"""Whether HumanSim's crowd can reach what a world and scenario hand it: every model resolves, and every point people
are walked to (an interactable's use point and first queue slots, seat standing points, the scenario's sources and
sinks, the humans' starts, waypoints and go_to stops) lies within arrival distance of the floor its walkers reach, no
wall between a queue's slots, and no room people are sent to is partly shut off by objects or locked doors.

The points are the ones the runtime hands HumanSim: an object's pose turned to its annotated `face` (`queue_theta`),
HumanSim's own `LineFormation` for the queue slots, the seats and standing points of the runtime's sit slots. The floor
is the level's rooms, door spans and cabins, less the walls, the doors locked to everyone and the bounding box of every
object whose model resolves, kept `WALKER_CLEARANCE` from them. A walker reaches the connected part of it that it
starts in, and an interactable or a sink must lie by a part some walker starts in: a free pocket behind a desk does not
count.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import shapely
import shapely.affinity
from arena_humansim.core.agents import BUILTIN_AGENTS, AgentType
from arena_humansim.core.agents.loader import load_agent_type_from_file
from arena_humansim.core.agents.types import GoToStepDef, StepDef
from arena_humansim.core.formation import Formation
from arena_humansim.core.formation.anchor import PoseAnchor
from arena_humansim.utils import DISTANCE_TOLERANCE
from arena_humansim.utils.types import Pose2D
from scipy import ndimage

from arena_simulation_setup.metrics.nav_at_r import CUT_OFF_AREA, RESOLUTION, clearance_of, components
from arena_simulation_setup.shared import Obstacle
from arena_simulation_setup.shared.semantics import authored_semantics
from arena_simulation_setup.shared.world import CABIN_DOOR_THRESHOLD, Door
from arena_simulation_setup.tree import Identifier, NetResolver
from arena_simulation_setup.tree.World import Level, World, WorldView, world_assets
from arena_simulation_setup.tree.World.model import elevator_doorway_pose
from arena_simulation_setup.tree.World.Scenario import RegionAssignment, Scenario
from arena_simulation_setup.utils.affordance import placed_slots, seat_approach, seat_reach, seat_slot
from arena_simulation_setup.utils.face import front_angle, queue_theta
from arena_simulation_setup.utils.geometry import Pose, Position

#: HumanSim's planner keeps walkers this far from walls and object boxes (its `global_planner.inflation_radius`) [m]
WALKER_CLEARANCE = 0.38
#: Queue slots behind a line's use point that must be reachable too
QUEUE_SLOTS = 3
#: How close a walker must come to a sink's point to be absorbed, without an `absorption_radius` (the adapter's default) [m]
ABSORPTION_RADIUS = 0.5

Stand = Callable[[float], float]


@attrs.frozen
class UsePoint:
    """A point HumanSim walks people to, in the frame of the object's level, and how close they must come."""

    object: str
    point: str
    x: float
    y: float
    reach: float
    #: the point before it in a queue, which no wall may stand between
    previous: tuple[float, float] | None = None
    #: the human or source whose walkers go there, None for anyone of the crowd
    walker: str | None = None


@attrs.frozen
class Finding:
    """Why an object or region fails: a model that does not resolve (`distance` NaN), or a point `distance` m from the
    floor a walker reaches, `reach` or more."""

    object: str
    point: str
    why: str
    level: str = ""
    x: float = math.nan
    y: float = math.nan
    distance: float = math.nan
    reach: float = math.nan

    def __str__(self) -> str:
        if math.isnan(self.distance):
            return f"{self.object}: {self.point} {self.why}"
        if self.point == "floor":
            return f"zone {self.object} {self.why} ({self.x:.2f}, {self.y:.2f}) on level {self.level}"
        far = "no floor a walker reaches" if math.isinf(self.distance) else f"{self.distance:.2f} m from the floor a walker reaches"
        return f"{self.object}: {self.point} at ({self.x:.2f}, {self.y:.2f}) on level {self.level} {self.why}, {far} (needs < {self.reach:g} m)"


def describe(findings: Sequence[Finding], limit: int = 10) -> str:
    """The findings in one line: the model findings grouped by model and answer with their objects, then the point findings, each list cut at `limit`."""
    models: dict[str, list[str]] = {}
    for finding in findings:
        if finding.point == "model":
            models.setdefault(finding.why, []).append(finding.object)
    lines = [f"model {why}, HumanSim never gets its {len(names)} object(s) and walkers pass through them: {', '.join(names[:limit])}{', ...' if len(names) > limit else ''}" for why, names in models.items()]
    points = [str(finding) for finding in findings if finding.point != "model"]
    lines += points[:limit] + ([f"... {len(points) - limit} more"] if len(points) > limit else [])
    return "; ".join(lines)


def region_point(polygon: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The point HumanSim gets as a region's pose: the mean of its vertices, or a point inside the polygon where that mean is not."""
    if not polygon:
        return 0.0, 0.0
    cx = sum(x for x, _ in polygon) / len(polygon)
    cy = sum(y for _, y in polygon) / len(polygon)
    if len(polygon) < 3:
        return cx, cy
    # the polygonal part only: a zero-width spike would otherwise count its line as inside
    area = shapely.make_valid(shapely.Polygon(polygon), method="structure", keep_collapsed=False)
    if area.is_empty or area.contains(shapely.Point(cx, cy)):
        return cx, cy
    inner = area.point_on_surface()
    return float(inner.x), float(inner.y)


def line_slots(x: float, y: float, theta: float, params: Mapping[str, float], count: int = QUEUE_SLOTS + 1) -> list[tuple[float, float]]:
    """The first `count` slots of HumanSim's line formation anchored at `x`, `y`, `theta` with `params`, the use point first."""
    line = Formation.create("line", anchor=PoseAnchor(fixed=Pose2D(x=x, y=y, theta=theta)), agent_lookup=lambda _agent: None, **{str(k): float(v) for k, v in params.items()})
    for agent in range(count):
        line.on_join(agent)
    targets = line.tick(0.0)
    return [(targets[agent].x, targets[agent].y) for agent in range(count)]


def _target_reach(humansim: Mapping[str, object]) -> float:
    """How close a walker comes to an object it walks to without a line or a seat: its `interaction_radius`, else its formation's base plus HumanSim's tolerance."""
    radius = humansim.get("interaction_radius")
    if isinstance(radius, int | float) and radius > 0:
        return float(radius)
    params = (humansim.get("formation") or {}).get("params") or {}  # pyright: ignore[reportAttributeAccessIssue]
    base = float(params.get("base_radius", params.get("base_step", 0.0)))
    return base + DISTANCE_TOLERANCE if base > 0.0 else DISTANCE_TOLERANCE


def object_points(obstacle: Obstacle, annotation: Mapping[str, object] | None, stand: Stand | None = None) -> list[UsePoint]:
    """The points HumanSim walks people to at an interactable (an object with a `humansim:` block), in the obstacle's
    frame: a line formation's use point and first `QUEUE_SLOTS` slots, each after the point before it (the object's box
    centre before the use point), else every seat's standing point (with `stand`, the
    runtime's distance ahead of a seat's front edge), else the object itself within its interaction radius."""
    humansim = obstacle.humansim
    if not humansim:
        return []
    x, y, yaw = obstacle.pose.position.x, obstacle.pose.position.y, obstacle.pose.orientation.to_yaw()
    annotation = annotation or {}
    theta = queue_theta(yaw, front_angle(annotation.get("face")))  # pyright: ignore[reportArgumentType]
    formation = humansim.get("formation") or {}
    if formation.get("type") == "line":
        slots = line_slots(x, y, theta, formation.get("params") or {})
        box = object_box(obstacle, annotation)
        chain = [(x, y) if box is None else (box.centroid.x, box.centroid.y), *slots]
        return [UsePoint(obstacle.name, "use point" if k == 0 else f"queue slot {k}", sx, sy, DISTANCE_TOLERANCE, chain[k]) for k, (sx, sy) in enumerate(slots)]
    seats = _seat_points(obstacle, annotation, stand) if stand is not None else []
    return seats or [UsePoint(obstacle.name, "target", x, y, _target_reach(humansim))]


def _seat_points(obstacle: Obstacle, annotation: Mapping[str, object], stand: Stand) -> list[UsePoint]:
    """The standing point before every seat as the runtime hands them HumanSim (scenario `seats`, else the sit slots at their centres in each allowed yaw)."""
    try:
        slots = placed_slots({"sit": list((annotation.get("affordances") or {}).get("sit") or [])})  # pyright: ignore[reportAttributeAccessIssue]
    except (TypeError, ValueError):
        slots = []
    box = annotation.get("bounding_box")
    bounding_box = box if isinstance(box, Sequence) else None
    if "seats" in obstacle.humansim:
        seats = [(i, Pose.parse(seat)) for i, seat in enumerate(obstacle.humansim["seats"])]
        edges = [None if (slot := seat_slot(seat, slots)) is None else seat_reach(seat, slot) for _, seat in seats]
    else:
        seats = [(i, Pose(Position(*slot.center), orientation)) for i, slot in enumerate(slots) for orientation in slot.facing.orientations]
        edges = [seat_reach(seat, slots[i]) for i, seat in seats]
    points = []
    for (i, seat), edge in zip(seats, edges, strict=True):
        placed = obstacle.pose * seat
        ahead = seat_approach(seat, edge, bounding_box, stand)
        yaw = placed.orientation.to_yaw()
        points.append(UsePoint(obstacle.name, f"seat {i} standing point", placed.position.x + ahead * math.cos(yaw), placed.position.y + ahead * math.sin(yaw), DISTANCE_TOLERANCE))
    return points


def region_points(name: str, region: RegionAssignment) -> list[UsePoint]:
    """The point of a sink a walker must reach (within its absorption radius), or a source's point, which must lie on or by the floor it spawns on."""
    polygon = [(p.x, p.y) for p in region.polygon]
    x, y = region_point(polygon)
    reach = float(region.config.get("absorption_radius", ABSORPTION_RADIUS)) if region.type == "sink" else DISTANCE_TOLERANCE
    return [UsePoint(name, region.type or "region", x, y, reach, walker=name if region.type == "source" else None)]


@attrs.define(eq=False)
class Floor:
    """The floor of one level a walker reaches: the occupancy raster (255 free), the label of every walkable cell, the
    boxes of the objects and the raster's frame."""

    grid: np.ndarray
    labels: np.ndarray
    boxes: tuple[tuple[str, shapely.Polygon], ...]
    origin: tuple[float, float]
    resolution: float
    _to: dict[frozenset[int], np.ndarray] = attrs.field(factory=dict)

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        rows, cols = self.grid.shape
        col = math.floor((x - self.origin[0]) / self.resolution)
        row = rows - 1 - math.floor((y - self.origin[1]) / self.resolution)
        return (row, col) if 0 <= row < rows and 0 <= col < cols else None

    def parts_at(self, x: float, y: float) -> frozenset[int]:
        """The walkable part a walker standing at a point is in: the part of the nearest walkable cell within arrival distance."""
        cell = self.cell(x, y)
        if cell is None:
            return frozenset()
        reach = math.ceil(DISTANCE_TOLERANCE / self.resolution)
        r0, c0 = max(cell[0] - reach, 0), max(cell[1] - reach, 0)
        window = self.labels[r0 : cell[0] + reach + 1, c0 : cell[1] + reach + 1]
        found = np.argwhere(window > 0)
        if not len(found):
            return frozenset()
        gaps = np.hypot(found[:, 0] + r0 - cell[0], found[:, 1] + c0 - cell[1])
        k = int(gaps.argmin())
        return frozenset({int(window[found[k, 0], found[k, 1]])}) if gaps[k] * self.resolution < DISTANCE_TOLERANCE else frozenset()

    def parts_in(self, polygon: shapely.Polygon) -> frozenset[int]:
        window = self._cells(polygon)
        if window is None:
            return frozenset()
        return frozenset(int(label) for label in np.unique(self.labels[window[0], window[1]][window[4]]) if label)

    def area(self, parts: frozenset[int]) -> float:
        return float(np.count_nonzero(np.isin(self.labels, list(parts)))) * self.resolution**2 if parts else 0.0

    def largest(self) -> frozenset[int]:
        counts = np.bincount(self.labels.ravel())
        return frozenset({int(np.argmax(counts[1:])) + 1}) if len(counts) > 1 else frozenset()

    def distance(self, x: float, y: float, parts: frozenset[int]) -> float:
        """How far a point lies from the walkable `parts`, inf off the raster or without parts."""
        cell = self.cell(x, y)
        if cell is None or not parts:
            return math.inf
        if parts not in self._to:
            to = ndimage.distance_transform_edt(~np.isin(self.labels, list(parts))) * self.resolution
            assert isinstance(to, np.ndarray)
            self._to[parts] = to
        return float(self._to[parts][cell])

    def why(self, x: float, y: float, own: str) -> str:
        """Why a point lies off the floor a walker reaches."""
        inside = [name for name, box in self.boxes if box.contains(shapely.Point(x, y))]
        if inside:
            others = [name for name in inside if name != own]
            return f"lies inside the box of {', '.join(others)}" if others else "lies inside its own box"
        cell = self.cell(x, y)
        if cell is None or self.grid[cell] != 255:
            return "lies off the floor or on a wall"
        if self.labels[cell]:
            return "lies on floor its walkers cannot reach"
        return "lies too close to walls or objects"

    def _cells(self, polygon: shapely.Polygon) -> tuple[slice, slice, np.ndarray, np.ndarray, np.ndarray] | None:
        """The raster window over a polygon, its cell centres and which of them the polygon holds."""
        x0, y0, x1, y1 = polygon.bounds
        rows, cols = self.grid.shape
        c0, c1 = max(math.floor((x0 - self.origin[0]) / self.resolution), 0), min(math.floor((x1 - self.origin[0]) / self.resolution) + 1, cols)
        r0, r1 = max(rows - 1 - math.floor((y1 - self.origin[1]) / self.resolution), 0), min(rows - math.floor((y0 - self.origin[1]) / self.resolution), rows)
        if c1 <= c0 or r1 <= r0:
            return None
        gx, gy = np.meshgrid(self.origin[0] + (np.arange(c0, c1) + 0.5) * self.resolution, self.origin[1] + (rows - 1 - np.arange(r0, r1) + 0.5) * self.resolution)
        return slice(r0, r1), slice(c0, c1), gx, gy, shapely.contains_xy(polygon, gx, gy)

    def cut_off(self, polygon: shapely.Polygon, parts: frozenset[int], joined: np.ndarray) -> tuple[float, float, float] | None:
        """The walkable floor of a polygon outside `parts` but among the `joined` cells, its area and one of its points, None below `CUT_OFF_AREA`."""
        window = self._cells(polygon)
        if window is None:
            return None
        rows, cols, gx, gy, inside = window
        labels = self.labels[rows, cols]
        stranded = inside & (labels > 0) & ~np.isin(labels, list(parts)) & joined[rows, cols]
        area = float(np.count_nonzero(stranded)) * self.resolution**2
        if area < CUT_OFF_AREA:
            return None
        k = np.argwhere(stranded)[0]
        return area, float(gx[k[0], k[1]]), float(gy[k[0], k[1]])


def object_box(obstacle: Obstacle, annotation: Mapping[str, object] | None) -> shapely.Polygon | None:
    """The object's bounding box on the floor as HumanSim walls it in (x and y bounds turned by its yaw), None without one."""
    box = (annotation or {}).get("bounding_box")
    if not isinstance(box, Sequence) or len(box) < 2:
        return None
    (x_min, x_max), (y_min, y_max) = box[0], box[1]  # pyright: ignore[reportGeneralTypeIssues]
    polygon = shapely.affinity.rotate(shapely.box(x_min, y_min, x_max, y_max), obstacle.pose.orientation.to_yaw(), origin=(0, 0), use_radians=True)
    return shapely.affinity.translate(polygon, xoff=obstacle.pose.position.x, yoff=obstacle.pose.position.y)


def level_floor(level: Level, boxes: Iterable[tuple[str, shapely.Polygon]], *, resolution: float = RESOLUTION, clearance: float = WALKER_CLEARANCE) -> Floor:
    """The floor of `level` among `boxes` (name, level-frame polygon): its rooms, door spans and cabins, walls and boxes
    occupied, walkable `clearance` from them (less half a cell, the raster's error at a wall line), in connected parts."""
    boxes = tuple(boxes)
    grid, origin = level.render_grid(resolution=resolution, asset_color="black", asset_name_color=None, static_objects=boxes)
    walkable = clearance_of(grid, resolution) >= clearance - resolution / 2
    labels, _, _ = components(walkable, walkable)
    return Floor(grid, labels, boxes, origin, resolution)


def _unresolved(model: Identifier) -> str:
    """Why a model does not resolve: what each asset bucket answered (it does not hold it: a stale name; an expired token,
    an unreachable host: the machine), else that no asset root holds it."""
    answers = NetResolver.answers(model)
    if answers:
        return "; ".join(answers)
    return "the asset network is off and no asset root holds it" if NetResolver.offline() else "no asset root holds it"


async def _annotations(objects: Sequence[Obstacle]) -> list[dict | str]:
    """Each object's annotation as the runtime reads it (an empty dict for an object without a model or a model without an
    annotation.yaml: a target without an obstacle), or why its model does not resolve; each model resolved once."""

    async def one(obstacle: Obstacle) -> dict | str:
        try:
            view = await obstacle.model.resolve()
        except FileNotFoundError:
            return f"{obstacle.model.name!r} does not resolve ({_unresolved(obstacle.model)})"
        return view.annotation or {}

    models = {str(o.model.relpath()): o for o in objects if not o.target_only}
    resolved = dict(zip(models, await asyncio.gather(*(one(obstacle) for obstacle in models.values())), strict=True))
    return [{} if o.target_only else resolved[str(o.model.relpath())] for o in objects]


def locked(door: Door) -> bool:
    """Whether a door is shut to everyone: a gate locked with nobody authorized."""
    return any(item.get("preset") == "gate" and (item.get("params") or {}).get("locked") and not (item.get("params") or {}).get("authorized") for item in authored_semantics(door.semantics))


def agent_type(name: str, base: Path | None) -> AgentType | None:
    """The agent type a scenario names: a yaml file relative to `base`, or a builtin name; None when neither loads."""
    if not ("/" in name or name.endswith(".yaml")):
        return BUILTIN_AGENTS.get(name)
    path = Path(name) if Path(name).is_absolute() or base is None else base / name
    return load_agent_type_from_file(path) if path.is_file() else None


def _stops(definition: AgentType, owner: str) -> list[UsePoint]:
    """The go_to poses of an agent type."""
    return [UsePoint(owner, f"go_to {sequence}.{name} of {definition.name}", step.target_pose.x, step.target_pose.y, DISTANCE_TOLERANCE, walker=owner) for sequence, spec in definition.sequences.items() for name, step in spec.steps.items() if isinstance(step, GoToStepDef) and step.target_pose is not None]


def _targets(definition: AgentType) -> set[str]:
    """The object ids and types an agent type's steps walk to."""
    return {step.target for spec in definition.sequences.values() for step in spec.steps.values() if isinstance(step, StepDef | GoToStepDef) and step.target}


def targeted(obstacle: Obstacle, targets: Iterable[str]) -> bool:
    """Whether people seek an interactable: it satisfies a need, or a step names it or its type."""
    names = {obstacle.name, str(obstacle.humansim.get("type", ""))}
    return bool(obstacle.humansim.get("satisfies")) or bool(names & set(targets))


def _crossing(walls: shapely.Geometry, a: tuple[float, float], b: tuple[float, float]) -> bool:
    return bool(shapely.intersects(walls, shapely.LineString([a, b])))


def check(
    view: WorldView,
    world: World,
    scenario: Scenario | None = None,
    *,
    scenario_dir: Path | None = None,
    origins: Mapping[str, tuple[float, float]] | None = None,
    stand: Stand | None = None,
    only: Callable[[str], bool] | None = None,
) -> list[Finding]:
    """The findings of a world (its level objects) and a `scenario` (its objects, humans with their starts, waypoints
    and agent types' go_to poses, sources and sinks; in the map frame, each level shifted by its entry of `origins`;
    agent type files relative to `scenario_dir`), every model resolved through `view`. Of the interactables only those
    people seek count (`targeted`), seats only with `stand`, and only the objects, humans and regions whose name `only`
    takes when given."""
    origins = origins or {}
    scenario = scenario or Scenario()
    first = next(iter(world.levels))
    wanted = only or (lambda _name: True)

    def place(level: str, x: float, y: float) -> tuple[str, float, float]:
        """The level of a scenario point and the point in its frame: an explicit level's frame, else the map frame of the level whose zones hold it."""
        if level in world.levels:
            return level, x, y
        for level_id, held in world.levels.items():
            dx, dy = origins.get(level_id, (0.0, 0.0))
            if any(len(z.polygon) >= 3 and z.shape().covers(shapely.Point(x - dx, y - dy)) for z in held.zones):
                return level_id, x - dx, y - dy
        dx, dy = origins.get(first, (0.0, 0.0))
        return first, x - dx, y - dy

    placed = [(level_id, o) for level_id, level in world.levels.items() for o in level.objects]
    for obstacle in scenario.objects:
        level_id, x, y = place(obstacle.level, obstacle.pose.position.x, obstacle.pose.position.y)
        placed.append((level_id, attrs.evolve(obstacle, pose=Pose(Position(x, y, obstacle.pose.position.z), obstacle.pose.orientation))))
    with world_assets(view, world):
        annotations = Identifier._run_sync(_annotations([o for _, o in placed]))
    findings = [Finding(o.name, "model", why) for (_, o), why in zip(placed, annotations, strict=True) if isinstance(why, str)]

    points: dict[str, list[UsePoint]] = {first: []}

    def add(level: str, point: UsePoint) -> None:
        level_id, x, y = place(level, point.x, point.y)
        points.setdefault(level_id, []).append(attrs.evolve(point, x=x, y=y))

    targets: set[str] = set()
    for human in scenario.humans:
        agent = human.agent.get("agent_type", "adult") if isinstance(human.agent, dict) else human.agent
        definition = agent_type(agent, human.included_from or scenario_dir) if isinstance(agent, str) else None
        targets |= _targets(definition) if definition is not None else set()
        add(human.level, UsePoint(human.name, "start", human.pose.position.x, human.pose.position.y, DISTANCE_TOLERANCE, walker=human.name))
        for i, w in enumerate(human.waypoints):
            if w.elevator is None:  # an elevator waypoint is a ride, its cabin's
                add(w.level or human.level, UsePoint(human.name, f"waypoint {i}", w.x, w.y, DISTANCE_TOLERANCE, walker=human.name))
        for stop in _stops(definition, human.name) if definition is not None else []:
            add(human.level, stop)
    sources: list[tuple[str, UsePoint, shapely.Polygon]] = []
    for name, region in scenario.regions.items():
        agent = (region.config.get("agent") or {}).get("agent_type", "adult") if region.type == "source" else None
        definition = agent_type(agent, scenario_dir) if isinstance(agent, str) else None
        targets |= _targets(definition) if definition is not None else set()
        if len(region.polygon) < 3:
            continue
        level_id, _, _ = place("", *region_point([(p.x, p.y) for p in region.polygon]))
        dx, dy = origins.get(level_id, (0.0, 0.0))
        polygon = shapely.Polygon([(p.x - dx, p.y - dy) for p in region.polygon])
        for point in region_points(name, region):
            if region.type == "source":
                sources.append((level_id, attrs.evolve(point, x=point.x - dx, y=point.y - dy), polygon))
            else:
                add("", point)
        for stop in _stops(definition, name) if definition is not None else []:
            add("", stop)
    for (level_id, obstacle), annotation in zip(placed, annotations, strict=True):
        if targeted(obstacle, targets):
            points.setdefault(level_id, []).extend(object_points(obstacle, annotation if isinstance(annotation, dict) else None, stand))

    floors: dict[str, Floor] = {}
    # the floor without objects and with every door open: a pocket the walls close off by themselves is no room cut off
    bare: dict[str, Floor] = {}
    walls: dict[str, shapely.Geometry] = {}
    for level_id, level in world.levels.items():
        shut = [door for door in level.doors if locked(door)]
        boxes = [(o.name, box) for (lid, o), annotation in zip(placed, annotations, strict=True) if lid == level_id and isinstance(annotation, dict) and (box := object_box(o, annotation)) is not None]
        boxes += [(f"locked door {door.name}", shapely.LineString([(door.start.x, door.start.y), (door.end.x, door.end.y)]).buffer(RESOLUTION)) for door in shut]
        floors[level_id] = level_floor(level, boxes)
        bare[level_id] = level_floor(level, [])
        walls[level_id] = shapely.union_all([shapely.LineString([(w.start.x, w.start.y), (w.end.x, w.end.y)]) for w in level.all_walls] + [shapely.LineString([(d.start.x, d.start.y), (d.end.x, d.end.y)]) for d in shut])

    def at(level_id: str, x: float, y: float) -> set[tuple[str, int]]:
        return {(level_id, label) for label in floors[level_id].parts_at(x, y)}

    starts: dict[str, set[tuple[str, int]]] = {}
    for level_id, here in points.items():
        for p in here:
            if p.point == "start" and p.walker is not None:
                starts.setdefault(p.walker, set()).update(at(level_id, p.x, p.y))
    for level_id, point, polygon in sources:
        starts.setdefault(point.object, set()).update((level_id, label) for label in floors[level_id].parts_in(polygon))
    links = _links(world, scenario, place, at)

    def closure(nodes: set[tuple[str, int]]) -> set[tuple[str, int]]:
        seen, todo = set(nodes), list(nodes)
        while todo:
            for node in links.get(todo.pop(), ()):
                if node not in seen:
                    seen.add(node)
                    todo.append(node)
        return seen

    reach = {walker: closure(nodes) for walker, nodes in starts.items()}
    everyone = set().union(*reach.values()) if reach else set()

    def parts(level_id: str, nodes: set[tuple[str, int]]) -> frozenset[int]:
        return frozenset(label for lid, label in nodes if lid == level_id)

    for level_id, here in sorted(points.items()):
        floor = floors[level_id]
        crowd = parts(level_id, everyone) or floor.largest()
        seen_points: set[tuple[str, str]] = set()
        for point in here:
            if (point.object, point.point) in seen_points:
                continue
            seen_points.add((point.object, point.point))
            if point.point == "start":
                # a start in a pocket smaller than a room reaches nothing a walker is sent to
                reached = parts(level_id, reach.get(point.walker or "", set()))
                distance = 0.0 if floor.area(reached) >= CUT_OFF_AREA else floor.distance(point.x, point.y, floor.largest())
            else:
                own = reach.get(point.walker or "")
                if own is not None and not own:
                    continue  # the walker's start already fails
                distance = floor.distance(point.x, point.y, (parts(level_id, own) if own else frozenset()) or crowd)
            if distance >= point.reach:
                findings.append(Finding(point.object, point.point, floor.why(point.x, point.y, point.object), level_id, point.x, point.y, distance, point.reach))
            elif point.previous is not None and _crossing(walls[level_id], point.previous, (point.x, point.y)):
                findings.append(Finding(point.object, point.point, "lies behind a wall from the object or the queue slot before it", level_id, point.x, point.y, distance, point.reach))
        # the zones people are sent to; with nobody sent anywhere, every zone of a scenario with demand (a crowd read from it later);
        # only floor that objects, locks or doors cut off counts, floor the crowd's part reaches on the bare floor
        demand = any(p.walker is None and p.point != "sink" for p in here)
        open_parts = np.unique(bare[level_id].labels[np.isin(floor.labels, list(crowd))])
        joined = np.isin(bare[level_id].labels, open_parts[open_parts > 0])
        for zone in world.levels[level_id].zones:
            if len(zone.polygon) < 3:
                continue
            shape = zone.shape()
            if (starts or not demand) and not any(shape.covers(shapely.Point(p.x, p.y)) for p in here):
                continue
            if (cut := floor.cut_off(shape, crowd, joined)) is not None:
                area, x, y = cut
                findings.append(Finding(zone.name, "floor", f"has {area:.1f} m2 of floor walkers cannot reach from where they start, around", level_id, x, y, floor.distance(x, y, crowd), 0.0))
    for level_id, point, polygon in sources:
        if not floors[level_id].parts_in(polygon):
            findings.append(Finding(point.object, "source", "holds no floor a walker reaches", level_id, point.x, point.y, floors[level_id].distance(point.x, point.y, floors[level_id].largest()), point.reach))
    return [finding for finding in findings if wanted(finding.object)]


def _links(
    world: World,
    scenario: Scenario,
    place: Callable[[str, float, float], tuple[str, float, float]],
    at: Callable[[str, float, float], set[tuple[str, int]]],
) -> dict[tuple[str, int], set[tuple[str, int]]]:
    """The floor parts that elevators, stair halves and the scenario's teleport zones join, both ways."""
    ends: dict[str, set[tuple[str, int]]] = {}
    joins: list[tuple[str, str]] = []
    for level_id, level in world.levels.items():
        for elevator in level.elevators:
            door = elevator_doorway_pose(elevator, standoff=CABIN_DOOR_THRESHOLD + DISTANCE_TOLERANCE)
            ends[elevator.name] = at(level_id, door.position.x, door.position.y)
            joins += [(elevator.name, other) for other in elevator.destinations]
        for stair in level.stairs:
            ends[stair.name] = at(level_id, stair.mouth.x, stair.mouth.y)
            joins += [(stair.name, stair.to)] if stair.to else []
    for zone in scenario.teleport_zones:
        ends[zone.name] = at(*place("", zone.entry[0], zone.entry[1]))
        joins += [(zone.name, other) for other in zone.destinations]
    links: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for a, b in joins:
        for one in ends.get(a, set()):
            for other in ends.get(b, set()):
                links.setdefault(one, set()).add(other)
                links.setdefault(other, set()).add(one)
    return links


def check_scenario(view: WorldView, scenario: str, *, stand: Stand | None = None, seed: int = 0) -> list[Finding]:
    """The findings of a world on disk and one of its scenarios, the objects and humans posed by zone name drawn with
    `seed` on walkable floor of the world, as the runtime draws them clear of its map."""
    world = view.load(validate=False)
    origins = view.level_origins() or dict.fromkeys(world.levels, (0.0, 0.0))
    with world_assets(view, world):
        annotations = {level_id: Identifier._run_sync(_annotations(level.objects)) for level_id, level in world.levels.items()}
    floors = {level_id: level_floor(level, [(o.name, box) for o, a in zip(level.objects, annotations[level_id], strict=True) if isinstance(a, dict) and (box := object_box(o, a)) is not None]) for level_id, level in world.levels.items()}

    def walkable(point: Position) -> bool:
        for level_id, floor in floors.items():
            dx, dy = origins.get(level_id, (0.0, 0.0))
            cell = floor.cell(point.x - dx, point.y - dy)
            if cell is not None and floor.labels[cell] > 0:
                return True
        return False

    scenario_view = view.scenario(scenario).resolve_sync()
    loaded = scenario_view.load(converter=world.compact_world(origins=origins).zone_converter(np.random.default_rng(seed), is_valid=walkable))
    return check(view, world, loaded, scenario_dir=scenario_view.path, origins=origins, stand=stand)


def check_world(view: WorldView, *, stand: Stand | None = None) -> list[Finding]:
    """The findings of a world on disk without a scenario: its level objects."""
    return check(view, view.load(validate=False), stand=stand)


__all__ = ["QUEUE_SLOTS", "WALKER_CLEARANCE", "Finding", "Floor", "UsePoint", "check", "check_scenario", "check_world", "describe", "level_floor", "line_slots", "agent_type", "locked", "object_box", "object_points", "region_point", "region_points", "targeted"]
