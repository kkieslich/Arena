"""Read a FLOWEDIT graph and crowd settings into a scenario: no hand-written person, the graph and the world's geometry only.

| Graph | Scenario |
|---|---|
| object demand (`at`, `serves`, `has_formation`, `satisfies`) | `objects:` with their `humansim:` block, one per graph object the world does not hold |
| `satisfies need:<n>` | every person's type declares need `n` and seeks an object satisfying it once it is low |
| `affords interaction:<kind>` | the step that uses the object: `kind` at it, or a `go_to` and `kind` beside it (`meet`) |
| `dwells` | the `duration` of that step |
| `has_purpose` of rooms | where people start and which rooms they walk between (populate's rules) |
| doors in one room only | a source and a sink inside the door: people enter, use the site, and leave there |

The world comes from `open_site`, an edited world as well as a shipped one. FLOWEDIT compares a run before an edit with a run
after it, so the two may differ by the edit only: the people (count, kind, speed, start room, route, idle points) are drawn
from the unedited base world, its graph and the seed alone; the edited site gives only the objects, which object serves which
need, the dwell, and the entrances.
"""

from __future__ import annotations

import hashlib
import math
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import yaml

from arena_simulation_setup.kg import Scene, Value, dumps

from ..scenario import populate as P
from ..scenario.formats import first_level, scenario_text
from ..scenario.materialize import GENERATED_MARKER, WouldOverwrite, is_generated
from . import settings as S
from .edit import EDITS, Site, open_site
from .interactions import by_rule, handle

if TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.shared.world import Door
    from arena_simulation_setup.tree.World.model import World

TYPES_DIR = "types"
PROVENANCE = "provenance.yaml"
#: Clearance a sampled standing point keeps from walls and furniture (m).
PERSON_CLEAR = 0.4
#: Depths of an entrance's sink and source strips inside its door (m): the sink at the door, the source behind it.
SINK_DEPTH = (0.1, 0.6)
SOURCE_DEPTH = (0.8, 1.8)
#: Footprint half-size a scenario object is stamped with on the sampling grid (m).
OBJECT_STAMP = 0.5
WALK_PATIENCE = {"mean": 60.0, "std": 10.0, "clip_low": 30.0, "clip_high": 120.0}
#: How far a world object's pose in the graph may differ from world.yaml before it counts as moved (m).
POSE_TOLERANCE = 0.01

Point = tuple[float, float]


def patience_for(base: Mapping[str, float], hold: Mapping[str, float]) -> dict[str, float]:
    """`base` patience shifted by the longest `hold`: HumanSim's step patience covers nav, queue and the hold itself, and a
    patience that runs out mid-hold ends every use at once (sitters stand up together)."""
    longest = float(hold.get("clip_high", hold["mean"] + 3 * hold.get("std", 0.0)))
    out = {"mean": float(base["mean"]) + longest}
    if "std" in base:
        out["std"] = float(base["std"])
    for key in ("clip_low", "clip_high"):
        if key in base:
            out[key] = float(base[key]) + longest
    return out


@attrs.frozen
class Affordance:
    kind: str
    meet: bool = False
    origin: str = ""


@attrs.define
class Thing:
    """An object of the graph."""

    name: str
    model: str = ""
    floor: str = ""
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    pose_ref: str | None = None
    service: str | None = None
    capacity: int | None = None
    interaction_radius: float | None = None
    formation: dict[str, Any] | None = None
    #: need -> amount
    satisfies: dict[str, float] = attrs.Factory(dict)
    dwell: dict[str, float] | None = None
    affords: list[Affordance] = attrs.Factory(list)

    def block(self) -> dict[str, Any]:
        """The object's `humansim:` block as a scenario writes it."""
        out: dict[str, Any] = {}
        if self.service:
            out["type"] = self.service
        if self.capacity is not None:
            out["capacity"] = self.capacity
        if self.interaction_radius is not None:
            out["interaction_radius"] = self.interaction_radius
        if self.satisfies:
            out["satisfies"] = dict(self.satisfies)
        if self.formation:
            out["formation"] = self.formation
        return out

    def usable(self) -> list[Affordance]:
        """The affordances a person can run alone at this object, object-bound first: no SERVICE (it needs a provider) and no agent target.

        An object with no demand at all (no `humansim:` block) is scenery and affords nothing; one with demand but no `affords`
        (a graph from before `affords`) is read by HumanSim's rules as the builder would.
        """
        found = self.affords
        if not found and self.block() and (rule := by_rule(self.block(), self.model)) is not None:
            found = [Affordance(rule[0], rule[1], rule[2])]
        ok = [a for a in found if handle(a.kind) in ("object", "none")]
        return sorted(ok, key=lambda a: (a.meet, a.kind))


@attrs.define
class Graph:
    objects: dict[str, Thing]
    purposes: dict[str, str]
    #: door name -> the rooms that contain it
    doors: dict[str, list[str]]
    text_sha: str = ""

    def exterior_doors(self) -> list[str]:
        """Doors only one room contains: a way in from outside the zones."""
        return sorted(d for d, rooms in self.doors.items() if len(rooms) == 1)


def _name(ref: Value) -> str:
    return str(ref).partition(":")[2]


def read_graph(scene: Scene) -> Graph:
    """The objects, room purposes and doors of a FLOWEDIT graph."""
    objects = {_name(id): Thing(_name(id), model=str(e.props.get("model", ""))) for id, e in scene.entities.items() if id.startswith("obj:")}
    purposes: dict[str, str] = {}
    doors: dict[str, list[str]] = {_name(id): [] for id in scene.entities if id.startswith("door:")}
    for f in scene.facts:
        p = dict(f.params)
        if f.verb == "has_purpose" and f.subject.startswith("room:"):
            purposes[_name(f.subject)] = _name(f.object)
        elif f.verb == "contains" and f.subject.startswith("room:") and str(f.object).startswith("door:"):
            doors.setdefault(_name(f.object), []).append(_name(f.subject))
        if not f.subject.startswith("obj:"):
            continue
        t = objects.setdefault(_name(f.subject), Thing(_name(f.subject)))
        if f.verb == "at":
            t.floor, t.x, t.y, t.yaw = _name(f.object), float(p["x"]), float(p["y"]), float(p.get("yaw", 0.0))
            t.pose_ref = str(p["pose_ref"]) if p.get("pose_ref") is not None else None
        elif f.verb == "serves":
            t.service = _name(f.object)
            t.capacity = int(p["capacity"]) if p.get("capacity") is not None else None
            t.interaction_radius = float(p["interaction_radius"]) if p.get("interaction_radius") is not None else None
        elif f.verb == "satisfies":
            # `from` only records where the value came from: every satisfies is the object's own
            t.satisfies[_name(f.object)] = float(p.get("amount", 100.0))
        elif f.verb == "has_formation":
            params = {k: v for k, v in p.items() if k not in ("anchor_kind", "anchor_ref")}
            formation: dict[str, Any] = {"type": _name(f.object), "params": params}
            if "anchor_kind" in p:
                formation["anchor"] = {"kind": p["anchor_kind"], **({"ref": p["anchor_ref"]} if "anchor_ref" in p else {})}
            t.formation = formation
        elif f.verb == "dwells":
            t.dwell = {"mean": float(f.object), **{k: float(v) for k, v in p.items() if k in ("std", "clip_low", "clip_high")}}
        elif f.verb == "affords":
            t.affords.append(Affordance(_name(f.object), bool(p.get("meet", False)), str(p.get("from", ""))))
    # an object without `affords` (inserted by an edit, say) is used as the objects of its service type are
    by_service: dict[str, tuple[str, list[Affordance]]] = {}
    for name, t in sorted(objects.items()):
        if t.affords and t.service and t.service not in by_service:
            by_service[t.service] = (name, t.affords)
    for t in objects.values():
        if not t.affords and t.service in by_service:
            source, affords = by_service[t.service]
            t.affords = [attrs.evolve(a, origin=f"obj:{source} (same service type)") for a in affords]
    return Graph(objects, purposes, {d: sorted(set(r)) for d, r in doors.items()})


@attrs.frozen
class NeedPlan:
    """How a person meets need `need`: `kind` at (or, `meet`, beside) `target`, an object id or a service type."""

    need: str
    target: str
    kind: str
    meet: bool
    amount: float
    dwell: dict[str, float]
    objects: tuple[str, ...]
    synthesized: bool = False

    def steps(self, patience: Mapping[str, float]) -> dict[str, Any]:
        use = {"interaction": self.kind, "duration": dict(self.dwell), "patience": patience_for(patience, self.dwell), "satisfies": {self.need: self.amount}}
        if self.meet:
            return {"go": {"kind": "go_to", "target": self.target, "patience": dict(patience)}, "use": use}
        return {"use": {"target": self.target, **use}}


def plan_needs(graph: Graph, settings: Mapping[str, Any], floor: str, notes: list[str]) -> dict[str, NeedPlan]:
    """Each need the graph's objects on `floor` satisfy, with the objects that serve it and how they are used."""
    by_need: dict[str, list[tuple[Thing, Affordance]]] = {}
    synthesized: set[str] = set()
    for name, t in sorted(graph.objects.items()):
        usable = t.usable()
        if t.floor != floor:
            if t.satisfies or usable:
                notes.append(f"obj:{name}: on floor {t.floor!r}, people walk floor {floor!r} only, left out of the needs")
            continue
        if not usable:
            if t.satisfies:
                notes.append(f"obj:{name}: satisfies {sorted(t.satisfies)} but affords nothing a person runs alone (SERVICE needs a provider), left out of the needs")
            continue
        needs = dict(t.satisfies)
        if not needs and settings["synthesize_needs"]:
            need = f"visit_{t.service or name}"
            needs = {need: 100.0}
            synthesized.add(need)
        for need in needs:
            by_need.setdefault(need, []).append((t, usable[0]))
    plans: dict[str, NeedPlan] = {}
    for need, users in sorted(by_need.items()):
        groups: dict[tuple, list[tuple[Thing, Affordance]]] = {}
        for t, a in users:
            dwell = t.dwell or dict(settings["dwell"])
            groups.setdefault((t.service, a.kind, a.meet, tuple(sorted(dwell.items()))), []).append((t, a))
        # HumanSim holds the dwell on the person's step, so one step (one kind, one dwell, one target) serves a need: objects
        # that differ for one need would silently drop all but one, and an edit to the others would change nothing in PEDS
        if len(groups) > 1:
            parts = "; ".join(f"{sorted(t.name for t, _ in g)}: type {k[0]!r}, {k[1]}{' meet' if k[2] else ''}, dwell {dict(k[3])}" for k, g in sorted(groups.items(), key=lambda kv: kv[1][0][0].name))
            raise ValueError(f"need:{need}: the objects that satisfy it differ ({parts}), and one step serves a need until HumanSim holds the dwell on the object: give them one type, interaction and dwell")
        key, chosen = next(iter(groups.items()))
        names = tuple(t.name for t, _ in chosen)
        service = key[0]
        all_of_type = service is not None and {t.name for t in graph.objects.values() if t.service == service and t.floor == floor} == set(names)
        if len(names) > 1 and not all_of_type:
            raise ValueError(f"need:{need}: {list(names)} satisfy it, but people seek several objects only by a service type all of them, and nothing else on the floor, have")
        target = service if len(names) > 1 else names[0]
        t0, a0 = chosen[0]
        if t0.dwell is None:
            notes.append(f"obj:{t0.name}: no dwell in the graph, the settings' dwell used")
        plans[need] = NeedPlan(need, target, a0.kind, a0.meet, t0.satisfies.get(need, 100.0), dict(t0.dwell or settings["dwell"]), names, need in synthesized)
    return plans


@attrs.define
class Floor:
    """A world as the reader samples it: the people's level, its zones and a free-space grid."""

    root: Path
    world: World
    level_id: str
    zones: list[dict[str, Any]]
    origin: Point
    grid: Any
    statics: list[dict[str, Any]]

    def shape(self, room: str) -> shapely.Polygon | None:
        import shapely

        zone = next((z for z in self.zones if z["name"] == room), None)
        return None if zone is None else shapely.Polygon(P.polygon(zone))

    def point(self, room: str, rng: random.Random) -> Point | None:
        """A free standing point of `room` in the map frame, None when the room has none."""
        import shapely

        shape = self.shape(room)
        if shape is None or shape.is_empty:
            return None
        minx, miny, maxx, maxy = shape.bounds
        for _ in range(300):
            pt = (rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if shape.contains(shapely.Point(pt)) and (self.grid is None or self.grid.free(pt, PERSON_CLEAR)):
                return self.to_map(pt)
        c = shape.representative_point()
        near = None if self.grid is None else self.grid.nearest_free((c.x, c.y), PERSON_CLEAR, 3.0)
        return self.to_map(near or (c.x, c.y))

    def to_map(self, pt: Point) -> Point:
        return (round(pt[0] + self.origin[0], 2), round(pt[1] + self.origin[1], 2))


def _origins(root: Path, world: World) -> dict[str, tuple[float, float]]:
    from arena_simulation_setup.tree.World.view import WorldView

    return WorldView(root).level_origins() or {lid: (0.0, 0.0) for lid in world.levels}


def _floor(root: Path, world: World, graph: Graph) -> Floor:
    """The level people walk, with the graph's modelled objects the world does not hold stamped onto its grid."""
    from ..scenario.occupancy import level_grid, world_statics
    from ..scenario.promptgen.world import Grid

    level = first_level(root, world)
    if level is None:
        raise ValueError(f"{root.name}: no level with zones to place people on")
    level_id = str(level["id"])
    held = {o.name for lv in world.levels.values() for o in lv.objects}
    statics = list(world_statics(root, world))
    for t in sorted(graph.objects.values(), key=lambda t: t.name):
        # a demand-only node (no model) is a HumanSim target, not an obstacle
        if t.name not in held and t.floor == level_id and t.model:
            statics.append({"name": t.name, "model": t.model, "pose": [t.x, t.y], "radius": OBJECT_STAMP, "resolved": False})
    loaded = level_grid(root, statics, world)
    grid = None if loaded is None else Grid(occupied=loaded[0], res=loaded[1], ox=loaded[2], oy=loaded[3])
    return Floor(root, world, level_id, list(level.get("zones") or []), _origins(root, world).get(level_id, (0.0, 0.0)), grid, statics)


def _objects(graph: Graph, floor: Floor, notes: list[str]) -> list[dict[str, Any]]:
    """The graph's objects the world does not hold, posed where the graph says, in the map frame.

    An object without a model (the `<name>_demand` node an edit leaves behind) is written with an empty model: a HumanSim
    target with its demand and no obstacle.
    """
    origins = _origins(floor.root, floor.world)
    held = {o.name: (lid, o) for lid, lv in floor.world.levels.items() for o in lv.objects}
    out = []
    for name, t in sorted(graph.objects.items()):
        if name in held:
            lid, o = held[name]
            moved = lid != t.floor or math.dist((o.pose.position.x - origins.get(lid, (0, 0))[0], o.pose.position.y - origins.get(lid, (0, 0))[1]), (t.x, t.y)) > POSE_TOLERANCE
            if moved:
                raise ValueError(f"obj:{name} is an object of world.yaml, a scenario cannot move it: move it with `flowedit edit`, which moves it in world.yaml too")
            continue
        ox, oy = origins.get(t.floor, (0.0, 0.0))
        entry: dict[str, Any] = {"name": name, "model": t.model, "pose": [round(t.x + ox, 3), round(t.y + oy, 3), round(t.yaw, 4)]}
        if t.floor != floor.level_id and len(floor.world.levels) > 1:
            entry["level"] = t.floor
            entry["pose"] = [round(t.x, 3), round(t.y, 3), round(t.yaw, 4)]
        if block := t.block():
            entry["humansim"] = block
        if not t.model:
            if not block:
                notes.append(f"obj:{name}: no model and no demand, left out")
                continue
            notes.append(f"obj:{name}: no model, written as a HumanSim target without an obstacle")
        if t.pose_ref:
            notes.append(f"obj:{name}: posed at zone {t.pose_ref!r} in the graph, written at the graph's point so the scenario is the same every run")
        out.append(entry)
    return out


def _robot(floor: Floor, settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The robot of the settings as a scenario runs it, None without one: the reader never invents a robot."""
    from ..scenario.route import robot_for

    if not settings["robot"]:
        return None
    return robot_for(dict(settings["robot"]), floor.zones)


def _homes(floor: Floor, robot: Mapping[str, Any] | None) -> frozenset[str]:
    from ..scenario.zone_screen import ref_safe_zones, screen_zones

    g = floor.grid
    clearance_map = None if g is None else (g.dist, g.res, g.ox, g.oy)
    route = P.route_points(robot, floor.zones) if robot else []
    verdicts = screen_zones(floor.zones, world_root=floor.root, route=route, statics=floor.statics, min_route_clearance=P.MIN_ROUTE_CLEARANCE, robot_zones=P.robot_zones(robot or {}), clearance_map=clearance_map, world=floor.world)
    return frozenset(ref_safe_zones(verdicts))


def _type_doc(name: str, base: str, settings: Mapping[str, Any], needs: Mapping[str, NeedPlan], main: str, sequences: dict[str, Any]) -> dict[str, Any]:
    """A need-driven HumanSim type: `main` runs `sequences[main]`, preempts into `seek_<need>` once that need is low, and comes back to `main`."""
    doc: dict[str, Any] = {"name": name, "mode": "behavior_tree", "extends": base}
    if settings["local_planner"]:
        doc["local_planner"] = settings["local_planner"]
    if settings["local_planner_params"]:
        doc["local_planner_params"] = dict(settings["local_planner_params"])
    if needs:
        doc["needs"] = {n: {k: S.need(settings, n)[k] for k in ("initial", "decay_rate")} for n in needs}
        doc["utility_weights"] = {n: 1.0 for n in needs}
    seqs = dict(sequences)
    if needs:
        seqs[main] = {**seqs[main], "transitions": [{"when": {n: {"below": float(S.need(settings, n)["below"])}}, "goto": f"seek_{n}"} for n in needs]}
    for n, plan in needs.items():
        seqs[f"seek_{n}"] = {"steps": plan.steps(settings["patience"]), "then": main, "on_failure": main}
    doc["initial_sequence"] = main
    doc["sequences"] = seqs
    return doc


def _walk(points: Sequence[Point], stand: Mapping[str, float]) -> dict[str, Any]:
    return {f"to_{i + 1}": {"kind": "go_to", "target_pose": {"x": x, "y": y, "theta": 0.0}, "duration": dict(stand), "patience": patience_for(WALK_PATIENCE, stand)} for i, (x, y) in enumerate(points)}


@attrs.define
class Base:
    """What the people are drawn from: the unedited base world and its graph, the same for every edit of it."""

    name: str
    floor: Floor
    graph: Graph
    homes: frozenset[str]
    crowd: P.Crowd


def _base(site: Site, base: Site | None, settings: Mapping[str, Any], robot_spec: Mapping[str, Any] | None, seed: int, notes: list[str]) -> Base:
    """The crowd of `site.base`: `site` itself when unedited, else `base`, else the base world opened by name."""
    if site.edits:
        if base is None:
            from ..scenario.cli import world_root

            base = open_site(world_root(None, site.base), site.scenario, site.seed, [])
        if base.edits or base.name != site.base:
            raise ValueError(f"{base.name} is not the unedited base world {site.base!r} of {site.name}")
    else:
        base = site
    graph = read_graph(base.graph)
    floor = _floor(base.root, base.world, graph)
    homes = _homes(floor, robot_spec)
    crowd = P.crowd(
        floor.zones,
        overrides=dict(graph.purposes),
        homes=homes,
        forbidden=P.robot_zones(robot_spec or {}),
        seed=seed,
        world=site.base,
        count=settings["count"],
        scale=float(settings["scale"]),
        tag="flowedit",
    )
    if not crowd.people and settings["count"] != 0:
        notes.append("people: no room of a purpose to start in, nobody placed")
    return Base(site.base, floor, graph, homes, crowd)


def _start_in(floor: Floor, start: Point, name: str, notes: list[str]) -> Point:
    """`start`, drawn in the base world, as the edited `floor` lets a person stand there: unchanged while it is free, else
    the nearest free point, so only the people an edit's footprint or wall covers start elsewhere."""
    if floor.grid is None:
        return start
    local = (start[0] - floor.origin[0], start[1] - floor.origin[1])
    if floor.grid.free(local, PERSON_CLEAR):
        return start
    near = floor.grid.nearest_free(local, PERSON_CLEAR, 3.0)
    if near is None:
        notes.append(f"people: {name} starts at ({start[0]}, {start[1]}), not free in this world and no free point within 3 m, left there")
        return start
    moved = floor.to_map(near)
    notes.append(f"people: {name} starts at ({start[0]}, {start[1]}), not free in this world, moved to ({moved[0]}, {moved[1]})")
    return moved


def _people(base: Base, settings: Mapping[str, Any], needs: Mapping[str, NeedPlan], seed: int, floor: Floor | None = None, notes: list[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The people present at the start, as populate places and routes them in the base world, each with its own need-driven type.

    `floor` is the world the scenario is read into: a start point its edit made unfree moves to the nearest free point there."""
    notes = notes if notes is not None else []
    humans, types = [], {}
    for i, person in enumerate(base.crowd.people):
        name = f"flow_{i + 1}"
        point_rng = random.Random(f"flowedit:{base.name}:{seed}:{name}")
        points = [pt for room in person.rooms if (pt := base.floor.point(room, point_rng)) is not None]
        ref = f"./{TYPES_DIR}/flow__{name}.yaml"
        types[ref] = _type_doc(f"flow__{name}", person.kind, settings, needs, "idle", {"idle": {"steps": _walk(points, settings["idle"]["stand"]), "then": "idle", "on_failure": "idle"}})
        # a start point drawn here, not the home zone's name: the runtime draws a zone's point per world, so the base and
        # each of its edits would start the same person at another spot
        home_rng = random.Random(f"flowedit:{base.name}:{seed}:{name}:home")
        start = base.floor.point(person.home, home_rng)
        yaw = round(home_rng.uniform(-math.pi, math.pi), 2)
        if start is not None and floor is not None:
            start = _start_in(floor, start, name, notes)
        pose: Any = person.home if start is None else [start[0], start[1], yaw]
        humans.append({"name": name, "model": person.model, "pose": pose, "agent": {"agent_type": ref, "desired_velocity": person.speed}})
    return humans, types


def _strip(door: Door, inward: Point, depth: tuple[float, float], origin: Point) -> list[dict[str, float]]:
    (ax, ay), (bx, by) = (door.start.x, door.start.y), (door.end.x, door.end.y)
    length = math.dist((ax, ay), (bx, by))
    ux, uy = (bx - ax) / length, (by - ay) / length
    inset = min(0.15, length / 4)
    a = (ax + ux * inset, ay + uy * inset)
    b = (bx - ux * inset, by - uy * inset)
    near, far = depth
    corners = [(a[0] + inward[0] * near, a[1] + inward[1] * near), (b[0] + inward[0] * near, b[1] + inward[1] * near), (b[0] + inward[0] * far, b[1] + inward[1] * far), (a[0] + inward[0] * far, a[1] + inward[1] * far)]
    return [{"x": round(x + origin[0], 3), "y": round(y + origin[1], 3)} for x, y in corners]


def _entrances(floor: Floor, graph: Graph, base: Base, settings: Mapping[str, Any], needs: Mapping[str, NeedPlan], seed: int, notes: list[str]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """A source and a sink inside every exterior door of the edited site's level, and the visitor type each source releases,
    its stops drawn in the base world."""
    import shapely

    cfg = settings["entrances"]
    if float(cfg["rate"]) <= 0:
        return {}, {}
    level = floor.world.levels[floor.level_id]
    doors = {d.name: d for d in level.doors}
    regions: dict[str, Any] = {}
    types: dict[str, dict[str, Any]] = {}
    exterior = [d for d in graph.exterior_doors() if d in doors]
    if not exterior:
        notes.append("entrances: no door of the people's level lies in one room only, so no sources or sinks")
    for name in exterior:
        door, room = doors[name], graph.doors[name][0]
        shape = floor.shape(room)
        if shape is None:
            notes.append(f"door:{name}: its room {room!r} has no zone on the people's level, no entrance")
            continue
        (ax, ay), (bx, by) = (door.start.x, door.start.y), (door.end.x, door.end.y)
        length = math.dist((ax, ay), (bx, by))
        if length < 0.3:
            notes.append(f"door:{name}: span under 0.3 m, no entrance")
            continue
        normal = (-(by - ay) / length, (bx - ax) / length)
        mid = ((ax + bx) / 2, (ay + by) / 2)
        probe = sum(SOURCE_DEPTH) / 2
        inward = next((n for n in (normal, (-normal[0], -normal[1])) if shape.contains(shapely.Point(mid[0] + n[0] * probe, mid[1] + n[1] * probe))), None)
        if inward is None:
            notes.append(f"door:{name}: neither side lies in room {room!r}, no entrance")
            continue
        sink, source = f"{name}_sink", f"{name}_source"
        sink_poly = _strip(door, inward, SINK_DEPTH, floor.origin)
        leave_at = (round(sum(p["x"] for p in sink_poly) / 4, 2), round(sum(p["y"] for p in sink_poly) / 4, 2))
        rng = random.Random(f"flowedit:{base.name}:{seed}:visitor:{name}")
        rooms = [r for r in base.crowd.destinations if r != room]
        stops = rng.sample(rooms, min(int(cfg["stops"]), len(rooms))) if rooms else []
        points = [pt for r in stops if (pt := base.floor.point(r, rng)) is not None]
        ref = f"./{TYPES_DIR}/visitor__{name}.yaml"
        leave = {"leave": {"steps": {"exit": {"kind": "go_to", "target_pose": {"x": leave_at[0], "y": leave_at[1], "theta": 0.0}, "patience": dict(WALK_PATIENCE)}}, "then": "leave", "on_failure": "leave"}}
        visit = {"steps": _walk(points, settings["idle"]["stand"]), "then": "leave", "on_failure": "leave"} if points else {"steps": {"look": {"duration": dict(settings["idle"]["stand"])}}, "then": "leave"}
        types[ref] = _type_doc(f"visitor__{name}", "adult", settings, needs, "visit", {"visit": visit, **leave})
        regions[source] = {
            "type": "source",
            "polygon": _strip(door, inward, SOURCE_DEPTH, floor.origin),
            "config": {
                "rate_profile": [{"t": 0.0, "rate": float(cfg["rate"])}],
                "max_concurrent": int(cfg["max_concurrent"]),
                "max_total": int(cfg["max_total"]),
                "agent": {"agent_type": ref, "desired_velocity": dict(cfg["speed"]), "sink_affinity": [{"sink": sink, "weight": 1.0}]},
            },
        }
        regions[sink] = {"type": "sink", "polygon": sink_poly, "config": {"capacity": -1}}
    return regions, types


@attrs.define
class Result:
    doc: dict[str, Any]
    types: dict[str, dict[str, Any]]
    needs: dict[str, NeedPlan]
    notes: list[str]
    provenance: dict[str, Any]
    #: the world the scenario is written into
    root: Path


def read(site: Site, settings: Mapping[str, Any] | None = None, seed: int = 0, notes: list[str] | None = None, *, base: Site | None = None) -> Result:
    """The scenario of `site` (from `open_site`) under crowd `settings` and `seed`.

    The people come from the unedited base world (`base`, else opened by `site.base`), so every edit of one world gets the same
    people for the same settings and seed; the objects, needs, dwell and entrances come from `site`.
    """
    notes = notes if notes is not None else []
    settings = S.load(settings)
    graph = read_graph(site.graph)
    text = dumps(site.graph)
    floor = _floor(site.root, site.world, graph)
    objects = _objects(graph, floor, notes)
    needs = plan_needs(graph, settings, floor.level_id, notes)
    if not needs:
        notes.append("needs: no object on the people's level satisfies a need or affords a use, people only walk")
    robot = _robot(floor, settings)
    people_from = _base(site, base, settings, robot, seed, notes)
    if people_from.floor.level_id != floor.level_id:
        raise ValueError(f"{site.name}: people walk level {floor.level_id!r}, its base {site.base} level {people_from.floor.level_id!r}")
    humans, types = _people(people_from, settings, needs, seed, floor, notes)
    regions, visitor_types = _entrances(floor, graph, people_from, settings, needs, seed, notes)
    doc: dict[str, Any] = {"objects": objects, "humans": humans, "regions": regions}
    if robot is not None:
        doc["robots"] = [robot]
    provenance = {
        "generator": "flowedit read",
        "world": site.name,
        "base": site.base,
        "scenario": site.scenario,
        "edits": [e.id for e in site.edits],
        "graph_sha256": hashlib.sha256(text.encode()).hexdigest()[:12],
        "seed": seed,
        "settings_digest": S.digest(settings),
        "settings": dict(settings),
        "needs": {n: {"target": p.target, "interaction": p.kind, "meet": p.meet, "amount": p.amount, "dwell": p.dwell, "objects": list(p.objects), "synthesized": p.synthesized} for n, p in needs.items()},
        "entrances": sorted(k[: -len("_source")] for k in regions if k.endswith("_source")),
        "notes": list(notes),
    }
    return Result(doc, {**types, **visitor_types}, needs, notes, provenance, Path(site.root))


def _dump(doc: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(doc), sort_keys=False, default_flow_style=None, width=1 << 16)


def render(result: Result, name: str) -> dict[str, str]:
    """The files of `result` as `scenarios/<name>/` holds them, relative path to text, checked by the scenario model."""
    p = result.provenance
    header = "\n".join(
        (
            f"# FLOWEDIT scenario: {name} ({len(result.doc['humans'])} people, {len(p['entrances'])} entrance(s), needs: {', '.join(result.needs) or 'none'})",
            f"# generator: flowedit read, world {p['world']} (base {p['base']}), graph sha256:{p['graph_sha256']}, settings {p['settings_digest']}, seed {p['seed']}",
            "#",
            GENERATED_MARKER,
            "",
            "",
        )
    )
    files = {ref[2:]: _dump(doc) for ref, doc in sorted(result.types.items())}
    with tempfile.TemporaryDirectory() as stage:
        for rel, text in files.items():
            (Path(stage) / rel).parent.mkdir(parents=True, exist_ok=True)
            (Path(stage) / rel).write_text(text)
        try:
            files["scenario.yaml"] = scenario_text(result.doc, world_root=result.root, types_dir=Path(stage), header=header)
        except ValueError as exc:
            raise ValueError(f"scenarios/{name}: {exc}") from exc
    files[PROVENANCE] = _dump(p)
    return files


def write(result: Result, name: str, force: bool = False) -> Path:
    """Write `scenarios/<name>/` into the site's world, replacing only a generated scenario unless `force`, its stale type files removed."""
    if result.provenance["edits"] and not (result.root / EDITS).is_file():
        raise ValueError(f"{result.provenance['world']} is not saved: `flowedit edit --write` saves an edited world before its scenario is read into it")
    out = result.root / "scenarios" / name
    existing = out / "scenario.yaml"
    if existing.is_file() and not force and not is_generated(existing):
        raise WouldOverwrite(f"{existing} was not generated, pass --force to replace it or choose another --name")
    files = render(result, name)
    if (out / TYPES_DIR).is_dir():
        shutil.rmtree(out / TYPES_DIR)
    for rel, text in files.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text)
    return out
