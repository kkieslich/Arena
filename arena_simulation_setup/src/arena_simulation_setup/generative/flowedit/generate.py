"""Sample FLOWEDIT edits of every type from a seed: valid targets and plausible values, the same edits for the same seed."""

from __future__ import annotations

import enum
import math
import random
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from arena_simulation_setup.kg.scene import kind_of
from arena_simulation_setup.utils.face import front_angle as annotated_front
from arena_simulation_setup.utils.geometry import Position

from .build import presets, zones_at
from .edit import DEMAND_SUFFIX, Edit, EditType, LayoutOp, Site, apply, world_object

if TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.kg import Fact
    from arena_simulation_setup.metrics.use_points import Finding
    from arena_simulation_setup.shared.world import Door, Schedule
    from arena_simulation_setup.tree.World.model import Level, Zone

GENERATOR_VERSION = 6
MANIFEST_FORMAT = "flowedit-generated/1"
MAX_TRIES = 50
#: Function fields the generator samples. The reader writes `dwell` into the `duration` of the step at the object; `satisfies` is not drawn yet.
FIELDS = ("capacity", "dwell")
#: Plausible `capacity` per service type: the queue length at which a needs-driven ped passes the object over for another of its type.
CAPACITY: Mapping[str, tuple[int, int]] = {"fountain": (1, 4), "desk": (1, 6), "chair": (1, 2), "patient_bed": (1, 2)}
DEFAULT_CAPACITY = (1, 4)
#: Plausible mean dwell (s) per service type: 3-20 s for dispensers (study design). Other types are still to decide and take that range.
DWELL_S: Mapping[str, tuple[float, float]] = {"fountain": (3.0, 20.0)}
DEFAULT_DWELL_S = (3.0, 20.0)
SCHEDULE_START_S = (30, 300)
SCHEDULE_LENGTH_S = (30, 180)
#: Share of moves that stay in the object's own room; the rest go to another room of its level.
LOCAL_SHARE = 0.3
LOCAL_DISTANCE_M = (0.5, 3.0)
#: Zones no object is moved or inserted into, by name substring.
EXCLUDED_ZONES = ("stair", "elevator", "lift", "shaft")
#: A world object standing higher (m) sits on furniture or a wall: spots are drawn on the floor, so it is neither moved nor copied.
FLOOR_HEIGHT_M = 0.05
#: Suffix of the `inert_object` control's copy.
INERT_SUFFIX = "_inert"
#: A wall running along a zone edge at most this far (m) off the edge's line stands behind that part of the edge, and objects go against it.
WALL_SNAP_M = 0.3
#: Most a wall may turn away from a zone edge's direction (rad) and still run along it.
WALL_PARALLEL_RAD = math.radians(10.0)
#: Shorter walled parts of a zone edge (jogs round a pillar, a door reveal, a wall stub) take no object.
WALL_EDGE_MIN_M = 0.6
#: Slack (m) past the wall gap, so spots rounded to `DECIMALS` stay valid.
WALL_SLACK_M = 0.01
POINT_TRIES = 30
MARGIN_M = 0.3
WALL_GAP_M = 0.05
DOOR_CLEARANCE_M = 0.8
#: Half side of the box standing in for an object whose asset gives no footprint.
FALLBACK_HALF_M = 0.4
DECIMALS = 2
#: Farthest (m) rounding a spot to `DECIMALS` moves it.
ROUNDING_M = math.sqrt(2.0) * 0.5 * 10.0**-DECIMALS
#: Inset (m) of the back's ends from the footprint's sides where a wall must stand behind it too.
BACK_END_INSET_M = 0.1
#: Turns of an object at a wall spot from front into the room, tried in order: into the room, then along the wall either way
TURNS = (0.0, math.pi / 2, -math.pi / 2)
#: Door layout ops and their weights; a door is moved along its wall by `DOOR_SHIFT_M`, an added door spans `DOOR_SPAN_M`.
DOOR_OPS = ((LayoutOp.MOVE, 0.5), (LayoutOp.ADD, 0.25), (LayoutOp.REMOVE, 0.25))
DOOR_SHIFT_M = (0.5, 2.0)
DOOR_SPAN_M = 1.5
DOOR_END_GAP_M = 0.5


def _round(value: float) -> float:
    """`value` to `DECIMALS`, without a negative zero, so equal spots give equal edit ids."""
    return round(value, DECIMALS) + 0.0


class Scope(enum.StrEnum):
    LOCAL = "local"
    ROOM = "room"


class NoCandidate(Exception):
    """The site offers no target for an edit type."""


class Redraw(Exception):
    """A draw that gave no usable proposal, drawn again."""


@attrs.frozen
class Sampled:
    """One proposed edit, `scope` naming how far a move or insert reaches."""

    type: EditType
    target: str
    params: Mapping[str, Any]
    scope: str | None = None


@attrs.frozen
class Generated:
    """A chain of sampled edits: the edited site and, per edit, what the manifest records."""

    site: Site
    records: tuple[Mapping[str, Any], ...]


#: Whether an edit of a type, target and params passes the use-point check (`refusal`), for a sampler trying spots
Fits = Callable[[EditType, str, Mapping[str, Any]], bool]
#: Whether a pose x, y, yaw of the object being placed passes
PoseFits = Callable[[float, float, float], bool]


def rng_for(site: Site, seed: int, step: int, edit_type: EditType, index: int, purpose: str = "edit") -> random.Random:
    """The RNG of one step: keyed by the current world's key (`Site.key`, `<base>__<last edit id>`), which names the chain
    so far through its edit ids; the readable part of the world's name never enters a draw."""
    return random.Random(f"{site.key}:{site.scenario}:{seed}:{step}:{edit_type.value}:{index}:{purpose}")


def _facts(site: Site, subject: str, verb: str) -> list[Fact]:
    return [f for f in site.graph.facts if f.subject == subject and f.verb == verb]


def objects(site: Site) -> list[str]:
    """The placed objects of the graph, by id."""
    return sorted({f.subject for f in site.graph.facts if f.verb == "at" and kind_of(f.subject) == "obj"})


def floor_objects(site: Site) -> list[str]:
    """The placed objects with a model that stand on the floor, by id: the ones a move or insert may draw."""
    raised = {f"obj:{o.name}" for level in site.world.levels.values() for o in level.objects if o.pose.position.z > FLOOR_HEIGHT_M}
    return [obj for obj in objects(site) if obj not in raised and _model(site, obj) is not None]


def demand_objects(site: Site) -> list[str]:
    """The objects that satisfy a need, by id: in PEDS only needs send people to an object."""
    return sorted({f.subject for f in site.graph.facts if f.verb == "satisfies"})


def _needs(site: Site, obj: str) -> set[str]:
    return {str(f.object) for f in _facts(site, obj, "satisfies")}


def sole_server(site: Site, obj: str) -> bool:
    """Whether no other object satisfies a need of `obj`. The reader takes one dwell per need, so only then may a dwell edit differ."""
    needs = _needs(site, obj)
    return all(other == obj or not needs & _needs(site, other) for other in demand_objects(site))


def _service(site: Site, obj: str) -> tuple[str, dict[str, Any]] | None:
    fact = next(iter(_facts(site, obj, "serves")), None)
    return None if fact is None else (str(fact.object).partition(":")[2], dict(fact.params))


def _prefer(rng: random.Random, preferred: Sequence[str], others: Sequence[str], wide: bool = False) -> str:
    """One of `preferred` if there is any, else one of `others`; any of `others` when `wide` (a retry after a duplicate)."""
    pool = list(others) if wide else list(preferred) or list(others)
    if not pool:
        raise NoCandidate("no target")
    return rng.choice(pool)


_FOOTPRINTS: dict[tuple[Path, str], shapely.Polygon | None] = {}
_FACES: dict[tuple[Path, str], str | None] = {}
#: Why a model did not resolve, by world root and model: its footprint and front are then the fallback's
_UNRESOLVED: dict[tuple[Path, str], str] = {}


def _docked_models(site: Site) -> set[str]:
    """The models, as the graph names them, of the objects of the site's world that have a dock."""
    from arena_simulation_setup.tree.World.write import short_identifier

    names = [short_identifier(o.model) for level in site.world.levels.values() for o in level.objects if o.dock is not None]
    return {str(name[0]) if isinstance(name, list) else name for name in names}


def resolve_footprints(site: Site, models: Sequence[str], strict: bool = False) -> None:
    """Resolve the footprint ring and annotated front (`face`) of each of `models` at the origin, as the site's world
    resolves its assets, None when unannotated. A model that does not resolve keeps the fallback, a `FALLBACK_HALF_M`
    box with its front at pi, so the generator still runs on a world with unresolved decor; but ValueError naming the
    model when `strict` or when an object of the site's world with that model has a dock, which the fallback front can
    turn to the wall."""
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.World.view import WorldView, world_assets

    missing = sorted({m for m in models if (site.root, m) not in _FOOTPRINTS})
    if missing:
        with world_assets(WorldView(site.root), site.world):
            for model in missing:
                probe = Obstacle.parse({"name": "probe", "model": model, "pose": [0.0, 0.0, 0.0]})
                try:
                    view = Identifier._run_sync(probe.model.resolve())
                    polygon = Identifier._run_sync(probe.footprint())
                    face = (view.annotation or {}).get("face")
                except Exception as exc:  # noqa: BLE001 - any failure to resolve leaves the fallback box, its front at pi
                    polygon, face = None, None
                    _UNRESOLVED[(site.root, model)] = f"{type(exc).__name__}: {exc}"
                _FOOTPRINTS[(site.root, model)] = polygon
                _FACES[(site.root, model)] = None if face is None else str(face)
    unresolved = sorted({m for m in models if (site.root, m) in _UNRESOLVED})
    if not unresolved:
        return
    docked = _docked_models(site)
    refused = [
        f"model {m!r} of a docked object does not resolve ({_UNRESOLVED[(site.root, m)]}): the fallback box faces pi and can turn its dock to the wall"
        if m in docked
        else f"model {m!r} does not resolve ({_UNRESOLVED[(site.root, m)]}): its footprint and front are unknown"
        for m in unresolved
        if strict or m in docked
    ]
    if refused:
        raise ValueError("; ".join(refused))


def footprint(root: Path, model: str, x: float, y: float, yaw: float) -> shapely.Polygon:
    """The level-frame footprint of `model` posed at `x`, `y`, `yaw`: its asset's once resolved, else a `FALLBACK_HALF_M` box."""
    import shapely
    import shapely.affinity

    local = _FOOTPRINTS.get((root, model))
    if local is None:
        local = shapely.box(-FALLBACK_HALF_M, -FALLBACK_HALF_M, FALLBACK_HALF_M, FALLBACK_HALF_M)
    return shapely.affinity.translate(shapely.affinity.rotate(local, yaw, origin=(0, 0), use_radians=True), xoff=x, yoff=y)


def front_angle(root: Path, model: str) -> float:
    """The angle in `model`'s frame of the side people queue at, once resolved (`resolve_footprints`): its annotated front, else pi, behind the yaw as HumanSim queues an object without a known `face`."""
    front = annotated_front(_FACES.get((root, model)))
    return math.pi if front is None else front


def allowed_zones(level: Level) -> list[Zone]:
    """The zones objects may go to, by name."""
    return sorted((z for z in level.zones if len(z.polygon) >= 3 and not any(n in z.name.lower() for n in EXCLUDED_ZONES)), key=lambda z: z.name)


def _at(site: Site, obj: str) -> tuple[str, dict[str, Any]] | None:
    fact = next(iter(_facts(site, obj, "at")), None)
    return None if fact is None else (str(fact.object).partition(":")[2], dict(fact.params))


def _model(site: Site, obj: str) -> str | None:
    entity = site.graph.entities.get(obj)
    model = entity.props.get("model") if entity is not None else None
    return None if model is None else str(model)


Segment = tuple[tuple[float, float], tuple[float, float]]


@attrs.frozen
class WallPiece:
    """A part `line` of a zone edge with a wall behind it: the edge's unit `direction` and unit `normal` into the zone,
    a point `origin` on the edge's line, and the `walls` (start, end) running along the edge."""

    line: shapely.LineString
    origin: tuple[float, float]
    direction: tuple[float, float]
    normal: tuple[float, float]
    walls: tuple[Segment, ...]

    def depth(self, x: float, y: float) -> float:
        """How far `x`, `y` lies into the zone from the edge's line (m), negative behind it."""
        return (x - self.origin[0]) * self.normal[0] + (y - self.origin[1]) * self.normal[1]

    def offset(self, x: float, y: float) -> float | None:
        """The depth of the wall behind `x`, `y`: of the walls along the edge at `x`, `y`'s place along it and within
        `WALL_SNAP_M` of its line, the one lying furthest into the zone; None without one."""
        (ox, oy), (ux, uy) = self.origin, self.direction
        t = (x - ox) * ux + (y - oy) * uy
        best = None
        for (sx, sy), (ex, ey) in self.walls:
            ts, te = (sx - ox) * ux + (sy - oy) * uy, (ex - ox) * ux + (ey - oy) * uy
            if te == ts or not min(ts, te) <= t <= max(ts, te):
                continue
            k = (t - ts) / (te - ts)
            d = self.depth(sx + (ex - sx) * k, sy + (ey - sy) * k)
            if abs(d) <= WALL_SNAP_M and (best is None or d > best):
                best = d
        return best


def wall_pieces(shape: shapely.Polygon, walls: Sequence[Segment]) -> list[WallPiece]:
    """The parts of `shape`'s outer edges with a wall behind them, in ring order: where a wall runs along the edge, at
    most `WALL_PARALLEL_RAD` off its direction and `WALL_SNAP_M` off its line. Each part is at least `WALL_EDGE_MIN_M`
    long, so a door gap only cuts its edge and a short wall stub or a wall meeting the edge across gives no part."""
    import shapely

    ring = list(shape.exterior.coords)
    out: list[WallPiece] = []
    for a, b in zip(ring, ring[1:], strict=False):
        length = math.dist(a, b)
        if length < WALL_EDGE_MIN_M:
            continue
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        nx, ny = (-uy, ux) if shape.covers(shapely.Point(mx - uy * 0.1, my + ux * 0.1)) else (uy, -ux)
        along: list[Segment] = []
        spans: list[tuple[float, float]] = []
        for (sx, sy), (ex, ey) in walls:
            wl = math.hypot(ex - sx, ey - sy)
            if wl == 0.0 or abs(ux * (ey - sy) - uy * (ex - sx)) / wl > math.sin(WALL_PARALLEL_RAD):
                continue
            ts, te = (sx - a[0]) * ux + (sy - a[1]) * uy, (ex - a[0]) * ux + (ey - a[1]) * uy
            ds, de = (sx - a[0]) * nx + (sy - a[1]) * ny, (ex - a[0]) * nx + (ey - a[1]) * ny
            # the share of the wall within WALL_SNAP_M of the edge's line, its depth changing linearly along it
            if de == ds:
                lo, hi = (0.0, 1.0) if abs(ds) <= WALL_SNAP_M else (1.0, 0.0)
            else:
                k1, k2 = (-WALL_SNAP_M - ds) / (de - ds), (WALL_SNAP_M - ds) / (de - ds)
                lo, hi = max(0.0, min(k1, k2)), min(1.0, max(k1, k2))
            if lo > hi:
                continue
            t0, t1 = sorted((ts + (te - ts) * lo, ts + (te - ts) * hi))
            t0, t1 = max(t0, 0.0), min(t1, length)
            if t1 > t0:
                along.append(((sx, sy), (ex, ey)))
                spans.append((t0, t1))
        merged: list[list[float]] = []
        for t0, t1 in sorted(spans):
            if merged and t0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], t1)
            else:
                merged.append([t0, t1])
        for t0, t1 in merged:
            if t1 - t0 >= WALL_EDGE_MIN_M:
                line = shapely.LineString([(a[0] + ux * t0, a[1] + uy * t0), (a[0] + ux * t1, a[1] + uy * t1)])
                out.append(WallPiece(line, (a[0], a[1]), (ux, uy), (nx, ny), tuple(along)))
    return out


class Spots:
    """Valid spots on one level: in an allowed zone, clear of walls, door spans and every object's footprint, a moved
    object's own included. Spots stand against a wall of their zone, as close as those gaps allow, the model's annotated
    front turned away from the wall into the room."""

    def __init__(self, site: Site, level_id: str) -> None:
        import shapely

        self.site, self.level_id = site, level_id
        self.level = site.world.levels[level_id]
        self.wall_segments: list[Segment] = [((w.start.x, w.start.y), (w.end.x, w.end.y)) for w in self.level.all_walls]
        walls = [shapely.LineString(segment) for segment in self.wall_segments]
        doors = [shapely.LineString([(d.start.x, d.start.y), (d.end.x, d.end.y)]) for d in self.level.doors]
        self.wall_lines = shapely.union_all(walls) if walls else shapely.GeometryCollection()
        self.walls = self.wall_lines.buffer(WALL_GAP_M) if walls else shapely.Polygon()
        self.doors = shapely.union_all(doors).buffer(DOOR_CLEARANCE_M) if doors else shapely.Polygon()
        self.shapes = {z.name: z.shape() for z in self.level.zones if len(z.polygon) >= 3}
        self.inner = {name: shape.buffer(-WALL_GAP_M) for name, shape in self.shapes.items()}
        here = {obj: (placed[1], model) for obj in objects(site) if (placed := _at(site, obj)) is not None and placed[0] == level_id and (model := _model(site, obj)) is not None}
        resolve_footprints(site, [model for _, model in here.values()])
        self.placed: dict[str, shapely.Polygon] = {obj: footprint(site.root, model, float(pose["x"]), float(pose["y"]), float(pose["yaw"])) for obj, (pose, model) in here.items()}

    def valid(self, model: str, zone: Zone, x: float, y: float, yaw: float) -> bool:
        resolve_footprints(self.site, [model])
        shape = footprint(self.site.root, model, x, y, yaw)
        if not self.inner[zone.name].covers(shape) or shape.intersects(self.walls) or shape.intersects(self.doors):
            return False
        grown = shape.buffer(MARGIN_M)
        return not any(other.intersects(grown) for other in self.placed.values())

    def wall_edges(self, zone: Zone) -> list[WallPiece]:
        """The parts of `zone`'s edges with a wall behind them (`wall_pieces`), in ring order."""
        return wall_pieces(self.shapes[zone.name], self.wall_segments)

    def against_wall(self, rng: random.Random, model: str, zone: Zone, accept: Callable[[float, float], bool], around: tuple[float, float, float] | None = None, fits: PoseFits | None = None) -> tuple[float, float, float] | None:
        """A valid pose in `zone` against one of its walls that `accept` and `fits` take: a point drawn along the walled
        parts of its edges by length (only those within `around` = x, y, radius when given), the model turned so its
        front points along the edge's normal, else along the wall either way (`turns`), its back as close to the wall
        behind and the zone edge as the wall gap allows, a wall behind the middle and both ends of the back
        (`backs_onto_wall`); the next point only when no turn at this one passes."""
        import shapely
        import shapely.affinity

        resolve_footprints(self.site, [model])
        pieces = self.wall_edges(zone)
        if around is not None:
            disk = shapely.Point(around[0], around[1]).buffer(around[2])
            near: list[WallPiece] = []
            for piece in pieces:
                clipped = piece.line.intersection(disk)
                near += [attrs.evolve(piece, line=part) for part in getattr(clipped, "geoms", [clipped]) if isinstance(part, shapely.LineString) and part.length > 0.05]
            pieces = near
        if not pieces:
            return None
        lengths = [piece.line.length for piece in pieces]
        local = footprint(self.site.root, model, 0.0, 0.0, 0.0)
        for _ in range(POINT_TRIES):
            piece = rng.choices(pieces, weights=lengths)[0]
            nx, ny = piece.normal
            p = piece.line.interpolate(rng.uniform(0.0, 1.0), normalized=True)
            offset = piece.offset(p.x, p.y)
            if offset is None:
                continue
            for turn in TURNS:
                yaw = math.atan2(ny, nx) + turn - front_angle(self.site.root, model)
                yaw = round(math.atan2(math.sin(yaw), math.cos(yaw)), 4) + 0.0
                turned = shapely.affinity.rotate(local, yaw, origin=(0, 0), use_radians=True)
                back = max(-(vx * nx + vy * ny) for vx, vy in turned.exterior.coords)
                # the back stops at the wall gap from both the wall line and the zone edge, whichever lies further in
                behind = max(offset, 0.0) + WALL_GAP_M + WALL_SLACK_M
                x, y = _round(p.x + nx * (behind + back)), _round(p.y + ny * (behind + back))
                if accept(x, y) and backs_onto_wall(piece, turned, x, y) and self.valid(model, zone, x, y, yaw) and (fits is None or fits(x, y, yaw)):
                    return x, y, yaw
        return None

    def in_zone(self, rng: random.Random, model: str, zone: Zone, fits: PoseFits | None = None) -> tuple[float, float, float] | None:
        """A valid pose against a wall of `zone`."""
        return self.against_wall(rng, model, zone, lambda _x, _y: True, fits=fits)

    def near(self, rng: random.Random, model: str, zone: Zone, x: float, y: float, fits: PoseFits | None = None) -> tuple[float, float, float] | None:
        """A valid pose against a wall of `zone` `LOCAL_DISTANCE_M` away from `x`, `y`."""
        lo, hi = LOCAL_DISTANCE_M
        # wall points a little beyond reach still give spots within it, the object standing out from the wall
        return self.against_wall(rng, model, zone, lambda nx, ny: lo <= math.dist((nx, ny), (x, y)) <= hi, around=(x, y, hi + 1.0), fits=fits)


def backs_onto_wall(piece: WallPiece, turned: shapely.Polygon, x: float, y: float) -> bool:
    """Whether a wall of `piece` stands behind the footprint `turned` placed at `x`, `y`: behind the middle of its back
    at the wall gap, the slack and the rounding from the wall line or from the zone edge, whichever lies further in, and
    behind both ends of its back (`BACK_END_INSET_M` in from its sides), so the object never backs onto an opening."""
    (nx, ny), (ux, uy) = piece.normal, piece.direction
    coords = list(turned.exterior.coords)
    back = piece.depth(x, y) - max(-(vx * nx + vy * ny) for vx, vy in coords)
    lo, hi = min(vx * ux + vy * uy for vx, vy in coords), max(vx * ux + vy * uy for vx, vy in coords)
    inset = min(BACK_END_INSET_M, (hi - lo) / 4.0)
    for i, along in enumerate((0.0, lo + inset, hi - inset)):
        wall = piece.offset(x + ux * along, y + uy * along)
        if wall is None:
            return False
        gap = back - wall
        most = WALL_GAP_M + WALL_SLACK_M + max(-wall, 0.0) + ROUNDING_M if i == 0 else math.inf
        if not WALL_GAP_M - ROUNDING_M <= gap <= most:
            return False
    return True


_SPOTS: list[Spots] = []


def spots_for(site: Site, level_id: str) -> Spots:
    """The `Spots` of a level of `site`, kept while the same site is sampled from so retries do not recompute every footprint."""
    if any(s.site is not site for s in _SPOTS):
        _SPOTS.clear()
    hit = next((s for s in _SPOTS if s.level_id == level_id), None)
    if hit is None:
        hit = Spots(site, level_id)
        _SPOTS.append(hit)
    return hit


def _move(site: Site, rng: random.Random, wide: bool = False, fits: Fits | None = None) -> Sampled:
    """Move an object on the floor, one with demand first: within its own room (`local`) or to another room of its level (`room`)."""
    movable = floor_objects(site)
    target = _prefer(rng, [obj for obj in demand_objects(site) if obj in movable], movable, wide)
    placed = _at(site, target)
    if placed is None:
        raise NoCandidate(f"{target} has no pose")
    level_id, pose = placed
    spots = spots_for(site, level_id)
    zones = allowed_zones(spots.level)
    here = sorted(set(zones_at(spots.level, _position(pose))) & {z.name for z in zones})
    model = _model(site, target) or ""
    placing = None if fits is None else lambda x, y, yaw: fits(EditType.MOVE, target, {"x": x, "y": y, "yaw": yaw})
    if here and rng.random() < LOCAL_SHARE:
        zone = next(z for z in zones if z.name == here[0])
        spot, scope = spots.near(rng, model, zone, float(pose["x"]), float(pose["y"]), placing), Scope.LOCAL
    else:
        others = [z for z in zones if z.name not in here]
        if not others:
            raise NoCandidate("no other room to move to")
        spot, scope = spots.in_zone(rng, model, rng.choice(others), placing), Scope.ROOM
    if spot is None:
        raise Redraw("no valid spot")
    return Sampled(EditType.MOVE, target, {"x": spot[0], "y": spot[1], "yaw": spot[2]}, scope.value)


def _position(pose: Mapping[str, Any]) -> Position:
    return Position(float(pose["x"]), float(pose["y"]))


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.rsplit("/", 1)[-1].lower()).strip("_") or "object"


def _clone_demand(site: Site, obj: str) -> dict[str, Any]:
    """The function params that give a new object the demand of `obj`: its service, needs, dwell and formation (anchored at the new object), copied rather than sampled."""
    service = _service(site, obj)
    if service is None:
        return {}
    kind, params = service
    out: dict[str, Any] = {"type": kind, **{k: params[k] for k in ("capacity", "interaction_radius") if k in params}}
    satisfies = {str(f.object).partition(":")[2]: f.params["amount"] for f in _facts(site, obj, "satisfies")}
    if satisfies:
        out["satisfies"] = satisfies
    dwell = next(iter(_facts(site, obj, "dwells")), None)
    if dwell is not None:
        out["dwell"] = {"mean": float(dwell.object), **{k: v for k, v in dwell.params.items() if k != "from"}}
    formation = next(iter(_facts(site, obj, "has_formation")), None)
    if formation is not None:
        out["formation"] = {"type": str(formation.object).partition(":")[2], "params": {k: v for k, v in formation.params.items() if k not in ("anchor_kind", "anchor_ref")}}
    return out


def _clone_dock(site: Site, obj: str) -> dict[str, Any]:
    """The `dock` and `charge_power_w` params that give a new object the dock of `obj` and, for a charger, its charge
    power, none for an object without a dock in `world.yaml`."""
    found = world_object(site.world, obj.partition(":")[2])
    source = None if found is None else found[0].objects[found[1]]
    if source is None or source.dock is None:
        return {}
    out: dict[str, Any] = {"dock": [[p.x, p.y] for p in source.dock]}
    if source.charge_power_w is not None:
        out["charge_power_w"] = source.charge_power_w
    return out


def _insert(site: Site, rng: random.Random, wide: bool = False, fits: Fits | None = None) -> Sampled:
    """Insert a copy of an object on the floor, one with demand first, its demand, dock and charge power copied, in a room of its level (`room`)."""
    modelled = floor_objects(site)
    source = _prefer(rng, [obj for obj in demand_objects(site) if obj in modelled], modelled, wide)
    placed = _at(site, source)
    if placed is None:
        raise NoCandidate(f"{source} has no pose")
    level_id = placed[0]
    model = _model(site, source) or ""
    taken = set(site.graph.entities) | {f"obj:{n}" for names in site.world.names().values() for n in names}
    name = next(f"obj:gen_{_slug(model)}_{i}" for i in range(1, len(taken) + 2) if f"obj:gen_{_slug(model)}_{i}" not in taken)
    spots = spots_for(site, level_id)
    zones = allowed_zones(spots.level)
    if not zones:
        raise NoCandidate("no zone to insert into")
    copied = {**_clone_demand(site, source), **_clone_dock(site, source)}
    placing = None if fits is None else lambda x, y, yaw: fits(EditType.INSERT, name, {"model": model, "x": x, "y": y, "yaw": yaw, "level": level_id, **copied})
    spot = spots.in_zone(rng, model, rng.choice(zones), placing)
    if spot is None:
        raise Redraw("no valid spot")
    return Sampled(EditType.INSERT, name, {"model": model, "x": spot[0], "y": spot[1], "yaw": spot[2], "level": level_id, **copied}, Scope.ROOM.value)


def _door_rooms(site: Site) -> dict[str, set[str]]:
    """The rooms each door connects, by door id."""
    rooms: dict[str, set[str]] = {}
    for fact in site.graph.facts:
        if fact.verb == "connects" and fact.params.get("door"):
            rooms.setdefault(str(fact.params["door"]), set()).update({fact.subject, str(fact.object)})
    return rooms


def _layout(site: Site, rng: random.Random, wide: bool = False, fits: Fits | None = None) -> Sampled:
    """Move a door along its wall, add one between two rooms, or remove one whose rooms keep another door."""
    ops = [op for op, _ in DOOR_OPS]
    weights = [w for _, w in DOOR_OPS]
    first = rng.choices(ops, weights)[0]
    for op in [first, *(o for o in ops if o is not first)]:
        sampled = {LayoutOp.MOVE: _move_door, LayoutOp.ADD: _add_door, LayoutOp.REMOVE: _remove_door}[op](site, rng)
        if sampled is not None:
            return sampled
    raise NoCandidate("no door to change")


def _doors(site: Site) -> list[tuple[str, Door]]:
    return sorted(((level_id, door) for level_id, level in site.world.levels.items() for door in level.doors), key=lambda item: item[1].name)


def _move_door(site: Site, rng: random.Random) -> Sampled | None:
    doors = _doors(site)
    if not doors:
        return None
    _, door = rng.choice(doors)
    length = math.hypot(door.end.x - door.start.x, door.end.y - door.start.y)
    if length == 0:
        return None
    shift = rng.uniform(*DOOR_SHIFT_M) * rng.choice((-1.0, 1.0))
    offset = [_round(shift * (door.end.x - door.start.x) / length), _round(shift * (door.end.y - door.start.y) / length)]
    return Sampled(EditType.LAYOUT, f"door:{door.name}", {"op": LayoutOp.MOVE.value, "offset": offset})


def spare_doors(site: Site) -> list[str]:
    """The doors between rooms that each keep another door, so removing or locking one cuts no room off."""
    rooms = _door_rooms(site)
    doors_of: dict[str, set[str]] = {}
    for door, connected in rooms.items():
        for room in connected:
            doors_of.setdefault(room, set()).add(door)
    return sorted(door for door, connected in rooms.items() if all(len(doors_of[room]) > 1 for room in connected))


def _remove_door(site: Site, rng: random.Random) -> Sampled | None:
    removable = spare_doors(site)
    return Sampled(EditType.LAYOUT, rng.choice(removable), {"op": LayoutOp.REMOVE.value}) if removable else None


def _add_door(site: Site, rng: random.Random) -> Sampled | None:
    """A door of `DOOR_SPAN_M` on a wall with a different room on each side, clear of the doors already there."""
    import shapely

    candidates = []
    for level_id, level in sorted(site.world.levels.items()):
        doors = [shapely.LineString([(d.start.x, d.start.y), (d.end.x, d.end.y)]) for d in level.doors]
        for wall in level.all_walls:
            length = math.hypot(wall.end.x - wall.start.x, wall.end.y - wall.start.y)
            if length >= DOOR_SPAN_M + 2 * DOOR_END_GAP_M:
                candidates.append((level_id, level, wall, length, doors))
    if not candidates:
        return None
    taken = {f"door:{n}" for names in site.world.names().values() for n in names} | set(site.graph.entities)
    name = next(f"door:gen_door_{i}" for i in range(1, len(taken) + 2) if f"door:gen_door_{i}" not in taken)
    for _ in range(POINT_TRIES):
        level_id, level, wall, length, doors = rng.choice(candidates)
        dx, dy = (wall.end.x - wall.start.x) / length, (wall.end.y - wall.start.y) / length
        t = rng.uniform(DOOR_END_GAP_M + DOOR_SPAN_M / 2, length - DOOR_END_GAP_M - DOOR_SPAN_M / 2)
        cx, cy = wall.start.x + dx * t, wall.start.y + dy * t
        span = [[_round(cx - dx * DOOR_SPAN_M / 2), _round(cy - dy * DOOR_SPAN_M / 2)], [_round(cx + dx * DOOR_SPAN_M / 2), _round(cy + dy * DOOR_SPAN_M / 2)]]
        line = shapely.LineString(span)
        if any(line.distance(d) < DOOR_END_GAP_M for d in doors):
            continue
        sides = [set(zones_at(level, _position({"x": cx - dy * s, "y": cy + dx * s}))) for s in (0.3, -0.3)]
        if sides[0] and sides[1] and not sides[0] & sides[1]:
            return Sampled(EditType.LAYOUT, name, {"op": LayoutOp.ADD.value, "level": level_id, "span": span})
    return None


def _remove(site: Site, rng: random.Random, wide: bool = False, fits: Fits | None = None) -> Sampled:
    """Remove an object, one with demand when the site has any."""
    return Sampled(EditType.REMOVE, _prefer(rng, demand_objects(site), objects(site), wide), {})


def _function(site: Site, rng: random.Random, wide: bool = False, fits: Fits | None = None) -> Sampled:
    """Change the `capacity` of an object with a need, shared service types first, or the `dwell` of a `sole_server`."""
    needy = demand_objects(site)
    served = {obj: service for obj in needy if (service := _service(site, obj)) is not None and service[1].get("capacity", 1) != 0 and _capacities(*service)}
    by_type: dict[str, list[str]] = {}
    for obj, (kind, _) in served.items():
        by_type.setdefault(kind, []).append(obj)
    sized = [obj for obj in served if len(by_type[served[obj][0]]) > 1] or list(served)
    options = [(obj, "capacity") for obj in sized] + [(obj, "dwell") for obj in needy if sole_server(site, obj)]
    if not options:
        raise NoCandidate("no object with a need")
    target, field = rng.choice(options)
    if field == "capacity":
        kind, params = served[target]
        return Sampled(EditType.FUNCTION, target, {"capacity": rng.choice(_capacities(kind, params))})
    return Sampled(EditType.FUNCTION, target, {"dwell": _new_dwell(site, target, rng)})


def _new_dwell(site: Site, obj: str, rng: random.Random) -> dict[str, float]:
    """A new mean dwell in the range of `obj`'s service type, with the authored spread (`std`, `clip_low`, `clip_high`) scaled to it."""
    kind = (_service(site, obj) or ("", {}))[0]
    low, high = DWELL_S.get(kind, DEFAULT_DWELL_S)
    fact = next(iter(_facts(site, obj, "dwells")), None)
    current = None if fact is None else float(fact.object)
    mean = round(rng.uniform(low, high), 1)
    if mean == current:
        raise Redraw("drew the current dwell")
    spread = {k: float(v) for k, v in (fact.params if fact is not None else {}).items() if k in ("std", "clip_low", "clip_high")}
    scale = mean / current if current else 1.0
    return {"mean": mean, **{k: round(v * scale, 2) for k, v in spread.items()}}


def _capacities(kind: str, params: Mapping[str, Any]) -> list[int]:
    """The plausible capacities of a `kind` object other than its current one."""
    low, high = CAPACITY.get(kind, DEFAULT_CAPACITY)
    return [n for n in range(low, high + 1) if n != int(params.get("capacity", 1))]


def _access(site: Site, rng: random.Random, wide: bool = False, fits: Fits | None = None) -> Sampled:
    """Lock a spare door, unlock or ungate a gated one, or give a schedule a window."""
    gated_doors = sorted({f.subject for f in site.graph.facts if f.verb == "gated"})
    doors = sorted(set(spare_doors(site)) | set(gated_doors))
    schedules = sorted(f"schedule:{s.name}" for s in site.world.schedules if _schedule_regime(s))
    target = rng.choice(doors + schedules) if doors or schedules else None
    if target is None:
        raise NoCandidate("no door or schedule")
    if kind_of(target) == "schedule":
        start = rng.randrange(SCHEDULE_START_S[0], SCHEDULE_START_S[1] + 1, 10)
        end = start + rng.randrange(SCHEDULE_LENGTH_S[0], SCHEDULE_LENGTH_S[1] + 1, 10)
        return Sampled(EditType.ACCESS, target, {"windows": [{"start": start, "end": end}]})
    gated = _facts(site, target, "gated")
    if not gated:
        return Sampled(EditType.ACCESS, target, {"locked": True})
    locked = bool(gated[0].object)
    if rng.random() < 0.5 or (not locked and target not in spare_doors(site)):
        return Sampled(EditType.ACCESS, target, {"remove": True})
    return Sampled(EditType.ACCESS, target, {"locked": not locked})


def _schedule_regime(schedule: Schedule) -> str | None:
    return next((p.get("regime") for p in presets(schedule.semantics, "schedule") if p.get("regime")), None)


#: Samplers by type: `(site, rng, wide, fits)`, `wide` set on a retry after a duplicate so a move, insert or remove may
#: take any object, `fits` the use-point check a move or insert tries its spots with.
SAMPLERS: dict[EditType, Callable[[Site, random.Random, bool, Fits | None], Sampled]] = {
    EditType.MOVE: _move,
    EditType.INSERT: _insert,
    EditType.REMOVE: _remove,
    EditType.FUNCTION: _function,
    EditType.ACCESS: _access,
    EditType.LAYOUT: _layout,
}


def function_carrying(site: Site, sampled: Sampled) -> bool:
    """Whether the edit moves, adds, removes or changes an object's demand, rather than geometry alone."""
    if sampled.type is EditType.FUNCTION:
        return True
    if sampled.type is EditType.INSERT:
        return "satisfies" in sampled.params
    if sampled.type in (EditType.MOVE, EditType.REMOVE):
        return sampled.target in demand_objects(site)
    return False


_FINDINGS: dict[tuple[Path, str], list[Finding]] = {}


def site_findings(site: Site) -> list[Finding]:
    """The use-point findings of a site as the reader hands it to the runtime, before any crowd is read from it
    (`metrics.use_points.check`): its world and the graph's objects the world does not hold, kept by the site's key."""
    from arena_simulation_setup.metrics.use_points import check
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.Scenario import Scenario
    from arena_simulation_setup.tree.World.view import WorldView

    from .read import read_graph

    key = (site.root, site.key)
    if key not in _FINDINGS:
        held = {o.name for level in site.world.levels.values() for o in level.objects}
        things = read_graph(site.graph).objects
        objects = [Obstacle.parse({"name": name, "model": t.model, "pose": [t.x, t.y, t.yaw], "level": t.floor, "humansim": t.block()}) for name, t in sorted(things.items()) if name not in held]
        _FINDINGS[key] = check(WorldView(site.root), site.world, Scenario(objects=objects))
    return _FINDINGS[key]


def refusal(site: Site, edited: Site, touched: Collection[str]) -> str | None:
    """Why `edited` is refused against `site`, its parent: a point finding of an object the edit `touched` (by name), or
    one `site` did not have. Unresolved models are `resolve_footprints`' to refuse."""
    before = {(f.object, f.point) for f in site_findings(site)}
    for finding in site_findings(edited):
        if finding.point != "model" and (finding.object in touched or (finding.object, finding.point) not in before):
            return str(finding)
    return None


def _touched(edit_type: EditType, target: str, params: Mapping[str, Any]) -> set[str]:
    """The objects, by name, whose use points an edit moves, adds or gives demand: its target and a left-behind demand node."""
    name = target.partition(":")[2]
    if edit_type not in (EditType.MOVE, EditType.INSERT, EditType.FUNCTION):
        return set()
    return {name, f"{name}{DEMAND_SUFFIX}"} if params.get("leave_demand") else {name}


def _edit_refusal(site: Site, edit_type: EditType, target: str, params: Mapping[str, Any], step: int) -> tuple[Site, str | None]:
    """`site` with the edit and why it is refused (`refusal`), its geometry-only controls checked too on a chain's first step (ValueError when `apply` refuses it)."""
    edited = apply(site, edit_type, target, params)
    why = refusal(site, edited, _touched(edit_type, target, params))
    if why is None and step == 0 and function_carrying(site, Sampled(edit_type, target, params)):
        for control, variant, touched in _control_sites(site, edited.edits[-1]):
            if (why := refusal(site, variant, touched)) is not None:
                return edited, f"{control.value} control: {why}"
    return edited, why


def sample_step(site: Site, edit_type: EditType, seed: int, step: int, index: int, notes: list[str], taken: set[str] | None = None) -> tuple[Site, dict[str, Any]] | None:
    """`site` with one valid edit of `edit_type` whose world is not in `taken` yet and whose use points pass (`refusal`),
    and its manifest record, else None and a note. A move or insert tries the turns of the model at a spot first."""
    sampler = SAMPLERS.get(edit_type)
    if sampler is None:
        notes.append(f"{site.name}: no sampler for {edit_type.value} edits")
        return None
    taken = taken if taken is not None else set()
    rng = rng_for(site, seed, step, edit_type, index)
    wide = False
    refused = "no draw gave a proposal"

    def fits(kind: EditType, target: str, params: Mapping[str, Any]) -> bool:
        nonlocal refused
        try:
            why = _edit_refusal(site, kind, target, params, step)[1]
        except ValueError as exc:
            why = str(exc)
        if why is not None:
            refused = f"last refusal: {why}"
        return why is None

    for _ in range(MAX_TRIES):
        try:
            sampled = sampler(site, rng, wide, fits)
        except NoCandidate as exc:
            notes.append(f"{site.name}: no {edit_type.value} edit, {exc}")
            return None
        except Redraw:
            continue
        try:
            edited, why = _edit_refusal(site, sampled.type, sampled.target, sampled.params, step)
        except ValueError as exc:
            refused = f"last refusal: {exc}"
            continue
        if why is not None:
            refused = f"last refusal: {why}"
            continue
        if edited.key in taken:
            wide = True
            continue
        taken.add(edited.key)
        record = {"type": sampled.type.value, "target": sampled.target, "id": edited.edits[-1].id, "function_carrying": function_carrying(site, sampled)}
        if sampled.scope is not None:
            record["scope"] = sampled.scope
        return edited, record
    notes.append(f"{site.name}: no valid {edit_type.value} edit in {MAX_TRIES} tries, {refused}")
    return None


def _step_types(rng: random.Random, kinds: Sequence[EditType]) -> list[EditType]:
    """The type drawn for a later step first, then the other types in a drawn order."""
    first = rng.choice(kinds)
    rest = [k for k in kinds if k is not first]
    rng.shuffle(rest)
    return [first, *rest]


def generate(site: Site, seed: int = 0, per_type: int = 1, steps: int = 1, types: Sequence[EditType | str] = tuple(EditType), notes: list[str] | None = None) -> list[Generated]:
    """Per type, `per_type` chains of up to `steps` edits on `site`, each chain starting with an edit of that type."""
    notes = notes if notes is not None else []
    if steps < 1 or per_type < 1:
        raise ValueError("steps and per_type must be at least 1")
    kinds = [EditType(t) for t in types]
    chains: list[Generated] = []
    taken: list[set[str]] = [set() for _ in range(steps)]
    for edit_type in kinds:
        for index in range(per_type):
            current, records = site, []
            for step in range(steps):
                order = [edit_type] if step == 0 else _step_types(rng_for(current, seed, step, edit_type, index, "type"), kinds)
                tried: list[str] = []
                result = None
                for step_type in order:
                    result = sample_step(current, step_type, seed, step, index, tried, taken[step])
                    if result is not None:
                        break
                if result is None:
                    notes.extend(tried)
                    break
                current, record = result
                records.append(record)
            if records:
                chains.append(Generated(current, tuple(records)))
    return chains


class Control(enum.StrEnum):
    """Geometry-only controls: only the footprint changes, the demand stays where it was."""

    INERT_OBJECT = "inert_object"
    MOVED_WITHOUT_DEMAND = "moved_without_demand"


def own_edits(site: Site, chain: Site) -> list[Edit]:
    """The edits `chain` adds to `site`, its base; ValueError when `site` is not its base."""
    if chain.edits[: len(site.edits)] != site.edits:
        raise ValueError(f"{site.name} is not the base of {chain.name}")
    return chain.edits[len(site.edits) :]


def _control_sites(site: Site, first: Edit) -> list[tuple[Control, Site, set[str]]]:
    """The geometry-only controls of a function-carrying first edit, applied to `site`, its base, each with the objects it touches."""
    if first.type is EditType.MOVE:
        model = _model(site, first.target)
        placed = _at(site, first.target)
        if model is None or placed is None:
            return []
        level_id, pose = placed
        spot = {k: first.params.get(k, pose[k]) for k in ("x", "y", "yaw")}
        inert = {"model": model, **spot, "level": str(first.params.get("level", level_id)), **_clone_dock(site, first.target)}
        moved = {**{k: v for k, v in first.params.items() if k in ("x", "y", "yaw", "level")}, "leave_demand": True}
        return [
            (Control.INERT_OBJECT, apply(site, EditType.INSERT, f"{first.target}{INERT_SUFFIX}", inert), {f"{first.target.partition(':')[2]}{INERT_SUFFIX}"}),
            (Control.MOVED_WITHOUT_DEMAND, apply(site, EditType.MOVE, first.target, moved), _touched(EditType.MOVE, first.target, moved)),
        ]
    if first.type is EditType.INSERT:
        keep = ("model", "x", "y", "yaw", "level", "dock", "charge_power_w")
        return [(Control.INERT_OBJECT, apply(site, EditType.INSERT, f"{first.target}{INERT_SUFFIX}", {k: v for k, v in first.params.items() if k in keep}), {f"{first.target.partition(':')[2]}{INERT_SUFFIX}"})]
    return []


def controls(site: Site, chain: Generated) -> list[tuple[Control, Site]]:
    """The geometry-only controls of a chain's first edit when it carries function, applied to `site`, the chain's base."""
    first = own_edits(site, chain.site)[0]
    if not chain.records[0]["function_carrying"]:
        return []
    return [(control, variant) for control, variant, _ in _control_sites(site, first)]


def dump_manifest(site: Site, chains: Sequence[Generated], seed: int, per_type: int, steps: int, worlds: Mapping[str, Path] | None = None, chain_controls: Mapping[str, Sequence[tuple[Control, Site]]] | None = None) -> dict[str, Any]:
    """The manifest of a run: what was generated from which site, one entry per chain, with its controls when made."""
    entries = []
    for chain in chains:
        entry: dict[str, Any] = {"world": chain.site.name, "edits": [dict(r) for r in chain.records]}
        if worlds is not None and chain.site.name in worlds:
            entry["path"] = str(worlds[chain.site.name])
        made = (chain_controls or {}).get(chain.site.name) or []
        if made:
            entry["controls"] = [{"kind": kind.value, "world": control.name, "ids": [e.id for e in own_edits(site, control)], **({"path": str(worlds[control.name])} if worlds is not None and control.name in worlds else {})} for kind, control in made]
        entries.append(entry)
    return {"format": MANIFEST_FORMAT, "generator_version": GENERATOR_VERSION, "base": site.name, "scenario": site.scenario, "seed": seed, "per_type": per_type, "steps": steps, "fields": list(FIELDS), "chains": entries}
