"""Sample FLOWEDIT edits of every type from a seed: valid targets and plausible values, the same edits for the same seed."""

from __future__ import annotations

import enum
import math
import random
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from arena_simulation_setup.kg.scene import kind_of

from .build import zones_at
from .edit import EditType, LayoutOp, Site, apply, open_site

if TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.tree.World.model import Level, Zone

GENERATOR_VERSION = 1
MANIFEST_FORMAT = "flowedit-generated/1"
MAX_TRIES = 50
#: Function fields the generator samples; `satisfies` and `dwell` stay open until it is settled where PEDS reads them.
FIELDS = ("capacity",)
#: Plausible `capacity` per service type: the queue length at which a needs-driven ped passes the object over for another of its type.
CAPACITY: Mapping[str, tuple[int, int]] = {"fountain": (1, 4), "desk": (1, 6), "chair": (1, 2), "patient_bed": (1, 2)}
DEFAULT_CAPACITY = (1, 4)
SCHEDULE_START_S = (30, 300)
SCHEDULE_LENGTH_S = (30, 180)
#: Share of moves that stay in the object's own room; the rest go to another room of its level.
LOCAL_SHARE = 0.3
LOCAL_DISTANCE_M = (0.5, 3.0)
#: Zones no object is moved or inserted into, by name substring.
EXCLUDED_ZONES = ("stair", "elevator", "lift", "shaft")
YAWS = (0.0, 1.5708, 3.1416, -1.5708)
POINT_TRIES = 30
MARGIN_M = 0.3
WALL_GAP_M = 0.05
DOOR_CLEARANCE_M = 0.8
#: Half side of the box standing in for an object whose asset gives no footprint.
FALLBACK_HALF_M = 0.4
DECIMALS = 2
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


def rng_for(site: Site, seed: int, step: int, edit_type: EditType, index: int, purpose: str = "edit") -> random.Random:
    """The RNG of one step: keyed by the current world's name, which names the chain so far through its edit ids."""
    return random.Random(f"{site.name}:{site.scenario}:{seed}:{step}:{edit_type.value}:{index}:{purpose}")


def _facts(site: Site, subject: str, verb: str) -> list[Any]:
    return [f for f in site.graph.facts if f.subject == subject and f.verb == verb]


def objects(site: Site) -> list[str]:
    """The placed objects of the graph, by id."""
    return sorted({f.subject for f in site.graph.facts if f.verb == "at" and kind_of(f.subject) == "obj"})


def demand_objects(site: Site) -> list[str]:
    """The objects that serve something, by id."""
    return sorted({f.subject for f in site.graph.facts if f.verb == "serves"})


def _service(site: Site, obj: str) -> tuple[str, dict[str, Any]] | None:
    fact = next(iter(_facts(site, obj, "serves")), None)
    return None if fact is None else (str(fact.object).partition(":")[2], dict(fact.params))


def _prefer(rng: random.Random, preferred: Sequence[str], others: Sequence[str]) -> str:
    """One of `preferred` if there is any, else one of `others`."""
    pool = list(preferred) or list(others)
    if not pool:
        raise NoCandidate("no target")
    return rng.choice(pool)


_FOOTPRINTS: dict[tuple[Path, str], tuple[tuple[float, float], ...] | None] = {}


def resolve_footprints(site: Site, models: Sequence[str]) -> None:
    """Resolve the footprint ring of each of `models` at the origin, as the site's world resolves its assets, None when unannotated."""
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.World.view import WorldView, world_assets

    missing = sorted({m for m in models if (site.root, m) not in _FOOTPRINTS})
    if not missing:
        return
    with world_assets(WorldView(site.root), site.world):
        for model in missing:
            try:
                polygon = Identifier._run_sync(Obstacle.parse({"name": "probe", "model": model, "pose": [0.0, 0.0, 0.0]}).footprint())
            except Exception:  # noqa: BLE001 - any failure to resolve leaves the fallback box
                polygon = None
            _FOOTPRINTS[(site.root, model)] = None if polygon is None else tuple(polygon.exterior.coords)


def footprint(root: Path, model: str, x: float, y: float, yaw: float) -> shapely.Polygon:
    """The level-frame footprint of `model` posed at `x`, `y`, `yaw`: its asset's once resolved, else a `FALLBACK_HALF_M` box."""
    import shapely
    import shapely.affinity

    ring = _FOOTPRINTS.get((root, model))
    local = shapely.Polygon(ring) if ring else shapely.box(-FALLBACK_HALF_M, -FALLBACK_HALF_M, FALLBACK_HALF_M, FALLBACK_HALF_M)
    return shapely.affinity.translate(shapely.affinity.rotate(local, yaw, origin=(0, 0), use_radians=True), xoff=x, yoff=y)


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


class Spots:
    """Valid spots on one level: inside an allowed zone, clear of walls, door spans and the other objects' footprints."""

    def __init__(self, site: Site, level_id: str) -> None:
        import shapely

        self.site, self.level_id = site, level_id
        self.level = site.world.levels[level_id]
        walls = [shapely.LineString([(w.start.x, w.start.y), (w.end.x, w.end.y)]) for w in self.level.all_walls]
        doors = [shapely.LineString([(d.start.x, d.start.y), (d.end.x, d.end.y)]) for d in self.level.doors]
        self.walls = shapely.union_all(walls).buffer(WALL_GAP_M) if walls else shapely.Polygon()
        self.doors = shapely.union_all(doors).buffer(DOOR_CLEARANCE_M) if doors else shapely.Polygon()
        self.shapes = {z.name: z.shape() for z in self.level.zones if len(z.polygon) >= 3}
        self.inner = {name: shape.buffer(-WALL_GAP_M) for name, shape in self.shapes.items()}
        here = {obj: (placed[1], model) for obj in objects(site) if (placed := _at(site, obj)) is not None and placed[0] == level_id and (model := _model(site, obj)) is not None}
        resolve_footprints(site, [model for _, model in here.values()])
        self.placed: dict[str, shapely.Polygon] = {obj: footprint(site.root, model, float(pose["x"]), float(pose["y"]), float(pose["yaw"])) for obj, (pose, model) in here.items()}

    def valid(self, obj: str, model: str, zone: Zone, x: float, y: float, yaw: float) -> bool:
        resolve_footprints(self.site, [model])
        shape = footprint(self.site.root, model, x, y, yaw)
        if not self.inner[zone.name].covers(shape) or shape.intersects(self.walls) or shape.intersects(self.doors):
            return False
        grown = shape.buffer(MARGIN_M)
        return not any(other.intersects(grown) for ref, other in self.placed.items() if ref != obj)

    def in_zone(self, rng: random.Random, obj: str, model: str, zone: Zone) -> tuple[float, float, float] | None:
        """A valid pose drawn uniformly in `zone`."""
        import shapely

        shape = self.shapes[zone.name]
        x0, y0, x1, y1 = shape.bounds
        for _ in range(POINT_TRIES):
            x, y, yaw = _round(rng.uniform(x0, x1)), _round(rng.uniform(y0, y1)), rng.choice(YAWS)
            if shape.covers(shapely.Point(x, y)) and self.valid(obj, model, zone, x, y, yaw):
                return x, y, yaw
        return None

    def near(self, rng: random.Random, obj: str, model: str, zone: Zone, x: float, y: float) -> tuple[float, float, float] | None:
        """A valid pose in `zone` `LOCAL_DISTANCE_M` away from `x`, `y`."""
        import shapely

        shape = self.shapes[zone.name]
        for _ in range(POINT_TRIES):
            r, theta = rng.uniform(*LOCAL_DISTANCE_M), rng.uniform(0.0, 2 * math.pi)
            nx, ny, yaw = _round(x + r * math.cos(theta)), _round(y + r * math.sin(theta)), rng.choice(YAWS)
            if shape.covers(shapely.Point(nx, ny)) and self.valid(obj, model, zone, nx, ny, yaw):
                return nx, ny, yaw
        return None


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


def _move(site: Site, rng: random.Random) -> Sampled:
    """Move an object, one with demand first: within its own room (`local`) or to another room of its level (`room`)."""
    movable = [obj for obj in objects(site) if _model(site, obj) is not None]
    target = _prefer(rng, [obj for obj in demand_objects(site) if obj in movable], movable)
    level_id, pose = _at(site, target)  # type: ignore[misc] - every object of `objects` is placed
    spots = spots_for(site, level_id)
    zones = allowed_zones(spots.level)
    here = sorted(set(zones_at(spots.level, _position(pose))) & {z.name for z in zones})
    model = _model(site, target) or ""
    if here and rng.random() < LOCAL_SHARE:
        zone = next(z for z in zones if z.name == here[0])
        spot, scope = spots.near(rng, target, model, zone, float(pose["x"]), float(pose["y"])), Scope.LOCAL
    else:
        others = [z for z in zones if z.name not in here]
        if not others:
            raise NoCandidate("no other room to move to")
        spot, scope = spots.in_zone(rng, target, model, rng.choice(others)), Scope.ROOM
    if spot is None:
        raise ValueError("no valid spot")
    return Sampled(EditType.MOVE, target, {"x": spot[0], "y": spot[1], "yaw": spot[2]}, scope.value)


def _position(pose: Mapping[str, Any]) -> Any:  # noqa: ANN401 - a geometry Position
    from arena_simulation_setup.utils.geometry import Position

    return Position(float(pose["x"]), float(pose["y"]))


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.rsplit("/", 1)[-1].lower()).strip("_") or "object"


def _clone_demand(site: Site, obj: str) -> dict[str, Any]:
    """The function params that give a new object the demand of `obj`: its service, needs and dwell, copied rather than sampled."""
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
    return out


def _insert(site: Site, rng: random.Random) -> Sampled:
    """Insert a copy of an object, one with demand first, its demand copied, in a room of its level (`room`)."""
    modelled = [obj for obj in objects(site) if _model(site, obj) is not None]
    source = _prefer(rng, [obj for obj in demand_objects(site) if obj in modelled], modelled)
    level_id, _ = _at(site, source)  # type: ignore[misc] - every object of `objects` is placed
    model = _model(site, source) or ""
    taken = set(site.graph.entities) | {f"obj:{n}" for names in site.world.names().values() for n in names}
    name = next(f"obj:gen_{_slug(model)}_{i}" for i in range(1, len(taken) + 2) if f"obj:gen_{_slug(model)}_{i}" not in taken)
    spots = spots_for(site, level_id)
    zones = allowed_zones(spots.level)
    if not zones:
        raise NoCandidate("no zone to insert into")
    spot = spots.in_zone(rng, name, model, rng.choice(zones))
    if spot is None:
        raise ValueError("no valid spot")
    return Sampled(EditType.INSERT, name, {"model": model, "x": spot[0], "y": spot[1], "yaw": spot[2], "level": level_id, **_clone_demand(site, source)}, Scope.ROOM.value)


def _door_rooms(site: Site) -> dict[str, set[str]]:
    """The rooms each door connects, by door id."""
    rooms: dict[str, set[str]] = {}
    for fact in site.graph.facts:
        if fact.verb == "connects" and fact.params.get("door"):
            rooms.setdefault(str(fact.params["door"]), set()).update({fact.subject, str(fact.object)})
    return rooms


def _layout(site: Site, rng: random.Random) -> Sampled:
    """Move a door along its wall, add one between two rooms, or remove one whose rooms keep another door."""
    ops = [op for op, _ in DOOR_OPS]
    weights = [w for _, w in DOOR_OPS]
    first = rng.choices(ops, weights)[0]
    for op in [first, *(o for o in ops if o is not first)]:
        sampled = {LayoutOp.MOVE: _move_door, LayoutOp.ADD: _add_door, LayoutOp.REMOVE: _remove_door}[op](site, rng)
        if sampled is not None:
            return sampled
    raise NoCandidate("no door to change")


def _doors(site: Site) -> list[tuple[str, Any]]:
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


def _remove(site: Site, rng: random.Random) -> Sampled:
    """Remove an object, one with demand when the site has any."""
    return Sampled(EditType.REMOVE, _prefer(rng, demand_objects(site), objects(site)), {})


def _function(site: Site, rng: random.Random) -> Sampled:
    """Change the `capacity` of a serving object, objects whose service type others share first, as only they let peds pass one over.

    An object with capacity 0 queues without bound and is left alone.
    """
    served = {obj: service for obj in demand_objects(site) if (service := _service(site, obj)) is not None and service[1].get("capacity", 1) != 0}
    by_type: dict[str, list[str]] = {}
    for obj, (kind, _) in served.items():
        by_type.setdefault(kind, []).append(obj)
    changeable = [obj for obj, (kind, params) in served.items() if _capacities(kind, params)]
    target = _prefer(rng, [obj for obj in changeable if len(by_type[served[obj][0]]) > 1], changeable)
    kind, params = served[target]
    return Sampled(EditType.FUNCTION, target, {"capacity": rng.choice(_capacities(kind, params))})


def _capacities(kind: str, params: Mapping[str, Any]) -> list[int]:
    """The plausible capacities of a `kind` object other than its current one."""
    low, high = CAPACITY.get(kind, DEFAULT_CAPACITY)
    return [n for n in range(low, high + 1) if n != int(params.get("capacity", 1))]


def _access(site: Site, rng: random.Random) -> Sampled:
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


def _schedule_regime(schedule: Any) -> str | None:  # noqa: ANN401 - a world schedule
    from .build import presets

    return next((p.get("regime") for p in presets(schedule.semantics, "schedule") if p.get("regime")), None)


SAMPLERS: dict[EditType, Callable[[Site, random.Random], Sampled]] = {
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
        return "type" in sampled.params
    if sampled.type in (EditType.MOVE, EditType.REMOVE):
        return sampled.target in demand_objects(site)
    return False


def sample_step(site: Site, edit_type: EditType, seed: int, step: int, index: int, notes: list[str]) -> tuple[Site, dict[str, Any]] | None:
    """`site` with one valid edit of `edit_type` applied and its manifest record, None when the site gives none in `MAX_TRIES` tries."""
    sampler = SAMPLERS.get(edit_type)
    if sampler is None:
        notes.append(f"{site.name}: no sampler for {edit_type.value} edits")
        return None
    rng = rng_for(site, seed, step, edit_type, index)
    for _ in range(MAX_TRIES):
        try:
            sampled = sampler(site, rng)
            edited = apply(site, sampled.type, sampled.target, sampled.params)
        except NoCandidate as exc:
            notes.append(f"{site.name}: no {edit_type.value} edit, {exc}")
            return None
        except ValueError:
            continue
        record = {"type": sampled.type.value, "target": sampled.target, "id": edited.edits[-1].id, "function_carrying": function_carrying(site, sampled)}
        if sampled.scope is not None:
            record["scope"] = sampled.scope
        return edited, record
    notes.append(f"{site.name}: no valid {edit_type.value} edit in {MAX_TRIES} tries")
    return None


def generate(site: Site, seed: int = 0, per_type: int = 1, steps: int = 1, types: Sequence[EditType | str] = tuple(EditType), notes: list[str] | None = None) -> list[Generated]:
    """Per type, `per_type` chains of up to `steps` edits on `site`, each chain starting with an edit of that type.

    A chain's later steps draw their type from `types`. A chain whose first edit fails is left out, one whose later edit fails ends there,
    both with a note. The same site, seed and code give the same chains, and a shorter chain is a prefix of a longer one.
    """
    notes = notes if notes is not None else []
    if steps < 1 or per_type < 1:
        raise ValueError("steps and per_type must be at least 1")
    kinds = [EditType(t) for t in types]
    chains: list[Generated] = []
    for edit_type in kinds:
        for index in range(per_type):
            current, records = site, []
            for step in range(steps):
                step_type = edit_type if step == 0 else rng_for(current, seed, step, edit_type, index, "type").choice(kinds)
                result = sample_step(current, step_type, seed, step, index, notes)
                if result is None:
                    break
                current, record = result
                records.append(record)
            if records:
                chains.append(Generated(current, tuple(records)))
    return chains


class Control(enum.StrEnum):
    INERT_OBJECT = "inert_object"
    MOVED_WITHOUT_DEMAND = "moved_without_demand"


def controls(site: Site, chain: Generated) -> list[tuple[Control, Site]]:
    """The geometry-only controls of a chain's first edit, when it carries function, applied to `site`, the chain's base.

    For a move or insert: the same model, without demand, at the edit's spot (`inert_object`), the original left in place.
    For a move also: the object moved and stripped of its demand (`moved_without_demand`). Other edits get none.
    """
    first = chain.site.edits[0]
    if not chain.records[0]["function_carrying"] or first.type not in (EditType.MOVE, EditType.INSERT):
        return []
    if first.type is EditType.MOVE:
        model = _model(site, first.target)
        placed = _at(site, first.target)
        if model is None or placed is None:
            return []
        level_id, pose = placed
        spot = {k: first.params.get(k, pose[k]) for k in ("x", "y", "yaw")}
        inert_name = f"{first.target}_inert"
        moved = apply(site, EditType.MOVE, first.target, first.params)
        return [
            (Control.INERT_OBJECT, apply(site, EditType.INSERT, inert_name, {"model": model, **spot, "level": str(first.params.get("level", level_id))})),
            (Control.MOVED_WITHOUT_DEMAND, apply(moved, EditType.FUNCTION, first.target, {"clear": True})),
        ]
    keep = ("model", "x", "y", "yaw", "level")
    return [(Control.INERT_OBJECT, apply(site, EditType.INSERT, f"{first.target}_inert", {k: v for k, v in first.params.items() if k in keep}))]


def dump_manifest(site: Site, chains: Sequence[Generated], seed: int, per_type: int, steps: int, worlds: Mapping[str, Path] | None = None, chain_controls: Mapping[str, Sequence[tuple[Control, Site]]] | None = None) -> dict[str, Any]:
    """The manifest of a run: what was generated from which site, one entry per chain, with its controls when made."""
    entries = []
    for chain in chains:
        entry: dict[str, Any] = {"world": chain.site.name, "edits": [dict(r) for r in chain.records]}
        if worlds is not None and chain.site.name in worlds:
            entry["path"] = str(worlds[chain.site.name])
        made = (chain_controls or {}).get(chain.site.name) or []
        if made:
            entry["controls"] = [{"kind": kind.value, "world": control.name, "ids": [e.id for e in control.edits], **({"path": str(worlds[control.name])} if worlds is not None and control.name in worlds else {})} for kind, control in made]
        entries.append(entry)
    return {"format": MANIFEST_FORMAT, "generator_version": GENERATOR_VERSION, "base": site.name, "scenario": site.scenario, "seed": seed, "per_type": per_type, "steps": steps, "fields": list(FIELDS), "chains": entries}

