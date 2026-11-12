"""Edit a FLOWEDIT site: one operation per edit type, each pairing the graph change with the world change, saved as a new world."""

from __future__ import annotations

import copy
import enum
import hashlib
import json
import math
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import yaml

from arena_simulation_setup.kg import Scene, load, write
from arena_simulation_setup.kg.scene import DERIVED, kind_of
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

from .build import build, presets, sync_access, zones_at

if TYPE_CHECKING:
    import shapely

    from arena_simulation_setup.shared.entities import Named, Obstacle
    from arena_simulation_setup.shared.walls import Wall
    from arena_simulation_setup.shared.world import Door
    from arena_simulation_setup.tree.World.model import Level, World

FORMAT = "flowedit-edits/1"
GRAPH = "flowgraph.yaml"
EDITS = "edits.yaml"
SEPARATOR = "__"
ID_LENGTH = 8
#: Longest readable part of an edited world's name (`describe`), the id after it keeps names unique.
SLUG_LENGTH = 40
SERVICE_KEYS = ("capacity", "interaction_radius")
DWELL_KEYS = ("mean", "std", "clip_low", "clip_high")
#: How an object is used (`affords`) is part of its demand: it moves with a left demand and goes with a cleared one.
DEMAND_VERBS = ("serves", "satisfies", "has_formation", "dwells", "affords")
FUNCTION_PARAMS = ("type", *SERVICE_KEYS, "satisfies", "dwell", "formation", "clear")
#: Params that only switch an operation on: false names the same edit as leaving them out.
FLAGS = ("leave_demand", "clear", "remove")
#: Suffix of the demand-only node a `move` with `leave_demand` leaves at the object's old spot.
DEMAND_SUFFIX = "_demand"
#: The state a schedule window written without a `value` takes while open
SCHEDULE_ON = "on"
DOOR_FIELDS = ("kind", "width", "height", "material", "activation_distance", "transition_time", "hold_time", "swing_direction")
WALL_FIELDS = ("closed", "kind", "material")
_ON_WALL_M = 0.05


class EditType(enum.StrEnum):
    MOVE = "move"
    INSERT = "insert"
    REMOVE = "remove"
    FUNCTION = "function"
    ACCESS = "access"
    LAYOUT = "layout"


class LayoutOp(enum.StrEnum):
    ADD = "add"
    MOVE = "move"
    REMOVE = "remove"


def _as_floats(value: Any) -> Any:  # noqa: ANN401 - YAML input
    """`value` with every int a float, so `15` and `15.0` hash alike."""
    if isinstance(value, Mapping):
        return {k: _as_floats(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_as_floats(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def _canonical(params: Mapping[str, Any]) -> dict[str, Any]:
    """`params` as they name an edit: ints as floats, false `FLAGS` left out."""
    return {k: _as_floats(v) for k, v in params.items() if v or k not in FLAGS}


@attrs.frozen
class Edit:
    """One edit: its `type`, the `target` id it changes, its `params`, and the world it was applied to."""

    type: EditType = attrs.field(converter=EditType)
    target: str
    params: Mapping[str, Any]
    parent: str

    @property
    def id(self) -> str:
        """Short hash of the whole record, the parent included, so it also names the chain."""
        record = json.dumps({"parent": self.parent, "type": self.type, "target": self.target, "params": _canonical(self.params)}, sort_keys=True)
        return hashlib.sha256(record.encode()).hexdigest()[:ID_LENGTH]

    def dump(self) -> dict[str, Any]:
        return {"id": self.id, "parent": self.parent, "type": self.type.value, "target": self.target, "params": dict(self.params)}


@attrs.define
class Site:
    """A world without its scenario objects, its FLOWEDIT graph, and the edits that led to them from the world `base`."""

    world: World
    graph: Scene
    root: Path
    base: str
    scenario: str | None = None
    seed: int = 0
    edits: list[Edit] = attrs.field(factory=list)

    @property
    def key(self) -> str:
        """The site's identity: `base` for an unedited site, else `<base>__<last edit id>`; an edit names its parent by it."""
        return f"{self.base}{SEPARATOR}{self.edits[-1].id}" if self.edits else self.base

    @property
    def name(self) -> str:
        """The world name: `base` for an unedited site, else `<base>__<what the edits did>__<last edit id>`."""
        return SEPARATOR.join((self.base, describe(self.edits), self.edits[-1].id)) if self.edits else self.base


def describe(edits: Sequence[Edit]) -> str:
    """What `edits` did, readable in a world name: the last edit's type, layout op, target and switched-on flags, after
    the chain's length when it has more than one edit. Lowercase words joined by single underscores, at most SLUG_LENGTH."""
    last = edits[-1]
    words = [last.type.value, str(last.params.get("op", "")), last.target.partition(":")[2] or last.target, *(k for k in FLAGS if last.params.get(k))]
    if len(edits) > 1:
        words.insert(0, f"{len(edits)}_edits")
    slug = re.sub(r"[^a-z0-9]+", "_", "_".join(w for w in words if w).lower()).strip("_")
    return slug[:SLUG_LENGTH].rstrip("_") or last.type.value


def _load_world(root: Path) -> World:
    """The world at `root` with only its stated KG, as it is written back."""
    from arena_simulation_setup.tree.World.view import WorldView

    world = WorldView(root.resolve()).load(derive="off")
    if world.kg is not None and not world.kg.entities and not world.kg.facts and not world.kg.constraints:
        world.kg = None
    return world


def open_site(root: Path, scenario: str | None = None, seed: int | None = None, notes: list[str] | None = None) -> Site:
    """The site of the world at `root`: an edited world from its `edits.yaml` and graph, else the graph `build` makes of it and `scenario`."""
    if (root / EDITS).is_file():
        doc = yaml.safe_load((root / EDITS).read_text()) or {}
        if doc.get("format") != FORMAT:
            raise ValueError(f"{root / EDITS}: format: expected {FORMAT!r}, got {doc.get('format')!r}")
        if scenario is not None and scenario != doc.get("scenario"):
            raise ValueError(f"{root.name} was edited from scenario {doc.get('scenario')!r}, not {scenario!r}")
        saved_seed = int(doc.get("seed", 0))
        if seed is not None and seed != saved_seed:
            raise ValueError(f"{root.name} was edited with seed {saved_seed}, not {seed}")
        edits = [Edit(type=e["type"], target=e["target"], params=e.get("params") or {}, parent=e["parent"]) for e in doc.get("edits") or []]
        return Site(_load_world(root), load(root / GRAPH), root, str(doc["base"]), doc.get("scenario"), saved_seed, edits)
    path = root / "scenarios" / scenario / "scenario.yaml" if scenario else None
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"no scenario {scenario!r} at {path}")
    seed = seed or 0
    return Site(_load_world(root), build(root, path, seed, notes), root, root.name, scenario, seed)


def _copy(site: Site) -> Site:
    from arena_simulation_setup.kg import dump

    return attrs.evolve(site, world=copy.deepcopy(site.world), graph=load(dump(site.graph)), edits=list(site.edits))


def _name(target: str, kind: str) -> str:
    if kind_of(target) != kind:
        raise ValueError(f"target {target!r} is not of the form {kind}:<name>")
    return target.partition(":")[2]


def _layout_op(params: dict[str, Any]) -> LayoutOp:
    """Pop the `op` param, `move` by default."""
    op = params.pop("op", LayoutOp.MOVE)
    if op not in tuple(LayoutOp):
        raise ValueError(f"op {op!r} is not one of {', '.join(LayoutOp)}")
    return LayoutOp(op)


def _at(graph: Scene, obj: str) -> tuple[str, dict[str, Any]] | None:
    """The level id and pose params of `obj`'s `at` fact."""
    fact = next((f for f in graph.facts if f.subject == obj and f.verb == "at"), None)
    return None if fact is None else (str(fact.object).partition(":")[2], dict(fact.params))


def _find(world: World, items: Callable[[Level], Sequence[Named]], name: str) -> tuple[Level, int] | None:
    """The level and index of the item `name` among `items(level)`."""
    for level in world.levels.values():
        for i, item in enumerate(items(level)):
            if item.name == name:
                return level, i
    return None


def world_object(world: World, name: str) -> tuple[Level, int] | None:
    """The level and index of the world object `name`, None when no level has it."""
    return _find(world, lambda level: level.objects, name)


def _door(world: World, name: str) -> tuple[Level, int] | None:
    return _find(world, lambda level: level.doors, name)


def _level(world: World, level_id: object) -> Level:
    if level_id is None:
        if len(world.levels) != 1:
            raise ValueError(f"the world has levels {list(world.levels)}, name one as level")
        return next(iter(world.levels.values()))
    if str(level_id) not in world.levels:
        raise ValueError(f"no level {level_id!r} (levels: {list(world.levels)})")
    return world.levels[str(level_id)]


def _check_params(params: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(f"unknown params {unknown}, expected some of {sorted(allowed)}")


def _point(value: Any, what: str) -> Position:  # noqa: ANN401 - YAML input
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{what}: expected [x, y], got {value!r}")
    return Position(float(value[0]), float(value[1]))


def _rederive(world: World, graph: Scene) -> None:
    """Recompute `contains` and `connects` of `graph` from the world's geometry and the graph's own object poses."""
    from arena_simulation_setup.tree.World.membership import derive

    graph.drop_derived()
    derive(attrs.evolve(world, kg=graph))
    world_objects = {f"obj:{o.name}" for level in world.levels.values() for o in level.objects}
    stated = {str(f.object) for f in graph.stated() if f.verb == "contains"}
    for obj in [ref for ref in graph.entities if kind_of(ref) == "obj" and ref not in world_objects and ref not in stated]:
        placed = _at(graph, obj)
        if placed is None or placed[0] not in world.levels:
            continue
        level_id, pose = placed
        for room in zones_at(world.levels[level_id], Position(float(pose["x"]), float(pose["y"]))):
            graph.add_entity(f"room:{room}")
            graph.add_fact(f"room:{room}", "contains", obj, source=DERIVED)


def _front_pose(site: Site, model: object, pose: Mapping[str, Any]) -> dict[str, Any]:
    """`pose` turned so the demand-only node is queued at from the side people used `model` from. The runtime queues an
    object without an annotated `face`, as a node without a model is, behind its yaw, and `model` at
    `generate.front_angle`, so the node's yaw is the model's yaw plus that angle minus pi."""
    from .generate import front_angle, resolve_footprints  # generate imports this module

    if not isinstance(model, str) or not model:
        return dict(pose)
    resolve_footprints(site, [model])
    turn = math.remainder(front_angle(site.root, model) - math.pi, math.tau)
    if abs(turn) < 1e-9:
        return dict(pose)
    yaw = float(pose.get("yaw", 0.0)) + turn
    return {**pose, "yaw": round(math.atan2(math.sin(yaw), math.cos(yaw)), 4)}


def _leave_demand(site: Site, obj: str, level_id: str, pose: Mapping[str, Any]) -> None:
    """Move the demand facts of `obj` (`DEMAND_VERBS`, so also how people use it) onto a new graph-only node
    `<obj>_demand` without a model, posed where `obj` stands and turned so people queue at the same side (`_front_pose`)."""
    demand = [f for f in site.graph.facts if f.subject == obj and f.verb in DEMAND_VERBS]
    if not demand:
        raise ValueError(f"{obj} has no demand to leave")
    old_name = _name(obj, "obj")
    node = f"{obj}{DEMAND_SUFFIX}"
    name = _name(node, "obj")
    if node in site.graph.entities or any(name in names for names in site.world.names().values()):
        raise ValueError(f"{name!r} is already taken")
    site.graph.add_entity(node)
    site.graph.add_fact(node, "at", f"floor:{level_id}", _front_pose(site, site.graph.entities[obj].props.get("model"), pose))
    for fact in demand:
        params = dict(fact.params)
        if fact.verb == "has_formation" and params.get("anchor_ref") == old_name:
            params["anchor_ref"] = name
        site.graph.remove_fact(fact.subject, fact.verb, fact.object, fact.params)
        site.graph.add_fact(node, fact.verb, fact.object, params, source=fact.source)
    found = world_object(site.world, old_name)
    if found is not None:
        level, i = found
        level.objects[i] = attrs.evolve(level.objects[i], humansim={k: v for k, v in level.objects[i].humansim.items() if k == "seats"})


def _move(site: Site, edit: Edit) -> None:
    """Move an object: `x`, `y`, `yaw` (default: unchanged), `level` (default: its own), `leave_demand` to leave its demand at the old spot."""
    _check_params(edit.params, ("x", "y", "yaw", "level", "leave_demand"))
    placed = _at(site.graph, edit.target)
    if placed is None:
        raise ValueError(f"{edit.target} is no object of the graph")
    old_level_id, pose = placed
    level_id = str(edit.params.get("level", old_level_id))
    _level(site.world, level_id)
    x, y, yaw = (float(edit.params.get(k, pose[k])) for k in ("x", "y", "yaw"))
    if edit.params.get("leave_demand"):
        _leave_demand(site, edit.target, old_level_id, pose)
    site.graph.remove_fact(edit.target, "at", f"floor:{old_level_id}")
    site.graph.add_fact(edit.target, "at", f"floor:{level_id}", {"x": x, "y": y, "yaw": yaw})
    found = world_object(site.world, _name(edit.target, "obj"))
    if found is not None:
        level, i = found
        old = level.objects[i].pose
        roll, pitch, _ = old.orientation.to_euler()
        obstacle = attrs.evolve(level.objects[i], pose=Pose(position=Position(x, y, old.position.z), orientation=Orientation.from_euler((roll, pitch, yaw))))
        if level is site.world.levels[level_id]:
            level.objects[i] = obstacle
        else:
            del level.objects[i]
            site.world.levels[level_id].objects.append(obstacle)
    _rederive(site.world, site.graph)


def _insert(site: Site, edit: Edit) -> None:
    """Insert a world object: `model`, `x`, `y`, `z` (default 0), `yaw` (default 0), `level`, `dock` (where a robot stands to use it, a polygon in its frame), `charge_power_w` (the power in W a robot in the dock charges with, which makes it a charger), and the demand `function` takes."""
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.write import short_identifier

    params = dict(edit.params)
    for key in ("model", "x", "y"):
        if key not in params:
            raise ValueError(f"insert needs {key!r}")
    name = _name(edit.target, "obj")
    if edit.target in site.graph.entities or any(name in names for names in site.world.names().values()):
        raise ValueError(f"{name!r} is already taken")
    level = _level(site.world, params.pop("level", None))
    level_id = next(k for k, v in site.world.levels.items() if v is level)
    x, y, z, yaw = (float(params.pop(k, 0.0)) for k in ("x", "y", "z", "yaw"))
    obstacle: Obstacle = Obstacle.parse({"name": name, "model": params.pop("model"), "pose": [x, y, z, 0.0, 0.0, yaw], "dock": params.pop("dock", None), "charge_power_w": params.pop("charge_power_w", None)})
    level.objects.append(obstacle)
    model = short_identifier(obstacle.model)
    site.graph.add_entity(edit.target, {"model": model[0] if isinstance(model, list) else model})
    site.graph.add_fact(edit.target, "at", f"floor:{level_id}", {"x": x, "y": y, "yaw": yaw})
    if params:
        _set_demand(site, edit.target, params, f"edit:{edit.id}")
    _rederive(site.world, site.graph)


def _remove(site: Site, edit: Edit) -> None:
    """Remove an object, from the graph and the world, with every fact about it."""
    _check_params(edit.params, ())
    name = _name(edit.target, "obj")
    if edit.target not in site.graph.entities:
        raise ValueError(f"{edit.target} is no object of the graph")
    site.graph.remove_entity(edit.target)
    found = world_object(site.world, name)
    if found is not None:
        level, i = found
        del level.objects[i]
        if site.world.kg is not None:
            site.world.kg.remove_entity(edit.target)
    _rederive(site.world, site.graph)


def _set_service(graph: Scene, obj: str, params: Mapping[str, Any], block: dict[str, Any] | None) -> None:
    """Set the `serves` fact of `obj` from `type`, `capacity` and `interaction_radius`."""
    serves = next((f for f in graph.facts if f.subject == obj and f.verb == "serves"), None)
    if serves is not None:
        kind = str(params.get("type", str(serves.object).partition(":")[2]))
        service = dict(serves.params)
        graph.remove_fact(obj, "serves", serves.object)
    elif "type" in params:
        kind = str(params["type"])
        service = {}
    else:
        raise ValueError(f"{obj} serves nothing, give it a type")
    changed = {k: params[k] for k in SERVICE_KEYS if k in params}
    service.update(changed)
    graph.add_fact(obj, "serves", f"service:{kind}", {k: v for k, v in service.items() if v is not None})
    if block is not None:
        block.update({"type": kind, **changed})


def _set_satisfies(graph: Scene, obj: str, satisfies: Mapping[str, Any], block: dict[str, Any] | None) -> None:
    """Set or, for a None amount, drop the `satisfies` fact of `obj` per need."""
    for need, amount in satisfies.items():
        graph.remove_fact(obj, "satisfies", f"need:{need}")
        if amount is not None:
            graph.add_fact(obj, "satisfies", f"need:{need}", {"amount": float(amount)})
        if block is not None:
            block["satisfies"] = {k: v for k, v in {**(block.get("satisfies") or {}), need: amount}.items() if v is not None}
    if block is not None and not block.get("satisfies"):
        block.pop("satisfies", None)


def _set_dwell(graph: Scene, obj: str, dwell: Any, origin: str) -> None:  # noqa: ANN401 - YAML input
    """Replace the `dwells` fact of `obj` with seconds or `{mean, std, clip_low, clip_high}`, or drop it for None."""
    for fact in [f for f in graph.facts if f.subject == obj and f.verb == "dwells"]:
        graph.remove_fact(obj, "dwells", fact.object)
    if isinstance(dwell, Mapping):
        _check_params(dwell, DWELL_KEYS)
        if "mean" not in dwell:
            raise ValueError("dwell needs a mean")
        graph.add_fact(obj, "dwells", float(dwell["mean"]), {**{k: float(v) for k, v in dwell.items() if k != "mean"}, "from": origin})
    elif dwell is not None:
        graph.add_fact(obj, "dwells", float(dwell), {"from": origin})


def _set_formation(graph: Scene, obj: str, formation: Any, block: dict[str, Any] | None) -> None:  # noqa: ANN401 - YAML input
    """Replace the `has_formation` fact of `obj` with `{type, params}`, anchored at the object, or drop it for None."""
    for fact in [f for f in graph.facts if f.subject == obj and f.verb == "has_formation"]:
        graph.remove_fact(obj, "has_formation", fact.object, fact.params)
    if block is not None:
        block.pop("formation", None)
    if formation is None:
        return
    if not isinstance(formation, Mapping) or not formation.get("type"):
        raise ValueError(f"formation is {{type, params}}, got {formation!r}")
    _check_params(formation, ("type", "params"))
    params = {str(k): float(v) for k, v in (formation.get("params") or {}).items()}
    graph.add_fact(obj, "has_formation", f"formation:{formation['type']}", params)
    if block is not None:
        block["formation"] = {"type": str(formation["type"]), "params": params}


def _set_demand(site: Site, obj: str, params: Mapping[str, Any], origin: str) -> None:
    """Set the demand of `obj` from `FUNCTION_PARAMS`, its dwell traced to `origin`."""
    _check_params(params, FUNCTION_PARAMS)
    if obj not in site.graph.entities or kind_of(obj) != "obj":
        raise ValueError(f"{obj} is no object of the graph")
    found = world_object(site.world, _name(obj, "obj"))
    block = None
    if found is not None:
        level, i = found
        block = dict(level.objects[i].humansim)
    if params.get("clear"):
        for fact in [f for f in site.graph.facts if f.subject == obj and f.verb in DEMAND_VERBS]:
            site.graph.remove_fact(fact.subject, fact.verb, fact.object, fact.params)
        if block is not None:
            block = {k: v for k, v in block.items() if k == "seats"}
    if "type" in params or any(k in params for k in SERVICE_KEYS):
        _set_service(site.graph, obj, params, block)
    if params.get("satisfies"):
        _set_satisfies(site.graph, obj, params["satisfies"], block)
    if "dwell" in params:
        _set_dwell(site.graph, obj, params["dwell"], origin)
    if "formation" in params:
        _set_formation(site.graph, obj, params["formation"], block)
    if found is not None and block is not None:
        level, i = found
        level.objects[i] = attrs.evolve(level.objects[i], humansim=block)


def _function(site: Site, edit: Edit) -> None:
    """Change an object's demand: `type`, `capacity`, `interaction_radius`, `satisfies`, `dwell`, `formation`, `clear` to drop it all first."""
    _set_demand(site, edit.target, edit.params, f"edit:{edit.id}")


def _set_presets(semantics: list[Any], preset: str, params: Mapping[str, Any] | None) -> list[Any]:
    """`semantics` as authored, its `preset` item set to `params`, or dropped for None."""
    from arena_simulation_setup.shared.semantics import authored_semantics, parse_semantics

    items = [item for item in authored_semantics(semantics) if item.get("preset") != preset]
    if params is not None:
        items.append({"preset": preset, "params": dict(params)})
    return parse_semantics(items)


def _access(site: Site, edit: Edit) -> None:
    """Change a door's gate (`locked`, `authorized`, `unlock_on`, or `remove`) or a schedule (`regime`, `windows`, `default`, or `remove`)."""
    from arena_simulation_setup.shared.world import Schedule

    params, world = edit.params, site.world
    remove = bool(params.get("remove", False))
    settings = {k: v for k, v in params.items() if k != "remove"}
    if kind_of(edit.target) == "door":
        _check_params(params, ("locked", "authorized", "unlock_on", "remove"))
        found = _door(world, _name(edit.target, "door"))
        if found is None:
            raise ValueError(f"{edit.target} is no door of the world")
        level, i = found
        door = level.doors[i]
        gate = {"authorized": [], "locked": False, **next(iter(presets(door.semantics, "gate")), {})}
        gate.update(settings)
        level.doors[i] = attrs.evolve(door, semantics=_set_presets(door.semantics, "gate", None if remove else {k: v for k, v in gate.items() if v is not None}))
    elif kind_of(edit.target) == "schedule":
        _check_params(params, ("regime", "windows", "default", "remove"))
        name = _name(edit.target, "schedule")
        i = next((i for i, s in enumerate(world.schedules) if s.name == name), None)
        if remove:
            if i is None:
                raise ValueError(f"{edit.target} is no schedule of the world")
            del world.schedules[i]
        else:
            if i is None:
                world.schedules.append(Schedule(name=name))
                i = len(world.schedules) - 1
            schedule = world.schedules[i]
            current = {"windows": [], **next(iter(presets(schedule.semantics, "schedule")), {})}
            current.update(settings)
            if not current.get("regime"):
                raise ValueError(f"{edit.target} needs a regime")
            windows = current.get("windows") or []
            if not isinstance(windows, list) or any(not isinstance(w, Mapping) or not {"start", "end"} <= set(w) for w in windows):
                raise ValueError(f"{edit.target}: windows are a list of {{start, end[, value]}}")
            # the runtime reads every window's `value`, the schedule's state while it is open; the regime holds in any window
            current["windows"] = [{**w, "value": w.get("value", SCHEDULE_ON)} for w in windows]
            world.schedules[i] = attrs.evolve(schedule, semantics=_set_presets(schedule.semantics, "schedule", {k: v for k, v in current.items() if v is not None}))
    else:
        raise ValueError(f"access edits a door:<name> or schedule:<name>, not {edit.target!r}")
    sync_access(site.graph, world)


def _segment(start: Position, end: Position) -> shapely.LineString:
    import shapely

    return shapely.LineString([(start.x, start.y), (end.x, end.y)])


def _on_wall(level: Level, door: Door) -> bool:
    """Whether the walls of `level` without the door cover the door's span."""
    import shapely

    walls = attrs.evolve(level, doors=[d for d in level.doors if d.name != door.name]).all_walls
    lines = shapely.union_all([_segment(w.start, w.end) for w in walls]).buffer(_ON_WALL_M)
    return bool(lines.covers(_segment(door.start, door.end)))


def _collinear(a: Position, b: Position, p: Position) -> bool:
    length = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5
    return length > 0 and abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / length <= _ON_WALL_M


def _fill_gap(level: Level, door: Door) -> None:
    """Close the gap a door leaves in walls drawn around it, with a wall path styled as the wall at its start."""
    import shapely

    from arena_simulation_setup.tree.World.model import WallPath

    if _on_wall(level, door):
        return
    walls = attrs.evolve(level, doors=[d for d in level.doors if d.name != door.name]).all_walls
    in_line = [w for w in walls if _collinear(door.start, door.end, w.start) and _collinear(door.start, door.end, w.end)]

    def wall_at(end: Position) -> Wall | None:
        spot = shapely.Point(end.x, end.y)
        return next((w for w in in_line if _segment(w.start, w.end).distance(spot) <= _ON_WALL_M), None)

    first, second = wall_at(door.start), wall_at(door.end)
    if first is not None and second is not None:
        level.walls.append(WallPath(path=[door.start, door.end], kind=first.kind, material=first.material))


def _shifted(points: list[Position], offset: Any) -> list[Position]:  # noqa: ANN401 - YAML input
    d = _point(offset, "offset")
    return [Position(p.x + d.x, p.y + d.y) for p in points]


def _layout_door(site: Site, edit: Edit) -> None:
    from arena_simulation_setup.shared.world import Door

    params = dict(edit.params)
    op = _layout_op(params)
    name = _name(edit.target, "door")
    found = _door(site.world, name)
    if op is LayoutOp.ADD:
        _check_params(params, ("level", "span", *DOOR_FIELDS))
        if found is not None or any(name in names for names in site.world.names().values()):
            raise ValueError(f"{name!r} is already taken")
        level = _level(site.world, params.pop("level", None))
        if "span" not in params:
            raise ValueError("adding a door needs a span")
        level.doors.append(Door.parse({"name": name, **params}))
        door = level.doors[-1]
        site.graph.add_entity(edit.target)
    else:
        if found is None:
            raise ValueError(f"{edit.target} is no door of the world")
        level, i = found
        if op is LayoutOp.REMOVE:
            _check_params(params, ())
            _fill_gap(level, level.doors[i])
            del level.doors[i]
            site.graph.remove_entity(edit.target)
            if site.world.kg is not None:
                site.world.kg.remove_entity(edit.target)
            _rederive(site.world, site.graph)
            return
        _check_params(params, ("span", "offset", *DOOR_FIELDS))
        door = level.doors[i]
        if "span" in params and "offset" in params:
            raise ValueError("give a span or an offset, not both")
        start, end = door.start, door.end
        if "span" in params:
            span = params.pop("span")
            if not isinstance(span, list | tuple) or len(span) != 2:
                raise ValueError(f"span: expected [[x, y], [x, y]], got {span!r}")
            start, end = _point(span[0], "span[0]"), _point(span[1], "span[1]")
        elif "offset" in params:
            start, end = _shifted([start, end], params.pop("offset"))
        if (start, end) != (door.start, door.end):
            _fill_gap(level, door)
        door = attrs.evolve(door, start=start, end=end, **params)
        level.doors[i] = door
    if not _on_wall(level, door):
        raise ValueError(f"door {name!r} at {[[door.start.x, door.start.y], [door.end.x, door.end.y]]} lies on no wall")
    _rederive(site.world, site.graph)


def _layout_wall(site: Site, edit: Edit) -> None:
    from arena_simulation_setup.tree.World.model import WallPath

    params = dict(edit.params)
    op = _layout_op(params)
    level_id, _, index = _name(edit.target, "wall").partition("/")
    level = _level(site.world, level_id)
    if op is LayoutOp.ADD:
        _check_params(params, ("path", *WALL_FIELDS))
        if index:
            raise ValueError(f"adding a wall targets wall:<level>, not {edit.target!r}")
        if "path" not in params:
            raise ValueError("adding a wall needs a path")
        level.walls.append(WallPath(path=[_point(p, "path") for p in params.pop("path")], **params))
    else:
        if not index.isdigit() or int(index) >= len(level.walls):
            raise ValueError(f"no wall {edit.target!r}: level {level_id!r} has wall paths 0..{len(level.walls) - 1}")
        i = int(index)
        if op is LayoutOp.REMOVE:
            _check_params(params, ())
            del level.walls[i]
        else:
            _check_params(params, ("path", "offset", *WALL_FIELDS))
            if "path" in params and "offset" in params:
                raise ValueError("give a path or an offset, not both")
            wall = level.walls[i]
            path = wall.path
            if "path" in params:
                path = [_point(p, "path") for p in params.pop("path")]
            elif "offset" in params:
                path = _shifted(wall.path, params.pop("offset"))
            level.walls[i] = attrs.evolve(wall, path=path, **params)
    _rederive(site.world, site.graph)


def _layout(site: Site, edit: Edit) -> None:
    """Change a door (`door:<name>`) or a wall path (`wall:<level>/<index>`, `wall:<level>` to add): `op` add, move (default) or remove."""
    if kind_of(edit.target) == "door":
        _layout_door(site, edit)
    elif kind_of(edit.target) == "wall":
        _layout_wall(site, edit)
    else:
        raise ValueError(f"layout edits a door:<name> or wall:<level>/<index>, not {edit.target!r}")


OPERATIONS: dict[EditType, Callable[[Site, Edit], None]] = {
    EditType.MOVE: _move,
    EditType.INSERT: _insert,
    EditType.REMOVE: _remove,
    EditType.FUNCTION: _function,
    EditType.ACCESS: _access,
    EditType.LAYOUT: _layout,
}


def apply(site: Site, edit_type: EditType | str, target: str, params: Mapping[str, Any] | None = None) -> Site:
    """A new site: `site` with the edit applied and recorded, `site` itself unchanged."""
    if edit_type not in tuple(EditType):
        raise ValueError(f"edit type {edit_type!r} is not one of {', '.join(EditType)}")
    edit = Edit(type=EditType(edit_type), target=target, params=dict(params or {}), parent=site.key)
    edited = _copy(site)
    OPERATIONS[edit.type](edited, edit)
    edited.world.validate()
    problems = edited.graph.validate()
    if problems:
        raise ValueError("; ".join(problems))
    edited.edits.append(edit)
    return edited


def dump_edits(site: Site) -> dict[str, Any]:
    return {"format": FORMAT, "base": site.base, "scenario": site.scenario, "seed": site.seed, "edits": [e.dump() for e in site.edits]}


def _link_or_copy(source: str, target: str) -> str:
    """Hardlink `source` at `target`, a copy where the link fails (another filesystem, one without hardlinks)."""
    try:
        os.link(source, target)
    except OSError:
        return shutil.copy2(source, target)
    return target


def save(site: Site) -> Path:
    """Write the edited world to the writable local worlds root under `site.name`, its graph and `edits.yaml` beside
    `world.yaml`. The files of the source world's `assets/` are hardlinked, not copied (a world can hold gigabytes of
    splats and meshes), so they are shared with the source and must be replaced, never written in place; every file
    the save writes is its own."""
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.tree.World.view import LEVEL_ORIGINS_FILE, world_assets
    from arena_simulation_setup.tree.World.write import to_yaml

    if not site.edits:
        raise ValueError(f"{site.name} has no edits to save")
    view = WorldIdentifier(site.name).resolve_write_sync()
    if view.path.resolve() == site.root.resolve():
        raise ValueError(f"{site.name} would overwrite the world it was edited from")
    if view.path.exists():
        shutil.rmtree(view.path)
    view.path.mkdir(parents=True)
    if (site.root / "assets").is_dir():
        shutil.copytree(site.root / "assets", view.path / "assets", copy_function=_link_or_copy)
    if (site.root / LEVEL_ORIGINS_FILE).is_file():
        shutil.copy2(site.root / LEVEL_ORIGINS_FILE, view.path / LEVEL_ORIGINS_FILE)
    with world_assets(view, site.world):
        view.save(site.world)
    write(site.graph, view.path / GRAPH)
    (view.path / EDITS).write_text(to_yaml(dump_edits(site)))
    return view.path
