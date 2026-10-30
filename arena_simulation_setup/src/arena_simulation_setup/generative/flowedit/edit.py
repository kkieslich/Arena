"""Edit a FLOWEDIT site: one operation per edit type, each pairing the graph change with the world change, saved as a new world."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import yaml

from arena_simulation_setup.kg import Scene, load, write
from arena_simulation_setup.kg.scene import DERIVED
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

from .build import build, sync_access

if TYPE_CHECKING:
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.model import Level, World

FORMAT = "flowedit-edits/1"
GRAPH = "flowgraph.yaml"
EDITS = "edits.yaml"
SEPARATOR = "__"
ID_LENGTH = 8
SERVICE_KEYS = ("capacity", "interaction_radius")
DWELL_KEYS = ("mean", "std", "clip_low", "clip_high")
DOOR_FIELDS = ("kind", "width", "height", "material", "activation_distance", "transition_time", "hold_time", "swing_direction")
WALL_FIELDS = ("closed", "kind", "material")
_ON_WALL_M = 0.05


@attrs.frozen
class Edit:
    """One edit: its `type`, the `target` id it changes, its `params`, and the world it was applied to."""

    type: str
    target: str
    params: Mapping[str, Any]
    parent: str

    @property
    def id(self) -> str:
        """Short hash of the whole record, the parent included, so it also names the chain."""
        record = json.dumps({"parent": self.parent, "type": self.type, "target": self.target, "params": self.params}, sort_keys=True)
        return hashlib.sha256(record.encode()).hexdigest()[:ID_LENGTH]

    def dump(self) -> dict[str, Any]:
        return {"id": self.id, "parent": self.parent, "type": self.type, "target": self.target, "params": dict(self.params)}


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
    def name(self) -> str:
        """The world name: `base` for an unedited site, else `<base>__<last edit id>`."""
        return f"{self.base}{SEPARATOR}{self.edits[-1].id}" if self.edits else self.base


def _load_world(root: Path) -> World:
    """The world at `root` with only its stated KG, as it is written back."""
    from arena_simulation_setup.tree.World.view import WorldView

    world = WorldView(root.resolve()).load(derive="off")
    if world.kg is not None and not world.kg.entities and not world.kg.facts and not world.kg.constraints:
        world.kg = None
    return world


def open_site(root: Path, scenario: str | None = None, seed: int = 0, notes: list[str] | None = None) -> Site:
    """The site of the world at `root`: an edited world from its `edits.yaml` and graph, else the graph `build` makes of it and `scenario`."""
    if (root / EDITS).is_file():
        doc = yaml.safe_load((root / EDITS).read_text()) or {}
        if doc.get("format") != FORMAT:
            raise ValueError(f"{root / EDITS}: format: expected {FORMAT!r}, got {doc.get('format')!r}")
        if scenario is not None and scenario != doc.get("scenario"):
            raise ValueError(f"{root.name} was edited from scenario {doc.get('scenario')!r}, not {scenario!r}")
        edits = [Edit(type=e["type"], target=e["target"], params=e.get("params") or {}, parent=e["parent"]) for e in doc.get("edits") or []]
        return Site(_load_world(root), load(root / GRAPH), root, str(doc["base"]), doc.get("scenario"), int(doc.get("seed", 0)), edits)
    path = root / "scenarios" / scenario / "scenario.yaml" if scenario else None
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"no scenario {scenario!r} at {path}")
    return Site(_load_world(root), build(root, path, seed, notes), root, root.name, scenario, seed)


def _copy(site: Site) -> Site:
    from arena_simulation_setup.kg import dump

    return attrs.evolve(site, world=copy.deepcopy(site.world), graph=load(dump(site.graph)), edits=list(site.edits))


# --- lookups ---


def _name(target: str, kind: str) -> str:
    prefix, sep, name = target.partition(":")
    if prefix != kind or not sep or not name:
        raise ValueError(f"target {target!r} is not of the form {kind}:<name>")
    return name


def _at(graph: Scene, obj: str) -> tuple[str, dict[str, Any]] | None:
    """The level id and pose params of `obj`'s `at` fact."""
    fact = next((f for f in graph.facts if f.subject == obj and f.verb == "at"), None)
    return None if fact is None else (str(fact.object).partition(":")[2], dict(fact.params))


def _world_object(world: World, name: str) -> tuple[Level, int] | None:
    for level in world.levels.values():
        for i, o in enumerate(level.objects):
            if o.name == name:
                return level, i
    return None


def _door(world: World, name: str) -> tuple[Level, int] | None:
    for level in world.levels.values():
        for i, d in enumerate(level.doors):
            if d.name == name:
                return level, i
    return None


def _level(world: World, level_id: object) -> Level:
    if level_id is None:
        if len(world.levels) != 1:
            raise ValueError(f"the world has levels {list(world.levels)}, name one as level")
        return next(iter(world.levels.values()))
    if str(level_id) not in world.levels:
        raise ValueError(f"no level {level_id!r} (levels: {list(world.levels)})")
    return world.levels[str(level_id)]


def _unknown(params: Mapping[str, Any], allowed: tuple[str, ...]) -> None:
    unknown = sorted(set(params) - set(allowed))
    if unknown:
        raise ValueError(f"unknown params {unknown}, expected some of {sorted(allowed)}")


def _point(value: Any, what: str) -> Position:  # noqa: ANN401 - YAML input
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{what}: expected [x, y], got {value!r}")
    return Position(float(value[0]), float(value[1]))


# --- derived facts ---


def _rederive(world: World, graph: Scene) -> None:
    """Recompute `contains` and `connects` of `graph` from the world's geometry and the graph's own object poses."""
    from arena_simulation_setup.tree.World.membership import _inside, derive

    graph.drop_derived()
    derive(attrs.evolve(world, kg=graph))
    world_objects = {f"obj:{o.name}" for level in world.levels.values() for o in level.objects}
    stated = {str(f.object) for f in graph.stated() if f.verb == "contains"}
    for obj in [id for id in graph.entities if id.startswith("obj:") and id not in world_objects and id not in stated]:
        placed = _at(graph, obj)
        if placed is None or placed[0] not in world.levels:
            continue
        level_id, pose = placed
        for room in _inside(world.levels[level_id], Position(float(pose["x"]), float(pose["y"]))):
            graph.add_entity(f"room:{room}")
            graph.add_fact(f"room:{room}", "contains", obj, source=DERIVED)


# --- operations: each changes `site` in place ---


def _move(site: Site, edit: Edit) -> None:
    """Move an object: `x`, `y`, `yaw` (default: unchanged), `level` (default: its own)."""
    _unknown(edit.params, ("x", "y", "yaw", "level"))
    placed = _at(site.graph, edit.target)
    if placed is None:
        raise ValueError(f"{edit.target} is no object of the graph")
    level_id, pose = placed
    level_id = str(edit.params.get("level", level_id))
    _level(site.world, level_id)
    x, y, yaw = (float(edit.params.get(k, pose[k])) for k in ("x", "y", "yaw"))
    site.graph.remove_fact(edit.target, "at", f"floor:{placed[0]}")
    site.graph.add_fact(edit.target, "at", f"floor:{level_id}", {"x": x, "y": y, "yaw": yaw})
    found = _world_object(site.world, _name(edit.target, "obj"))
    if found is not None:
        level, i = found
        obstacle = attrs.evolve(level.objects[i], pose=Pose(position=Position(x, y), orientation=Orientation.from_yaw(yaw)))
        if level is site.world.levels[level_id]:
            level.objects[i] = obstacle
        else:
            del level.objects[i]
            site.world.levels[level_id].objects.append(obstacle)
    _rederive(site.world, site.graph)


def _insert(site: Site, edit: Edit) -> None:
    """Insert a world object: `model`, `x`, `y`, `yaw` (default 0), `level`, and the demand `function` takes."""
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
    x, y, yaw = float(params.pop("x")), float(params.pop("y")), float(params.pop("yaw", 0.0))
    obstacle: Obstacle = Obstacle.parse({"name": name, "model": params.pop("model"), "pose": [x, y, yaw]})
    level.objects.append(obstacle)
    model = short_identifier(obstacle.model)
    site.graph.add_entity(edit.target, {"model": model[0] if isinstance(model, list) else model})
    site.graph.add_fact(edit.target, "at", f"floor:{level_id}", {"x": x, "y": y, "yaw": yaw})
    if params:
        _function(site, attrs.evolve(edit, params=params))
    _rederive(site.world, site.graph)


def _remove(site: Site, edit: Edit) -> None:
    """Remove an object, from the graph and the world, with every fact about it."""
    _unknown(edit.params, ())
    name = _name(edit.target, "obj")
    if edit.target not in site.graph.entities:
        raise ValueError(f"{edit.target} is no object of the graph")
    site.graph.remove_entity(edit.target)
    found = _world_object(site.world, name)
    if found is not None:
        level, i = found
        del level.objects[i]
        if site.world.kg is not None:
            site.world.kg.remove_entity(edit.target)
    _rederive(site.world, site.graph)


def _function(site: Site, edit: Edit) -> None:
    """Change an object's demand: `type`, `capacity`, `interaction_radius`, `satisfies: {<need>: amount or null}`, `dwell: seconds, {mean, std, clip_low, clip_high} or null`."""
    _unknown(edit.params, ("type", *SERVICE_KEYS, "satisfies", "dwell"))
    graph, obj, params = site.graph, edit.target, edit.params
    if obj not in graph.entities or not obj.startswith("obj:"):
        raise ValueError(f"{obj} is no object of the graph")
    found = _world_object(site.world, _name(obj, "obj"))
    block = dict(found[0].objects[found[1]].humansim) if found is not None else None
    serves = next((f for f in graph.facts if f.subject == obj and f.verb == "serves"), None)
    if "type" in params or any(k in params for k in SERVICE_KEYS):
        if serves is None and "type" not in params:
            raise ValueError(f"{obj} serves nothing, give it a type")
        kind = str(params["type"]) if "type" in params else str(serves.object).partition(":")[2]  # type: ignore[union-attr]
        service = dict(serves.params) if serves is not None else {}
        service.update({k: params[k] for k in SERVICE_KEYS if k in params})
        if serves is not None:
            graph.remove_fact(obj, "serves", serves.object)
        graph.add_fact(obj, "serves", f"service:{kind}", {k: v for k, v in service.items() if v is not None})
        if block is not None:
            block.update({"type": kind, **{k: params[k] for k in SERVICE_KEYS if k in params}})
    for need, amount in (params.get("satisfies") or {}).items():
        graph.remove_fact(obj, "satisfies", f"need:{need}")
        if amount is not None:
            graph.add_fact(obj, "satisfies", f"need:{need}", {"amount": float(amount)})
        if block is not None:
            satisfies = {**(block.get("satisfies") or {}), need: amount}
            block["satisfies"] = {k: v for k, v in satisfies.items() if v is not None}
    if "dwell" in params:
        for fact in [f for f in graph.facts if f.subject == obj and f.verb == "dwells"]:
            graph.remove_fact(obj, "dwells", fact.object)
        dwell = params["dwell"]
        if isinstance(dwell, Mapping):
            _unknown(dwell, DWELL_KEYS)
            if "mean" not in dwell:
                raise ValueError("dwell needs a mean")
            graph.add_fact(obj, "dwells", float(dwell["mean"]), {**{k: float(v) for k, v in dwell.items() if k != "mean"}, "from": f"edit:{edit.id}"})
        elif dwell is not None:
            graph.add_fact(obj, "dwells", float(dwell), {"from": f"edit:{edit.id}"})
    if found is not None and block is not None:
        level, i = found
        if not block.get("satisfies"):
            block.pop("satisfies", None)
        level.objects[i] = attrs.evolve(level.objects[i], humansim=block)


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

    from .build import presets

    params, world = edit.params, site.world
    remove = bool(params.get("remove", False))
    if edit.target.startswith("door:"):
        _unknown(params, ("locked", "authorized", "unlock_on", "remove"))
        found = _door(world, _name(edit.target, "door"))
        if found is None:
            raise ValueError(f"{edit.target} is no door of the world")
        level, i = found
        door = level.doors[i]
        gate = {"authorized": [], "locked": False, **next(iter(presets(door.semantics, "gate")), {})}
        gate.update({k: v for k, v in params.items() if k != "remove"})
        level.doors[i] = attrs.evolve(door, semantics=_set_presets(door.semantics, "gate", None if remove else {k: v for k, v in gate.items() if v is not None}))
    elif edit.target.startswith("schedule:"):
        _unknown(params, ("regime", "windows", "default", "remove"))
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
            current.update(params)
            if not current.get("regime"):
                raise ValueError(f"{edit.target} needs a regime")
            world.schedules[i] = attrs.evolve(schedule, semantics=_set_presets(schedule.semantics, "schedule", {k: v for k, v in current.items() if v is not None}))
    else:
        raise ValueError(f"access edits a door:<name> or schedule:<name>, not {edit.target!r}")
    sync_access(site.graph, world)


def _on_wall(level: Level, door: Any) -> bool:  # noqa: ANN401 - Door
    """Whether the walls of `level` without the door cover the door's span."""
    import shapely

    walls = attrs.evolve(level, doors=[d for d in level.doors if d.name != door.name]).all_walls
    lines = shapely.union_all([shapely.LineString([(w.start.x, w.start.y), (w.end.x, w.end.y)]) for w in walls]).buffer(_ON_WALL_M)
    return bool(lines.covers(shapely.LineString([(door.start.x, door.start.y), (door.end.x, door.end.y)])))


def _collinear(a: Position, b: Position, p: Position) -> bool:
    length = ((b.x - a.x) ** 2 + (b.y - a.y) ** 2) ** 0.5
    return length > 0 and abs((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / length <= _ON_WALL_M


def _fill_gap(level: Level, door: Any) -> None:  # noqa: ANN401 - Door
    """Close the gap a door leaves in walls drawn around it, with a wall path styled as the wall at its start."""
    import shapely

    from arena_simulation_setup.tree.World.model import WallPath

    if _on_wall(level, door):
        return
    walls = attrs.evolve(level, doors=[d for d in level.doors if d.name != door.name]).all_walls
    touching = []
    for end in (door.start, door.end):
        spot = shapely.Point(end.x, end.y)
        touching.append(next((w for w in walls if _collinear(door.start, door.end, w.start) and _collinear(door.start, door.end, w.end) and shapely.LineString([(w.start.x, w.start.y), (w.end.x, w.end.y)]).distance(spot) <= _ON_WALL_M), None))
    if all(w is not None for w in touching):
        level.walls.append(WallPath(path=[door.start, door.end], kind=touching[0].kind, material=touching[0].material))


def _shifted(points: list[Position], offset: Any) -> list[Position]:  # noqa: ANN401 - YAML input
    d = _point(offset, "offset")
    return [Position(p.x + d.x, p.y + d.y) for p in points]


def _layout_door(site: Site, edit: Edit) -> None:
    from arena_simulation_setup.shared.world import Door

    params = dict(edit.params)
    op = params.pop("op", "move")
    name = _name(edit.target, "door")
    found = _door(site.world, name)
    if op == "add":
        _unknown(params, ("level", "span", *DOOR_FIELDS))
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
        if op == "remove":
            _unknown(params, ())
            _fill_gap(level, level.doors[i])
            del level.doors[i]
            site.graph.remove_entity(edit.target)
            if site.world.kg is not None:
                site.world.kg.remove_entity(edit.target)
            _rederive(site.world, site.graph)
            return
        if op != "move":
            raise ValueError(f"op {op!r} is not one of add, move, remove")
        _unknown(params, ("span", "offset", *DOOR_FIELDS))
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
    op = params.pop("op", "move")
    level_id, _, index = _name(edit.target, "wall").partition("/")
    level = _level(site.world, level_id)
    if op == "add":
        _unknown(params, ("path", *WALL_FIELDS))
        if index:
            raise ValueError(f"adding a wall targets wall:<level>, not {edit.target!r}")
        if "path" not in params:
            raise ValueError("adding a wall needs a path")
        level.walls.append(WallPath(path=[_point(p, "path") for p in params.pop("path")], **params))
    else:
        if not index.isdigit() or int(index) >= len(level.walls):
            raise ValueError(f"no wall {edit.target!r}: level {level_id!r} has wall paths 0..{len(level.walls) - 1}")
        i = int(index)
        if op == "remove":
            _unknown(params, ())
            del level.walls[i]
        elif op == "move":
            _unknown(params, ("path", "offset", *WALL_FIELDS))
            if "path" in params and "offset" in params:
                raise ValueError("give a path or an offset, not both")
            wall = level.walls[i]
            path = [_point(p, "path") for p in params.pop("path")] if "path" in params else _shifted(wall.path, params.pop("offset")) if "offset" in params else wall.path
            level.walls[i] = attrs.evolve(wall, path=path, **params)
        else:
            raise ValueError(f"op {op!r} is not one of add, move, remove")
    _rederive(site.world, site.graph)


def _layout(site: Site, edit: Edit) -> None:
    """Change a door (`door:<name>`) or a wall path (`wall:<level>/<index>`, `wall:<level>` to add): `op` add, move (default) or remove."""
    if edit.target.startswith("door:"):
        _layout_door(site, edit)
    elif edit.target.startswith("wall:"):
        _layout_wall(site, edit)
    else:
        raise ValueError(f"layout edits a door:<name> or wall:<level>/<index>, not {edit.target!r}")


OPERATIONS: dict[str, Callable[[Site, Edit], None]] = {
    "move": _move,
    "insert": _insert,
    "remove": _remove,
    "function": _function,
    "access": _access,
    "layout": _layout,
}


def apply(site: Site, type: str, target: str, params: Mapping[str, Any] | None = None) -> Site:
    """A new site: `site` with the edit applied and recorded, `site` itself unchanged."""
    if type not in OPERATIONS:
        raise ValueError(f"edit type {type!r} is not one of {', '.join(OPERATIONS)}")
    edit = Edit(type=type, target=target, params=dict(params or {}), parent=site.name)
    edited = _copy(site)
    OPERATIONS[type](edited, edit)
    edited.world.validate()
    problems = edited.graph.validate()
    if problems:
        raise ValueError("; ".join(problems))
    edited.edits.append(edit)
    return edited


def dump_edits(site: Site) -> dict[str, Any]:
    return {"format": FORMAT, "base": site.base, "scenario": site.scenario, "seed": site.seed, "edits": [e.dump() for e in site.edits]}


def save(site: Site) -> Path:
    """Write the edited world to the writable local worlds root under `site.name`, its graph and `edits.yaml` beside `world.yaml`."""
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.tree.World.view import LEVEL_ORIGINS_FILE, world_assets

    if not site.edits:
        raise ValueError(f"{site.name} has no edits to save")
    view = WorldIdentifier(site.name).resolve_write_sync()
    if view.path.resolve() == site.root.resolve():
        raise ValueError(f"{site.name} would overwrite the world it was edited from")
    if view.path.exists():
        shutil.rmtree(view.path)
    view.path.mkdir(parents=True)
    if (site.root / "assets").is_dir():
        shutil.copytree(site.root / "assets", view.path / "assets")
    if (site.root / LEVEL_ORIGINS_FILE).is_file():
        shutil.copy2(site.root / LEVEL_ORIGINS_FILE, view.path / LEVEL_ORIGINS_FILE)
    with world_assets(view, site.world):
        view.save(site.world)
    write(site.graph, view.path / GRAPH)
    (view.path / EDITS).write_text(yaml.safe_dump(dump_edits(site), sort_keys=False, default_flow_style=None, width=1 << 16))
    return view.path
