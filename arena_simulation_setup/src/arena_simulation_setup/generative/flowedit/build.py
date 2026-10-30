"""Build the FLOWEDIT graph of a world and one of its scenarios: a KG scene with each object's demand."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import yaml

from arena_simulation_setup.kg import Scene, Value, Vocabulary
from arena_simulation_setup.utils.geometry import Pose, Position

from ..scenario.formats import read_scenario
from ..scenario.zones import KIND, PACKAGE, VERB, classify

if TYPE_CHECKING:
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.model import World

USES = ("core@2", PACKAGE, "flow@1")
ACCESS_VERBS = ("gated", "authorizes", "unlocks_on", "asserts", "window")
Duration = float | Mapping[str, float]


def _scene(world: World) -> Scene:
    """The world's derived KG under the FLOWEDIT vocabulary."""
    from arena_simulation_setup.tree.World.membership import derive

    scene = derive(world)
    scene.vocab = Vocabulary([*scene.uses, *(u for u in USES if u not in scene.uses)])
    return scene


def _placed(world: World, origins: Mapping[str, tuple[float, float]], obstacle: Obstacle) -> str:
    """The level whose zones hold the map-frame `obstacle`, the only level when none does."""
    import shapely

    spot = shapely.Point(obstacle.pose.position.x, obstacle.pose.position.y)
    for level_id, level in world.levels.items():
        if any(len(z.polygon) >= 3 and z.shape().covers(spot) for z in level.shifted(*origins[level_id]).zones):
            return level_id
    if len(world.levels) == 1:
        return next(iter(world.levels))
    raise ValueError(f"scenario object {obstacle.name!r} lies in no zone of any level")


def _add_scenario_objects(world: World, root: Path, objects: list[Any], seed: int) -> dict[str, str]:
    """Add the scenario's objects to their levels as the runtime places them, the zone names they are posed at by object name."""
    import numpy as np

    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.view import WorldView

    origins = WorldView(root).level_origins() or {level_id: (0.0, 0.0) for level_id in world.levels}
    converter = world.compact_world(origins).zone_converter(np.random.default_rng(seed))
    taken = {o.name for level in world.levels.values() for o in level.objects}
    refs = {}
    for i, raw in enumerate(objects):
        try:
            obstacle = converter.structure(raw, Obstacle)
        except Exception as exc:
            raise ValueError(f"objects[{i}]: {exc}") from exc
        if obstacle.name in taken:
            raise ValueError(f"objects[{i}]: {obstacle.name!r} is already an object of the world")
        taken.add(obstacle.name)
        level_id = _placed(world, origins, obstacle)
        ox, oy = origins[level_id]
        position = Position(obstacle.pose.position.x - ox, obstacle.pose.position.y - oy)
        world.levels[level_id].objects.append(attrs.evolve(obstacle, pose=Pose(position=position, orientation=obstacle.pose.orientation)))
        if isinstance(raw.get("pose"), str):
            refs[obstacle.name] = raw["pose"]
    return refs


def _agent_type_files(scenario: Mapping[str, Any], scenario_dir: Path) -> list[Path]:
    """The scenario-local agent type files its humans and regions name, in order of first use."""
    refs = [(h.get("agent") or {}).get("agent_type") for h in scenario.get("humans") or []]
    refs += [((r.get("config") or {}).get("agent") or {}).get("agent_type") for r in (scenario.get("regions") or {}).values()]
    files = []
    for ref in refs:
        if isinstance(ref, str) and ref.startswith("./") and (path := scenario_dir / ref[2:]) not in files:
            files.append(path)
    return files


def _timed_steps(files: list[Path]) -> Iterator[tuple[str, str, Duration]]:
    """`(target, origin, duration)` of every step with both, origin as `<file>#<sequence>.<step>`."""
    for path in files:
        raw = yaml.safe_load(path.read_text()) or {}
        for sequence, spec in (raw.get("sequences") or {}).items():
            for step, body in ((spec or {}).get("steps") or {}).items():
                if isinstance(body, Mapping) and isinstance(body.get("target"), str) and "duration" in body:
                    yield body["target"], f"{path.name}#{sequence}.{step}", body["duration"]


def _dwell(name: str, kind: str | None, steps: list[tuple[str, str, Duration]], notes: list[str]) -> tuple[Duration, str] | None:
    """The step duration that uses object `name`, by its id before its type; a note when steps disagree."""
    found = [(origin, d) for target, origin, d in steps if target == name] or [(origin, d) for target, origin, d in steps if kind and target == kind]
    if not found:
        return None
    origin, duration = found[0]
    others = [o for o, d in found[1:] if d != duration]
    if others:
        notes.append(f"obj:{name}: dwell from {origin}, the steps {others} give other durations")
    return duration, origin


def _add_demand(scene: Scene, obj: str, block: Mapping[str, Any], dwell: tuple[Duration, str] | None, notes: list[str]) -> None:
    """Add the demand facts of `obj` from its `humansim:` block and dwell."""
    service = {k: block[k] for k in ("capacity", "interaction_radius") if block.get(k) is not None}
    if block.get("type"):
        scene.add_fact(obj, "serves", f"service:{block['type']}", service)
    for need, amount in (block.get("satisfies") or {}).items():
        scene.add_fact(obj, "satisfies", f"need:{need}", {"amount": float(amount)})
    if formation := block.get("formation"):
        params = dict(formation.get("params") or {})
        if anchor := formation.get("anchor"):
            if "pose" in anchor:
                raise ValueError(f"{obj}: a formation anchored at a pose is not in flow@1")
            params["anchor_kind"] = anchor.get("kind", "object")
            if anchor.get("ref") is not None:
                params["anchor_ref"] = str(anchor["ref"])
        scene.add_fact(obj, "has_formation", f"formation:{formation.get('type', '')}", params)
    if "seats" in block:
        notes.append(f"{obj}: seats are not in flow@1, left out")
    if dwell is not None:
        duration, origin = dwell
        if isinstance(duration, Mapping):
            if "mean" not in duration:
                raise ValueError(f"{obj}: the duration of {origin} has no mean")
            rest = {k: float(v) for k, v in duration.items() if k != "mean"}
            scene.add_fact(obj, "dwells", float(duration["mean"]), {**rest, "from": origin})
        else:
            scene.add_fact(obj, "dwells", float(duration), {"from": origin})


def presets(semantics: list[Any], preset: str) -> list[dict[str, Any]]:
    """The `params` of every `preset` item `semantics` was authored with."""
    from arena_simulation_setup.shared.semantics import authored_semantics

    return [dict(item.get("params") or {}) for item in authored_semantics(semantics) if item.get("preset") == preset]


def sync_access(scene: Scene, world: World, notes: list[str] | None = None) -> None:
    """Replace the access facts of `scene` with the gates on the world's doors and its schedules."""
    notes = notes if notes is not None else []
    for fact in [f for f in scene.facts if f.verb in ACCESS_VERBS]:
        scene.remove_fact(fact.subject, fact.verb, fact.object, fact.params)
    schedules = {f"schedule:{s.name}" for s in world.schedules}
    for id in [id for id in scene.entities if id.startswith("schedule:") and id not in schedules]:
        scene.remove_entity(id)
    for level in world.levels.values():
        for door in level.doors:
            for gate in presets(door.semantics, "gate"):
                ref = f"door:{door.name}"
                scene.add_entity(ref)
                scene.add_fact(ref, "gated", bool(gate.get("locked", True)))
                for name in gate.get("authorized") or []:
                    scene.add_fact(ref, "authorizes", str(name))
                if gate.get("unlock_on"):
                    scene.add_fact(ref, "unlocks_on", f"regime:{gate['unlock_on']}")
    for schedule in world.schedules:
        for params in presets(schedule.semantics, "schedule"):
            ref = f"schedule:{schedule.name}"
            scene.add_entity(ref)
            if not params.get("regime"):
                notes.append(f"{ref}: a schedule without a regime asserts nothing, left out")
                continue
            scene.add_fact(ref, "asserts", f"regime:{params['regime']}", {"default": params["default"]} if params.get("default") is not None else {})
            for window in params.get("windows") or []:
                scene.add_fact(ref, "window", float(window["start"]), {"end": float(window["end"]), **({"value": window["value"]} if window.get("value") is not None else {})})
    used = {str(f.object) for f in scene.facts}
    for id in [id for id in scene.entities if id.startswith("regime:") and id not in used]:
        scene.remove_entity(id)


def build(root: Path, scenario: Path | None = None, seed: int = 0, notes: list[str] | None = None) -> Scene:
    """The FLOWEDIT graph of the world at `root` and `scenario`, what it leaves out appended to `notes`."""
    from arena_simulation_setup.tree.World.write import ANGLE_DECIMALS, POSITION_DECIMALS, short_identifier

    from ..scenario.formats import load_world

    notes = notes if notes is not None else []
    world = load_world(root)
    doc = read_scenario(scenario) if scenario is not None else {}
    refs = _add_scenario_objects(world, root, list(doc.get("objects") or []), seed)
    scene = _scene(world)
    steps = list(_timed_steps(_agent_type_files(doc, scenario.parent))) if scenario is not None else []
    stated = {f.subject for f in scene.facts if f.verb == VERB}
    for level_id, level in world.levels.items():
        for zone in level.zones:
            room = f"room:{zone.name}"
            if room not in stated:
                scene.add_fact(room, VERB, f"{KIND}:{classify(zone.name).value}", {"by": "rule"})
        for o in level.objects:
            obj = f"obj:{o.name}"
            model = short_identifier(o.model)
            if isinstance(model, list):
                notes.append(f"{obj}: model modifiers {model[1]} are not in flow@1, left out")
                model = model[0]
            scene.add_entity(obj, {"model": model})
            pose: dict[str, Value] = {"x": round(o.pose.position.x, POSITION_DECIMALS), "y": round(o.pose.position.y, POSITION_DECIMALS), "yaw": round(o.pose.orientation.to_yaw(), ANGLE_DECIMALS)}
            if o.name in refs:
                pose["pose_ref"] = refs[o.name]
            scene.add_fact(obj, "at", f"floor:{level_id}", pose)
            dwell = _dwell(o.name, o.humansim.get("type"), steps, notes)
            if o.humansim or dwell is not None:
                _add_demand(scene, obj, o.humansim, dwell, notes)
    sync_access(scene, world, notes)
    problems = scene.validate()
    if problems:
        raise ValueError("; ".join(problems))
    return scene
