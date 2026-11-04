"""The scenario side of a FLOWEDIT site: its own objects follow the graph, and its folder travels with an edited world."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from arena_simulation_setup.kg import Scene

#: `humansim:` keys a function edit changes; the formation is kept as authored while the graph has one, the seats always.
DEMAND_KEYS = ("type", "capacity", "interaction_radius", "satisfies")


def read_doc(path: Path) -> dict[str, Any]:
    """The format 2 mapping of the scenario file at `path`, its `format` key included."""
    from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT

    from ..scenario.formats import read_scenario

    return {"format": SCENARIO_FORMAT, **read_scenario(path)}


def _at(graph: Scene, obj: str) -> tuple[str, dict[str, Any]] | None:
    fact = next((f for f in graph.facts if f.subject == obj and f.verb == "at"), None)
    return None if fact is None else (str(fact.object).partition(":")[2], dict(fact.params))


def _demand(graph: Scene, obj: str, block: Mapping[str, Any]) -> dict[str, Any]:
    """`block` with the demand keys the graph states for `obj`, its formation while the graph keeps one, its seats as they were."""
    out = {k: v for k, v in block.items() if k not in DEMAND_KEYS}
    if not any(f.subject == obj and f.verb == "has_formation" for f in graph.facts):
        out.pop("formation", None)
    serves = next((f for f in graph.facts if f.subject == obj and f.verb == "serves"), None)
    if serves is not None:
        out["type"] = str(serves.object).partition(":")[2]
        out.update({k: serves.params[k] for k in ("capacity", "interaction_radius") if k in serves.params})
    satisfies = {str(f.object).partition(":")[2]: f.params["amount"] for f in graph.facts if f.subject == obj and f.verb == "satisfies"}
    if satisfies:
        out["satisfies"] = satisfies
    return {k: out[k] for k in [*DEMAND_KEYS, *block] if k in out}


def project(doc: Mapping[str, Any], graph: Scene, origins: Mapping[str, tuple[float, float]], notes: list[str] | None = None) -> dict[str, Any]:
    """`doc` with its objects as the graph has them: removed ones dropped, every one pinned at its map-frame pose, demand from the graph.

    Objects posed at a zone name are pinned too: the runtime draws their points in object order from one RNG, so removing or adding
    an object would move every later one. Pinned, an object the edit leaves alone stands where it stood before.
    """
    from arena_simulation_setup.tree.World.write import ANGLE_DECIMALS, POSITION_DECIMALS

    notes = notes if notes is not None else []
    out = copy.deepcopy(dict(doc))
    kept = []
    for raw in out.get("objects") or []:
        obj = f"obj:{raw.get('name')}"
        placed = _at(graph, obj)
        if obj not in graph.entities or placed is None:
            notes.append(f"{obj}: removed from the scenario, steps or formations naming it are left as they are")
            continue
        level_id, pose = placed
        ox, oy = origins.get(level_id, (0.0, 0.0))
        raw["pose"] = [round(float(pose["x"]) + ox, POSITION_DECIMALS), round(float(pose["y"]) + oy, POSITION_DECIMALS), round(float(pose["yaw"]), ANGLE_DECIMALS)]
        if raw.get("humansim") or any(f.subject == obj and f.verb in ("serves", "satisfies") for f in graph.facts):
            raw["humansim"] = _demand(graph, obj, raw.get("humansim") or {})
            if not raw["humansim"]:
                del raw["humansim"]
        kept.append(raw)
    if "objects" in out:
        out["objects"] = kept
    return out


def origins_of(root: Path, levels: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """The map-frame origin of each level of the world at `root`, as the runtime places scenario objects."""
    from arena_simulation_setup.tree.World.view import WorldView

    return WorldView(root).level_origins() or {level_id: (0.0, 0.0) for level_id in levels}


def write_scenario(source: Path, target: Path, doc: Mapping[str, Any]) -> None:
    """Copy the scenario folder `source` to `target`, its local agent types included, with `doc` as its `scenario.yaml`."""
    import shutil

    from arena_simulation_setup.tree.World.write import to_yaml

    shutil.copytree(source, target, ignore=shutil.ignore_patterns("flowgraph.yaml", "scenario.json"))
    (target / "scenario.yaml").write_text(to_yaml(dict(doc)))

