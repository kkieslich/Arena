"""Reading and writing the `kg.yaml` kg of a world."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .scene import GENERATED, Scene, Value

FORMAT = "arena-kg/1"

_TOP_KEYS = ("format", "uses", "entities", "facts", "constraints")
_ENTITY_KEYS = ("id", "props")
_FACT_KEYS = ("subject", "verb", "object", "params", "source")
_CONSTRAINT_KEYS = ("id", "kind", "scope", "params", "hard")


def _keys(item: object, where: str, allowed: tuple[str, ...], required: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(item).__name__}")
    for key in item:
        if key not in allowed:
            raise ValueError(f"{where}: unknown key {key!r}")
    for key in required:
        if key not in item:
            raise ValueError(f"{where}: missing key {key!r}")
    return item


def _list(raw: object, where: str) -> list[Any]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"{where}: expected a list, got {type(raw).__name__}")
    return raw


def _string(raw: object, where: str) -> str:
    if not isinstance(raw, str):
        raise ValueError(f"{where}: expected a string, got {type(raw).__name__}")
    return raw


def _scalar(raw: object, where: str) -> Value:
    if not isinstance(raw, str | int | float | bool):
        raise ValueError(f"{where}: expected a string, number or boolean, got {type(raw).__name__}")
    return raw


def _scalars(raw: object, where: str) -> dict[str, Value]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a mapping, got {type(raw).__name__}")
    return {_string(k, f"{where}: key"): _scalar(v, f"{where}.{k}") for k, v in raw.items()}


def _parse(data: object) -> Scene:
    top = _keys(data, "top level", _TOP_KEYS, ("format", "uses"))
    if top["format"] != FORMAT:
        raise ValueError(f"format: expected {FORMAT!r}, got {top['format']!r}")
    uses = [_string(u, f"uses[{i}]") for i, u in enumerate(_list(top["uses"], "uses"))]
    if not uses:
        raise ValueError("uses: expected at least one package")
    scene = Scene(uses)
    for i, raw in enumerate(_list(top.get("entities"), "entities")):
        where = f"entities[{i}]"
        item = _keys(raw, where, _ENTITY_KEYS, ("id",))
        id = _string(item["id"], f"{where}.id")
        if id in scene.entities:
            raise ValueError(f"{where}: duplicate entity {id!r}")
        try:
            scene.add_entity(id, _scalars(item.get("props"), f"{where}.props"))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from e
    for i, raw in enumerate(_list(top.get("facts"), "facts")):
        where = f"facts[{i}]"
        item = _keys(raw, where, _FACT_KEYS, ("subject", "verb", "object"))
        subject = _string(item["subject"], f"{where}.subject")
        verb = _string(item["verb"], f"{where}.verb")
        obj = _scalar(item["object"], f"{where}.object")
        params = _scalars(item.get("params"), f"{where}.params")
        if scene.get_fact(subject, verb, obj, params) is not None:
            raise ValueError(f"{where}: duplicate fact {(subject, verb, obj)}")
        try:
            scene.add_fact(subject, verb, obj, params, _string(item.get("source", GENERATED), f"{where}.source"))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from e
    for i, raw in enumerate(_list(top.get("constraints"), "constraints")):
        where = f"constraints[{i}]"
        item = _keys(raw, where, _CONSTRAINT_KEYS, ("id", "kind", "scope"))
        id = _string(item["id"], f"{where}.id")
        if id in scene.constraints:
            raise ValueError(f"{where}: duplicate constraint {id!r}")
        scope = [_string(s, f"{where}.scope[{j}]") for j, s in enumerate(_list(item["scope"], f"{where}.scope"))]
        hard = item.get("hard", True)
        if not isinstance(hard, bool):
            raise ValueError(f"{where}.hard: expected a boolean, got {type(hard).__name__}")
        scene.add_constraint(id, _string(item["kind"], f"{where}.kind"), scope, _scalars(item.get("params"), f"{where}.params"), hard)
    return scene


def load(source: str | os.PathLike[str] | Mapping[str, Any]) -> Scene:
    """Parse a KG from a file path or an already loaded mapping, rejecting unknown keys."""
    if isinstance(source, Mapping):
        return _parse(source)
    path = Path(source)
    with path.open() as f:
        data = yaml.safe_load(f)
    try:
        return _parse(data)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def dump(scene: Scene) -> dict[str, Any]:
    """The KG mapping of `scene`, in insertion order, without empty or default fields."""
    entities = []
    for e in scene.entities.values():
        entities.append({"id": e.id, **({"props": dict(e.props)} if e.props else {})})
    facts = []
    for f in scene.facts:
        item: dict[str, Any] = {"subject": f.subject, "verb": f.verb, "object": f.object}
        if f.params:
            item["params"] = dict(f.params)
        if f.source != GENERATED:
            item["source"] = f.source
        facts.append(item)
    constraints = []
    for c in scene.constraints.values():
        item = {"id": c.id, "kind": c.kind, "scope": list(c.scope)}
        if c.params:
            item["params"] = dict(c.params)
        if not c.hard:
            item["hard"] = False
        constraints.append(item)
    return {"format": FORMAT, "uses": scene.uses, "entities": entities, "facts": facts, "constraints": constraints}


def dumps(scene: Scene) -> str:
    """`scene` as KG YAML text."""
    return yaml.safe_dump(dump(scene), sort_keys=False, default_flow_style=None, width=1 << 16)


def write(scene: Scene, path: str | os.PathLike[str]) -> None:
    """Write `scene` as YAML to `path`."""
    Path(path).write_text(dumps(scene))
