"""Cross-checks of a KG scene against the names its `world.yaml` declares."""

from __future__ import annotations

from collections.abc import Collection, Mapping

from .scene import Scene

RESOLVED_KINDS = ("room", "door", "floor", "elevator", "stair", "light", "sound", "microphone")
GEOMETRY_PROPS = ("boundary", "line", "pose", "scale", "model", "area", "sim")


def check_against_world(scene: Scene, names: Mapping[str, Collection[str]]) -> list[str]:
    """Problems of `scene` against `names`, the world names per kind; KG-only kinds are never checked."""
    problems = []
    for e in scene.entities.values():
        if e.kind in RESOLVED_KINDS:
            required = True
        elif e.kind == "obj":
            required = e.props.get("placed", True) is not False
        else:
            continue
        if e.name not in names.get(e.kind, ()):
            if required:
                problems.append(f"{e.id}: no {e.kind} named {e.name!r} in world.yaml")
            continue
        for prop in e.props:
            if prop in GEOMETRY_PROPS:
                problems.append(f"{e.id}: geometry prop {prop!r} on an entity that resolves in world.yaml")
    return problems
