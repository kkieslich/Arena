"""How people use an object, read from HumanSim: the agent type steps that name it, else its `humansim:` block against HumanSim's interaction registry."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import attrs
import yaml

Duration = float | Mapping[str, float]

#: Word endings of an object's type or model that give the posture HumanSim's object-bound kinds declare (`armchair` is seated, `bedside` is not prone).
POSTURE_WORDS: dict[str, tuple[str, ...]] = {
    "prone": ("bed", "stretcher", "cot"),
    "seated": ("chair", "seat", "sofa", "bench", "stool", "couch"),
}


def handle(kind: str) -> str | None:
    """HumanSim's handle of interaction `kind`: `object`, `none` (symmetric), `tag` (SERVICE) or `agent` (BLOCK), None for no kind."""
    from arena_humansim.core.interaction_kinds import InteractionType

    try:
        return InteractionType[kind].kind.handle.kind.value
    except KeyError:
        return None


def kinds(handle_kind: str | None = None) -> list[str]:
    """HumanSim's interaction kinds in registry order, those of `handle_kind` only when given."""
    from arena_humansim.core.interaction_kinds import InteractionType

    return [t.name for t in InteractionType if handle_kind is None or t.kind.handle.kind.value == handle_kind]


def posture(kind: str) -> str:
    from arena_humansim.core.interaction_kinds import InteractionType

    return InteractionType[kind].kind.posture


@attrs.frozen
class Use:
    """One way an agent type uses an object: `kind` at `target` (an object id or type)."""

    target: str
    kind: str
    origin: str
    duration: Duration | None = None
    satisfies: Mapping[str, float] = attrs.field(factory=dict)
    #: A symmetric (or SERVICE) kind run beside the object after a `go_to` to it, not on the object itself.
    meet: bool = False
    tag: str | None = None
    offer: bool = False


def _use(target: str, kind: str, origin: str, body: Mapping[str, Any], *, meet: bool, extra: Mapping[str, float] | None = None) -> Use:
    satisfies = {str(k): float(v) for k, v in {**(extra or {}), **(body.get("satisfies") or {})}.items()}
    tag = body.get("target") if handle(kind) == "tag" and isinstance(body.get("target"), str) else None
    return Use(target, kind, origin, body.get("duration"), satisfies, meet, tag, bool(body.get("offer")))


def step_uses(files: list[Path]) -> Iterator[Use]:
    """Every object use the agent type `files` author, in file, sequence and step order.

    An object-bound step (`target` + USE, QUEUE_USE, SIT_ON, LIE_ON) uses its target. A `go_to` to an object followed by a
    step whose kind takes no object (GROUP_CONVERSATION, TALK_TO, SERVICE, ...) uses that object as a meeting point, the
    authored way to "meet at an object" in HumanSim.
    """
    for path in files:
        raw = yaml.safe_load(path.read_text()) or {}
        for sequence, spec in (raw.get("sequences") or {}).items():
            steps = [(name, body) for name, body in ((spec or {}).get("steps") or {}).items() if isinstance(body, Mapping)]
            for i, (name, body) in enumerate(steps):
                target, kind = body.get("target"), body.get("interaction")
                if isinstance(kind, str) and isinstance(target, str) and handle(kind) == "object":
                    yield _use(target, kind, f"{path.name}#{sequence}.{name}", body, meet=False)
                elif body.get("kind") == "go_to" and isinstance(target, str) and i + 1 < len(steps):
                    after_name, after = steps[i + 1]
                    after_kind = after.get("interaction")
                    if isinstance(after_kind, str) and handle(after_kind) in ("none", "tag"):
                        yield _use(target, after_kind, f"{path.name}#{sequence}.{after_name}", after, meet=True, extra=body.get("satisfies"))


def by_rule(block: Mapping[str, Any], model: str = "") -> tuple[str, bool, str] | None:
    """`(kind, meet, why)` for an object no step uses, read off its `humansim:` block against HumanSim's registry, None for a marker.

    In order: seats, or a posture word in its type, pick the object-bound kind of that posture (a bed is LIE_ON, a chair
    SIT_ON); a formation picks the kind whose default formation it is (an object-bound one first: `line` is QUEUE_USE,
    `f_formation` the symmetric GROUP_CONVERSATION at the object); a posture word in its model as for the type; a
    capacity alone is USE.
    """
    from arena_humansim.core.interaction_kinds import InteractionType

    def by_posture(words: str, seats: bool = False) -> tuple[str, bool, str] | None:
        tokens = [t for t in re.split(r"[^a-z]+", words.lower()) if t]
        for pose, hints in POSTURE_WORDS.items():
            if any(t.endswith(h) for t in tokens for h in hints) or (pose == "seated" and seats):
                return next(k for k in kinds("object") if posture(k) == pose), False, f"rule:{pose}"
        return None

    if found := by_posture(str(block.get("type", "")), bool(block.get("seats"))):
        return found
    formation = (block.get("formation") or {}).get("type")
    if formation:
        matching = [t.name for t in InteractionType if t.kind.formation_default is not None and t.kind.formation_default.type == formation]
        for wanted, meet in (("object", False), ("none", True)):
            for kind in matching:
                if handle(kind) == wanted:
                    return kind, meet, f"rule:formation {formation}"
    if found := by_posture(model):
        return found
    if int(block.get("capacity") or 0) > 0:
        return "USE", False, "rule:capacity"
    return None
