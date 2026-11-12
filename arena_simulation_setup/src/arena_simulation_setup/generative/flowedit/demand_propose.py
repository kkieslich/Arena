"""Model-proposed demand classes for the models `demand.prefill` leaves undecided, merged into the proposals file as `by: llm` entries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from ..scenario.propose.client import DEFAULT_MODEL, Unavailable, ask, prompt_sha256, strip_fence
from .demand import PROPOSALS, PROPOSALS_FORMAT, UNDECIDED, Class, Stage, Tags, coverage, load_proposals, model_of, objects, proposed, source

if TYPE_CHECKING:
    from arena_simulation_setup.kg import Scene

#: The model's answer for a model it cannot place: never written, the model stays undecided for a human
UNKNOWN = "unknown"
WHY_WORDS = 12
HEADER = """\
# FLOWEDIT demand proposals: the demand class (`demand_classes.yaml`) of asset models no class rule covers, by model.
# `flowedit demand <world> --propose --write` asks a model and adds its answers with `by: llm`; `flowedit build --prefill
# llm` pre-fills them marked `by: llm`. To accept or correct a proposal, set its class and drop `by: llm`: an entry
# without it is a human's, never replaced, and pre-filled by `--prefill class`.
"""


def system(classes: Mapping[str, Class]) -> str:
    """The system half of the request: the classes of the table and how to answer."""
    width = max(len(name) for name in [*classes, UNKNOWN])
    rows = "\n".join(f"  {name:{width}s} - {c.desc}" for name, c in classes.items())
    return f"""\
You decide what people do with the objects of a building in a pedestrian simulator. Each object model gets one class,
and the class decides whether simulated people walk to the object and how they use it.

Answer with one class per model, from exactly this set:

{rows}
  {UNKNOWN:{width}s} - the evidence does not support any of the above

Pick a class with demand only for an object a visitor or staff member would walk to and use as the class says. Machines,
instruments and furniture that only a specialist operates during a procedure are `inert`: nobody walks to them in the
simulation. `{UNKNOWN}` is a real answer and the right one when the name and the evidence do not say what the object is. A
wrong confident answer sends a crowd to a cupboard; an honest `{UNKNOWN}` sends the model to a human.

Reply with JSON only, no prose and no code fence:
{{"<model>": {{"class": "<class>", "why": "<at most {WHY_WORDS} words>"}}}}
"""


@attrs.frozen(kw_only=True)
class Proposal:
    model: str
    cls: str
    why: str = ""
    llm: str = ""
    prompt_sha: str = ""

    @property
    def confident(self) -> bool:
        return self.cls != UNKNOWN


def unproposed(scene: Scene, proposals: Mapping[str, Mapping[str, Any]] | None = None) -> list[str]:
    """The models of the undecided objects of `scene` that have no entry in the proposals yet, sorted."""
    proposals = proposals if proposals is not None else load_proposals()
    return sorted({model_of(scene, obj) for obj in objects(scene) if source(scene, obj) == UNDECIDED} - set(proposals))


def _evidence(scene: Scene, model: str, tags: Tags) -> dict[str, Any]:
    """What the graph and the asset say about one model: how many stand where, and its tags."""
    mine = {obj for obj in objects(scene) if model_of(scene, obj) == model}
    purposes = {f.subject: str(f.object).partition(":")[2] for f in scene.facts if f.verb == "has_purpose"}
    rooms = sorted({f.subject for f in scene.facts if f.verb == "contains" and f.object in mine and f.subject.startswith("room:")})
    out: dict[str, Any] = {"count": len(mine), "rooms": {room.partition(":")[2]: purposes.get(room, UNKNOWN) for room in rooms} or "none"}
    if found := tags(model):
        out["tags"] = list(found)
    return out


def build_prompt(scene: Scene, models: Sequence[str], tags: Tags, *, world: str) -> str:
    """The user half of the request: every model of the world with what is decided, then the models in question."""
    decided: dict[str, str] = {}
    for row in coverage(scene):
        if row.source != UNDECIDED:
            decided.setdefault(row.model, row.decision)
    asked = {model: _evidence(scene, model, tags) for model in models}
    return f"World: {world}\n\nEvery object model of this world that is already decided, with its service type (and needs) or inert:\n{json.dumps(decided, indent=2)}\n\nClassify only these, using their names, where they stand and what the rest of the world suggests:\n{json.dumps(asked, indent=2)}\n"


def parse(text: str, asked: Sequence[str], classes: Mapping[str, Class], *, model: str, sha: str) -> tuple[list[Proposal], list[str]]:
    """Proposals and complaints. Anything unasked-for or unrecognized is dropped, never coerced."""
    try:
        answer = json.loads(strip_fence(text))
    except json.JSONDecodeError as exc:
        return [], [f"response was not JSON: {exc}"]
    if not isinstance(answer, dict):
        return [], [f"response was {type(answer).__name__}, expected an object"]

    proposals: list[Proposal] = []
    complaints: list[str] = []
    wanted = set(asked)
    for name, entry in answer.items():
        if name not in wanted:
            complaints.append(f"{name}: not one of the models asked about")
            continue
        cls = str(entry.get("class") if isinstance(entry, Mapping) else entry)
        if cls != UNKNOWN and cls not in classes:
            complaints.append(f"{name}: {cls!r} is not a class")
            continue
        # a `why` is free text: a colon would let the graph read `<word>: ...` as a reference to an entity
        why = " ".join(str(entry.get("why", "")).replace(":", " ").split()) if isinstance(entry, Mapping) else ""
        proposals.append(Proposal(model=name, cls=cls, why=why, llm=model, prompt_sha=sha))

    for missing in sorted(wanted - {p.model for p in proposals}):
        complaints.append(f"{missing}: no answer returned")
    return proposals, complaints


def propose(scene: Scene, models: Sequence[str], classes: Mapping[str, Class], tags: Tags, *, world: str, model: str | None = None) -> tuple[list[Proposal], list[str]]:
    """Ask the model about `models` of this world. One call, or none if `models` is empty."""
    asked = list(models)
    if not asked:
        return [], []
    head, user = system(classes), build_prompt(scene, asked, tags, world=world)
    sha = prompt_sha256(head, user)
    return parse(ask(head, user, model=model).text, asked, classes, model=model or DEFAULT_MODEL, sha=sha)


def merge(proposals: Sequence[Proposal], path: Path = PROPOSALS) -> tuple[int, list[str]]:
    """Write confident proposals into the proposals file at `path`, never over a human's entry, as (written, notes)."""
    from arena_simulation_setup.tree.World.write import to_yaml

    entries = load_proposals(path)
    written = 0
    kept: list[str] = []
    for proposal in proposals:
        if not proposal.confident:
            kept.append(f"{proposal.model}: model returned `{UNKNOWN}` - left for a human")
            continue
        current = entries.get(proposal.model)
        if current is not None and not proposed(current):
            kept.append(f"{proposal.model}: reviewed by a human as {current['class']!r}")
            continue
        entry = {"class": proposal.cls, "by": Stage.LLM.value, "model": proposal.llm, "prompt_sha": proposal.prompt_sha, "why": proposal.why}
        entry = {k: v for k, v in entry.items() if v}
        if current == entry:
            continue
        entries[proposal.model] = entry
        written += 1
    if written:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(HEADER + to_yaml({"format": PROPOSALS_FORMAT, "models": dict(sorted(entries.items()))}), encoding="utf-8")
    return written, kept


__all__ = ["UNKNOWN", "Proposal", "Unavailable", "build_prompt", "merge", "parse", "propose", "system", "unproposed"]
