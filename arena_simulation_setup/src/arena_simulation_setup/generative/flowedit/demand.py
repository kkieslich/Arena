"""Pre-fill the demand of a FLOWEDIT graph's objects nobody authored: by the class of the object's asset, then by a model's proposal."""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import yaml

from .build import Duration, add_demand, add_uses
from .edit import DEMAND_VERBS, INERT

if TYPE_CHECKING:
    from arena_simulation_setup.kg import Scene, Value
    from arena_simulation_setup.tree.World.model import World

CLASSES = Path(__file__).with_name("demand_classes.yaml")
PROPOSALS = Path(__file__).with_name("demand_proposals.yaml")
CLASSES_FORMAT = "flowedit-demand-classes/1"
PROPOSALS_FORMAT = "flowedit-demand-proposals/1"
CLASS_KEYS = ("desc", "kinds", "hoi", "words", "humansim", "dwell")
PROPOSAL_KEYS = ("class", "by", "model", "prompt_sha", "why")
#: In a class's `humansim:` block, the name of the object the block is given to (a formation anchored at the object itself)
SELF = "{name}"
#: `why` of a fact pre-filled from a proposal a human accepted
REVIEWED = "reviewed proposal"
#: The source of a fact without `by`
AUTHORED = "authored"
UNDECIDED = "undecided"
Tags = Callable[[str], Sequence[str] | None]


class Stage(enum.StrEnum):
    """What may decide an object nobody authored, the `by` of the facts it adds."""

    CLASS = "class"
    LLM = "llm"


@attrs.frozen
class Class:
    """One class of the table: the `humansim:` block its objects get, None for an inert class."""

    name: str
    desc: str = ""
    block: Mapping[str, Any] | None = None
    dwell: Duration | None = None
    kinds: tuple[str, ...] = ()
    hoi: tuple[str, ...] = ()
    words: tuple[str, ...] = ()


def load_classes(path: Path = CLASSES) -> dict[str, Class]:
    """The class table at `path` by class name, in file order."""
    doc = yaml.safe_load(path.read_text()) or {}
    if doc.get("format") != CLASSES_FORMAT:
        raise ValueError(f"{path}: format: expected {CLASSES_FORMAT!r}, got {doc.get('format')!r}")
    classes: dict[str, Class] = {}
    taken: dict[tuple[str, str], str] = {}
    for name, raw in (doc.get("classes") or {}).items():
        raw = raw or {}
        if unknown := sorted(set(raw) - set(CLASS_KEYS)):
            raise ValueError(f"{path}: classes.{name}: unknown keys {unknown}, known: {list(CLASS_KEYS)}")
        block = raw.get("humansim")
        if raw.get("dwell") is not None and not (block or {}).get("satisfies"):
            raise ValueError(f"{path}: classes.{name}: a dwell needs a need the class satisfies")
        for key in ("kinds", "hoi", "words"):
            for value in raw.get(key) or []:
                if (other := taken.setdefault((key, str(value)), name)) != name:
                    raise ValueError(f"{path}: classes.{name}: {key} {value!r} is already of class {other!r}")
        classes[name] = Class(name, str(raw.get("desc", "")), block, raw.get("dwell"), *(tuple(str(v) for v in raw.get(key) or []) for key in ("kinds", "hoi", "words")))
    return classes


def load_proposals(path: Path = PROPOSALS) -> dict[str, dict[str, Any]]:
    """The proposals at `path` by model: `{class, by, model, prompt_sha, why}`, an entry without `by: llm` a human's."""
    if not path.is_file():
        return {}
    doc = yaml.safe_load(path.read_text()) or {}
    if doc.get("format") != PROPOSALS_FORMAT:
        raise ValueError(f"{path}: format: expected {PROPOSALS_FORMAT!r}, got {doc.get('format')!r}")
    proposals = {}
    for model, raw in (doc.get("models") or {}).items():
        if not isinstance(raw, Mapping) or not raw.get("class"):
            raise ValueError(f"{path}: models.{model}: needs a class")
        if unknown := sorted(set(raw) - set(PROPOSAL_KEYS)):
            raise ValueError(f"{path}: models.{model}: unknown keys {unknown}, known: {list(PROPOSAL_KEYS)}")
        proposals[str(model)] = dict(raw)
    return proposals


def proposed(entry: Mapping[str, Any]) -> bool:
    """Whether a model proposed `entry` and no human has accepted it."""
    return entry.get("by") == Stage.LLM


def name_words(model: str) -> list[str]:
    """The words of a model's name, split at non-letters and case changes, then each two neighbours joined, lowercase (`SM_WheelChair_01a`: sm, wheel, chair, a, smwheel, wheelchair, chaira)."""
    words = [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", model.rsplit("/", 1)[-1])]
    return words + [a + b for a, b in zip(words, words[1:], strict=False)]


def by_tags(tags: Sequence[str] | None, classes: Mapping[str, Class]) -> tuple[str, str] | None:
    """`(class, why)` of an asset by its `kind` tag, else a `hoi` tag, None when no class lists one of them."""
    from arena_assets.semantic import parse_tags

    facets = parse_tags([str(t) for t in tags or []])
    for facet, values in (("kind", "kinds"), ("hoi", "hoi")):
        for value in facets.get(facet, []):
            for c in classes.values():
                if value in getattr(c, values):
                    return c.name, f"{facet}::{value}"
    return None


def by_name(model: str, classes: Mapping[str, Class]) -> tuple[str, str] | None:
    """`(class, why)` of an asset by the longest class word that ends a word of its name, None when none does."""
    words = name_words(model)
    found = [(len(word), word, c.name) for c in classes.values() for word in c.words if any(w.endswith(word) for w in words)]
    if not found:
        return None
    _, word, name = max(found)
    return name, f"name:{word}"


def classify(model: str, tags: Sequence[str] | None, classes: Mapping[str, Class]) -> tuple[str, str] | None:
    """`(class, why)` of an asset: by its tags, else by its name."""
    return by_tags(tags, classes) or by_name(model, classes)


def asset_tags(root: Path, world: World, models: Sequence[str]) -> dict[str, list[str] | None]:
    """The annotation tags of each of `models`, resolved as the world at `root` resolves its assets, None for a model that does not resolve."""
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.World.view import WorldView, world_assets

    tags: dict[str, list[str] | None] = {}
    with world_assets(WorldView(root), world):
        for model in sorted(set(models)):
            try:
                view = Identifier._run_sync(Obstacle.parse({"name": "probe", "model": model, "pose": [0.0, 0.0, 0.0]}).model.resolve())
                found = (view.annotation or {}).get("tags")
            except Exception:  # noqa: BLE001 - any failure to resolve leaves the model to its name
                found = None
            tags[model] = [str(t) for t in found] if found is not None else None
    return tags


def world_tags(root: Path, scene: Scene) -> Tags:
    """The tags of a model as the world at `root` resolves its asset, for the models of the objects of `scene`."""
    from ..scenario.formats import load_world

    tags = asset_tags(root, load_world(root), [model_of(scene, obj) for obj in objects(scene)])
    return tags.get


def model_of(scene: Scene, obj: str) -> str:
    return str(scene.entities[obj].props.get("model", ""))


def objects(scene: Scene) -> list[str]:
    """The objects of `scene` that stand for an asset, by id: a demand-only node has no model and is not an object to decide."""
    return sorted(id for id in scene.entities if id.startswith("obj:") and model_of(scene, id))


def source(scene: Scene, obj: str) -> str:
    """Who decided the demand of `obj`: `authored`, `class`, `llm`, or `undecided` without demand and without `inert`."""
    facts = [f for f in scene.facts if f.subject == obj and (f.verb in DEMAND_VERBS or f.verb == INERT)]
    if not facts:
        return UNDECIDED
    if any("by" not in f.params for f in facts):
        return AUTHORED
    return Stage.LLM.value if any(f.params["by"] == Stage.LLM for f in facts) else Stage.CLASS.value


def undecided(scene: Scene) -> list[str]:
    """The objects of `scene` with neither demand nor `inert`."""
    return [obj for obj in objects(scene) if source(scene, obj) == UNDECIDED]


def check(scene: Scene) -> list[str]:
    """Every object of `scene` that is `inert` and has demand, which no object may."""
    inert = {f.subject for f in scene.facts if f.verb == INERT}
    return [f"{obj}: is inert and has demand" for obj in sorted({f.subject for f in scene.facts if f.verb in DEMAND_VERBS} & inert)]


def _like(scene: Scene, block: Mapping[str, Any]) -> str | None:
    """An object that already serves a need of `block`, the first by id: the reader takes one dwell and one use per need."""
    needs = {f"need:{need}" for need in block.get("satisfies") or {}}
    return min((f.subject for f in scene.facts if f.verb == "satisfies" and str(f.object) in needs), default=None)


def _fill(scene: Scene, obj: str, cls: Class, mark: Mapping[str, Value], notes: list[str]) -> None:
    """Give `obj` the demand of `cls`, each fact marked with `mark`."""
    if cls.block is None:
        scene.add_fact(obj, INERT, True, mark)
        return
    name = obj.partition(":")[2]
    block = yaml.safe_load(yaml.safe_dump(dict(cls.block)).replace(SELF, name))
    before = {f.key for f in scene.facts if f.subject == obj}
    like = _like(scene, block)
    dwell: tuple[Duration, str] | None = None
    if like is not None:
        # as the object that already serves the need: one step serves a need, so its objects share their type, use and dwell
        for f in [f for f in scene.facts if f.subject == like and f.verb in ("dwells", "affords")]:
            scene.add_fact(obj, f.verb, f.object, f.params)
        served = next((str(f.object) for f in scene.facts if f.subject == like and f.verb == "serves"), None)
        if served != f"service:{block.get('type')}":
            notes.append(f"{obj}: pre-filled as service:{block.get('type')}, but {like} serves the same need as {served}: the reader refuses objects of one need with different types")
    elif cls.dwell is not None and block.get("satisfies"):
        dwell = (cls.dwell, f"{CLASSES.name}#{cls.name}")
    add_demand(scene, obj, block, dwell, notes)
    if like is None:
        add_uses(scene, obj, block, model_of(scene, obj), [])
    for f in scene.facts:
        if f.subject == obj and f.key not in before and f.verb in DEMAND_VERBS:
            f.params.update(mark)


def prefill(
    scene: Scene,
    stages: Sequence[Stage],
    tags: Tags,
    notes: list[str] | None = None,
    classes: Mapping[str, Class] | None = None,
    proposals: Mapping[str, Mapping[str, Any]] | None = None,
) -> Scene:
    """Decide every undecided object of `scene`, in place, by the `stages` given.

    `Stage.CLASS` takes the class a human set for the object's model in the proposals, else the class of its asset: by
    its tags (`tags(model)`, None for an asset that does not resolve), else its name. `Stage.LLM` takes a model's
    proposal for what is still undecided. An object gets its class's `humansim:` block as `build` reads an authored one,
    or `inert`; every fact carries `by` and `why`. Authored demand and decided objects are left as they are.
    """
    notes = notes if notes is not None else []
    classes = classes if classes is not None else load_classes()
    proposals = proposals if proposals is not None else load_proposals()
    for obj in undecided(scene):
        model = model_of(scene, obj)
        entry = proposals.get(model)
        found: tuple[str, dict[str, Value]] | None = None
        if Stage.CLASS in stages:
            if entry is not None and not proposed(entry):
                found = str(entry["class"]), {"by": Stage.CLASS.value, "why": REVIEWED}
            elif (classed := classify(model, tags(model), classes)) is not None:
                found = classed[0], {"by": Stage.CLASS.value, "why": classed[1]}
        if found is None and Stage.LLM in stages and entry is not None and proposed(entry):
            found = str(entry["class"]), {"by": Stage.LLM.value, **{k: str(entry[k]) for k in ("model", "prompt_sha", "why") if entry.get(k)}}
        if found is None:
            continue
        name, mark = found
        if name not in classes:
            raise ValueError(f"{obj}: model {model!r} is proposed as class {name!r}, known: {sorted(classes)}")
        _fill(scene, obj, classes[name], mark, notes)
    problems = [*scene.validate(), *check(scene)]
    if problems:
        raise ValueError("; ".join(problems))
    return scene


@attrs.frozen
class Row:
    """The objects of one model and what decided them."""

    model: str
    count: int
    decision: str
    source: str
    why: str = ""


def _decision(scene: Scene, obj: str) -> tuple[str, str]:
    """What `obj` is decided as (`inert`, its service type and needs, or `undecided`) and the `why` of the decision."""
    facts = [f for f in scene.facts if f.subject == obj and (f.verb in DEMAND_VERBS or f.verb == INERT)]
    why = next((str(f.params["why"]) for f in facts if f.params.get("why")), "")
    if any(f.verb == INERT for f in facts):
        return INERT, why
    service = next((str(f.object).partition(":")[2] for f in facts if f.verb == "serves"), "")
    needs = sorted(str(f.object).partition(":")[2] for f in facts if f.verb == "satisfies")
    return (" ".join(filter(None, [service, f"({', '.join(needs)})" if needs else ""])) or ("demand" if facts else UNDECIDED)), why


def coverage(scene: Scene) -> list[Row]:
    """One row per model, source and decision of the objects of `scene`, undecided models last."""
    counted: dict[tuple[str, str, str, str], int] = {}
    for obj in objects(scene):
        decision, why = _decision(scene, obj)
        key = (model_of(scene, obj), decision, source(scene, obj), why)
        counted[key] = counted.get(key, 0) + 1
    rows = [Row(model, count, decision, by, why) for (model, decision, by, why), count in counted.items()]
    return sorted(rows, key=lambda r: (r.source == UNDECIDED, r.model, r.source, r.decision))
