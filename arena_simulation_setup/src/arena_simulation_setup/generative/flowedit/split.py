"""Splits of edited worlds into train, held-out edits and held-out worlds, and the sweep that records every run of one.

A split file names worlds only. Every edited world in it is made again, under the same name, by `flowedit generate` with
its base's `sources` entry, so the file is all a sweep needs:

    format: flowedit-split/1
    sources:                 # base world: the generate call its edited worlds come from
      hospital_1: {scenario: queue, seed: 0, per_type: 3, steps: 1, controls: true}
    duration: 300            # seconds recorded per run
    settings: {entrances: {rate: 0}}  # crowd settings of every run, a sweep's --settings merged over them (optional)
    seeds:                   # crowd seeds: one list for every partition, or a list per partition
      train: [1, 2, 3, 4, 5] # or apart for its base worlds and its edits: {bases: [1, 2, 3, 4, 5], edits: [1, 2]}
      held_out_edits: [6, 7, 8]
      held_out_worlds: [6, 7, 8]
    baseline_seeds:          # base worlds of a partition also run at these seeds, for baselines only (optional)
      held_out_worlds: [1, 2, 3, 4, 5]
    partitions:
      train: [hospital_1, hospital_1__move_fountain__6e1d6873, hospital_1__insert_fountain_inert__4a32a4fc, ...]
      held_out_edits: [...]  # other edits of the train worlds
      held_out_worlds: [...] # worlds, and their edits, with no run in train
    control_worlds:          # edit world: its geometry-only controls (`controls: true`), listed in the edit's partition
      hospital_1__move_fountain__6e1d6873: [hospital_1__insert_fountain_inert__4a32a4fc, ...]
    crowds:                  # test crowds (optional): a settings file, relative to the split, and the partitions it runs on
      test_orca: {settings: crowds/test_orca.yaml, partitions: [held_out_edits]}
      # optional: only some worlds of its partitions, and its own baseline seeds
      test_size: {settings: crowds/test_size.yaml, partitions: [held_out_edits], worlds: [...], baseline_seeds: [1, 2, 3]}
    robot_task:              # the robot's task in every run of `flowedit suite` (optional), a scenario `robots:` entry
      start: [10.0, 2.0, 1.5708]
      phases: [{loop: [{goto: [14.8, 22.9, -0.82]}, {goto: [10.0, 2.0, -1.5708]}]}]
    contest:                 # the planners `flowedit suite` runs every stage with (optional, needs `robot_task`)
      - {name: dwb_blind, mobile: {driver: nav2, local_planner: dwb}}

A control's name does not say which edit it controls (an inert insert is named after its model), so `control_worlds`
holds the link, as `flowedit generate --controls` writes it in its manifest; a sweep checks it against that manifest.

Every edited world is scored against its base world run with the same seed. A partition that lists edits but not their
base world (held-out edits run at test seeds) gets the base world at its seeds as a reference run, never a training run.
A partition's seeds may differ for its base worlds and its edits (`{bases, edits}`): a base world then runs at more seeds
than its edits, and every edit seed still needs a base run at it, its own or a reference.

A baseline that predicts a scored run from recorded runs (stale flow, transplanted flow) must not see the scored run's
crowd draw, so it is built from base runs at other seeds, and the across-seed floor compares a world's runs across seeds.
`baseline_seeds` gives a partition's base worlds (its listed ones and the bases of its edits) these runs: each runs once
more per baseline seed, as a baseline run (`Job.baseline`). A baseline run belongs to its held-out partition, outside
that partition's own seeds, so code that trains on `train`, or scores a partition at `seeds_of`, never reads one. A base
world that already runs at a seed (a train world under the base crowd) is not run again. A test crowd's `baseline_seeds`
do the same for the base worlds it runs.

The split's own `settings` are its base crowd, which runs on every partition. A test crowd's file is merged over them and
runs only on its partitions, its reference runs included, so an edit still pairs with its base world under one crowd.
A test crowd's `worlds` keeps it to those worlds of its partitions (each still with its base world's reference runs).

Without `robot_task` the robot parks off the map, unseen, and the runs record the crowd alone. With it, `flowedit suite`
adds the task to every run's scenario after the crowd is read, so the people are the same as without it, and the people
see the robot. Every goto of the task is a pose: a named target can resolve differently in an edited world (a moved
object takes its dock along), and a pose keeps the task the same in every world. `contest` lists arena_evaluation
contestants (`name` plus the contestant's keys, as a contest file lists them). `flowedit sweep` records without a robot
and refuses a split with a `robot_task`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPLIT_FORMAT = "flowedit-split/1"
PARTITIONS = ("train", "held_out_edits", "held_out_worlds")
SOURCE_KEYS = ("scenario", "seed", "per_type", "steps", "controls", "generator_version")
CROWD_KEYS = ("settings", "partitions")
#: Keys a test crowd may add to CROWD_KEYS: a subset of its partitions' worlds, and seeds of baseline runs of their base worlds
CROWD_OPTIONAL = ("worlds", "baseline_seeds")
#: The two seed lists of a partition whose base worlds and edits run at different seeds
SEED_KINDS = ("bases", "edits")
#: Top-level keys of a split that `flowedit suite` reads and `flowedit sweep` cannot record
ROBOT_TASK = "robot_task"
CONTEST = "contest"
#: The crowd every split has: its own `settings`, on every partition
BASE_CROWD = "base"
#: `--crowd` of every crowd, so no crowd takes the name
ALL_CROWDS = "all"
_CROWD_NAME = re.compile(r"^[a-z0-9_]+$")


def base_of(world: str) -> str:
    """The base world of an edited world's name (`<base>__<edit id>`), or the name itself."""
    return world.partition("__")[0]


def is_base(world: str) -> bool:
    """Whether `world` is a base world, not an edit of one: a base world runs its source's scenario, an edited world names its own."""
    return base_of(world) == world


def duration_of(doc: Mapping[str, Any]) -> float:
    """Sim seconds every run of the split lasts: its `duration`, else the default of `flowedit run`."""
    from .run import DURATION_S

    return float(doc.get("duration", DURATION_S))


def crowd(doc: Mapping[str, Any], over: Path | None = None, name: str = BASE_CROWD) -> dict[str, Any]:
    """The crowd settings of crowd `name`'s runs: the defaults, the split's `settings`, the crowd's file, then the file `over`."""
    from . import settings as S

    merged = S.load(doc.get("settings") or {})
    files = [Path(f) for f in ([] if name == BASE_CROWD else [_crowd_entry(doc, name)["settings"]])]
    for file in [*files, *([Path(over)] if over is not None else [])]:
        raw = yaml.safe_load(file.read_text()) or {}
        if not isinstance(raw, Mapping):
            raise S.SettingsError(f"{file}: crowd settings are a mapping")
        merged = S.load(S._merge(merged, raw))
    return merged


def crowd_names(doc: Mapping[str, Any]) -> list[str]:
    """The split's crowds: the base crowd, then its test crowds in file order."""
    return [BASE_CROWD, *(doc.get("crowds") or {})]


def partitions_of(doc: Mapping[str, Any], name: str) -> list[str]:
    """The partitions crowd `name` runs on: every one for the base crowd."""
    return list(PARTITIONS) if name == BASE_CROWD else list(_crowd_entry(doc, name)["partitions"])


def _crowd_entry(doc: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    entry = (doc.get("crowds") or {}).get(name)
    if entry is None:
        raise ValueError(f"unknown crowd {name!r}, the split has {crowd_names(doc)}")
    return entry


def _check_crowds(doc: Mapping[str, Any]) -> None:
    """Raise ValueError unless every test crowd has a name, a settings file and partitions to run on."""
    table = doc.get("crowds") or {}
    if not isinstance(table, Mapping):
        raise ValueError("crowds maps a crowd name to its settings file and partitions")
    for name, entry in table.items():
        if not isinstance(name, str) or not _CROWD_NAME.match(name) or name in (BASE_CROWD, ALL_CROWDS):
            raise ValueError(f"crowds: {name!r} is no crowd name (lower case, digits and _, not {BASE_CROWD!r} or {ALL_CROWDS!r})")
        if not isinstance(entry, Mapping) or not set(CROWD_KEYS) <= set(entry) or set(entry) - set(CROWD_KEYS) - set(CROWD_OPTIONAL):
            raise ValueError(f"crowds.{name}: a mapping of {list(CROWD_KEYS)}, optionally {list(CROWD_OPTIONAL)}")
        if not isinstance(entry["settings"], str):
            raise ValueError(f"crowds.{name}.settings is the path of a crowd settings file")
        parts = entry["partitions"]
        if not isinstance(parts, list) or not parts:
            raise ValueError(f"crowds.{name}: names no partition to run on")
        if unknown := sorted(set(parts) - set(PARTITIONS)):
            raise ValueError(f"crowds.{name}: unknown partitions {unknown}, known: {list(PARTITIONS)}")
        if "worlds" in entry:
            worlds = entry["worlds"]
            listed = {w for p in parts for w in (doc.get("partitions") or {}).get(p) or []}
            if not isinstance(worlds, list) or not worlds:
                raise ValueError(f"crowds.{name}.worlds: a non-empty list of worlds of its partitions")
            if stray := sorted(set(worlds) - listed):
                raise ValueError(f"crowds.{name}.worlds: {stray} are in none of its partitions {parts}")
        if "baseline_seeds" in entry:
            _seed_list(f"crowds.{name}.baseline_seeds", entry["baseline_seeds"])
            if clash := sorted({int(s) for s in entry["baseline_seeds"]} & {s for p in parts for s in seeds_of(doc, p)}):
                raise ValueError(f"crowds.{name}.baseline_seeds: {clash} are also seeds its partitions are scored at")


def check(doc: Mapping[str, Any]) -> None:
    """Raise ValueError unless `doc` is a split whose partitions are disjoint and whose held-out worlds never run in train."""
    if doc.get("format") != SPLIT_FORMAT:
        raise ValueError(f"a split has format {SPLIT_FORMAT}, got {doc.get('format')!r}")
    parts = doc.get("partitions") or {}
    if unknown := sorted(set(parts) - set(PARTITIONS)):
        raise ValueError(f"unknown partitions {unknown}, known: {list(PARTITIONS)}")
    sources = doc.get("sources") or {}
    for base, source in sources.items():
        if unknown := sorted(set(source) - set(SOURCE_KEYS)):
            raise ValueError(f"sources.{base}: unknown keys {unknown}, known: {list(SOURCE_KEYS)}")
    seen: dict[str, str] = {}
    for part, worlds in parts.items():
        for world in worlds:
            if world in seen:
                raise ValueError(f"{world} is in both {seen[world]} and {part}")
            seen[world] = part
            if base_of(world) not in sources:
                raise ValueError(f"{part}: {world} has no sources entry for its base {base_of(world)}")
    train = {base_of(w) for w in parts.get("train", [])}
    if leaked := sorted({base_of(w) for w in parts.get("held_out_worlds", [])} & train):
        raise ValueError(f"held-out worlds {leaked} also run in train")
    if stray := sorted({base_of(w) for w in parts.get("held_out_edits", [])} - train):
        raise ValueError(f"held-out edits of {stray}, worlds that never run in train")
    seeds = doc.get("seeds", [0])
    if isinstance(seeds, Mapping):
        if unknown := sorted(set(seeds) - set(PARTITIONS)):
            raise ValueError(f"seeds of unknown partitions {unknown}, known: {list(PARTITIONS)}")
        lists = {part: seeds.get(part) for part, worlds in parts.items() if worlds}
    else:
        lists = {"every partition": seeds}
    for part, values in lists.items():
        if isinstance(values, Mapping):
            if sorted(values) != sorted(SEED_KINDS):
                raise ValueError(f"seeds of {part}: a list, or a mapping of exactly {list(SEED_KINDS)}, got {sorted(values)}")
            for kind in SEED_KINDS:
                _seed_list(f"seeds of {part} ({kind})", values[kind])
        else:
            _seed_list(f"seeds of {part}", values)
    baseline = doc.get("baseline_seeds") or {}
    if not isinstance(baseline, Mapping):
        raise ValueError("baseline_seeds maps a partition to the seeds its base worlds also run at, for baselines only")
    if unknown := sorted(set(baseline) - set(PARTITIONS)):
        raise ValueError(f"baseline_seeds of unknown partitions {unknown}, known: {list(PARTITIONS)}")
    for part, values in baseline.items():
        _seed_list(f"baseline_seeds of {part}", values)
        if part == "train":
            raise ValueError("baseline_seeds of train: a train world's runs are training runs, give it more seeds instead")
        if clash := sorted({int(v) for v in values} & set(seeds_of(doc, part))):
            raise ValueError(f"baseline_seeds of {part}: {clash} are also seeds it is scored at")
    if not isinstance(doc.get("settings") or {}, Mapping):
        raise ValueError("settings is a mapping of crowd settings, as a settings file holds them")
    crowd(doc)
    _check_crowds(doc)
    _check_controls(doc, seen)
    robot_task_of(doc)
    contest_of(doc)


def robot_task_of(doc: Mapping[str, Any]) -> dict[str, Any] | None:
    """The split's `robot_task` as a scenario `robots:` entry, None without one; ValueError unless every goto is a pose."""
    from arena_simulation_setup.shared.task import GoToPhase, LoopPhase
    from arena_simulation_setup.tree.World.Scenario import RobotGoal

    task = doc.get(ROBOT_TASK)
    if task is None:
        return None
    if not isinstance(task, Mapping):
        raise ValueError(f"{ROBOT_TASK} is a scenario robots entry: start, phases")
    if (doc.get("settings") or {}).get("robot"):
        raise ValueError(f"{ROBOT_TASK} and settings.robot both give the robot; settings.robot also moves the people, keep only {ROBOT_TASK}")
    if isinstance(task.get("start"), str):
        raise ValueError(f"{ROBOT_TASK}.start is a pose [x, y, yaw], not the zone {task['start']!r}")
    try:
        goal = RobotGoal.parse(dict(task))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{ROBOT_TASK}: {exc}") from exc
    if not goal.phases:
        raise ValueError(f"{ROBOT_TASK} has no phases: the robot would park, leave the task out for a parked robot")
    gotos = [inner for phase in goal.phases for inner in (phase.phases if isinstance(phase, LoopPhase) else [phase]) if isinstance(inner, GoToPhase)]
    if named := [g.target or ", ".join(g.via) for g in gotos if g.target is not None or g.via]:
        raise ValueError(f"{ROBOT_TASK}: gotos to {named} name a target, give every goto as a pose [x, y, yaw] so it is the same in every world")
    return dict(task)


def contest_of(doc: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The split's `contest`, None without one; ValueError unless it is a list of named contestants and the split has a robot task."""
    contest = doc.get(CONTEST)
    if contest is None:
        return None
    if not isinstance(contest, list) or not contest or not all(isinstance(c, Mapping) and isinstance(c.get("name"), str) and c["name"] for c in contest):
        raise ValueError(f"{CONTEST} is a non-empty list of contestants, each a mapping with a name")
    names = [c["name"] for c in contest]
    if dup := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"{CONTEST}: contestants {dup} appear twice")
    if doc.get(ROBOT_TASK) is None:
        raise ValueError(f"{CONTEST} without a {ROBOT_TASK}: the planners would drive a parked robot nowhere")
    return [dict(c) for c in contest]


def _check_controls(doc: Mapping[str, Any], partition_of: Mapping[str, str]) -> None:
    """Raise ValueError unless every control world sits in its edit's partition, made by a source with `controls: true`."""
    sources = doc.get("sources") or {}
    for base, source in sources.items():
        if not isinstance(source.get("controls", False), bool):
            raise ValueError(f"sources.{base}.controls is true or false, got {source['controls']!r}")
    links = doc.get("control_worlds") or {}
    if not isinstance(links, Mapping):
        raise ValueError("control_worlds maps an edit world to the list of its control worlds")
    controlled: dict[str, str] = {}
    for edit, worlds in links.items():
        if edit not in partition_of:
            raise ValueError(f"control_worlds: {edit} is in no partition")
        if not sources.get(base_of(edit), {}).get("controls"):
            raise ValueError(f"control_worlds: {edit}'s base {base_of(edit)} has no `controls: true`, its controls are not made again")
        for control in worlds or []:
            if control in controlled:
                raise ValueError(f"control_worlds: {control} controls both {controlled[control]} and {edit}")
            controlled[control] = edit
            if base_of(control) != base_of(edit):
                raise ValueError(f"control_worlds: {control} is no edit of {edit}'s base {base_of(edit)}")
            # the control answers whether the edit's change of flow is its demand's, so it is scored where the edit is
            if partition_of.get(control) != partition_of[edit]:
                raise ValueError(f"control_worlds: {control} is in {partition_of.get(control, 'no partition')}, its edit {edit} in {partition_of[edit]}")


def _seed_list(where: str, values: object) -> None:
    if not isinstance(values, list) or not values or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        raise ValueError(f"{where}: a non-empty list of crowd seeds, got {values!r}")


def seeds_of(doc: Mapping[str, Any], partition: str, world: str | None = None) -> list[int]:
    """The crowd seeds `partition` runs at: its own list when `seeds` is a mapping, else the one list. A partition with
    seeds apart for its base worlds and its edits gives `world`'s list, and without `world` every seed of either, in order."""
    seeds = doc.get("seeds", [0])
    entry = seeds.get(partition) or [] if isinstance(seeds, Mapping) else seeds
    if isinstance(entry, Mapping):
        if world is not None:
            return [int(s) for s in entry["bases" if is_base(world) else "edits"]]
        return [int(s) for s in dict.fromkeys([*entry["bases"], *entry["edits"]])]
    return [int(s) for s in entry]


def baseline_seeds_of(doc: Mapping[str, Any], partition: str, crowd: str = BASE_CROWD) -> list[int]:
    """The seeds the base worlds of `partition` also run at under crowd `crowd`, for baselines only."""
    if crowd == BASE_CROWD:
        values = (doc.get("baseline_seeds") or {}).get(partition) or []
    else:
        entry = _crowd_entry(doc, crowd)
        values = entry.get("baseline_seeds") or [] if partition in entry["partitions"] else []
    return [int(s) for s in values]


def load(path: Path) -> dict[str, Any]:
    """The split at `path`, checked, its test crowds' settings files resolved against the split's directory and loaded."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    check(doc)
    for name, entry in (doc.get("crowds") or {}).items():
        file = (Path(path).parent / entry["settings"]).resolve()
        if not file.is_file():
            raise ValueError(f"crowds.{name}: no settings file {file}")
        entry["settings"] = str(file)
        crowd(doc, name=name)
    return doc


@dataclass(frozen=True)
class Job:
    partition: str
    world: str
    seed: int
    #: a base world run at a partition's seeds only for its edits to be scored against, never trained on
    reference: bool = False
    crowd: str = BASE_CROWD
    #: a base world run at a baseline seed: only for the baselines and the across-seed floor, never trained on or scored
    baseline: bool = False

    @property
    def label(self) -> str:
        where = self.partition if self.crowd == BASE_CROWD else f"{self.crowd}/{self.partition}"
        return f"[{where}] {self.world} seed {self.seed}" + (" (reference)" if self.reference else "") + (" (baseline only)" if self.baseline else "")

    @property
    def log_name(self) -> str:
        """The job's log file in a sweep's `_sweep/`."""
        return ("" if self.crowd == BASE_CROWD else f"{self.crowd}__") + f"{self.world}__s{self.seed}.log"

    def argv(self, scenario: str | None, out: Path, duration: float, settings: Path | None) -> list[str]:
        """`flowedit run` of this job: an edited world names its own scenario, a base world the source's."""
        args = ["run", self.world, *([scenario] if scenario and is_base(self.world) else []), "--seed", str(self.seed), "--duration", str(duration), "--out", str(out)]
        return [*args, *(["--settings", str(settings)] if settings is not None else [])]


def jobs(doc: Mapping[str, Any], partitions: Iterable[str] = PARTITIONS, crowd: str = BASE_CROWD) -> list[Job]:
    """Every run of crowd `crowd` on the chosen partitions it runs on: each world once per seed of its partition (its base
    worlds' or its edits' seeds when they differ), in file order, a test crowd's `worlds` only. An edited world whose base
    world runs at none of its seeds among the chosen partitions gets that base world as a reference run at them, first in
    the partition. Then the partition's base worlds run at its baseline seeds, leaving out every (world, seed) the crowd
    runs in any partition."""
    entry = None if crowd == BASE_CROWD else _crowd_entry(doc, crowd)
    keep = set(entry["worlds"]) if entry is not None and "worlds" in entry else None
    every = partitions_of(doc, crowd)
    parts = [p for p in partitions if p in every]

    def listed_in(part: str) -> list[Job]:
        return [Job(part, world, seed, crowd=crowd) for world in (doc["partitions"].get(part) or []) if keep is None or world in keep for seed in seeds_of(doc, part, world)]

    listed = {part: listed_in(part) for part in every}
    have = {(j.world, j.seed) for part in parts for j in listed[part]}
    # a baseline run repeats no run of the crowd, in a chosen partition or not
    ran = have | {(j.world, j.seed) for part in every for j in listed[part]}
    out: list[Job] = []
    for part in parts:
        for job in listed[part]:
            base = base_of(job.world)
            if base != job.world and (base, job.seed) not in have:
                have.add((base, job.seed))
                out.append(Job(part, base, job.seed, reference=True, crowd=crowd))
        out.extend(listed[part])
        for base in dict.fromkeys(base_of(j.world) for j in listed[part]):
            for seed in baseline_seeds_of(doc, part, crowd):
                if (base, seed) not in have | ran:
                    have.add((base, seed))
                    out.append(Job(part, base, seed, crowd=crowd, baseline=True))
    return out


def generate_argv(base: str, source: Mapping[str, Any], manifest: Path | None = None) -> list[str]:
    """`flowedit generate` that saves the edited worlds of `base`, and their controls when the source asks, again under
    the names the split gives them, its manifest at `manifest` when given."""
    return [
        "generate",
        base,
        *(["--scenario", str(source["scenario"])] if source.get("scenario") else []),
        "--seed",
        str(source.get("seed", 0)),
        "--per-type",
        str(source.get("per_type", 1)),
        "--steps",
        str(source.get("steps", 1)),
        *(["--controls"] if source.get("controls") else []),
        *(["--manifest", str(manifest)] if manifest is not None else []),
        "--write",
    ]


def verify(doc: Mapping[str, Any], base: str, manifest: Mapping[str, Any]) -> None:
    """Raise ValueError unless the generate manifest of `base` made every edited world the split lists for it, with the
    controls `control_worlds` gives them: a name the generator no longer makes (its edits changed) would not resolve. A
    source's `generator_version`, when given, must be the manifest's."""
    version = manifest.get("generator_version")
    wanted = (doc.get("sources") or {}).get(base, {}).get("generator_version")
    if wanted is not None and wanted != version:
        raise ValueError(f"{base}: the split's names come from generator version {wanted}, `flowedit generate` is version {version}: regenerate the split's names")
    made: dict[str, list[str]] = {}
    for chain in manifest.get("chains") or []:
        made[chain["world"]] = [c["world"] for c in chain.get("controls") or []]
    controls = {c for cs in made.values() for c in cs}
    listed = {w for worlds in (doc.get("partitions") or {}).values() for w in worlds or [] if base_of(w) == base and w != base}
    if missing := sorted(listed - set(made) - controls):
        raise ValueError(f"{base}: `flowedit generate` (generator version {version}) did not make {missing}, regenerate the split's names")
    for edit, worlds in (doc.get("control_worlds") or {}).items():
        if base_of(edit) == base and (wrong := sorted(set(worlds or []) - set(made.get(edit, [])))):
            raise ValueError(f"{base}: {wrong} are no controls of {edit} in the generate manifest")


def _flowedit(args: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "arena_simulation_setup.generative.flowedit.cli", *args]


def regenerate_worlds(doc: Mapping[str, Any], manifests: Path, log: Callable[[str], None] = print) -> None:
    """Make the split's edited worlds again with `flowedit generate`, manifests under `manifests`, and check every listed name came out."""
    manifests.mkdir(parents=True, exist_ok=True)
    for base, source in doc["sources"].items():
        log(f"generating the edited worlds of {base}")
        manifest = manifests / f"{base}.manifest.yaml"
        subprocess.run(_flowedit(generate_argv(base, source, manifest)), check=True, stdout=subprocess.DEVNULL)
        verify(doc, base, yaml.safe_load(manifest.read_text()) or {})


def sweep(
    doc: Mapping[str, Any],
    out: Path,
    *,
    partitions: Sequence[str] = PARTITIONS,
    crowds: Sequence[str] = (BASE_CROWD,),
    settings: Path | None = None,
    jobs_at_once: int = 1,
    domain_base: int = 150,
    regenerate: bool = True,
    stagger: float = 30.0,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Record every run of the chosen crowds on the chosen partitions under `out`, `jobs_at_once` launches at a time, each
    on its own ROS domain. `settings` is merged over every crowd's settings.

    The edited worlds are generated again first, so the names in the split resolve. A finished run is skipped. Returns the
    count of runs that finished, were skipped and failed; a failed run's reason is in its launch.log. Launches start at
    least `stagger` seconds apart: runtimes coming up at the same moment on a loaded host have stalled before registering."""
    if doc.get(ROBOT_TASK) is not None:
        raise ValueError(f"the split has a {ROBOT_TASK}, which `flowedit run` does not drive: record it with `flowedit suite`")
    logs = out / "_sweep"
    logs.mkdir(parents=True, exist_ok=True)
    if regenerate:
        regenerate_worlds(doc, logs, log)
    from . import settings as S
    from .run import done, scenario_name

    files: dict[str, Path | None] = {}
    queue = []
    counts = {"done": 0, "skipped": 0, "failed": 0}
    for name in crowds:
        settings_doc = crowd(doc, settings, name)
        files[name] = settings
        if doc.get("settings") or settings is not None or name != BASE_CROWD:
            # one file holding the merged settings, so every run reads exactly what the skip check below names its run by
            files[name] = logs / f"settings_{S.digest(settings_doc)}.yaml"
            files[name].write_text(yaml.safe_dump(settings_doc, sort_keys=True))
        for job in jobs(doc, partitions, name):
            # a finished run is skipped here, without a launch slot or a stagger
            if done(out / f"{job.world}__{scenario_name(doc['sources'][base_of(job.world)].get('scenario'), settings_doc, job.seed)}"):
                counts["skipped"] += 1
                log(f"{job.label}: already recorded")
            else:
                queue.append(job)
    duration = duration_of(doc)
    running: dict[int, tuple[Job, subprocess.Popen]] = {}
    last_start = -float("inf")
    while queue or running:
        for slot in [s for s in range(jobs_at_once) if s not in running]:
            if not queue or time.monotonic() - last_start < stagger:
                break
            last_start = time.monotonic()
            job = queue.pop(0)
            env = {**os.environ, "ROS_DOMAIN_ID": str(domain_base + slot)}
            argv = job.argv(doc["sources"][base_of(job.world)].get("scenario"), out, duration, files[job.crowd])
            with (logs / job.log_name).open("w") as fh:
                running[slot] = (job, subprocess.Popen(_flowedit(argv), stdout=fh, stderr=subprocess.STDOUT, env=env))
            log(f"{job.label}: started on domain {domain_base + slot}")
        time.sleep(2.0)
        for slot, (job, proc) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            text = (logs / job.log_name).read_text()
            state = "failed" if proc.returncode else "skipped" if "already recorded" in text else "done"
            counts[state] += 1
            log(f"{job.label}: {state}" + (f" ({text.strip().splitlines()[-1] if text.strip() else proc.returncode})" if state == "failed" else ""))
    return counts


__all__ = ["ALL_CROWDS", "BASE_CROWD", "CONTEST", "PARTITIONS", "ROBOT_TASK", "SEED_KINDS", "SPLIT_FORMAT", "Job", "base_of", "baseline_seeds_of", "check", "contest_of", "crowd", "crowd_names", "duration_of", "generate_argv", "is_base", "jobs", "load", "partitions_of", "regenerate_worlds", "robot_task_of", "seeds_of", "sweep", "verify"]
