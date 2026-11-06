"""Splits of edited worlds into train, held-out edits and held-out worlds, and the sweep that records every run of one.

A split file names worlds only. Every edited world in it is made again, under the same name, by `flowedit generate` with
its base's `sources` entry, so the file is all a sweep needs:

    format: flowedit-split/1
    sources:                 # base world: the generate call its edited worlds come from
      hospital_1: {scenario: queue, seed: 0, per_type: 3, steps: 1, controls: true}
    duration: 300            # seconds recorded per run
    settings: {entrances: {rate: 0}}  # crowd settings of every run, a sweep's --settings merged over them (optional)
    seeds:                   # crowd seeds: one list for every partition, or a list per partition
      train: [1, 2, 3, 4, 5]
      held_out_edits: [6, 7, 8]
      held_out_worlds: [6, 7, 8]
    partitions:
      train: [hospital_1, hospital_1__move_fountain__6e1d6873, hospital_1__insert_fountain_inert__4a32a4fc, ...]
      held_out_edits: [...]  # other edits of the train worlds
      held_out_worlds: [...] # worlds, and their edits, with no run in train
    control_worlds:          # edit world: its geometry-only controls (`controls: true`), listed in the edit's partition
      hospital_1__move_fountain__6e1d6873: [hospital_1__insert_fountain_inert__4a32a4fc, ...]

A control's name does not say which edit it controls (an inert insert is named after its model), so `control_worlds`
holds the link, as `flowedit generate --controls` writes it in its manifest; a sweep checks it against that manifest.

Every edited world is scored against its base world run with the same seed. A partition that lists edits but not their
base world (held-out edits run at test seeds) gets the base world at its seeds as a reference run, never a training run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SPLIT_FORMAT = "flowedit-split/1"
PARTITIONS = ("train", "held_out_edits", "held_out_worlds")
SOURCE_KEYS = ("scenario", "seed", "per_type", "steps", "controls")


def base_of(world: str) -> str:
    """The base world of an edited world's name (`<base>__<edit id>`), or the name itself."""
    return world.partition("__")[0]


def crowd(doc: Mapping[str, Any], over: Path | None = None) -> dict[str, Any]:
    """The crowd settings every run of the split gets: the defaults, the split's `settings`, then the file `over`."""
    from . import settings as S

    merged = S.load(doc.get("settings") or {})
    if over is not None:
        raw = yaml.safe_load(Path(over).read_text()) or {}
        if not isinstance(raw, Mapping):
            raise S.SettingsError(f"{over}: crowd settings are a mapping")
        merged = S.load(S._merge(merged, raw))
    return merged


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
        if not isinstance(values, list) or not values or not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            raise ValueError(f"seeds of {part}: a non-empty list of crowd seeds, got {values!r}")
    if not isinstance(doc.get("settings") or {}, Mapping):
        raise ValueError("settings is a mapping of crowd settings, as a settings file holds them")
    crowd(doc)
    _check_controls(doc, seen)


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


def seeds_of(doc: Mapping[str, Any], partition: str) -> list[int]:
    """The crowd seeds `partition` runs at: its own list when `seeds` is a mapping, else the one list."""
    seeds = doc.get("seeds", [0])
    return [int(s) for s in (seeds.get(partition) or [] if isinstance(seeds, Mapping) else seeds)]


def load(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text()) or {}
    check(doc)
    return doc


@dataclass(frozen=True)
class Job:
    partition: str
    world: str
    seed: int
    #: a base world run at a partition's seeds only for its edits to be scored against, never trained on
    reference: bool = False

    @property
    def label(self) -> str:
        return f"[{self.partition}] {self.world} seed {self.seed}" + (" (reference)" if self.reference else "")

    def argv(self, scenario: str | None, out: Path, duration: float, settings: Path | None) -> list[str]:
        """`flowedit run` of this job: an edited world names its own scenario, a base world the source's."""
        args = ["run", self.world, *([scenario] if scenario and "__" not in self.world else []), "--seed", str(self.seed), "--duration", str(duration), "--out", str(out)]
        return [*args, *(["--settings", str(settings)] if settings is not None else [])]


def jobs(doc: Mapping[str, Any], partitions: Iterable[str] = PARTITIONS) -> list[Job]:
    """Every run of the chosen partitions: each world once per seed of its partition, in file order. An edited world whose
    base world runs at none of its seeds gets that base world as a reference run at them, first in the partition."""
    parts = list(partitions)
    listed = [Job(part, world, seed) for part in parts for world in (doc["partitions"].get(part) or []) for seed in seeds_of(doc, part)]
    have = {(j.world, j.seed) for j in listed}
    out: list[Job] = []
    for part in parts:
        for job in (j for j in listed if j.partition == part):
            base = base_of(job.world)
            if base != job.world and (base, job.seed) not in have:
                have.add((base, job.seed))
                out.append(Job(part, base, job.seed, reference=True))
        out.extend(j for j in listed if j.partition == part)
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
    controls `control_worlds` gives them: a name the generator no longer makes (its edits changed) would not resolve."""
    made: dict[str, list[str]] = {}
    for chain in manifest.get("chains") or []:
        made[chain["world"]] = [c["world"] for c in chain.get("controls") or []]
    controls = {c for cs in made.values() for c in cs}
    listed = {w for worlds in (doc.get("partitions") or {}).values() for w in worlds or [] if base_of(w) == base and w != base}
    if missing := sorted(listed - set(made) - controls):
        raise ValueError(f"{base}: `flowedit generate` did not make {missing}, regenerate the split's names")
    for edit, worlds in (doc.get("control_worlds") or {}).items():
        if base_of(edit) == base and (wrong := sorted(set(worlds or []) - set(made.get(edit, [])))):
            raise ValueError(f"{base}: {wrong} are no controls of {edit} in the generate manifest")


def _flowedit(args: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "arena_simulation_setup.generative.flowedit.cli", *args]


def sweep(
    doc: Mapping[str, Any],
    out: Path,
    *,
    partitions: Sequence[str] = PARTITIONS,
    settings: Path | None = None,
    jobs_at_once: int = 1,
    domain_base: int = 150,
    regenerate: bool = True,
    stagger: float = 30.0,
    log=print,  # noqa: ANN001 - a print-like callable
) -> dict[str, int]:
    """Record every run of the chosen partitions under `out`, `jobs_at_once` launches at a time, each on its own ROS domain.

    The edited worlds are generated again first, so the names in the split resolve. A finished run is skipped. Returns the
    count of runs that finished, were skipped and failed; a failed run's reason is in its launch.log. Launches start at
    least `stagger` seconds apart: runtimes coming up at the same moment on a loaded host have stalled before registering."""
    logs = out / "_sweep"
    logs.mkdir(parents=True, exist_ok=True)
    if regenerate:
        for base, source in doc["sources"].items():
            log(f"generating the edited worlds of {base}")
            manifest = logs / f"{base}.manifest.yaml"
            subprocess.run(_flowedit(generate_argv(base, source, manifest)), check=True, stdout=subprocess.DEVNULL)
            verify(doc, base, yaml.safe_load(manifest.read_text()) or {})
    from . import settings as S
    from .run import done, scenario_name

    settings_doc = crowd(doc, settings)
    if doc.get("settings") or settings is not None:
        # one file holding the merged settings, so every run reads exactly what the skip check below names its run by
        settings = logs / f"settings_{S.digest(settings_doc)}.yaml"
        settings.write_text(yaml.safe_dump(settings_doc, sort_keys=True))
    duration = float(doc.get("duration", 300.0))
    counts = {"done": 0, "skipped": 0, "failed": 0}
    queue = []
    for job in jobs(doc, partitions):
        # a finished run is skipped here, without a launch slot or a stagger
        if done(out / f"{job.world}__{scenario_name(doc['sources'][base_of(job.world)].get('scenario'), settings_doc, job.seed)}"):
            counts["skipped"] += 1
            log(f"{job.label}: already recorded")
        else:
            queue.append(job)
    running: dict[int, tuple[Job, subprocess.Popen]] = {}
    last_start = -float("inf")
    while queue or running:
        for slot in [s for s in range(jobs_at_once) if s not in running]:
            if not queue or time.monotonic() - last_start < stagger:
                break
            last_start = time.monotonic()
            job = queue.pop(0)
            env = {**os.environ, "ROS_DOMAIN_ID": str(domain_base + slot)}
            argv = job.argv(doc["sources"][base_of(job.world)].get("scenario"), out, duration, settings)
            with (logs / f"{job.world}__s{job.seed}.log").open("w") as fh:
                running[slot] = (job, subprocess.Popen(_flowedit(argv), stdout=fh, stderr=subprocess.STDOUT, env=env))
            log(f"{job.label}: started on domain {domain_base + slot}")
        time.sleep(2.0)
        for slot, (job, proc) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            text = (logs / f"{job.world}__s{job.seed}.log").read_text()
            state = "failed" if proc.returncode else "skipped" if "already recorded" in text else "done"
            counts[state] += 1
            log(f"{job.label}: {state}" + (f" ({text.strip().splitlines()[-1] if text.strip() else proc.returncode})" if state == "failed" else ""))
    return counts


__all__ = ["PARTITIONS", "SPLIT_FORMAT", "Job", "base_of", "check", "crowd", "generate_argv", "jobs", "load", "seeds_of", "sweep", "verify"]
