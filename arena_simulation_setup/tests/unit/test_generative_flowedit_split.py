from __future__ import annotations

import copy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from arena_simulation_setup.generative.flowedit import run as R
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit import split as SP
from arena_simulation_setup.generative.flowedit.cli import main

SPLITS = Path(SP.__file__).parent / "splits"


@pytest.fixture
def doc() -> dict[str, Any]:
    return {
        "format": SP.SPLIT_FORMAT,
        "sources": {"hospital_1": {"scenario": "queue", "seed": 0, "per_type": 2}, "office_1": {"scenario": "default"}},
        "duration": 5,
        "seeds": [0, 1],
        "partitions": {"train": ["hospital_1", "hospital_1__aa"], "held_out_edits": ["hospital_1__bb"], "held_out_worlds": ["office_1"]},
    }


@pytest.mark.parametrize("path", sorted(SPLITS.glob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_splits_hold_out_what_they_say(path: Path) -> None:
    doc = SP.load(path)
    assert set(doc["partitions"]) == set(SP.PARTITIONS)
    assert all(doc["partitions"][p] for p in SP.PARTITIONS)


def test_the_pilot_runs_train_and_test_seeds_with_references_for_held_out_edits() -> None:
    doc = SP.load(SPLITS / "peds_pilot.yaml")
    runs = SP.jobs(doc)
    assert len(runs) == 155
    assert [(j.world, j.seed) for j in runs if j.reference] == [("hospital_1", 6), ("hospital_1", 7), ("hospital_1", 8)]
    assert {j.seed for j in runs if j.partition == "train"} == {1, 2, 3, 4, 5} and {j.seed for j in runs if j.partition != "train"} == {6, 7, 8}
    assert SP.crowd(doc)["entrances"]["rate"] == 0


def _mapped(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` with a seed list per partition and two held-out edits of hospital_1."""
    doc = copy.deepcopy(doc)
    doc["seeds"] = {"train": [1, 2], "held_out_edits": [6, 7], "held_out_worlds": [6]}
    doc["partitions"]["held_out_edits"].append("hospital_1__cc")
    return doc


def test_held_out_edits_get_their_base_world_at_their_seeds_as_a_reference(doc: dict[str, Any]) -> None:
    runs = SP.jobs(_mapped(doc))
    held = [j for j in runs if j.partition == "held_out_edits"]
    # the base first, once per seed however many edits it has, then the edits; office_1 is listed itself, so no reference
    assert [(j.world, j.seed, j.reference) for j in held[:2]] == [("hospital_1", 6, True), ("hospital_1", 7, True)]
    assert len(held) == 6 and not any(j.reference for j in runs if j.partition != "held_out_edits")
    assert [(j.world, j.seed) for j in runs if j.partition == "train"][:2] == [("hospital_1", 1), ("hospital_1", 2)]
    assert [j for j in SP.jobs(_mapped(doc), ["held_out_edits"]) if j.reference] == [j for j in held if j.reference]
    assert held[0].label == "[held_out_edits] hospital_1 seed 6 (reference)" and held[2].label == "[held_out_edits] hospital_1__bb seed 6"
    # one seed list for every partition: the base runs in train at the same seeds, no reference needed
    assert not any(j.reference for j in SP.jobs(doc))


def test_seeds_are_a_list_or_a_list_per_partition(doc: dict[str, Any]) -> None:
    SP.check(_mapped(doc))
    assert SP.seeds_of(_mapped(doc), "held_out_worlds") == [6] and SP.seeds_of(doc, "held_out_worlds") == [0, 1]
    for seeds, message in (
        ({"train": [1], "held_out_edits": [6]}, "seeds of held_out_worlds"),
        ({"train": [1], "held_out_edits": [6], "held_out_worlds": [6], "test": [9]}, "unknown partitions"),
        ({"train": [1], "held_out_edits": ["6"], "held_out_worlds": [6]}, "seeds of held_out_edits"),
        ([True], "seeds of every partition"),
    ):
        bad = copy.deepcopy(doc)
        bad["seeds"] = seeds
        with pytest.raises(ValueError, match=message):
            SP.check(bad)


def _controlled(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` with controls on hospital_1: hospital_1__aa (train) controlled by hospital_1__aa_inert."""
    doc = copy.deepcopy(doc)
    doc["sources"]["hospital_1"]["controls"] = True
    doc["partitions"]["train"].append("hospital_1__aa_inert")
    doc["control_worlds"] = {"hospital_1__aa": ["hospital_1__aa_inert"]}
    return doc


def test_a_control_sits_in_the_partition_of_the_edit_it_controls(doc: dict[str, Any]) -> None:
    SP.check(_controlled(doc))
    for change, message in (
        (lambda d: (d["partitions"]["train"].remove("hospital_1__aa_inert"), d["partitions"]["held_out_edits"].append("hospital_1__aa_inert")), "is in held_out_edits, its edit hospital_1__aa in train"),
        (lambda d: d["partitions"]["train"].remove("hospital_1__aa_inert"), "is in no partition, its edit"),
        (lambda d: d["sources"]["hospital_1"].update(controls=False), "has no `controls: true`"),
        (lambda d: d["sources"]["hospital_1"].update(controls="yes"), "is true or false"),
        (lambda d: d["control_worlds"].update({"hospital_1__zz": []}), "hospital_1__zz is in no partition"),
        (lambda d: d["control_worlds"].update({"hospital_1__bb": ["hospital_1__aa_inert"]}), "controls both"),
        (lambda d: d["control_worlds"].update({"hospital_1__aa": ["office_1"]}), "no edit of hospital_1__aa's base"),
    ):
        bad = _controlled(doc)
        change(bad)
        with pytest.raises(ValueError, match=message):
            SP.check(bad)


def test_a_sweep_checks_the_generator_still_makes_every_listed_name(doc: dict[str, Any], tmp_path: Path) -> None:
    controlled = _controlled(doc)
    assert SP.generate_argv("hospital_1", controlled["sources"]["hospital_1"], tmp_path / "m.yaml")[-4:] == ["--controls", "--manifest", str(tmp_path / "m.yaml"), "--write"]
    manifest = {"chains": [{"world": "hospital_1__aa", "controls": [{"kind": "inert_object", "world": "hospital_1__aa_inert"}]}, {"world": "hospital_1__bb"}]}
    SP.verify(controlled, "hospital_1", manifest)
    SP.verify(controlled, "office_1", {"chains": []})
    with pytest.raises(ValueError, match=r"did not make \['hospital_1__bb'\]"):
        SP.verify(controlled, "hospital_1", {"chains": manifest["chains"][:1]})
    with pytest.raises(ValueError, match="no controls of hospital_1__aa"):
        SP.verify(controlled, "hospital_1", {"chains": [{"world": "hospital_1__aa"}, {"world": "hospital_1__bb"}, {"world": "hospital_1__aa_inert"}]})


def test_a_splits_crowd_settings_reach_every_run_under_a_sweeps_own(doc: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    doc = copy.deepcopy(doc)
    doc["settings"] = {"entrances": {"rate": 0}}
    SP.check(doc)
    over = tmp_path / "test_crowd.yaml"
    over.write_text("count: 3\n")
    merged = SP.crowd(doc, over)
    assert merged["entrances"]["rate"] == 0 and merged["count"] == 3 and SP.crowd(doc)["count"] is None
    assert SP.crowd({})["entrances"] == S.load()["entrances"]
    for bad, message in (({"entrances": {"rte": 0}}, "unknown crowd setting"), (["rate"], "settings is a mapping")):
        with pytest.raises(ValueError, match=message):
            SP.check({**doc, "settings": bad})
    # every run reads one file with the merged settings, and a run finished under them is skipped
    seen: list[list[str]] = []
    monkeypatch.setattr(SP, "_flowedit", lambda args: (seen.append(list(args)), [sys.executable, "-c", "pass"])[1])
    finished = tmp_path / "runs" / f"hospital_1__{R.scenario_name('queue', merged, 0)}"
    finished.mkdir(parents=True)
    (finished / R.RUN_FILE).write_text("format: flowedit-run/1\n")
    (finished / R.TRACKS).write_text("")
    counts = SP.sweep(doc, tmp_path / "runs", settings=over, regenerate=False, stagger=0.0, log=lambda _: None)
    assert counts["skipped"] == 1 and counts["done"] == len(SP.jobs(doc)) - 1
    files = {a[a.index("--settings") + 1] for a in seen}
    assert len(files) == 1 and S.load(Path(files.pop())) == merged


def test_a_split_keeps_its_partitions_apart(doc: dict[str, Any]) -> None:
    SP.check(doc)
    for change, message in (
        (lambda d: d["partitions"]["held_out_edits"].append("hospital_1__aa"), "both train and held_out_edits"),
        (lambda d: d["partitions"]["held_out_worlds"].append("hospital_1__cc"), r"held-out worlds \['hospital_1'\] also run in train"),
        (lambda d: d["partitions"]["held_out_edits"].append("office_1__dd"), r"held-out edits of \['office_1'\]"),
        (lambda d: d["partitions"]["train"].append("airport"), "no sources entry"),
        (lambda d: d["partitions"].update(test=[]), "unknown partitions"),
        (lambda d: d["sources"]["office_1"].update(steps_per=2), "unknown keys"),
        (lambda d: d.update(format="x"), "format"),
        (lambda d: d.update(seeds=[]), "seeds"),
    ):
        bad = copy.deepcopy(doc)
        change(bad)
        with pytest.raises(ValueError, match=message):
            SP.check(bad)


def test_every_world_runs_once_per_seed(doc: dict[str, Any]) -> None:
    runs = SP.jobs(doc)
    assert [(j.partition, j.world, j.seed) for j in runs[:2]] == [("train", "hospital_1", 0), ("train", "hospital_1", 1)]
    assert len(runs) == 8 and len(SP.jobs(doc, ["held_out_worlds"])) == 2
    base, edited = runs[0], runs[2]
    assert base.argv("queue", Path("/o"), 5.0, None)[:3] == ["run", "hospital_1", "queue"]
    assert edited.argv("queue", Path("/o"), 5.0, Path("s.yaml"))[:2] == ["run", "hospital_1__aa"] and edited.argv("queue", Path("/o"), 5.0, Path("s.yaml"))[-2:] == ["--settings", "s.yaml"]
    assert "queue" not in edited.argv("queue", Path("/o"), 5.0, None)
    assert SP.generate_argv("hospital_1", doc["sources"]["hospital_1"]) == ["generate", "hospital_1", "--scenario", "queue", "--seed", "0", "--per-type", "2", "--steps", "1", "--write"]


def test_a_sweep_runs_every_job_and_counts_how_each_ended(doc: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a stand-in `flowedit`: office_1 fails, hospital_1__bb is already recorded, every launch prints its ROS domain
    script = "import os, sys; a = sys.argv[1:]; print('domain', os.environ['ROS_DOMAIN_ID']); w = a[1]; print('already recorded' if w == 'hospital_1__bb' else ''); sys.exit(3 if w == 'office_1' else 0)"

    def fake(args: Sequence[str]) -> list[str]:
        return [sys.executable, "-c", script, *args]

    monkeypatch.setattr(SP, "_flowedit", fake)
    # a finished run (run.yaml and tracks) is skipped without a launch
    finished = tmp_path / f"hospital_1__{R.scenario_name('queue', S.load(), 1)}"
    finished.mkdir(parents=True)
    (finished / R.RUN_FILE).write_text("format: flowedit-run/1\n")
    (finished / R.TRACKS).write_text("")
    lines: list[str] = []
    counts = SP.sweep(doc, tmp_path, jobs_at_once=2, domain_base=170, regenerate=False, stagger=0.0, log=lines.append)
    assert counts == {"done": 3, "skipped": 3, "failed": 2}
    assert not any("(reference)" in line for line in lines)
    # a reference run says so in the sweep's log
    mapped = _mapped(doc)
    lines.clear()
    SP.sweep(mapped, tmp_path / "mapped", partitions=["held_out_edits"], regenerate=False, stagger=0.0, log=lines.append)
    assert "[held_out_edits] hospital_1 seed 6 (reference): started on domain 150" in lines
    logs = sorted((tmp_path / "_sweep").iterdir())
    assert len(logs) == 7 and {line for f in logs for line in f.read_text().splitlines() if line.startswith("domain")} == {"domain 170", "domain 171"}


def test_cli_lists_a_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sweep", str(SPLITS / "peds_pilot.yaml"), "--partition", "held_out_worlds", "--dry-run"]) == 0
    rows = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
    assert rows and {r[0] for r in rows} == {"held_out_worlds"} and rows[0][1] == "office_1"
