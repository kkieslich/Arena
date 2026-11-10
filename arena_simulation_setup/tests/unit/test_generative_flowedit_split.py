from __future__ import annotations

import copy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
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
    if SP.robot_task_of(doc) is None:
        assert all(doc["partitions"][p] for p in SP.PARTITIONS)
    else:
        # a split with a robot task scores planners on its worlds, not a forecaster on held-out ones
        assert doc["partitions"]["train"] and SP.contest_of(doc)


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


def _v2(doc: dict[str, Any]) -> dict[str, Any]:
    """`doc` laid out as the v2 split: train bases at more seeds than train edits, held-out runs at test seeds, the
    held-out world's base also at baseline seeds."""
    doc = copy.deepcopy(doc)
    doc["seeds"] = {"train": {"bases": [1, 2, 3], "edits": [1]}, "held_out_edits": [6, 7], "held_out_worlds": [6]}
    doc["partitions"]["held_out_worlds"].append("office_1__ee")
    doc["baseline_seeds"] = {"held_out_worlds": [1, 2], "held_out_edits": [1, 2]}
    return doc


def test_a_partition_may_run_its_base_worlds_at_more_seeds_than_its_edits(doc: dict[str, Any]) -> None:
    v2 = _v2(doc)
    SP.check(v2)
    train = [(j.world, j.seed) for j in SP.jobs(v2) if j.partition == "train"]
    assert train == [("hospital_1", 1), ("hospital_1", 2), ("hospital_1", 3), ("hospital_1__aa", 1)]
    assert SP.seeds_of(v2, "train") == [1, 2, 3] and SP.seeds_of(v2, "train", "hospital_1__aa") == [1] and SP.seeds_of(v2, "train", "hospital_1") == [1, 2, 3]
    # an edit seed its base runs at nowhere in the partition gets the base as a reference run
    v2["seeds"]["train"] = {"bases": [1], "edits": [1, 4]}
    assert [(j.world, j.seed) for j in SP.jobs(v2, ["train"]) if j.reference] == [("hospital_1", 4)]
    for seeds, message in (({"bases": [1]}, r"exactly \['bases', 'edits'\]"), ({"bases": [1], "edits": []}, r"seeds of train \(edits\)"), ({"bases": ["1"], "edits": [1]}, r"seeds of train \(bases\)")):
        bad = copy.deepcopy(v2)
        bad["seeds"]["train"] = seeds
        with pytest.raises(ValueError, match=message):
            SP.check(bad)


def test_baseline_runs_give_held_out_bases_other_seeds_and_are_never_trained_or_scored_on(doc: dict[str, Any]) -> None:
    v2 = _v2(doc)
    runs = SP.jobs(v2)
    baseline = [(j.partition, j.world, j.seed) for j in runs if j.baseline]
    # office_1 runs at 1 and 2 for baselines only; hospital_1 already runs at 1 and 2 in train, so no second run of it
    assert baseline == [("held_out_worlds", "office_1", 1), ("held_out_worlds", "office_1", 2)]
    assert not any(j.reference for j in runs if j.baseline) and all(SP.is_base(j.world) for j in runs if j.baseline)
    assert not any(j.baseline for j in runs if j.partition == "train")
    assert SP.seeds_of(v2, "held_out_worlds") == [6] and SP.baseline_seeds_of(v2, "held_out_worlds") == [1, 2]
    assert [j.label for j in runs if j.baseline][0] == "[held_out_worlds] office_1 seed 1 (baseline only)"
    # every (world, seed, crowd) runs once, so stage names stay unique
    assert len({(j.world, j.seed) for j in runs}) == len(runs)
    # a partition written alone still carries its baseline runs, never the train runs it would repeat
    assert [(j.world, j.seed) for j in SP.jobs(v2, ["held_out_worlds"]) if j.baseline] == [("office_1", 1), ("office_1", 2)]
    assert not any(j.baseline for j in SP.jobs(v2, ["held_out_edits"]))
    for change, message in (
        (lambda d: d["baseline_seeds"].update(held_out_worlds=[6]), r"baseline_seeds of held_out_worlds: \[6\] are also seeds it is scored at"),
        (lambda d: d["baseline_seeds"].update(train=[4]), "baseline_seeds of train"),
        (lambda d: d["baseline_seeds"].update(test=[4]), "baseline_seeds of unknown partitions"),
        (lambda d: d["baseline_seeds"].update(held_out_worlds=[]), "baseline_seeds of held_out_worlds: a non-empty list"),
        (lambda d: d.update(baseline_seeds=[1]), "baseline_seeds maps a partition"),
    ):
        bad = _v2(doc)
        change(bad)
        with pytest.raises(ValueError, match=message):
            SP.check(bad)


def test_a_test_crowd_may_run_some_worlds_of_its_partitions_with_baseline_runs_of_its_own(doc: dict[str, Any], tmp_path: Path) -> None:
    v2 = _v2(doc)
    v2["partitions"]["held_out_edits"].append("hospital_1__cc")
    (tmp_path / "orca.yaml").write_text("local_planner: orca\n")
    v2["crowds"] = {"test_orca": {"settings": str(tmp_path / "orca.yaml"), "partitions": ["held_out_edits"], "worlds": ["hospital_1__cc"], "baseline_seeds": [1]}}
    SP.check(v2)
    runs = SP.jobs(v2, crowd="test_orca")
    # the chosen edit with its base's reference runs at the test seeds, then the base at the crowd's baseline seed: under
    # this crowd hospital_1 runs nowhere else, so its baseline run is not left out
    assert [(j.world, j.seed, j.reference, j.baseline) for j in runs] == [("hospital_1", 6, True, False), ("hospital_1", 7, True, False), ("hospital_1__cc", 6, False, False), ("hospital_1__cc", 7, False, False), ("hospital_1", 1, False, True)]
    assert SP.baseline_seeds_of(v2, "held_out_edits", "test_orca") == [1] and SP.baseline_seeds_of(v2, "held_out_worlds", "test_orca") == []
    for change, message in (
        (lambda c: c.update(worlds=["office_1"]), r"crowds.test_orca.worlds: \['office_1'\] are in none of its partitions"),
        (lambda c: c.update(worlds=[]), "crowds.test_orca.worlds: a non-empty list"),
        (lambda c: c.update(baseline_seeds=[6]), r"crowds.test_orca.baseline_seeds: \[6\] are also seeds"),
        (lambda c: c.update(seeds=[1]), "optionally"),
    ):
        bad = copy.deepcopy(v2)
        change(bad["crowds"]["test_orca"])
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


def test_a_base_world_is_one_whose_name_holds_no_edit() -> None:
    assert SP.is_base("hospital_1") and SP.is_base("office_1")
    assert not SP.is_base("hospital_1__move_fountain__6e1d6873")


def test_runs_last_the_splits_duration_else_the_run_default(doc: dict[str, Any]) -> None:
    assert SP.duration_of(doc) == 5.0
    assert SP.duration_of({k: v for k, v in doc.items() if k != "duration"}) == R.DURATION_S


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
    assert rows and {(r[0], r[1]) for r in rows} == {(SP.BASE_CROWD, "held_out_worlds")} and rows[0][2] == "office_1"
    # a test crowd runs on its partitions only: the held-out edits at seeds 6-8 and their base world as reference
    assert main(["sweep", str(SPLITS / "peds_pilot.yaml"), "--crowd", "test_orca", "--dry-run"]) == 0
    rows = [line.split("\t") for line in capsys.readouterr().out.splitlines()]
    assert len(rows) == 30 and {(r[0], r[1]) for r in rows} == {("test_orca", "held_out_edits")} and sum(r[-1] == "reference" for r in rows) == 3
    assert main(["sweep", str(SPLITS / "peds_pilot.yaml"), "--crowd", "all", "--dry-run"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 155 + 5 * 30
    assert main(["sweep", str(SPLITS / "peds_pilot.yaml"), "--crowd", "sfm", "--dry-run"]) == 1


def _crowded(doc: dict[str, Any], tmp_path: Path) -> Path:
    """`doc` written next to a test crowd `orca` on the held-out edits, with the split's own settings under it."""
    doc = {**copy.deepcopy(doc), "settings": {"entrances": {"rate": 0}}, "crowds": {"orca": {"settings": "crowds/orca.yaml", "partitions": ["held_out_edits"]}}}
    (tmp_path / "crowds").mkdir()
    (tmp_path / "crowds" / "orca.yaml").write_text("local_planner: orca\ncount: 3\n")
    (tmp_path / "split.yaml").write_text(yaml.safe_dump(doc))
    return tmp_path / "split.yaml"


def test_a_test_crowd_is_merged_over_the_splits_settings_and_runs_on_its_partitions(doc: dict[str, Any], tmp_path: Path) -> None:
    loaded = SP.load(_crowded(doc, tmp_path))
    assert SP.crowd_names(loaded) == [SP.BASE_CROWD, "orca"] and SP.partitions_of(loaded, "orca") == ["held_out_edits"]
    orca = SP.crowd(loaded, name="orca")
    assert (orca["local_planner"], orca["count"], orca["entrances"]["rate"]) == ("orca", 3, 0)
    assert SP.crowd(loaded)["local_planner"] is None and S.digest(orca) != S.digest(SP.crowd(loaded))
    # the base crowd keeps every partition; the test crowd gets the held-out edit and its base as reference, per seed
    assert len(SP.jobs(loaded)) == 8 and SP.jobs(loaded, ["train"], "orca") == []
    assert [(j.world, j.reference, j.crowd) for j in SP.jobs(loaded, crowd="orca")] == [("hospital_1", True, "orca")] * 2 + [("hospital_1__bb", False, "orca")] * 2
    with pytest.raises(ValueError, match="unknown crowd"):
        SP.jobs(loaded, crowd="sfm")
    (tmp_path / "crowds" / "orca.yaml").unlink()
    with pytest.raises(ValueError, match="no settings file"):
        SP.load(tmp_path / "split.yaml")


def test_a_split_names_its_crowds_well(doc: dict[str, Any]) -> None:
    for crowds, message in (
        ({"orca": {"settings": "o.yaml", "partitions": ["test"]}}, r"crowds.orca: unknown partitions"),
        ({"orca": {"settings": "o.yaml", "partitions": []}}, "names no partition"),
        ({"orca": {"settings": "o.yaml"}}, "optionally"),
        ({"orca": {"planner": "orca", "settings": "o.yaml", "partitions": ["train"]}}, "optionally"),
        ({"orca": {"settings": None, "partitions": ["train"]}}, "path of a crowd settings file"),
        ({SP.BASE_CROWD: {"settings": "o.yaml", "partitions": ["train"]}}, "no crowd name"),
        ({"Test Orca": {"settings": "o.yaml", "partitions": ["train"]}}, "no crowd name"),
        ({SP.ALL_CROWDS: {"settings": "o.yaml", "partitions": ["train"]}}, "no crowd name"),
        (["orca"], "crowds maps"),
    ):
        with pytest.raises(ValueError, match=message):
            SP.check({**doc, "crowds": crowds})


def test_a_sweep_records_each_crowd_under_its_own_settings(doc: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = SP.load(_crowded(doc, tmp_path))
    seen: list[list[str]] = []
    monkeypatch.setattr(SP, "_flowedit", lambda args: (seen.append(list(args)), [sys.executable, "-c", "pass"])[1])
    counts = SP.sweep(loaded, tmp_path / "runs", crowds=[SP.BASE_CROWD, "orca"], regenerate=False, stagger=0.0, log=lambda _: None)
    assert counts == {"done": 12, "skipped": 0, "failed": 0}
    files = {a[a.index("--settings") + 1] for a in seen}
    assert {S.digest(S.load(Path(f))) for f in files} == {S.digest(SP.crowd(loaded)), S.digest(SP.crowd(loaded, name="orca"))}
    assert len(list((tmp_path / "runs" / "_sweep").glob("orca__*.log"))) == 4


@pytest.mark.parametrize("name", [n for n in SP.crowd_names(SP.load(SPLITS / "peds_pilot.yaml")) if n != SP.BASE_CROWD])
def test_every_test_crowd_of_the_pilot_differs_from_its_base_crowd(name: str) -> None:
    doc = SP.load(SPLITS / "peds_pilot.yaml")
    base, test = SP.crowd(doc), SP.crowd(doc, name=name)
    changed = {
        "rates": S.need(test, "thirst")["decay_rate"] != S.need(base, "thirst")["decay_rate"],
        "size": (test["scale"], test["count"]) != (base["scale"], base["count"]),
        "planner": (test["local_planner"] or "sfm") != (base["local_planner"] or "sfm"),
    }
    assert any(changed.values()) and test["entrances"] == base["entrances"]
    if changed["rates"]:
        # the test draws start above where nearly every base draw ends (mean + 3 std)
        base_rate = S.need(base, "thirst")["decay_rate"]
        assert S.need(test, "thirst")["decay_rate"]["clip_low"] > base_rate["mean"] + base_rate["std"]


def test_data_v2_holds_out_what_the_design_says_with_entrances_on() -> None:
    v2, pilot = SP.load(SPLITS / "peds_v2.yaml"), SP.load(SPLITS / "peds_pilot.yaml")
    parts = v2["partitions"]
    further = {"food_court", "hallway_lab", "hallway_office", "hallway_school", "office_open_plan", "waiting_hall_clinic"}
    assert {SP.base_of(w) for w in parts["train"]} == {"hospital_1", "office_1", *further}
    assert {w for w in parts["held_out_worlds"] if SP.is_base(w)} == {"reception", "hallway_clinic", "library", "station_hall"}
    assert all(v2["sources"][w]["per_type"] == 5 for w in further) and v2["sources"]["hospital_1"]["per_type"] == 10
    assert SP.seeds_of(v2, "train", "hospital_1") == [1, 2, 3, 4, 5] and SP.seeds_of(v2, "train", "hospital_1__x") == [1, 2]
    assert SP.seeds_of(v2, "held_out_edits") == SP.seeds_of(v2, "held_out_worlds") == [6, 7, 8]
    assert SP.crowd(v2)["entrances"]["rate"] > 0
    # the pilot (the development set) ran the first chains: every one of them is a train edit in v2, none a test edit
    pilot_edits = {w for p in pilot["partitions"].values() for w in p if not SP.is_base(w)}
    assert pilot_edits <= set(parts["train"])
    # 20 % of each train world's chains held out: 2 of 10 per edit type, 1 of 5 on the further worlds
    controls = {c for cs in v2["control_worlds"].values() for c in cs}
    held = {base: [w for w in parts["held_out_edits"] if SP.base_of(w) == base and w not in controls] for base in ("hospital_1", "office_1", *further)}
    assert len(held["hospital_1"]) == len(held["office_1"]) == 12
    # one chain per edit type the generator makes there: five, six where the plan has a lockable spare door
    assert all(len(held[base]) in (5, 6) for base in further)
    runs = SP.jobs(v2)
    assert len(runs) == 1243 and len({(j.world, j.seed) for j in runs}) == 1243
    baseline = sorted((j.world, j.seed) for j in runs if j.baseline)
    assert baseline == [(w, s) for w in ("hallway_clinic", "library", "reception", "station_hall") for s in (1, 2, 3, 4, 5)]
    assert all(j.partition == "held_out_worlds" for j in runs if j.baseline)
    for crowd in ("test_rates", "test_size", "test_orca", "test_hsfm", "test_all"):
        crowd_runs = SP.jobs(v2, crowd=crowd)
        assert len(crowd_runs) == 66 and {j.seed for j in crowd_runs if not j.baseline} == {6, 7, 8}
        assert sorted({j.world for j in crowd_runs if j.baseline}) == ["hospital_1", "office_1"]
