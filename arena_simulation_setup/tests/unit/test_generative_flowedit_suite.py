from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import run as R
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit import split as SP
from arena_simulation_setup.generative.flowedit import suite as SU
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.generative.flowedit.edit import EditType, open_site

pytestmark = pytest.mark.usefixtures("local_root")

HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"
CROWD = {"count": 2, "entrances": {"rate": 0}}
STAGE_KEYS = {"name", "map", "robot", "modes", "episodes", "seed", "timeout", "config"}


@pytest.fixture
def worlds(local_root: Path) -> Path:
    root = local_root / "worlds"
    shutil.copytree(HOSPITAL, root / "hospital_1")
    return root


@pytest.fixture
def split(worlds: Path, tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A split over hospital_1 and two of its edits saved as `flowedit generate` would, the second held out at test seeds."""
    base = open_site(worlds / "hospital_1", "queue")
    crowd = S.load(CROWD)
    train = R.prepare(base, base, [R.Edit(EditType.FUNCTION, "obj:fountain", {"dwell": 20})], crowd, 0).site.name
    held = R.prepare(base, base, [R.Edit(EditType.FUNCTION, "obj:fountain", {"dwell": 5})], crowd, 0).site.name
    doc = {
        "format": SP.SPLIT_FORMAT,
        "sources": {"hospital_1": {"scenario": "queue"}},
        "duration": 120,
        "settings": CROWD,
        "seeds": {"train": [1, 2], "held_out_edits": [6]},
        "partitions": {"train": ["hospital_1", train], "held_out_edits": [held], "held_out_worlds": []},
    }
    path = tmp_path / "pilot.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path, doc


def _suite(split: Path, worlds: Path, out: Path, *extra: str) -> dict[str, Any]:
    assert main(["suite", str(split), "--worlds", str(worlds), "--no-generate", "--out", str(out), *extra]) == 0
    text = out.read_text()
    assert text.startswith(SU.MARKER)
    return yaml.safe_load(text)


def test_stage_names_parse_back() -> None:
    assert SU.parse_stage_name(SU.stage_name("hospital_1__move_fountain__6e1d6873", 7)) == ("hospital_1__move_fountain__6e1d6873", 7)
    assert SU.parse_stage_name("hospital_1__s12") == ("hospital_1", 12)
    for bad in ("hospital_1", "hospital_1__sx", "__s3"):
        with pytest.raises(ValueError, match="no flowedit stage name"):
            SU.parse_stage_name(bad)
    with pytest.raises(ValueError, match="would not parse back"):
        SU.stage_name("odd__s4", 1)


def test_every_job_is_one_stage_and_every_edit_pairs_with_its_base_at_its_seed(split: tuple[Path, dict[str, Any]], worlds: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path, doc = split
    suite = _suite(path, worlds, tmp_path / "flowedit_pilot.yaml")
    stages = suite["stages"]
    jobs = SP.jobs(doc)
    assert [(s["map"], s["seed"]) for s in stages] == [(j.world, j.seed) for j in jobs]
    assert len({s["name"] for s in stages}) == len(stages) == 6
    assert all(set(s) == STAGE_KEYS and SU.parse_stage_name(s["name"]) == (s["map"], s["seed"]) for s in stages)
    by_name = {s["name"]: s for s in stages}
    pairs = SU.pairs(suite)
    edits = [s for s in stages if SP.base_of(s["map"]) != s["map"]]
    assert set(pairs) == {s["name"] for s in edits}
    for edit, base in pairs.items():
        assert by_name[edit]["seed"] == by_name[base]["seed"] and by_name[base]["map"] == "hospital_1"
        # the same reader settings and seed name the same scenario in both worlds: the same people
        assert by_name[edit]["config"]["scenario"] == by_name[base]["config"]["scenario"]
    # the held-out edit runs at seed 6, so the base world gets a reference stage there
    assert "hospital_1__s6" in by_name and any(j.reference and j.seed == 6 for j in jobs)
    out = capsys.readouterr().out
    assert "arena evaluation benchmark --suite flowedit_pilot" in out and "--lanes 2" in out


def test_stages_park_an_unseen_robot_and_last_the_split_s_duration(split: tuple[Path, dict[str, Any]], worlds: Path, tmp_path: Path) -> None:
    path, _ = split
    suite = _suite(path, worlds, tmp_path / "s.yaml", "--robot", "mpo700")
    assert suite["references"] is False and suite["launch"]["human.robots.visible"] == "false" and suite["launch"]["lockstep"] == "true"
    for s in suite["stages"]:
        assert s["timeout"] == 120.0 and s["episodes"] == 1 and s["robot"] == "mpo700"
        assert s["modes"] == {"robots": "stationary", "obstacles": "scenario"}
        assert s["config"]["stationary"] == {"pos_x": SU.PARK, "pos_y": SU.PARK}


def test_every_stage_s_scenario_is_written_into_its_world(split: tuple[Path, dict[str, Any]], worlds: Path, tmp_path: Path) -> None:
    path, _ = split
    suite = _suite(path, worlds, tmp_path / "s.yaml")
    for s in suite["stages"]:
        scenario = worlds / s["map"] / "scenarios" / s["config"]["scenario"]["file"] / "scenario.yaml"
        assert scenario.is_file(), scenario
        assert s["config"]["scenario"]["file"] == R.scenario_name("queue", SP.crowd(yaml.safe_load(path.read_text())), s["seed"])


def test_a_partition_and_a_dry_run(split: tuple[Path, dict[str, Any]], worlds: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path, doc = split
    out = tmp_path / "dry.yaml"
    assert main(["suite", str(path), "--worlds", str(worlds), "--no-generate", "--out", str(out), "--partition", "held_out_edits", "--dry-run"]) == 0
    assert not out.exists()
    printed = yaml.safe_load(capsys.readouterr().out)
    assert [s["name"] for s in printed["stages"]] == [SU.stage_name(j.world, j.seed) for j in SP.jobs(doc, ["held_out_edits"])]
    held = doc["partitions"]["held_out_edits"][0]
    assert not (worlds / held / "scenarios" / printed["stages"][-1]["config"]["scenario"]["file"]).exists()


def test_a_built_suite_has_one_stage_per_job_and_a_dry_build_writes_no_scenario(split: tuple[Path, dict[str, Any]], worlds: Path) -> None:
    _, doc = split
    lines: list[str] = []
    suite = SU.build(doc, SP.crowd(doc), worlds=worlds, robot="mpo700", regenerate=False, dry_run=True, log=lines.append)
    assert [(s["name"], s["robot"]) for s in suite["stages"]] == [(SU.stage_name(j.world, j.seed), "mpo700") for j in SP.jobs(doc)]
    assert not any((worlds / s["map"] / "scenarios" / s["config"]["scenario"]["file"]).exists() for s in suite["stages"])
    assert all(line.startswith("note: [") for line in lines)


def test_a_suite_flowedit_did_not_write_is_kept(split: tuple[Path, dict[str, Any]], worlds: Path, tmp_path: Path) -> None:
    path, _ = split
    out = tmp_path / "theirs.yaml"
    out.write_text("stages: []\n")
    assert main(["suite", str(path), "--worlds", str(worlds), "--no-generate", "--out", str(out)]) == 1
    assert out.read_text() == "stages: []\n"
    assert main(["suite", str(path), "--worlds", str(worlds), "--no-generate", "--out", str(out), "--force"]) == 0


def test_a_suite_without_an_edit_s_base_stage_is_refused() -> None:
    with pytest.raises(ValueError, match="no stage hospital_1__s3"):
        SU.suite_doc([SU.stage("hospital_1__function_fountain__ab", 3, "x", 60.0)])
    with pytest.raises(ValueError, match="appear twice"):
        SU.suite_doc([SU.stage("hospital_1", 1, "x", 60.0)] * 2)


def test_the_shipped_pilot_lists_one_stage_per_job() -> None:
    doc = SP.load(Path(SP.__file__).parent / "splits" / "peds_pilot.yaml")
    names = [SU.stage_name(j.world, j.seed) for j in SP.jobs(doc)]
    assert len(names) == len(set(names)) == len(SP.jobs(doc))
    assert all(SU.parse_stage_name(n)[1] in (1, 2, 3, 4, 5, 6, 7, 8) for n in names)
