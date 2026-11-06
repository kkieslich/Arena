from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import run as R
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.generative.flowedit.edit import EditType, open_site

pd = pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("arena_humansim")
pytestmark = pytest.mark.usefixtures("local_root")

HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"
DWELL = R.Edit(EditType.FUNCTION, "obj:fountain", {"dwell": 20})
FEW = S.load(count=2)


@pytest.fixture
def worlds(local_root: Path) -> Path:
    root = local_root / "worlds"
    shutil.copytree(HOSPITAL, root / "hospital_1")
    return root


def _types(scenario: Path) -> dict[str, dict]:
    return {p.name: yaml.safe_load(p.read_text()) for p in sorted((scenario / "types").glob("flow__*.yaml"))}


def test_the_scenario_is_named_after_the_settings_and_seed() -> None:
    assert R.scenario_name("queue", FEW, 0) == R.scenario_name("queue", S.load(count=2), 0)
    assert R.scenario_name("queue", FEW, 0) != R.scenario_name("queue", FEW, 1)
    assert R.scenario_name("queue", FEW, 0) != R.scenario_name("queue", S.load(count=3), 0)
    assert R.scenario_name(None, FEW, 0).startswith("flowedit_world_")


def test_the_launch_records_humansim_into_the_run() -> None:
    command = R.launch_command("hospital_1__ab", "flowedit_queue_x_s0", Path("/runs/r/humansim"), ["robot:=jackal"])
    assert command[:2] == ["arena", "launch"]
    assert {"world:=hospital_1__ab", "task.scenario.file:=flowedit_queue_x_s0", "human:=arena", "humansim.record:=true", "humansim.record_dir:=/runs/r/humansim"} <= set(command)
    assert command[-1] == "robot:=jackal"


def test_an_edit_is_saved_and_its_scenario_has_the_people_of_the_base(worlds: Path) -> None:
    base = open_site(worlds / "hospital_1", "queue")
    plain = R.prepare(base, base, [], FEW, 0)
    edited = R.prepare(base, base, [DWELL], FEW, 0)
    assert plain.site.root == worlds / "hospital_1" and edited.site.root.parent == worlds
    assert edited.site.name.startswith("hospital_1__") and edited.run_id == f"{edited.site.name}__{edited.scenario}"
    assert edited.scenario == plain.scenario
    before, after = _types(plain.site.root / "scenarios" / plain.scenario), _types(edited.site.root / "scenarios" / edited.scenario)
    assert before.keys() == after.keys() and before != after
    for name in before:
        assert before[name]["sequences"]["idle"] == after[name]["sequences"]["idle"]
    assert any(t["sequences"].get("seek_thirst", {}).get("steps", {}).get("use", {}).get("duration") == {"mean": 20.0} for t in after.values())


def test_a_dry_run_saves_and_writes_nothing(worlds: Path) -> None:
    base = open_site(worlds / "hospital_1", "queue")
    prepared = R.prepare(base, base, [DWELL], FEW, 0, dry_run=True)
    assert prepared.site.edits and sorted(p.name for p in worlds.iterdir()) == ["hospital_1"]
    assert not (worlds / "hospital_1" / "scenarios" / prepared.scenario).exists()


def test_tracks_pair_released_people_by_source_and_placed_people_by_name() -> None:
    df = pd.DataFrame(
        {
            "time": [10.0, 10.0, 10.0, 10.0, 75.0],
            "agent_id": [1, 2, 3, -1, 1],
            "x": [0.0] * 5,
            "y": [0.0] * 5,
            "source_name": ["", "door_a", "", "", ""],
            "source_seq": [0, 4, 0, 0, 0],
        }
    )
    out = R.keyed(df, {1: "env_1/flow_1", 2: "env_1/door_a_4"}, duration=60.0)
    assert list(out["key"]) == ["flow_1", "door_a#4", "id3", "id-1"]
    assert list(out["time"]) == [0.0] * 4
    assert R.summary(out) == {"frames": 1, "agents": 3, "released": 1, "seconds": 0.0}
    assert R.keyed(df.iloc[:0], {}).empty


def test_a_reused_agent_id_is_named_after_the_agent_under_it_at_that_frame() -> None:
    # HumanSim hands a left agent's id to the next one: a whole-run name map would give flow_1 the visitor's name
    df = pd.DataFrame({"time": [10.0, 60.0, 61.0], "agent_id": [1, 1, 2], "x": [0.0] * 3, "y": [0.0] * 3, "source_name": ["", "door_a", ""], "source_seq": [0, 4, 0]})
    # agent 2 joins after the snapshot at 50 s: the first later one names it
    names = [(0.0, {1: "env_1/flow_1"}), (50.0, {1: "env_1/door_a_4"}), (70.0, {2: "env_1/flow_2"})]
    out = R.keyed(df, names)
    assert list(out["name"]) == ["flow_1", "door_a_4", "flow_2"]
    assert list(out["key"]) == ["flow_1", "door_a#4", "flow_2"]
    assert list(R.names_at([5.0], [1], [(10.0, {1: "a"})])) == ["a"] and list(R.names_at([5.0], [3], names)) == [""]
    # a snapshot listing an agent unnamed is the agent's name at that time, not a gap to fill
    assert list(R.names_at([60.0], [2], [(50.0, {2: ""}), (70.0, {2: "flow_2"})])) == [""]


def _interactions(live: list[tuple[int, int, str, bool, list[int], list[int]]], events: list[tuple[int, int, int, list[int]]] = ()) -> SimpleNamespace:
    """An InteractionFrame: per live interaction (id, type, object, holding, members, roles), and edges (id, type, event, participants)."""
    offsets = [0]
    for *_, members, _ in live:
        offsets.append(offsets[-1] + len(members))
    return SimpleNamespace(
        interaction_id=[i[0] for i in live],
        interaction_type=[i[1] for i in live],
        outcome=[1] * len(live),
        object_id=[i[2] for i in live],
        arrived=[i[3] for i in live],
        holding=[i[3] for i in live],
        duration=[8.0] * len(live),
        member_offset=offsets,
        member_id=[m for i in live for m in i[4]],
        member_role=[r for i in live for r in i[5]],
        member_seat=[-1 for i in live for _ in i[4]],
        events=[SimpleNamespace(interaction_id=e[0], interaction_type=e[1], event=e[2], participants=e[3]) for e in events],
    )


def test_interactions_are_keyed_like_the_tracks_on_their_time_base() -> None:
    tracks = R.keyed(pd.DataFrame({"time": [100.0, 100.0, 100.0], "agent_id": [1, 2, 3], "x": [0.0] * 3, "y": [0.0] * 3, "source_name": ["", "door_a", ""], "source_seq": [0, 4, 0]}), {1: "flow_1", 3: "flow_3"})
    frames = [
        # latched before the first agent frame: what is live at the start, its edges already past
        (98.0, _interactions([(7, 6, "fountain", False, [1], [0])], [(7, 6, 0, [1])])),
        (99.0, _interactions([(7, 6, "fountain", True, [1, 2], [0, 1])], [(7, 6, 1, [1])])),
        (104.0, _interactions([(7, 6, "fountain", True, [2, 3], [0, 1]), (9, 0, "", False, [3], [0])], [(7, 6, 2, [1])])),
        (200.0, _interactions([(7, 6, "fountain", True, [3], [0])])),
    ]
    rows = R.interaction_rows(frames, 100.0, tracks, duration=60.0)
    assert set(rows["time"]) == {0.0, 4.0}
    start = rows[rows["time"] == 0.0]
    assert list(start["record"]) == ["member", "member"] and list(start["key"]) == ["flow_1", "door_a#4"] and list(start["role"]) == ["participant", "queued"]
    assert list(start["type"]) == ["queue_use", "queue_use"] and list(start["position"]) == [0, 1]
    released = rows[rows["record"] == "event"]
    assert list(released["event"]) == ["released"] and list(released["object_id"]) == ["fountain"] and list(released["key"]) == ["flow_1"]
    talk = rows[rows["interaction_id"] == 9]
    assert list(talk["type"]) == ["talk_to"] and list(talk["object_id"]) == [""]
    assert R.uses(rows) == {"fountain": {"uses": 1, "users": 2, "queue_max": 1}}
    empty = R.interaction_rows([], 0.0)
    assert empty.empty and list(empty.columns) == list(R.INTERACTION_COLUMNS) and R.uses(empty) == {}


def test_provenance_names_each_checkout_and_whether_it_was_changed(tmp_path: Path) -> None:
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_text("a")
    git("add", "a.txt")
    git("commit", "-qm", "a")
    head, dirty = R._commit(tmp_path)
    assert len(head or "") == 40 and dirty is False
    # an untracked file, a generated scenario among them, leaves a checkout clean; a changed tracked file does not
    (tmp_path / "b.txt").write_text("b")
    assert R._commit(tmp_path) == (head, False)
    (tmp_path / "a.txt").write_text("changed")
    assert R._commit(tmp_path) == (head, True)
    assert R._commit(tmp_path / "missing") == (None, None)
    assert {"code", "code_dirty", "humansim", "humansim_dirty"} <= set(R.provenance())


def _fake_arena(bin_dir: Path, behaviour: str) -> None:
    """An `arena` on PATH that opens HumanSim's bag dir like a launch, then `behaviour`."""
    script = bin_dir / "arena"
    script.write_text(f"#!/bin/bash\nfor a in \"$@\"; do case $a in humansim.record_dir:=*) d=${{a#humansim.record_dir:=}};; esac; done\necho \"$@\" > \"$(dirname \"$d\")/args\"\n{behaviour}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def fake_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return bin_dir


def _frames(bag: Path, duration: float | None = None) -> R.Extracted:
    df = R.keyed(pd.DataFrame({"time": [0.0, 1.0], "agent_id": [1, 1], "x": [0.0, 1.0], "y": [0.0, 0.0], "source_name": ["", ""], "source_seq": [0, 0]}), {1: "flow_1"}, duration)
    return R.Extracted(df, R.interaction_rows([(0.5, _interactions([(7, 6, "fountain", True, [1], [0])]))], 0.0, df, duration))


def test_a_run_records_until_the_duration_and_stops_the_launch(worlds: Path, tmp_path: Path, fake_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_arena(fake_bin, 'mkdir -p "$d/bag"; exec sleep 600')
    monkeypatch.setattr(R, "extract", _frames)
    base = open_site(worlds / "hospital_1", "queue")
    prepared = R.prepare(base, base, [DWELL], FEW, 3)
    started = time.monotonic()
    run_dir = R.record(prepared, tmp_path / "runs", 1.0, launch_args=["robot:=jackal"], margin=0.0, grace=5.0, log=lambda _: None)
    assert time.monotonic() - started < 20
    assert R.done(run_dir) and run_dir.name == prepared.run_id
    doc = yaml.safe_load((run_dir / R.RUN_FILE).read_text())
    assert doc["world"] == prepared.site.name and doc["base"] == "hospital_1" and doc["seed"] == 3
    assert [e["id"] for e in doc["edits"]] == [e.id for e in prepared.site.edits]
    assert doc["settings_digest"] == S.digest(FEW) and doc["tracks"]["agents"] == 1
    assert "robot:=jackal" in (run_dir / "args").read_text()
    assert len(pd.read_parquet(run_dir / R.TRACKS)) == 2
    assert list(pd.read_parquet(run_dir / R.INTERACTIONS)["key"]) == ["flow_1"]
    assert doc["uses"] == {"fountain": {"uses": 1, "users": 1, "queue_max": 0}}
    assert {"code", "code_dirty", "humansim", "humansim_dirty"} <= set(doc)


def test_a_launch_that_dies_is_reported_and_leaves_no_finished_run(worlds: Path, tmp_path: Path, fake_bin: Path) -> None:
    _fake_arena(fake_bin, "echo boom; exit 3")
    base = open_site(worlds / "hospital_1", "queue")
    prepared = R.prepare(base, base, [], FEW, 0)
    with pytest.raises(RuntimeError, match=r"exited \(3\)"):
        R.record(prepared, tmp_path / "runs", 1.0, margin=0.0, log=lambda _: None)
    run_dir = tmp_path / "runs" / prepared.run_id
    assert not R.done(run_dir) and "boom" in (run_dir / "launch.log").read_text()


def test_a_run_stops_when_a_vital_node_dies(worlds: Path, tmp_path: Path, fake_bin: Path) -> None:
    # the launch keeps running when the env node dies, so the run would otherwise wait out its whole duration
    _fake_arena(fake_bin, 'mkdir -p "$d/bag"; sleep 3; echo "[ERROR] [arena_env_node-5]: process has died [pid 1, exit code 1]"; exec sleep 600')
    base = open_site(worlds / "hospital_1", "queue")
    prepared = R.prepare(base, base, [], FEW, 0)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="arena_env_node died while recording"):
        R.record(prepared, tmp_path / "runs", 300.0, grace=5.0, log=lambda _: None)
    assert time.monotonic() - started < 30


def test_cli_dry_run_prints_the_launch(worlds: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", "hospital_1", "queue", "--worlds", str(worlds), "--count", "2", "--edit", "function", "obj:fountain", "-p", "dwell=20", "--out", str(tmp_path)]
    assert main([*args, "--dry-run"]) == 0
    doc = yaml.safe_load(capsys.readouterr().out)
    assert doc["base"] == "hospital_1" and doc["edits"][0]["params"] == {"dwell": 20}
    assert f"task.scenario.file:={doc['scenario']}" in doc["launch"] and doc["run"].startswith(str(tmp_path))
    assert main(["run", "hospital_1", "queue", "--worlds", str(worlds), "--edit", "spin", "obj:fountain", "--dry-run"]) == 1
    assert main(["run", "hospital_1", "queue", "--worlds", str(worlds), "-p", "dwell=20", "--dry-run"]) == 1


def test_cli_skips_a_finished_run(worlds: Path, tmp_path: Path, fake_bin: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _fake_arena(fake_bin, "echo launched >> $(dirname $d)/count; exit 9")
    args = ["run", "hospital_1", "queue", "--worlds", str(worlds), "--count", "2", "--out", str(tmp_path), "--duration", "1"]
    assert main(args) == 1
    run_dir = next((tmp_path).iterdir())
    (run_dir / R.RUN_FILE).write_text("format: flowedit-run/1\n")
    (run_dir / R.TRACKS).write_text("")
    capsys.readouterr()
    assert main(args) == 0 and "already recorded" in capsys.readouterr().out
    assert (run_dir / "count").read_text().count("launched") == 1
