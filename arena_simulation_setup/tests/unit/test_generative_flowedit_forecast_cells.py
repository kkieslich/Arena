"""Flow cells from several run dirs read as one (`--cells` and `--benchmark`, repeatable): one split recorded on two machines,
on one Arena and humansim commit."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
shapely = pytest.importorskip("shapely")
pl = pytest.importorskip("polars")
yaml = pytest.importorskip("yaml")

from arena_simulation_setup.generative.flowedit.forecast import baselines as B  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast import maps as M  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast.evaluate import Runs, crowds_of, read_cells  # noqa: E402

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
GRID = M.FlowGrid(x0=0.0, y0=0.0, nx=4, ny=3)
CROWD = "eed0f29c"


def _cells(world: str, seed: int, scale: float = 1.0, cell: float = 0.5, headings: int = M.HEADINGS, planner: str = "peds_only") -> pl.DataFrame:
    """A run's flow cells as arena_evaluation writes them: a few cells, a moving bin and the standing bin each."""
    rows = []
    for i in range(GRID.ny):
        for j in range(GRID.nx):
            b = (i + j + seed) % headings
            rows.append((cell * (j + 0.5), cell * (i + 0.5), -math.pi + (b + 0.5) * 2 * math.pi / headings, scale * (1 + i + seed * j)))
            rows.append((cell * (j + 0.5), cell * (i + 0.5), None, scale * (0.5 + seed)))
    x, y, heading, seconds = zip(*rows, strict=True)
    n = len(rows)
    return pl.DataFrame(
        {
            "planner": [planner] * n,
            "stage": [f"{world}__s{seed}"] * n,
            "flow_world": [world] * n,
            "flow_base": [world.split("__")[0]] * n,
            "flow_seed": [seed] * n,
            "x": list(x),
            "y": list(y),
            "cell_m": [cell] * n,
            "heading_rad": pl.Series(heading, dtype=pl.Float64),
            "ped_seconds": list(seconds),
        }
    )


def _commits(arena: str = "a" * 40, humansim: str = "b" * 40, dirty: bool = False) -> dict:
    """The `commits` of arena_evaluation's run manifest."""
    return {"arena": {"path": "/opt/arena_ws/src/Arena", "commit": arena, "dirty": dirty}, "humansim": {"path": "/opt/arena_ws/src/Arena/humansim", "commit": humansim, "dirty": False, "pinned": humansim}, "arena_evaluation": {"path": "/opt/arena_ws/src/Arena/arena_evaluation", "commit": "c" * 40, "dirty": False, "pinned": "c" * 40}}


def _run_dir(root: Path, name: str, frames: list[pl.DataFrame], crowd: str | None = None, commits: dict | None = None) -> Path:
    """A processed run dir: flow_cells.parquet and, with `crowd` or `commits`, the manifest naming each stage's flowedit
    scenario file and the commits the run ran."""
    run = root / name
    run.mkdir(parents=True)
    cells = pl.concat(frames, how="diagonal")
    cells.write_parquet(run / "flow_cells.parquet")
    if crowd is not None or commits is not None:
        stages = [{"name": s, "config": {"scenario": {"file": f"flowedit_demand_{crowd or CROWD}_s{s.rpartition('__s')[2]}"}}} for s in cells["stage"].unique(maintain_order=True)]
        (run / "manifest.yaml").write_text(yaml.safe_dump({"suite_name": "flowedit_peds_v2", "suite": {"stages": stages}, **({} if commits is None else {"commits": commits})}))
    return run / "flow_cells.parquet"


def _maps(runs: Runs) -> dict:
    return {k: v.tolist() for k, v in sorted(runs.maps.items())}


def test_heading_bins_are_read_from_the_bin_centres() -> None:
    centres = [M.bin_centre(b) for b in range(M.HEADINGS)]
    assert M.heading_bins(centres) == 8
    assert M.heading_bins(centres[2:5]) == 8
    assert M.heading_bins([-math.pi + (b + 0.5) * 2 * math.pi / 16 for b in range(0, 16, 2)]) == 16
    assert M.heading_bins([]) is None
    with pytest.raises(ValueError, match="heading bins"):
        M.heading_bins([0.123456789])


def test_two_run_dirs_read_as_the_one_combined_file(tmp_path: Path) -> None:
    runs = {(w, s): _cells(w, s) for w in ("hospital_1", "hospital_1__move_fountain__ab") for s in (1, 2, 3)}
    home = _run_dir(tmp_path, "home", [runs[k] for k in runs if k[1] < 3], CROWD)
    institute = _run_dir(tmp_path, "institute", [runs[k] for k in runs if k[1] == 3], CROWD)
    combined = _run_dir(tmp_path, "combined", list(runs.values()))
    grid_of = lambda base: GRID  # noqa: E731
    assert _maps(Runs.load([home, institute], grid_of)) == _maps(Runs.load(combined, grid_of)) == _maps(Runs.load([institute, home], grid_of))
    assert set(Runs.load([home, institute], grid_of).maps) == set(runs)


def test_one_file_reads_as_before(tmp_path: Path) -> None:
    path = _run_dir(tmp_path, "one", [_cells("hospital_1", 1), _cells("hospital_1", 1, planner="dwb").with_columns(pl.lit("extra").alias("note"))])
    assert read_cells(path, "dwb").equals(pl.read_parquet(path).filter(pl.col("planner") == "dwb"))
    assert read_cells([path], "dwb").columns == pl.read_parquet(path).columns
    with pytest.raises(ValueError, match=r"contestants \['dwb', 'peds_only'\], pick one with --planner"):
        read_cells(path)


def test_a_stage_in_two_files_is_read_once_when_its_rows_agree(tmp_path: Path) -> None:
    a = _run_dir(tmp_path, "a", [_cells("hospital_1", 1), _cells("hospital_1", 2)], CROWD)
    # the same run's cells, rows in another order (a run dir processed again)
    b = _run_dir(tmp_path, "b", [_cells("hospital_1", 2).reverse(), _cells("hospital_1", 3)], CROWD)
    cells = read_cells([a, b])
    assert cells.height == 3 * _cells("hospital_1", 1).height
    one = Runs.load(_run_dir(tmp_path, "one", [_cells("hospital_1", s) for s in (1, 2, 3)]), lambda base: GRID)
    assert _maps(Runs.load([a, b], lambda base: GRID)) == _maps(one)
    assert read_cells([a, a]).height == read_cells(a).height


def test_a_stage_in_two_files_with_other_rows_fails(tmp_path: Path) -> None:
    a = _run_dir(tmp_path, "a", [_cells("hospital_1", 1), _cells("hospital_1", 2)])
    b = _run_dir(tmp_path, "b", [_cells("hospital_1", 2, scale=1.5)])
    with pytest.raises(ValueError, match=r"stage hospital_1__s2 \(peds_only\) is in .*a/flow_cells.parquet and in .*b/flow_cells.parquet with different flow cells"):
        read_cells([a, b])
    # another contestant at that stage is another run, not a clash
    c = _run_dir(tmp_path, "c", [_cells("hospital_1", 2, scale=1.5, planner="dwb")])
    assert read_cells([a, c], "dwb").height == _cells("hospital_1", 2).height


def test_files_on_other_flow_grids_fail(tmp_path: Path) -> None:
    a = _run_dir(tmp_path, "a", [_cells("hospital_1", 1)])
    coarse = _run_dir(tmp_path, "coarse", [_cells("hospital_1", 2, cell=1.0)])
    with pytest.raises(ValueError, match=r"different grids, cell size: .*a/flow_cells.parquet \(0.5,\), .*coarse/flow_cells.parquet \(1.0,\)"):
        read_cells([a, coarse])
    fine = _run_dir(tmp_path, "fine", [_cells("hospital_1", 2, headings=16)])
    with pytest.raises(ValueError, match="different grids, heading bins: .* 8, .* 16"):
        read_cells([a, fine])


def test_files_of_other_crowds_fail_where_the_manifest_says(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "base", [_cells("hospital_1", 1)], CROWD)
    test_crowd = _run_dir(tmp_path, "test_orca", [_cells("hospital_1__move_fountain__ab", 6)], "0123abcd")
    assert crowds_of(base, ["hospital_1__s1"]) == {(CROWD, None)}
    with pytest.raises(ValueError, match=f"different crowds: .*base/flow_cells.parquet crowd settings {CROWD}; .*test_orca/flow_cells.parquet crowd settings 0123abcd"):
        read_cells([base, test_crowd])
    # a run with the robot task is another crowd too (H2)
    h2 = tmp_path / "h2"
    h2.mkdir()
    _cells("hospital_1", 7).write_parquet(h2 / "flow_cells.parquet")
    (h2 / "manifest.yaml").write_text(yaml.safe_dump({"suite": {"stages": [{"name": "hospital_1__s7", "config": {"scenario": {"file": f"flowedit_queue_{CROWD}_r89abcdef_s7"}}}]}}))
    with pytest.raises(ValueError, match="with robot task 89abcdef"):
        read_cells([base, h2 / "flow_cells.parquet"])
    # without a manifest there is nothing to compare
    bare = _run_dir(tmp_path, "bare", [_cells("hospital_1__move_fountain__ab", 6)])
    assert read_cells([base, bare]).height == 2 * _cells("hospital_1", 1).height


def test_the_track_store_reads_several_run_dirs(tmp_path: Path) -> None:
    for name, stage, x in (("home", "w__e__s1", 1.0), ("institute", "w__e__s6", 6.0), ("again", "w__e__s1", 9.0)):
        run = tmp_path / name
        run.mkdir()
        pl.DataFrame({"episode": [0], "stage": [stage], "planner": ["peds_only"], "robot": ["env_0_jackal"]}).write_parquet(run / "combined_metrics.parquet")
        derived = run / "episodes" / "episode_000" / "topics" / ".derived" / "ped_tracks.v2"
        derived.mkdir(parents=True)
        pl.DataFrame({"key": ["a"], "x": [x]}).write_parquet(derived / "env_0.parquet")
    tracks = B.Tracks.load([tmp_path / "home", tmp_path / "institute", tmp_path / "again"])
    assert tracks("w__e", 1)["x"].to_list() == [1.0] and tracks("w__e", 6)["x"].to_list() == [6.0]
    assert B.Tracks.load(tmp_path / "institute")("w__e", 6)["x"].to_list() == [6.0]


def test_run_dirs_of_other_arena_or_humansim_commits_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = _run_dir(tmp_path, "home", [_cells("hospital_1", 1)], commits=_commits())
    same = _run_dir(tmp_path, "same", [_cells("hospital_1", 2)], commits=_commits())
    assert read_cells([home, same]).height == 2 * _cells("hospital_1", 1).height
    assert "warning" not in capsys.readouterr().err
    humansim = _run_dir(tmp_path, "humansim", [_cells("hospital_1", 3)], commits=_commits(humansim="d" * 40))
    with pytest.raises(ValueError, match=f"different humansim commits: .*home {'b' * 12}, .*humansim {'d' * 12}; read only runs of one Arena release together"):
        read_cells([home, humansim])
    arena = _run_dir(tmp_path, "arena", [_cells("hospital_1", 3)], commits=_commits(arena="e" * 40))
    with pytest.raises(ValueError, match="different arena commits"):
        read_cells([home, arena])
    # one file is read as before, its commits not looked at
    assert read_cells(humansim).height == _cells("hospital_1", 3).height


def test_run_dirs_without_commits_are_read_with_a_warning(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    home = _run_dir(tmp_path, "home", [_cells("hospital_1", 1)], commits=_commits())
    older = _run_dir(tmp_path, "older", [_cells("hospital_1", 2)], CROWD)
    bare = _run_dir(tmp_path, "bare", [_cells("hospital_1", 3)])
    unread = _run_dir(tmp_path, "unread", [_cells("hospital_1", 4)], commits={**_commits(), "humansim": {"path": None, "commit": None, "dirty": None, "reason": "module arena_humansim not installed"}})
    dirty = _run_dir(tmp_path, "dirty", [_cells("hospital_1", 5)], commits=_commits(dirty=True))
    assert read_cells([home, older, bare, unread, dirty]).height == 5 * _cells("hospital_1", 1).height
    err = capsys.readouterr().err
    assert f"warning: {older.parent}: its manifest.yaml records no commits (written before arena_evaluation recorded them), so its Arena and humansim commits are not compared" in err
    assert f"warning: {bare.parent}: no manifest.yaml" in err
    assert f"warning: {unread.parent}: no humansim commit recorded (module arena_humansim not installed)" in err
    assert f"warning: {dirty.parent}: its arena checkout (/opt/arena_ws/src/Arena) was dirty at {'a' * 12}" in err
    assert f"{home.parent}:" not in err


def test_the_track_store_refuses_run_dirs_of_other_commits(tmp_path: Path) -> None:
    for name, humansim in (("home", "b" * 40), ("institute", "d" * 40)):
        run = tmp_path / name
        run.mkdir()
        pl.DataFrame({"episode": [0], "stage": ["w__e__s1"], "planner": ["peds_only"], "robot": ["env_0_jackal"]}).write_parquet(run / "combined_metrics.parquet")
        (run / "manifest.yaml").write_text(yaml.safe_dump({"commits": _commits(humansim=humansim)}))
    with pytest.raises(ValueError, match="different humansim commits"):
        B.Tracks.load([tmp_path / "home", tmp_path / "institute"])
    assert B.Tracks.load(tmp_path / "institute").where


def _actions(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], argparse.Action]]:
    out = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out += _actions(sub, (*path, name))
        else:
            out.append((path, action))
    return out


def test_every_forecast_verb_takes_cells_and_benchmark_repeatedly() -> None:
    from arena_simulation_setup.generative.flowedit.cli import _parser

    found = [(path, a) for path, a in _actions(_parser()) if path[:1] == ("forecast",) and a.dest in ("cells", "benchmark")]
    verbs = {" ".join(path[1:]) for path, _ in found}
    assert {"fit", "select", "train", "score", "scaling", "curve", "strength", "baseline cliff", "baseline controls", "baseline geometry", "baseline deepflow tune", "baseline deepflow train", "baseline deepflow predict"} <= verbs
    assert all(isinstance(a, argparse._AppendAction) for _, a in found), [" ".join(p) for p, a in found if not isinstance(a, argparse._AppendAction)]


# --- the fit and the score on hospital_1 ------------------------------------------------------------------------------


@pytest.mark.usefixtures("local_root")
def test_fit_and_score_on_two_files_of_one_split_are_those_on_the_combined_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("arena_humansim")
    from arena_simulation_setup.generative.flowedit.cli import main

    monkeypatch.setenv("ARENA_WORLD_PATH", str(WORLDS))
    forecast = tmp_path / "forecast.parquet"
    assert main(["forecast", "predict", "hospital_1", "--scenario", "queue", "--duration", "60", "--out", str(forecast)]) == 0
    rows = pl.read_parquet(forecast)
    # recordings of three seeds, each the forecast with more weight on one side; two seeds' runs on one machine, the third's on the other, one run on both
    median = rows["x"].median()
    seeds = {s: rows.with_columns((pl.col("ped_seconds") * pl.when(pl.col("x") > median).then(1 + 0.2 * s).otherwise(1.0)).alias("ped_seconds"), pl.lit("peds_only").alias("planner"), pl.lit(f"hospital_1__s{s}").alias("stage"), pl.lit("hospital_1").alias("flow_base"), pl.lit(s).alias("flow_seed")) for s in (1, 2, 3)}
    split = {"format": "flowedit-split/1", "sources": {"hospital_1": {"scenario": "queue", "seed": 0, "per_type": 1, "steps": 1}}, "duration": 60, "seeds": [1, 2, 3], "partitions": {"train": ["hospital_1"], "held_out_edits": [], "held_out_worlds": []}}
    (tmp_path / "split.yaml").write_text(yaml.safe_dump(split))
    home = _run_dir(tmp_path, "home", [seeds[1], seeds[2]], CROWD, _commits())
    institute = _run_dir(tmp_path, "institute", [seeds[3], seeds[2]], CROWD, _commits())
    combined = _run_dir(tmp_path, "combined", [seeds[1], seeds[2], seeds[3]])
    out = {}
    for name, cells in (("two", [home, institute]), ("one", [combined])):
        args = [a for c in cells for a in ("--cells", str(c))]
        assert main(["forecast", "fit", str(tmp_path / "split.yaml"), *args, "--max-evals", "8", "--out", str(tmp_path / f"{name}.yaml")]) == 0
        assert main(["forecast", "score", str(tmp_path / "split.yaml"), *args, "--params", str(tmp_path / f"{name}.yaml"), "--partition", "train", "--out", str(tmp_path / f"{name}_score.yaml")]) == 0
        out[name] = (yaml.safe_load((tmp_path / f"{name}.yaml").read_text()), yaml.safe_load((tmp_path / f"{name}_score.yaml").read_text()))
    (fit_two, score_two), (fit_one, score_one) = out["two"], out["one"]
    assert fit_two["fit"]["cells"] == [str(home), str(institute)] and fit_one["fit"]["cells"] == str(combined)
    assert fit_two["params"] == fit_one["params"]
    assert {k: v for k, v in fit_two["fit"].items() if k != "cells"} == {k: v for k, v in fit_one["fit"].items() if k != "cells"}
    assert fit_two["fit"]["seeds"] == [1, 2, 3] and fit_two["fit"]["worlds"] == 1
    assert score_two["partitions"] == score_one["partitions"] and score_two["partitions"]["train"]["rows"]
    # the third seed's run recorded twice, differently: the fit stops instead of pooling both
    capsys.readouterr()
    clash = _run_dir(tmp_path, "clash", [seeds[3].with_columns(pl.col("ped_seconds") * 2)], CROWD, _commits())
    assert main(["forecast", "fit", str(tmp_path / "split.yaml"), "--cells", str(institute), "--cells", str(clash), "--max-evals", "8"]) == 1
    assert "stage hospital_1__s3 (peds_only)" in capsys.readouterr().err
    # the other machine on another humansim commit
    other = _run_dir(tmp_path, "other", [seeds[3]], CROWD, _commits(humansim="d" * 40))
    assert main(["forecast", "fit", str(tmp_path / "split.yaml"), "--cells", str(home), "--cells", str(other), "--max-evals", "8"]) == 1
    assert "different humansim commits" in capsys.readouterr().err
