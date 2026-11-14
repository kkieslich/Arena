"""The learned forecaster (raster, network, training) and the validity checks (fair baselines, floor, change, tracks seen)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")
shapely = pytest.importorskip("shapely")

from arena_simulation_setup.generative.flowedit.forecast import maps as M  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast import validity as V  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast.evaluate import Runs  # noqa: E402

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
GRID = M.FlowGrid.from_polygons([[(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]])


def _walk(key: str, start: tuple[float, float], velocity: tuple[float, float], n: int = 20, dt: float = 0.1) -> pl.DataFrame:
    t = np.arange(n) * dt
    return pl.DataFrame({"key": [key] * n, "t": t, "x": start[0] + velocity[0] * t, "y": start[1] + velocity[1] * t, "vx": [velocity[0]] * n, "vy": [velocity[1]] * n})


# --- maps.counts mirrors arena_evaluation ------------------------------------------------------------------------------


def test_counts_match_arena_evaluation() -> None:
    flow = pytest.importorskip("arena_evaluation.processing.flow")
    rng = np.random.default_rng(3)
    angles = np.linspace(-math.pi, math.pi, 41)
    n = len(angles)
    speed = np.r_[np.ones(n - 4), 0.1, 0.1, 0.1, 0.1]
    tracks = pl.DataFrame({"key": ["a"] * (n // 2) + ["b"] * (n - n // 2), "t": np.r_[np.arange(n // 2) * 0.1, np.arange(n - n // 2) * 0.25], "x": rng.uniform(0.1, 5.9, n), "y": rng.uniform(0.1, 3.9, n), "vx": np.cos(angles) * speed, "vy": np.sin(angles) * speed})
    tracks = tracks.with_columns(pl.when(pl.int_range(pl.len()) % 7 == 0).then(float("nan")).otherwise(pl.col("vx")).alias("vx"))
    theirs = flow.FlowGrid.from_polygons([[(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]])
    assert np.allclose(M.counts(tracks, GRID), theirs.counts(tracks))


# --- symmetries ------------------------------------------------------------------------------------------------------


def test_a_quarter_turn_and_a_mirror_move_the_cells_and_the_heading_bins_with_the_world() -> None:
    torch = pytest.importorskip("torch")
    from arena_simulation_setup.generative.flowedit.forecast.net import transform, untransform

    square = M.FlowGrid.from_polygons([[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]])
    walk = _walk("a", (0.61, 1.33), (math.cos(0.2), math.sin(0.2)))  # off every cell edge and heading-bin edge
    turned = walk.select("key", "t", (5.0 - pl.col("y")).alias("x"), pl.col("x").alias("y"), (-pl.col("vy")).alias("vx"), pl.col("vx").alias("vy"))
    mirrored = walk.with_columns((5.0 - pl.col("x")).alias("x"), (-pl.col("vx")).alias("vx"))
    east, north, west = (M.counts(t, square) for t in (walk, turned, mirrored))
    assert east[:, :, 4].sum() > 0 and north[:, :, 6].sum() > 0 and west[:, :, 7].sum() > 0
    as_t = lambda m: torch.as_tensor(np.moveaxis(m, -1, 0))  # noqa: E731
    assert torch.allclose(transform(as_t(east), False, 1, bins=True), as_t(north))
    assert torch.allclose(transform(as_t(east), True, 0, bins=True), as_t(west))
    for flip in (False, True):
        for k in range(4):
            assert torch.allclose(untransform(transform(as_t(east), flip, k, bins=True), flip, k, bins=True), as_t(east))


# --- fair baselines, floor, change -----------------------------------------------------------------------------------


def _runs(maps: dict[tuple[str, int], np.ndarray]) -> Runs:
    return Runs(dict(maps))


def test_stale_and_the_floor_use_the_base_worlds_other_seeds_never_the_scored_ones() -> None:
    rng = np.random.default_rng(0)
    m = {(w, s): rng.random((2, 3, M.BINS)) for w in ("base", "base__move_x__1") for s in (1, 2, 3)}
    runs = _runs(m)
    stale = V.stale(runs, [3])
    assert np.allclose(stale("base__move_x__1"), M.pooled([m[("base", 1)], m[("base", 2)]]))
    assert V.other_seeds(runs, "base", [3]) == [1, 2]
    # equal pools: the one scored seed against each one other seed
    f = V.floor(runs, "base", [3])
    assert (f.seeds, f.pairs) == (1, 2) and f.mean == pytest.approx(np.mean([M.jsd(m[("base", 3)], m[("base", s)]) for s in (1, 2)]))
    two = V.floor(runs, "base", [2, 3])
    assert (two.seeds, two.pairs) == (1, 2)  # one other seed: pools of one, the scored side drawn both ways
    only = _runs({("office", 6): m[("base", 1)], ("office__move_x__1", 6): m[("base", 2)]})
    assert V.stale(only, [6])("office__move_x__1") is None and V.floor(only, "office", [6]) is None


def test_scores_and_changes_take_any_method() -> None:
    rng = np.random.default_rng(1)
    m = {(w, s): rng.random((2, 3, M.BINS)) for w in ("base", "base__move_x__1") for s in (1, 2)}
    runs = _runs(m)
    doc = {"partitions": {"held_out_edits": ["base", "base__move_x__1"]}, "seeds": {"held_out_edits": [1, 2]}}
    found = V.truths(runs, doc, "held_out_edits")
    perfect = {t.world: t.map for t in found}
    rows = V.score(found, {"perfect": perfect.get, "none": lambda w: None}, doc)
    assert [r["perfect"] for r in rows] == [0.0, 0.0] and rows[0]["none"] is None and rows[1]["what"] == "move_x"
    (change,) = V.changes(found, runs, {"perfect": perfect.get}, doc)
    assert change["perfect"] > 0 and change["recorded"] == pytest.approx(change["perfect"])
    assert "| mean over edits |" in V.table(rows, ["perfect", "none"])


def test_predicted_maps_round_trip_through_their_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    maps = {"w": rng.random((GRID.ny, GRID.nx, M.BINS)), "w__move_x__1": rng.random((GRID.ny, GRID.nx, M.BINS))}
    path = V.write_predicted(tmp_path / "p.parquet", maps, lambda base: GRID)
    assert set(pl.read_parquet(path).columns) >= set(V.PREDICTED_COLUMNS)
    loaded = V.Predicted.load(path, lambda base: GRID)
    assert np.allclose(loaded("w"), maps["w"]) and loaded("other") is None


# --- tracks seen -----------------------------------------------------------------------------------------------------


def test_the_dirichlet_update_starts_at_the_prior_and_ends_at_what_it_saw() -> None:
    mask = GRID.room_cells()
    prior = np.random.default_rng(4).random(GRID.shape)
    seen = M.counts(_walk("a", (0.5, 0.5), (1.0, 0.5)), GRID)
    assert np.allclose(V.dirichlet(prior, 50.0, np.zeros(GRID.shape), mask), M.normalized(prior))
    assert M.jsd(V.dirichlet(prior, 1e-6, seen, mask), seen) == pytest.approx(0.0, abs=1e-6)


def test_the_curve_falls_with_tracks_from_a_uniform_prior() -> None:
    walks = [_walk(f"p{i}", (0.3, 0.4 + 0.2 * i), (1.0, 0.0), n=50) for i in range(8)]
    tracks = pl.concat(walks)
    truth = M.counts(tracks, GRID)
    runs = _runs({("w", 1): truth, ("w", 2): truth})
    res = V.curve("w", [1, 2], lambda w, s: tracks, runs, GRID, {"uniform": (V.uniform(GRID.room_cells()), 10.0), "perfect": (truth, 10.0)}, counts=(0, 2, 8), orders=3)
    assert res["uniform"][0] > res["uniform"][1] > res["uniform"][2]
    assert res["perfect"][0] == pytest.approx(0.0, abs=1e-9)
    assert res[V.ONE_SEED_FLOOR] == [pytest.approx(0.0, abs=1e-9)] * 3  # the seen run equals the other seeds' pool here


# --- on hospital_1: the raster and a few training steps ----------------------------------------------------------------


@pytest.fixture(scope="module")
def queue():  # noqa: ANN201
    pytest.importorskip("arena_humansim")
    from arena_simulation_setup.generative.flowedit.edit import open_site

    return open_site(HOSPITAL, "queue")


def _raster(site, base, demand: bool = True):  # noqa: ANN001, ANN202
    from arena_simulation_setup.generative.flowedit import settings as S
    from arena_simulation_setup.generative.flowedit.forecast.raster import raster
    from arena_simulation_setup.generative.flowedit.forecast.scene import base_people, scene_of

    settings = S.load({"entrances": {"rate": 0}})
    grid = M.FlowGrid.from_world(HOSPITAL)
    scene = scene_of(site, settings, 300.0, grid=grid, people=base_people(base, settings, (1,)))
    return raster(site, scene, settings, demand=demand), scene


@pytest.mark.usefixtures("local_root")
def test_the_raster_holds_the_graph_and_geometry_and_the_ablation_drops_the_demand(queue) -> None:  # noqa: ANN001
    from arena_simulation_setup.generative.flowedit.edit import apply
    from arena_simulation_setup.generative.flowedit.forecast.raster import CHANNELS, NEED_CHANNELS

    r, scene = _raster(queue, queue)
    assert r.channels.shape == (len(CHANNELS), r.grid.ny, r.grid.nx)
    assert np.array_equal(r.channels[CHANNELS.index("rooms")] > 0, r.grid.room_cells())
    assert r.needs == ("thirst",) and r.need_channels.shape == (1, len(NEED_CHANNELS), r.grid.ny, r.grid.nx)
    (fountain,) = scene.demands
    i, j, _ = r.grid.index(*fountain.spot(0))
    assert r.need_channels[0, NEED_CHANNELS.index("front_capacity"), i, j] == fountain.capacity
    near = r.need_channels[0, NEED_CHANNELS.index("near")]
    assert near[i, j] == near.max() > 0.9
    assert set(r.nodes["demand"].names) == {"fountain"} and len(r.edges["room_need"].src) == len(r.nodes["room"].names)
    assert r.channels[CHANNELS.index("locked")].sum() == 0
    locked, _ = _raster(apply(queue, "access", "door:patient_ward_door_2", {"locked": True}), queue)
    assert locked.channels[CHANNELS.index("locked")].sum() > 0 and locked.nodes["door"].x[:, 1].sum() == 1
    geometry, _ = _raster(queue, queue, demand=False)
    assert geometry.needs == () and geometry.need_channels.shape[0] == 0 and len(geometry.nodes["demand"].names) == 0
    assert np.array_equal(geometry.channels, r.channels)


@pytest.mark.usefixtures("local_root")
def test_a_few_training_steps_lower_the_loss_and_a_saved_net_forecasts_the_same(queue, tmp_path: Path) -> None:  # noqa: ANN001
    pytest.importorskip("torch")
    from arena_simulation_setup.generative.flowedit.forecast.learn import Sample, load, predict, save, train

    r, scene = _raster(queue, queue)
    # a "recording": everyone stands at the fountain's front for the whole run
    counts = np.zeros((M.BINS, r.grid.ny, r.grid.nx), np.float32)
    i, j, _ = r.grid.index(*scene.demands[0].spot(0))
    counts[M.STANDING, i, j] = r.people * 300.0

    class Fixed:
        def __call__(self, world: str, demand: bool = True, demand_model=None):  # noqa: ANN001, ANN204
            return r

    model, summary = train(Fixed(), [Sample(queue.name, 1, counts)], steps=60, batch=2, device="cpu", log=lambda _: None)
    assert summary["loss_last"] < summary["loss_first"]
    forecast = predict(model, r)
    assert forecast.shape == (r.grid.ny, r.grid.nx, M.BINS) and (forecast >= 0).all()
    assert forecast[~r.grid.room_cells()].sum() == 0
    save(tmp_path / "net.pt", model, {"demand": True})
    again, meta = load(tmp_path / "net.pt")
    assert np.allclose(predict(again, r), forecast, atol=1e-5) and meta["demand"] is True


@pytest.mark.usefixtures("local_root")
def test_the_demand_model_variant_takes_the_forecast_as_input_layers_and_keeps_them_through_save(queue, tmp_path: Path) -> None:  # noqa: ANN001
    pytest.importorskip("torch")
    import attrs

    from arena_simulation_setup.generative.flowedit.forecast.learn import Sample, load, predict, save, train
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    r, scene = _raster(queue, queue)
    extra = np.zeros((M.BINS, r.grid.ny, r.grid.nx), np.float32)
    i, j, _ = r.grid.index(*scene.demands[0].spot(0))
    extra[M.STANDING, i, j] = 5.0
    with_input = attrs.evolve(r, extra=extra)
    counts = np.zeros((M.BINS, r.grid.ny, r.grid.nx), np.float32)
    counts[M.STANDING, i, j] = r.people * 300.0
    asked = []

    class Fixed:
        def __call__(self, world: str, demand: bool = True, demand_model=None):  # noqa: ANN001, ANN204
            asked.append(demand_model)
            return with_input if demand_model is not None else r

    params = Params(sigma=0.3)
    model, _ = train(Fixed(), [Sample(queue.name, 1, counts)], steps=30, batch=2, device="cpu", demand_model=params, log=lambda _: None)
    assert model.extra == M.BINS and asked == [params]
    forecast = predict(model, with_input)
    with pytest.raises(ValueError, match="extra input layers"):
        predict(model, r)
    save(tmp_path / "net.pt", model, {})
    again, _ = load(tmp_path / "net.pt")
    assert again.extra == M.BINS and again.demand_model == params and np.allclose(predict(again, with_input), forecast, atol=1e-5)
    with pytest.raises(ValueError, match="demand ablation"):
        train(Fixed(), [Sample(queue.name, 1, counts)], demand=False, steps=1, device="cpu", demand_model=params, log=lambda _: None)


def test_chains_keep_an_edit_with_its_controls() -> None:
    from arena_simulation_setup.generative.flowedit.forecast.learn import chains, pick

    doc = {"partitions": {"train": ["b", "b__move_f__1", "b__insert_f_inert__2", "b__remove_s__3"]}, "control_worlds": {"b__move_f__1": ["b__insert_f_inert__2"]}}
    assert chains(doc, "train") == [["b__move_f__1", "b__insert_f_inert__2"], ["b__remove_s__3"]]
    half = pick(doc, "train", 0.5, 0)
    assert half[0] == "b" and len(half) in (2, 3)
    assert pick(doc, "train", 1.0, 0) == ["b", "b__move_f__1", "b__insert_f_inert__2", "b__remove_s__3"]


def test_training_runs_leave_out_the_seeds_the_held_out_partitions_score() -> None:
    from arena_simulation_setup.generative.flowedit.forecast.learn import held_out_seeds, samples

    doc = {"partitions": {"train": ["b", "b__move_f__1"], "held_out_edits": ["b__move_f__2"], "held_out_worlds": ["o"]}, "seeds": {"train": [1, 2], "held_out_edits": [6], "held_out_worlds": [7]}}
    m = np.zeros((2, 3, M.BINS))
    runs = _runs({("b", s): m for s in (1, 2, 3, 4, 6)} | {("b__move_f__1", 1): m, ("b__move_f__2", 6): m, ("o", 7): m})
    assert held_out_seeds(doc, "b") == {6} and held_out_seeds(doc, "o") == {7}
    # the base world's extra seeds 3 and 4 train too; its reference run at the held-out seed 6 does not
    assert [(s.world, s.seed) for s in samples(runs, ["b", "b__move_f__1"], doc)] == [("b", 1), ("b", 2), ("b", 3), ("b", 4), ("b__move_f__1", 1)]


def test_a_split_with_seeds_apart_for_bases_and_edits_scores_each_at_its_own_and_never_trains_on_baseline_seeds() -> None:
    from arena_simulation_setup.generative.flowedit.forecast.learn import held_out_seeds

    doc = {
        "partitions": {"train": ["b", "b__move_f__1"], "held_out_edits": ["b__move_f__2"], "held_out_worlds": ["o", "o__move_f__3"]},
        "seeds": {"train": {"bases": [1, 2, 3], "edits": [1]}, "held_out_edits": [6], "held_out_worlds": [6]},
        "baseline_seeds": {"held_out_worlds": [9], "held_out_edits": [10]},
    }
    assert V.world_seeds(doc, "train", "b") == [1, 2, 3] and V.world_seeds(doc, "train", "b__move_f__1") == [1]
    m = np.random.default_rng(5).random((2, 3, M.BINS))
    runs = _runs({("o", 6): m, ("o", 9): m, ("o__move_f__3", 6): m})
    found = V.truths(runs, doc, "held_out_worlds")
    assert [(t.world, t.seeds) for t in found] == [("o", (6,)), ("o__move_f__3", (6,))]
    assert V.other_seeds(runs, "o", [6]) == [9]
    assert held_out_seeds(doc, "b") == {6, 10}


def test_jsd_stays_finite_with_subnormal_entries() -> None:
    p = np.zeros((2, 2, M.BINS))
    q = np.zeros((2, 2, M.BINS))
    p[0, 0, 0], p[1, 1, 1] = 1.0, 5e-324
    q[0, 0, 0] = 1.0
    assert M.jsd(p, q) == pytest.approx(0.0, abs=1e-12)


def test_the_demand_model_layers_turn_their_heading_bins_with_the_grid() -> None:
    torch = pytest.importorskip("torch")
    from arena_simulation_setup.generative.flowedit.forecast.net import DIM, NEED_DIM, WRITTEN, Forecaster, transform
    from arena_simulation_setup.generative.flowedit.forecast.raster import CHANNELS

    square = M.FlowGrid.from_polygons([[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)]])
    walk = _walk("a", (0.61, 1.33), (math.cos(0.2), math.sin(0.2)))
    turned = walk.select("key", "t", (5.0 - pl.col("y")).alias("x"), pl.col("x").alias("y"), (-pl.col("vy")).alias("vx"), pl.col("vx").alias("vy"))
    east, north = (torch.as_tensor(np.moveaxis(M.counts(t, square), -1, 0), dtype=torch.float32) for t in (walk, turned))
    net = Forecaster(extra=M.BINS)
    c0 = len(CHANNELS)
    inputs = torch.zeros((c0 + M.BINS + NEED_DIM + DIM * len(WRITTEN), square.ny, square.nx))
    inputs[c0 : c0 + M.BINS] = east
    out = net.oriented(inputs, False, 1)
    # the demand model's map of the turned world, as the turned target holds it; the other layers only move
    assert torch.allclose(out[c0 : c0 + M.BINS], north)
    assert torch.allclose(out[:c0], transform(inputs[:c0], False, 1))
    with pytest.raises(ValueError, match="9 bins"):
        Forecaster(extra=3)


def test_strength_folds_hold_out_two_seeds_and_the_choice_reports_the_spread() -> None:
    assert V.seed_folds([3, 1, 2]) == [(1, 2), (2, 3), (3, 1)]
    with pytest.raises(ValueError, match="at least 3"):
        V.seed_folds([1, 2])
    chosen, spread = V.choose_strength([{10.0: 0.3, 100.0: 0.2}, {10.0: 0.25, 100.0: 0.26}])
    assert chosen == 100.0 and spread["fold_best"] == [100.0, 10.0] and spread["score_at_chosen"]["max"] == 0.26


def test_a_prior_that_matches_the_held_out_truth_wants_a_heavy_weight_and_a_uniform_one_a_light_one() -> None:
    walks = [_walk(f"p{i}", (0.3, 0.4 + 0.2 * i), (1.0, 0.0), n=50) for i in range(8)]
    tracks = pl.concat(walks)
    truth = M.counts(tracks, GRID)
    runs = _runs({("w", 1): truth, ("w", 2): truth})
    perfect = V.fold_strengths(["w"], (1, 2), lambda w, s: tracks, runs, lambda base: GRID, lambda w: truth, strengths=(10.0, 1000.0))
    flat = V.fold_strengths(["w"], (1, 2), lambda w, s: tracks, runs, lambda base: GRID, lambda w: V.uniform(GRID.room_cells()), strengths=(10.0, 1000.0))
    assert perfect[1000.0] <= perfect[10.0] and flat[10.0] < flat[1000.0]


def test_strength_cross_validates_the_demand_model_input_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import yaml

    from arena_simulation_setup.generative.flowedit import split as SP
    from arena_simulation_setup.generative.flowedit.cli import main
    from arena_simulation_setup.generative.flowedit.forecast import cli, learn
    from arena_simulation_setup.generative.flowedit.forecast.learn import Sample
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    params = Params(sigma=0.3)
    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": params.dump(), "fit": {"partition": "train", "per_world": {"w": 0.4}, "seeds": [0]}}))
    monkeypatch.setattr(SP, "load", lambda path: {"partitions": {"train": ["w"]}, "duration": 60})
    monkeypatch.setattr(SP, "crowd", lambda doc, over=None: {})
    monkeypatch.setattr(learn.Rasters, "of", classmethod(lambda cls, doc, settings, worlds=None: types.SimpleNamespace(sites=types.SimpleNamespace(grid=lambda base: GRID))))
    monkeypatch.setattr(cli, "_recorded", lambda benchmarks, grid_of, planner: (None, None))
    monkeypatch.setattr(learn, "samples", lambda runs, worlds, doc: [Sample("w", s, None) for s in (1, 2, 3)])
    trained, forecast_with = [], []

    def fake_train(rasters, found, *, demand, steps, seed, device, demand_model, log):  # noqa: ANN001, ANN202
        trained.append(demand_model)
        return types.SimpleNamespace(demand=demand, demand_model=demand_model), {"steps": steps}

    def fake_forecast(model, rasters, world):  # noqa: ANN001, ANN202
        forecast_with.append(model.demand_model)
        return np.ones((GRID.ny, GRID.nx, M.BINS))

    def fake_fold(worlds, fold, tracks, runs, grid_of, prior):  # noqa: ANN001, ANN202
        prior("w")
        return {100.0: 0.2, 1000.0: 0.3}

    monkeypatch.setattr(learn, "train", fake_train)
    monkeypatch.setattr(learn, "forecast", fake_forecast)
    monkeypatch.setattr(V, "fold_strengths", fake_fold)
    out = tmp_path / "strength.yaml"
    assert main(["forecast", "strength", "split.yaml", "--benchmark", str(tmp_path), "--demand-model-input", str(params_file), "--folds", "1", "--out", str(out)]) == 0
    # the fold network takes the demand model's forecast, and the prior it scores is that network's forecast with it
    assert trained == [params] and forecast_with == [params]
    doc = yaml.safe_load(out.read_text())
    assert doc["prior"] == "network + demand model input" and doc["demand_model_input"] == str(params_file) and "held-out seeds included" in doc["note"]
    assert doc["net"]["chosen"] == 100.0 and doc["demand_model"] == params.dump()
    with pytest.raises(SystemExit, match="cannot combine with --no-demand"):
        main(["forecast", "strength", "split.yaml", "--benchmark", str(tmp_path), "--demand-model-input", str(params_file), "--no-demand", "--out", str(out)])
    no_fit = tmp_path / "params_no_fit.yaml"
    no_fit.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": params.dump()}))
    assert main(["forecast", "strength", "split.yaml", "--benchmark", str(tmp_path), "--demand-model-input", str(no_fit), "--folds", "1", "--out", str(out)]) == 0
    assert "records no fit" in yaml.safe_load(out.read_text())["note"]
    other_split = tmp_path / "params_other_split.yaml"
    other_split.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": params.dump(), "fit": {"partition": "train", "split": "/x/peds_pilot.yaml", "per_world": {"w": 0.4, "pilot_world": 0.5}, "seeds": [0]}}))
    with pytest.raises(SystemExit, match="fitted on 1 world.s. outside the train partition"):
        main(["forecast", "strength", "split.yaml", "--benchmark", str(tmp_path), "--demand-model-input", str(other_split), "--out", str(out)])
    with pytest.raises(SystemExit, match="train: --demand-model-input .* was fitted on the held_out_edits partition"):
        held = tmp_path / "params_held.yaml"
        held.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": params.dump(), "fit": {"partition": "held_out_edits", "per_world": {"w": 0.4}, "seeds": [0]}}))
        main(["forecast", "train", "split.yaml", "--cells", str(tmp_path / "cells.parquet"), "--demand-model-input", str(held), "--out", str(tmp_path / "net.pt")])
    held_out = tmp_path / "params_held_out.yaml"
    held_out.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": params.dump(), "fit": {"partition": "held_out_edits", "per_world": {"w": 0.4}, "seeds": [0]}}))
    with pytest.raises(SystemExit, match="fitted on the held_out_edits partition"):
        main(["forecast", "strength", "split.yaml", "--benchmark", str(tmp_path), "--demand-model-input", str(held_out), "--out", str(out)])


def test_the_strength_scores_do_not_depend_on_scoring_one_world_at_a_time() -> None:
    # the per-strength loop over worlds with a shared track cache gave these values; scoring every strength in one curve
    # call per world (one world's track maps held at a time) must give the same
    rng = np.random.default_rng(5)
    worlds = {"w": [_walk(f"p{i}", (0.3, 0.2 + 0.17 * i), (1.0, 0.0), n=40) for i in range(20)], "v": [_walk(f"q{i}", (5.5, 0.2 + 0.18 * i), (-0.8, 0.0), n=30) for i in range(18)]}
    tracks = {(w, s): pl.concat(ws[s - 1 :] + ws[: s - 1]) for w, ws in worlds.items() for s in (1, 2, 3)}
    runs = _runs({key: M.counts(t, GRID) + rng.uniform(0, 0.1, (GRID.ny, GRID.nx, M.BINS)) for key, t in tracks.items()})
    prior = {"w": M.counts(tracks[("w", 1)], GRID) + 0.05, "v": V.uniform(GRID.room_cells())}
    grid = (10.0, 100.0, 1000.0)
    old = {}
    cache: dict = {}
    for c in grid:
        vals = []
        for w in worlds:
            res = V.curve(w, [1, 2, 3], lambda w, s: tracks[(w, s)], runs, GRID, {"p": (prior[w], c)}, counts=(1, 4, 16), orders=4, cache=cache)
            vals.extend(v for v in res["p"] if v is not None)
        old[c] = vals
    scores = V.strength_scores([(w, [1, 2, 3]) for w in worlds], lambda w, s: tracks[(w, s)], runs, lambda base: GRID, lambda w: prior[w], grid)
    # every count is scored in both worlds (the runs hold more than 16 tracks), and the scores are the old ones exactly
    assert all(len(v) == 2 * 3 for v in old.values()) and scores == old
    old = {c: float(np.mean(v)) for c, v in old.items()}
    assert V.fit_strength([(w, [1, 2, 3]) for w in worlds], lambda w, s: tracks[(w, s)], runs, lambda base: GRID, lambda w: prior[w], grid=grid) == min(grid, key=lambda c: (old[c], grid.index(c)))
    folds_old: dict = {c: [] for c in grid}
    cache = {}
    for w in worlds:
        res = V.curve(w, [1, 2], lambda w, s: tracks[(w, s)], runs, GRID, {str(c): (prior[w], c) for c in grid}, counts=V.STRENGTH_COUNTS, orders=10, cache=cache)
        for c in grid:
            folds_old[c].extend(v for v in res[str(c)] if v is not None)
    new = V.fold_strengths(list(worlds), (1, 2), lambda w, s: tracks[(w, s)], runs, lambda base: GRID, lambda w: prior[w], strengths=grid)
    assert new == pytest.approx({c: float(np.mean(v)) for c, v in folds_old.items()}, abs=1e-12)


def test_params_are_checked_by_the_worlds_and_seeds_they_were_fitted_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import yaml

    from arena_simulation_setup.generative.flowedit import split as SP
    from arena_simulation_setup.generative.flowedit.cli import main
    from arena_simulation_setup.generative.flowedit.forecast import cli
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    doc = {"partitions": {"train": ["w", "w__e__1"], "held_out_edits": ["w__e__2"]}, "seeds": {"train": {"bases": [1, 2, 3], "edits": [1, 2]}, "held_out_edits": [6, 7]}, "duration": 60}
    monkeypatch.setattr(SP, "load", lambda path: doc)

    def params_file(name: str, **fit: object) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": Params(sigma=0.3).dump(), **({"fit": fit} if fit else {})}))
        return path

    # valid params under another name and path (a copy, a link, another machine's checkout) are accepted
    valid = params_file("copied.yaml", partition="train", split="/elsewhere/renamed_split.yaml", seeds=[1, 2, 3], per_world={"w": 0.4, "w__e__1": 0.5})
    assert cli.checked_params(valid, tmp_path / "split.yaml", "strength") == (Params(sigma=0.3), "train")
    # seeds compare as a set
    assert cli.checked_params(params_file("reordered.yaml", partition="train", seeds=[3, 1, 2], per_world={"w": 0.4}), tmp_path / "split.yaml", "strength")[1] == "train"
    # a hand-written file records no fit: not checked, and it says so
    assert cli.checked_params(params_file("hand.yaml"), tmp_path / "split.yaml", "strength") == (Params(sigma=0.3), None)
    assert "records no fit, so where its params come from is not checked" in capsys.readouterr().err
    # an incomplete fit record is refused
    with pytest.raises(SystemExit, match="incomplete fit record .no per_world, seeds."):
        cli.checked_params(params_file("partial.yaml", partition="train"), tmp_path / "split.yaml", "strength")
    other_worlds = params_file("older.yaml", partition="train", split="split.yaml", seeds=[1, 2, 3], per_world={"w": 0.4, "w__e__9": 0.5})
    with pytest.raises(SystemExit, match="fitted on 1 world.s. outside the train partition .*w__e__9"):
        cli.checked_params(other_worlds, tmp_path / "split.yaml", "strength")
    other_seeds = params_file("seeds.yaml", partition="train", seeds=[1, 2], per_world={"w": 0.4})
    with pytest.raises(SystemExit, match="fitted at seeds .1, 2., the train partition .* runs at .1, 2, 3."):
        cli.checked_params(other_seeds, tmp_path / "split.yaml", "strength")
    # every verb that takes params refuses them before any work
    with pytest.raises(SystemExit, match="train: --demand-model-input .* outside the train partition"):
        main(["forecast", "train", "split.yaml", "--cells", str(tmp_path / "cells.parquet"), "--demand-model-input", str(other_worlds), "--out", str(tmp_path / "net.pt")])
    with pytest.raises(SystemExit, match="select: --params .* outside the train partition"):
        main(["forecast", "select", "split.yaml", "--cells", str(tmp_path / "cells.parquet"), "--params", str(other_worlds), "--out", str(tmp_path / "select.yaml")])
    with pytest.raises(SystemExit, match="score: --params .* fitted at seeds"):
        main(["forecast", "score", "split.yaml", "--cells", str(tmp_path / "cells.parquet"), "--params", str(other_seeds), "--out", str(tmp_path / "score.yaml")])


def test_score_writes_whether_its_params_were_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import yaml

    from arena_simulation_setup.generative.flowedit import split as SP
    from arena_simulation_setup.generative.flowedit.cli import main
    from arena_simulation_setup.generative.flowedit.forecast import cli, evaluate, learn
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    doc = {"partitions": {"train": ["w"], "held_out_edits": ["w__e__1"]}, "seeds": {"train": [1, 2], "held_out_edits": [6]}, "duration": 60}
    monkeypatch.setattr(SP, "load", lambda path: doc)
    monkeypatch.setattr(SP, "crowd", lambda doc, over=None: {})
    monkeypatch.setattr(learn.Rasters, "of", classmethod(lambda cls, doc, settings, worlds=None: types.SimpleNamespace(sites=types.SimpleNamespace(grid=lambda base: GRID), scenario_of={})))
    monkeypatch.setattr(evaluate.Runs, "load", classmethod(lambda cls, paths, grid_of, planner=None: types.SimpleNamespace(maps={})))
    monkeypatch.setattr(V, "truths", lambda runs, doc, partition, log: [])
    for name, fit in (("hand.yaml", None), ("fitted.yaml", {"partition": "train", "per_world": {"w": 0.4}, "seeds": [1, 2]})):
        path = tmp_path / name
        path.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": Params().dump(), **({"fit": fit} if fit else {})}))
        out = tmp_path / f"score_{name}"
        assert main(["forecast", "score", "split.yaml", "--cells", str(tmp_path / "cells.parquet"), "--params", str(path), "--partition", "held_out_edits", "--out", str(out)]) == 0
        assert yaml.safe_load(out.read_text())["params_checked"] is (fit is not None)


def test_a_strength_file_belongs_to_one_network() -> None:
    import types

    from arena_simulation_setup.generative.flowedit.forecast.cli import strength_mismatch
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    plain = types.SimpleNamespace(demand=True, demand_model=None)
    variant = types.SimpleNamespace(demand=True, demand_model=Params(sigma=0.3))
    ablation = types.SimpleNamespace(demand=False, demand_model=None)
    plain_file, ablation_file = {"demand": True, "demand_model": None}, {"demand": False, "demand_model": None}
    variant_file = {"demand": True, "demand_model": Params(sigma=0.3).dump()}
    assert strength_mismatch(plain_file, plain) is None and strength_mismatch(variant_file, variant) is None and strength_mismatch(ablation_file, ablation) is None
    # a file from before strength recorded its demand model is a plain network's
    assert strength_mismatch({"demand": True}, plain) is None
    assert strength_mismatch(plain_file, variant) == "the file is the plain network's, the network is the one with the demand model input"
    assert strength_mismatch(variant_file, plain) == "the file is the network with the demand model input's, the network is plain"
    assert strength_mismatch(ablation_file, plain) == "the file is the demand ablation's, the network is a network with demand"
    assert strength_mismatch(ablation_file, variant) == "the file is the demand ablation's, the network is a network with demand"
    assert strength_mismatch(plain_file, ablation) == "the file is the network with demand's, the network is the demand ablation"
    assert strength_mismatch({"demand": True, "demand_model": Params(sigma=0.5).dump()}, variant).startswith("its demand model params")


def test_a_strength_file_from_before_the_params_were_recorded_is_a_variants_when_it_names_a_demand_model_input(tmp_path: Path) -> None:
    import types

    import yaml

    from arena_simulation_setup.generative.flowedit.forecast import cli
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    params_file = tmp_path / "params.yaml"
    params_file.write_text(yaml.safe_dump({"format": cli.FORMAT, "params": Params(sigma=0.3).dump()}))
    old_variant = {"demand": True, "demand_model_input": str(params_file)}
    plain = types.SimpleNamespace(demand=True, demand_model=None)
    assert "network is plain" in cli.strength_mismatch(old_variant, plain)
    assert cli.strength_mismatch(old_variant, types.SimpleNamespace(demand=True, demand_model=Params(sigma=0.3))) is None
    assert cli.strength_mismatch(old_variant, types.SimpleNamespace(demand=True, demand_model=Params(sigma=0.5))).startswith("its demand model params")
    params_file.unlink()
    assert "run strength again" in cli.strength_mismatch(old_variant, types.SimpleNamespace(demand=True, demand_model=Params(sigma=0.3)))


def test_curve_refuses_a_strength_file_of_another_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    import yaml

    from arena_simulation_setup.generative.flowedit import split as SP
    from arena_simulation_setup.generative.flowedit.cli import main
    from arena_simulation_setup.generative.flowedit.forecast import cli, learn
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    monkeypatch.setattr(SP, "load", lambda path: {"partitions": {"train": ["w"], "held_out_edits": ["w__e__1"]}, "duration": 60})
    monkeypatch.setattr(SP, "crowd", lambda doc, over=None: {})
    monkeypatch.setattr(learn.Rasters, "of", classmethod(lambda cls, doc, settings, worlds=None: types.SimpleNamespace(sites=types.SimpleNamespace(grid=lambda base: GRID))))
    monkeypatch.setattr(cli, "_recorded", lambda benchmarks, grid_of, planner: (None, None))

    def variant_method(world):  # noqa: ANN001, ANN202
        return np.ones((GRID.ny, GRID.nx, M.BINS))

    variant_method.model = types.SimpleNamespace(demand=True, demand_model=Params(sigma=0.3))
    monkeypatch.setattr(cli, "_net_methods", lambda nets, rasters, device, chosen=None: {"forecaster": variant_method})
    plain_file = tmp_path / "strength_plain.yaml"
    plain_file.write_text(yaml.safe_dump({"demand": True, "demand_model": None, "net": {"chosen": 1000.0}, "uniform": {"chosen": 100.0}}))
    net = tmp_path / "net.pt"
    net.touch()
    with pytest.raises(SystemExit, match="not this network's weight: the file is the plain network's"):
        main(["forecast", "curve", "split.yaml", "--benchmark", str(tmp_path), "--net", str(net), "--strength-file", f"forecaster={plain_file}", "--out", str(tmp_path / "curve.yaml")])


def test_tracks_come_from_the_newest_track_store_version_that_holds_them(tmp_path: Path) -> None:
    from arena_simulation_setup.generative.flowedit.forecast.cli import track_file

    derived = tmp_path / "topics" / ".derived"
    for version in ("v1", "v2", "v10", "vx"):
        (derived / f"ped_tracks.{version}").mkdir(parents=True)
    for version in ("v1", "v2", "vx"):
        (derived / f"ped_tracks.{version}" / "env_3.parquet").touch()
    assert track_file(derived, "env_3.parquet") == derived / "ped_tracks.v2" / "env_3.parquet"
    assert track_file(derived, "env_4.parquet") is None and track_file(tmp_path / "missing", "env_3.parquet") is None


def test_a_curve_step_with_an_empty_posterior_is_not_scored() -> None:
    tracks = _walk("p", (0.3, 0.5), (1.0, 0.0), n=40)
    truth = M.counts(tracks, GRID)
    runs = _runs({("w", 1): truth, ("w", 2): truth})
    res = V.curve("w", [1, 2], lambda w, s: tracks, runs, GRID, {"empty": (np.zeros(GRID.shape), 10.0)}, counts=(0, 1), orders=2)
    assert res["empty"][0] is None and res["empty"][1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.usefixtures("local_root")
def test_select_holds_out_each_training_world_for_both_arms_and_never_reads_the_test_partitions(queue) -> None:  # noqa: ANN001
    pytest.importorskip("torch")
    import attrs

    from arena_simulation_setup.generative.flowedit.forecast import selection as SE
    from arena_simulation_setup.generative.flowedit.forecast.model import Params

    r, scene = _raster(queue, queue)
    with_input = attrs.evolve(r, extra=np.zeros((M.BINS, r.grid.ny, r.grid.nx), np.float32))
    asked: list[str] = []

    class Fixed:
        def __call__(self, world: str, demand: bool = True, demand_model=None):  # noqa: ANN001, ANN204
            asked.append(world)
            return with_input if demand_model is not None else r

    rng = np.random.default_rng(7)
    m = lambda: rng.random((r.grid.ny, r.grid.nx, M.BINS))  # noqa: E731
    doc = {
        "partitions": {"train": ["a", "a__move_f__1", "b"], "held_out_edits": ["a__move_f__2"], "held_out_worlds": ["c"]},
        "seeds": {"train": [1, 2], "held_out_edits": [6], "held_out_worlds": [6]},
    }
    runs = _runs({("a", 1): m(), ("a", 2): m(), ("a", 6): m(), ("a__move_f__1", 1): m(), ("b", 1): m(), ("b", 2): m(), ("a__move_f__2", 6): m(), ("c", 6): m()})
    assert sorted(SE.train_runs(runs, doc).maps) == [("a", 1), ("a", 2), ("a__move_f__1", 1), ("b", 1), ("b", 2)]
    out = SE.select(Fixed(), runs, doc, params=Params(), steps=30, device="cpu", log=lambda _: None)
    assert [f["held_out"] for f in out["folds"]] == ["a", "b"] and out["folds"][0]["worlds"] == 2
    assert out["chosen"] in SE.ARMS and set(out["mean"]) == set(SE.ARMS) and "note" in out
    assert set(asked) <= {"a", "a__move_f__1", "b"}
    with pytest.raises(ValueError, match="at least 2 training worlds"):
        SE.select(Fixed(), runs, {**doc, "partitions": {**doc["partitions"], "train": ["a", "a__move_f__1"]}}, params=Params(), steps=30, device="cpu")


def test_the_chosen_arm_is_the_forecaster_and_the_other_goes_by_its_own_name() -> None:
    from types import SimpleNamespace

    from arena_simulation_setup.generative.flowedit.forecast.selection import DEMAND_MODEL_INPUT, PLAIN, arm_label

    plain, variant, ablation = SimpleNamespace(demand=True, demand_model=None), SimpleNamespace(demand=True, demand_model=object()), SimpleNamespace(demand=False, demand_model=None)
    assert (arm_label(plain, None), arm_label(variant, None), arm_label(ablation, None)) == ("forecaster", "forecaster + demand model", "demand ablation")
    assert (arm_label(plain, DEMAND_MODEL_INPUT), arm_label(variant, DEMAND_MODEL_INPUT)) == ("plain network", "forecaster")
    assert (arm_label(plain, PLAIN), arm_label(variant, PLAIN), arm_label(ablation, PLAIN)) == ("forecaster", "network + demand model input", "demand ablation")
