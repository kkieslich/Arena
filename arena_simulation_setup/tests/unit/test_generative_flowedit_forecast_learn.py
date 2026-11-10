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
        def __call__(self, world: str, demand: bool = True):  # noqa: ANN204
            return r

    model, summary = train(Fixed(), [Sample(queue.name, 1, counts)], steps=60, batch=2, device="cpu", log=lambda _: None)
    assert summary["loss_last"] < summary["loss_first"]
    forecast = predict(model, r)
    assert forecast.shape == (r.grid.ny, r.grid.nx, M.BINS) and (forecast >= 0).all()
    assert forecast[~r.grid.room_cells()].sum() == 0
    save(tmp_path / "net.pt", model, {"demand": True})
    again, meta = load(tmp_path / "net.pt")
    assert np.allclose(predict(again, r), forecast, atol=1e-5) and meta["demand"] is True


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
