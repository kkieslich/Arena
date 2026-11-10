from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
shapely = pytest.importorskip("shapely")
pl = pytest.importorskip("polars")

from arena_simulation_setup.generative.flowedit.forecast import baselines as B  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast import maps as M  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast.cliff import CliffMap, CliffParams, Observations, build, observations  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast.evaluate import Runs  # noqa: E402

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
ROOM = M.FlowGrid.from_polygons([list(shapely.box(0, 0, 12, 6).exterior.coords)[:-1]])


def _track(key: str, start: tuple[float, float], velocity: tuple[float, float], seconds: float, rng: np.random.Generator, hz: float = 15.0) -> pl.DataFrame:
    n = int(seconds * hz)
    t = np.arange(n) / hz
    return pl.DataFrame(
        {
            "key": [key] * n,
            "t": t,
            "x": start[0] + velocity[0] * t + rng.normal(0, 0.02, n),
            "y": start[1] + velocity[1] * t + rng.normal(0, 0.02, n),
            "vx": velocity[0] + rng.normal(0, 0.05, n),
            "vy": velocity[1] + rng.normal(0, 0.05, n),
        }
    )


def _cell_bins(grid: M.FlowGrid, m: np.ndarray, point: tuple[float, float]) -> np.ndarray:
    i, j, _ = grid.index(*point)
    v = m[int(i), int(j)]
    return v / v.sum()


# --- the CLiFF-map ---------------------------------------------------------------------------------------------------


def test_observations_downsample_and_wrap_the_direction() -> None:
    track = _track("a", (1.0, 3.0), (-1.0, 0.0), 10.0, np.random.default_rng(0))
    obs = observations(track, 2.0)
    assert len(obs) == 20
    assert np.all((obs.theta >= 0) & (obs.theta < 2 * math.pi))
    assert np.allclose(obs.theta, math.pi, atol=0.3) and np.allclose(obs.rho, 1.0, atol=0.2)


def test_em_keeps_a_mode_across_the_wrap_point_whole() -> None:
    rng = np.random.default_rng(1)
    theta = np.mod(rng.normal(0.0, 0.1, 200), 2 * math.pi)
    rho = rng.normal(1.2, 0.05, 200)
    m, mu, _ = build([np.arange(200)], theta, rho, CliffParams())
    (k,) = np.flatnonzero(m[0] > 0.05)
    assert m[0, k] > 0.95
    assert min(mu[0, k, 0], 2 * math.pi - mu[0, k, 0]) < 0.05 and mu[0, k, 1] == pytest.approx(1.2, abs=0.02)


def test_mean_shift_finds_two_opposite_flows() -> None:
    rng = np.random.default_rng(2)
    theta = np.concatenate([rng.normal(0.5 * math.pi, 0.1, 120), rng.normal(1.5 * math.pi, 0.1, 60)])
    rho = np.concatenate([rng.normal(1.0, 0.05, 120), rng.normal(0.8, 0.05, 60)])
    m, mu, _ = build([np.arange(180)], theta, rho, CliffParams())
    big = np.argsort(-m[0])[:2]
    assert m[0, big].sum() > 0.98 and m[0, big[0]] == pytest.approx(2 / 3, abs=0.03)
    assert sorted(np.round(mu[0, big, 0] / math.pi, 1)) == [0.5, 1.5]


def test_map_projects_motion_onto_heading_bins_and_standing() -> None:
    rng = np.random.default_rng(3)
    cm = CliffMap.empty(ROOM, CliffParams(decay=1.0))
    # axis-aligned headings lie on bin edges; 3 pi / 8 is the centre of bin floor((3 pi / 8 + pi) / (pi / 4)) = 5
    heading = 3 * math.pi / 8
    walker = _track("n", (5.0, 0.5), (math.cos(heading), math.sin(heading)), 5.0, rng)
    stand = _track("s", (2.25, 4.25), (0.0, 0.0), 20.0, rng)
    for t in (walker, stand):
        cm.update(observations(t, 2.0))
    m = cm.map()
    # every observation within the radius of a location counts there, so the total is at least the observations
    assert m.sum() >= 10 + 40 and m.min() >= 0.0
    on_path = (5.0 + 2.5 * math.cos(heading), 0.5 + 2.5 * math.sin(heading))
    assert _cell_bins(ROOM, m, on_path)[5] > 0.9
    assert _cell_bins(ROOM, m, (2.25, 4.25))[M.STANDING] == pytest.approx(1.0)
    assert np.allclose(cm.shares()[cm.built].sum(axis=1), 1.0)
    # no denormal tail: a JSD halves every entry, and the smallest denormals halve to 0
    assert not ((m > 0) & (m < 1e-300)).any()


def test_online_update_follows_a_reversed_flow_and_adds_its_component() -> None:
    rng = np.random.default_rng(4)
    east = [_track(f"e{i}", (0.5, 3.0), (1.2, 0.0), 9.0, rng) for i in range(6)]
    west = [_track(f"w{i}", (11.5, 3.0), (-1.2, 0.0), 9.0, rng) for i in range(6)]
    cm = CliffMap.empty(ROOM, CliffParams(decay=0.5))
    for t in east:
        cm.update(observations(t, 2.0))
    i, j, _ = ROOM.index(6.0, 3.0)
    loc = int(i) * ROOM.nx + int(j)
    before = _cell_bins(ROOM, cm.map(), (6.0, 3.0))
    assert before[3] + before[4] > 0.9  # heading 0 lies on the edge of bins 3 and 4
    for t in west:
        cm.update(observations(t, 2.0))
    after = _cell_bins(ROOM, cm.map(), (6.0, 3.0))
    assert after[0] + after[7] > 0.9  # heading pi: bins 7 and 0
    assert (cm.m[loc] > 0).sum() >= 2  # the west flow came in as a new component, not by moving the east one


def test_a_copy_updates_apart_from_its_original() -> None:
    rng = np.random.default_rng(7)
    cm = CliffMap.empty(ROOM, CliffParams())
    cm.update(observations(_track("e", (0.5, 3.0), (1.2, 0.0), 9.0, rng), 2.0))
    before = cm.map()
    twin = cm.copy()
    twin.update(observations(_track("w", (11.5, 3.0), (-1.2, 0.0), 9.0, rng), 2.0))
    assert np.array_equal(cm.map(), before) and not np.array_equal(twin.map(), before)
    assert twin.grid is cm.grid


def test_decay_one_keeps_the_history_and_low_decay_forgets_it() -> None:
    rng = np.random.default_rng(5)
    east = [_track(f"e{i}", (0.5, 3.0), (1.2, 0.0), 9.0, rng) for i in range(4)]
    west = [_track(f"w{i}", (11.5, 3.0), (-1.2, 0.0), 9.0, rng) for i in range(4)]

    def west_share(decay: float) -> float:
        cm = CliffMap.empty(ROOM, CliffParams(decay=decay))
        for t in east + west:
            cm.update(observations(t, 2.0))
        b = _cell_bins(ROOM, cm.map(), (6.0, 3.0))
        return float(b[0] + b[7])

    assert west_share(1.0) == pytest.approx(0.5, abs=0.1)
    assert west_share(0.3) > 0.85


# --- protocol ---------------------------------------------------------------------------------------------------------


def _runs(maps: dict[tuple[str, int], np.ndarray]) -> Runs:
    return Runs(maps)


def _blob(at: tuple[float, float], b: int, mass: float = 10.0, grid: M.FlowGrid = ROOM) -> np.ndarray:
    m = grid.empty()
    i, j, _ = grid.index(*at)
    m[int(i), int(j), b] = mass
    return m


def test_folds_score_against_the_other_seeds_only() -> None:
    maps = {("w__e", s): _blob((1.0 + s, 3.0), 0) for s in (6, 7, 8)}
    runs = _runs(maps)
    folds = B.folds(runs, "w__e", [6, 7, 8])
    assert [f.seen for f in folds] == [6, 7, 8]
    for f in folds:
        assert f.seen not in f.rest
        assert np.allclose(f.truth, M.pooled([maps[("w__e", s)] for s in f.rest]))
    # each seed's run is a blob of its own: one run never predicts the others
    assert B.one_seed_floor(runs, "w__e", [6, 7, 8]) == pytest.approx(1.0)
    assert B.outside(runs, "w__e", [7, 8]) == [6]


def test_controls_score_the_control_at_the_seen_seed() -> None:
    maps = {}
    for s in (6, 7, 8):
        maps[("w", s)] = _blob((1.0, 1.0), 0)
        maps[("w__edit", s)] = _blob((6.0, 3.0), 2) + _blob((1.0, 1.0), 0, 1.0)
        maps[("w__ctrl", s)] = _blob((6.0, 3.0), 2)
    doc = {"seeds": {"held_out_edits": [6, 7, 8]}, "partitions": {"held_out_edits": ["w__edit", "w__ctrl"]}, "control_worlds": {"w__edit": ["w__ctrl"]}}
    (row,) = B.controls(_runs(maps), doc, "held_out_edits")
    truth = maps[("w__edit", 7)] / maps[("w__edit", 7)].sum()
    assert row["control"] == "w__ctrl" and row["seeds"] == [6, 7, 8]
    assert row["geometry_simulation"] == pytest.approx(M.jsd(maps[("w__ctrl", 6)], truth))
    assert row["one_seed_floor"] == pytest.approx(0.0, abs=1e-9)
    assert row["stale_at_seen"] > row["geometry_simulation"]
    # the base world ran only at the scored seeds, so the runs outside a fold's truth are the seen seed's alone
    assert B.stale_outside(_runs(maps), "w__edit", [6, 7, 8]) == pytest.approx(row["stale_at_seen"])


def test_curve_feeds_the_seen_seed_and_matches_the_dirichlet_curves_track_orders() -> None:
    rng = np.random.default_rng(6)
    tracks = {s: pl.concat([_track(f"p{k}", (0.5, 1.0 + k), (1.0, 0.0), 10.0, rng) for k in range(4)]) for s in (6, 7, 8)}
    grid = ROOM
    maps = {("w__e", s): M.FlowGrid.dense(grid, t["x"].to_numpy(), t["y"].to_numpy(), np.arctan2(t["vy"].to_numpy(), t["vx"].to_numpy()), np.full(len(t), 1 / 15)) for s, t in tracks.items()}
    runs = _runs(maps)
    seen: list[tuple[int, list[str]]] = []

    class Spy:
        def start(self, world: str, fold: B.Fold) -> list[str]:
            seen.append((fold.seen, []))
            return seen[-1][1]

        def feed(self, state: list[str], track: pl.DataFrame) -> list[str]:
            state.append(str(track["key"][0]))
            return state

        def map(self, state: list[str]) -> np.ndarray | None:
            return None

    B.curve("w__e", [6, 7, 8], lambda w, s: tracks[s], runs, {"spy": Spy()}, counts=(0, 1, 4), orders=3)
    # the learned forecaster's loop: one Random(0) per world, sorted keys shuffled once per order, folds in seed order
    ref = random.Random(0)
    expected = []
    for s in (6, 7, 8):
        for _ in range(3):
            order = sorted(f"p{k}" for k in range(4))
            ref.shuffle(order)
            expected.append((s, order))
    assert seen == expected
    # a CLiFF-map fed every track of a seed lands near the one-seed floor; the empty map has no forecast at zero tracks
    cliff = B.OnlineCliff(lambda base: grid, lambda w, s: tracks[s], runs, CliffParams(decay=1.0))
    res = B.curve("w__e", [6, 7, 8], lambda w, s: tracks[s], runs, {"cliff": cliff}, counts=(0, 1, 4), orders=2)["cliff"]
    assert res[0] is None and res[1] > res[2]
    assert res[2] < B.one_seed_floor(runs, "w__e", [6, 7, 8]) + 0.2


def test_write_maps_reads_back_as_the_same_map(tmp_path: Path) -> None:
    m = _blob((6.0, 3.0), 2) + _blob((1.0, 1.0), M.STANDING, 3.0)
    path = B.write_maps(tmp_path / "maps.parquet", {"w__e": m}, lambda base: ROOM)
    cells = pl.read_parquet(path)
    assert tuple(cells.columns) == B.MAP_COLUMNS
    back = ROOM.dense(cells["x"].to_numpy(), cells["y"].to_numpy(), cells["heading_rad"].to_numpy(), cells["ped_seconds"].to_numpy())
    assert np.allclose(back, m)


# --- on hospital_1 ----------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def queue():  # noqa: ANN201
    pytest.importorskip("arena_humansim")
    from arena_simulation_setup.generative.flowedit.edit import open_site

    return open_site(HOSPITAL, "queue")


@pytest.mark.usefixtures("local_root")
def test_geometry_prior_walks_between_the_doors_and_skips_a_locked_one(queue) -> None:  # noqa: ANN001
    from arena_simulation_setup.generative.flowedit import settings as S
    from arena_simulation_setup.generative.flowedit.edit import apply

    settings = S.load({"entrances": {"rate": 0}})
    grid = M.FlowGrid.from_world(HOSPITAL)
    m, info = B.geometry_prior(queue, grid, settings, 300.0)
    assert info["kind"].startswith("every open door") and len(info["entrances"]) >= 2
    assert info["walks"] == len(info["entrances"]) * (len(info["entrances"]) - 1) - len(info["unreachable"])
    assert m.sum() > 0 and m[:, :, M.STANDING].sum() == 0.0
    assert not m[~grid.room_cells()].any()
    locked = apply(queue, "access", "door:patient_ward_door_2", {"locked": True})
    _, info2 = B.geometry_prior(locked, grid, settings, 300.0)
    assert "patient_ward_door_2" in info["entrances"] and "patient_ward_door_2" not in info2["entrances"]


# --- Deep Crowd-Flow --------------------------------------------------------------------------------------------------


def test_merge_lines_merges_identical_lines_first() -> None:
    from arena_simulation_setup.generative.flowedit.forecast.deepflow import merge_lines

    E = np.zeros((6, 4))
    E[0] = 1
    E[3, 1] = 1
    lines = merge_lines(E, side=4)
    assert lines.size == 4
    # rows 1 and 2 (open) merge, rows 4 and 5 (open) merge; the walls of rows 0 and 3 stay their own
    assert lines.group == ((0,), (1, 2), (3,), (4, 5))
    assert list(lines.index()) == [0, 1, 1, 2, 3, 3]


def test_cage_encoding_keeps_the_mass_and_decoding_spreads_it_back() -> None:
    from arena_simulation_setup.generative.flowedit.forecast.deepflow import SIDE, Cage, Lines

    rows = Lines(((0,), (1, 2), (3,)))
    cols = Lines(((0, 1), (2,)))
    c = Cage(None, rows, cols, (4, 3))
    v = np.arange(12, dtype=float).reshape(4, 3, 1)
    enc = c.encode(v)
    assert enc.shape == (SIDE, SIDE, 1)
    cap = np.array([[2, 1], [4, 2], [2, 1]])
    assert (enc[:3, :2, 0] * cap).sum() == pytest.approx(v.sum())
    dec = c.decode(enc)
    assert dec.shape == v.shape and dec.sum() == pytest.approx(v.sum())


def test_segnet_maps_cage_to_nine_rates_per_cell() -> None:
    torch = pytest.importorskip("torch")
    from arena_simulation_setup.generative.flowedit.forecast.deepflow import CHANNELS, SIDE, segnet

    net = segnet()
    net.eval()
    with torch.no_grad():
        y = net(torch.rand(1, len(CHANNELS), SIDE, SIDE))
    assert tuple(y.shape) == (1, M.BINS, SIDE, SIDE) and float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_deepflow_samples_pool_a_worlds_runs_and_train_runs_the_updates() -> None:
    torch = pytest.importorskip("torch")
    from arena_simulation_setup.generative.flowedit.forecast import deepflow as D

    grid = ROOM
    c = D.Cage(np.zeros((len(D.CHANNELS), D.SIDE, D.SIDE), np.float32), D.Lines(tuple((i,) for i in range(grid.ny))), D.Lines(tuple((j,) for j in range(grid.nx))), (grid.ny, grid.nx))
    maps = {("w", 1): _blob((1.0, 1.0), 0, 10.0), ("w", 2): _blob((6.0, 3.0), 2, 30.0), ("v", 1): _blob((2.0, 2.0), 1, 5.0)}
    found = D.samples(lambda world: c, _runs(maps), ["w", "v", "u"], [1, 2])
    assert [(f[0], f[1]) for f in found] == [("w", (1, 2)), ("v", (1,))]
    y, top = D.target(c, M.pooled([maps[("w", 1)], maps[("w", 2)]]) * 20.0)
    assert np.allclose(found[0][3], y) and found[0][4] == pytest.approx(top)
    # one sample per world: the pooled map gives both runs' cells, each with half the mass
    i1, j1, _ = grid.index(1.0, 1.0)
    i2, j2, _ = grid.index(6.0, 3.0)
    assert found[0][3][0, int(i1), int(j1)] == pytest.approx(found[0][3][2, int(i2), int(j2)])
    torch.manual_seed(0)
    model, meta = D.train(found, steps=2, device="cpu", log=lambda m: None)
    assert meta["steps"] == 2 and meta["optimizer"] == "adam" and meta["validation"] == [] and meta["kept_epoch"] is None
    out = D.forecast(model.eval(), meta, c, grid)
    assert out.shape == grid.shape and out.min() >= 0.0


@pytest.mark.usefixtures("local_root")
def test_cage_of_hospital_marks_walls_people_and_the_fountain(queue) -> None:  # noqa: ANN001
    from arena_simulation_setup.generative.flowedit import settings as S
    from arena_simulation_setup.generative.flowedit.forecast.deepflow import SIDE, cage
    from arena_simulation_setup.generative.flowedit.forecast.scene import base_people

    settings = S.load({"entrances": {"rate": 0}})
    grid = M.FlowGrid.from_world(HOSPITAL)
    people = base_people(queue, settings, (1, 2))
    c = cage(queue, grid, settings, 300.0, people)
    Cx, Cy, A, G, E = c.x
    assert c.x.shape == (5, SIDE, SIDE) and (c.rows.size, c.cols.size) == (grid.ny, grid.nx)
    assert np.all(Cx == 1.0) and np.all(Cy == 1.0)
    assert E[grid.ny :, :].min() == 1.0 and E[: grid.ny, : grid.nx].min() == 0.0
    assert A.sum() == pytest.approx(min(len(people) / 2, A.sum()), rel=0.2) and A.max() <= 1.0
    # G is 0 at the fountain's front and grows away from it
    i, j, _ = grid.index(11.0, 9.0)
    assert G[: grid.ny, : grid.nx][E[: grid.ny, : grid.nx] == 0].min() < 0.02 and G[int(i), int(j)] < 0.2


def test_cli_lists_the_baselines(capsys: pytest.CaptureFixture[str]) -> None:
    from arena_simulation_setup.generative.flowedit.cli import main

    with pytest.raises(SystemExit):
        main(["forecast", "baseline", "--help"])
    out = capsys.readouterr().out
    assert all(v in out for v in ("cliff", "controls", "geometry", "deepflow"))
