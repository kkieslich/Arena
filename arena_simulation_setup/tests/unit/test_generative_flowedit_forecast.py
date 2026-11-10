from __future__ import annotations

import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
shapely = pytest.importorskip("shapely")

from arena_simulation_setup.generative.flowedit.forecast import maps as M  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast.evaluate import anchored, transplanted  # noqa: E402
from arena_simulation_setup.generative.flowedit.forecast.model import Demand, Need, Params, Person, Scene, heading_table, layers, predict, schedule  # noqa: E402

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
THIRST = Need(initial={"mean": 40.0, "std": 0.0}, decay={"mean": 0.5, "std": 0.0}, below=30.0)
SHARP = Params(sigma=0.05, heading=0.05)


def _scene(rooms: list[shapely.Polygon], people: list[Person], demands: list[Demand] = (), walls: shapely.Geometry | None = None, duration: float = 120.0) -> Scene:
    """A toy scene: free space is the rooms shrunk by 0.3 m, minus `walls`."""
    grid = M.FlowGrid.from_polygons([list(r.exterior.coords)[:-1] for r in rooms])
    free_space = shapely.union_all(rooms).buffer(-0.3)
    if walls is not None:
        free_space = free_space.difference(walls)
    xs, ys = grid.centres(2)
    return Scene(grid=grid, free=shapely.contains_xy(free_space, xs, ys), sub=2, people=list(people), demands=list(demands), needs={"thirst": THIRST}, duration=duration)


def _near(grid: M.FlowGrid, counts: np.ndarray, point: tuple[float, float], radius: float, bins: slice = slice(None)) -> float:
    xs, ys = grid.centres()
    return float(counts[np.hypot(xs - point[0], ys - point[1]) <= radius][:, bins].sum())


ROOM = shapely.box(0, 0, 12, 6)
WALKER = Person(start=(2.0, 3.0), stops=((2.0, 3.0), (10.0, 3.0)), speed=1.2)
FOUNTAIN = Demand("fountain", "thirst", at=(6.0, 0.3), facing=math.pi / 2, dwell=5.0)


def test_forecast_is_deterministic() -> None:
    a = predict(layers(_scene([ROOM], [WALKER], [FOUNTAIN])), Params())
    b = predict(layers(_scene([ROOM], [WALKER], [FOUNTAIN])), Params())
    assert np.array_equal(a, b)


@pytest.mark.parametrize("params", [Params(), Params(sigma=1.5, walk=4.0, need=0.2), Params(sigma=0.1, walk=0.1, need=10.0, queue=5.0)])
def test_forecast_holds_the_crowds_pedestrian_seconds(params: Params) -> None:
    people = [WALKER, Person(start=(9.0, 2.0), stops=((9.0, 2.0), (3.0, 5.0), (11.0, 5.0)), speed=1.0, weight=0.5)]
    out = predict(layers(_scene([ROOM], people, [FOUNTAIN], duration=300.0)), params)
    assert out.shape == (12, 24, M.BINS) and out.min() >= 0.0
    assert out.sum() == pytest.approx(1.5 * 300.0)


def test_mass_stays_in_the_rooms() -> None:
    # an L: the bounding box's upper right quarter is no room, and a wide kernel would spill into it
    rooms = [shapely.box(0, 0, 10, 4), shapely.box(0, 4, 4, 10)]
    person = Person(start=(2.0, 8.0), stops=((2.0, 8.0), (8.0, 2.0)), speed=1.2)
    scene = _scene(rooms, [person], [Demand("f", "thirst", at=(3.7, 6.0), facing=math.pi, dwell=5.0)])
    out = predict(layers(scene), Params(sigma=2.0))
    inside = scene.grid.room_cells()
    assert (~inside).any() and out[~inside].sum() == 0.0
    assert out.sum() == pytest.approx(120.0)


def test_headings_follow_the_path_and_split_on_a_bin_edge() -> None:
    corridor = shapely.box(0, 0, 20, 2)
    person = Person(start=(2.0, 1.0), stops=((2.0, 1.0), (18.0, 1.0)), speed=1.2)
    out = predict(layers(_scene([corridor], [person])), SHARP)
    walking = out[:, :, : M.HEADINGS].sum(axis=(0, 1))
    # east (0 rad) is the edge of bins 3 and 4, west (pi) of bins 7 and 0, as arena_evaluation bins a velocity
    assert walking[[0, 3, 4, 7]].sum() / walking.sum() > 0.97
    assert all(walking[b] / walking.sum() > 0.15 for b in (0, 3, 4, 7))


def test_heading_table() -> None:
    table = heading_table(0.05)
    centre = M.bin_centre(5)
    row = table[int((centre + math.pi) / (2 * math.pi) * table.shape[0])]
    assert row[5] > 0.99
    edge = heading_table(0.3)[table.shape[0] // 2]  # just above 0 rad, the edge of bins 3 and 4
    assert edge[3] == pytest.approx(edge[4], abs=0.02) and edge[3] + edge[4] > 0.8
    assert np.allclose(table.sum(axis=1), 1.0)


def test_moving_a_demand_object_moves_its_mass() -> None:
    moved = Demand("fountain", "thirst", at=(6.0, 5.7), facing=-math.pi / 2, dwell=5.0)
    before = predict(layers(_scene([ROOM], [WALKER], [FOUNTAIN])), SHARP)
    after = predict(layers(_scene([ROOM], [WALKER], [moved])), SHARP)
    grid = M.FlowGrid.from_polygons([list(ROOM.exterior.coords)[:-1]])
    standing = slice(M.STANDING, M.STANDING + 1)
    assert _near(grid, after, moved.spot(0), 1.0, standing) > 3 * _near(grid, before, moved.spot(0), 1.0, standing)
    assert _near(grid, before, FOUNTAIN.spot(0), 1.0, standing) > 3 * _near(grid, after, FOUNTAIN.spot(0), 1.0, standing)
    assert after.sum() == pytest.approx(before.sum())


def test_objects_of_one_need_share_its_trips_by_distance() -> None:
    near = Demand("a", "thirst", at=(2.0, 0.3), facing=math.pi / 2, dwell=5.0)
    far = Demand("b", "thirst", at=(11.0, 5.7), facing=-math.pi / 2, dwell=5.0)
    person = Person(start=(2.0, 3.0), stops=((2.0, 3.0), (3.0, 4.0)), speed=1.2)
    lay = layers(_scene([ROOM], [person], [near, far]))
    from arena_simulation_setup.generative.flowedit.forecast.model import trip_shares

    sharp, soft = trip_shares(lay, Params(tau=0.1))[0], trip_shares(lay, Params(tau=50.0))[0]
    assert sharp[0] > 0.99 * sharp.sum() and 0.3 < soft[0] / soft.sum() < 0.7


def test_a_person_cut_off_from_its_stops_stands_where_it_starts() -> None:
    wall = shapely.box(5.6, -1, 6.4, 7)
    stuck = Person(start=(2.0, 3.0), stops=((9.0, 3.0), (10.0, 5.0)), speed=1.2)
    scene = _scene([ROOM], [stuck], [FOUNTAIN], walls=wall)
    out = predict(layers(scene), SHARP)
    assert _near(scene.grid, out, stuck.start, 0.5, slice(M.STANDING, M.STANDING + 1)) == pytest.approx(out.sum(), rel=1e-6)


def test_schedule_walks_the_idle_cycle_in_order() -> None:
    # first walk 10 s, then stand 8 at stop 0, walk 20 to stop 1, stand 8, walk 20 back: one cycle of 56 s and 34 s more
    cover, trips, origin = schedule(10.0, [8.0, 20.0, 8.0, 20.0], [], 100.0)
    assert cover.tolist() == pytest.approx([10.0, 16.0, 40.0, 14.0, 20.0])
    assert len(trips) == 0 and origin.shape == (0, 2)


def test_schedule_interrupts_the_cycle_for_a_trip_and_restarts_at_the_first_stop() -> None:
    # thirst 40 falls to 30 at 20 s, 2 s into the walk from stop 0 (a tenth of it); the 30 s trip ends at 50 s back at
    # stop 0, and the next trip would start at 50 + 70 / 0.5 = 190 s, after the run
    cover, trips, origin = schedule(10.0, [8.0, 20.0, 8.0, 20.0], [(THIRST, [30.0, 30.0])], 100.0)
    assert cover.tolist() == pytest.approx([10.0, 16.0, 22.0, 8.0, 14.0])
    assert cover.sum() + 30.0 == pytest.approx(100.0)
    assert trips.tolist() == pytest.approx([1.0]) and origin[0].tolist() == pytest.approx([0.9, 0.1])
    # the run's end cuts the second trip of a longer run by its share, and a need that never falls sends nobody
    assert schedule(10.0, [8.0, 20.0, 8.0, 20.0], [(THIRST, [30.0, 30.0])], 205.0)[1].tolist() == pytest.approx([1.5])
    never = Need(initial={"mean": 100.0, "std": 0.0}, decay={"mean": 0.1, "std": 0.0}, below=30.0)
    assert schedule(10.0, [8.0, 20.0], [(never, [30.0])], 300.0)[1].tolist() == [0.0]


def test_params_load_dump_and_bounds() -> None:
    p = Params(sigma=0.7, heading=0.2, walk=2.0, need=0.5, tau=3.0, queue=1.5)
    assert Params.load(p.dump()) == p
    assert Params.from_vector([100.0, 0.0, 1.0, 1.0, 1.0, 1.0]).sigma == 3.0
    with pytest.raises(ValueError, match="unknown"):
        Params.load({"kernel": 1.0})


def test_transplant_moves_copies_and_drops_the_mass_near_an_object() -> None:
    grid = M.FlowGrid.from_polygons([[(0, 0), (10, 0), (10, 10), (0, 10)]])
    base = np.zeros(grid.shape)
    i, j, _ = grid.index(2.25, 2.25)
    base[i, j, 4] = 10.0  # walking east by the object
    base[5, 15, M.STANDING] = 10.0  # far away
    moved = transplanted(base, grid, [((2.25, 2.25, 0.0), (7.25, 7.25, math.pi / 2), "move")])
    ni, nj, _ = grid.index(7.25, 7.25)
    assert moved[ni, nj, 6] == pytest.approx(10.0) and moved[i, j].sum() == 0.0  # turned a quarter: two bins on
    copied = transplanted(base, grid, [((2.25, 2.25, 0.0), (7.25, 7.25, 0.0), "copy")])
    assert copied.sum() == pytest.approx(base.sum()) and copied[ni, nj, 4] == pytest.approx(copied[i, j, 4])
    dropped = transplanted(base, grid, [((2.25, 2.25, 0.0), None, "drop")])
    assert dropped[i, j].sum() == 0.0 and dropped.sum() == pytest.approx(base.sum())


def test_an_anchored_forecast_without_change_is_the_base_map() -> None:
    rng = np.random.default_rng(1)
    base, forecast = rng.random((4, 5, M.BINS)), rng.random((4, 5, M.BINS))
    assert np.allclose(anchored(base, forecast, forecast), base)


def test_maps_match_arena_evaluation() -> None:
    flow = pytest.importorskip("arena_evaluation.processing.flow")
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    p, q = rng.random((5, 6, M.BINS)), rng.random((5, 6, M.BINS))
    p[0, 0] = 0.0
    assert M.jsd(p, q) == pytest.approx(flow.jsd(p, q))
    assert np.allclose(M.pooled([p, q]), flow.pooled([p, q]))
    assert np.allclose(M.occupancy(p), flow.occupancy(p))
    polygons = [[(0.0, 0.0), (4.2, 0.0), (4.2, 3.1), (0.0, 3.1)], [(4.2, 1.0), (6.0, 1.0), (6.0, 2.0), (4.2, 2.0)]]
    ours, theirs = M.FlowGrid.from_polygons(polygons), flow.FlowGrid.from_polygons(polygons)
    assert (ours.x0, ours.y0, ours.nx, ours.ny, ours.cell) == (theirs.x0, theirs.y0, theirs.nx, theirs.ny, theirs.cell)
    assert M.HEADINGS == flow.HEADINGS
    # every heading, the bin edges among them, standing too: our dense map of their cells is their map
    angles = np.linspace(-math.pi, math.pi, 33)
    n = len(angles)
    tracks = pl.DataFrame({"key": ["a"] * n, "t": np.arange(n) * 0.1, "x": rng.uniform(0.1, 5.9, n), "y": rng.uniform(1.1, 1.9, n), "vx": np.cos(angles) * np.r_[np.ones(n - 3), 0.1, 0.1, 0.1], "vy": np.sin(angles) * np.r_[np.ones(n - 3), 0.1, 0.1, 0.1]})
    counts = theirs.counts(tracks)
    cells = theirs.cells(counts)
    dense = ours.dense(cells["x"].to_numpy(), cells["y"].to_numpy(), cells["heading_rad"].to_numpy(), cells["ped_seconds"].to_numpy())
    assert np.allclose(dense, counts)
    assert np.array_equal(M.heading_bin(angles), np.floor((angles + math.pi) / (2 * math.pi) * flow.HEADINGS).astype(int) % flow.HEADINGS)
    rows = ours.rows(counts)
    assert np.allclose(ours.dense(np.array(rows["x"]), np.array(rows["y"]), rows["heading_rad"], np.array(rows["ped_seconds"])), counts)


# --- on hospital_1: the reader's people and the graph's demand --------------------------------------------------------


@pytest.fixture(scope="module")
def queue():  # noqa: ANN201
    pytest.importorskip("arena_humansim")
    from arena_simulation_setup.generative.flowedit.edit import open_site

    return open_site(HOSPITAL, "queue")


@pytest.mark.usefixtures("local_root")
def test_people_are_the_readers_people(queue) -> None:  # noqa: ANN001
    from arena_simulation_setup.generative.flowedit import settings as S
    from arena_simulation_setup.generative.flowedit.forecast.scene import base_people
    from arena_simulation_setup.generative.flowedit.read import read

    settings = S.load({"entrances": {"rate": 0}})
    people = base_people(queue, settings, (3,))
    result = read(queue, settings, seed=3)
    assert len(people) == len(result.doc["humans"]) > 0
    for person, human in zip(people, result.doc["humans"], strict=True):
        steps = result.types[human["agent"]["agent_type"]]["sequences"]["idle"]["steps"].values()
        assert person.start == tuple(human["pose"][:2])
        assert person.stops == tuple((s["target_pose"]["x"], s["target_pose"]["y"]) for s in steps)
        assert person.speed == human["agent"]["desired_velocity"] and person.weight == 1.0


@pytest.mark.usefixtures("local_root")
def test_hospital_forecast_follows_the_moved_fountain_and_a_locked_door(queue) -> None:  # noqa: ANN001
    from arena_simulation_setup.generative.flowedit import settings as S
    from arena_simulation_setup.generative.flowedit.edit import apply
    from arena_simulation_setup.generative.flowedit.forecast.scene import base_people, scene_of

    settings = S.load({"entrances": {"rate": 0}})
    grid = M.FlowGrid.from_world(HOSPITAL)
    people = base_people(queue, settings, (1, 2))
    base = scene_of(queue, settings, 300.0, grid=grid, people=people)
    (fountain,) = base.demands
    assert fountain.name == "fountain" and fountain.at == (11.0, 10.0) and fountain.dwell == 5.0
    moved = scene_of(apply(queue, "move", "obj:fountain", {"x": 15.0, "y": 11.5}), settings, 300.0, grid=grid, people=people)
    assert moved.demands[0].at == (15.0, 11.5)
    before, after = predict(layers(base), Params()), predict(layers(moved), Params())
    standing = slice(M.STANDING, M.STANDING + 1)
    spot = moved.demands[0].spot(0)
    assert _near(grid, after, spot, 1.0, standing) > 2 * _near(grid, before, spot, 1.0, standing)
    locked = scene_of(apply(queue, "access", "door:patient_ward_door_2", {"locked": True}), settings, 300.0, grid=grid, people=people)
    door = shapely.LineString([(8.0, 20.35), (8.0, 21.85)]).buffer(0.3)
    xs, ys = grid.centres(2)
    at_door = shapely.contains_xy(door, xs, ys)
    assert base.free[at_door].any() and not locked.free[at_door].any()
    assert any("patient_ward_door_2" in n for n in locked.notes)


@pytest.mark.usefixtures("local_root")
def test_cli_predicts_fits_and_scores(queue, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: ANN001
    import yaml

    pl = pytest.importorskip("polars")
    from arena_simulation_setup.generative.flowedit.cli import main

    monkeypatch.setenv("ARENA_WORLD_PATH", str(WORLDS))
    out = tmp_path / "forecast.parquet"
    assert main(["forecast", "predict", "hospital_1", "--scenario", "queue", "--duration", "60", "--out", str(out)]) == 0
    rows = pl.read_parquet(out)
    assert rows["ped_seconds"].sum() > 0 and set(rows["flow_world"]) == {"hospital_1"}
    assert "thirst: obj:fountain" in capsys.readouterr().out
    # a "recording" that is the default forecast itself: the fit starts there, and the score finds it
    split = {"format": "flowedit-split/1", "sources": {"hospital_1": {"scenario": "queue", "seed": 0, "per_type": 1, "steps": 1}}, "duration": 60, "seeds": [1], "partitions": {"train": ["hospital_1"], "held_out_edits": [], "held_out_worlds": []}}
    (tmp_path / "split.yaml").write_text(yaml.safe_dump(split))
    cells = rows.with_columns(pl.lit("peds_only").alias("planner"), pl.lit("hospital_1__s1").alias("stage"), pl.lit("hospital_1").alias("flow_base"), pl.lit(1).alias("flow_seed"))
    cells.write_parquet(tmp_path / "flow_cells.parquet")
    params = tmp_path / "params.yaml"
    assert main(["forecast", "fit", str(tmp_path / "split.yaml"), "--cells", str(tmp_path / "flow_cells.parquet"), "--max-evals", "8", "--out", str(params)]) == 0
    fitted = yaml.safe_load(params.read_text())
    assert fitted["format"] == "flowedit-forecast/1" and fitted["fit"]["mean_jsd"] <= fitted["fit"]["init_mean_jsd"] < 0.02
    capsys.readouterr()
    assert main(["forecast", "score", str(tmp_path / "split.yaml"), "--cells", str(tmp_path / "flow_cells.parquet"), "--params", str(params), "--predicted", f"recorded={out}", "--partition", "train", "--out", str(tmp_path / "score.yaml")]) == 0
    assert "| hospital_1 | base |" in capsys.readouterr().out
    (row,) = yaml.safe_load((tmp_path / "score.yaml").read_text())["partitions"]["train"]["rows"]
    assert row["demand model"] < 0.02 and row["recorded"] == 0.0 and row["stale"] is None
