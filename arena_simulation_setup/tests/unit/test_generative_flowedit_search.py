from __future__ import annotations

import itertools
import json
import math
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import place, search
from arena_simulation_setup.generative.flowedit.edit import Site, open_site
from arena_simulation_setup.generative.flowedit.place import Layout
from arena_simulation_setup.generative.flowedit.search import PIN_SNAP_M, SEARCHED, Occupancy, Search, TaskMix

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
shapely = pytest.importorskip("shapely")

HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"


@pytest.fixture(scope="module")
def assets(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """The docked element models under a module-wide local root, as the placement tests write them per test."""
    from flowedit_docks import write_docked_assets

    root = tmp_path_factory.mktemp("local")
    previous = os.environ.get("ARENA_ASSETS_DIR_LOCAL")
    os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(root)
    write_docked_assets(root)
    try:
        yield root
    finally:
        if previous is None:
            del os.environ["ARENA_ASSETS_DIR_LOCAL"]
        else:
            os.environ["ARENA_ASSETS_DIR_LOCAL"] = previous


@pytest.fixture(scope="module")
def hospital(assets: Path) -> Site:
    return open_site(HOSPITAL)


@pytest.fixture(scope="module")
def loading(hospital: Site) -> place.Spot:
    """The first valid loading point spot of the central hallway."""
    return next(s for s in place.candidate_spots(hospital, "loading_point") if s.valid and s.zone == "central_hallway")


@pytest.fixture(scope="module")
def base(hospital: Site, loading: place.Spot):  # noqa: ANN201 - a NavrBase
    """hospital_1's NAV@r base without its furniture, whose models the tests' fake assets do not have (a base refuses
    objects that do not resolve)."""
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    level = hospital.world.levels[loading.level]
    return NavrBase(level, exclude={o.name for o in level.objects})


@pytest.fixture(scope="module")
def space(hospital: Site, loading: place.Spot, base) -> Search:  # noqa: ANN001
    return Search(hospital, loading=loading, base=base)


def _corridor_level(tmp_path: Path):  # noqa: ANN202 - a World Level
    """A 10 x 4 m room cut by a wall from (5, 0) to (5, 3), so (2.5, 1) and (7.5, 1) are 5 m apart but the floor
    between them runs round the wall's end."""
    from arena_simulation_setup.tree.World import FORMAT, WorldView

    root = tmp_path / "cut"
    root.mkdir()
    level = {
        "id": "0",
        "zones": [{"name": "room", "polygon": [[0, 0], [10, 0], [10, 4], [0, 4]]}],
        "walls": [{"path": [[0, 0], [10, 0], [10, 4], [0, 4]], "closed": True}, {"path": [[5, 0], [5, 3]]}],
    }
    (root / "world.yaml").write_text(yaml.safe_dump({"format": FORMAT, "levels": [level]}))
    return WorldView(root).load(validate=False).levels["0"]


def test_paths_run_round_obstacles_on_the_jackals_floor(tmp_path: Path) -> None:
    from arena_simulation_setup.metrics.navr_filter import JACKAL_RADIUS, NavrBase

    base = NavrBase(_corridor_level(tmp_path))
    paths = search.RobotPaths.from_base(base)
    d = paths.lengths([(2.5, 1.0), (7.5, 1.0)], [(2.5, 1.0), (7.5, 1.0), (2.5, 3.0)])
    assert d[0, 0] < 0.1 and d[1, 1] < 0.1
    assert d[0, 1] == pytest.approx(d[1, 0], abs=1e-6)
    # round the wall's end at (5, 3), kept r off it: at least the taut string, at most 8 % over it (8-connected cells)
    taut = 2 * math.hypot(2.5, 2.0 + JACKAL_RADIUS)
    assert taut - 0.1 < d[0, 1] < 1.08 * taut + 0.1
    assert d[0, 2] == pytest.approx(2.0, abs=0.15)  # straight up the room, on the grid's axis
    assert np.isinf(paths.lengths([(2.5, 1.0)], [(20.0, 1.0)])).all()  # off the floor
    coarse = search.RobotPaths.from_base(base, resolution=0.1)
    assert coarse.cells < paths.cells / 3
    assert coarse.lengths([(2.5, 1.0)], [(7.5, 1.0)])[0, 0] == pytest.approx(d[0, 1], rel=0.05)


def test_the_search_space_holds_valid_spots_compatible_with_the_loading_point(hospital: Site, loading: place.Spot, space: Search) -> None:
    assert space.loading is loading
    for name in SEARCHED:
        assert space.spots[name], name
        for s in space.spots[name]:
            assert s.valid and s.level == loading.level
            assert not place.check_layout(hospital, Layout.of([loading, s]))
        found = place.candidate_spots(hospital, name)
        dropped = space.dropped[name]
        assert dropped["spots"] == len(found)
        assert len(space.spots[name]) == dropped["spots"] - sum(v for k, v in dropped.items() if k != "spots")


def test_candidates_are_exactly_the_pairwise_compatible_spot_triples(hospital: Site, space: Search) -> None:
    def ok(a: place.Spot, b: place.Spot) -> bool:
        return not place.check_layout(hospital, Layout.of([a, b]))

    chargers, bays, lockers = (space.spots[name] for name in SEARCHED)
    expected = {(c, b, k) for (c, sc), (b, sb), (k, sk) in itertools.product(enumerate(chargers), enumerate(bays), enumerate(lockers)) if ok(sc, sb) and ok(sc, sk) and ok(sb, sk)}
    assert {tuple(int(v) for v in row) for row in space.candidates} == expected
    assert len(space.candidates) == len(expected)


def test_a_far_pair_needs_no_check_because_check_layout_passes_it(hospital: Site, space: Search) -> None:
    far = [(a, b) for a in space.spots["charger_pair"] for b in place.candidate_spots(hospital, "locker") if b.valid and not search._reach(a).intersects(search._reach(b))]
    assert far
    for a, b in far:
        assert not place.check_layout(hospital, Layout.of([a, b]))


def test_layouts_carry_the_loading_point_and_their_ids_stay_the_same(hospital: Site, loading: place.Spot, space: Search) -> None:
    again = Search(hospital, loading=loading.objects[0].goto)
    assert again.loading.id == loading.id
    assert (again.candidates == space.candidates).all()
    for index in range(len(space.candidates)):
        layout = space.layout(index)
        assert [s.element.name for s in layout.spots] == list(place.CATALOG)
        assert layout.layout_id == again.layout(index).layout_id
    assert len({space.layout(i).layout_id for i in range(len(space.candidates))}) == len(space.candidates)


def test_a_loading_point_far_from_every_valid_spot_is_refused(hospital: Site) -> None:
    with pytest.raises(ValueError, match="nearest valid loading_point spot"):
        Search(hospital, loading=(-100.0, -100.0))


def test_a_loading_point_marked_behind_its_bay_snaps_to_it(hospital: Site, loading: place.Spot) -> None:
    import math

    bay = loading.objects[0]
    far = max(bay.footprint.exterior.coords, key=lambda c: math.dist(c, bay.goto))
    away = math.dist(far, bay.goto)
    behind = (far[0] + (far[0] - bay.goto[0]) / away * 0.2, far[1] + (far[1] - bay.goto[1]) / away * 0.2)
    assert math.dist(behind, bay.goto) > PIN_SNAP_M
    assert Search(hospital, loading=behind).loading.id == loading.id


def test_pins_fix_a_searched_element_to_one_spot(hospital: Site, loading: place.Spot, space: Search) -> None:
    locker = space.spots["locker"][3]
    pinned = Search(hospital, loading=loading, pins={"locker": locker.objects[0].goto}, base=space.base)
    assert [s.id for s in pinned.spots["locker"]] == [locker.id]
    assert set(pinned.candidates[:, 2].tolist()) == {0}
    assert locker.id in {s.id for s in search.robot_only(pinned).layout.spots}
    with pytest.raises(ValueError, match="pins name"):
        Search(hospital, loading=loading, pins={"loading_point": loading})


def test_keep_out_polygons_drop_the_spots_they_touch(hospital: Site, loading: place.Spot, space: Search) -> None:
    locker = space.spots["locker"][0]
    kept = Search(hospital, loading=loading, keep_out=[locker.clear.buffer(0.1)])
    assert locker.id not in {s.id for s in kept.spots["locker"]}
    assert kept.dropped["locker"]["keep_out"] >= 1
    assert len(kept.candidates) < len(space.candidates)


def test_the_full_check_is_check_layout_and_the_navr_filter(hospital: Site, space: Search) -> None:
    from arena_simulation_setup.metrics.navr_filter import navr_check

    for index in range(len(space.candidates)):
        checked = space.check(index)
        layout = space.layout(index)
        assert checked.problems == tuple(place.check_layout(hospital, layout))
        assert checked.navr.ok == navr_check(space.base, place.placed_elements(layout)).ok
        assert checked.valid == (not checked.problems and checked.navr.ok)


def test_task_mix_weights_follow_the_job_cycle() -> None:
    w = TaskMix(rate_per_h=12.0, pickup_share=0.0, robots=4, episode_s=1500.0, charges_per_robot=0.0).weights()
    assert w == {"L_bay": 0.5, "L_charger": 0.5, "K_bay": 0.5, "K_charger": 0.5}  # deliveries home -> L -> K -> home
    mix = TaskMix(rate_per_h=24.0, pickup_share=0.25, robots=4, episode_s=1500.0, charges_per_robot=1.0)
    assert mix.charges_per_job == pytest.approx(4 / 10)
    w = mix.weights()
    assert w["L_bay"] + w["L_charger"] == pytest.approx(1.0) and w["K_bay"] + w["K_charger"] == pytest.approx(1.0)
    assert w["K_charger"] > w["L_charger"]  # most jobs are deliveries, which end at the locker before a charge
    assert TaskMix(rate_per_h=1.0, charges_per_robot=1.0).charges_per_job == 1.0


def test_travel_is_the_job_cycle_over_the_path_lengths(space: Search) -> None:
    index = len(space.candidates) // 2
    layout = space.layout(index)
    goto = {o.name: o.goto for o in layout.objects}
    homes = [goto[h] for h in search.HOMES]
    d = space.paths.lengths([goto["loading_point"], goto["locker_1"]], homes)
    lk = space.paths.lengths([goto["loading_point"]], [goto["locker_1"]])[0, 0]
    p, q = space.mix.pickup_share, space.mix.charges_per_job
    home_l, home_k = d[0].mean(), d[1].mean()
    charger_l, charger_k = d[0, 2:].mean(), d[1, 2:].mean()
    delivery = home_l + lk + (1 - q) * home_k + q * charger_k
    pickup = home_k + lk + (1 - q) * home_l + q * charger_l
    assert space.travel(index)[0] == pytest.approx((1 - p) * delivery + p * pickup, rel=1e-6)


def test_robot_only_picks_the_valid_layout_with_the_shortest_travel(space: Search) -> None:
    pick = search.robot_only(space)
    travel = space.travel()
    best = min((float(travel[i]), i) for i in range(len(space.candidates)) if space.valid(i))
    assert (pick.travel_m, pick.index) == pytest.approx(best)
    assert pick.score == pick.travel_m and pick.navr.ok
    assert search.robot_only(space).layout_id == pick.layout_id


def test_robot_only_weighs_travel_with_its_task_mix(hospital: Site, loading: place.Spot, base) -> None:  # noqa: ANN001
    for mix in (TaskMix(charges_per_robot=0.0), TaskMix(rate_per_h=4.0, pickup_share=0.9)):
        space = Search(hospital, loading=loading, mix=mix, base=base)
        pick = search.robot_only(space)
        assert pick.travel_m == pytest.approx(min(float(space.travel(i)[0]) for i in range(len(space.candidates)) if space.valid(i)))


def test_random_valid_layouts_are_valid_distinct_seeded_and_exclude(space: Search) -> None:
    first = search.random_valid(space, 5, seed=3)
    assert len(first) == 5 and len({s.layout_id for s in first}) == 5
    assert all(space.valid(s.index) for s in first)
    assert [s.layout_id for s in search.random_valid(space, 5, seed=3)] == [s.layout_id for s in first]
    assert [s.layout_id for s in search.random_valid(space, 5, seed=4)] != [s.layout_id for s in first]
    skip = {s.layout_id for s in first[:2]}
    assert not skip & {s.layout_id for s in search.random_valid(space, 5, seed=3, exclude=skip)}
    every = search.random_valid(space, len(space.candidates) + 10, seed=0)
    assert len(every) == sum(space.valid(i) for i in range(len(space.candidates)))


def test_occupancy_integrates_the_density_of_flow_cells() -> None:
    # two 0.5 m cells, one with two heading rows; 100 s of recording in all
    occ = Occupancy.from_cells([0.25, 0.25, 0.75], [0.25, 0.25, 0.25], [10.0, 15.0, 5.0], 0.5, 100.0)
    assert occ.density.shape == (1, 2)
    assert occ.density[0, 0] == pytest.approx(25.0 / 100.0 / 0.25)  # 1 pedestrian per m^2 on average
    assert occ.people(shapely.box(0.0, 0.0, 0.5, 0.5)) == pytest.approx(0.25)
    assert occ.people(shapely.box(0.0, 0.0, 0.25, 0.5)) == pytest.approx(0.125)
    assert occ.people(shapely.box(0.0, 0.0, 1.0, 0.5)) == pytest.approx(0.25 + 0.05)
    assert occ.people(shapely.box(5.0, 5.0, 6.0, 6.0)) == 0.0
    assert occ.mean_density == pytest.approx((1.0 + 0.2) / 2)
    with pytest.raises(ValueError):
        Occupancy.from_cells([0.25], [0.25], [1.0], 0.5, 0.0)


def test_occupancy_reads_a_worlds_runs_from_arena_evaluations_flow_cells(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rows = [
        {"planner": "peds_only", "stage": "site__s1", "flow_world": "site", "flow_base": "site", "flow_seed": 1, "x": 0.25, "y": 0.25, "cell_m": 0.5, "heading_rad": 0.0, "ped_seconds": 30.0},
        {"planner": "peds_only", "stage": "site__s2", "flow_world": "site", "flow_base": "site", "flow_seed": 2, "x": 0.25, "y": 0.25, "cell_m": 0.5, "heading_rad": None, "ped_seconds": 10.0},
        {"planner": "peds_only", "stage": "site__layout_x__s1", "flow_world": "site__layout_x", "flow_base": "site", "flow_seed": 1, "x": 0.25, "y": 0.25, "cell_m": 0.5, "heading_rad": 0.0, "ped_seconds": 99.0},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "flow_cells.parquet")
    occ = Occupancy.from_flow_cells(tmp_path, "site", episode_s=20.0)
    assert occ.density[0, 0] == pytest.approx(40.0 / (2 * 20.0) / 0.25)
    with pytest.raises(ValueError, match="no flow cells"):
        Occupancy.from_flow_cells(tmp_path / "flow_cells.parquet", "other", episode_s=20.0)


def test_joint_ranks_sampled_layouts_by_the_pedestrians_under_their_elements(space: Search) -> None:
    crowded = space.spots["locker"][0]
    # a crowd on the cells under the first locker spot's footprint, dock and access zone only
    x0, y0, x1, y1 = search.approach_area(crowded).bounds
    xs, ys = np.meshgrid(np.arange(math.floor(x0 * 2) / 2 + 0.25, x1, 0.5), np.arange(math.floor(y0 * 2) / 2 + 0.25, y1, 0.5))
    occ = Occupancy.from_cells(xs.ravel(), ys.ravel(), np.full(xs.size, 100.0), 0.5, 100.0)
    ranked = search.joint(space, occ, n=len(space.candidates), top=len(space.candidates), seed=0)
    assert len(ranked) == sum(space.valid(i) for i in range(len(space.candidates)))
    assert [r.score for r in ranked] == sorted(r.score for r in ranked)
    for r in ranked:
        assert r.occupancy == pytest.approx(sum(occ.people(search.approach_area(s)) for s in r.layout.spots))
        assert r.score == pytest.approx(r.occupancy + occ.mean_density * r.floor_loss_m2)
    under = occ.people(search.approach_area(crowded))
    assert under > 0
    assert crowded.id not in {s.id for s in ranked[0].layout.spots} and ranked[0].score < under
    assert all(r.score >= under - 1e-9 for r in ranked if crowded.id in {s.id for s in r.layout.spots})
    top = search.joint(space, occ, n=20, top=3, seed=5)
    assert len(top) == 3 and [r.layout_id for r in top] == [r.layout_id for r in search.joint(space, occ, n=20, top=3, seed=5)]


def test_scored_layouts_are_plain_data(space: Search) -> None:
    record = search.robot_only(space).record()
    assert json.loads(json.dumps(record)) == record
    assert [s["element"] for s in record["spots"]] == list(place.CATALOG)
    assert {o["name"] for s in record["spots"] for o in s["objects"]} == {"charger_1", "charger_2", "bay_1", "bay_2", "locker_1", "loading_point"}


def test_the_site_facts_filter_the_spots_every_pick_draws_from(hospital: Site, loading: place.Spot, base, space: Search) -> None:  # noqa: ANN001
    facts = place.SiteFacts((place.Entrance("main", "building", ((2.0, 1.5),), recipients=True),))
    ruled = Search(hospital, loading=loading, base=base, facts=facts)
    assert 0 < len(ruled.spots["locker"]) < len(space.spots["locker"])
    assert all(s.verdict[place.Rule.RECIPIENT_DISTANCE] is place.Verdict.PASS for s in ruled.spots["locker"])
    assert ruled.dropped["locker"]["invalid"] > space.dropped["locker"]["invalid"]
    assert [s.id for s in ruled.spots["charger_pair"]] == [s.id for s in space.spots["charger_pair"]]
    pick = search.robot_only(ruled)
    locker = next(s for s in pick.layout.spots if s.element.name == "locker")
    assert locker.id in {s.id for s in ruled.spots["locker"]}


def test_placements_keep_each_element_in_its_zones_areas_or_pin(hospital: Site, loading: place.Spot, base, space: Search) -> None:  # noqa: ANN001
    from arena_simulation_setup.generative.flowedit.search import Placement

    zones = {s.zone for s in space.spots["locker"]}
    assert len(zones) > 1
    zone = sorted(zones)[0]
    inside = Search(hospital, loading=loading, base=base, placements={"locker": Placement(zones=(zone,))})
    assert inside.spots["locker"] and {s.zone for s in inside.spots["locker"]} == {zone}
    assert inside.spots["charger_pair"] == space.spots["charger_pair"]
    # a pattern matches zone names
    pattern = Search(hospital, loading=loading, base=base, placements={"locker": Placement(zones=(zone[:3] + "*",))})
    assert {s.zone for s in pattern.spots["locker"]} == {z for z in zones if z.startswith(zone[:3])}
    # an area keeps the spots whose every object stands in it
    first = space.spots["bay_pair"][0]
    box = shapely.box(*shapely.union_all([o.footprint for o in first.objects]).buffer(0.5).bounds)
    area = Search(hospital, loading=loading, base=base, placements={"bay_pair": Placement(areas=(box,))})
    assert first in area.spots["bay_pair"] and all(all(box.covers(shapely.Point(o.x, o.y)) for o in s.objects) for s in area.spots["bay_pair"])
    assert 0 < len(area.spots["bay_pair"]) < len(space.spots["bay_pair"]) and area.dropped["bay_pair"]["outside_areas"] > 0
    # the loading point and a searched element pinned by their placements
    locker = space.spots["locker"][3]
    pinned = Search(hospital, base=base, placements={"loading_point": Placement(pin=loading.objects[0].goto), "locker": Placement(pin=locker.objects[0].goto)})
    assert pinned.loading.id == loading.id and pinned.spots["locker"] == (locker,)
    with pytest.raises(ValueError, match="no zone of hospital_1 matches"):
        Search(hospital, loading=loading, base=base, placements={"locker": Placement(zones=("no_such_zone*",))})
    with pytest.raises(ValueError, match="no loading point"):
        Search(hospital, base=base)
    with pytest.raises(ValueError, match="not elements of a layout"):
        Search(hospital, loading=loading, base=base, placements={"fountain": Placement()})


def test_the_joint_search_ranks_only_layouts_the_travel_prefilter_keeps(space: Search) -> None:
    robot = search.robot_only(space)
    cap = search.travel_cap(robot)
    assert cap == pytest.approx(search.TRAVEL_SLACK * robot.travel_m)
    within = space.within_travel(cap)
    travel = space.travel()
    assert len(within) and np.array_equal(within, np.flatnonzero(travel <= cap))
    occupancy = Occupancy(0.0, 0.0, 1.0, np.full((40, 40), 0.1))
    ranked = search.joint(space, occupancy, n=50, top=5, seed=3, max_travel_m=cap)
    kept = set(within.tolist())
    assert ranked and all(s.travel_m <= cap + 1e-9 and s.index in kept for s in ranked)
    assert [s.score for s in ranked] == sorted(s.score for s in ranked)
    assert all(s.score == pytest.approx(search.ranking_score(space, occupancy, s.index)) for s in ranked)
    # without the cap the sample reaches layouts the cap drops
    loose = search.joint(space, occupancy, n=200, top=200, seed=3)
    assert max(s.travel_m for s in loose) > cap
    # sampling among a set draws only from it, in a seeded order
    assert set(space.sample(10, 1, among=within[:30])) <= set(within[:30].tolist())
    assert space.sample(10, 1, among=within[:30]) == space.sample(10, 1, among=within[:30])


def test_the_premise_pair_puts_every_element_at_the_narrowest_and_at_the_widest_section(space: Search) -> None:
    narrow, wide = search.premise_pair(space)
    assert narrow.layout_id != wide.layout_id
    assert space.valid(narrow.index) and space.valid(wide.index)
    narrowest, widest = space.corridor()
    travel = space.travel()
    # every candidate sorted before a pick is invalid: none has a narrower widest corridor (or a wider narrowest one)
    for index in np.flatnonzero(widest < widest[narrow.index] - 1e-9)[:300]:
        assert not (math.isfinite(travel[index]) and space.valid(int(index)))
    for index in np.flatnonzero(narrowest > narrowest[wide.index] + 1e-9)[:300]:
        assert not (math.isfinite(travel[index]) and space.valid(int(index)))
    assert widest[narrow.index] <= narrowest[wide.index]
    spots = {s.element.name: s for s in narrow.layout.spots}
    assert widest[narrow.index] == pytest.approx(max(spots[name].corridor_m for name in SEARCHED))


def test_a_pinned_element_in_the_keep_out_is_refused(hospital: Site, loading: place.Spot, base, space: Search) -> None:  # noqa: ANN001
    over_loading = shapely.Point(loading.objects[0].goto).buffer(0.3)
    with pytest.raises(ValueError, match="the pinned loading_point .* overlaps a PEDS source or sink"):
        Search(hospital, loading=loading, base=base, keep_out=[over_loading])
    locker = space.spots["locker"][2]
    with pytest.raises(ValueError, match="the pinned locker"):
        Search(hospital, loading=loading, base=base, pins={"locker": locker}, keep_out=[shapely.Point(locker.objects[0].goto).buffer(0.3)])


# --- conflict exposure, top-up trips, distinct top, guideline ----------------------------------------------------------


def test_path_integrals_sum_the_values_along_the_shortest_paths(tmp_path: Path) -> None:
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    paths = search.RobotPaths.from_base(NavrBase(_corridor_level(tmp_path)))
    sources, targets = [(2.5, 1.0), (7.5, 1.0)], [(7.5, 1.0), (2.5, 3.0), (20.0, 1.0)]
    ones = np.ones((1, paths.cells))
    lengths, integrals = paths.integrals(sources, targets, ones)
    assert np.allclose(lengths[np.isfinite(lengths)], paths.lengths(sources, targets)[np.isfinite(lengths)])
    assert np.allclose(integrals[0][np.isfinite(lengths)], lengths[np.isfinite(lengths)])  # a value of 1 integrates to the length
    assert np.isinf(lengths[:, 2]).all() and np.isinf(integrals[0][:, 2]).all()  # off the floor
    # the right half of the room holds the value: the path round the wall's end spends about half its length there
    right = (paths.centres[:, 0] > 5.0).astype(float)[None, :]
    _, part = paths.integrals([(2.5, 1.0)], [(7.5, 1.0)], np.vstack([right, 2.0 * right]))
    assert 0.4 * lengths[0, 0] < part[0, 0, 0] < 0.6 * lengths[0, 0] and part[1, 0, 0] == pytest.approx(2.0 * part[0, 0, 0])
    with pytest.raises(ValueError, match="cells per row"):
        paths.integrals(sources, targets, np.ones((1, 3)))


def test_the_band_counts_the_walkers_within_its_radius() -> None:
    occ = Occupancy(0.0, 0.0, 0.5, np.full((80, 80), 0.2))  # 0.2 walkers per m^2 on 40 x 40 m
    assert occ.present == pytest.approx(0.2 * 40 * 40)
    band = occ.band(2.0)
    assert band[40, 40] == pytest.approx(0.2 * math.pi * 4.0, rel=1e-9)
    assert band[0, 0] < band[40, 40] / 2  # a corner cell sees little more than a quarter disc
    assert occ.sample(band, np.array([[20.1, 20.1], [-5.0, 3.0]])).tolist() == [pytest.approx(band[40, 40]), 0.0]


def test_group_cells_give_an_occupancy_per_group(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rows = [
        {"world": "site", "stage": "site__s1", "planner": "peds_only", "group": "adult", "x": 0.25, "y": 0.25, "cell_m": 0.5, "ped_seconds": 40.0},
        {"world": "site", "stage": "site__s2", "planner": "peds_only", "group": "adult", "x": 0.75, "y": 0.25, "cell_m": 0.5, "ped_seconds": 20.0},
        {"world": "site", "stage": "site__s1", "planner": "peds_only", "group": "wheelchair", "x": 0.25, "y": 0.25, "cell_m": 0.5, "ped_seconds": 10.0},
        {"world": "site", "stage": "site__s1", "planner": "peds_only", "group": "long_cane", "x": 0.25, "y": 0.25, "cell_m": 0.5, "ped_seconds": 0.0},
        {"world": "other", "stage": "other__s1", "planner": "peds_only", "group": "adult", "x": 0.25, "y": 0.25, "cell_m": 0.5, "ped_seconds": 99.0},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "infra_occupancy.parquet")
    groups = Occupancy.from_group_cells(tmp_path, "site", episode_s=20.0)
    assert sorted(groups) == ["adult", "wheelchair"]  # no walker time, no group
    assert groups["adult"].present == pytest.approx(60.0 / (2 * 20.0)) and groups["wheelchair"].present == pytest.approx(10.0 / 40.0)
    with pytest.raises(ValueError, match="no group cells"):
        Occupancy.from_group_cells([tmp_path / "infra_occupancy.parquet"], "nowhere", episode_s=20.0)


def test_top_ups_add_round_trips_between_bays_and_chargers(hospital: Site, loading: place.Spot, base) -> None:  # noqa: ANN001
    plain = Search(hospital, loading=loading, base=base, mix=TaskMix(rate_per_h=6.0))
    topped = Search(hospital, loading=loading, base=base, mix=TaskMix(rate_per_h=6.0, top_ups_per_job=0.5))
    rows = plain.candidates
    assert np.allclose(topped.travel() - plain.travel(), 2 * 0.5 * plain.bay_charger[rows[:, 0], rows[:, 1]])
    index = len(rows) // 3
    chargers = [o.goto for o in plain.spots[SEARCHED[0]][rows[index, 0]].objects]
    bays = [o.goto for o in plain.spots[SEARCHED[1]][rows[index, 1]].objects]
    assert plain.bay_charger[rows[index, 0], rows[index, 1]] == pytest.approx(plain.paths.lengths(chargers, bays).mean())
    with pytest.raises(ValueError, match="top_ups_per_job"):
        TaskMix(top_ups_per_job=-1.0)


def test_conflict_exposure_on_a_uniform_crowd_follows_robot_travel(hospital: Site, loading: place.Spot, base) -> None:  # noqa: ANN001
    space = Search(hospital, loading=loading, base=base, mix=TaskMix(rate_per_h=6.0, top_ups_per_job=0.3))
    occ = Occupancy(-20.0, -20.0, 0.5, np.full((200, 200), 0.1))  # 0.1 walkers per m^2 everywhere on the level
    per_metre = 0.1 * math.pi * search.CONFLICT_BAND_M**2
    jobs = space.job_exposure({"adult": occ})["adult"]
    travel = space.travel()
    finite = np.isfinite(travel)
    expected = per_metre * (travel / search.ROBOT_SPEED_MPS + sum(search.HOLD_S))
    assert np.allclose(jobs[finite], expected[finite], rtol=1e-6)
    scores = space.conflict({"adult": occ, "wheelchair": occ}, rate_per_h=6.0)
    elements = space.element_people(occ)
    assert np.allclose(scores["adult"][finite], (6.0 * jobs[finite] + 3600.0 * elements[finite]) / occ.present)
    assert np.allclose(scores["adult"], scores["wheelchair"], equal_nan=True)
    index = int(np.flatnonzero(finite)[0])
    assert elements[index] == pytest.approx(space.occupancy(occ, index))
    with pytest.raises(ValueError, match="no walker time"):
        space.conflict({"adult": Occupancy(0.0, 0.0, 0.5, np.zeros((4, 4)))}, rate_per_h=6.0)


def test_the_joint_exposure_ranking_returns_distinct_valid_layouts_best_first(space: Search) -> None:
    crowded = space.spots["locker"][0]
    x0, y0, x1, y1 = search.approach_area(crowded).bounds
    xs, ys = np.meshgrid(np.arange(math.floor(x0 * 2) / 2 + 0.25, x1 + 4.0, 0.5), np.arange(math.floor(y0 * 2) / 2 + 0.25, y1 + 4.0, 0.5))
    groups = {"adult": Occupancy.from_cells(xs.ravel(), ys.ravel(), np.full(xs.size, 100.0), 0.5, 100.0)}
    robot = search.robot_only(space)
    cap = 1.5 * robot.travel_m
    ranked = search.joint_exposure(space, groups, rate_per_h=6.0, top=5, max_travel_m=cap)
    assert ranked and len(ranked) <= 5 and all(space.valid(r.index) and r.travel_m <= cap + 1e-9 for r in ranked)
    assert [r.score for r in ranked] == sorted(r.score for r in ranked)
    scores = space.conflict(groups, rate_per_h=6.0)
    for r in ranked:
        assert r.exposure is not None and r.score == pytest.approx(max(r.exposure.values()))
        assert r.score == pytest.approx(scores["adult"][r.index] + search.Search.conflict_floor(groups, r.floor_loss_m2)["adult"])
        assert r.exposure == pytest.approx(search.exposure_of(space, groups, r.index, rate_per_h=6.0))
    centres = space.spot_centres()
    for a, b in itertools.combinations(ranked, 2):
        far = [float(np.linalg.norm(centres[n][space.candidates[a.index, i]] - centres[n][space.candidates[b.index, i]])) for i, n in enumerate(SEARCHED)]
        assert max(far) >= search.DISTINCT_M
    # the best valid layout within the cap leads, whatever the distinct rule
    within = space.within_travel(cap)
    best = min((float(scores["adult"][i]), int(i)) for i in within if space.valid(int(i)))
    assert ranked[0].score == pytest.approx(best[0] + search.Search.conflict_floor(groups, space.check(best[1]).navr.floor_loss_m2)["adult"])
    # with no distance rule the top is the plain best by score
    loose = search.joint_exposure(space, groups, rate_per_h=6.0, top=3, max_travel_m=cap, distinct_m=0.0)
    assert [r.score for r in loose] == sorted(r.score for r in loose) and loose[0].score == pytest.approx(ranked[0].score)
    assert json.loads(json.dumps(ranked[0].record()))["exposure"] == pytest.approx(ranked[0].exposure, abs=1e-3)


def test_the_guideline_places_bays_locker_and_chargers_by_walking_distance(hospital: Site, loading: place.Spot, base) -> None:  # noqa: ANN001
    facts = place.SiteFacts((place.Entrance("main", "building", ((2.0, 1.5),), recipients=True),))
    space = Search(hospital, loading=loading, base=base, facts=facts)
    pick = search.guideline(space)
    assert space.valid(pick.index) and pick.score == pytest.approx(pick.travel_m)
    walk = place.walking_paths(hospital, loading.level)
    centres = space.spot_centres()
    rows = space.candidates
    bay_d = walk.lengths([loading.objects[0].goto], centres[SEARCHED[1]])[0]
    lockers = [s.objects[0].goto for s in space.spots[SEARCHED[2]]]
    locker_d = walk.lengths([(2.0, 1.5)], lockers)[0]
    c, b, k = (int(v) for v in rows[pick.index])
    charger_d = walk.lengths([tuple(centres[SEARCHED[1]][b])], centres[SEARCHED[0]])[0]
    # lexicographic in the rule order: no valid layout has a nearer bay, or as near a bay and a nearer locker, or
    # both as near and a charger nearer those bays
    for index in range(len(rows)):
        cc, bb, kk = (int(v) for v in rows[index])
        if (bay_d[bb], locker_d[kk]) < (bay_d[b], locker_d[k]) or (bb == b and kk == k and charger_d[cc] < charger_d[c]):
            assert not space.valid(index)
    with pytest.raises(ValueError, match="recipient entrances"):
        search.guideline(Search(hospital, loading=loading, base=base))
