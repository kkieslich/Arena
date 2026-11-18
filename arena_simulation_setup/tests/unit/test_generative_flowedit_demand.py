from __future__ import annotations

import itertools
import json
import math
import os
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import cli, demand, infra, place, search
from arena_simulation_setup.generative.flowedit.demand import Cost, Demand, EntranceSearch, Walks
from arena_simulation_setup.generative.flowedit.edit import Site, open_site
from arena_simulation_setup.generative.flowedit.place import Entrance, Layout
from arena_simulation_setup.generative.flowedit.search import Occupancy, Search, TaskMix

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
shapely = pytest.importorskip("shapely")

HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"
CHARGERS, BAYS, LOCKER = search.SEARCHED


@pytest.fixture(scope="module")
def assets(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
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
    return next(s for s in place.candidate_spots(hospital, "loading_point") if s.valid and s.zone == "central_hallway")


@pytest.fixture(scope="module")
def base(hospital: Site, loading: place.Spot):  # noqa: ANN201 - a NavrBase
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    level = hospital.world.levels[loading.level]
    return NavrBase(level, exclude={o.name for o in level.objects})


@pytest.fixture(scope="module")
def inner(hospital: Site, loading: place.Spot, base) -> Search:  # noqa: ANN001
    """The first study's search space with the lockers of the whole site (no recipient distance rule), every other
    locker spot kept so that the tests can list every layout."""
    full = Search(hospital, loading=loading, base=base, elements={LOCKER: demand.locker_element()}, path_resolution=0.1)
    full.spots = {name: tuple(spots[::2]) if name == LOCKER else spots for name, spots in full.spots.items()}
    full._pairs = {(a, b): full._pair_matrix(a, b) for i, a in enumerate(search.SEARCHED) for b in search.SEARCHED[i + 1 :]}
    return full


@pytest.fixture(scope="module")
def entrances(inner: Search) -> tuple[Entrance, ...]:
    """Four entrances at the goto points of locker spots spread over the site: two near each other, two apart."""
    spots = inner.spots[LOCKER]
    gotos = np.array([s.objects[0].goto for s in spots])
    first = 0
    far = int(np.argmax(np.linalg.norm(gotos - gotos[first], axis=1)))
    near = int(np.argsort(np.linalg.norm(gotos - gotos[first], axis=1))[1])
    middle = int(np.argmin(np.abs(np.linalg.norm(gotos - gotos[first], axis=1) - np.linalg.norm(gotos[far] - gotos[first]) / 2)))
    picks = list(dict.fromkeys([first, near, middle, far]))
    types = ["building", "lift", "building", "lift"]
    return tuple(Entrance(f"door_{i}", types[i], (tuple(float(v) for v in gotos[k]),)) for i, k in enumerate(picks))


@pytest.fixture(scope="module")
def crowd() -> dict[str, Occupancy]:
    """A crowd that is denser in the east for adults and in the north for wheelchair users."""
    y, x = np.mgrid[0:200, 0:200]
    return {
        "adult": Occupancy(-20.0, -20.0, 0.5, 0.02 + 0.2 * (x / 200.0)),
        "wheelchair": Occupancy(-20.0, -20.0, 0.5, 0.02 + 0.2 * (y / 200.0)),
        "long_cane": Occupancy(-20.0, -20.0, 0.5, np.full((200, 200), 0.05)),
    }


@pytest.fixture(scope="module")
def space(inner: Search, entrances: tuple[Entrance, ...], crowd: dict[str, Occupancy]) -> EntranceSearch:
    return EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, charging_trips_per_job=0.2), groups=crowd)


def _reach(space: EntranceSearch, count: int) -> float:
    """The largest reach at which the layouts have `count` lockers: the most locker spots to choose from."""
    for reach in sorted({float(v) for v in np.round(space.walk_m.ravel(), 3) if np.isfinite(v)}, reverse=True):
        try:
            if space.cover(reach).count == count:
                return reach
        except ValueError:
            break
    raise AssertionError(f"no reach with {count} lockers")


def _listed(space: EntranceSearch, cover: demand.Cover, objective: Sequence[demand._Channel], caps: Sequence[tuple[demand._Channel, float]] = (), tie: demand._Channel | None = None) -> list[tuple[float, float, tuple[int, int, tuple[int, ...]]]]:
    """Every layout at `cover`'s reach whose spots go together pairwise and that holds the caps, by brute force: (the
    largest objective channel, the tie channel, (charger, bay, lockers)), best first."""
    out = []
    kept = range(len(space.lockers))
    for lockers in itertools.combinations(kept, cover.count):
        if not all(cover.within[e, list(lockers)].any() for e in range(len(space.entrances))):
            continue
        if not all(space.locker_pairs[a, b] for a, b in itertools.combinations(lockers, 2)):
            continue
        order, _ = space.served(lockers)
        for charger, bay in itertools.product(range(len(space.search.spots[CHARGERS])), range(len(space.search.spots[BAYS]))):
            if not space._cb[charger, bay] or not space._ck[charger, list(order)].all() or not space._bk[bay, list(order)].all():
                continue
            if any(space.value(ch, charger, bay, order) > cap + 1e-9 for ch, cap in caps):
                continue
            out.append((max(space.value(ch, charger, bay, order) for ch in objective), 0.0 if tie is None else space.value(tie, charger, bay, order), (charger, bay, order)))
    return sorted(out)


def _first_valid(space: EntranceSearch, listed: list[tuple[float, float, tuple[int, int, tuple[int, ...]]]]) -> tuple[float, float, tuple[int, int, tuple[int, ...]]]:
    return next(item for item in listed if space.check(*item[2]).valid)


def test_recipient_entrances_are_the_building_doors_and_lifts() -> None:
    facts = place.SiteFacts((
        Entrance("door", "building", ((1.0, 1.0),)),
        Entrance("exit", "mrt", ((2.0, 1.0),)),
        Entrance("lift_a", "lift", ((3.0, 1.0),), level="1"),
        Entrance("cafe", "shop", ((4.0, 1.0),)),
        Entrance("lift_b", "lift", ((5.0, 1.0),), recipients=True),
    ))
    assert [e.name for e in demand.recipient_entrances(facts)] == ["door", "lift_a", "lift_b"]
    assert [e.name for e in demand.recipient_entrances(facts, level="0")] == ["door", "lift_b"]
    assert [e.name for e in demand.recipient_entrances(facts, types=("shop",))] == ["cafe"]
    twin = infra.site_config("smu_link_full_v2")
    names = [e.name for e in demand.recipient_entrances(twin.facts)]
    assert names == ["outdoor_space_scis1", "scis1_entrance", "connexion_entrance", "lift_library", "lift_school_of_accountancy", "lift_school_of_social_science", "lift_scis1", "lift_scis1_staff"]


def _cut_room(tmp_path: Path):  # noqa: ANN202 - a RobotPaths
    """The walking floor of a 10 x 4 m room cut by a wall from (5, 0) to (5, 3): the floor between (2.5, 1) and
    (7.5, 1) runs round the wall's end."""
    from arena_simulation_setup.metrics.navr_filter import NavrBase
    from arena_simulation_setup.tree.World import FORMAT, WorldView

    root = tmp_path / "cut"
    root.mkdir()
    level = {"id": "0", "zones": [{"name": "room", "polygon": [[0, 0], [10, 0], [10, 4], [0, 4]]}], "walls": [{"path": [[0, 0], [10, 0], [10, 4], [0, 4]], "closed": True}, {"path": [[5, 0], [5, 3]]}]}
    (root / "world.yaml").write_text(yaml.safe_dump({"format": FORMAT, "levels": [level]}))
    return search.RobotPaths.from_base(NavrBase(WorldView(root).load(validate=False).levels["0"]), resolution=0.1)


def _half(name: str, kind: str, x: float, to: Sequence[str], delay: float = 0.0) -> dict:
    return {"name": name, "kind": kind, "destinations": list(to), "entry": {"x": x, "y": 1.0, "theta": 0.0}, "delay": delay}


def test_people_walk_over_stairs_and_escalators_and_step_free_walks_stay_on_the_floor(tmp_path: Path) -> None:
    floor = _cut_room(tmp_path)
    west, east = (2.5, 1.0), (7.5, 1.0)
    plain = Walks.of(floor)
    round_the_wall = plain.lengths([west], [east], climbs=True)[0, 0]
    assert round_the_wall == pytest.approx(floor.lengths([west], [east])[0, 0], abs=1e-6) and round_the_wall > 6.0
    assert plain.lengths([west], [east], climbs=False)[0, 0] == pytest.approx(round_the_wall)
    stairs = Walks.of(floor, [_half("top", "stairs", 4.0, ["bottom"]), _half("bottom", "stairs", 6.0, ["top"])])
    # 1.5 m to the stair half, the 2 m flight, 1.5 m on: both ways, since each half leads to the other
    assert stairs.lengths([west], [east], climbs=True)[0, 0] == pytest.approx(5.0, abs=0.15)
    assert stairs.lengths([east], [west], climbs=True)[0, 0] == pytest.approx(5.0, abs=0.15)
    assert stairs.lengths([west], [east], climbs=False)[0, 0] == pytest.approx(round_the_wall)
    # the flight at the stairs' speed, the floor at the walker's
    assert stairs.seconds([west], [east], 1.0, climbs=True)[0, 0] == pytest.approx(3.0 + 2.0 / demand.LINK_SPEED_MPS["stairs"], abs=0.2)
    assert stairs.seconds([west], [east], 0.5, climbs=False)[0, 0] == pytest.approx(round_the_wall / 0.5)
    # an escalator runs one way
    escalator = Walks.of(floor, [_half("up", "escalator", 4.0, ["landing"]), _half("landing", "escalator", 6.0, [])])
    assert escalator.lengths([west], [east], climbs=True)[0, 0] == pytest.approx(5.0, abs=0.15)
    assert escalator.lengths([east], [west], climbs=True)[0, 0] == pytest.approx(round_the_wall)
    # a link without steps (a lift cabin) is step-free and takes its delay
    cabin = Walks.of(floor, [_half("car_west", "lift", 4.0, ["car_east"], delay=20.0), _half("car_east", "lift", 6.0, ["car_west"], delay=20.0)])
    assert cabin.lengths([west], [east], climbs=False)[0, 0] == pytest.approx(5.0, abs=0.15)
    assert cabin.seconds([west], [east], 1.0, climbs=False)[0, 0] == pytest.approx(min(3.0 + 20.0, round_the_wall), abs=0.2)
    assert np.isinf(stairs.lengths([west], [(20.0, 1.0)], climbs=True)).all()
    with pytest.raises(ValueError, match="which is no link"):
        Walks.of(floor, [_half("top", "stairs", 4.0, ["nowhere"])])
    with pytest.raises(ValueError, match="from the walking floor"):
        Walks.of(floor, [_half("top", "stairs", 40.0, ["bottom"]), _half("bottom", "stairs", 6.0, [])])


def test_the_locker_keeps_every_rule_but_the_recipient_distance(hospital: Site, loading: place.Spot, base) -> None:  # noqa: ANN001
    catalog = place.CATALOG[LOCKER]
    locker = demand.locker_element()
    assert set(catalog.rules) - set(locker.rules) == {place.Rule.RECIPIENT_DISTANCE} and locker.name == LOCKER and locker.dock == catalog.dock
    assert set(catalog.rules) - set(demand.locker_element(outlet=False).rules) == {place.Rule.RECIPIENT_DISTANCE, place.Rule.OUTLET_CABLE}
    # an entrance marked for recipients filters the first study's lockers by its fixed distance, not these
    facts = place.SiteFacts((Entrance("main", "building", ((2.0, 1.5),), recipients=True),))
    old = Search(hospital, loading=loading, base=base, facts=facts)
    new = Search(hospital, loading=loading, base=base, facts=facts, elements={LOCKER: locker})
    free = Search(hospital, loading=loading, base=base)
    assert len(old.spots[LOCKER]) < len(new.spots[LOCKER]) == len(free.spots[LOCKER])
    assert {s.id for s in new.spots[LOCKER]} == {s.id for s in free.spots[LOCKER]}
    assert [s.id for s in new.spots[CHARGERS]] == [s.id for s in old.spots[CHARGERS]]
    with pytest.raises(ValueError, match="not elements of a layout"):
        Search(hospital, loading=loading, base=base, elements={"fountain": locker})
    assert np.array_equal(free.compatible(LOCKER, CHARGERS), free.compatible(CHARGERS, LOCKER).T)
    with pytest.raises(ValueError, match="no pair of searched elements"):
        free.compatible(LOCKER, LOCKER)


def test_numbered_lockers_keep_the_first_ones_names(hospital: Site, inner: Search) -> None:
    a, b = inner.spots[LOCKER][0], inner.spots[LOCKER][-1]
    assert demand.numbered(a, 1) is a
    second = demand.numbered(b, 2)
    assert second.element.name == "locker_2" and [o.name for o in second.objects] == ["locker_2"] and second.id != b.id
    assert (second.objects[0].x, second.objects[0].y, second.objects[0].goto) == (b.objects[0].x, b.objects[0].y, b.objects[0].goto)
    layout = Layout.of([inner.loading, a, second])
    assert [o.name for o in layout.objects] == ["locker_1", "loading_point", "locker_2"]
    assert not place.check_layout(hospital, layout)
    with pytest.raises(ValueError, match="one spot per element"):
        Layout.of([a, b])


def test_demand_and_cost_check_their_inputs() -> None:
    assert Demand(rate_per_h=10.0, pickup_share=0.3).deliveries_per_h == pytest.approx(7.0)
    for bad in ({"rate_per_h": 0.0}, {"pickup_share": 1.5}, {"bay_share": -0.1}, {"charging_trips_per_job": -1.0}):
        with pytest.raises(ValueError):
            Demand(**bad)
    cost = Cost.pooled({"adult": 0.3, "wheelchair": 0.5}, recipients_per_h=4.0, present=200.0, speeds={"adult": 1.0, "wheelchair": 0.8})
    # 4 recipients per hour among 200 walkers: each metre of mean walk costs 2 m / speed, for 0.02 of the members
    assert cost.exposure == {"adult": 0.3, "wheelchair": 0.5} and cost.walk == pytest.approx({"adult": 0.04, "wheelchair": 0.05})
    assert cost.of({"adult": 10.0, "wheelchair": 10.0}, {"adult": 50.0, "wheelchair": 100.0}) == pytest.approx({"adult": 5.0, "wheelchair": 10.0})
    assert Cost(walk={"wheelchair": 1.0}).groups == ("wheelchair",)
    with pytest.raises(ValueError, match="at least one walker group"):
        Cost()
    with pytest.raises(ValueError, match="negative"):
        Cost(exposure={"adult": -1.0})
    with pytest.raises(ValueError, match="present must be positive"):
        Cost.pooled({"adult": 1.0}, 4.0, 0.0)


def test_the_search_keeps_the_locker_spots_within_the_largest_reach(inner: Search, entrances: tuple[Entrance, ...], space: EntranceSearch) -> None:
    assert len(space.lockers) == len(inner.spots[LOCKER]) and space.counts()["spots"][LOCKER] == len(space.lockers)
    walks = Walks.of(place.walking_paths(inner.site, inner.loading.level))
    gotos = [s.objects[0].goto for s in space.lockers]
    assert np.allclose(space.walk_m, walks.lengths([e.anchor for e in entrances], gotos, climbs=True))
    assert np.allclose(space.walk_m, space.free_m)  # the hospital has no stairs
    assert np.allclose(space.walk_seconds("wheelchair"), space.free_m / demand.WALK_SPEED_MPS["wheelchair"])
    reach = float(np.sort(space.walk_m.min(axis=0))[len(space.lockers) // 2])
    near = EntranceSearch(inner, entrances, reach_max_m=reach, legs=None)
    assert [s.id for s in near.lockers] == [s.id for s, d in zip(space.lockers, space.walk_m.min(axis=0), strict=True) if d <= reach + 1e-9]
    assert 0 < len(near.lockers) < len(space.lockers)
    with pytest.raises(ValueError, match="over the search's largest"):
        near.cover(reach + 1.0)
    with pytest.raises(ValueError, match="no recipient entrances"):
        EntranceSearch(inner, [])
    with pytest.raises(ValueError, match="not on the loading point's level"):
        EntranceSearch(inner, [Entrance("upstairs", "lift", ((2.0, 1.5),), level="7")])


def test_a_layout_has_the_fewest_lockers_that_cover_every_entrance(space: EntranceSearch) -> None:
    n = len(space.entrances)
    seen = set()
    for count in range(1, n + 1):
        try:
            reach = _reach(space, count)
        except AssertionError:
            continue
        cover = space.cover(reach)
        seen.add(count)
        within = space.walk_m <= reach + 1e-9
        assert np.array_equal(cover.within, within) and np.array_equal(cover.lockers, np.flatnonzero(within.any(axis=0)))
        covering = [c for c in itertools.combinations(range(len(space.lockers)), count) if within[:, list(c)].any(axis=1).all()]
        assert covering  # that many cover every entrance
        if count > 1:  # and no fewer do
            assert not any(within[:, list(c)].any(axis=1).all() for c in itertools.combinations(range(len(space.lockers)), count - 1))
        # the ways to split the entrances hold the split of every covering set: each entrance at its nearest locker
        for lockers in covering:
            order, serves = space.served(lockers)
            assert sorted(e for group in serves for e in group) == list(range(n)) and all(serves)
            assert serves in cover.partitions
            for k, group in zip(order, serves, strict=True):
                assert all(space.walk_m[e, k] == space.walk_m[e, list(lockers)].min() and within[e, k] for e in group)
        assert all(sorted(e for group in parts for e in group) == list(range(n)) and len(parts) == count for parts in cover.partitions)
        assert all(any(within[list(group)][:, k].all() for k in range(len(space.lockers))) for parts in cover.partitions for group in parts)
        assert all([group[0] for group in parts] == sorted(group[0] for group in parts) for parts in cover.partitions)
    assert len(seen) >= 2 and max(seen) >= 2
    with pytest.raises(ValueError, match="no locker spot within 0.01 m as people walk of"):
        space.cover(0.01)


def test_locker_pairs_are_the_spots_that_stand_together(hospital: Site, space: EntranceSearch) -> None:
    pairs = space.locker_pairs
    assert pairs.shape == (len(space.lockers),) * 2 and not pairs.diagonal().any() and np.array_equal(pairs, pairs.T)
    for i, j in itertools.combinations(range(len(space.lockers)), 2):
        assert pairs[i, j] == (not place.check_layout(hospital, Layout.of([space.lockers[i], demand.numbered(space.lockers[j], 2)])))


def test_travel_with_one_locker_is_the_first_studys_job_cycle_without_charges_at_a_jobs_end(inner: Search, entrances: tuple[Entrance, ...], space: EntranceSearch) -> None:
    # one entrance, one locker: the old model with no charge at a job's end and the charging trips as top-ups
    one = EntranceSearch(inner, entrances[:1], reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, charging_trips_per_job=0.2), legs=None)
    inner.mix = TaskMix(rate_per_h=6.0, charges_per_robot=0.0, top_ups_per_job=0.2)
    rows = inner.candidates
    old = inner.travel()
    new = np.array([one.travel(int(c), int(b), [int(k)]) for c, b, k in rows])
    assert np.isfinite(old).all() and np.allclose(new, old, rtol=1e-9)
    assert np.allclose(one.legs["m_LK"], inner.lengths["LK"]) and np.allclose(one.legs["m_K_bay"], inner.lengths[f"K_{BAYS}"]) and np.allclose(one.legs["m_L_charger"], inner.lengths[f"L_{CHARGERS}"])
    assert np.allclose(one.legs["m_BC"], inner.bay_charger)
    # with several entrances a job's locker legs are those of the entrance it is addressed to, each entrance the same share
    reach = _reach(space, 2)
    cover = space.cover(reach)
    lockers = next(c for c in itertools.combinations(range(len(space.lockers)), 2) if cover.within[:, list(c)].any(axis=1).all())
    order, serves = space.served(lockers)
    m, d = space.legs, space.demand
    c, b = len(inner.spots[CHARGERS]) - 1, len(inner.spots[BAYS]) - 1
    expected = 0.5 * m["m_L_bay"][b] + 0.5 * m["m_L_charger"][c] + d.charging_trips_per_job * m["m_BC"][c, b]
    for k, group in zip(order, serves, strict=True):
        expected += len(group) / len(space.entrances) * (m["m_LK"][k] + 0.5 * m["m_K_bay"][k, b] + 0.5 * m["m_K_charger"][k, c])
    assert space.travel(c, b, lockers) == pytest.approx(expected)
    # all robots waiting at the bays: the charger legs drop out
    bays_only = EntranceSearch(inner, space.entrances, reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, bay_share=1.0), legs={k: v[:0] if k.startswith("x_") else v for k, v in space.legs.items()})
    assert bays_only.travel(c, b, lockers) == pytest.approx(m["m_L_bay"][b] + sum(len(g) / len(space.entrances) * (m["m_LK"][k] + m["m_K_bay"][k, b]) for k, g in zip(order, serves, strict=True)))


def test_exposure_on_a_uniform_crowd_follows_robot_travel(inner: Search, entrances: tuple[Entrance, ...]) -> None:
    occ = Occupancy(-20.0, -20.0, 0.5, np.full((200, 200), 0.1))
    space = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, charging_trips_per_job=0.3), groups={"adult": occ}, bands=(2.0, 1.0))
    reach = _reach(space, 2)
    lockers = next(c for c in itertools.combinations(range(len(space.lockers)), 2) if space.cover(reach).within[:, list(c)].any(axis=1).all())
    c, b = len(inner.spots[CHARGERS]) - 1, len(inner.spots[BAYS]) - 1
    for band in (2.0, 1.0):
        total, traffic = space.exposure(c, b, lockers, band)
        per_metre = 0.1 * math.pi * band**2
        assert traffic["adult"] == pytest.approx(6.0 * per_metre * space.travel(c, b, lockers) / search.ROBOT_SPEED_MPS / occ.present, rel=1e-6)
        floor = sum(occ.people(search.approach_area(s)) for s in (inner.loading, inner.spots[CHARGERS][c], inner.spots[BAYS][b], *(space.lockers[k] for k in lockers)))
        assert total["adult"] - traffic["adult"] == pytest.approx(3600.0 * floor / occ.present, rel=1e-6)
    with pytest.raises(ValueError, match="no occupancy of group"):
        space.exposure_channel("adult", 3.0)
    # one entrance and one locker: the first study's conflict exposure without charges at a job's end
    one = EntranceSearch(inner, entrances[:1], reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, charging_trips_per_job=0.3), groups={"adult": occ})
    inner.mix = TaskMix(rate_per_h=6.0, charges_per_robot=0.0, top_ups_per_job=0.3)
    inner._exposure.clear()
    old = inner.conflict({"adult": occ}, rate_per_h=6.0)["adult"]
    rows = inner.candidates
    for index in range(0, len(rows), max(1, len(rows) // 25)):
        c, b, k = (int(v) for v in rows[index])
        assert one.exposure(c, b, [k])[0]["adult"] == pytest.approx(old[index], rel=1e-6)


def test_the_walk_to_the_locker_is_each_entrances_walk_to_the_locker_that_serves_it(space: EntranceSearch) -> None:
    reach = _reach(space, 2)
    lockers = next(c for c in itertools.combinations(range(len(space.lockers)), 2) if space.cover(reach).within[:, list(c)].any(axis=1).all())
    walk = space.walk(lockers)
    nearest = space.walk_m[:, list(lockers)].min(axis=1)
    assert set(walk) == set(search.GROUPS)
    assert walk["adult"]["mean_m"] == pytest.approx(nearest.mean()) and walk["adult"]["longest_m"] == pytest.approx(nearest.max()) and nearest.max() <= reach + 1e-9
    assert walk["long_cane"]["mean_m"] == pytest.approx(walk["adult"]["mean_m"]) and walk["wheelchair"]["mean_m"] == pytest.approx(nearest.mean())
    assert walk["adult"]["mean_s"] == pytest.approx(nearest.mean() / demand.WALK_SPEED_MPS["adult"], rel=1e-6)
    assert walk["long_cane"]["longest_s"] == pytest.approx(nearest.max() / demand.WALK_SPEED_MPS["long_cane"], rel=1e-6)
    assert space.value(space.walk_channel("adult"), 0, 0, lockers) == pytest.approx(nearest.mean())


@pytest.mark.parametrize("count", [1, 2, 3])
def test_the_travel_optimal_layout_is_the_valid_layout_with_the_least_robot_travel(space: EntranceSearch, count: int) -> None:
    reach = _reach(space, count)
    cover = space.cover(reach)
    pick = demand.travel_optimal(space, reach)
    listed = _listed(space, cover, [space.travel_channel()])
    value, _, key = _first_valid(space, listed)
    assert pick.key == key and pick.travel_m == pytest.approx(value) and pick.value == pytest.approx(value)
    assert len(pick.lockers) == count and space.check(*pick.key).valid and pick.reach_m == reach
    assert pick.serves == space.served(pick.lockers)[1] and pick.lockers == space.served(pick.lockers)[0]
    names = [o.name for o in pick.layout.objects]
    assert sorted(names) == sorted(["charger_1", "charger_2", "bay_1", "bay_2", "loading_point", *(f"locker_{i + 1}" for i in range(count))])


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("margin", [1.0, 1.1, 1.5])
def test_the_pedestrian_aware_pick_is_the_least_cost_of_the_worst_off_group_within_the_travel_margin(space: EntranceSearch, count: int, margin: float) -> None:
    reach = _reach(space, count)
    cover = space.cover(reach)
    optimal = demand.travel_optimal(space, reach)
    cap = margin * optimal.travel_m
    travel = space.travel_channel()
    cost = Cost.pooled({"adult": 0.4, "wheelchair": 0.7, "long_cane": 0.4}, recipients_per_h=4.0, present=40.0)
    channels = list(space.cost_channels(cost).values())
    pick = demand.pedestrian_aware(space, reach, cost, cap, start=[optimal])
    value, _, key = _first_valid(space, _listed(space, cover, channels, [(travel, cap)], travel))
    assert pick.key == key and pick.value == pytest.approx(value) and pick.travel_m <= cap + 1e-6
    assert demand.pedestrian_aware(space, reach, cost, cap).key == key  # the same without a layout to start from
    exposure, _ = space.exposure(*pick.key)
    walk = {g: v["mean_m"] for g, v in space.walk(pick.lockers).items()}
    assert max(cost.of(exposure, walk).values()) == pytest.approx(value)
    if margin == 1.0:
        assert pick.travel_m == pytest.approx(optimal.travel_m)
    assert pick.value <= max(space.value(ch, *optimal.key) for ch in channels) + 1e-9


def test_costs_rank_in_order_the_first_decides_and_the_next_breaks_its_ties(space: EntranceSearch) -> None:
    reach = _reach(space, 2)
    cover = space.cover(reach)
    optimal = demand.travel_optimal(space, reach)
    cap = 1.5 * optimal.travel_m
    travel = space.travel_channel()
    trip = Cost(walk={g: 2.0 / demand.WALK_SPEED_MPS[g] for g in search.GROUPS})
    exposure = Cost(exposure=dict.fromkeys(search.GROUPS, 1.0))
    pick = demand.pedestrian_aware(space, reach, [trip, exposure], cap)
    walks = list(space.cost_channels(trip).values())
    shortest = _first_valid(space, _listed(space, cover, walks, [(travel, cap)], travel))[0]
    assert max(space.value(ch, *pick.key) for ch in walks) == pytest.approx(shortest)
    channels = list(space.cost_channels(exposure).values())
    value, _, key = _first_valid(space, _listed(space, cover, channels, [(travel, cap), *((ch, shortest) for ch in walks)], travel))
    assert pick.key == key and pick.value == pytest.approx(value)
    # the walk alone leaves the chargers and bays to the tie: the shorter robot travel
    alone = demand.pedestrian_aware(space, reach, trip, cap)
    assert alone.value == pytest.approx(shortest) and alone.travel_m <= pick.travel_m + 1e-9
    with pytest.raises(ValueError, match="no valid layout within"):
        demand.pedestrian_aware(space, reach, trip, 0.5 * optimal.travel_m)


def test_the_practice_based_layout_places_bays_lockers_and_chargers_by_walking_distance(inner: Search, space: EntranceSearch) -> None:
    reach = _reach(space, 2)
    cover = space.cover(reach)
    pick = demand.practice_based(space, reach)
    assert space.check(*pick.key).valid and len(pick.lockers) == 2
    walk = place.walking_paths(inner.site, inner.loading.level)
    centres = inner.spot_centres()
    bay_d = walk.lengths([inner.loading.objects[0].goto], centres[BAYS])[0]
    charger_d = walk.lengths([tuple(centres[BAYS][pick.bay])], centres[CHARGERS])[0]

    def mean_walk(lockers: Sequence[int]) -> float:
        return float(space.walk_m[:, list(lockers)].min(axis=1).mean())

    # lexicographic in the rule order: no valid layout has a nearer bay, or as near a bay and lockers with a shorter
    # mean walk, or both as near and a charger nearer those bays
    for _, _, (c, b, lockers) in _listed(space, cover, [space.travel_channel()]):
        if (bay_d[b], mean_walk(lockers)) < (bay_d[pick.bay], mean_walk(pick.lockers) - 1e-9) or (b == pick.bay and lockers == pick.lockers and charger_d[c] < charger_d[pick.charger]):
            assert not space.check(c, b, lockers).valid
    sets = list(itertools.islice(space.nearest_sets(cover, pick.bay), 50))
    assert [mean_walk(s) for s in sets] == sorted(mean_walk(s) for s in sets) and len(set(sets)) == len(sets)
    assert all(space._bk[pick.bay, list(s)].all() and space.locker_pairs[s[0], s[1]] for s in sets)


def test_a_layouts_record_is_plain_data(space: EntranceSearch) -> None:
    reach = _reach(space, 2)
    pick = demand.travel_optimal(space, reach)
    cost = Cost.pooled({"adult": 0.4, "wheelchair": 0.7, "long_cane": 0.4}, recipients_per_h=4.0, present=40.0)
    record = json.loads(json.dumps(space.record(pick, cost)))
    assert record["layout_id"] == pick.layout_id and record["lockers"] == 2 and record["reach_m"] == reach
    assert sorted(e for names in record["serves"].values() for e in names) == sorted(e.name for e in space.entrances) and list(record["serves"]) == ["locker_1", "locker_2"]
    assert set(record["walk"]) == set(search.GROUPS) and set(record["walk"]["adult"]) == {"mean_m", "longest_m", "mean_s", "longest_s"}
    assert set(record["walk_by_entrance"]) == {e.name for e in space.entrances}
    assert set(record["exposure"]["2"]) == set(search.GROUPS) and all(record["exposure"]["2"][g] >= record["exposure_traffic"]["2"][g] for g in search.GROUPS)
    assert record["worst_group"] == max(record["cost"], key=record["cost"].get)
    assert [s["element"] for s in record["spots"]] == [CHARGERS, BAYS, LOCKER, "loading_point", "locker_2"]


@pytest.fixture
def hospital_copy(tmp_path: Path, assets: Path) -> Path:
    """hospital_1 in the local worlds root with a crowd scenario `mix` and a site config beside it."""
    root = assets / "worlds" / f"hospital_copy_{tmp_path.name}"
    shutil.copytree(HOSPITAL, root)
    doc = {"format": "arena-scenario/2", "regions": {"far": {"type": "sink", "polygon": [{"x": 9.0, "y": 30.0}, {"x": 11.0, "y": 30.0}, {"x": 11.0, "y": 31.0}, {"x": 9.0, "y": 31.0}]}}}
    (root / "scenarios" / "mix").mkdir(parents=True)
    (root / "scenarios" / "mix" / "scenario.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return root


def test_cli_search_reports_the_layouts_per_reach(hospital_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    site = open_site(hospital_copy)
    loading = next(s for s in place.candidate_spots(site, "loading_point") if s.valid and s.zone == "central_hallway")
    level = site.world.levels[loading.level]
    base = NavrBase(level, exclude={o.name for o in level.objects})
    monkeypatch.setattr(search.Search, "base", property(lambda self: base))
    lockers = Search(site, loading=loading, base=base, elements={LOCKER: demand.locker_element()}, keep_out=infra.keep_out(hospital_copy / "scenarios" / "mix")).spots[LOCKER]
    gotos = np.array([s.objects[0].goto for s in lockers])
    far = int(np.argmax(np.linalg.norm(gotos - gotos[0], axis=1)))
    x, y = loading.objects[0].goto
    config = tmp_path / "site.yaml"
    config.write_text(yaml.safe_dump({
        "site": site.key,
        "scenario": "mix",
        "courier_access": [x, y],
        "entrances": [
            {"name": "ward", "type": "building", "at": [float(v) for v in gotos[0]]},
            {"name": "lift", "type": "lift", "at": [float(v) for v in gotos[far]]},
            {"name": "kiosk", "type": "shop", "at": [float(v) for v in gotos[len(gotos) // 2]]},
        ],
    }))
    out = tmp_path / "report.json"
    args = ["infra", "search", "--site", str(hospital_copy), "--site-config", str(config), "--reach", "0.01,8", "--margins", "1.1,1.5", "--rate", "6", "--charging-trips", "0.1", "--rank", "walk", "--out", str(out)]
    assert cli.main(args) == 0
    report = json.loads(out.read_text())
    assert report["recipient_entrances"] == ["ward", "lift"] and report["demand"]["charging_trips_per_job"] == 0.1 and "recipient_distance" not in report["locker_rules"]
    none, some = report["reaches"]
    assert none["valid"] is False and "no locker spot within 0.01 m" in none["reason"]
    assert some["valid"] is True and some["lockers"] == 2 and [r["role"] for r in some["layouts"]] == ["travel_optimal", "pedestrian_aware", "pedestrian_aware", "practice_based"]
    optimal = some["layouts"][0]
    for pick in some["layouts"][1:3]:
        assert pick["travel_m"] <= pick["margin"] * optimal["travel_m"] + 1e-3 and max(pick["cost"].values()) <= max(optimal["cost"].values()) + 1e-6
        assert max(v["longest_m"] for v in pick["walk"].values()) <= 8.0 + 1e-6
    assert "no world or suite written" in capsys.readouterr().err
    assert cli.main(args) == 1 and cli.main([*args, "--force"]) == 0  # an existing report is kept unless forced
    assert cli.main([*args[:-2], "--rank", "pooled"]) == 1  # no occupancy to predict exposure from
