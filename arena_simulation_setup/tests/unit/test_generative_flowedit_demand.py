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
    facts = place.SiteFacts(
        (
            Entrance("door", "building", ((1.0, 1.0),)),
            Entrance("exit", "mrt", ((2.0, 1.0),)),
            Entrance("lift_a", "lift", ((3.0, 1.0),), level="1"),
            Entrance("cafe", "shop", ((4.0, 1.0),)),
            Entrance("lift_b", "lift", ((5.0, 1.0),), recipients=True),
        )
    )
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
    walks = Walks.of(place.walking_paths(inner.site, inner.loading.level, neighbourhood=demand.WALK_NEIGHBOURHOOD))
    gotos = [s.objects[0].goto for s in space.lockers]
    assert np.allclose(space.walk_m, walks.lengths([e.anchor for e in entrances], gotos, climbs=True))
    assert np.allclose(space.walk_m, space.free_m)  # the hospital has no stairs
    assert np.allclose(space.walk_seconds("wheelchair"), space.free_m / demand.WALK_SPEED_MPS["wheelchair"])
    reach = float(np.sort(space.walk_m.min(axis=0))[len(space.lockers) // 2])
    asked = []

    def from_cache(found: Search, lockers: Sequence[place.Spot], values: np.ndarray) -> dict:
        asked.append(([s.id for s in lockers], values.shape))
        return demand.leg_integrals(found, lockers, values)

    near = EntranceSearch(inner, entrances, reach_max_m=reach, legs=from_cache)
    assert [s.id for s in near.lockers] == [s.id for s, d in zip(space.lockers, space.walk_m.min(axis=0), strict=True) if d <= reach + 1e-9]
    assert 0 < len(near.lockers) < len(space.lockers)
    assert asked == [([s.id for s in near.lockers], (0, inner.paths.cells))]  # the legs' function gets the kept lockers
    with pytest.raises(ValueError, match="the leg integrals hold"):
        EntranceSearch(inner, entrances, reach_max_m=reach, legs=space.legs)
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
    # a third of the robots waiting at the bays: each leg to a waiting spot takes its own share
    third = EntranceSearch(inner, space.entrances, reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, bay_share=0.3, charging_trips_per_job=0.2), legs={k: v[:0] if k.startswith("x_") else v for k, v in space.legs.items()})
    expected = 0.3 * m["m_L_bay"][b] + 0.7 * m["m_L_charger"][c] + 0.2 * m["m_BC"][c, b]
    for k, group in zip(order, serves, strict=True):
        expected += len(group) / len(space.entrances) * (m["m_LK"][k] + 0.3 * m["m_K_bay"][k, b] + 0.7 * m["m_K_charger"][k, c])
    assert third.travel(c, b, lockers) == pytest.approx(expected)
    # all robots waiting at the bays: the charger legs drop out
    bays_only = EntranceSearch(inner, space.entrances, reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, bay_share=1.0), legs={k: v[:0] if k.startswith("x_") else v for k, v in space.legs.items()})
    assert bays_only.travel(c, b, lockers) == pytest.approx(m["m_L_bay"][b] + sum(len(g) / len(space.entrances) * (m["m_LK"][k] + m["m_K_bay"][k, b]) for k, g in zip(order, serves, strict=True)))


def test_exposure_on_a_uniform_crowd_follows_robot_travel(inner: Search, entrances: tuple[Entrance, ...]) -> None:
    occ = Occupancy(-20.0, -20.0, 0.5, np.full((200, 200), 0.1))
    space = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=Demand(rate_per_h=6.0, bay_share=0.3, charging_trips_per_job=0.3), groups={"adult": occ}, bands=(2.0, 1.0))
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
    assert pick.key == key and pick.value == pytest.approx(shortest)  # the first cost's value: the one that decides
    # the walk alone leaves the chargers and bays to the tie: the shorter robot travel
    alone = demand.pedestrian_aware(space, reach, trip, cap)
    assert alone.value == pytest.approx(shortest) and alone.travel_m <= pick.travel_m + 1e-9
    with pytest.raises(ValueError, match="no valid layout within"):
        demand.pedestrian_aware(space, reach, trip, 0.5 * optimal.travel_m)


def test_a_spot_that_fails_the_full_check_on_its_own_is_left_out(inner: Search, entrances: tuple[Entrance, ...], space: EntranceSearch, monkeypatch: pytest.MonkeyPatch) -> None:
    import attrs
    from arena_simulation_setup.metrics import navr_filter

    assert space.alone_invalid == dict.fromkeys(search.SEARCHED, 0) and space.counts()["invalid_alone"] == space.alone_invalid
    reach = _reach(space, 2)
    pick = demand.travel_optimal(space, reach)
    # the NAV@r filter refuses every layout with the pick's first locker or its bay spot
    bad_locker = space.lockers[pick.lockers[0]].objects[0].goto
    bad_bay = inner.spots[BAYS][pick.bay].objects[0].goto
    real = navr_filter.navr_check

    def refusing(base, elements, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        verdict = real(base, elements, *args, **kwargs)
        if any(e.dock_point in (bad_locker, bad_bay) for e in elements):
            return attrs.evolve(verdict, ok=False, reasons=("refused by the test",))
        return verdict

    monkeypatch.setattr(navr_filter, "navr_check", refusing)
    screened = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=space.demand)
    # every bay pair with a slot on the refused one fails (a pair's neighbour one step along shares a slot)
    refused_bays = sum(any(o.goto == bad_bay for o in s.objects) for s in inner.spots[BAYS])
    assert refused_bays >= 1 and screened.alone_invalid == {CHARGERS: 0, BAYS: refused_bays, LOCKER: 1}
    assert bad_locker not in [s.objects[0].goto for s in screened.lockers] and len(screened.lockers) == len(space.lockers) - 1
    assert not screened._cb[:, pick.bay].any()
    again = demand.travel_optimal(screened, _reach(screened, 2))
    assert again.bay != pick.bay and screened.check(*again.key).valid and again.travel_m >= pick.travel_m - 1e-9
    # without the screen the search still returns a valid layout: it finds the refused ones invalid one by one
    plain = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=space.demand, screen=False, screen_pairs=False)
    assert plain.alone_invalid == dict.fromkeys(search.SEARCHED, 0) and len(plain.lockers) == len(space.lockers)
    slow = demand.travel_optimal(plain, _reach(plain, 2))
    assert plain.check(*slow.key).valid and slow.bay != pick.bay and bad_locker not in [plain.lockers[k].objects[0].goto for k in slow.lockers]
    if _reach(plain, 2) == _reach(screened, 2):
        assert slow.travel_m == pytest.approx(again.travel_m)


def test_the_practice_based_layout_places_bays_lockers_and_chargers_by_walking_distance(inner: Search, space: EntranceSearch) -> None:
    reach = _reach(space, 2)
    cover = space.cover(reach)
    pick = demand.practice_based(space, reach)
    assert space.check(*pick.key).valid and len(pick.lockers) == 2
    walk = place.walking_paths(inner.site, inner.loading.level, neighbourhood=demand.WALK_NEIGHBOURHOOD)
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
    config.write_text(
        yaml.safe_dump(
            {
                "site": site.key,
                "scenario": "mix",
                "courier_access": [x, y],
                "entrances": [
                    {"name": "ward", "type": "building", "at": [float(v) for v in gotos[0]]},
                    {"name": "lift", "type": "lift", "at": [float(v) for v in gotos[far]]},
                    {"name": "kiosk", "type": "shop", "at": [float(v) for v in gotos[len(gotos) // 2]]},
                ],
            }
        )
    )
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


# --- walks without the grid's bias, link times, separate weights, rule S, groupings, failures, ties ------------------


def _open_room(tmp_path: Path, neighbourhood: int):  # noqa: ANN202 - a RobotPaths
    """The walking floor of an open 40 x 20 m room at 0.1 m."""
    from arena_simulation_setup.metrics.navr_filter import NavrBase
    from arena_simulation_setup.tree.World import FORMAT, WorldView

    root = tmp_path / f"open{neighbourhood}"
    root.mkdir()
    level = {"id": "0", "zones": [{"name": "room", "polygon": [[0, 0], [40, 0], [40, 20], [0, 20]]}], "walls": [{"path": [[0, 0], [40, 0], [40, 20], [0, 20]], "closed": True}]}
    (root / "world.yaml").write_text(yaml.safe_dump({"format": FORMAT, "levels": [level]}))
    return search.RobotPaths.from_base(NavrBase(WorldView(root).load(validate=False).levels["0"]), resolution=0.1, neighbourhood=neighbourhood)


def test_walks_with_long_moves_are_near_the_straight_line(tmp_path: Path) -> None:
    a = (2.0, 2.0)
    worst, floors = {}, {}
    for neighbourhood in (8, 16, 32):
        floor = floors[neighbourhood] = _open_room(tmp_path, neighbourhood)
        ratios = []
        for angle in np.arange(0.0, 45.01, 2.5):
            b = (2.0 + 16.0 * math.cos(math.radians(angle)), 2.0 + 16.0 * math.sin(math.radians(angle)))
            ratios.append(Walks.of(floor).lengths([a], [b], climbs=True)[0, 0] / math.dist(a, b))
        worst[neighbourhood] = max(ratios)
    assert worst[8] > 1.07 and worst[16] <= 1.028 + 0.005 and worst[32] <= 1.0131 + 0.005
    assert demand.WALK_NEIGHBOURHOOD == 32
    # a reach of 30 m takes a spot 29.5 m away in a straight line at any angle (the 8-connected grid did not)
    for angle in (0.0, 10.0, 22.5, 30.0):
        b = (2.0 + 29.5 * math.cos(math.radians(angle)), 2.0 + 29.5 * math.sin(math.radians(angle)))
        assert Walks.of(floors[32]).lengths([a], [b], climbs=True)[0, 0] <= 30.0
    assert Walks.of(floors[8]).lengths([a], [(2.0 + 29.5 * math.cos(math.radians(22.5)), 2.0 + 29.5 * math.sin(math.radians(22.5)))], climbs=True)[0, 0] > 30.0
    # a long move never crosses a wall: round the end of the cut room's wall a walk is at least the taut string
    from arena_simulation_setup.metrics.navr_filter import NavrBase
    from arena_simulation_setup.tree.World import WorldView

    _cut_room(tmp_path)
    cut = search.RobotPaths.from_base(NavrBase(WorldView(tmp_path / "cut").load(validate=False).levels["0"]), resolution=0.1, neighbourhood=32)
    taut = 2 * math.hypot(2.5, 2.0 + 0.325)
    assert taut - 0.15 < cut.lengths([(2.5, 1.0)], [(7.5, 1.0)])[0, 0] < 1.0135 * taut + 0.15
    with pytest.raises(ValueError, match="neighbourhood must be one of"):
        search.RobotPaths.from_base(NavrBase(WorldView(tmp_path / "cut").load(validate=False).levels["0"]), neighbourhood=12)


def test_a_links_seconds_are_humansims_traverse_from_the_site_configs_fields(tmp_path: Path) -> None:
    def half(name: str, kind: str, x: float, to: Sequence[str], delay: float, belt: float, facing: float) -> dict:
        return {"name": name, "kind": kind, "destinations": list(to), "entry": {"x": x, "y": 1.0, "theta": facing}, "hold": {"x": x + 0.7 * math.cos(facing), "y": 1.0, "theta": facing}, "delay": delay, "belt": belt}

    floor = _cut_room(tmp_path)
    walks = Walks.of(floor, [half("up", "escalator", 4.0, ["land"], 3.0, 0.5, 0.0), half("land", "escalator", 6.0, [], 0.0, -0.5, math.pi), half("s1", "stairs", 3.5, ["s2"], 2.0, 0.0, 0.0), half("s2", "stairs", 6.5, ["s1"], 2.0, 0.0, math.pi)])
    up, stair = walks.links[0], walks.links[1]
    assert up.board_m == pytest.approx(0.7) and up.land_m == pytest.approx(0.7)
    assert up.seconds(1.1) == pytest.approx(0.7 / 0.5 + 3.0 + 0.7 / 0.5)  # each half at belt speed
    assert stair.seconds(1.1) == pytest.approx(0.7 / 0.7 + 2.0 + 0.7 / 0.7)  # a still stair caps at 0.7 m/s
    assert stair.seconds(0.5) == pytest.approx(0.7 / 0.5 + 2.0 + 0.7 / 0.5)  # a slower walker keeps its own speed
    # the twin's links: HumanSim's formula on the site config's own fields, within 0.25 s of its comment's traverse
    config = infra.site_config("smu_link_full_v2")
    by = {z["name"]: z for z in config.links}
    got = {}
    for z in config.links:
        for dest in z.get("destinations") or []:
            link = demand._Link(0, 0, 0.0, z["kind"], float(z.get("delay") or 0.0), demand._hold_m(z), demand._hold_m(by[dest]), float(z.get("belt") or 0.0), float(by[dest].get("belt") or 0.0))
            got[z["name"]] = link.seconds(1.1)
    assert got["tj_escalator_south_bottom"] == pytest.approx(24.84, abs=0.01) and abs(got["tj_escalator_south_bottom"] - 24.6) < 0.25
    assert got["tj_escalator_north_top"] == pytest.approx(23.93, abs=0.01) and abs(got["tj_escalator_north_top"] - 23.7) < 0.25
    assert got["tj_stairs_south_top"] == pytest.approx(16.05, abs=0.01) and abs(got["tj_stairs_south_top"] - 15.9) < 0.25


def test_time_on_element_floor_has_its_own_weight(space: EntranceSearch) -> None:
    reach = _reach(space, 2)
    pick = demand.travel_optimal(space, reach)
    total, traffic = space.exposure(*pick.key)
    cost = Cost(exposure={"adult": 0.4, "wheelchair": 0.6}, element={"adult": 2.0, "wheelchair": 1.0})
    values = {g: space.value(ch, *pick.key) for g, ch in space.cost_channels(cost).items()}
    for g, (w_t, w_e) in {"adult": (0.4, 2.0), "wheelchair": (0.6, 1.0)}.items():
        assert values[g] == pytest.approx(w_t * traffic[g] + w_e * (total[g] - traffic[g]))
        assert space.value(space.element_channel(g), *pick.key) == pytest.approx(total[g] - traffic[g])
    walk = {g: v["mean_m"] for g, v in space.walk(pick.lockers).items()}
    assert cost.of(total, walk, traffic) == pytest.approx(values)
    with pytest.raises(ValueError, match="time near robots alone"):
        cost.of(total, walk)
    # without element weights the exposure weights cover both, as before
    same = Cost(exposure={"adult": 0.4})
    assert space.value(space.cost_channels(same)["adult"], *pick.key) == pytest.approx(0.4 * total["adult"]) and same.element_weight("adult") == 0.4
    pooled = Cost.pooled({"adult": 0.47}, 4.0, 200.0, delay_per_element={"adult": 1.22})
    assert pooled.exposure == {"adult": 0.47} and pooled.element == {"adult": 1.22} and pooled.walk["adult"] == pytest.approx(4.0 * 2.0 / (200.0 * 1.1))
    assert demand.delay_per_exposure(1.0) == demand.DELAY_PER_ELEMENT and demand.delay_per_exposure(2.0) == demand.DELAY_PER_EXPOSURE
    with pytest.raises(ValueError, match="no default delay per exposure at a 1.5 m band"):
        demand.delay_per_exposure(1.5)


@pytest.fixture(scope="module")
def stair(space: EntranceSearch, crowd: dict[str, Occupancy]) -> tuple[EntranceSearch, int, float, float]:
    """A stair link on the hospital floor from a floor point A to the dock point of one locker spot j, where the straight
    flight is much shorter than the nearest step-free walk from A to any locker spot; one recipient entrance at A.
    Returns the search, j, a reach that meets j only over the stair, and the nearest step-free walk from A."""
    gotos = np.array([s.objects[0].goto for s in space.lockers])
    centres = space.walks.floor.centres[:: max(1, space.walks.floor.cells // 300)]
    free = space.walks.lengths([tuple(c) for c in centres], [tuple(g) for g in gotos], climbs=False)
    straight = np.linalg.norm(centres[:, None, :] - gotos[None, :, :], axis=2)
    nearest_free = free.min(axis=1)
    gain = np.where(np.isfinite(nearest_free), nearest_free - straight.min(axis=1), -np.inf)
    a = int(np.argmax(gain))
    j = int(np.argmin(straight[a]))
    point = tuple(float(v) for v in centres[a])
    links = [
        {"name": "flight_a", "kind": "stairs", "destinations": ["flight_b"], "entry": {"x": point[0], "y": point[1], "theta": 0.0}, "delay": 0.0},
        {"name": "flight_b", "kind": "stairs", "destinations": ["flight_a"], "entry": {"x": float(gotos[j][0]), "y": float(gotos[j][1]), "theta": 0.0}, "delay": 0.0},
    ]
    found = EntranceSearch(space.search, [Entrance("up", "lift", (point,))], links=links, reach_max_m=1000.0, demand=space.demand, groups=crowd, legs=space.legs)
    assert gain[a] > 3.0, "the hospital has no floor point with a stair worth taking"
    reach = (float(found.walk_m[0, j]) + float(nearest_free[a])) / 2.0
    return found, j, reach, float(nearest_free[a])


def test_wheelchair_users_walk_step_free_and_the_reach_is_as_people_walk(stair: tuple[EntranceSearch, int, float, float]) -> None:
    found, j, reach, nearest_free = stair
    assert found.walk_m[0, j] < found.free_m[0, j] - 3.0  # over the stair as people walk, round the walls step-free
    cover = found.cover(reach)
    assert cover.count == 1 and cover.within[0, j] and found.free_m[0, j] > reach and (found.free_m[0, cover.within[0]] > reach).all()
    walk = found.walk([j])
    assert walk["wheelchair"]["mean_m"] == pytest.approx(found.free_m[0, j]) and walk["wheelchair"]["longest_m"] == pytest.approx(found.free_m[0, j])
    assert walk["adult"]["mean_m"] == pytest.approx(found.walk_m[0, j]) and walk["long_cane"]["mean_m"] == pytest.approx(found.walk_m[0, j])
    assert walk["wheelchair"]["mean_s"] == pytest.approx(found.free_m[0, j] / demand.WALK_SPEED_MPS["wheelchair"], rel=1e-6)
    assert found.value(found.beyond_channel(reach), 0, 0, [j]) == 1.0
    record = json.loads(json.dumps(found.record(demand.travel_optimal(found, reach)), allow_nan=False))
    assert record["walk_by_entrance"]["up"]["step_free_m"] > reach


@pytest.mark.parametrize("wide", [False, True])
def test_rule_s_puts_step_free_layouts_first_where_the_margin_holds_one(stair: tuple[EntranceSearch, int, float, float], wide: bool) -> None:
    found, j, reach, nearest_free = stair
    reach = nearest_free + 2.0 if wide else reach
    cover = found.cover(reach)
    optimal = demand.travel_optimal(found, reach)
    travel, beyond = found.travel_channel(), found.beyond_channel(reach)
    cost = Cost.pooled({"adult": 0.4, "wheelchair": 0.7, "long_cane": 0.4}, recipients_per_h=4.0, present=40.0, delay_per_element={"adult": 1.2, "wheelchair": 2.2, "long_cane": 1.3})
    channels = list(found.cost_channels(cost).values())
    cap = 3.0 * optimal.travel_m
    plain = demand.pedestrian_aware(found, reach, cost, cap, start=[optimal])
    s = demand.pedestrian_aware(found, reach, cost, cap, start=[optimal], step_free=True)
    assert plain.key == _first_valid(found, _listed(found, cover, channels, [(travel, cap)], travel))[2] and plain.step_free_set is None
    if wide:
        assert s.step_free_set is True and found.value(beyond, *s.key) == 0.0
        assert s.key == _first_valid(found, _listed(found, cover, channels, [(travel, cap), (beyond, 0.0)], travel))[2]
    else:
        assert not _listed(found, cover, channels, [(travel, cap), (beyond, 0.0)], travel)  # no step-free layout at this reach
        assert s.step_free_set is False and s.key == plain.key
    assert json.loads(json.dumps(found.record(s, cost), allow_nan=False))["step_free_set"] is s.step_free_set


@pytest.fixture(scope="module")
def groupings(space: EntranceSearch, crowd: dict[str, Occupancy]) -> tuple[EntranceSearch, float]:
    """Three entrances at locker dock points and a reach at which two lockers cover them in more than one way, and the
    layout with the least mean walk for adults serves them in another grouping than the first the search lists."""
    gotos = [s.objects[0].goto for s in space.lockers]
    walk = space.walks.lengths(gotos, gotos, climbs=True)
    n, built = len(gotos), 0
    whole = frozenset(range(3))
    for trio in itertools.combinations(range(n), 3):
        rows = walk[list(trio)]
        for reach in sorted({round(float(v), 6) for v in rows.ravel() if 0 < v < 1000}):
            within = rows <= reach + 1e-9
            if within.all(axis=0).any() or not within.any(axis=1).all():
                continue
            joint = {frozenset(e for e in range(3) if within[e, k]) for k in range(n)}
            coverable = {g for g in (frozenset(c) for r in (1, 2) for c in itertools.combinations(range(3), r)) if any(g <= h for h in joint)}
            splits = {frozenset((g, whole - g)) for g in coverable if whole - g in coverable}
            if len(splits) < 2:
                continue
            entrances = [Entrance(f"door_{i}", "building", (tuple(float(v) for v in gotos[k]),)) for i, k in enumerate(trio)]
            found = EntranceSearch(space.search, entrances, reach_max_m=1000.0, demand=space.demand, groups=crowd, legs=space.legs)
            built += 1
            cover = found.cover(reach)
            if cover.count == 2 and len(cover.partitions) >= 2:
                try:
                    pick = demand.pedestrian_aware(found, reach, Cost(walk={"adult": 1.0}), 1e9)
                except ValueError:
                    pick = None
                if pick is not None and found.served(pick.lockers)[1] != cover.partitions[0]:
                    return found, reach
            if built >= 25:
                pytest.fail("no three entrances on the hospital floor give two groupings among the first 25 tried")
    pytest.fail("no three entrances on the hospital floor give two groupings")


def test_the_search_covers_every_way_to_split_the_entrances(groupings: tuple[EntranceSearch, float]) -> None:
    found, reach = groupings
    cover = found.cover(reach)
    assert len(cover.partitions) >= 2
    travel = found.travel_channel()
    walk = Cost(walk={"adult": 1.0})
    channels = list(found.cost_channels(walk).values())
    value, _, key = _first_valid(found, _listed(found, cover, channels, [(travel, 1e9)], travel))
    pick = demand.pedestrian_aware(found, reach, walk, 1e9)
    assert pick.key == key and pick.value == pytest.approx(value)
    assert found.served(pick.lockers)[1] != cover.partitions[0]  # the best layout lies in a later grouping
    assert demand.travel_optimal(found, reach).key == _first_valid(found, _listed(found, cover, [travel]))[2]


def test_a_layout_that_fails_the_full_check_only_as_a_whole_is_passed_over(inner: Search, entrances: tuple[Entrance, ...], space: EntranceSearch, crowd: dict[str, Occupancy], monkeypatch: pytest.MonkeyPatch) -> None:
    import attrs
    from arena_simulation_setup.metrics import navr_filter

    reach = _reach(space, 2)
    best = demand.travel_optimal(space, reach)
    docks = {inner.spots[CHARGERS][best.charger].objects[0].goto, inner.spots[BAYS][best.bay].objects[0].goto, space.lockers[best.lockers[0]].objects[0].goto}
    real = navr_filter.navr_check

    def refusing(base, elements, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        verdict = real(base, elements, *args, **kwargs)
        return attrs.evolve(verdict, ok=False, reasons=("refused by the test",)) if docks <= {e.dock_point for e in elements} else verdict

    monkeypatch.setattr(navr_filter, "navr_check", refusing)
    fresh = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=space.demand, groups=crowd, legs=space.legs)
    assert fresh.alone_invalid == dict.fromkeys(search.SEARCHED, 0) and all(v["left_out"] == 0 for v in fresh.pairs_screened.values())
    assert not fresh.check(*best.key).valid  # every pair of it passes, the whole does not
    again = demand.travel_optimal(fresh, reach)
    listed = _listed(fresh, fresh.cover(reach), [fresh.travel_channel()])
    assert again.key != best.key and again.key == _first_valid(fresh, listed)[2] and again.travel_m >= best.travel_m - 1e-9


def test_ties_go_to_the_shorter_robot_travel_then_the_lower_spots(space: EntranceSearch) -> None:
    reach = _reach(space, 2)
    cover = space.cover(reach)
    travel = space.travel_channel()
    walk = Cost(walk={"adult": 1.0})
    channels = list(space.cost_channels(walk).values())
    listed = _listed(space, cover, channels, [(travel, 1e9)], travel)
    least = listed[0][0]
    assert sum(1 for v, _, _ in listed if v == pytest.approx(least)) > 1  # the walk does not tell the chargers and bays apart
    pick = demand.pedestrian_aware(space, reach, walk, 1e9)
    assert pick.key == _first_valid(space, listed)[2]


def test_a_cost_no_layout_can_meet_gives_no_pick(inner: Search, entrances: tuple[Entrance, ...], space: EntranceSearch, crowd: dict[str, Occupancy]) -> None:
    fresh = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=space.demand, groups=crowd, legs=space.legs)
    reach = _reach(fresh, 2)
    best = demand.pedestrian_aware(fresh, reach, Cost(walk={"wheelchair": 1.0}), 1e9)
    fresh.free_m = fresh.free_m.copy()
    fresh.free_m[:, best.lockers[0]] = np.inf  # wheelchair users cannot reach the first locker of the best layout
    other = demand.pedestrian_aware(fresh, reach, Cost(walk={"wheelchair": 1.0}), 1e9)
    assert best.lockers[0] not in other.lockers and math.isfinite(other.value)
    fresh.free_m[:] = np.inf
    with pytest.raises(ValueError, match="no pick: no valid layout within .* has a finite cost"):
        demand.pedestrian_aware(fresh, reach, Cost(walk={"wheelchair": 1.0}), 1e9)
    with pytest.raises(ValueError, match="no pick"):  # a known layout to start from is no pick either when its cost is infinite
        demand.pedestrian_aware(fresh, reach, Cost(walk={"wheelchair": 1.0}), 1e9, start=[best])
    record = fresh.record(best, Cost(walk={"wheelchair": 1.0}))
    assert record["cost"]["wheelchair"] is None and record["walk"]["wheelchair"]["mean_m"] is None
    json.dumps(record, allow_nan=False)


def test_near_pairs_that_fail_together_are_left_out(inner: Search, entrances: tuple[Entrance, ...], space: EntranceSearch, crowd: dict[str, Occupancy], monkeypatch: pytest.MonkeyPatch) -> None:
    import attrs
    from arena_simulation_setup.metrics import navr_filter

    spots = {CHARGERS: inner.spots[CHARGERS], BAYS: inner.spots[BAYS], LOCKER: space.lockers}
    matrices = {(CHARGERS, BAYS): space._cb, (CHARGERS, LOCKER): space._ck, (BAYS, LOCKER): space._bk}
    near = next(((a, b, i, j) for (a, b), m in matrices.items() for i, x in enumerate(spots[a]) for j, y in enumerate(spots[b]) if m[i, j] and search._reach(x).intersects(search._reach(y))), None)
    assert near is not None, "the hospital subset has no near pair"
    first, second, i, j = near
    docks = {spots[first][i].objects[0].goto, spots[second][j].objects[0].goto}
    real = navr_filter.navr_check

    def refusing(base, elements, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        verdict = real(base, elements, *args, **kwargs)
        return attrs.evolve(verdict, ok=False, reasons=("refused by the test",)) if docks <= {e.dock_point for e in elements} else verdict

    monkeypatch.setattr(navr_filter, "navr_check", refusing)
    fresh = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=space.demand, groups=crowd, legs=space.legs)
    left = {(CHARGERS, BAYS): fresh._cb, (CHARGERS, LOCKER): fresh._ck, (BAYS, LOCKER): fresh._bk}[(first, second)]
    assert not left[i, j] and fresh.pairs_screened[f"{first}/{second}"]["left_out"] >= 1
    kept = EntranceSearch(inner, entrances, reach_max_m=1000.0, demand=space.demand, groups=crowd, legs=space.legs, screen_pairs=False)
    assert {(CHARGERS, BAYS): kept._cb, (CHARGERS, LOCKER): kept._ck, (BAYS, LOCKER): kept._bk}[(first, second)][i, j] and kept.pairs_screened == {}
    reach = _reach(fresh, 2)
    assert fresh.check(*demand.travel_optimal(fresh, reach).key).valid and demand.travel_optimal(fresh, reach).key == demand.travel_optimal(kept, reach).key


def test_cli_search_ranks_by_each_rule(hospital_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
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
    config.write_text(yaml.safe_dump({"site": site.key, "scenario": "mix", "courier_access": [x, y], "entrances": [{"name": "ward", "type": "building", "at": [float(v) for v in gotos[0]]}, {"name": "lift", "type": "lift", "at": [float(v) for v in gotos[far]]}]}))
    rows = [{"world": site.key, "stage": f"{site.key}__s1", "planner": "peds_only", "group": g, "x": cx + 0.5, "y": cy + 0.5, "cell_m": 1.0, "ped_seconds": 10.0 + (cx if g == "adult" else cy)} for g in search.GROUPS for cx in range(-5, 40) for cy in range(-5, 45)]
    pd.DataFrame(rows).to_parquet(tmp_path / "cells.parquet")
    common = ["infra", "search", "--site", str(hospital_copy), "--site-config", str(config), "--reach", "8", "--margins", "1.5", "--rate", "6", "--group-occupancy", str(tmp_path / "cells.parquet")]
    for rank in ("step-free", "pooled", "exposure", "walk"):
        out = tmp_path / f"{rank}.json"
        assert cli.main([*common, "--rank", rank, "--out", str(out)]) == 0
        report = json.loads(out.read_text())
        assert report["rank"] == rank and report["search"]["walk_neighbourhood"] == demand.WALK_NEIGHBOURHOOD
        _, pick = [r for r in report["reaches"][0]["layouts"] if r["role"] in ("travel_optimal", "pedestrian_aware")]
        assert pick["travel_m"] <= 1.5 * report["reaches"][0]["layouts"][0]["travel_m"] + 1e-3
        if rank == "exposure":
            assert pick["value"] == pytest.approx(max(pick["exposure"]["2"].values()), abs=1e-3)
        else:
            assert pick["value"] == pytest.approx(max(pick["cost"].values()), abs=1e-3)
        assert ("step_free_set" in pick) == (rank == "step-free")
        if rank in ("step-free", "pooled"):
            assert "element's floor times adult 1.22, wheelchair 2.19, long_cane 1.29" in report["ranking"]
    assert cli.main([*common, "--band", "1.5", "--rank", "pooled", "--out", str(tmp_path / "band.json")]) == 1  # no default weights at 1.5 m
    assert not (tmp_path / "band.json").exists()
