"""The walkway rule (generative/flowedit/walkway.py) on synthetic floors: one-way and two-way corridors, necks between
furniture and docks (the office_1 layouts b5aeadcb and e52cbd21), a slot beside a pillar on and off the crowd's way
(the twin's travel-optimal bays), the walkers moved into a narrow passage, intermittent docks, the crowd's demand and
routes, and the rule in `place` on hospital_1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import shapely

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from arena_simulation_setup.generative.flowedit import place, walkway  # noqa: E402
from arena_simulation_setup.generative.flowedit.walkway import ONE_WAY_M, TWO_WAY_M, Occupied, Routes, WalkerDemand, WalkerFloor  # noqa: E402

#: The raster's cells make a gap up to about a cell wider or narrower than the geometry's
CELL_M = 0.06


def _box(x0: float, y0: float, x1: float, y1: float) -> list[dict[str, float]]:
    return [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}]


def _crowd(ends: dict[str, tuple[float, float, float, float]], pairs: list[tuple[str, str]], rate: float = 0.05, agent: str = "adult") -> dict[str, Any]:
    """A scenario: a source and a sink on each end's box, and walkers at `rate` [ped/s] for each (from, to) pair."""
    regions: dict[str, Any] = {f"{name}_sink": {"type": "sink", "polygon": _box(*box)} for name, box in ends.items()}
    for a, b in pairs:
        regions[f"{a}_to_{b}"] = {"type": "source", "polygon": _box(*ends[a]), "config": {"rate_profile": [{"t": 0.0, "rate": rate}], "agent": {"agent_type": agent, "sink_affinity": [{"sink": f"{b}_sink", "weight": 1.0}]}}}
    return {"format": "arena-scenario/2", "regions": regions}


def _floor(box: tuple[float, float, float, float], *obstacles: tuple[str, Any]) -> WalkerFloor:
    x0, y0, x1, y1 = box
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    walls = tuple((corners[i], corners[(i + 1) % 4]) for i in range(4))
    return WalkerFloor("0", shapely.box(*box), walls, tuple(Occupied(name, g) for name, g in obstacles))


# --- office_1: a 4 m hallway with two plants on its north wall (the layouts of PINCH_FURNITURE_CHECK) -----------------

HALL = (0.0, 10.0, 34.0, 14.0)
PLANTS = (("plant_3", shapely.box(17.85, 12.54, 19.16, 14.08)), ("plant_4", shapely.box(19.85, 12.54, 21.16, 14.08)))
ENDS = {"west": (0.5, 10.5, 3.0, 13.5), "east": (31.0, 10.5, 33.5, 13.5)}
#: b5aeadcb: the charger pair's docks, 1.02 m from plant_3's corner on the diagonal; e52cbd21: the bay pair, 1.28 m
B5AEADCB = shapely.box(15.25, 10.06, 17.25, 11.71)
E52CBD21 = shapely.box(16.25, 10.06, 18.25, 11.26)


@pytest.fixture(scope="module")
def office() -> WalkerFloor:
    return _floor(HALL, *PLANTS)


@pytest.mark.parametrize(("layout", "docks", "gap"), [("b5aeadcb", B5AEADCB, 1.02), ("e52cbd21", E52CBD21, 1.28)])
def test_a_neck_between_docks_and_a_plant_fails_a_two_way_hallway_and_passes_a_one_way_one(office: WalkerFloor, layout: str, docks: Any, gap: float) -> None:  # noqa: ANN401
    two_way = Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, [("west", "east"), ("east", "west")])))
    (found,) = walkway.narrowings(office, two_way, [Occupied(layout, docks)])
    assert found.two_way and found.required == TWO_WAY_M
    assert found.after == pytest.approx(gap, abs=CELL_M)
    # before the layout the hallway's narrowest cross-section is between the plants and the south wall
    assert found.before == pytest.approx(12.54 - 10.05, abs=CELL_M)
    assert found.by == (layout,) and not found.intermittent
    assert found.flow_ab == pytest.approx(180.0) and found.flow_ba == pytest.approx(180.0)
    one_way = Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, [("west", "east")])))
    stretches: list[walkway.Stretch] = []
    assert walkway.narrowings(office, one_way, [Occupied(layout, docks)], stretches=stretches) == []
    assert [s.required for s in stretches] == [ONE_WAY_M] and stretches[0].after == pytest.approx(gap, abs=CELL_M)


def test_an_element_that_leaves_the_hallway_wide_passes(office: WalkerFloor) -> None:
    routes = Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, [("west", "east"), ("east", "west")])))
    stretches: list[walkway.Stretch] = []
    # a bay pair against the south wall far from the plants: 4 m less 1.2 m
    assert walkway.narrowings(office, routes, [Occupied("bays", shapely.box(8.0, 10.06, 10.0, 11.26))], stretches=stretches) == []
    assert stretches and min(s.after for s in stretches if s.after is not None) == pytest.approx(14.0 - 11.26 - 0.05, abs=CELL_M)


def test_the_requirement_follows_the_lesser_direction_of_the_flow(office: WalkerFloor) -> None:
    # 3 walkers per hour back: under FLOW_MIN_PED_H, so the hallway is one-way
    doc = _crowd(ENDS, [("west", "east")])
    doc["regions"]["east_to_west"] = {"type": "source", "polygon": _box(*ENDS["east"]), "config": {"rate_profile": [{"t": 0.0, "rate": 3.0 / 3600.0}], "agent": {"agent_type": "adult", "sink_affinity": [{"sink": "west_sink"}]}}}
    routes = Routes.build(office, WalkerDemand.from_scenario(doc))
    assert walkway.narrowings(office, routes, [Occupied("b5aeadcb", B5AEADCB)]) == []
    doc["regions"]["east_to_west"]["config"]["rate_profile"] = [{"t": 0.0, "rate": 2.0 * walkway.FLOW_MIN_PED_H / 3600.0}]
    routes = Routes.build(office, WalkerDemand.from_scenario(doc))
    assert [n.required for n in walkway.narrowings(office, routes, [Occupied("b5aeadcb", B5AEADCB)])] == [TWO_WAY_M]


def test_a_neck_a_robot_occupies_only_while_it_delivers_counts_and_says_so(office: WalkerFloor) -> None:
    routes = Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, [("west", "east"), ("east", "west")])))
    (found,) = walkway.narrowings(office, routes, [Occupied("locker:locker_1", shapely.box(15.25, 10.06, 16.0, 10.85)), Occupied("locker:locker_1#dock", shapely.box(16.0, 10.06, 17.25, 11.71), intermittent=True)])
    assert found.intermittent and found.by == ("locker:locker_1",)


# --- the twin's travel-optimal bays: a slot between a bay and a pillar, on the crowd's way or off it ------------------

CONCOURSE = (0.0, 0.0, 30.0, 10.6)
#: a 0.9 m pillar 2.21 m off the south wall; the bay pair against the wall below it leaves 0.96 m
PILLAR = ("pillar", shapely.box(14.0, 2.21, 14.9, 3.11))
BAY = shapely.box(13.5, 0.05, 15.5, 1.25)


@pytest.fixture(scope="module")
def concourse() -> WalkerFloor:
    return _floor(CONCOURSE, PILLAR)


def test_a_slot_beside_a_pillar_off_the_crowds_way_passes(concourse: WalkerFloor) -> None:
    # the crowd crosses the middle of the concourse: the slot by the wall is over 10 % longer than its way
    ends = {"west": (0.5, 5.0, 2.5, 8.0), "east": (27.5, 5.0, 29.5, 8.0)}
    routes = Routes.build(concourse, WalkerDemand.from_scenario(_crowd(ends, [("west", "east"), ("east", "west")])))
    assert walkway.narrowings(concourse, routes, [Occupied("bay_pair:bay_1", BAY)]) == []


def test_a_slot_beside_a_pillar_on_the_crowds_way_fails_however_wide_the_other_side(concourse: WalkerFloor) -> None:
    # the crowd walks along the south wall (a door at either end there): its shortest way runs through the slot, 7.5 m
    # of floor beyond the pillar or not
    ends = {"west": (0.5, 0.5, 2.5, 2.2), "east": (27.5, 0.5, 29.5, 2.2)}
    routes = Routes.build(concourse, WalkerDemand.from_scenario(_crowd(ends, [("west", "east"), ("east", "west")])))
    found = walkway.narrowings(concourse, routes, [Occupied("bay_pair:bay_1", BAY)])
    assert found and found[0].two_way and found[0].after == pytest.approx(0.96, abs=CELL_M)
    assert found[0].before == pytest.approx(2.21 - 0.05, abs=CELL_M)


def test_an_element_that_moves_the_walkers_into_a_narrow_passage_fails(concourse: WalkerFloor) -> None:
    # a 3 m corridor split by a long island: 2.0 m north of it, 1.0 m south; the crowd takes the north side, an element
    # in it moves the walkers into the 1.0 m south passage
    floor = _floor((0.0, 0.0, 30.0, 4.0), ("island", shapely.box(10.0, 1.05, 20.0, 1.95)))
    ends = {"west": (0.5, 0.5, 2.5, 3.5), "east": (27.5, 0.5, 29.5, 3.5)}
    routes = Routes.build(floor, WalkerDemand.from_scenario(_crowd(ends, [("west", "east"), ("east", "west")])))
    found = walkway.narrowings(floor, routes, [Occupied("blocker", shapely.box(14.0, 2.0, 15.0, 3.95))])
    assert found and found[0].two_way and found[0].after < ONE_WAY_M + 0.2


# --- the crowd's demand and routes ------------------------------------------------------------------------------------


def test_the_demand_splits_each_source_over_its_sinks_by_weight_and_wheelchairs_do_not_climb() -> None:
    doc = _crowd({"a": (0, 0, 1, 1), "b": (5, 0, 6, 1), "c": (10, 0, 11, 1)}, [])
    doc["regions"]["a_wheelchair"] = {"type": "source", "polygon": _box(0, 0, 1, 1), "config": {"rate_profile": [{"t": 0.0, "rate": 0.4}], "agent": {"agent_type": "wheelchair_manual", "sink_affinity": [{"sink": "b_sink", "weight": 3.0}, {"sink": "c_sink", "weight": 1.0}]}}}
    demand = WalkerDemand.from_scenario(doc)
    assert {(f.sink, f.rate, f.climbs) for f in demand.flows} == {("b_sink", pytest.approx(0.3), False), ("c_sink", pytest.approx(0.1), False)}
    assert hash(demand) == hash(WalkerDemand.from_scenario(doc))


def test_climbing_walkers_take_a_teleport_zone_and_wheelchair_users_do_not() -> None:
    # two floors with no way between them on the level but a stair, as the twin's deck and lower concourse
    floor = WalkerFloor("0", shapely.union_all([shapely.box(0, 0, 10, 4), shapely.box(20, 0, 30, 4)]), (), ())
    doc = _crowd({"west": (0.5, 0.5, 2.0, 3.5), "east": (28.0, 0.5, 29.5, 3.5)}, [("west", "east")])
    doc["regions"]["west_wheelchair"] = {"type": "source", "polygon": _box(0.5, 0.5, 2.0, 3.5), "config": {"rate_profile": [{"t": 0.0, "rate": 0.01}], "agent": {"agent_type": "wheelchair_manual", "sink_affinity": [{"sink": "east_sink"}]}}}
    doc["teleport_zones"] = [{"name": "up", "entry": {"x": 9.0, "y": 2.0}, "destinations": ["down"], "delay": 10.0}, {"name": "down", "entry": {"x": 21.0, "y": 2.0}, "destinations": []}]
    routes = Routes.build(floor, WalkerDemand.from_scenario(doc))
    assert [(m.source, m.climbs) for m in routes.missing] == [("west_wheelchair", False)]
    assert len(routes.routes) == 2  # the climbers' route, one piece on each floor


def test_the_walker_floor_exports_what_humansim_gets(office: WalkerFloor) -> None:
    doc = office.export()
    assert len(doc["walls"]) == 4 and [o["name"] for o in doc["obstacles"]] == ["plant_3", "plant_4"]
    assert doc["bare_edges_m"] == 0.0
    assert office.free.area == pytest.approx(34.0 * 4.0 - 2 * 1.31 * 1.46 - 0.1 * (2 * 34 + 2 * 4) / 2.0, rel=0.02)


# --- the rule in place: hospital_1 with a crowd -----------------------------------------------------------------------


HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"


@pytest.mark.usefixtures("local_root")
def test_the_walkway_rule_runs_against_the_sites_crowd_and_check_layout_finds_the_necks_two_elements_make(local_root: Path) -> None:
    from flowedit_docks import write_docked_assets

    from arena_simulation_setup.generative.flowedit.edit import open_site

    write_docked_assets(local_root)
    site = open_site(HOSPITAL)
    level = next(iter(site.world.levels.values()))
    zones = sorted((z for z in level.zones if len(z.polygon) >= 3), key=lambda z: z.shape().area, reverse=True)
    # the crowd walks between the two largest zones' middles, both ways
    ends = {}
    for name, zone in (("a", zones[0]), ("b", zones[1])):
        c = zone.shape().representative_point()
        ends[name] = (c.x - 0.75, c.y - 0.75, c.x + 0.75, c.y + 0.75)
    facts = place.SiteFacts(walkers=WalkerDemand.from_scenario(_crowd(ends, [("a", "b"), ("b", "a")], rate=0.2)))
    spots = place.candidate_spots(site, "charger_pair", facts=facts)
    checked = [s for s in spots if place.Rule.WALKWAY_WIDTH in s.verdict and not [r for r in s.failed if r is not place.Rule.WALKWAY_WIDTH]]
    assert checked and {s.verdict[place.Rule.WALKWAY_WIDTH] for s in checked} <= {place.Verdict.PASS, place.Verdict.FAIL}
    assert all(s.walkers is facts.walkers for s in spots)
    for s in checked:
        if s.verdict[place.Rule.WALKWAY_WIDTH] is place.Verdict.FAIL:
            assert s.narrowings and all(n.after < n.required for n in s.narrowings)
        else:
            assert not s.narrowings
    # without the crowd the rule says unknown and filters nothing
    assert {s.verdict[place.Rule.WALKWAY_WIDTH] for s in place.candidate_spots(site, "charger_pair")} == {place.Verdict.UNKNOWN}
