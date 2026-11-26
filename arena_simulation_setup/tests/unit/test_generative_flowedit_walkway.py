"""The walkway rule (generative/flowedit/walkway.py) on synthetic floors: one-way and two-way corridors, necks between
furniture and docks (the office_1 layouts b5aeadcb and e52cbd21), a slot beside a pillar on and off the crowd's way
(the twin's travel-optimal bays), the walkers moved into a narrow passage, intermittent docks, the crowd's demand and
routes, and the rule in `place` on hospital_1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import shapely
import shapely.affinity

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
    assert found.by == (layout,) and not found.docked
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


def test_a_locker_dock_is_no_obstacle_to_the_walkway_widths_and_keeps_a_lane_while_a_robot_stands_in_it(office: WalkerFloor) -> None:
    routes = Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, [("west", "east"), ("east", "west")])))
    locker = Occupied("locker:locker_1", shapely.box(15.25, 10.06, 16.0, 10.85))

    def dock(top: float) -> Occupied:
        return Occupied("locker:locker_1#dock", shapely.box(16.0, 10.06, 17.25, top), intermittent=True)

    # the dock leaves 1.06 m to the plant: under the two-way width, but only while a robot delivers, and a lane is left
    stretches: list[walkway.Stretch] = []
    assert walkway.narrowings(office, routes, [locker, dock(11.71)], stretches=stretches) == []
    assert walkway.narrowings(office, routes, [locker, dock(11.71)], docked=True, stretches=stretches) == []
    assert {s.docked for s in stretches} == {False, True}
    assert min(s.after for s in stretches if s.docked) == pytest.approx(1.06, abs=CELL_M)
    # a dock reaching 0.65 m from the plant closes the hallway while a robot stands in it
    (found,) = walkway.narrowings(office, routes, [locker, dock(12.3)], docked=True)
    assert found.docked and found.required == pytest.approx(ONE_WAY_M) and found.after == 0.0
    assert found.by == ("locker:locker_1",)
    assert walkway.narrowings(office, routes, [locker, dock(12.3)]) == []
    # 1440 walkers an hour each way need more than the 1.06 m lane taking turns
    busy = Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, [("west", "east"), ("east", "west")], rate=0.4)))
    (found,) = walkway.narrowings(office, busy, [locker, dock(11.71)], docked=True)
    assert found.required == pytest.approx(walkway.docked_width(1440.0, 1440.0)) and found.after == pytest.approx(1.06, abs=CELL_M)


def test_the_lane_beside_a_docked_robot_is_a_wheelchair_or_what_the_flow_needs_taking_turns() -> None:
    assert walkway.docked_width(180.0, 180.0) == pytest.approx(ONE_WAY_M)
    # 3600 walkers an hour in all: 1 ped/s over half of 1.22 ped/(m s)
    assert walkway.docked_width(1800.0, 1800.0) == pytest.approx(1.0 / (0.5 * 1.22))
    assert walkway.dock_share(6.0, 10.0) == pytest.approx(60.0 / 3600.0)


def test_a_site_dock_that_is_intermittent_is_free_floor_until_docked() -> None:
    floor = WalkerFloor("0", shapely.box(0, 0, 10, 4), (), (Occupied("obj:locker_1#dock", shapely.box(4, 0, 5, 1), intermittent=True),))
    assert floor.free.area == pytest.approx(40.0)
    assert floor.free_docked.area == pytest.approx(39.0)


# --- the twin's travel-optimal bays: a slot between a bay and a pillar, on the crowd's way or off it ------------------

CONCOURSE = (0.0, 0.0, 30.0, 10.6)
#: a 0.9 m pillar 2.21 m off the south wall; the bay pair against the wall below it leaves 0.96 m
PILLAR = ("pillar", shapely.box(14.0, 2.21, 14.9, 3.11))
BAY = shapely.box(13.5, 0.05, 15.5, 1.25)


@pytest.fixture(scope="module")
def concourse() -> WalkerFloor:
    return _floor(CONCOURSE, PILLAR)


def test_a_slot_beside_a_pillar_off_the_crowds_way_passes(concourse: WalkerFloor) -> None:
    # the crowd crosses the far side of the concourse: a way through the slot by the wall is over 10 % longer than its own
    ends = {"west": (0.5, 8.5, 2.5, 10.0), "east": (27.5, 8.5, 29.5, 10.0)}
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


def test_an_element_that_moves_the_walkers_into_a_narrow_passage_fails() -> None:
    # an island splits a 4 m corridor: 2.0 m north of it, 1.0 m south; the crowd walks along the north side between two
    # doors there, the south passage over 10 % longer; an element closing the north side moves the walkers into the
    # 1.0 m south passage, which it does not touch
    floor = _floor((0.0, 0.0, 30.0, 4.0), ("island", shapely.box(10.0, 1.05, 20.0, 1.95)))
    ends = {"west": (8.0, 2.6, 9.5, 3.6), "east": (20.5, 2.6, 22.0, 3.6)}
    routes = Routes.build(floor, WalkerDemand.from_scenario(_crowd(ends, [("west", "east"), ("east", "west")])))
    (found,) = walkway.narrowings(floor, routes, [Occupied("blocker", shapely.box(14.0, 1.95, 15.0, 3.95))])
    assert found.two_way and found.moved and found.after == pytest.approx(1.0, abs=CELL_M) and found.before > TWO_WAY_M


def test_walkers_from_two_openings_who_meet_head_on_count_as_two_way() -> None:
    # a T-junction: a 4 m corridor west-east with a branch north at x 14-18; walkers go from the west end up the branch
    # and from the east end to the west end. Each pair of openings carries one direction, but west of the junction the two
    # meet head-on: an element there leaving 1.1 m needs the two-way width
    floor = WalkerFloor("0", shapely.union_all([shapely.box(0.0, 0.0, 30.0, 4.0), shapely.box(14.0, 3.9, 18.0, 14.0)]), (), ())
    ends = {"west": (0.5, 0.5, 2.0, 3.5), "east": (28.0, 0.5, 29.5, 3.5), "north": (14.5, 12.0, 17.5, 13.5)}
    routes = Routes.build(floor, WalkerDemand.from_scenario(_crowd(ends, [("west", "north"), ("east", "west")], rate=500.0 / 3600.0)))
    element = Occupied("element", shapely.box(8.0, 0.0, 10.0, 2.9))
    found = walkway.narrowings(floor, routes, [element])
    assert found and found[0].two_way and found[0].after == pytest.approx(1.1, abs=CELL_M)
    assert found[0].flow_ab == pytest.approx(500.0, rel=0.01) and found[0].flow_ba == pytest.approx(500.0, rel=0.01)


def test_a_neck_between_two_sources_and_sinks_inside_the_window_is_measured(office: WalkerFloor) -> None:
    # both ends of the walkers' way lie inside the element's window: a region-to-region stretch
    ends = {"west": (11.0, 10.5, 12.5, 13.5), "east": (21.5, 10.5, 23.0, 13.5)}
    routes = Routes.build(office, WalkerDemand.from_scenario(_crowd(ends, [("west", "east"), ("east", "west")])))
    (found,) = walkway.narrowings(office, routes, [Occupied("b5aeadcb", B5AEADCB)])
    assert found.two_way and found.after == pytest.approx(1.02, abs=CELL_M) and all(p.startswith("end:") for p in found.ports)


def test_a_neck_at_a_stairs_mouth_inside_the_window_is_measured() -> None:
    # walkers come up a stair whose mouth lies in the element's window and walk on along a 4 m corridor; the element
    # beside the mouth's way leaves 1.0 m: the stretch from the mouth is measured, not dropped
    floor = WalkerFloor("0", shapely.union_all([shapely.box(0.0, 0.0, 20.0, 4.0), shapely.box(40.0, 0.0, 50.0, 4.0)]), (), ())
    doc = _crowd({"west": (0.5, 0.5, 2.0, 3.5), "east": (48.0, 0.5, 49.5, 3.5)}, [("west", "east"), ("east", "west")])
    doc["teleport_zones"] = [{"name": "up", "entry": {"x": 19.6, "y": 2.0}, "destinations": ["down"], "delay": 5.0}, {"name": "down", "entry": {"x": 40.4, "y": 2.0}, "destinations": ["up"], "delay": 5.0}]
    routes = Routes.build(floor, WalkerDemand.from_scenario(doc))
    assert not routes.missing
    found = walkway.narrowings(floor, routes, [Occupied("element", shapely.box(15.0, 1.0, 17.0, 4.0))])
    assert found and found[0].two_way and found[0].after == pytest.approx(1.0, abs=CELL_M)
    assert any(p.startswith("end:") for p in found[0].ports)


def test_a_way_with_no_neck_near_the_element_does_not_end_the_search_for_the_others(concourse: WalkerFloor) -> None:
    # the crowd's shortest way runs north of the pillar, with no neck near the bays; the way through the slot is a few per
    # cent longer, within the slack, and is measured too: it fails
    ends = {"west": (0.5, 4.0, 2.5, 5.0), "east": (27.5, 4.0, 29.5, 5.0)}
    routes = Routes.build(concourse, WalkerDemand.from_scenario(_crowd(ends, [("west", "east"), ("east", "west")])))
    found = walkway.narrowings(concourse, routes, [Occupied("bay_pair:bay_1", BAY)])
    assert found and found[0].kind == "narrowed" and found[0].after == pytest.approx(0.96, abs=CELL_M)


def test_the_verdict_does_not_depend_on_where_the_grids_start(office: WalkerFloor) -> None:
    # the same office shifted by a fraction of a cell of both grids: the same narrowing, to a cell
    dx, dy = 0.033, 0.021
    moved = WalkerFloor("0", shapely.affinity.translate(office.floor, dx, dy), tuple(((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in office.segments), tuple(Occupied(o.name, shapely.affinity.translate(o.geometry, dx, dy)) for o in office.obstacles))
    pairs = [("west", "east"), ("east", "west")]
    shifted_ends = {k: (x0 + dx, y0 + dy, x1 + dx, y1 + dy) for k, (x0, y0, x1, y1) in ENDS.items()}
    for docks in (B5AEADCB, E52CBD21, shapely.box(8.0, 10.06, 10.0, 11.26)):
        here = walkway.narrowings(office, Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, pairs))), [Occupied("e", docks)])
        there = walkway.narrowings(moved, Routes.build(moved, WalkerDemand.from_scenario(_crowd(shifted_ends, pairs))), [Occupied("e", shapely.affinity.translate(docks, dx, dy))])
        assert [(n.kind, n.required) for n in here] == [(n.kind, n.required) for n in there]
        assert [n.after for n in here] == pytest.approx([n.after for n in there], abs=CELL_M)


def test_two_elements_that_leave_a_neck_together_are_both_named(office: WalkerFloor) -> None:
    routes = Routes.build(office, WalkerDemand.from_scenario(_crowd(ENDS, [("west", "east"), ("east", "west")])))
    south = Occupied("charger_pair:charger_1", shapely.box(6.0, 10.06, 7.0, 11.71))
    north = Occupied("locker:locker_1", shapely.box(6.0, 13.0, 7.0, 13.95))
    assert walkway.narrowings(office, routes, [south]) == [] and walkway.narrowings(office, routes, [north]) == []
    (found,) = walkway.narrowings(office, routes, [south], [north])
    assert set(found.by) == {"charger_pair:charger_1", "locker:locker_1"} and found.after == pytest.approx(13.0 - 11.71, abs=CELL_M)


def test_a_way_ending_at_a_free_standing_pillar_is_measured() -> None:
    # the walkers' way ends at a sink inside the window, heading for a free-standing pillar the floor wraps round: its
    # run-on goes on through the pillar to the window's edge, so the way still has two sides and its neck is read
    floor = WalkerFloor("0", shapely.box(0.0, 0.0, 30.0, 6.0), (), (Occupied("pillar", shapely.box(12.8, 2.6, 13.4, 3.4)),))
    ends = {"west": (0.5, 2.6, 2.0, 3.4), "east": (11.0, 2.8, 11.6, 3.2)}
    routes = Routes.build(floor, WalkerDemand.from_scenario(_crowd(ends, [("west", "east"), ("east", "west")])))
    assert walkway.narrowings(floor, routes, [Occupied("element", shapely.box(8.0, 0.0, 9.0, 1.2))]) == []
    (found,) = walkway.narrowings(floor, routes, [Occupied("element", shapely.box(8.0, 0.0, 9.0, 4.9))])
    assert found.kind == "narrowed" and found.two_way and found.after == pytest.approx(1.1, abs=CELL_M)


def test_a_way_whose_sides_the_window_does_not_part_is_unmeasured_not_wide() -> None:
    import numpy as np
    from arena_simulation_setup.generative.flowedit.walkway import _Grid, _width

    # a path along the window's top edge parts no left side from a right one: no width, never a pass
    grid = _Grid(0.0, 0.0, 0.05, (60, 60))
    free = np.ones(grid.shape, dtype=bool)
    width = _width(free, grid, np.full(60, 59), np.arange(60))
    assert not width.measured


# --- the crowd's demand and routes ------------------------------------------------------------------------------------


def test_the_demand_splits_each_source_over_its_sinks_by_weight_and_wheelchairs_do_not_climb() -> None:
    doc = _crowd({"a": (0, 0, 1, 1), "b": (5, 0, 6, 1), "c": (10, 0, 11, 1)}, [])
    doc["regions"]["a_wheelchair"] = {"type": "source", "polygon": _box(0, 0, 1, 1), "config": {"rate_profile": [{"t": 0.0, "rate": 0.4}], "agent": {"agent_type": "wheelchair_manual", "sink_affinity": [{"sink": "b_sink", "weight": 3.0}, {"sink": "c_sink", "weight": 1.0}]}}}
    demand = WalkerDemand.from_scenario(doc)
    assert sorted((f.sink, round(f.rate, 6), f.climbs) for f in demand.flows) == [("b_sink", 0.3, False), ("c_sink", 0.1, False)]
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


def test_a_crowd_some_of_whose_walkers_find_no_route_is_refused_not_passed() -> None:
    # the east end lies on a floor no way reaches: its walkers find no route, and a rule measured without them refuses
    floor = WalkerFloor("0", shapely.union_all([shapely.box(0.0, 0.0, 10.0, 4.0), shapely.box(20.0, 0.0, 30.0, 4.0)]), (), ())
    routes = Routes.build(floor, WalkerDemand.from_scenario(_crowd({"west": (0.5, 0.5, 2.0, 3.5), "east": (28.0, 0.5, 29.5, 3.5)}, [("west", "east")])))
    assert routes.missing
    with pytest.raises(ValueError, match="find no route"):
        routes.require_all()


def test_the_walker_floor_exports_what_humansim_gets(office: WalkerFloor) -> None:
    doc = office.export()
    assert len(doc["walls"]) == 4 and [o["name"] for o in doc["obstacles"]] == ["plant_3", "plant_4"]
    assert doc["bare_edges_m"] == 0.0
    assert office.free.area == pytest.approx(34.0 * 4.0 - 2 * 1.31 * 1.46 - 0.1 * (2 * 34 + 2 * 4) / 2.0, rel=0.02)


# --- the rule in place: hospital_1 with a crowd -----------------------------------------------------------------------


HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"


@pytest.mark.usefixtures("local_root")
def test_the_walkway_rule_runs_against_the_sites_crowd_and_check_layout_finds_the_necks_two_elements_make(local_root: Path) -> None:
    from arena_simulation_setup.generative.flowedit.edit import open_site
    from flowedit_docks import write_docked_assets

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


@pytest.mark.usefixtures("local_root")
def test_check_layout_names_the_neck_two_facing_elements_leave_in_a_hallway_the_crowd_walks(local_root: Path) -> None:
    from arena_simulation_setup.generative.flowedit.edit import open_site
    from arena_simulation_setup.generative.flowedit.place import Layout
    from flowedit_docks import write_docked_assets

    write_docked_assets(local_root)
    site = open_site(HOSPITAL)
    level = next(iter(site.world.levels.values()))
    hall = next(z for z in level.zones if z.name == "central_hallway").shape()
    x0, y0, x1, y1 = hall.bounds
    # the crowd walks the central hallway from end to end, both ways
    if x1 - x0 >= y1 - y0:
        mid = hall.intersection(shapely.LineString([((x0 + x1) / 2.0, y0), ((x0 + x1) / 2.0, y1)])).centroid.y
        ends = {"a": (x0 + 0.6, mid - 0.75, x0 + 2.1, mid + 0.75), "b": (x1 - 2.1, mid - 0.75, x1 - 0.6, mid + 0.75)}
    else:
        mid = hall.intersection(shapely.LineString([(x0, (y0 + y1) / 2.0), (x1, (y0 + y1) / 2.0)])).centroid.x
        ends = {"a": (mid - 0.75, y0 + 0.6, mid + 0.75, y0 + 2.1), "b": (mid - 0.75, y1 - 2.1, mid + 0.75, y1 - 0.6)}
    facts = place.SiteFacts(walkers=WalkerDemand.from_scenario(_crowd(ends, [("a", "b"), ("b", "a")], rate=0.2)))
    chargers = [s for s in place.candidate_spots(site, "charger_pair", facts=facts) if s.zone == "central_hallway" and s.valid]
    lockers = [s for s in place.candidate_spots(site, "locker", facts=facts) if s.zone == "central_hallway"]
    pairs = [(c, k) for c in chargers for k in lockers if c.frame.n[0] * k.frame.n[0] + c.frame.n[1] * k.frame.n[1] < -0.9]
    assert pairs
    charger, locker = min(pairs, key=lambda p: p[0].taken.distance(p[1].taken))
    problems = place.check_layout(site, Layout.of([charger, locker]))
    walk = [p for p in problems if "walkway" in p and (p.startswith("charger_pair: with locker") or p.startswith("locker: with charger_pair"))]
    assert walk, problems
