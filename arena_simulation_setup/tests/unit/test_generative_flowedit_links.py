"""The world's stairs and escalators in flowedit (generative/flowedit/links.py): as HumanSim teleport zones, priced as
HumanSim prices a ride, taken by the crowd's routes for climbers only, kept clear of elements at their mouths, and the
crowd builder on a site without a scenario to take its robots from."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import shapely

pytest.importorskip("numpy")
pytest.importorskip("scipy")

from arena_simulation_setup.generative.flowedit import links, walkway  # noqa: E402
from arena_simulation_setup.shared.world import STAIRS_JOINED_INSET, Stairs  # noqa: E402
from arena_simulation_setup.utils.geometry import Position  # noqa: E402


def _half(name: str, x: float, mouth_side: str, to: str = "", kind: str = "stairs", speed: float = 0.5) -> Stairs:
    """A 3 m by 1.5 m half on its own level centred at (x, 2), its mouth on `mouth_side`."""
    return Stairs(name=name, position=Position(x, 2.0, 0.0), kind=kind, size=[3.0, 1.5], mouth_side=mouth_side, to=to, speed=speed, same_level=True)


def _level(*halves: Stairs) -> SimpleNamespace:
    return SimpleNamespace(stairs=list(halves))


def test_world_halves_become_teleport_zones_as_the_runtime_hands_them_to_humansim() -> None:
    # a two-way stair (a and b, each the other's partner) and a one-way escalator (c boards, d lands); a ramp is floor
    level = _level(_half("a", 5.0, "-x", "b"), _half("b", 15.0, "+x", "a"), _half("c", 5.0, "-x", "d", "escalator"), _half("d", 15.0, "+x", "", "escalator"), _half("r", 25.0, "-x", kind="ramp"))
    zones = {z["name"]: z for z in links.world_links(level)}
    assert sorted(zones) == ["a", "b", "c", "d"]
    assert zones["a"]["destinations"] == ["b"] and zones["c"]["destinations"] == ["d"] and zones["d"]["destinations"] == []
    # the mouth of a (-x side) at x 3.5, facing +x; its hold just past the seam on a half joined on its own level
    assert (zones["a"]["entry"]["x"], zones["a"]["entry"]["y"]) == pytest.approx((3.5, 2.0))
    assert zones["a"]["entry"]["theta"] == pytest.approx(0.0)
    assert zones["a"]["hold"]["x"] == pytest.approx(3.5 + 3.0 - STAIRS_JOINED_INSET)
    assert abs(math.remainder(zones["b"]["entry"]["theta"] - math.pi, math.tau)) < 1e-5
    # the escalator's belt runs towards the seam on the boarding half, away from it on the landing half
    assert zones["c"]["belt"] == pytest.approx(0.5) and zones["d"]["belt"] == pytest.approx(-0.5) and zones["a"]["belt"] == 0.0


def test_a_ride_costs_what_humansim_prices_it_at() -> None:
    level = _level(_half("a", 5.0, "-x", "b"), _half("b", 15.0, "+x", "a"), _half("c", 5.0, "-x", "d", "escalator"), _half("d", 15.0, "+x", "", "escalator"))
    found = {link.name: link for link in walkway.links_of(links.world_links(level))}
    assert sorted(found) == ["a>b", "b>a", "c>d"]
    run = 3.0 - STAIRS_JOINED_INSET  # entry to hold on each half
    assert found["a>b"].cost_m == pytest.approx(2 * run / walkway.STAIR_SPEED_MPS * walkway.LINK_SPEED_MPS, abs=1e-3)
    assert found["c>d"].cost_m == pytest.approx(2 * run / 0.5 * walkway.LINK_SPEED_MPS, abs=1e-3)
    assert found["a>b"].entry == pytest.approx((3.5, 2.0)) and found["a>b"].exits[0] == pytest.approx((16.5, 2.0))


def test_the_crowd_takes_the_worlds_stairs_and_wheelchair_users_do_not() -> None:
    # two floors with no way between them on the level but a stair from the west one's east end to the east one's west end
    floor = walkway.WalkerFloor("0", shapely.union_all([shapely.box(-6.0, 0.0, 3.5, 4.0), shapely.box(16.5, 0.0, 26.0, 4.0)]), (), ())
    level = _level(_half("up", 5.0, "-x", "down"), _half("down", 15.0, "+x", "up"))
    regions = {"west": [{"x": -5.5, "y": 0.5}, {"x": -4.0, "y": 0.5}, {"x": -4.0, "y": 3.5}, {"x": -5.5, "y": 3.5}], "east": [{"x": 24.0, "y": 0.5}, {"x": 25.5, "y": 0.5}, {"x": 25.5, "y": 3.5}, {"x": 24.0, "y": 3.5}]}
    doc = {"regions": {"east_sink": {"type": "sink", "polygon": regions["east"]}}}
    for name, agent in (("west_adult", "adult"), ("west_wheelchair", "wheelchair_manual")):
        doc["regions"][name] = {"type": "source", "polygon": regions["west"], "config": {"rate_profile": [{"t": 0.0, "rate": 0.01}], "agent": {"agent_type": agent, "sink_affinity": [{"sink": "east_sink"}]}}}
    demand = walkway.WalkerDemand.from_scenario(doc)
    assert not demand.links
    alone = walkway.Routes.build(floor, demand)
    assert {f.source for f in alone.missing} == {"west_adult", "west_wheelchair"}
    over = walkway.Routes.build(floor, demand.with_links(links.world_links(level)))
    assert [(f.source, f.climbs) for f in over.missing] == [("west_wheelchair", False)]
    assert len(over.routes) == 2  # the adults' route, a piece on each floor


def test_no_element_stands_on_a_world_stair_or_in_front_of_its_mouth() -> None:
    zones = links.world_links(_level(_half("a", 5.0, "-x", "b"), _half("b", 15.0, "+x", "a")))
    taken = shapely.union_all(links.link_keep_out(zones))
    for x, y in ((4.0, 2.0), (3.5 - links.LINK_APRON_M + 0.05, 2.5), (16.5 + links.LINK_APRON_M - 0.05, 1.5)):
        assert taken.contains(shapely.Point(x, y)), (x, y)
    for x, y in ((3.5 - links.LINK_APRON_M - 0.1, 2.0), (2.0, 3.0), (10.0, 2.0)):
        assert not taken.contains(shapely.Point(x, y)), (x, y)


def test_a_world_stair_and_a_site_config_link_of_one_name_are_refused() -> None:
    site = SimpleNamespace(world=SimpleNamespace(levels={"0": _level(_half("a", 5.0, "-x", "b"), _half("b", 15.0, "+x", "a"))}))
    assert [z["name"] for z in links.site_links(site, "0", [{"name": "lift", "entry": {"x": 0.0, "y": 0.0}}])] == ["lift", "a", "b"]
    with pytest.raises(ValueError, match="both world stairs and site config links"):
        links.site_links(site, "0", [{"name": "a", "entry": {"x": 0.0, "y": 0.0}}])


@pytest.mark.usefixtures("local_root")
def test_the_crowd_builder_takes_its_robots_from_the_sites_default_scenario_or_none(tmp_path: Path) -> None:
    import shutil

    import yaml
    from arena_simulation_setup.generative.flowedit import infra
    from arena_simulation_setup.generative.flowedit.edit import open_site
    from arena_simulation_setup.generative.flowedit.infra_cli import _crowd_robots

    hospital = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"
    root = tmp_path / "worlds" / "fresh"
    shutil.copytree(hospital, root, ignore=shutil.ignore_patterns("scenarios"))
    config = infra.SiteConfig(site="fresh", scenario="study_entrances")
    assert _crowd_robots(open_site(root), config, None) == ([], "none (the site has no scenario to take them from)")
    (root / "scenarios" / "default").mkdir(parents=True)
    (root / "scenarios" / "default" / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", "robots": [{"start": [1.0, 2.0, 0.0], "phases": [{"goto": [5.0, 2.0, 0.0]}]}]}))
    robots, where = _crowd_robots(open_site(root), config, None)
    assert robots == [{"start": [1.0, 2.0, 0.0], "phases": []}] and "default" in where


# --- flights that meet on a landing (the world draws it as stair, no zone covers it) ------------------------------------


def _flights() -> SimpleNamespace:
    """Two flights of two halves each between floor A (x <= 3.5) and floor B (x >= 11.5), meeting on a 1 m landing at
    x 7-8 that no floor covers, as the twin's T-junction stairs."""

    def half(name: str, x: float, mouth_side: str, to: str) -> Stairs:
        return Stairs(name=name, position=Position(x, 2.0, 0.0), kind="stairs", size=[1.75, 1.5], mouth_side=mouth_side, to=to, same_level=True)

    return _level(half("lower_bottom", 4.375, "-x", "lower_top"), half("lower_top", 6.125, "+x", "lower_bottom"), half("upper_bottom", 8.875, "-x", "upper_top"), half("upper_top", 10.625, "+x", "upper_bottom"))


def _on_floor(p: tuple[float, float]) -> bool:
    return p[0] <= 3.6 or p[0] >= 11.4


def test_flights_that_meet_on_a_landing_become_one_link_from_floor_to_floor() -> None:
    zones = links.world_links(_flights())
    found = {z["name"]: z for z in links.floor_links(zones, _on_floor)}
    # the landing halves are left out; each floor half leads over the landing to the other floor
    assert sorted(found) == ["lower_bottom", "lower_bottom>upper_bottom>upper_top", "upper_top", "upper_top>lower_top>lower_bottom"]
    assert found["lower_bottom"]["destinations"] == [] and found["lower_bottom>upper_bottom>upper_top"]["destinations"] == ["upper_top"]
    by = {z["name"]: z for z in zones}
    run = math.dist((by["lower_top"]["entry"]["x"], by["lower_top"]["entry"]["y"]), (by["lower_top"]["hold"]["x"], by["lower_top"]["hold"]["y"]))
    landing = 8.0 - 7.0
    assert found["lower_bottom>upper_bottom>upper_top"]["delay"] == pytest.approx((run + landing + run) / links.STAIR_SPEED_MPS, abs=1e-3)


def test_the_crowd_climbs_over_a_landing_and_a_link_the_routes_cannot_reach_is_refused() -> None:
    floor = walkway.WalkerFloor("0", shapely.union_all([shapely.box(-6.0, 0.0, 3.5, 4.0), shapely.box(11.5, 0.0, 21.0, 4.0)]), (), ())
    regions = {"west": [{"x": -5.5, "y": 0.5}, {"x": -4.0, "y": 0.5}, {"x": -4.0, "y": 3.5}, {"x": -5.5, "y": 3.5}], "east": [{"x": 19.0, "y": 0.5}, {"x": 20.5, "y": 0.5}, {"x": 20.5, "y": 3.5}, {"x": 19.0, "y": 3.5}]}
    doc = {"regions": {"east_sink": {"type": "sink", "polygon": regions["east"]}, "west_adult": {"type": "source", "polygon": regions["west"], "config": {"rate_profile": [{"t": 0.0, "rate": 0.01}], "agent": {"agent_type": "adult", "sink_affinity": [{"sink": "east_sink"}]}}}}}
    demand = walkway.WalkerDemand.from_scenario(doc)
    zones = links.world_links(_flights())
    over = walkway.Routes.build(floor, demand.with_links(links.floor_links(zones, _on_floor)))
    assert not over.missing and len(over.routes) == 2
    with pytest.raises(ValueError, match="compose it over the landing"):
        walkway.Routes.build(floor, demand.with_links(zones))


def test_a_flight_that_ends_on_a_landing_with_no_way_on_is_refused() -> None:
    level = _flights()
    level.stairs = level.stairs[:2]  # the lower flight alone: its top half lands on the landing, nothing goes on
    with pytest.raises(ValueError, match="no landing joins it to the floor"):
        links.floor_links(links.world_links(level), _on_floor)


def test_the_twins_stairs_take_walks_from_floor_to_floor() -> None:
    import os

    from arena_simulation_setup.generative.flowedit import demand
    from arena_simulation_setup.generative.flowedit.edit import open_site

    # a local world, outside the fake assets the tests run on: the checkout's own local assets
    root = Path(os.environ.get("ARENA_DIR", "/nonexistent")) / "arena_simulation_setup" / "assets" / "worlds" / "smu_link_full_v3"
    if not (root / "world.yaml").is_file():
        pytest.skip("the local world smu_link_full_v3 is not installed")
    site = open_site(root)
    level_id = next(iter(site.world.levels))
    halves = links.world_links(site.world.levels[level_id])
    assert len(halves) == 12
    walks = demand.site_walks(site, level_id)
    # both stair lanes and both escalators join the lower concourse (x ~132) and the deck (x ~123): the climb from the
    # stair foot to the deck is far shorter over the stairs than step-free round the west ramp
    foot, deck = (134.0, 41.0), (121.5, 41.0)
    climbing = float(walks.lengths([foot], [deck], climbs=True)[0, 0])
    step_free = float(walks.lengths([foot], [deck], climbs=False)[0, 0])
    assert climbing < 30.0 < step_free
