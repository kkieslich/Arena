"""A scenario's own teleport zones: stair and escalator halves joining floor of one level, as HumanSim reads them."""

from __future__ import annotations

import math

import pytest
from arena_simulation_setup.tree.World.Scenario import TeleportZone, parse_scenario

TOP = {
    "name": "up",
    "kind": "escalator",
    "destinations": ["down"],
    "entry": {"x": 1.0, "y": 2.0, "theta": 0.0},
    "wait": {"x": 0.2, "y": 2.0, "theta": 0.0},
    "hold": {"x": 2.5, "y": 2.0},
    "footprint": [{"x": 1.0, "y": 1.5}, {"x": 3.0, "y": 1.5}, {"x": 3.0, "y": 2.5}, {"x": 1.0, "y": 2.5}],
    "belt": 0.5,
}
BOTTOM = {"name": "down", "kind": "escalator", "entry": {"x": 9.0, "y": 2.0, "theta": math.pi}, "hold": {"x": 7.5, "y": 2.0, "theta": math.pi}, "footprint": [{"x": 7.0, "y": 1.5}, {"x": 9.0, "y": 1.5}, {"x": 9.0, "y": 2.5}, {"x": 7.0, "y": 2.5}]}


def test_teleport_zones_parse_in_humansim_form() -> None:
    zones = parse_scenario({"format": "arena-scenario/2", "teleport_zones": [TOP, BOTTOM]}).teleport_zones
    up, down = zones
    assert up == TeleportZone("up", ("down",), (1.0, 2.0, 0.0), (0.2, 2.0, 0.0), (2.5, 2.0, 0.0), ((1.0, 1.5), (3.0, 1.5), (3.0, 2.5), (1.0, 2.5)), True, 0.0, "escalator", 0.5)
    assert down.destinations == () and down.wait == down.entry and down.belt == 0.0 and down.open
    assert parse_scenario({"format": "arena-scenario/2", "teleport_zones": [{**TOP, "delay": 14.05}, BOTTOM]}).teleport_zones[0].delay == 14.05
    assert parse_scenario({"format": "arena-scenario/2"}).teleport_zones == []
    # a cabin's negative delay: HumanSim leaves its teleport to the caller
    cabins = [{"name": "lift_a", "destinations": ["lift_b"], "entry": {"x": 0.0, "y": 0.0}, "hold": {"x": 0.5, "y": 0.0}, "delay": -1.0}, {"name": "lift_b", "destinations": ["lift_a"], "entry": {"x": 9.0, "y": 0.0}, "hold": {"x": 9.5, "y": 0.0}, "delay": -1.0}]
    assert [z.delay for z in parse_scenario({"format": "arena-scenario/2", "teleport_zones": cabins}).teleport_zones] == [-1.0, -1.0]


@pytest.mark.parametrize(
    ("zones", "match"),
    [
        ([TOP], "unknown zones"),
        ([BOTTOM, BOTTOM], "repeat"),
        ([{**BOTTOM, "kind": "ladder"}], "kind"),
        ([{k: v for k, v in BOTTOM.items() if k != "footprint"}], "footprint"),
        ([{**BOTTOM, "colour": "red"}], "takes name"),
        ([{**BOTTOM, "hold": {"x": 1.0}}], "hold"),
        # as HumanSim checks them: a hold a walker reaches, finite numbers, no flight back in time, no landing in itself
        ([{**BOTTOM, "hold": {"x": 7.0, "y": 2.0}}], "more than 0.25 m inside the footprint"),
        ([{**BOTTOM, "hold": {"x": 7.2, "y": 2.0}}], "more than 0.25 m inside the footprint"),
        ([{**BOTTOM, "hold": {"x": 5.0, "y": 2.0}}], "more than 0.25 m inside the footprint"),
        ([{**BOTTOM, "hold": {"x": 7.5, "y": 1.7}}], "more than 0.25 m inside the footprint"),
        ([{**BOTTOM, "delay": float("nan")}], "finite"),
        ([{**BOTTOM, "entry": {"x": float("inf"), "y": 2.0}}], "finite"),
        ([{**BOTTOM, "delay": -1.0}], "half's delay must be >= 0"),
        ([{**BOTTOM, "destinations": ["down"]}], "lands in itself"),
        ([{**BOTTOM, "destinations": ["lift"]}, {"name": "lift", "entry": {"x": 0.0, "y": 0.0}, "hold": {"x": 0.5, "y": 0.0}}], "lands only in stair or escalator halves"),
    ],
)
def test_teleport_zones_are_checked(zones: list[dict], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_scenario({"format": "arena-scenario/2", "teleport_zones": zones})
