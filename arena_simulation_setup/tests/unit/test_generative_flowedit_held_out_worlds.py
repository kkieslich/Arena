"""FLOWEDIT data v2's further worlds read into a crowd with their authored demand."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit.edit import open_site
from arena_simulation_setup.generative.flowedit.generate import demand_objects
from arena_simulation_setup.generative.flowedit.read import read

pytestmark = pytest.mark.usefixtures("local_root")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"


@pytest.mark.parametrize(("world", "entrances"), [("reception", []), ("hallway_clinic", ["back_door", "front_door"])])
def test_a_held_out_world_reads_people_who_seek_its_authored_demand(world: str, entrances: list[str]) -> None:
    site = open_site(WORLDS / world, "demand", None, [])
    assert demand_objects(site) == ["obj:fountain"]
    notes: list[str] = []
    result = read(site, S.load(), 6, notes)
    thirst = result.provenance["needs"]["thirst"]["objects"]
    assert len(thirst) == 1 and str(thirst[0]).endswith("fountain")
    # people start in the rooms the kg gives a purpose, and entrances sit at the doors that lead outside
    assert len(result.doc["humans"]) >= 4
    assert result.provenance["entrances"] == entrances


def test_the_clinic_records_the_generator_call_its_layout_came_from() -> None:
    made = yaml.safe_load((WORLDS / "hallway_clinic" / "generated_from.yaml").read_text())
    assert made["generator"] == "hallway" and made["seed"] == 7 and made["config"]["rooms_per_side"] == 4
    world = yaml.safe_load((WORLDS / "hallway_clinic" / "world.yaml").read_text())
    zones = [z["name"] for z in world["levels"][0]["zones"]]
    assert len(zones) == len(set(zones)) == 9 and "main_hallway" in zones


NEW_WORLDS = {
    # world: entrances (its exterior doors)
    "hallway_office": ["east_entrance", "west_entrance"],
    "hallway_school": ["east_entrance", "west_entrance"],
    "hallway_lab": ["east_entrance", "west_entrance"],
    "waiting_hall_clinic": ["front_door_east", "front_door_west", "staff_exit"],
    "library": ["cafe_street_door", "front_door", "side_door"],
    "food_court": ["east_door", "south_door", "west_street_door"],
    "station_hall": ["east_door", "platform_1_stairs", "platform_2_stairs", "west_door"],
    "office_open_plan": ["back_door", "front_door"],
}


@pytest.mark.parametrize(("world", "entrances"), sorted(NEW_WORLDS.items()))
def test_each_further_world_reads_people_who_seek_water_and_enter_by_its_exterior_doors(world: str, entrances: list[str]) -> None:
    site = open_site(WORLDS / world, "demand", None, [])
    assert "obj:fountain" in demand_objects(site)
    result = read(site, S.load(), 6, [])
    assert result.provenance["needs"]["thirst"]["objects"] and len(result.doc["humans"]) >= 4
    assert result.provenance["entrances"] == entrances
    # a hallway variant names its generator call, a hand-drawn plan its rooms and doors
    zones = {z["name"] for z in yaml.safe_load((WORLDS / world / "world.yaml").read_text())["levels"][0]["zones"]}
    if world.startswith("hallway_"):
        assert yaml.safe_load((WORLDS / world / "generated_from.yaml").read_text())["generator"] == "hallway"
    else:
        assert set(yaml.safe_load((WORLDS / world / "plan.yaml").read_text())["rooms"]) == zones
