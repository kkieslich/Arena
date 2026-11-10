"""FLOWEDIT data v2's held-out worlds read into a crowd with their authored demand."""

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
