"""The crowd stays inside the buildings: HumanSim walks its walkers wherever the walls leave room, so the runtime hands it a
wall across every exterior door (`Level.outside_walls`, the crowd's only) where that shuts the building's outside."""

from __future__ import annotations

from pathlib import Path

import pytest
from arena_simulation_setup.tree.World import WorldView

pytest.importorskip("shapely")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
#: scenes whose outside is open other than at their doors (outdoor scenes, a level drawn without outer walls) and
#: hospital_2's 2.4 m2 pocket behind a gap in its outer wall: their doors stay open to the crowd
OPEN = {"map_empty", "mechanisms_elevator", "mechanisms_regimes", "three_storied_residential", "hospital_2"}
FLOWEDIT = ("food_court", "hallway_clinic", "hallway_lab", "hallway_office", "hallway_school", "hospital_1", "library", "office_1", "office_open_plan", "reception", "station_hall", "waiting_hall_clinic")


@pytest.mark.parametrize("world", sorted(p.name for p in WORLDS.iterdir() if (p / "world.yaml").is_file()))
def test_walkers_reach_no_floor_outside_a_building(world: str) -> None:
    for level in WorldView(WORLDS / world).load(validate=False).levels.values():
        walls = level.outside_walls()
        if world in OPEN:
            assert walls == []
        else:
            assert level.outside_reached(walls) < 0.5


@pytest.mark.parametrize("world", FLOWEDIT)
def test_the_exterior_doors_of_a_flowedit_world_are_shut_to_the_crowd(world: str) -> None:
    (level,) = WorldView(WORLDS / world).load(validate=False).levels.values()
    exterior = level.exterior_doors()
    # with the walls alone, walkers leave by one exterior door and walk outside along the building to the next
    assert (level.outside_reached() > 100.0) == bool(exterior)
    assert [(w.start, w.end) for w in level.outside_walls()] == [(d.start, d.end) for d in exterior]
