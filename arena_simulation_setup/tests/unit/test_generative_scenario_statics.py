from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.generative.scenario.formats import first_level
from arena_simulation_setup.generative.scenario.occupancy import RESOLUTION, level_grid, world_statics

_WORLD = {
    "format": "arena-world/2",
    "levels": [
        {
            "id": "0",
            "zones": [{"name": "lobby", "polygon": [[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0]], "floor": False}],
            "objects": [
                {"name": "counter", "pose": [4.0, 2.8, 0.0], "model": "Hospital/TestDesk"},
                {"name": "plant", "pose": [8.0, 5.0, 0.0], "model": "Hospital/TestPlant"},
                {"name": "kiosk", "pose": [2.0, 1.0, 0.0], "model": "Hospital/WorldKiosk"},
            ],
        }
    ],
}


@pytest.fixture
def world(tmp_path: Path) -> Path:
    from arena_simulation_setup.tree import DynamicPaths

    root = tmp_path / "world"
    root.mkdir()
    (root / "world.yaml").write_text(yaml.safe_dump(_WORLD))
    desk = tmp_path / "assets" / "Hospital" / "Object" / "TestDesk"
    desk.mkdir(parents=True)
    (desk / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[0.0, 2.42], [-2.76, 0.10], [0.0, 1.0]]}))
    (tmp_path / "assets" / "Hospital" / "Object" / "TestPlant").mkdir(parents=True)
    kiosk = root / "assets" / "Hospital" / "Object" / "WorldKiosk"
    kiosk.mkdir(parents=True)
    (kiosk / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]}))
    with DynamicPaths.ARENA.pinned(tmp_path / "assets"):
        yield root


def test_an_annotated_object_is_its_rotated_bounds(world: Path) -> None:
    statics = {s["name"]: s for s in world_statics(world)}
    xs, ys = zip(*statics["counter"]["polygon"], strict=True)
    assert (min(xs), round(max(xs), 2), round(min(ys), 2), round(max(ys), 2)) == (4.0, 6.42, 0.04, 2.9)
    assert statics["counter"]["radius"] == pytest.approx(math.hypot(2.42, 2.76))


def test_an_unannotated_object_is_the_runtime_box(world: Path) -> None:
    plant = next(s for s in world_statics(world) if s["name"] == "plant")
    xs, ys = zip(*plant["polygon"], strict=True)
    assert (min(xs), max(xs), min(ys), max(ys)) == (7.0, 9.0, 4.0, 6.0)
    assert not plant["resolved"]


def test_promptgen_names_the_assets_without_a_footprint(world: Path) -> None:
    from arena_simulation_setup.generative.scenario.promptgen.world import LoadedWorld

    unresolved = LoadedWorld.read(world).unresolved
    assert len(unresolved) == 1
    assert unresolved[0].endswith("TestPlant")


def test_an_annotated_object_is_marked_resolved(world: Path) -> None:
    assert {s["name"]: s["resolved"] for s in world_statics(world)} == {"counter": True, "plant": False, "kiosk": True}


def test_an_asset_shipped_with_the_world_resolves(world: Path) -> None:
    kiosk = next(s for s in world_statics(world) if s["name"] == "kiosk")
    xs, ys = zip(*kiosk["polygon"], strict=True)
    assert (min(xs), max(xs), min(ys), max(ys)) == (1.5, 2.5, 0.5, 1.5)


def test_the_desk_is_stamped_as_its_box(world: Path) -> None:
    occupied, res, ox, oy = level_grid(world, world_statics(world))
    assert res == RESOLUTION

    def cell(x: float, y: float) -> bool:
        return bool(occupied[occupied.shape[0] - 1 - int((y - oy) / res), int((x - ox) / res)])

    assert cell(5.0, 1.5) and cell(6.3, 0.2) and not cell(3.5, 1.5) and not cell(4.0, 4.0)


def test_the_bare_grid_is_the_rendered_level(world: Path) -> None:
    occupied, res, ox, oy = level_grid(world)

    def cell(x: float, y: float) -> bool:
        return bool(occupied[occupied.shape[0] - 1 - int((y - oy) / res), int((x - ox) / res)])

    assert not cell(5.0, 1.5) and not cell(8.0, 5.0)
    assert cell(-0.1, 3.0) and cell(5.0, 6.1)
    assert level_grid(world.parent / "nowhere") is None


def test_objects_sit_in_the_zone_that_covers_them(world: Path) -> None:
    (lobby,) = first_level(world)["zones"]
    assert [e["model"] for e in lobby["entities"]["static"]] == ["TestDesk", "TestPlant", "WorldKiosk"]
