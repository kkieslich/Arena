from __future__ import annotations

import asyncio

import pytest

from arena_simulation_setup.shared.walls import Wall
from arena_simulation_setup.utils.geometry import Position


def test_wall_iter_yields_start_end():
    w = Wall(start=Position(1.0, 0.0, 0.0), end=Position(2.0, 0.0, 0.0))
    points = list(w)
    assert len(points) == 2
    assert points[0].x == pytest.approx(1.0)
    assert points[1].x == pytest.approx(2.0)


def test_wall_serialize_no_kind_no_material():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0))
    d = w.serialize()
    assert "kind" not in d
    assert "material" not in d


def test_wall_serialize_with_kind():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0), kind="brick")
    d = w.serialize()
    assert d["kind"] == "brick"
    assert "material" not in d


def test_wall_serialize_with_material_no_kind():
    from arena_simulation_setup.tree.assets.Material import MaterialIdentifier

    mat = MaterialIdentifier("Marble")
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0), material=mat)
    d = w.serialize()
    assert "kind" not in d
    assert "material" in d


def test_wall_assets_no_kind_returns_realization():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0))
    walls_iter, obs_iter = asyncio.run(w.assets())
    walls = list(walls_iter)
    # Simple wall should produce at least one WallSegment
    assert len(walls) >= 1


def test_wall_assets_exception_fallback():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(1.0, 0.0, 0.0), kind="nonexistent_wall_type_xyz")
    walls_iter, obs_iter = asyncio.run(w.assets())
    walls = list(walls_iter)
    # Should fall back to a simple wall
    assert len(walls) >= 1


def test_a_sloped_wall_capped_by_top_follows_its_line():
    w = Wall(start=Position(0.0, 0.0, 0.0), end=Position(3.0, 0.0, 1.5), height=1.0)
    (segment,) = list(asyncio.run(w.assets())[0])
    assert (segment.start.z, segment.end.z, segment.height) == pytest.approx((0.0, 1.5, 1.0))


def test_a_lintel_on_a_sloped_wall_sits_bottom_above_the_line_at_both_ends():
    w = Wall(start=Position(0.0, 0.0, 1.0), end=Position(2.0, 0.0, 2.0), top=0.9, bottom=0.85)
    (segment,) = list(asyncio.run(w.assets())[0])
    assert (segment.start.z, segment.end.z, segment.height) == pytest.approx((1.85, 2.85, 0.05))


def test_height_cuts_a_layered_stack_drops_layers_above_it_and_top_only_raises():
    from arena_simulation_setup.tree.Wall import PlaceWallSegmentAsset, WallDescription

    layers = WallDescription(main=[PlaceWallSegmentAsset(height=1.0), PlaceWallSegmentAsset(height=0.75, z=1.0), PlaceWallSegmentAsset(height=0.25, z=1.75)])
    start, end = Position(0.0, 0.0, 0.5), Position(2.0, 0.0, 0.5)
    cut, _ = Wall(start=start, end=end, height=1.2)._close_to(layers.realize(start, end))
    assert [value for segment in cut for value in (segment.start.z, segment.height)] == pytest.approx([0.5, 1.0, 1.5, 0.2])
    raised, _ = Wall(start=start, end=end, top=1.2)._close_to(layers.realize(start, end))
    assert [value for segment in raised for value in (segment.start.z, segment.height)] == pytest.approx([0.5, 1.0, 1.5, 0.75, 2.25, 0.25])


def test_layers_and_tiles_of_a_sloped_wall_keep_their_height_above_its_line():
    from arena_simulation_setup.tree.Wall import PlaceWallSegmentAsset, TilingAsset, WallDescription

    rail = WallDescription(main=[PlaceWallSegmentAsset(height=0.05, z=0.85), TilingAsset(tile=[PlaceWallSegmentAsset(height=0.85, width=0.03)], every=0.9, width=0.03)])
    segments, _ = rail.realize(Position(0.0, 0.0, 0.0), Position(2.6, 0.0, 1.5))
    bar, *posts = list(segments)
    assert (bar.start.x, bar.start.z, bar.end.x, bar.end.z) == pytest.approx((0.0, 0.85, 2.6, 2.35))
    assert posts and all(post.start.z == pytest.approx(post.start.x * 1.5 / 2.6) for post in posts)
