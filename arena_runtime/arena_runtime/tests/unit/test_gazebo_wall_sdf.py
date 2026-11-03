"""Wall SDF generation honors the solid and shadows segment flags."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")

from arena_simulation_setup.tree.Wall import WallSegment  # noqa: E402
from arena_simulation_setup.utils.geometry import Position  # noqa: E402

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _generate_wall_sdf  # noqa: E402


def _links(*segments: WallSegment) -> list[ET.Element]:
    sdf = _generate_wall_sdf("wall", [(segment, {}) for segment in segments])
    return ET.fromstring(sdf).findall("./model/link")


def _segment(**flags: bool) -> WallSegment:
    return WallSegment(start=Position(0.0, 0.0, 0.0), end=Position(4.0, 0.0, 0.0), height=2.5, width=0.05, **flags)


def test_default_segment_has_visual_collision_and_shadows() -> None:
    (link,) = _links(_segment())
    assert link.find("visual") is not None
    assert link.find("collision") is not None
    assert link.findtext("visual/cast_shadows") == "true"


def test_non_solid_segment_has_no_collision() -> None:
    (link,) = _links(_segment(solid=False))
    assert link.find("visual") is not None
    assert link.find("collision") is None


def test_shadowless_segment_does_not_cast_shadows() -> None:
    (link,) = _links(_segment(shadows=False))
    assert link.findtext("visual/cast_shadows") == "false"


def test_camera_hidden_segment_stays_drawn_and_solid() -> None:
    (link,) = _links(_segment(visible=False))
    assert link.find("visual") is not None
    assert link.find("collision") is not None


def test_segments_keep_their_own_flags_in_one_model() -> None:
    line, backdrop = _links(_segment(visible=False, shadows=False), _segment(solid=False, shadows=False))
    assert line.find("collision") is not None
    assert backdrop.find("collision") is None
    assert [link.findtext("visual/cast_shadows") for link in (line, backdrop)] == ["false", "false"]


def test_sloped_segment_is_pitched_along_its_incline() -> None:
    rail = WallSegment(start=Position(0.0, 0.0, 0.0), end=Position(0.0, 3.0, 1.5), height=1.0, width=0.05)
    (link,) = _links(rail)
    x, y, z, roll, pitch, yaw = (float(v) for v in link.findtext("pose").split())
    assert (x, y, z) == pytest.approx((0.0, 1.5, 0.75 + 0.5))
    assert (roll, pitch, yaw) == pytest.approx((0.0, -math.atan2(1.5, 3.0), math.pi / 2.0))
    size = [float(v) for v in link.findtext("visual/geometry/box/size").split()]
    assert size == pytest.approx([math.hypot(3.0, 1.5), 0.05, math.cos(math.atan2(1.5, 3.0))])
    assert link.findtext("collision/geometry/box/size") == link.findtext("visual/geometry/box/size")


def test_a_post_on_a_sloped_line_stands_upright() -> None:
    post = WallSegment(start=Position(0.0, 0.0, 0.0), end=Position(0.03, 0.0, 0.017), height=0.85, width=0.03)
    (link,) = _links(post)
    _x, _y, z, roll, pitch, _yaw = (float(v) for v in link.findtext("pose").split())
    assert (roll, pitch) == (0.0, 0.0)
    assert z == pytest.approx(0.85 / 2.0)


def test_a_sloped_panel_stacks_into_strips_whose_ends_stand_within_a_centimetre_of_vertical() -> None:
    from arena_runtime.sim._walls import WALL_LEAN, sloped_box, upright_strips

    panel = WallSegment(start=Position(0.0, 0.0, 0.0), end=Position(2.6, 0.0, 1.5), height=1.0, width=0.05)
    *strips, low_cap, high_cap = upright_strips(panel)
    assert len(strips) > 1
    assert sum(strip.height for strip in strips) == pytest.approx(1.0)
    assert [strip.start.z for strip in strips] == pytest.approx([k * strips[0].height for k in range(len(strips))])
    for cap, end in ((low_cap, panel.start), (high_cap, panel.end)):
        assert sloped_box(cap) is None
        assert (cap.start.z, cap.end.z, cap.height) == (end.z, end.z, 1.0)
        assert (cap.start.x + cap.end.x) / 2.0 == pytest.approx(end.x)
        assert 0.0 < cap.end.x - cap.start.x <= 2.0 * WALL_LEAN + 1e-9
    for strip in strips:
        (_length, _width, thickness), pose = sloped_box(strip)
        assert thickness * math.sin(abs(pose.orientation.to_euler()[1])) / 2.0 <= WALL_LEAN + 1e-9
    level = WallSegment(start=Position(0.0, 0.0, 0.0), end=Position(2.6, 0.0, 0.0), height=1.0, width=0.05)
    assert upright_strips(level) == [level]
