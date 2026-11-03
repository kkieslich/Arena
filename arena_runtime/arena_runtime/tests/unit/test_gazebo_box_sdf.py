"""Box primitives carry the look of the material they were spawned with."""

from __future__ import annotations

import math

import pytest
from arena_simulation_setup.utils.geometry import Orientation

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _body_frame, _generate_box_sdf  # noqa: E402

SIZE = (1.0, 0.1, 2.0)


def test_box_sdf_binds_the_resolved_textures() -> None:
    sdf = _generate_box_sdf("door", SIZE, textures={"albedo": "/m/wood.png", "normal": "/m/wood_n.png"})
    assert "<albedo_map>/m/wood.png</albedo_map>" in sdf
    assert "<normal_map>/m/wood_n.png</normal_map>" in sdf
    assert "<size>1.0 0.1 2.0</size>" in sdf


@pytest.mark.parametrize("textures", [None, {}])
def test_box_sdf_is_flat_gray_without_textures(textures: dict[str, str] | None) -> None:
    sdf = _generate_box_sdf("door", SIZE, textures=textures)
    assert "<albedo_map>" not in sdf
    assert "<diffuse>0.7 0.7 0.7 1</diffuse>" in sdf


def test_static_box_is_static_and_a_moving_box_is_driven_without_gravity() -> None:
    still = _generate_box_sdf("step", SIZE)
    moving = _generate_box_sdf("tread", SIZE, velocity=(0.5, 0.0, 0.2))
    assert ("<static>true</static>" in still, "VelocityControl" in still) == (True, False)
    assert "<static>false</static>" in moving
    assert "<gravity>false</gravity>" in moving
    assert "<initial_linear>0.5 0.0 0.2</initial_linear>" in moving
    assert ("<visual" in moving, "<collision" in moving) == (True, True)


def test_moving_boxes_pass_through_comb_boxes_only() -> None:
    plain = _generate_box_sdf("step", SIZE)
    comb = _generate_box_sdf("plate", SIZE, comb=True)
    moving = _generate_box_sdf("tread", SIZE, velocity=(0.5, 0.0, 0.2), comb=True)
    assert "<collide_bitmask>" not in plain
    assert "<collide_bitmask>0x01</collide_bitmask>" in comb
    assert "<collide_bitmask>0xfffe</collide_bitmask>" in moving


def test_body_frame_undoes_the_box_orientation() -> None:
    yawed = Orientation.from_yaw(math.pi / 2.0)
    assert _body_frame(yawed, (0.0, 0.5, 0.2)) == pytest.approx((0.5, 0.0, 0.2))
    pitched = Orientation.from_euler((0.0, -0.4, 0.0))
    along = (0.5 * math.cos(0.4), 0.0, 0.5 * math.sin(0.4))
    assert _body_frame(pitched, along) == pytest.approx((0.5, 0.0, 0.0))
