from __future__ import annotations

import math

import pytest
from arena_simulation_setup.utils.face import front_angle


@pytest.mark.parametrize(
    ("face", "angle"),
    [("+x", 0.0), ("+y", math.pi / 2), ("-x", math.pi), ("-y", -math.pi / 2), ("xy", math.pi / 4), ("90", -math.pi / 2), (-180, 0.0)],
)
def test_a_face_turns_to_the_front_the_runtime_queues_at(face: str | int, angle: float) -> None:
    front = front_angle(face)
    assert front is not None
    assert math.cos(front) == pytest.approx(math.cos(angle), abs=1e-9)
    assert math.sin(front) == pytest.approx(math.sin(angle), abs=1e-9)


@pytest.mark.parametrize("face", ["up", "45", 45.0, True, None])
def test_a_face_that_names_no_side_has_no_front(face: str | float | None) -> None:
    assert front_angle(face) is None
