"""The front of a model from its annotated `face`."""

from __future__ import annotations

import math

from arena_assets.impl import Face


def _front(face: Face) -> float:
    return math.remainder(math.radians(face.angle + 180.0), math.tau)


def front_angle(face: str | float | None) -> float | None:
    """Angle in rad of an annotated `face` (an arena_assets `Face` value or a legacy numeric angle) in the model frame, None when it names no face."""
    if isinstance(face, str):
        try:
            return _front(Face(face))
        except ValueError:
            try:
                face = float(face)
            except ValueError:
                return None
    if isinstance(face, (int, float)) and not isinstance(face, bool):
        return next((_front(f) for f in Face if f.angle == float(face)), None)
    return None


def queue_theta(yaw: float, front: float | None) -> float:
    """The theta HumanSim gets for an object at `yaw` whose front lies at `front` in its frame: HumanSim lines people up behind theta, so theta turns until that side is the front; the yaw itself without a front."""
    if front is None:
        return yaw
    theta = yaw + front - math.pi
    return math.atan2(math.sin(theta), math.cos(theta))


__all__ = ["front_angle", "queue_theta"]
