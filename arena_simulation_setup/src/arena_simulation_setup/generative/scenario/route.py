"""The robot's route in a base scenario: the world's shipped route, cut or shuttled to fit the episode."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .populate import endpoint_xy, robot_endpoints, zone_centroids

MIN_EPISODE_S = 30.0

MAX_EPISODE_S = 60.0
CASE_RESERVE_S = 20.0

#: the robot's speed along its route, for how long it drives it [m/s]: the jackals' measured mean over all moving time in twin recordings on the synced stack
ROBOT_SPEED_MPS = 1.14


ROUTE_MARGIN = 1.2

MAX_LEGS = 8

MAX_ROUTE_M = (MAX_EPISODE_S - CASE_RESERVE_S) * ROBOT_SPEED_MPS


def straight_length(endpoints: Sequence[Any], zones: Sequence[Mapping[str, Any]]) -> float:
    """Meters of the straight lines joining `endpoints` in order, a zone name standing for its centroid."""
    centroids = zone_centroids(zones)
    pts = [p for p in (endpoint_xy(e, centroids) for e in endpoints) if p is not None]
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))


def robot_for(robot: Mapping[str, Any], zones: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The shipped route as `phases`: its shortest prefix lasting `MIN_EPISODE_S` at `ROBOT_SPEED_MPS`, else shuttled to it."""
    ends = robot_endpoints(robot)

    need = MIN_EPISODE_S * ROBOT_SPEED_MPS * ROUTE_MARGIN
    seq = [ends[0]]
    for point in ends[1:]:
        seq.append(point)
        if straight_length(seq, zones) >= min(need, MAX_ROUTE_M) or len(seq) - 1 >= MAX_LEGS:
            break

    if len(ends) >= 2:
        back, forth = list(reversed(ends[:-1])), list(ends[1:])
        bounce = [back, forth]
        turn = 0
        capped = False
        while not capped and straight_length(seq, zones) < need and len(seq) - 1 < MAX_LEGS:
            added = False
            for point in bounce[turn % 2]:
                if len(seq) - 1 >= MAX_LEGS:
                    break
                if straight_length([*seq, point], zones) > MAX_ROUTE_M:
                    capped = True
                    break
                seq.append(point)
                added = True
                if straight_length(seq, zones) >= need:
                    break
            turn += 1
            if not added:
                break
        del seq[MAX_LEGS + 1 :]

    spec: dict[str, Any] = {"start": seq[0], "phases": [{"goto": p} for p in seq[1:]]}
    for key in ("level", "goal_level"):
        if robot.get(key) not in (None, ""):
            spec[key] = robot[key]
    return spec


def route_seconds(robot: Mapping[str, Any], zones: Sequence[Mapping[str, Any]]) -> float:
    return straight_length(robot_endpoints(robot), zones) / ROBOT_SPEED_MPS


__all__ = [
    "CASE_RESERVE_S",
    "MAX_EPISODE_S",
    "MAX_LEGS",
    "MAX_ROUTE_M",
    "MIN_EPISODE_S",
    "ROBOT_SPEED_MPS",
    "ROUTE_MARGIN",
    "robot_for",
    "route_seconds",
    "straight_length",
]
