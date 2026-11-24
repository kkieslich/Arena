"""An object with an empty model is a HumanSim target only: the robot simulator never spawns, moves or deletes it."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import pytest

pytest.importorskip("rclpy")


class _Sim:
    def __init__(self) -> None:
        self.spawned: list[str] = []
        self.moved: list[str] = []

    async def spawn_objects(self, objects: Sequence) -> list[bool]:
        self.spawned += [o.name for o in objects]
        return [True] * len(objects)

    async def move_objects(self, objects: Sequence) -> list[bool]:
        self.moved += [o.name for o in objects]
        return [True] * len(objects)


def _humans(sim: _Sim):
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.humans.utils import KnownObstacles

    class Humans(NoopHumanSimulator):
        """The no-op human simulator without a node: its logger, registry and robot simulator set directly."""

        _logger = logging.getLogger("test")
        _simulator = sim

    humans = object.__new__(Humans)
    humans._known_obstacles = KnownObstacles()
    return humans


def _obstacle(name: str, model: str):
    from arena_runtime.shared import Obstacle

    return Obstacle.parse({"name": name, "model": model, "pose": [1.0, 2.0, 0.0], "humansim": {"type": "fountain", "capacity": 1}})


def test_a_target_only_object_is_registered_but_never_reaches_the_robot_simulator() -> None:
    sim = _Sim()
    humans = _humans(sim)
    target, body = _obstacle("fountain_demand", ""), _obstacle("fountain", "Hospital/SM_WaterDispenser_01a")
    assert target.target_only and not body.target_only
    asyncio.run(humans.spawn_obstacles([target, body]))
    assert sim.spawned == ["fountain"]
    assert humans._known_obstacles.get("fountain_demand").spawned
    asyncio.run(humans.spawn_obstacles([target, body]))
    assert sim.moved == ["fountain"]
