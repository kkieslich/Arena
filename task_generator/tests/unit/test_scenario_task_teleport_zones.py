"""The scenario task hands the scenario's teleport zones to the human simulator each episode, and fails when it refuses them."""

from __future__ import annotations

import asyncio
import math
import typing

import pytest

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World.Scenario import TeleportZone


class _Env:
    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.calls: list[list] = []

    async def setup_teleport_zones(self, zones: list) -> bool:
        self.calls.append(list(zones))
        return self.accept


def _zone() -> TeleportZone:
    from arena_simulation_setup.tree.World.Scenario import TeleportZone

    return TeleportZone("up", (), (1.0, 2.0, 0.0), (0.2, 2.0, 0.0), (2.0, 2.0, math.pi), ((1.0, 1.5), (3.0, 1.5), (3.0, 2.5), (1.0, 2.5)), True, 14.05, "stairs", 0.0)


def test_the_zones_go_to_the_environment_each_episode():
    from task_generator.tasks.obstacles.scenario.impl import hand_over_teleport_zones

    env = _Env(accept=True)
    asyncio.run(hand_over_teleport_zones(env, "study", [_zone()]))
    asyncio.run(hand_over_teleport_zones(env, "study", []))
    assert env.calls == [[_zone()], []]


def test_an_episode_whose_zones_the_human_simulator_refuses_fails():
    from task_generator.tasks.obstacles.scenario.impl import hand_over_teleport_zones

    with pytest.raises(RuntimeError, match="'study': the human simulator refused its teleport zones"):
        asyncio.run(hand_over_teleport_zones(_Env(accept=False), "study", [_zone()]))
