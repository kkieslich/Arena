"""Crowd-only walls round floor the walkers keep off (the fleet's parking spots): realized, and in place on return."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")


def test_keep_crowd_off_walls_each_polygon_in_the_map_frame_for_the_crowd_alone() -> None:
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.shared import Position

    sent = {}

    async def set_crowd_walls(key: str, segments: object) -> None:
        sent[key] = segments

    manager = SimpleNamespace(realize=lambda p: Position(p.x + 10.0, p.y, p.z), _human_simulator=SimpleNamespace(set_crowd_walls=set_crowd_walls))
    square = [Position(0.0, 0.0), Position(1.0, 0.0), Position(1.0, 2.0), Position(0.0, 2.0)]
    asyncio.run(EnvironmentManager.keep_crowd_off(manager, "parking", [square]))
    assert sent["parking"] == [((10.0, 0.0), (11.0, 0.0)), ((11.0, 0.0), (11.0, 2.0)), ((11.0, 2.0), (10.0, 2.0)), ((10.0, 2.0), (10.0, 0.0))]
    asyncio.run(EnvironmentManager.keep_crowd_off(manager, "parking", []))
    assert sent["parking"] is None


def test_crowd_walls_are_applied_before_the_call_returns_and_drop_a_pending_set_of_the_key() -> None:
    from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator

    applied = []

    async def apply(key: str, segments: object) -> None:
        applied.append((key, segments))

    simulator = SimpleNamespace(_mech_walls_desired={"parking": [((0.0, 0.0), (9.0, 9.0))], "door:a": None}, _apply_mechanism_walls=apply)
    asyncio.run(ArenaHumanSimulator.set_crowd_walls(simulator, "parking", [((0, 0), (1, 0))]))
    assert applied == [("parking", [((0.0, 0.0), (1.0, 0.0))])]
    assert simulator._mech_walls_desired == {"door:a": None}
