"""The env's sim tick: a robot whose tick awaits a late reply (a goal its action server does not accept, a lifecycle query)
runs on alone, the other robots are judged every tick, the late one again once its tick is done, reported with its duration."""

from __future__ import annotations

import asyncio
import typing
from types import SimpleNamespace

import pytest


class _Server:
    """A goto_pose action server that accepts a goal only when told to."""

    def __init__(self) -> None:
        self.accept = asyncio.Event()

    async def send_goal(self) -> None:
        await self.accept.wait()


def _robot_base() -> type:
    from arena_runtime.env.robot_manager.tick_in_flight import TickInFlight

    return TickInFlight


class _Robot(_robot_base()):  # pyright: ignore[reportUntypedBaseClass]
    def __init__(self, name: str, server: _Server | None = None, error: Exception | None = None) -> None:
        self.name = name
        self.runner = SimpleNamespace(active=None, changed=False)
        self.server = server
        self.error = error
        self.ticks = 0
        self.ended: list[str] = []

    def publish_task_pose(self, pose: object) -> None:
        del pose

    def publish_charge_power(self, watts: float) -> None:
        del watts

    async def tick(self, sample: object, zones: object) -> typing.Any:
        del sample, zones
        self.ticks += 1
        if self.error is not None:
            raise self.error
        if self.server is not None:
            await self.server.send_goal()  # dispatching its next phase: waits for the server to take the goal
        return SimpleNamespace(ended=f"{self.name} phase {self.ticks}")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> typing.Any:
    from arena_runtime.env import node

    monkeypatch.setattr(node, "_ROBOT_TICK_WAIT_S", 0.05)
    monkeypatch.setattr(node, "_LATE_ROBOT_TICK_S", 0.1)
    warned: list[str] = []
    flags: list[str] = []
    stub_type = type("Stub", (node.EnvNode,), {"get_logger": lambda self: SimpleNamespace(warning=warned.append), "_compromise": lambda self, reason: flags.append(reason)})
    stub_type.__abstractmethods__ = frozenset()
    stub = object.__new__(stub_type)
    stub._chargers = SimpleNamespace(step=lambda sample, zones: {})
    stub._sample = SimpleNamespace(robots={})
    stub._judge_zones = {}
    stub._semantics_dirty = False
    stub.warned, stub.flags = warned, flags
    stub.with_robots = lambda *robots: setattr(stub, "_robots_manager", SimpleNamespace(managers={r.name: r for r in robots}))
    return stub


def test_robots_that_answer_at_once_end_their_phases_in_the_same_tick(env: typing.Any) -> None:
    a, b = _Robot("a"), _Robot("b")
    env.with_robots(a, b)
    assert asyncio.run(env._tick_robots()) == {"a": "a phase 1", "b": "b phase 1"}
    assert env.flags == [] and a.tick_task is None and b.tick_task is None


def test_a_late_goal_reply_does_not_stop_the_other_robots(env: typing.Any) -> None:
    server = _Server()
    slow, fast = _Robot("slow", server), _Robot("fast")
    env.with_robots(slow, fast)

    async def scenario() -> list[dict[str, str]]:
        rounds = [await env._tick_robots() for _ in range(3)]
        await asyncio.sleep(0.15)  # the server still holds the goal, past the late mark
        rounds.append(await env._tick_robots())
        server.accept.set()
        await asyncio.sleep(0)
        rounds.append(await env._tick_robots())
        rounds.append(await env._tick_robots())
        return rounds

    rounds = asyncio.run(scenario())
    assert fast.ticks == 6 and slow.ticks == 2  # the slow one ticks again on the tick after its late one ended a phase
    assert [r.get("fast") for r in rounds] == [f"fast phase {i}" for i in range(1, 7)]
    assert [r.get("slow") for r in rounds] == [None, None, None, None, "slow phase 1", "slow phase 2"]
    assert any(w.startswith("robot slow: its tick has waited") for w in env.warned)
    assert len(env.flags) == 1 and env.flags[0].startswith("robot slow tick took ")


def test_a_failing_robot_tick_is_reported_and_the_others_keep_their_transitions(env: typing.Any) -> None:
    env.with_robots(_Robot("broken", error=RuntimeError("send_goal to 'goto_pose' got no response in 30s")), _Robot("fine"))
    assert asyncio.run(env._tick_robots()) == {"fine": "fine phase 1"}
    assert env.flags == ["robot broken tick failed after 0.0s: RuntimeError(\"send_goal to 'goto_pose' got no response in 30s\")"]


def test_a_reset_or_removal_of_the_robot_cancels_its_tick_in_flight(env: typing.Any) -> None:
    server = _Server()
    slow = _Robot("slow", server)
    env.with_robots(slow)

    async def scenario() -> None:
        await env._tick_robots()
        pending = slow.tick_task
        assert pending is not None
        await slow.cancel_tick()  # what RobotsManager.set_up (a new episode) and RobotManager.destroy do
        assert pending.cancelled() and slow.tick_task is None
        assert await env._tick_robots() == {}  # a fresh tick, waiting on the server again
        await slow.cancel_tick()

    asyncio.run(scenario())
    assert slow.ticks == 2
