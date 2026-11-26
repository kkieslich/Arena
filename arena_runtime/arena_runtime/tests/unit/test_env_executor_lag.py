"""The env warns when its 1 s watchdog timer (the executor) or its 0.5 s loop beat (the event loop) came late, with the sim time."""

from __future__ import annotations

import typing
from types import SimpleNamespace

import pytest


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> typing.Any:
    from arena_runtime.env import node

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(node, "time", SimpleNamespace(monotonic=lambda: clock.now))
    warned: list[str] = []
    stub_type = type("Stub", (node.EnvNode,), {"get_logger": lambda self: SimpleNamespace(warning=warned.append), "sim_time": SimpleNamespace(to_seconds=lambda: 812.34)})
    stub_type.__abstractmethods__ = frozenset()
    stub = object.__new__(stub_type)
    stub._watchdog_at = None
    stub.clock, stub.warned = clock, warned
    return stub


def _ticks(env: typing.Any, *gaps: float) -> None:
    env._warn_executor_lag()
    for gap in gaps:
        env.clock.now += gap
        env._warn_executor_lag()


def test_a_timer_on_time_warns_nothing(env: typing.Any) -> None:
    _ticks(env, 1.0, 1.02, 1.9, 1.0)
    assert env.warned == []


def test_a_late_timer_warns_its_lag(env: typing.Any) -> None:
    _ticks(env, 1.0, 6.5)
    assert env.warned == ["env executor ran 5.5s behind at sim 812.3s (its 1 s watchdog timer came late)"]


def test_each_late_firing_is_one_warning(env: typing.Any) -> None:
    _ticks(env, 4.0, 1.0, 29.0, 1.0)
    assert [w.split(" behind")[0] for w in env.warned] == ["env executor ran 3.0s", "env executor ran 28.0s"]


def test_a_late_loop_beat_warns_with_the_sim_time(env: typing.Any) -> None:
    env._note_loop_lag(0.3)
    env._note_loop_lag(8.25)
    assert env.warned == ["env event loop ran 8.2s behind at sim 812.3s (its 0.5 s beat came late)"]
