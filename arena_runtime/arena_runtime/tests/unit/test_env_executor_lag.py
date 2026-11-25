"""The env's 1 s watchdog timer warns when it ran late: the env executor's queue is that far behind."""

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
    stub_type = type("Stub", (node.EnvNode,), {"get_logger": lambda self: SimpleNamespace(warning=warned.append)})
    stub_type.__abstractmethods__ = frozenset()
    stub = object.__new__(stub_type)
    stub._watchdog_at = None
    stub._lag_warned_at = -node._LAG_WARN_EVERY_S
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
    assert env.warned == ["env executor ran 5.5s behind (its 1 s watchdog timer came late)"]


def test_lag_is_warned_at_most_every_thirty_seconds(env: typing.Any) -> None:
    _ticks(env, 4.0, 4.0, 4.0, 20.0, 4.0)
    assert len(env.warned) == 2
