"""The arena backend has the engine hold its clock after the reset's remove-all and starts it at the episode start."""

from __future__ import annotations

import asyncio
import typing

import pytest

pytest.importorskip("rclpy")
srv = pytest.importorskip("arena_humansim_msgs.srv")
if not hasattr(srv, "StartClock"):
    pytest.skip("arena_humansim_msgs built before StartClock", allow_module_level=True)

from arena_humansim_msgs.srv import StartClock
from arena_rclpy_mixins.Time import Time
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.humans.arena_humansim.engine_inputs import EngineInputs
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.srv import SetParametersAtomically


class _Client:
    def __init__(self, response: object, *more: object) -> None:
        self.responses = [response, *more]
        self.requests: list[object] = []

    async def call_timeout(self, request: object) -> object:
        self.requests.append(request)
        return self.responses[min(len(self.requests), len(self.responses)) - 1]

    async def ensure(self) -> bool:
        return True


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def warning(self, line: str) -> None:
        self.lines.append(line)

    error = info = warning

    def get_child(self, _name: str) -> _Logger:
        return self


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger


def _params(accept: bool) -> SetParametersAtomically.Response:
    return SetParametersAtomically.Response(result=SetParametersResult(successful=accept, reason="" if accept else "unknown parameter"))


def _adapter(accept: bool, started: bool = True, barrier: bool | None = None, gated: bool = True) -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._NodeInterface__node = _Node()  # pyright: ignore[reportAttributeAccessIssue]
    barrier = accept if barrier is None else barrier
    adapter._set_params_client = _Client(_params(barrier), _params(accept))
    adapter._start_clock_client = _Client(StartClock.Response(success=started, message="first tick" if started else "ticking since 0 ns"))
    adapter._clock_on_request = False
    adapter._clock_started = False
    adapter._input_barrier = False
    adapter._deferred_spawns = []
    adapter._deferred_unanchored = []
    adapter._feedback_loop_task = None
    adapter._engine_inputs = EngineInputs(lambda: 0)

    async def lockstep_gates() -> bool:
        return gated

    adapter._lockstep_gates = lockstep_gates  # pyright: ignore[reportAttributeAccessIssue]
    return adapter


def _run(coro: typing.Awaitable[None]) -> None:
    asyncio.run(coro)  # pyright: ignore[reportArgumentType]


def test_an_engine_that_takes_start_on_request_is_started_at_the_episode_start() -> None:
    adapter = _adapter(accept=True)
    _run(adapter._hold_clock_until_start())
    [request] = adapter._set_params_client.requests
    assert [(p.name, p.value.bool_value) for p in request.parameters] == [("start_on_request", True), ("input_barrier", True)]
    assert adapter._input_barrier
    epoch = Time(sec=41, nanosec=16_800_000)
    _run(adapter.start_clock(epoch))
    [start] = adapter._start_clock_client.requests
    assert (start.epoch.sec, start.epoch.nanosec) == (41, 16_800_000)


def test_an_engine_without_start_on_request_keeps_its_clock_and_is_never_asked_to_start() -> None:
    adapter = _adapter(accept=False)
    _run(adapter._hold_clock_until_start())
    _run(adapter.start_clock(Time(sec=3)))
    assert adapter._start_clock_client.requests == []
    assert any("seed-paired episodes can differ" in line for line in adapter.node.logger.lines)


def test_a_refused_start_is_logged_not_raised() -> None:
    adapter = _adapter(accept=True, started=False)
    _run(adapter._hold_clock_until_start())
    _run(adapter.start_clock(Time(sec=3)))
    assert any("did not start its clock" in line for line in adapter.node.logger.lines)


def test_an_engine_without_the_input_barrier_still_holds_its_clock() -> None:
    adapter = _adapter(accept=True, barrier=False)
    _run(adapter._hold_clock_until_start())
    first, second = adapter._set_params_client.requests
    assert [p.name for p in first.parameters] == ["start_on_request", "input_barrier"]
    assert [p.name for p in second.parameters] == ["start_on_request"]
    assert adapter._clock_on_request and not adapter._input_barrier
    assert any("takes no input barrier" in line for line in adapter.node.logger.lines)


def test_a_walker_delayed_before_the_start_comes_in_its_delay_after_the_epoch() -> None:
    adapter = _adapter(accept=True)
    _run(adapter._hold_clock_until_start())
    msg = object()
    adapter._deferred_unanchored.append((2.5, msg, {}))
    _run(adapter.start_clock(Time(sec=41, nanosec=500_000_000)))
    assert adapter._deferred_spawns == [(44.0, msg, {})]
    assert adapter._deferred_unanchored == []
    assert adapter._clock_started


def test_without_a_gating_lockstep_the_engine_does_not_wait_for_acknowledgements() -> None:
    adapter = _adapter(accept=True, gated=False)
    _run(adapter._hold_clock_until_start())
    [request] = adapter._set_params_client.requests
    assert [p.name for p in request.parameters] == ["start_on_request"]
    assert adapter._clock_on_request and not adapter._input_barrier
