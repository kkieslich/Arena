"""The crowd render and the flatsim relay coalesce a backlog of rate ticks into one pass and yield after it; ticks that
arrive one at a time are each handled, as before."""

from __future__ import annotations

import asyncio
import contextlib
import typing
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")


def _sim(ticks: asyncio.Queue[float], done: asyncio.Event) -> typing.Any:  # noqa: ANN401  (a stub built without __init__)
    from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator

    @contextlib.contextmanager
    def sim_time_rate(rate: float) -> typing.Iterator[tuple[asyncio.Event, asyncio.Queue[float]]]:
        del rate
        yield done, ticks

    sim = object.__new__(ArenaHumanSimulator)
    sim._ticks_coalesced = {"render": 0, "relay": 0}
    logger = SimpleNamespace(error=print, warning=print, get_child=lambda name: logger)
    sim._NodeInterface__node = SimpleNamespace(sim_time_rate=sim_time_rate, get_logger=lambda: logger)
    return sim


@pytest.mark.parametrize(("loop", "work", "counter"), [("_interpolation_loop", "_publish_interpolated", "render"), ("_pedestrian_update_loop", "_relay_once", "relay")])
def test_a_backlog_of_ticks_is_handled_once_and_the_loop_yields(loop: str, work: str, counter: str) -> None:
    async def scenario() -> tuple[list[str], dict[str, int]]:
        ticks: asyncio.Queue[float] = asyncio.Queue()
        done = asyncio.Event()
        sim = _sim(ticks, done)
        trace: list[str] = []

        async def handled(*args: object) -> None:
            trace.append("work")
            if trace.count("work") == 2:
                done.set()

        setattr(sim, work, handled)
        for _ in range(6):  # a backlog: six ticks queued while the loop was busy
            ticks.put_nowait(0.02)

        async def other() -> None:
            for _ in range(3):
                trace.append("other")
                await asyncio.sleep(0)

        task = asyncio.create_task(getattr(sim, loop)())
        await asyncio.sleep(0)
        await other()
        ticks.put_nowait(0.02)
        await asyncio.wait_for(task, 1.0)
        return trace, sim._ticks_coalesced

    trace, counts = asyncio.run(scenario())
    assert trace.count("work") == 2  # one pass for the six queued ticks, one for the next tick
    assert trace.index("other") < trace.index("work", trace.index("work") + 1)  # another coroutine ran in between
    assert counts[counter] == 5


@pytest.mark.parametrize(("loop", "work", "counter"), [("_interpolation_loop", "_publish_interpolated", "render"), ("_pedestrian_update_loop", "_relay_once", "relay")])
def test_ticks_that_arrive_one_at_a_time_are_each_handled(loop: str, work: str, counter: str) -> None:
    async def scenario() -> tuple[int, dict[str, int]]:
        ticks: asyncio.Queue[float] = asyncio.Queue()
        done = asyncio.Event()
        sim = _sim(ticks, done)
        handled_n = [0]

        async def handled(*args: object) -> None:
            handled_n[0] += 1

        setattr(sim, work, handled)
        task = asyncio.create_task(getattr(sim, loop)())
        for _ in range(10):
            ticks.put_nowait(0.02)
            for _ in range(3):
                await asyncio.sleep(0)
        done.set()
        ticks.put_nowait(0.02)
        await asyncio.wait_for(task, 1.0)
        return handled_n[0], sim._ticks_coalesced

    handled_n, counts = asyncio.run(scenario())
    assert handled_n >= 10 and counts[counter] == 0
