"""A robot's sim tick as a task of its own, owned by the robot so its reset and removal cancel it."""

from __future__ import annotations

import asyncio
import time
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Coroutine

    from arena_runtime.env.robot_manager.task_runner import Transition


class TickInFlight:
    """The robot's tick the env's sim tick loop started and has not collected yet.

    A tick can await a reply (the goal of the phase it dispatches, a cancel, a lifecycle query); the env does not wait for
    it beyond its short budget and judges the other robots meanwhile (`EnvNode._tick_robots`). The robot keeps the task:
    destroying the robot, or setting the robots up for a new episode, cancels it.
    """

    _tick_task: asyncio.Task[Transition] | None = None
    _tick_started: float = 0.0
    _tick_finished: float | None = None
    tick_reported: bool = False

    def start_tick(self, tick: Coroutine[typing.Any, typing.Any, Transition]) -> asyncio.Task[Transition]:
        task = asyncio.create_task(tick)
        self._tick_task, self._tick_started, self._tick_finished, self.tick_reported = task, time.monotonic(), None, False
        task.add_done_callback(lambda _: setattr(self, "_tick_finished", time.monotonic()))
        return task

    @property
    def tick_task(self) -> asyncio.Task[Transition] | None:
        return self._tick_task

    def tick_age(self, now: float) -> float:
        """Wall seconds the tick in flight has run by `now`, or took once it finished."""
        return (self._tick_finished if self._tick_finished is not None else now) - self._tick_started

    def take_tick(self) -> asyncio.Task[Transition] | None:
        """The finished tick, which the robot then forgets; None while it runs or without one."""
        task = self._tick_task
        if task is None or not task.done():
            return None
        self._tick_task = None
        return task

    async def cancel_tick(self) -> None:
        task, self._tick_task = self._tick_task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
