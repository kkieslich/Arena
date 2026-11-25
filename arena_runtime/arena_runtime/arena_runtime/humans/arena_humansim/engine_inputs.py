"""Engine inputs on the engine's tick boundaries.

With `input_barrier` the engine runs tick k + 1 only once the adapter has acknowledged tick k's frame with a world_state
stamped with that frame's stamp. Inputs the adapter sends between the frame and the acknowledgement land between ticks
k and k + 1 in every run. An input some other task decides while the engine ticks is held here and sent in the step
after the frame it was decided on, so it too lands on the same tick however the messages race the clock.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import typing

if typing.TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

_IN_STEP: contextvars.ContextVar[bool] = contextvars.ContextVar("arena_humansim_engine_step", default=False)


@dataclasses.dataclass(eq=False)
class _Held:
    stamp_ns: int
    send: Callable[[], Awaitable[typing.Any]]
    future: asyncio.Future[typing.Any]


class EngineInputs:
    """Holds the engine-changing calls made outside a step while `active` and sends them in the step that covers them.

    `stopped()` says the clock stands still with the engine caught up (a reset holding the lockstep clock): no frame comes
    to step on, and a call sent at once lands before the next tick, as it does in every run that stops the clock there.
    """

    POLL_S = 0.1

    def __init__(self, now_ns: Callable[[], int], stopped: Callable[[], bool] = lambda: False) -> None:
        self._now_ns = now_ns
        self._stopped = stopped
        self._held: list[_Held] = []
        self.active = False

    @property
    def held(self) -> int:
        return len(self._held)

    @contextlib.contextmanager
    def step(self) -> Iterator[None]:
        """Calls made inside go out at once: the step itself sends them before it acknowledges the frame."""
        token = _IN_STEP.set(True)
        try:
            yield
        finally:
            _IN_STEP.reset(token)

    async def call(self, send: Callable[[], Awaitable[typing.Any]]) -> typing.Any:  # noqa: ANN401
        """`send()` now when nothing is held back (or inside a step), else in the step after the frame it was decided on."""
        if not self.active or _IN_STEP.get():
            return await send()
        future: asyncio.Future[typing.Any] = asyncio.get_running_loop().create_future()
        held = _Held(self._now_ns(), send, future)
        self._held.append(held)
        while True:
            done, _ = await asyncio.wait({future}, timeout=self.POLL_S)
            if done:
                return future.result()
            if held in self._held and (not self.active or self._stopped()):
                self._held.remove(held)
                return await send()

    async def flush(self, before_ns: int | None = None) -> None:
        """Send the held calls decided before `before_ns` (all of them when None) in the order they were made, each once
        the previous one is answered, so the engine takes them in that order."""
        with self.step():
            while True:
                index = next((i for i, held in enumerate(self._held) if before_ns is None or held.stamp_ns < before_ns), None)
                if index is None:
                    return
                held = self._held.pop(index)
                if held.future.done():
                    continue
                # a step cancelled meanwhile (pause) leaves the call running, and its caller still gets the answer
                sent = asyncio.ensure_future(held.send())
                sent.add_done_callback(lambda task, future=held.future: _settle(future, task))
                with contextlib.suppress(Exception):
                    await asyncio.shield(sent)


def _settle(future: asyncio.Future[typing.Any], task: asyncio.Future[typing.Any]) -> None:
    if future.done():
        return
    if task.cancelled():
        future.cancel()
    elif (error := task.exception()) is not None:
        future.set_exception(error)
    else:
        future.set_result(task.result())


class SequencedClient:
    """A service client whose calls go through `EngineInputs`; everything else is the wrapped client's."""

    def __init__(self, inner: typing.Any, inputs: EngineInputs) -> None:  # noqa: ANN401
        self._inner = inner
        self._inputs = inputs

    async def call_timeout(self, request: object, *args: object, **kwargs: object) -> typing.Any:  # noqa: ANN401
        return await self._inputs.call(lambda: self._inner.call_timeout(request, *args, **kwargs))

    def __getattr__(self, name: str) -> typing.Any:  # noqa: ANN401
        return getattr(self._inner, name)
