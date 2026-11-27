"""The episode's sim-time deadline, kept off the env's event loop.

The termination watcher checks the episode budget on the env's asyncio event loop, every 0.5 s. A loop step that runs
long (a crowd render or relay working through a backlog of queued rate ticks) holds that check, so on a loaded lane an
episode with a 1200 s budget ran 2716-5899 s of sim time before its timeout came (lockstep off: the sim does not wait).
`EpisodeDeadline` checks the budget where /clock arrives instead, in the node's clock listeners (`TimeNode.on_clock`,
the executor side): the first /clock past start + budget calls `on_expire` once, whatever the event loop is doing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Protocol


class _Seconds(Protocol):
    def to_seconds(self) -> float: ...


class EpisodeDeadline:
    """One armed deadline at a time: `arm` at the episode start, `disarm` when the episode ends; `on_clock` (a /clock
    listener) calls `on_expire(episode_id, now_s)` once, on the listener's thread, at the first time past the deadline.
    A clock that goes back before the start (a reset moved it) does not fire."""

    def __init__(self, on_expire: Callable[[int, float], None]) -> None:
        self._on_expire = on_expire
        self._lock = threading.Lock()
        self._armed: tuple[int, float, float] | None = None  # (episode id, start s, deadline s)
        self.fired: tuple[int, float] | None = None

    def arm(self, episode_id: int, start: _Seconds | float, budget_s: float) -> None:
        start_s = start if isinstance(start, (int, float)) else start.to_seconds()
        with self._lock:
            self._armed = (episode_id, float(start_s), float(start_s) + float(budget_s))
            self.fired = None

    def disarm(self, episode_id: int | None = None) -> None:
        with self._lock:
            if self._armed is not None and (episode_id is None or self._armed[0] == episode_id):
                self._armed = None

    @property
    def armed(self) -> tuple[int, float, float] | None:
        return self._armed

    def on_clock(self, now: _Seconds | float) -> None:
        now_s = now if isinstance(now, (int, float)) else now.to_seconds()
        with self._lock:
            armed = self._armed
            if armed is None or now_s < armed[2] or now_s < armed[1]:
                return
            self._armed = None
            self.fired = (armed[0], now_s)
        self._on_expire(armed[0], now_s)
