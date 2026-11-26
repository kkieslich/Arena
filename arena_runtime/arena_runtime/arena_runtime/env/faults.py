"""Fault injection for checking the env on a real lane: off unless its test-only environment variable is set.

The switch is read from the process environment only, never from a ROS parameter, launch argument or suite config, so a
study run cannot turn it on (`tests/unit/test_faults.py` checks that nothing else reads the name)."""

from __future__ import annotations

import asyncio
import os

#: `<robot>@<sim s>=<delay s>` (`*` for any robot): hold that robot's first goal dispatched at or after `<sim s>` (sim time) for `<delay s>`
#: (wall) before it is sent, once per process, as a goal reply that comes late
GOAL_REPLY_DELAY_ENV = "ARENA_TEST_DELAY_GOAL_REPLY"

_spent: set[str] = set()


def goal_reply_delay_armed() -> bool:
    """Whether the switch is set at all; the caller reads nothing else (not even the sim time) without it."""
    return bool(os.environ.get(GOAL_REPLY_DELAY_ENV))


def _goal_reply_delay(robot: str, sim_s: float, spec: str | None) -> float:
    if not spec or GOAL_REPLY_DELAY_ENV in _spent:
        return 0.0
    target, _, rest = spec.partition("@")
    at, _, delay = rest.partition("=")
    if target not in (robot, "*") or sim_s < float(at):
        return 0.0
    _spent.add(GOAL_REPLY_DELAY_ENV)
    return float(delay)


async def hold_goal(robot: str, sim_s: float) -> float:
    """Wait as `ARENA_TEST_DELAY_GOAL_REPLY` asks for this robot's goal (once), returning the seconds held."""
    delay = _goal_reply_delay(robot, sim_s, os.environ.get(GOAL_REPLY_DELAY_ENV))
    if delay > 0.0:
        await asyncio.sleep(delay)
    return delay
