"""The episode budget is checked where /clock arrives, not on the env's event loop: a loop held for seconds by one long
step (a backlog of crowd ticks) still ends the episode on time."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from arena_runtime.env.deadline import EpisodeDeadline


def test_fires_once_at_the_budget() -> None:
    fired: list[tuple[int, float]] = []
    d = EpisodeDeadline(lambda e, t: fired.append((e, t)))
    d.arm(3, 100.0, 20.0)
    for t in (100.0, 110.0, 119.99):
        d.on_clock(t)
    assert fired == []
    d.on_clock(120.0)
    d.on_clock(120.05)
    assert fired == [(3, 120.0)] and d.armed is None


def test_disarmed_or_other_episode_does_not_fire() -> None:
    fired: list[tuple[int, float]] = []
    d = EpisodeDeadline(lambda e, t: fired.append((e, t)))
    d.arm(1, 0.0, 5.0)
    d.disarm(2)            # another episode: stays armed
    assert d.armed is not None
    d.disarm(1)
    d.on_clock(10.0)
    assert fired == []
    d.arm(2, 50.0, 5.0)    # the next episode, the clock went back before its start: no fire
    d.on_clock(40.0)
    d.on_clock(55.0)
    assert fired == [(2, 55.0)]


def test_a_starved_event_loop_does_not_delay_the_deadline() -> None:
    """The loop runs one step that blocks for 1.5 s (as a render working through a tick backlog did). A clock thread
    stands in for the executor's /clock listener at sim = wall. The 0.3 s budget fires within 0.1 s of its sim time; the
    loop-side abort runs only once the long step is over, after the deadline has already been published."""
    events: list[tuple[str, float]] = []
    t0 = time.monotonic()
    loop = asyncio.new_event_loop()

    def on_expire(episode_id: int, now_s: float) -> None:
        events.append(("published", time.monotonic() - t0))
        loop.call_soon_threadsafe(lambda: events.append(("loop abort", time.monotonic() - t0)))

    d = EpisodeDeadline(on_expire)

    async def starve() -> None:
        time.sleep(1.5)  # one step that never yields
        await asyncio.sleep(0.05)

    def clock() -> None:
        while time.monotonic() - t0 < 2.0:
            d.on_clock(time.monotonic() - t0)
            time.sleep(0.01)

    d.arm(1, 0.0, 0.3)
    th = threading.Thread(target=clock)
    th.start()
    loop.run_until_complete(starve())
    th.join()
    loop.close()
    published = dict(events)["published"]
    assert 0.3 <= published < 0.4, events
    assert dict(events)["loop abort"] >= 1.5 > published, events


def test_the_episode_node_publishes_its_timeout_from_the_clock() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_runtime_msgs")
    import arena_runtime_msgs.action

    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.env.task_layer import EpisodeRecord

    RUNNING = arena_runtime_msgs.action.RunEpisode.Result.RUNNING
    FAILED = arena_runtime_msgs.action.RunEpisode.Result.FAILED
    published, scheduled, warned = [], [], []

    class Stub(EpisodeNode):  # the node's event loop, config and logger are properties: a subclass stands in for them
        event_loop = SimpleNamespace(call_soon_threadsafe=lambda f, *a: scheduled.append((f, a)))
        conf = SimpleNamespace(Robot=SimpleNamespace(TIMEOUT=SimpleNamespace(value=1200.0)))

        def get_logger(self):  # noqa: ANN202
            return SimpleNamespace(warning=warned.append)

    node = object.__new__(Stub)
    node._episodes = SimpleNamespace(current=EpisodeRecord(episode_id=4, outcome_state=RUNNING))
    node._record_to_msg = lambda record, params: SimpleNamespace(episode_id=record.episode_id, outcome_state=record.outcome_state, outcome_info=record.outcome_info)
    node._mode_params = lambda modes: {}
    node._human_params = {}
    node._pub_state_episode = SimpleNamespace(publish=published.append)
    node._on_deadline(4, 1200.05)
    assert [(m.episode_id, m.outcome_state, m.outcome_info) for m in published] == [(4, FAILED, "timeout")]
    assert node._episodes.current.outcome_state == RUNNING  # the loop's own record is the loop's to change
    assert scheduled and scheduled[0][1] == ("timeout",)
    node._on_deadline(3, 1300.0)  # an older episode's deadline: nothing
    assert len(published) == 1
