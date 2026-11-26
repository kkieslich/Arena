"""The robot trail (a viz Path nothing records): sampled on a sim-time timer, sent only to a subscriber, at most every _TRAIL_SEND_S, at once to a new one."""

from __future__ import annotations

import math
import typing
from types import SimpleNamespace

import pytest


class _Publisher:
    def __init__(self) -> None:
        self.subscribers = 0
        self.sent: list[int] = []

    def get_subscription_count(self) -> int:
        return self.subscribers

    def publish(self, msg: typing.Any) -> None:
        self.sent.append(len(msg.poses))


class _Pose:
    def __init__(self, x: float) -> None:
        self.position = SimpleNamespace(x=x, y=0.0)

    def to_msg(self) -> typing.Any:
        import geometry_msgs.msg

        msg = geometry_msgs.msg.Pose()
        msg.position.x = self.position.x
        return msg


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> typing.Any:
    import nav_msgs.msg

    from arena_runtime.env.adapters import mobile

    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(mobile, "time", SimpleNamespace(monotonic=lambda: clock.now))
    trail_only = type("TrailOnly", (mobile.MobileAdapter,), {})
    trail_only.__abstractmethods__ = frozenset()
    adapter = object.__new__(trail_only)
    adapter.rm = SimpleNamespace(pose=_Pose(0.0), node=SimpleNamespace(sim_time=SimpleNamespace(to_msg=lambda: nav_msgs.msg.Path().header.stamp)))
    adapter._trail = nav_msgs.msg.Path()
    adapter._trail_pub = _Publisher()
    adapter._trail_grown = False
    adapter._trail_sent_at = -math.inf
    adapter._trail_sent_to = 0
    adapter.clock = clock
    return adapter


def _drive(adapter: typing.Any, metres: float, *, speed: float = 2.0) -> None:
    """Drive at `speed`, one timer sample every _TRAIL_SAMPLE_S."""
    from arena_runtime.env.adapters.mobile import _TRAIL_SAMPLE_S

    x = adapter.rm.pose.position.x
    for _ in range(round(metres / speed / _TRAIL_SAMPLE_S)):
        x += speed * _TRAIL_SAMPLE_S
        adapter.clock.now += _TRAIL_SAMPLE_S
        adapter.rm.pose = _Pose(x)
        adapter._extend_trail()


def test_the_trail_grows_but_is_not_sent_without_a_subscriber(adapter: typing.Any) -> None:
    _drive(adapter, 10.0)
    assert adapter._trail_pub.sent == []
    assert len(adapter._trail.poses) == 50


def test_a_new_subscriber_gets_the_whole_trail_at_once(adapter: typing.Any) -> None:
    _drive(adapter, 10.0)
    adapter._trail_pub.subscribers = 1
    adapter._extend_trail()
    assert adapter._trail_pub.sent == [50]


def test_a_driving_robot_sends_its_trail_at_most_twice_a_second(adapter: typing.Any) -> None:
    adapter._trail_pub.subscribers = 1
    _drive(adapter, 20.0)
    sent = adapter._trail_pub.sent
    assert 16 <= len(sent) <= 21
    assert len(adapter._trail.poses) - sent[-1] <= 6  # at most 0.6 s of driving not yet sent


def test_a_parked_robot_sends_nothing_new(adapter: typing.Any) -> None:
    adapter._trail_pub.subscribers = 1
    _drive(adapter, 1.0)
    before = len(adapter._trail_pub.sent)
    for _ in range(30):
        adapter.clock.now += 0.1
        adapter._extend_trail()
    assert len(adapter._trail_pub.sent) - before <= 1


def test_points_closer_than_the_step_are_not_added(adapter: typing.Any) -> None:
    _drive(adapter, 1.0, speed=0.3)  # 3 cm per sample
    assert all(b.pose.position.x - a.pose.position.x >= 0.05 - 1e-9 for a, b in zip(adapter._trail.poses, adapter._trail.poses[1:], strict=False))


def test_the_trail_keeps_its_last_2000_points(adapter: typing.Any) -> None:
    _drive(adapter, 450.0)
    assert len(adapter._trail.poses) == 2000


def test_a_lost_pose_adds_nothing(adapter: typing.Any) -> None:
    adapter.rm.pose = None
    adapter._extend_trail()
    assert adapter._trail.poses == []
