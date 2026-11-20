"""Nav2Adapter.on_reset: every episode starts with the goal cancelled and both costmaps cleared, the global one once."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("nav2_msgs.srv")

from arena_rclpy_mixins.shared import Namespace
from arena_robots.task_kinds import TaskKind
from arena_runtime.env.adapters.mobile.nav2 import Nav2Adapter
from nav2_msgs.srv import ClearEntireCostmap

NS = "/env_0/jackal_0"
GLOBAL = f"{NS}/global_costmap/clear_entirely_global_costmap"
LOCAL = f"{NS}/local_costmap/clear_entirely_local_costmap"
SLAM_RESET = f"{NS}/slam_toolbox/reset"


class _Client:
    def __init__(self, events: list[str], done: bool) -> None:
        self._events = events
        self._done = done

    def is_done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        self._events.append("cancel")


class _Logger:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def warning(self, message: str) -> None:
        self._events.append(f"warning: {message}")


def _reset(mapping: str, *, running: bool = True, start_pose: object | None = None, timeout_on: str | None = None, active: bool = True) -> list[str]:
    """Run on_reset with the service calls recorded instead of made; the events in their order."""
    events: list[str] = []
    adapter = Nav2Adapter.__new__(Nav2Adapter)
    adapter._bringup_kwargs = {"mapping": mapping}
    adapter._service_clients = {}
    adapter._clients = {TaskKind.GOTO_POSE: _Client(events, done=not running)}

    async def call_active(robot: object, node_name: str, srv_type: type, srv_name: str, req: object) -> bool:
        if srv_name == timeout_on:
            raise TimeoutError(f"{srv_name} not available after 20.0s")
        events.append(srv_name)
        assert srv_type is not ClearEntireCostmap or isinstance(req, ClearEntireCostmap.Request)
        return active

    async def move(pose: object) -> None:
        events.append("move")

    adapter._call_active = call_active
    robot = SimpleNamespace(namespace=Namespace(NS), node=SimpleNamespace(get_logger=lambda: _Logger(events)), move=move)
    asyncio.run(adapter.on_reset(robot, SimpleNamespace(start_pose=start_pose)))
    return events


def test_a_static_map_robot_cancels_moves_and_clears_the_global_then_the_local_costmap() -> None:
    assert _reset("static", start_pose=object()) == ["cancel", "move", GLOBAL, LOCAL]


def test_a_robot_without_a_running_goal_still_clears_both_costmaps() -> None:
    assert _reset("static", running=False) == [GLOBAL, LOCAL]


def test_the_global_clear_uses_the_robot_namespace() -> None:
    assert [e for e in _reset("static") if "global" in e] == ["/env_0/jackal_0/global_costmap/clear_entirely_global_costmap"]


def test_a_slam_robot_clears_the_global_costmap_once_after_its_map() -> None:
    assert _reset("slam") == ["cancel", SLAM_RESET, GLOBAL, LOCAL]


def test_a_slam_robot_whose_mapper_is_not_active_yet_clears_only_the_local_costmap() -> None:
    assert _reset("slam", active=False) == ["cancel", SLAM_RESET, LOCAL]


def test_a_global_costmap_still_coming_up_does_not_fail_the_reset() -> None:
    events = _reset("static", timeout_on=GLOBAL)
    assert events[0] == "cancel" and events[-1] == LOCAL
    assert len(events) == 3 and events[1].startswith(f"warning: {NS}: global costmap not cleared")


def test_an_inactive_global_costmap_is_logged_and_does_not_stop_the_local_clear() -> None:
    events = _reset("static", active=False)
    assert events[:2] == ["cancel", GLOBAL] and events[-1] == LOCAL
    assert len(events) == 4 and events[2] == f"warning: {NS}: global costmap not cleared, it is not active or the call failed"


def test_a_slam_robot_whose_mapper_is_still_coming_up_still_clears_the_local_costmap() -> None:
    events = _reset("slam", timeout_on=SLAM_RESET)
    assert events[0] == "cancel" and events[-1] == LOCAL
    assert len(events) == 3 and events[1].startswith(f"warning: {NS}: slam map not reset")


def test_a_local_costmap_still_coming_up_fails_the_reset() -> None:
    with pytest.raises(TimeoutError):
        _reset("static", timeout_on=LOCAL)
