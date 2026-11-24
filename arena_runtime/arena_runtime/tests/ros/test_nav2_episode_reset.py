"""Nav2Adapter.on_reset: every episode starts with the goal cancelled and both costmaps cleared; a moved robot's global
costmap is cleared twice, the second time once the sensors have reported from the start pose."""

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


SCAN = f"{NS}/lidar"
CLOUD = f"{NS}/lidar/points"
TICK_NS = 20_000_000  # one physics step
SCAN_NS = 100_000_000  # the lidar's 10 Hz


class _Rig:
    """A Nav2Adapter on a held sim: the service calls, sim steps, subscriptions and the move are recorded instead of
    made, and each step publishes on the subscribed scan topics at the lidar's rate, stamped with the sim time."""

    def __init__(
        self,
        mapping: str = "static",
        *,
        running: bool = True,
        timeout_on: str | None = None,
        active: bool = True,
        held: bool = True,
        topics: tuple[str, ...] = (),
        first: bool = False,
        refused: int = 0,
    ) -> None:
        self.events: list[str] = []
        self.timeout_on = timeout_on
        self.active = active
        self.held = held
        self.refused = refused  # steps refused before the sim is held again (another robot's unpause window)
        self.now = 5_000_000_000
        self.landed: int | None = None
        self.publishing = set(topics)  # topics with a publisher that publish on the steps
        self.callbacks: dict[str, object] = {}
        self.late: list[int] = []  # stamps delivered as soon as a subscription is made: scans in flight from before
        self.at_clear: list[dict[str, list[int]]] = []
        self.waited: dict[str, list[int]] = {}  # each topic's stamps when the wait ended
        self.adapter = Nav2Adapter.__new__(Nav2Adapter)
        self.adapter._bringup_kwargs = {"mapping": mapping}
        self.adapter._service_clients = {}
        self.adapter._clients = {TaskKind.GOTO_POSE: _Client(self.events, done=not running)}
        self.adapter._scan_topics = dict.fromkeys(topics, object)
        self.adapter._scan_subs = []
        self.adapter._scan_stamps = {}
        self.adapter._resets = 0 if first else 1
        self.adapter._call_active = self._call_active
        node = SimpleNamespace(
            get_logger=lambda: _Logger(self.events),
            step_tick=self._step_tick,
            create_subscription=self._subscribe,
            destroy_subscription=self._unsubscribe,
            count_publishers=lambda topic: int(topic in self.publishing),
        )
        self.robot = SimpleNamespace(namespace=Namespace(NS), node=node, move=self._move, pose_stamped=None)
        self.adapter.rm = self.robot

    def _subscribe(self, msg_type: type, topic: str, callback: object, qos: object) -> object:
        self.events.append(f"subscribe {topic}")
        self.callbacks[topic] = callback
        for stamp in self.late:
            self.publish(topic, stamp)
        return topic

    def _unsubscribe(self, sub: object) -> None:
        self.events.append(f"unsubscribe {sub}")
        self.waited.setdefault(str(sub), list(self.adapter._scan_stamps.get(str(sub), ())))
        self.callbacks.pop(str(sub))

    def publish(self, topic: str, stamp: int) -> None:
        self.callbacks[topic](SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000))))

    def tick(self) -> None:
        self.now += TICK_NS
        if self.now % SCAN_NS == 0:
            for topic in self.publishing & set(self.callbacks):
                self.publish(topic, self.now)

    async def _step_tick(self) -> float | None:
        self.events.append("step")
        if not self.held:
            return None
        if self.refused:
            self.refused -= 1
            return None
        self.tick()
        return TICK_NS / 1e9

    async def _call_active(self, robot: object, node_name: str, srv_type: type, srv_name: str, req: object) -> bool:
        if srv_name == self.timeout_on:
            raise TimeoutError(f"{srv_name} not available after 20.0s")
        self.events.append(srv_name)
        assert srv_type is not ClearEntireCostmap or isinstance(req, ClearEntireCostmap.Request)
        if srv_name == GLOBAL:
            self.at_clear.append({topic: list(stamps) for topic, stamps in self.adapter._scan_stamps.items()})
        return self.active

    async def _move(self, pose: object) -> None:
        self.events.append("move")
        self.tick()
        self.landed = self.now
        self.robot.pose_stamped = (pose, SimpleNamespace(nanoseconds=self.landed))

    def reset(self, start_pose: object | None = None) -> list[str]:
        asyncio.run(self.adapter.on_reset(self.robot, SimpleNamespace(start_pose=start_pose)))
        return self.events

    def fresh(self, topic: str) -> int:
        """Scans stamped after the landing that `topic` had delivered when the wait ended."""
        return sum(stamp > self.landed for stamp in self.waited[topic])


def _reset(mapping: str, *, start_pose: object | None = None, **kwargs: object) -> list[str]:
    """Run on_reset with the service calls recorded instead of made; the events in their order."""
    return _Rig(mapping, **kwargs).reset(start_pose)


def _steps_squeezed(events: list[str]) -> list[str]:
    return [e for i, e in enumerate(events) if e != "step" or (i and events[i - 1] != "step")]


WAIT = ["subscribe " + SCAN, "subscribe " + CLOUD, "step", "unsubscribe " + SCAN, "unsubscribe " + CLOUD]


def test_a_static_map_robot_cancels_moves_and_clears_the_global_then_the_local_costmap() -> None:
    # no scan topic: the second clear follows at once
    assert _reset("static", start_pose=object()) == ["cancel", "move", GLOBAL, LOCAL, GLOBAL]


def test_the_second_global_clear_waits_for_two_scans_from_the_start_pose_on_every_scan_topic() -> None:
    rig = _Rig(topics=(SCAN, CLOUD))
    events = rig.reset(start_pose=object())
    assert _steps_squeezed(events) == ["cancel", "move", GLOBAL, LOCAL, *WAIT, GLOBAL]
    assert rig.at_clear == [{}, {}] and rig.callbacks == {}  # no subscription at either clear
    assert rig.fresh(SCAN) >= 2 and rig.fresh(CLOUD) >= 2
    assert events.count("step") <= 2 * SCAN_NS // TICK_NS


def test_the_subscriptions_live_only_for_the_wait_and_hold_no_stamps_after_it() -> None:
    rig = _Rig(topics=(SCAN, CLOUD))
    rig.reset(start_pose=object())
    assert rig.adapter._scan_subs == [] and rig.adapter._scan_stamps == {} and rig.callbacks == {}


def test_teardown_destroys_subscriptions_still_open(monkeypatch: pytest.MonkeyPatch) -> None:
    async def base_teardown(self: object) -> None:
        return None

    base = next(c for c in Nav2Adapter.__mro__[1:] if "teardown" in c.__dict__)
    monkeypatch.setattr(base, "teardown", base_teardown)
    rig = _Rig(topics=(SCAN,))
    rig.adapter._watch_scans(rig.robot)
    asyncio.run(rig.adapter.teardown())
    assert rig.events == ["subscribe " + SCAN, "unsubscribe " + SCAN] and rig.adapter._scan_subs == []


def test_the_first_reset_of_an_env_clears_twice_at_once_without_a_step() -> None:
    rig = _Rig(topics=(SCAN, CLOUD), first=True)
    assert rig.reset(start_pose=object()) == ["cancel", "move", GLOBAL, LOCAL, GLOBAL]
    assert rig.adapter._resets == 1
    assert _steps_squeezed(rig.reset(start_pose=object()))[-6:] == [*WAIT, GLOBAL]


def test_a_scan_stamped_before_the_landing_is_no_scan_from_the_start_pose() -> None:
    rig = _Rig(topics=(SCAN,))
    rig.late = [rig.now + TICK_NS // 2] * 5  # stamped before the landing, delivered after it
    events = rig.reset(start_pose=object())
    assert rig.fresh(SCAN) >= 2
    assert events.count("step") >= SCAN_NS // TICK_NS


def test_a_topic_without_a_publisher_is_not_waited_for() -> None:
    rig = _Rig(topics=(SCAN, CLOUD))
    rig.publishing.discard(CLOUD)
    events = rig.reset(start_pose=object())
    assert _steps_squeezed(events) == ["cancel", "move", GLOBAL, LOCAL, "subscribe " + SCAN, "step", "unsubscribe " + SCAN, GLOBAL]
    assert rig.fresh(SCAN) >= 2


def test_a_refused_step_is_tried_again_and_stepping_resumes() -> None:
    rig = _Rig(topics=(SCAN,), refused=3)
    events = rig.reset(start_pose=object())
    assert rig.fresh(SCAN) >= 2 and not any(e.startswith("warning") for e in events)
    assert events.count("step") >= 3 + SCAN_NS // TICK_NS


def test_a_sim_that_runs_on_its_own_is_waited_for() -> None:
    rig = _Rig(topics=(SCAN,), held=False)

    async def main() -> None:
        async def sim() -> None:
            while True:
                await asyncio.sleep(0.005)
                rig.tick()

        clock = asyncio.create_task(sim())
        try:
            await rig.adapter.on_reset(rig.robot, SimpleNamespace(start_pose=object()))
        finally:
            clock.cancel()

    asyncio.run(main())
    assert _steps_squeezed(rig.events) == ["cancel", "move", GLOBAL, LOCAL, "subscribe " + SCAN, "step", "unsubscribe " + SCAN, GLOBAL]
    assert rig.fresh(SCAN) >= 2


def test_scans_that_do_not_come_end_the_wait_with_a_warning_and_the_clear_goes_ahead(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_runtime.env.adapters.mobile import nav2

    monkeypatch.setattr(nav2, "_FRESH_SCAN_TIMEOUT_S", 0.2)
    rig = _Rig(topics=(SCAN,), held=False)  # held=False: steps are refused and nothing publishes
    events = _steps_squeezed(rig.reset(start_pose=object()))
    assert events[:7] == ["cancel", "move", GLOBAL, LOCAL, "subscribe " + SCAN, "step", "unsubscribe " + SCAN]
    assert events[7].startswith(f"warning: {NS}: no 2 scans from the start pose within 0.2s") and events[-1] == GLOBAL


def test_a_robot_that_is_not_moved_gets_no_second_clear() -> None:
    rig = _Rig(topics=(SCAN,))
    assert rig.reset() == ["cancel", GLOBAL, LOCAL]


def test_the_scan_topics_are_the_robots_laserscan_and_pointcloud_sensors_in_its_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_robots.Sensor import SensorSpec, SensorType

    base = next(c for c in Nav2Adapter.__mro__[1:] if "launch_description" in c.__dict__)
    monkeypatch.setattr(base, "launch_description", lambda self, ctx: "group")
    adapter = Nav2Adapter.__new__(Nav2Adapter)
    adapter._scan_topics = {}
    sensors = [
        SensorSpec(name="lidar", type=SensorType.LASERSCAN, topic="${namespace}/lidar", frame=""),
        SensorSpec(name="lidar_points", type=SensorType.POINTCLOUD, topic="lidar/points", frame=""),
        SensorSpec(name="camera", type=SensorType.IMAGE, topic="${namespace}/camera", frame=""),
    ]
    assert adapter.launch_description(SimpleNamespace(namespace=Namespace(NS), sensors=sensors)) == "group"
    from sensor_msgs.msg import LaserScan, PointCloud2

    assert adapter._scan_topics == {SCAN: LaserScan, CLOUD: PointCloud2}


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
