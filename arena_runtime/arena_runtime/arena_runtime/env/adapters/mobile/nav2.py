"""Nav2 adapter: thin composer of Nav2Bringup + GotoPoseClient."""

from __future__ import annotations

import asyncio
import collections
import os
from typing import TYPE_CHECKING, ClassVar, Literal

import lifecycle_msgs.msg
from arena_rclpy_mixins.params import declare
from arena_robots.bringup.mobile.nav2 import Nav2Bringup
from arena_robots.clients.goto_pose import GotoPoseClient
from arena_robots.task_kinds import TaskKind
from arena_robots_msgs.action import GotoPose
from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec
from nav2_msgs.srv import ClearCostmapAroundRobot, ClearEntireCostmap
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2

from arena_runtime.env.adapters import AdapterDisplayHint, AdapterMeta
from arena_runtime.env.adapters.mobile import MobileAdapter
from arena_runtime.env.request import GoToPhase, TaskPhase
from arena_runtime.env.world_manager.shims import requires_map_server
from arena_runtime.node_params import NodeParams

if TYPE_CHECKING:
    import geometry_msgs.msg
    from arena_rclpy_mixins.Async import ClientWrapper

    from arena_runtime.env.adapters import AdapterCtx, ResetContext
    from arena_runtime.env.robot_manager.robot_manager import RobotManager
    from arena_runtime.shared import Pose

_COSTMAP_TIMEOUT_S = 20.0
_SLAM_NODE = "slam_toolbox"
_REDISPATCH_MIN_S = 2.0
#: The sensors the costmaps mark from, by SensorSpec type (arena_robots nav2.py `_TYPE_TO_NAV2`)
_SCAN_TYPES: dict[str, type] = {"laserscan": LaserScan, "pointcloud": PointCloud2}
#: Scans stamped after the move a topic must deliver before the second global clear: by the second, the costmap holds the
#: first, and with the global observation persistence 0 it marks from nothing older
_FRESH_SCANS = 2
_FRESH_SCAN_TIMEOUT_S = 3.0  # wall
_FRESH_SCAN_POLL_S = 0.02


@AdapterMeta.attach(
    accepts={TaskKind.GOTO_POSE},
    bringup=Nav2Bringup,
    client=GotoPoseClient,
    cap="mobile",
    displays=[
        AdapterDisplayHint(
            name="Local Costmap",
            topic="{ns}/local_costmap/costmap",
            topic_type="nav_msgs/OccupancyGrid",
            kind=DisplayKind.MAP,
            style_json=StyleSpec(alpha=0.7, extra={"rviz": {"Color Scheme": "costmap"}}).to_json(),
        ),
        AdapterDisplayHint(
            name="Global Costmap",
            topic="{ns}/global_costmap/costmap",
            topic_type="nav_msgs/OccupancyGrid",
            kind=DisplayKind.MAP,
            style_json=StyleSpec(alpha=0.7, enabled=False, extra={"rviz": {"Color Scheme": "costmap"}}).to_json(),
        ),
        AdapterDisplayHint(
            name="Local Plan",
            topic="{ns}/local_plan",
            topic_type="nav_msgs/Path",
            kind=DisplayKind.PATH,
            style_json=StyleSpec(color=(255, 0, 0), line_width=0.05).to_json(),
        ),
        AdapterDisplayHint(
            name="Robot Footprint",
            topic="{ns}/local_costmap/published_footprint",
            topic_type="geometry_msgs/PolygonStamped",
            kind=DisplayKind.POLYGON,
            style_json=StyleSpec(alpha=1.0).to_json(),
        ),
    ],
)
@requires_map_server
class Nav2Adapter(MobileAdapter):
    kind: ClassVar[str] = "nav2"
    mappings: ClassVar[frozenset[str]] = frozenset({"static", "slam"})

    def __init__(self, robot_manager: RobotManager, **kwargs: object):
        super().__init__(robot_manager, **kwargs)
        self._service_clients: dict[str, ClientWrapper] = {}
        self._last_redispatch: float = -_REDISPATCH_MIN_S
        self._scan_topics: dict[str, type] = {}
        self._scan_subs: list[object] = []
        self._scan_stamps: dict[str, collections.deque[int]] = {}

    def launch_description(self, ctx: AdapterCtx):  # noqa: ANN201 - GroupAction, as the base
        namespace = str(ctx.namespace).rstrip("/")
        for spec in ctx.sensors:
            kind = spec.type.value if hasattr(spec.type, "value") else str(spec.type)
            if kind in _SCAN_TYPES:
                topic = spec.topic.replace("${namespace}", namespace)
                self._scan_topics[topic if topic.startswith("/") else f"{namespace}/{topic}"] = _SCAN_TYPES[kind]
        return super().launch_description(ctx)

    async def teardown(self) -> None:
        for cli in self._service_clients.values():
            self.rm.node.destroy_client(cli.client)
        self._service_clients.clear()
        for sub in self._scan_subs:
            self.rm.node.destroy_subscription(sub)
        self._scan_subs.clear()
        await super().teardown()

    async def publish_goal_loop(self) -> None:
        # nav2 uses navigate_to_pose action; no topic republish.
        return

    async def dispatch_phase(
        self,
        phase: TaskPhase,
        robot: RobotManager,
    ) -> None:
        assert isinstance(phase, GoToPhase), f"Nav2Adapter only accepts GOTO_POSE phases; got {type(phase).__name__} (kind={phase.kind!r})"
        assert phase.pose is not None
        robot._goal_pos = phase.pose  # pylint: disable=protected-access
        if self.client.is_done() is False:
            self.client.cancel()
        goal = GotoPose.Goal()
        goal.target = self._phase_to_pose_stamped(phase, robot)
        goal.pose_tolerance = float(phase.tolerance_radius or 0.0)
        goal.yaw_tolerance = float(phase.tolerance_angle or 0.0)
        await self.client.send_goal(goal)

    def _phase_to_pose_stamped(
        self,
        phase: GoToPhase,
        robot: RobotManager,
    ) -> geometry_msgs.msg.PoseStamped:
        import geometry_msgs.msg

        msg = geometry_msgs.msg.PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = robot.node.sim_time.to_msg()
        assert phase.pose is not None
        msg.pose = phase.pose.to_msg()
        return msg

    def on_phase_tick(self, phase: TaskPhase, robot: RobotManager) -> None:
        if robot.waiting or not self.client.is_done():
            return
        now = robot.node.sim_time.to_seconds()
        if now - self._last_redispatch >= _REDISPATCH_MIN_S:
            self._last_redispatch = now
            robot.node.event_loop.create_task(self.dispatch_phase(phase, robot))

    async def wait_until_ready(
        self,
        robot: RobotManager,
        node_paths: set[str],
    ) -> None:
        # TMP: remove once rosnavrl decoupled from nav
        if declare(robot.node, NodeParams.ROBOT_TRAIN).value:
            await super().wait_until_ready(robot, node_paths)
            return
        bt_node_path = str(robot.namespace("bt_navigator"))
        await robot.node.poll(lambda: bt_node_path in node_paths, f"node {bt_node_path}", interval=0.01)
        await super().wait_until_ready(robot, node_paths)

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        if self.client.is_done() is False:
            self.client.cancel()
        await super().on_reset(robot, ctx)
        # the global costmap keeps the obstacles the robot marked in the last episode, which in this episode's layout and
        # from its start it may never see again to clear; under slam, reset_map clears it once, after the map
        if self._bringup_kwargs.get("mapping") == "slam":
            await self.reset_map(robot)
        else:
            await self._clear_global_costmap(robot)
        await self._clear_costmap(robot, "local")
        if ctx.start_pose is not None:
            await self._clear_global_costmap_from_start(robot)

    async def _clear_global_costmap_from_start(self, robot: RobotManager) -> None:
        """Clear the global costmap again once the sensors have reported from the start pose.

        The first clear lands right after the teleport, while the costmap still holds the last scan from the pose before
        it and marks it again: one sensor view of marks at the old pose, which the robot never sees again to clear. Once
        every scan topic that published before has delivered `_FRESH_SCANS` scans stamped after the robot landed, the
        costmap holds only scans from the start pose, and a clear drops the rest for good. The sim is stepped while the
        reset holds it. After `_FRESH_SCAN_TIMEOUT_S` the clear goes ahead with a warning."""
        self._watch_scans(robot)
        stamped = robot.pose_stamped
        landed = None if stamped is None else stamped[1].nanoseconds
        if not await self._await_fresh_scans(robot, landed):
            robot.node.get_logger().warning(f"{robot.namespace}: no {_FRESH_SCANS} scans from the start pose within {_FRESH_SCAN_TIMEOUT_S:g}s, global costmap cleared again without them")
        await self._clear_global_costmap(robot)

    def _watch_scans(self, robot: RobotManager) -> None:
        """Subscribe once to the robot's scan topics, keeping each topic's latest stamps."""
        if self._scan_subs or not self._scan_topics:
            return
        for topic, msg_type in self._scan_topics.items():
            stamps: collections.deque[int] = collections.deque(maxlen=8)
            self._scan_stamps[topic] = stamps

            def record(msg: LaserScan | PointCloud2, stamps: collections.deque[int] = stamps) -> None:
                stamps.append(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)

            self._scan_subs.append(robot.node.create_subscription(msg_type, topic, record, qos_profile_sensor_data))

    async def _await_fresh_scans(self, robot: RobotManager, landed: int | None) -> bool:
        """Whether every scan topic that has published delivers `_FRESH_SCANS` scans stamped after `landed` (after its latest
        stamp so far, without a TF stamp of the landing) within the timeout, stepping a held sim."""
        since = {topic: landed if landed is not None else stamps[-1] for topic, stamps in self._scan_stamps.items() if stamps}

        def fresh() -> bool:
            # a snapshot: the subscriptions may append from the executor's thread
            return all(sum(stamp > after for stamp in tuple(self._scan_stamps[topic])) >= _FRESH_SCANS for topic, after in since.items())

        loop = asyncio.get_running_loop()
        deadline = loop.time() + _FRESH_SCAN_TIMEOUT_S
        stepping = True
        while not fresh():
            if loop.time() >= deadline:
                return False
            if stepping:
                try:
                    advanced = await asyncio.wait_for(robot.node.step_tick(), timeout=max(0.0, deadline - loop.time()))
                except TimeoutError:
                    return fresh()
                stepping = advanced is not None  # None: the sim runs on its own or refuses a step, so wait for it
            await asyncio.sleep(_FRESH_SCAN_POLL_S)
        return True

    async def _clear_global_costmap(self, robot: RobotManager) -> bool:
        """Clear the global costmap; False while it is not active or still coming up, which does not fail the reset."""
        try:
            cleared = await self._clear_costmap(robot, "global")
        except TimeoutError as exc:
            robot.node.get_logger().warning(f"{robot.namespace}: global costmap not cleared, the navigation stack is still coming up ({exc})")
            return False
        if not cleared:
            robot.node.get_logger().warning(f"{robot.namespace}: global costmap not cleared, it is not active or the call failed")
        return cleared

    async def reset_map(self, robot: RobotManager) -> bool:
        """Under robot.mobile.mapping:=slam, empty slam_toolbox's map and the global costmap built on it."""
        if self._bringup_kwargs.get("mapping") != "slam":
            return False
        from slam_toolbox.srv import Reset  # noqa: PLC0415  # only slam mapping needs slam_toolbox

        node_name = robot.namespace(_SLAM_NODE)
        try:
            if not await self._call_active(robot, node_name, Reset, os.path.abspath(node_name("reset")), Reset.Request()):
                return False
            return await self._clear_costmap(robot, "global")
        except TimeoutError as exc:
            robot.node.get_logger().warning(f"{robot.namespace}: slam map not reset, the navigation stack is still coming up ({exc})")
            return False

    async def before_move(
        self,
        pose: Pose,
        robot: RobotManager,
    ) -> None:
        if self.client.is_done() is False:
            self.client.cancel()

    async def on_move(
        self,
        pose: Pose,
        robot: RobotManager,
    ) -> None:
        await self._clear_costmap(robot, "local")

        phase = robot.active_dispatch_phase()
        if isinstance(phase, GoToPhase):
            await self.dispatch_phase(phase, robot)

    async def on_cabin_open(self, robot: RobotManager) -> None:
        await self._clear_costmap(robot, "local")

    async def on_navigation_reset(self, robot: RobotManager) -> None:
        # a teleport clears only the local costmap (on_move); the global one keeps the obstacles the robot marked on its
        # way, which from the new pose it may never see again to clear, and a path they close stays closed for every goal
        await self._clear_costmap(robot, "global")
        await self._clear_costmap(robot, "local")

    async def _clear_costmap(
        self,
        robot: RobotManager,
        which: Literal["local", "global"] = "local",
        reset_distance: float = -1.0,
    ) -> bool:
        node_name = robot.namespace(f"{which}_costmap/{which}_costmap")

        if reset_distance < 0:
            srv_name = os.path.abspath(node_name(f"../clear_entirely_{which}_costmap"))
            srv_type = ClearEntireCostmap
            req = ClearEntireCostmap.Request()
        else:
            srv_name = os.path.abspath(node_name(f"../clear_around_{which}_costmap"))
            srv_type = ClearCostmapAroundRobot
            req = ClearCostmapAroundRobot.Request()
            req.reset_distance = reset_distance

        return await self._call_active(robot, node_name, srv_type, srv_name, req)

    async def _call_active(self, robot: RobotManager, node_name: str, srv_type: type, srv_name: str, req: object) -> bool:
        """Call a service of a lifecycle node, False while the node is not active or when the call fails."""
        state = await robot.node.get_lifecycle_state_async(node_name, timeout=_COSTMAP_TIMEOUT_S)
        if state.id != lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE:
            return False

        cli = self._service_clients.get(srv_name)
        if cli is None:
            cli = robot.node.create_client_wrapper(srv_type, srv_name)
            self._service_clients[srv_name] = cli
        if not await cli.ensure(timeout_sec=_COSTMAP_TIMEOUT_S):
            raise TimeoutError(f"{srv_name} not available after {_COSTMAP_TIMEOUT_S}s")

        result = await cli.call_timeout(req, timeout_sec=_COSTMAP_TIMEOUT_S)
        if result is None:
            robot.node.get_logger().error(f"service call failed for {srv_name}")
            return False
        return True
