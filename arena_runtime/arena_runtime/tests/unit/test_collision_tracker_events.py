from __future__ import annotations

import asyncio
import math
import threading
import time
import typing
import uuid

import numpy as np
import pytest

CLEAR = (2.0, 2.0)
IN_MAP = (8.5, 8.5)
ON_WALL = (2.5, 5.5)
CRATE = (5.5, 2.5)
BY_PED = (2.0, 8.2)
HALF_WIDTH = 0.2078
PED_RADIUS = 0.3
WHEELCHAIR_RADIUS = 0.325


async def _until(pred: typing.Callable[[], bool], what: str, timeout: float = 20.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out after {timeout}s waiting for {what}")
        await asyncio.sleep(0.01)


class Stage:
    def __init__(self, node, env, human, probe) -> None:
        import std_msgs.msg

        self.node = node
        self.env = env
        self.human = human
        self.probe = probe
        self.reset_end = probe.create_publisher(std_msgs.msg.Empty, "/reset_end", 1)
        self.reset_start = probe.create_publisher(std_msgs.msg.Empty, "/reset_start", 1)
        self.trackers: list = []
        self.armed_ns = 0
        self.listened: dict[int, tuple] = {}

    def robot_manager(self, name: str = "jackal"):
        from arena_rclpy_mixins.shared import Namespace
        from arena_robots.Robot import RobotIdentifier

        from arena_runtime.env.robot_manager.robot_manager import RobotManager
        from arena_runtime.shared import Pose, Robot

        robot = Robot(name=name, pose=Pose(), model=RobotIdentifier.parse("jackal"), adapters={"mobile": "none"}, extra={})
        rm = RobotManager(node=self.node, namespace=Namespace(self.node.get_namespace())(self.node.get_name()), environment_manager=self.env, robot=robot)
        self.trackers.append(rm._adapter._collision)
        return rm, rm._adapter._collision

    def tracker(self, rm, polygons: dict, footprint: list[list[float]] | None):
        from arena_runtime.env.robot_manager.collision_tracker import CollisionTrackerNode

        tracker = CollisionTrackerNode(rm, polygons, footprint=footprint)
        self.node.executor.add_node(tracker)
        self.trackers.append(tracker)
        return tracker

    def place(self, rm, x: float, y: float, yaw: float = 0.0, after: float = 1.0) -> None:
        import geometry_msgs.msg

        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = "map"
        t.header.stamp.sec, t.header.stamp.nanosec = divmod(self.armed_ns + round(after * 1e9), 10**9)
        t.child_frame_id = rm.base_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.node.tf_buffer.set_transform(t, "test")

    async def listen(self, tracker, state: bool = False, markers: bool = False) -> list:
        import arena_robots_msgs.msg
        import nav2_msgs.msg
        import visualization_msgs.msg

        received: list = []
        if markers:
            msg_type, topic, publisher = visualization_msgs.msg.MarkerArray, "collision_markers", tracker._pub_markers.publisher
        elif state:
            msg_type, topic, publisher = nav2_msgs.msg.CollisionMonitorState, "collision_monitor_state", tracker._pub_state.publisher
        else:
            msg_type, topic, publisher = arena_robots_msgs.msg.CollisionEvents, "collision_events", tracker._pub_events
        self.probe.create_subscription(msg_type, f"{tracker.get_namespace()}/{topic}", received.append, 10)
        await _until(lambda: publisher.get_subscription_count() > 0, f"the {topic} subscription")
        self.listened[id(received)] = (msg_type, publisher)
        return received

    async def arm(self, tracker) -> None:
        import std_msgs.msg

        await _until(lambda: self.reset_end.get_subscription_count() >= len(self.trackers), "the reset_end subscriptions")
        self.reset_end.publish(std_msgs.msg.Empty())
        await _until(lambda: tracker._valid_after is not None, "reset_end to arm the tracker")
        self.armed_ns = tracker._valid_after.to_nanoseconds()

    async def peds(self, tracker, *peds: tuple) -> None:
        """Publish a roster (name, x, y[, radius][, held device]) the way the env's human simulator does, which hands it to the trackers in-process."""
        from arena_people_msgs.msg import HeldDevice, Pedestrian, Pedestrians

        msg = Pedestrians()
        for i, (name, x, y, *extra) in enumerate(peds):
            radius = next((float(v) for v in extra if isinstance(v, (int, float))), 0.0)
            device = next((v for v in extra if isinstance(v, HeldDevice)), HeldDevice())
            ped = Pedestrian(name=name, id=i, radius=radius, device=device)
            ped.pose.position.x = x
            ped.pose.position.y = y
            ped.pose.orientation.w = 1.0
            msg.pedestrians.append(ped)
        self.human.publish_arena_peds(msg)
        assert tracker._peds_msg is not None and len(tracker._peds_msg.pedestrians) == len(peds)

    async def tick(self, tracker, received: list) -> list:
        seen = len(received)
        tracker._tick()
        await _until(lambda: len(received) > seen, "a published message")
        await asyncio.sleep(0.01)  # a collision reaches the robot manager through the event loop
        return list(received[-1].events)

    async def markers(self, tracker, received: list) -> list:
        seen = len(received)
        tracker._tick()
        await _until(lambda: len(received) > seen, "a published marker array")
        return list(received[-1].markers)

    async def silent_tick(self, tracker, received: list) -> None:
        msg_type, publisher = self.listened[id(received)]
        seen = len(received)
        tracker._tick()
        publisher.publish(msg_type())
        await _until(lambda: len(received) > seen, "the marker after a silent tick")
        assert received[seen:] == [msg_type()]
        received.pop()


@pytest.fixture
def world_dir(tmp_path):
    from arena_simulation_setup.tree import DynamicPaths

    crate = tmp_path / "assets" / "Common" / "Object" / "crate"
    crate.mkdir(parents=True)
    (crate / "annotation.yaml").write_text("bounding_box: [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]]\n")
    (tmp_path / "assets" / "Common" / "Material" / "plaster").mkdir(parents=True)
    previous = DynamicPaths.WORLD.path
    DynamicPaths.WORLD.path = tmp_path
    try:
        yield tmp_path
    finally:
        DynamicPaths.WORLD.path = previous


@pytest.fixture
def on_stage(world_dir):
    def run(body: typing.Callable[[Stage], typing.Awaitable[None]], host: typing.Callable[[], object] | None = None) -> None:
        import rclpy
        from arena_rclpy_mixins.spin import create_executor
        from arena_runtime.testing.env_host import env_host
        from arena_runtime.sim.dummy_simulator import DummySimulator

        from arena_runtime.env.constants import Constants
        from arena_runtime.env.environment_manager import EnvironmentManager
        from arena_runtime.env.realizer import Realizer
        from arena_runtime.acoustics import AcousticsSimulatorRegistry
        from arena_runtime.humans.noop import NoopHumanSimulator

        async def main() -> None:
            executor = create_executor()
            node = (host or env_host)()
            probe = rclpy.create_node(f"collision_probe_{uuid.uuid4().hex[:8]}")
            executor.add_node(node)
            executor.add_node(probe)
            spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
            stage = None
            try:
                realizer = Realizer(level_origins={"0": (0.0, 0.0)})
                sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
                acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
                human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
                env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
                await env.spawn_world_obstacles(_level(), "", None, _world_map())
                stage = Stage(node, env, human, probe)
                await body(stage)
            finally:
                node._heartbeat_stop.set()
                executor.shutdown()
                await spin
                for tracker in stage.trackers if stage is not None else ():
                    tracker.shutdown()
                probe.destroy_node()
                node.destroy_node()

        asyncio.run(main())

    return run


def _level():
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
    from arena_simulation_setup.tree.World import Level
    from arena_simulation_setup.tree.World.model import WallPath
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    crate = Obstacle(name="crate", pose=Pose(Position(*CRATE, 0.0), Orientation.from_yaw(0.0)), model=ObjectIdentifier("crate"))
    return Level(walls=[WallPath(path=[Position(0.0, 5.5), Position(4.0, 5.5)], material="plaster")], objects=[crate])


def _world_map():
    from arena_rclpy_mixins.Time import Time

    from arena_runtime.env.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from arena_runtime.shared import Position

    grid = np.full((10, 10), WorldOccupancy.EMPTY, dtype=np.uint8)
    grid[0:3, 7:10] = WorldOccupancy.FULL
    return WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=0.0, y=0.0), resolution=1.0, time=Time())


def _event(event) -> tuple:
    return event.kind, event.obstacle_id, event.distance, (event.obstacle_position.x, event.obstacle_position.y)


def _spec(name: str, kind: str, *, points=None, radius=None, action=None):
    from arena_robots.caps import PolygonSpec

    return PolygonSpec(name=name, type=kind, points=points, radius=radius, action_type=action, polygon_pub_topic=None, min_points=None, visualize=None, enabled=None, slowdown_ratio=None)


def test_mobile_adapter_tracks_the_robot_footprint_and_cap_polygons(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        assert tracker._footprint_base.bounds == pytest.approx((-0.229, -HALF_WIDTH, 0.229, HALF_WIDTH))
        assert {name: (entry["type"], entry["action_code"]) for name, entry in tracker._poly_cache.items()} == {"StopPolygon": ("polygon", 1), "SlowPolygon": ("circle", 2)}
        assert tracker._poly_cache["SlowPolygon"]["R_c"] == 0.25
        assert tracker.get_namespace() == str(rm.namespace)

    on_stage(body)


def test_map_cell_under_the_footprint_is_a_static_collision_at_the_robot(on_stage):
    import arena_robots_msgs.msg

    from arena_runtime.env.collision_grid import MAP_ID

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_STATIC, MAP_ID, 0.0, IN_MAP)]
        assert received[-1].header.frame_id == "map"

    on_stage(body)


def test_authored_wall_under_the_footprint_is_a_wall_collision(on_stage):
    import arena_robots_msgs.msg

    from arena_runtime.env.collision_grid import WALL_ID

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *ON_WALL)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_WALL, WALL_ID, 0.0, ON_WALL)]

    on_stage(body)


def test_static_obstacle_collision_reports_the_obstacle_centroid(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        assert set(stage.env.static_polygons) == {"crate"}
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, CRATE[0] + 0.1, CRATE[1], math.pi / 2)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_STATIC, "crate", 0.0, CRATE)]

    on_stage(body)


def test_free_space_publishes_an_empty_event_list(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *CLEAR)
        assert await stage.tick(tracker, received) == []

    on_stage(body)


def test_pedestrian_within_its_radius_collides_and_nan_pedestrians_are_skipped(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.peds(tracker, ("walker", 2.0, 8.6), ("far", 6.0, 6.0), ("lost", math.nan, 8.2))
        stage.place(rm, *BY_PED)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN, "walker", 0.0, (2.0, 8.6))]

    on_stage(body)


def test_pedestrian_just_beyond_its_radius_does_not_collide(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.peds(tracker, ("walker", BY_PED[0], BY_PED[1] + HALF_WIDTH + PED_RADIUS + 0.01))
        stage.place(rm, *BY_PED)
        assert await stage.tick(tracker, received) == []

    on_stage(body)


def test_pedestrian_collides_within_its_own_radius_beyond_the_default(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        beyond_default = BY_PED[1] + HALF_WIDTH + PED_RADIUS + 0.01
        await stage.peds(tracker, ("wheelchair", BY_PED[0], beyond_default, WHEELCHAIR_RADIUS))
        stage.place(rm, *BY_PED)
        assert [_event(e) for e in await stage.tick(tracker, received)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN, "wheelchair", 0.0, (BY_PED[0], beyond_default))]

    on_stage(body)


def test_wall_and_pedestrian_together_yield_both_events_wall_first(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.peds(tracker, ("walker", ON_WALL[0] + 0.3, ON_WALL[1]))
        stage.place(rm, *ON_WALL)
        assert [e.kind for e in await stage.tick(tracker, received)] == [arena_robots_msgs.msg.CollisionEvent.KIND_WALL, arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN]

    on_stage(body)


def _cane(x: float, y: float, heading: float):
    """A 1 m cane ahead of a pedestrian at (x, y) along `heading`, on the ground."""
    from arena_people_msgs.msg import HeldDevice
    from geometry_msgs.msg import Point32

    c, s = math.cos(heading), math.sin(heading)
    outline = [(0.3, -0.0125), (1.3, -0.0125), (1.3, 0.0125), (0.3, 0.0125)]
    return HeldDevice(kind="cane", footprint=[Point32(x=x + c * u - s * v, y=y + s * u + c * v, z=0.0) for u, v in outline])


def test_a_held_device_under_the_footprint_collides_apart_from_its_holder(on_stage):
    import arena_robots_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        holder = (CLEAR[0], CLEAR[1] + 1.3)
        await stage.peds(tracker, ("blind", *holder, _cane(*holder, -math.pi / 2)), ("walker", 6.0, 6.0))
        stage.place(rm, *CLEAR)
        (event,) = await stage.tick(tracker, received)
        assert (event.kind, event.device, event.obstacle_id) == (arena_robots_msgs.msg.CollisionEvent.KIND_DEVICE, "cane", "blind")
        assert (event.obstacle_position.x, event.obstacle_position.y) == pytest.approx((CLEAR[0], CLEAR[1] + HALF_WIDTH / 2), abs=0.02)
        hit = tracker._monitor_state(*CLEAR, 0.0)
        assert hit.polygon_name != ""

    on_stage(body)


def test_collision_markers_turn_the_footprint_and_the_touched_device_red(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker, markers=True)
        await stage.arm(tracker)
        holder, beside = (CLEAR[0], CLEAR[1] + 1.3), (CLEAR[0] + 3.0, CLEAR[1] + 1.3)
        await stage.peds(tracker, ("blind", *holder, _cane(*holder, -math.pi / 2)), ("other", *beside, _cane(*beside, -math.pi / 2)))
        stage.place(rm, *CLEAR)
        robot, devices, column = await stage.markers(tracker, received)
        assert (robot.ns, robot.color.r > 0.8) == ("footprint", True)
        assert robot.points
        assert len(devices.points) == len(devices.colors) == 16
        assert [c.r > 0.8 for c in devices.colors] == [True] * 8 + [False] * 8
        assert (column.ns, column.lifetime.sec) == ("contacts", 1)
        assert (column.pose.position.x, column.pose.position.y) == pytest.approx((CLEAR[0], CLEAR[1] + HALF_WIDTH / 2), abs=0.02)
        await stage.peds(tracker)
        idle, cleared = await stage.markers(tracker, received)
        assert idle.color.r < 0.8
        assert cleared.action == cleared.DELETE

    on_stage(body)


def test_a_device_beside_the_footprint_and_one_of_a_lost_pedestrian_do_not_collide(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        beside = (CLEAR[0] + 0.5, CLEAR[1] + 1.3)
        await stage.peds(tracker, ("blind", *beside, _cane(*beside, -math.pi / 2)), ("lost", math.nan, math.nan, _cane(math.nan, math.nan, 0.0)))
        stage.place(rm, *CLEAR)
        assert await stage.tick(tracker, received) == []

    on_stage(body)


def test_tracker_ignores_poses_before_the_first_reset_end(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        stage.place(rm, *IN_MAP, after=1e6)
        await stage.silent_tick(tracker, received)
        await stage.arm(tracker)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_tracker_ignores_poses_stamped_up_to_the_reset_end(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP, after=0.0)
        await stage.silent_tick(tracker, received)
        stage.place(rm, *IN_MAP, after=0.5)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_reset_start_disarms_the_tracker_until_the_next_reset_end(on_stage):
    import std_msgs.msg

    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.reset_start.publish(std_msgs.msg.Empty())
        await _until(lambda: tracker._valid_after is None, "reset_start to disarm the tracker")
        stage.place(rm, *IN_MAP, after=1e6)
        await stage.silent_tick(tracker, received)
        await stage.arm(tracker)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_tracker_is_silent_without_a_robot_transform(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        await stage.silent_tick(tracker, received)
        stage.place(rm, *IN_MAP)
        assert len(await stage.tick(tracker, received)) == 1

    on_stage(body)


def test_reset_end_drops_the_pedestrians_of_the_previous_episode(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        received = await stage.listen(tracker)
        await stage.peds(tracker, ("walker", 2.0, 8.6))
        await stage.arm(tracker)
        assert tracker._peds_msg is None
        stage.place(rm, *BY_PED)
        assert await stage.tick(tracker, received) == []

    on_stage(body)


def test_monitor_state_prefers_the_strictest_hit_action(on_stage):
    async def body(stage: Stage):
        _, tracker = stage.robot_manager()
        edge = tracker._monitor_state(IN_MAP[0] - 0.22, IN_MAP[1], 0.0)
        assert (edge.polygon_name, edge.action_type) == ("SlowPolygon", 2)
        inside = tracker._monitor_state(*IN_MAP, 0.0)
        assert (inside.polygon_name, inside.action_type) == ("StopPolygon", 1)
        clear = tracker._monitor_state(*CLEAR, 0.0)
        assert (clear.polygon_name, clear.action_type) == ("", 0)

    on_stage(body)


def test_monitor_state_is_published_on_tick(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        states = await stage.listen(tracker, state=True)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        tracker._tick()
        await _until(lambda: bool(states), "a collision monitor state")
        assert (states[-1].polygon_name, states[-1].action_type) == ("StopPolygon", 1)

    on_stage(body)


def test_monitor_state_counts_pedestrians_and_ranks_unknown_actions_last(on_stage):
    async def body(stage: Stage):
        rm, _ = stage.robot_manager()
        polygons = {
            "wide": _spec("wide", "circle", radius=1.0),
            "near": _spec("near", "polygon", points=[[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]], action="limit"),
            "line": _spec("line", "polygon", points=[[0.0, 0.0], [1.0, 0.0]], action="stop"),
            "open": _spec("open", "polygon", action="stop"),
        }
        tracker = stage.tracker(rm, polygons, footprint=None)
        assert set(tracker._poly_cache) == {"wide", "near"}
        assert tracker._pub_events is None
        await stage.peds(tracker, ("walker", 3.0, 2.0))
        hit = tracker._monitor_state(*CLEAR, 0.0)
        assert (hit.polygon_name, hit.action_type) == ("near", 4)
        turned = tracker._monitor_state(*CLEAR, math.pi)
        assert (turned.polygon_name, turned.action_type) == ("wide", 0)

    on_stage(body)


def test_circle_without_a_radius_is_rejected(on_stage):
    async def body(stage: Stage):
        rm, _ = stage.robot_manager()
        with pytest.raises(ValueError, match="'dot': circle without a radius"):
            stage.tracker(rm, {"dot": _spec("dot", "circle", action="stop")}, footprint=None)

    on_stage(body)


def test_tracker_without_a_footprint_publishes_only_the_monitor_state(on_stage):
    async def body(stage: Stage):
        rm, _ = stage.robot_manager()
        tracker = stage.tracker(rm, {"wide": _spec("wide", "circle", radius=1.0, action="slowdown")}, footprint=[[0.0, 0.0], [1.0, 0.0]])
        assert tracker._footprint_base is None
        assert tracker._pub_events is None
        states = await stage.listen(tracker, state=True)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        tracker._tick()
        await _until(lambda: any(state.polygon_name == "wide" for state in states), "the monitor state of the footprintless tracker")

    on_stage(body)


def test_collision_callback_fires_only_on_a_collision(on_stage):
    async def body(stage: Stage):
        rm, tracker = stage.robot_manager()
        reported: list[bool] = []
        rm.bind_collision(lambda: reported.append(True))
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        assert len(await stage.tick(tracker, received)) == 1
        assert reported == [True]
        stage.place(rm, *CLEAR, after=2.0)
        assert await stage.tick(tracker, received) == []
        assert reported == [True]

    on_stage(body)


def test_collision_fails_the_episode_only_under_fail_on_collision(on_stage, fake_task_layer):
    from types import SimpleNamespace

    from rclpy.parameter import Parameter

    from arena_runtime.env.episode_node import EpisodeNode

    async def body(stage: Stage):
        stage.node._task = SimpleNamespace()
        rm, tracker = stage.robot_manager()
        rm.bind_collision(stage.node._on_collision)
        received = await stage.listen(tracker)
        await stage.arm(tracker)
        stage.place(rm, *IN_MAP)
        assert len(await stage.tick(tracker, received)) == 1
        assert stage.node._abort_reason is None
        assert stage.node.set_parameters([Parameter("task.episode.fail_on_collision", value=True)])[0].successful
        stage.place(rm, *CLEAR, after=2.0)
        assert await stage.tick(tracker, received) == []
        assert stage.node._abort_reason is None
        stage.place(rm, *IN_MAP, after=3.0)
        assert len(await stage.tick(tracker, received)) == 1
        assert stage.node._abort_reason == "collision"

    on_stage(body, host=EpisodeNode)


def test_a_roster_is_indexed_once_for_every_tracker_of_the_env(on_stage, monkeypatch):
    import arena_robots_msgs.msg

    from arena_runtime.env.robot_manager import collision_tracker

    built: list[int] = []
    index = collision_tracker.PedestrianIndex

    class Counted(index):
        def __init__(self, peds, default_radius):
            built.append(len(peds))
            super().__init__(peds, default_radius)

    monkeypatch.setattr(collision_tracker, "PedestrianIndex", Counted)

    async def body(stage: Stage):
        robots = [stage.robot_manager(f"jackal_{i}") for i in range(3)]
        received = [await stage.listen(tracker) for _, tracker in robots]
        await stage.arm(robots[0][1])
        await stage.peds(robots[0][1], ("walker", *BY_PED), ("far", 6.0, 6.0))
        for rm, _ in robots:
            stage.place(rm, *BY_PED)
        built.clear()
        for (_, tracker), seen in zip(robots, received, strict=True):
            assert [_event(e) for e in await stage.tick(tracker, seen)] == [(arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN, "walker", 0.0, BY_PED)]
        assert built == [2]

    on_stage(body)


CROWD = 1500
ROBOTS = 4
PING_TIMEOUT = 2.0
PING_BUDGET = 1.0


def _crowd(seed: int) -> list[tuple[str, float, float]]:
    rng = np.random.default_rng(seed)
    return [(f"walker_{i}", float(x), float(y)) for i, (x, y) in enumerate(rng.uniform((20.0, 0.0), (170.0, 60.0), (CROWD, 2)))]


def test_trackers_keep_answering_beside_a_large_crowd(on_stage):
    """Four robots' trackers ticking on the sim clock beside a large roster renewed at 10 Hz, the env's main thread busy:
    each tracker answers its parameter service (spun by the env's side executor) within a second.

    The round trip is timed off the event loop, by a client the probe's executor spins, waited for in a worker thread:
    the loop itself is blocked by the test's own load (a 1500-walker roster takes 0.4-0.8 s to publish on 2 cores, plus
    gen-2 collections of 0.1-0.2 s), which would be timed instead of the tracker."""
    from rcl_interfaces.srv import GetParameters
    from rosgraph_msgs.msg import Clock

    from arena_rclpy_mixins.Time import Time

    async def body(stage: Stage):
        robots = [stage.robot_manager(f"jackal_{i}") for i in range(ROBOTS)]
        await stage.arm(robots[0][1])
        for i, (rm, _) in enumerate(robots):
            stage.place(rm, *CLEAR, yaw=0.1 * i)
        published = [await stage.listen(tracker) for _, tracker in robots]
        clock = stage.probe.create_publisher(Clock, "/clock", 10)
        clients = [stage.probe.create_client(GetParameters, f"{tracker.get_fully_qualified_name()}/get_parameters") for _, tracker in robots]
        for client in clients:
            assert await asyncio.to_thread(client.wait_for_service, timeout_sec=5.0)
        stop = asyncio.Event()

        async def sim() -> None:
            now = 2.0
            while not stop.is_set():
                now += 1.0 / 30.0
                clock.publish(Clock(clock=Time.from_float(now).to_msg()))
                await asyncio.sleep(1.0 / 30.0)

        async def crowd() -> None:
            seed = 0
            while not stop.is_set():
                seed += 1
                await stage.peds(robots[0][1], *_crowd(seed))
                await asyncio.sleep(0.1)

        async def main_thread_load() -> None:
            while not stop.is_set():
                end = time.perf_counter() + 0.004
                while time.perf_counter() < end:
                    pass
                await asyncio.sleep(0.004)

        def round_trip(client) -> float:
            answered = threading.Event()
            start = time.monotonic()
            future = client.call_async(GetParameters.Request(names=["use_sim_time"]))
            future.add_done_callback(lambda _: answered.set())
            if not answered.wait(PING_TIMEOUT):
                client.remove_pending_request(future)
                return math.inf
            return time.monotonic() - start

        async def ping(client) -> float:
            return await asyncio.to_thread(round_trip, client)

        tasks = [asyncio.create_task(job()) for job in (sim, crowd, main_thread_load)]
        try:
            await _until(lambda: all(len(seen) > 0 for seen in published), "the first ticks")
            latencies: list[float] = []
            for _ in range(12):
                latencies.extend(await asyncio.gather(*(ping(client) for client in clients)))
                await asyncio.sleep(0.5)
        finally:
            stop.set()
            await asyncio.gather(*tasks)
        assert all(len(seen) >= 10 for seen in published), f"the trackers ticked {[len(seen) for seen in published]} times"
        assert max(latencies) < PING_BUDGET, f"slowest parameter ping {max(latencies):.2f}s, median {sorted(latencies)[len(latencies) // 2]:.2f}s"

    on_stage(body)


def test_the_tracker_ticks_once_per_mark_of_the_sim_clock_and_skips_what_the_clock_jumped_past(on_stage):
    """Ticked from its node's /clock listener, not an rclpy sim-time timer: one tick per 0.1 s of sim time, a jump ticks once."""
    from arena_rclpy_mixins.Time import Time

    async def body(stage: Stage):
        _, tracker = stage.robot_manager()
        ticks: list[int] = []
        clock = [0]
        tracker._tick = lambda: ticks.append(clock[0])
        for ms in (2000, 2050, 2100, 2150, 2210, 2300, 7300, 7350, 7410):
            clock[0] = ms
            tracker._on_clock(Time.from_nanoseconds(ms * 1_000_000))
        assert ticks == [2100, 2210, 2300, 7300, 7410]
        tracker._on_clock(Time.from_nanoseconds(1_000_000_000))  # a new episode's clock starts over
        tracker._on_clock(Time.from_nanoseconds(1_100_000_000))
        assert len(ticks) == 6

    on_stage(body)


def test_the_tracker_ticks_on_the_side_executor_thread(on_stage):
    """The /clock listener that ticks the tracker runs on the env's side executor, like its parameter service, not behind the env executor's queue."""
    import threading

    from rosgraph_msgs.msg import Clock

    from arena_rclpy_mixins.Time import Time

    async def body(stage: Stage):
        _, tracker = stage.robot_manager()
        threads: list[str] = []
        tracker._tick = lambda: threads.append(threading.current_thread().name)
        clock = stage.probe.create_publisher(Clock, "/clock", 10)
        await _until(lambda: clock.get_subscription_count() >= 2, "the tracker's /clock subscription")
        now = 2.0
        for _ in range(100):
            now += 0.05
            clock.publish(Clock(clock=Time.from_float(now).to_msg()))
            await asyncio.sleep(0.01)
            if len(threads) >= 3:
                break
        assert threads and set(threads) == {f"{stage.node.get_name()}_side"}, threads

    on_stage(body)
