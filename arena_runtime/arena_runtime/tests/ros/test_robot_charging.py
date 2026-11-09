"""A charger charges the first robot parked in its dock with its power, each robot starts an episode with a charge drawn per seed and name on a battery of the episode's capacity, and the runtime reads the charge back."""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Callable
from types import SimpleNamespace

import pytest

try:
    import arena_robots_msgs.msg
    import rclpy
    import rclpy.node
    import std_msgs.msg
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_rclpy_mixins.shared import Namespace
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.shared.judge import Sample, Zones, zone_polygons
    from arena_simulation_setup.tree.World import Level
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState

    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.env.constants.runtime import Configuration
    from arena_runtime.env.robot_manager.charging import PARK_TIME, Chargers
    from arena_runtime.env.robot_manager.robot_manager import _LATCHED, RobotManager
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

DOCK = [[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]]
# every object faces -x, so each dock spans x 2.5..3.5: charger_1 y 1.5..2.5, the locker (no charger) y 3.5..4.5, charger_2 y 5.5..6.5
CHARGERS = {"charger_1": 1620.0, "charger_2": 800.0}
TICK = 1.0 / 30.0
TOPIC = "/test_robot_charging/jackal_0/power_publisher/charge_power"


def _zones() -> Zones:
    def obstacle(name: str, y: float, power: float | None) -> Obstacle:
        return Obstacle(name=name, model=name, pose=Pose(position=Position(4.0, y), orientation=Orientation.from_yaw(math.pi)), dock=DOCK, charge_power_w=power)

    return zone_polygons(Level(id="0", objects=[obstacle("charger_1", 2.0, 1620.0), obstacle("locker_1", 4.0, None), obstacle("charger_2", 6.0, 800.0)]))


def _at(t: float, **robots: tuple[float, float]) -> Sample:
    return Sample(t=t, robots={name: (x, y, 0.0) for name, (x, y) in robots.items()})


def test_a_robot_charges_once_parked_in_a_chargers_dock_and_not_while_driving_through() -> None:
    chargers, zones = Chargers(CHARGERS), _zones()
    for step in range(121):
        assert chargers.step(_at(step * TICK, jackal_0=(2.4 + step * 0.01, 2.0)), zones) == {"jackal_0": 0.0}
    stop = 121 * TICK
    assert chargers.step(_at(stop, jackal_0=(3.0, 2.0)), zones) == {"jackal_0": 0.0}
    assert chargers.step(_at(stop + PARK_TIME / 2, jackal_0=(3.0, 2.0)), zones) == {"jackal_0": 0.0}
    assert chargers.step(_at(stop + PARK_TIME, jackal_0=(3.0, 2.0)), zones) == {"jackal_0": 1620.0}
    assert chargers.step(_at(stop + PARK_TIME + 1.0, jackal_0=(3.0, 2.04)), zones) == {"jackal_0": 1620.0}


def test_a_robot_parked_in_a_docked_object_without_charge_power_does_not_charge() -> None:
    chargers, zones = Chargers(CHARGERS), _zones()
    chargers.step(_at(0.0, jackal_0=(3.0, 4.0)), zones)
    assert chargers.step(_at(PARK_TIME, jackal_0=(3.0, 4.0)), zones) == {"jackal_0": 0.0}


def test_a_charger_charges_the_first_robot_parked_in_its_dock_until_it_leaves() -> None:
    chargers, zones = Chargers(CHARGERS), _zones()
    chargers.step(_at(0.0, jackal_1=(1.0, 2.0), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones)
    assert chargers.step(_at(PARK_TIME / 2, jackal_1=(3.3, 2.3), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 0.0, "jackal_0": 0.0, "jackal_2": 0.0}
    assert chargers.step(_at(PARK_TIME, jackal_1=(3.3, 2.3), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 0.0, "jackal_0": 1620.0, "jackal_2": 800.0}
    assert chargers.step(_at(2 * PARK_TIME, jackal_1=(3.3, 2.3), jackal_0=(3.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 0.0, "jackal_0": 1620.0, "jackal_2": 800.0}
    assert chargers.step(_at(3 * PARK_TIME, jackal_1=(3.3, 2.3), jackal_0=(1.0, 2.0), jackal_2=(3.0, 6.0)), zones) == {"jackal_1": 1620.0, "jackal_0": 0.0, "jackal_2": 800.0}


def test_the_charge_power_goes_out_on_change() -> None:
    node = rclpy.node.Node("test_robot_charging")
    received: list[float] = []
    node.create_subscription(std_msgs.msg.Float64, TOPIC, lambda msg: received.append(msg.data), _LATCHED)
    manager = RobotManager.__new__(RobotManager)
    manager._charge_power_w = 0.0
    manager._charge_pub = node.create_publisher(std_msgs.msg.Float64, TOPIC, _LATCHED)

    def spin_until(count: int) -> None:
        deadline = time.monotonic() + 5.0
        while len(received) < count and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        manager.publish_charge_power(0.0)
        manager.publish_charge_power(1620.0)
        spin_until(1)
        manager.publish_charge_power(1620.0)
        manager.publish_charge_power(0.0)
        spin_until(2)
        assert received == [1620.0, 0.0]
    finally:
        node.destroy_node()


def _manager_in_env(env: int, name: str) -> RobotManager:
    manager = RobotManager.__new__(RobotManager)
    manager._namespace = Namespace(f"/arena/env_{env}/arena_env_node")
    manager._robot = SimpleNamespace(name=name)
    return manager


def _rng(seed: int) -> EpisodeRng:
    rng = EpisodeRng()
    rng.reseed(seed)
    return rng


def test_the_initial_charge_is_drawn_per_seed_and_robot_name_whatever_the_env_index() -> None:
    names = ("jackal", "jackal_1", "jackal_2", "jackal_3")
    socs = {(seed, name): _manager_in_env(0, name).initial_soc(_rng(seed), 40.0, 100.0) for seed in range(1, 11) for name in names}
    assert all(40.0 <= soc <= 100.0 for soc in socs.values())
    assert len(set(socs.values())) == len(socs)
    for env in (1, 7):
        assert _manager_in_env(env, "jackal").namespace == f"/arena/env_{env}/arena_env_node/jackal"
        assert _manager_in_env(env, "jackal").initial_soc(_rng(3), 40.0, 100.0) == socs[(3, "jackal")]


def test_the_initial_charge_ignores_the_episodes_other_draws() -> None:
    rng = _rng(3)
    rng.stream("robots", "target", "jackal").uniform()
    rng.stream("robot-adapter", "jackal").uniform()
    assert _manager_in_env(0, "jackal").initial_soc(rng, 40.0, 100.0) == _manager_in_env(0, "jackal").initial_soc(_rng(3), 40.0, 100.0)


def _soc_range(**soc: float) -> tuple[float, float]:
    node = ROSParamServer("test_battery_soc", parameter_overrides=[rclpy.Parameter("world", value="map_empty"), *(rclpy.Parameter(f"task.episode.battery.soc.{key}", value=value) for key, value in soc.items())])
    try:
        robot = Configuration(node).Robot
        return robot.BATTERY_SOC_MIN.value, robot.BATTERY_SOC_MAX.value
    finally:
        node.destroy_node()


def test_the_start_charge_range_defaults_to_40_to_100_percent_and_takes_launch_values() -> None:
    assert _soc_range() == (40.0, 100.0)
    assert _soc_range(min=0.0, max=55.5) == (0.0, 55.5)


@pytest.mark.parametrize(("key", "value"), [("min", -1.0), ("max", 150.0)])
def test_a_start_charge_bound_outside_0_to_100_percent_is_refused(key: str, value: float) -> None:
    with pytest.raises(RuntimeError, match=f"task.episode.battery.soc.{key} failed"):
        _soc_range(**{key: value})


def test_a_start_charge_range_with_its_min_above_its_max_is_refused() -> None:
    with pytest.raises(ValueError, match=r"task.episode.battery.soc.min \(80.0\) is above task.episode.battery.soc.max \(40.0\)"):
        _manager_in_env(0, "jackal").initial_soc(_rng(3), 80.0, 40.0)


def _capacity(value: object = None) -> float:
    overrides = [rclpy.Parameter("world", value="map_empty")]
    if value is not None:
        overrides.append(rclpy.Parameter("task.episode.battery.capacity_wh", value=value))
    node = ROSParamServer("test_battery_capacity", parameter_overrides=overrides)
    try:
        return Configuration(node).Robot.BATTERY_CAPACITY_WH.value
    finally:
        node.destroy_node()


def test_the_battery_capacity_defaults_to_0_for_the_robots_own_and_takes_launch_values() -> None:
    assert _capacity() == 0.0
    assert _capacity(15.0) == 15.0
    assert _capacity(15) == 15.0
    assert isinstance(_capacity(15), float)


@pytest.mark.parametrize("value", [-1.0, math.inf, math.nan, True], ids=["negative", "inf", "nan", "bool"])
def test_a_battery_capacity_below_0_or_not_a_finite_number_is_refused(value: object) -> None:
    with pytest.raises(RuntimeError, match="task.episode.battery.capacity_wh failed"):
        _capacity(value)


def test_the_reset_sends_the_battery_capacity_before_the_start_charge() -> None:
    manager = _manager_in_env(0, "jackal")
    sent: list[tuple[str, float]] = []
    manager._capacity_pub = SimpleNamespace(publish=lambda msg: sent.append(("capacity_wh", msg.data)))
    manager._soc_pub = SimpleNamespace(publish=lambda msg: sent.append(("soc_percent", msg.data)))
    manager.reset_battery(_rng(3), 15.0, 40.0, 100.0)
    manager.reset_battery(_rng(3), 0.0, 40.0, 100.0)
    soc = manager.initial_soc(_rng(3), 40.0, 100.0)
    assert sent == [("capacity_wh", 15.0), ("soc_percent", soc), ("capacity_wh", 0.0), ("soc_percent", soc)]


def test_a_start_charge_range_with_its_min_above_its_max_fails_the_reset_before_anything_is_sent() -> None:
    manager = _manager_in_env(0, "jackal")
    sent: list[float] = []
    manager._capacity_pub = manager._soc_pub = SimpleNamespace(publish=lambda msg: sent.append(msg.data))
    with pytest.raises(ValueError, match="above task.episode.battery.soc.max"):
        manager.reset_battery(_rng(3), 15.0, 80.0, 40.0)
    assert sent == []


_roots = itertools.count()


def _wired(node: rclpy.node.Node, name: str = "jackal") -> RobotManager:
    """A manager with its real battery topics on `node`, in an env namespace of its own."""
    manager = RobotManager.__new__(RobotManager)
    manager._namespace = Namespace(f"/test_robot_charging_{next(_roots)}/env_0/arena_env_node")
    manager._robot = SimpleNamespace(name=name)
    manager._NodeInterface__node = node
    manager._charge_power_w = 0.0
    manager._soc = None
    manager._set_up_battery()
    return manager


def _spin_until(executor: SingleThreadedExecutor, pred: Callable[[], bool], what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout} s waiting for {what}")
        executor.spin_once(timeout_sec=0.05)


def test_the_charge_reading_follows_the_power_publishers_energy_output_and_starts_at_the_start_charge() -> None:
    runtime = rclpy.node.Node("test_robot_charging_runtime")
    probe = rclpy.node.Node("test_robot_charging_power")
    executor = SingleThreadedExecutor()
    executor.add_node(runtime)
    executor.add_node(probe)
    try:
        manager = _wired(runtime)
        energy = probe.create_publisher(arena_robots_msgs.msg.Energy, str(manager.namespace("power_publisher", "energy")), 10)
        assert manager.soc is None
        # the power publisher sends energy only while someone subscribes, the manager's subscription is that someone
        _spin_until(executor, lambda: energy.get_subscription_count() == 1, "the manager's energy subscription")

        manager.publish_soc(55.0)
        assert manager.soc == 55.0
        for soc in (54.5, 30.25):
            energy.publish(arena_robots_msgs.msg.Energy(total_energy_consumed_wh=1.0, battery_soc_percent=soc))
            _spin_until(executor, lambda soc=soc: manager.soc == soc, f"a {soc} % reading")
    finally:
        executor.shutdown()
        runtime.destroy_node()
        probe.destroy_node()


def _joint_state(sec: int) -> JointState:
    msg = JointState()
    msg.header.stamp.sec = sec
    msg.name = ["front_left_wheel_joint"]
    msg.velocity = [0.0]
    msg.effort = [0.0]
    return msg


def test_a_real_power_publisher_runs_on_the_runtimes_capacity_start_charge_and_charge_power_and_the_runtime_reads_its_charge() -> None:
    from rclpy.parameter import Parameter

    from arena_robots.power_publisher import PowerPublisher

    runtime = rclpy.node.Node("test_robot_charging_runtime")
    sim = rclpy.node.Node("test_robot_charging_sim")
    manager = _wired(runtime)
    # a 36 W standby draw like flatsim's jackal, on its 270 Wh power.yaml battery
    power = PowerPublisher(namespace=str(manager.namespace), parameter_overrides=[Parameter("static_power_w", value=36.0), Parameter("battery_capacity_wh", value=270.0), Parameter("filter_tau_s", value=0.0)])
    joints = sim.create_publisher(JointState, str(manager.namespace("joint_states")), 10)
    executor = SingleThreadedExecutor()
    for node in (runtime, sim, power):
        executor.add_node(node)
    try:
        manager.reset_battery(_rng(3), 15.0, 40.0, 40.0)
        _spin_until(executor, lambda: power._battery_capacity_wh == 15.0 and power._stored_energy_wh == pytest.approx(6.0), "15 Wh at 40 %")
        _spin_until(executor, lambda: joints.get_subscription_count() == 1 and power._energy_pub.wanted, "the joint states and the energy subscription")

        # 36 W for 100 s is 1 Wh off 6 Wh
        joints.publish(_joint_state(0))
        joints.publish(_joint_state(100))
        _spin_until(executor, lambda: manager.soc == pytest.approx(5.0 / 15.0 * 100.0, abs=1e-4), "a 33.3 % reading")

        # a 125 W charger for 100 s puts back 89 W net, 2.47 Wh
        manager.publish_charge_power(125.0)
        _spin_until(executor, lambda: power._charge_power_w == 125.0, "the charge power")
        joints.publish(_joint_state(200))
        _spin_until(executor, lambda: manager.soc == pytest.approx((5.0 + 89.0 * 100.0 / 3600.0) / 15.0 * 100.0, abs=1e-4), "a 49.8 % reading")

        # back to the robot's own 270 Wh, at the same percentage
        manager.reset_battery(_rng(3), 0.0, 40.0, 40.0)
        _spin_until(executor, lambda: power._battery_capacity_wh == 270.0 and power._stored_energy_wh == pytest.approx(108.0), "270 Wh at 40 %")
    finally:
        executor.shutdown()
        power.destroy_node()
        runtime.destroy_node()
        sim.destroy_node()
