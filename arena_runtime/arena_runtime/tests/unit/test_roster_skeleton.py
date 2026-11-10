"""human.skeleton:=false: the roster goes out without joint states and body poses, every other field as with them."""

from __future__ import annotations

import asyncio
import copy
import math

from arena_people_msgs.msg import Pedestrian, Pedestrians


def _roster() -> Pedestrians:
    msg = Pedestrians()
    msg.header.stamp.sec = 3
    for i in range(6):
        ped = Pedestrian(name=f"ped_{i}", id=i, radius=0.3 + 0.01 * i)
        ped.pose.position.x = float(i)
        ped.pose.position.y = 2.0
        ped.pose.orientation.z = math.sin(0.2 * i)
        ped.pose.orientation.w = math.cos(0.2 * i)
        ped.twist.linear.x = 1.2
        ped.animation_state = Pedestrian.WALKING if i % 2 else Pedestrian.IDLE
        ped.gait_phase = 0.5 * i + 0.1
        msg.pedestrians.append(ped)
    return msg


def _published(skeleton: bool) -> list[Pedestrians]:
    from arena_rclpy_mixins.spin import create_executor

    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from arena_runtime.testing.env_host import env_host

    out: list[Pedestrians] = []

    async def main() -> None:
        executor = create_executor()
        node = env_host()
        executor.add_node(node)
        spin = asyncio.get_running_loop().run_in_executor(None, executor.spin)
        try:
            realizer = Realizer(level_origins={"0": (0.0, 0.0)})
            sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics, skeleton=skeleton)
            human.on_arena_peds(out.append)
            for _ in range(3):
                human.publish_arena_peds(copy.deepcopy(_roster()))
        finally:
            node._heartbeat_stop.set()
            executor.shutdown()
            await spin
            node.destroy_node()

    asyncio.run(main())
    return out


def _without_skeleton(ped: Pedestrian) -> Pedestrian:
    bare = copy.deepcopy(ped)
    bare.joint_state.name, bare.joint_state.position = [], []
    bare.joint_state.header = type(bare.joint_state.header)()
    bare.body = type(bare.body)()
    return bare


def test_the_roster_without_skeleton_is_the_full_roster_less_joints_and_body() -> None:
    full, lean = _published(True), _published(False)
    assert len(full) == len(lean) == 3
    assert all(ped.body.joints for msg in full for ped in msg.pedestrians if ped.animation_state == Pedestrian.WALKING)
    for with_skeleton, without in zip(full, lean, strict=True):
        assert without.header == with_skeleton.header
        for a, b in zip(with_skeleton.pedestrians, without.pedestrians, strict=True):
            assert list(b.joint_state.name) == [] and list(b.body.joints) == []
            assert b == _without_skeleton(a)
