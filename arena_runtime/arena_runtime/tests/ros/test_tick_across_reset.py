"""A robot tick still in flight when an episode reset starts (one awaiting a late goal reply) is cancelled before the
robots begin the next episode: it leaves no trace in that episode, even when its reply comes during the reset."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.msg")

TIMEOUT = 60.0


def _held_robot() -> object:
    from arena_runtime.env.robot_manager.tick_in_flight import TickInFlight

    class HeldRobot(TickInFlight):
        """A robot whose first tick awaits a goal reply that comes only once the next episode has begun."""

        def __init__(self) -> None:
            self.name, self.model_name = "held", "fake_held"
            self.pose = self.goal = None
            self.placed, self.waiting = True, False
            self.safe_distance = 0.3
            self.runner = SimpleNamespace(active=None, changed=False, phases=[], violated=[], serialize=dict, outcomes=lambda outcome: [])
            self.reply = asyncio.Event()
            self.episode = 0
            self.done: list[tuple[int, int]] = []  # (episode the tick started in, episode it ended in)

        def begin_episode(self) -> None:
            self.episode += 1
            self.reply.set()  # the late reply arrives inside the reset, once this episode has begun

        async def tick(self, sample: object, zones: object) -> object:
            del sample, zones
            started = self.episode
            if started == 0:
                await self.reply.wait()
            self.done.append((started, self.episode))
            return SimpleNamespace(ended=None)

        def publish_task_pose(self, pose: object) -> None:
            del pose

        def publish_charge_power(self, watts: float) -> None:
            del watts

        def set_episode_conditions(self, conditions: object) -> None:
            del conditions

        async def reset(self, ctx: object) -> dict:
            del ctx
            return {}

        def reset_battery(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def finish_episode(self) -> None:
            pass

    return HeldRobot()


def test_a_tick_held_across_a_reset_leaves_no_trace_in_the_next_episode(rclpy_context: object, fake_task_layer: object) -> None:
    import rclpy.parameter
    from arena_runtime_msgs.action import RunEpisode

    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.testing.env_host import arena_on_dummy

    fake_task_layer.task.finished = True

    async def scenario() -> None:
        async with arena_on_dummy(on_taken=pytest.skip) as rig:
            overrides = [
                rclpy.parameter.Parameter("env_id", value=rig.env_id),
                rclpy.parameter.Parameter("world", value="map_empty"),
                rclpy.parameter.Parameter("sim", value="dummy"),
                rclpy.parameter.Parameter("human", value="dummy"),
                rclpy.parameter.Parameter("acoustics", value="none"),
                rclpy.parameter.Parameter("task.episode.auto_reset", value=False),
            ]
            env = EpisodeNode(namespace=f"/{rig.namespace}", parameter_overrides=overrides)
            rig.executor.add_node(env)
            try:
                await asyncio.wait_for(env.setup(), TIMEOUT)
                env.trigger_activate()
                robot = _held_robot()
                env._robots_manager.managers[robot.name] = robot
                env._robots_manager._publish_fleet = lambda: None  # the fleet message describes real robots only
                try:
                    deadline = time.monotonic() + TIMEOUT
                    while robot.tick_task is None:
                        assert time.monotonic() < deadline, "the sim tick loop never ticked the robot"
                        await asyncio.sleep(0.05)
                    held = robot.tick_task
                    result = await asyncio.wait_for(env._run_episode(), TIMEOUT)
                    assert result.state == RunEpisode.Result.SUCCESS, result.info
                    await asyncio.sleep(0.2)
                    assert held.cancelled()
                    assert all(started == ended == 1 for started, ended in robot.done), robot.done
                    assert not any("held" in reason for reason in env._episodes.current.integrity)
                finally:
                    await robot.cancel_tick()
                    env._robots_manager.managers.pop(robot.name, None)
                    await asyncio.wait_for(env.teardown(), TIMEOUT)
            finally:
                rig.executor.remove_node(env)
                env.destroy_node()

    asyncio.run(scenario())
