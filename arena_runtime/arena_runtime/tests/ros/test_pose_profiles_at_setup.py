"""The env reads the gait library and resolves the builtin agent types' pose profiles during its setup, before any
episode starts: lazily, the first crowd of each episode did it and held the GIL for seconds after EPISODE STARTED."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.msg")

TIMEOUT = 60.0


def test_the_library_and_the_builtin_profiles_are_loaded_by_the_end_of_setup(rclpy_context: object, fake_task_layer: object, monkeypatch: pytest.MonkeyPatch) -> None:
    import rclpy.parameter

    from arena_runtime.env.episode_node import EpisodeNode
    from arena_runtime.humans import profile
    from arena_runtime.testing.env_host import arena_on_dummy

    monkeypatch.setattr(profile, "_LIBRARIES", {})
    del fake_task_layer

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
                assert env._episodes.current.episode_id == 0  # no episode has started
                assert Path(profile.LIBRARY_DIR) in profile._LIBRARIES
                builtin = env._human_simulator._builtin_agent_types()
                if builtin is not None and builtin.is_dir():
                    assert {path.stem for path in builtin.glob("*.yaml")} <= set(env._human_simulator._pose_profiles)
                await asyncio.wait_for(env.teardown(), TIMEOUT)
            finally:
                rig.executor.remove_node(env)
                env.destroy_node()

    asyncio.run(scenario())
