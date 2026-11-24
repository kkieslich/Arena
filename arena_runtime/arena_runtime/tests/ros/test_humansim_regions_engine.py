"""ArenaHumanSimulator flow regions against a live arena_humansim engine: a scenario's 29 regions all land, though the engine's service queues hold 10."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("tf2_ros")
    pytest.importorskip("arena_humansim_msgs.msg")
    pytest.importorskip("arena_humansim.core.agent_manager")


def _with_engine(scenario: Callable[..., Awaitable[None]]) -> None:
    import tf2_ros
    from arena_humansim.core.agent_manager import AgentManager
    from arena_humansim.params import MODE_SUBSYSTEM
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter

    from arena_runtime.acoustics.noop import NoopAcousticsSimulator
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator as Human

    class _Node(ServiceNamespace, AsyncNode):
        pass

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> None:
        node = _Node(f"regions_{suffix}", namespace=str(ns))
        executor = SingleThreadedExecutor()
        engine = None
        try:
            node.tf_buffer = tf2_ros.Buffer()
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=""))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            acoustics = NoopAcousticsSimulator(node=node, namespace=ns)
            human = Human(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            executor.add_node(node)
            engine = AgentManager(
                namespace=node.get_fully_qualified_name(),
                parameter_overrides=[Parameter("mode", value=MODE_SUBSYSTEM), Parameter("publish_markers", value=0)],
            )
            executor.add_node(engine)
            clients = (human._add_source_client, human._add_sink_client, human._remove_source_client, human._remove_sink_client)
            for client in clients:
                client._timeout = 2.0
                for _ in range(500):
                    if client.client.service_is_ready():
                        break
                    executor.spin_once(timeout_sec=0.02)

            async def drive(coro: Awaitable[object]) -> object:
                future = asyncio.ensure_future(coro)
                for _ in range(2000):
                    if future.done():
                        return future.result()
                    executor.spin_once(timeout_sec=0.01)
                    await asyncio.sleep(0)
                raise AssertionError("call never completed")

            await scenario(human, engine, drive)
        finally:
            executor.shutdown()
            if engine is not None:
                engine.destroy_node()
            node.destroy_node()

    asyncio.run(main())


def _regions(sources: int, sinks: int, prefix: str = "") -> list:
    from arena_runtime.shared import Position, Region

    square = [Position(0.0, 0.0), Position(2.0, 0.0), Position(2.0, 2.0), Position(0.0, 2.0)]
    return [Region(name=f"{prefix}src_{i:02d}", type="source", polygon=square, config={"agent": {"agent_type": "adult"}, "rate_profile": [{"t": 0.0, "rate": 0.1}]}) for i in range(sources)] + [Region(name=f"{prefix}sink_{i}", type="sink", polygon=square) for i in range(sinks)]


def test_all_29_regions_of_a_scenario_land_in_the_engine_in_order(rclpy_context):
    async def scenario(human, engine, drive) -> None:
        assert await drive(human.setup_regions(_regions(24, 5))) is True
        assert list(engine._spawn_scheduler._sources) == [f"src_{i:02d}" for i in range(24)]
        assert list(engine._despawn_monitor.sinks) == [f"sink_{i}" for i in range(5)]

    _with_engine(scenario)


def test_a_next_stage_finds_only_its_own_regions_in_the_engine(rclpy_context):
    async def scenario(human, engine, drive) -> None:
        await drive(human.setup_regions(_regions(24, 5, prefix="a_")))
        assert await drive(human.remove_all_regions()) is True
        assert dict(engine._spawn_scheduler._sources) == {}
        assert engine._despawn_monitor.sinks == {}
        assert await drive(human.setup_regions(_regions(3, 2, prefix="b_"))) is True
        assert list(engine._spawn_scheduler._sources) == ["b_src_00", "b_src_01", "b_src_02"]
        assert list(engine._despawn_monitor.sinks) == ["b_sink_0", "b_sink_1"]
        assert list(engine._spawn_scheduler._sinks) == ["b_sink_0", "b_sink_1"]

    _with_engine(scenario)
