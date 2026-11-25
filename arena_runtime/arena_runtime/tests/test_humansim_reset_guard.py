"""A reset that removed every agent guards the next episode: an update pass it overtook spawns and relays nothing, an
engine frame sent before it is dropped, and a failed body delete is logged without failing the reset."""

from __future__ import annotations

import asyncio
import typing

import pytest

pytest.importorskip("rclpy")
srv = pytest.importorskip("arena_humansim_msgs.srv")

from arena_humansim_msgs.srv import RemoveAgents
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_runtime.env.realizer import Realizer
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator


class _Client:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[object] = []

    async def call_timeout(self, request: object) -> object:
        self.requests.append(request)
        return self.response


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, line: str) -> None:
        self.lines.append(line)

    error = warning = debug = info

    def get_child(self, _name: str) -> _Logger:
        return self


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()
        from arena_rclpy_mixins.Time import Time

        self.sim_time = Time(sec=500, nanosec=250_000_000)

    def get_logger(self) -> _Logger:
        return self.logger


class _Simulator:
    def __init__(self) -> None:
        self.deleted: list[list[str]] = []

    async def pedestrian_delete(self, pedestrians: typing.Sequence[typing.Any]) -> typing.Sequence[bool]:
        self.deleted.append([pedestrian.sim_path for pedestrian in pedestrians])
        return [True] * len(pedestrians)


def _adapter(flow: set[int], bridge: set[int], *, removed: bool = True) -> tuple[ArenaHumanSimulator, _Simulator]:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._NodeInterface__node = _Node()  # pyright: ignore[reportAttributeAccessIssue]
    simulator = _Simulator()
    adapter._simulator = simulator  # pyright: ignore[reportAttributeAccessIssue]
    adapter._realizer = Realizer()  # pyright: ignore[reportAttributeAccessIssue]
    adapter._remove_client = _Client(RemoveAgents.Response(success=removed, message="removed"))  # pyright: ignore[reportAttributeAccessIssue]

    async def _no_op() -> None:
        return None

    adapter._seed_engine = _no_op  # pyright: ignore[reportAttributeAccessIssue]
    adapter._publish_interpolated = _no_op  # pyright: ignore[reportAttributeAccessIssue]
    adapter._apply_interactions = lambda held, holds: None  # pyright: ignore[reportAttributeAccessIssue]
    adapter._removals = 0  # pyright: ignore[reportAttributeAccessIssue]
    adapter._removed_at_ns = None  # pyright: ignore[reportAttributeAccessIssue]
    adapter._flow_agent_ids = set(flow)
    adapter._bridge_agent_ids = set(bridge)
    adapter._agent_names = {aid: f"flow_{aid}" for aid in flow} | {aid: f"scenario_ped_{aid}" for aid in bridge}
    adapter._input_barrier = False  # pyright: ignore[reportAttributeAccessIssue]
    for name in ("_deferred_spawns", "_deferred_unanchored"):
        setattr(adapter, name, [])
    for name in ("_itineraries", "_semantic_aliases", "_flow_meta_wait", "_tagged_models", "_teleport_stamps", "_ped_model_uris"):
        setattr(adapter, name, {})
    return adapter, simulator


def _ped(agent_id: int, name: str) -> Pedestrian:
    return Pedestrian(id=agent_id, name=name)


class _Rate:
    def __init__(self, done: asyncio.Event) -> None:
        self.done = done

    async def get(self) -> None:
        self.done.set()  # one pass of the loop


class _TickingNode(_Node):
    def sim_time_rate(self, _hz: float) -> typing.Any:
        import contextlib

        done = asyncio.Event()
        return contextlib.nullcontext((done, _Rate(done)))


def _loop_adapter(roster: list[tuple[int, str]], agent_ids: list[int]) -> tuple[ArenaHumanSimulator, list[list[str]], list[list[str]]]:
    """An adapter ready for one pass of its update loop: the engine reports `agent_ids`, the roster `roster`; the lists
    record what the pass spawns and relays."""
    from arena_humansim_msgs.msg import AgentFrame

    adapter, simulator = _adapter(flow=set(), bridge=set())
    adapter._NodeInterface__node = _TickingNode()  # pyright: ignore[reportAttributeAccessIssue]
    adapter._agents_lock = asyncio.Lock()  # pyright: ignore[reportAttributeAccessIssue]
    adapter._spawns_in_flight = 0  # pyright: ignore[reportAttributeAccessIssue]
    adapter._agent_types = {aid: "adult" for aid in agent_ids}  # pyright: ignore[reportAttributeAccessIssue]
    adapter._agent_names = {}
    adapter._from_engine = lambda x, y: (x, y)  # pyright: ignore[reportAttributeAccessIssue]
    msg = Pedestrians()
    msg.pedestrians = [_ped(aid, name) for aid, name in roster]
    adapter._arena_pedestrians = msg  # pyright: ignore[reportAttributeAccessIssue]
    adapter._curr_agent_states = AgentFrame(agent_id=agent_ids, x=[1.0] * len(agent_ids), y=[2.0] * len(agent_ids), desired_velocity=[1.2] * len(agent_ids))  # pyright: ignore[reportAttributeAccessIssue]
    spawned: list[list[str]] = []
    relayed: list[list[str]] = []

    async def _model(_aid: int) -> str:
        return ArenaHumanSimulator.FLOW_MODEL

    async def _spawn(pedestrians: typing.Sequence[typing.Any]) -> typing.Sequence[bool]:
        spawned.append([pedestrian.sim_path for pedestrian in pedestrians])
        return [True] * len(pedestrians)

    async def _relay(msg: Pedestrians) -> typing.Sequence[bool]:
        relayed.append([ped.name for ped in msg.pedestrians])
        return [True] * len(msg.pedestrians)

    adapter._flow_model = _model  # pyright: ignore[reportAttributeAccessIssue]
    simulator.pedestrian_spawn = _spawn  # pyright: ignore[reportAttributeAccessIssue]
    adapter.relay_pedestrian_update = _relay  # pyright: ignore[reportAttributeAccessIssue]
    return adapter, spawned, relayed


def test_an_update_pass_a_reset_overtakes_spawns_and_relays_nothing_of_the_old_episode() -> None:
    body = Realizer().prefix("flow_8")
    adapter, spawned, relayed = _loop_adapter([(8, body)], [8])

    async def _spawnable_across_a_reset(obstacles: typing.Sequence[typing.Any]) -> typing.Sequence[typing.Any]:
        # the reset runs while the pass waits here
        adapter._removals += 1
        return list(obstacles)

    adapter._ensure_spawnable = _spawnable_across_a_reset  # pyright: ignore[reportAttributeAccessIssue]
    asyncio.run(adapter._pedestrian_update_loop())
    assert spawned == [] and relayed == []
    assert adapter._flow_agent_ids == set() and adapter._agent_names == {}


def test_an_engine_frame_sent_before_the_reset_is_dropped_and_a_later_one_taken() -> None:
    from arena_humansim_msgs.msg import AgentFrame

    adapter, _ = _adapter(flow={3}, bridge=set())
    adapter._curr_agent_states = None  # pyright: ignore[reportAttributeAccessIssue]
    adapter._publish_pending = True  # pyright: ignore[reportAttributeAccessIssue]
    asyncio.run(adapter._remove_pedestrians_impl())
    assert adapter._removed_at_ns == 500_250_000_000

    def frame(sec: int, nanosec: int) -> AgentFrame:
        msg = AgentFrame(agent_id=[3], x=[1.0], y=[2.0])
        msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nanosec
        return msg

    adapter._agent_states_callback(frame(500, 200_000_000))
    adapter._agent_states_callback(frame(500, 250_000_000))
    assert adapter._curr_agent_states is None
    adapter._agent_states_callback(frame(500, 300_000_000))
    assert adapter._curr_agent_states is not None and list(adapter._curr_agent_states.agent_id) == [3]


def test_a_failed_body_delete_in_the_reset_is_logged_as_such_and_the_reset_still_publishes() -> None:
    adapter, simulator = _adapter(flow={4}, bridge=set())
    published: list[bool] = []

    async def _fail(pedestrians: typing.Sequence[typing.Any]) -> typing.Sequence[bool]:
        raise RuntimeError("service down")

    async def _publish() -> None:
        published.append(True)

    simulator.pedestrian_delete = _fail  # pyright: ignore[reportAttributeAccessIssue]
    adapter._publish_interpolated = _publish  # pyright: ignore[reportAttributeAccessIssue]
    assert asyncio.run(adapter._remove_pedestrians_impl())
    assert published == [True]
    lines = adapter._NodeInterface__node.logger.lines  # pyright: ignore[reportAttributeAccessIssue]
    assert any("deleting the simulator bodies of 1 removed walkers failed" in line for line in lines)
    assert not any("RemoveAgents call failed" in line for line in lines)
