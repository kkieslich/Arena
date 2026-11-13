"""A reset that removes every agent from the engine deletes the simulator's bodies of the flow agents too: a body left
in place would stand where its agent was, seen by the robots' sensors and by no crowd measure."""

from __future__ import annotations

import asyncio
import typing

import pytest

pytest.importorskip("rclpy")
srv = pytest.importorskip("arena_humansim_msgs.srv")

from arena_humansim_msgs.srv import RemoveAgents
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
    adapter._flow_agent_ids = set(flow)
    adapter._bridge_agent_ids = set(bridge)
    adapter._agent_names = {aid: f"flow_{aid}" for aid in flow} | {aid: f"scenario_ped_{aid}" for aid in bridge}
    for name in ("_deferred_spawns",):
        setattr(adapter, name, [])
    for name in ("_itineraries", "_semantic_aliases", "_flow_meta_wait", "_tagged_models", "_teleport_stamps", "_ped_model_uris"):
        setattr(adapter, name, {})
    return adapter, simulator


def test_a_reset_deletes_the_bodies_of_the_flow_agents_it_removed() -> None:
    adapter, simulator = _adapter(flow={1220, 7, 31}, bridge={100})
    assert asyncio.run(adapter._remove_pedestrians_impl())
    realizer = Realizer()
    # the scenario's own pedestrians stay with the obstacle lifecycle, which reuses or purges their bodies by name
    assert simulator.deleted == [[realizer.prefix(f"flow_{aid}") for aid in (7, 31, 1220)]]
    assert adapter._flow_agent_ids == set() and adapter._agent_names == {}
    # an update pass that a reset overtook drops the old episode's roster
    assert adapter._removals == 1


def test_a_reset_without_flow_agents_deletes_nothing() -> None:
    adapter, simulator = _adapter(flow=set(), bridge={100})
    assert asyncio.run(adapter._remove_pedestrians_impl())
    assert simulator.deleted == []


def test_the_bodies_go_even_when_the_engine_reports_a_failed_removal() -> None:
    adapter, simulator = _adapter(flow={4}, bridge=set(), removed=False)
    assert not asyncio.run(adapter._remove_pedestrians_impl())
    assert simulator.deleted == [[Realizer().prefix("flow_4")]]
