"""Humansim flow regions: added one at a time in scenario order, a timed-out call sent again in place, cleared with the engine's clear-all, and a region the engine does not take fails the episode."""

from __future__ import annotations

import asyncio
import types
import typing

import pytest
import shapely

pytest.importorskip("rclpy")

from arena_runtime.env.realizer import Realizer
from arena_runtime.humans import RegionSetupError
from arena_runtime.humans.arena_humansim.arena_humansim import REGION_CALL_ATTEMPTS, ArenaHumanSimulator
from arena_runtime.humans.noop import NoopHumanSimulator
from arena_runtime.shared import Position, Region

SQUARE = [Position(0.0, 0.0), Position(2.0, 0.0), Position(2.0, 2.0), Position(0.0, 2.0)]
QUEUE_DEPTH = 10  # the rmw default for a service


class _Client:
    """A service answering after a tick that drops a request arriving while QUEUE_DEPTH others wait (as a full rmw queue does), and the first `lose[name]` sends of a name."""

    def __init__(self, lose: dict[str, int] | None = None, refuse: frozenset[str] = frozenset()) -> None:
        self.names: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.lose = dict(lose or {})
        self.refuse = refuse

    async def call_timeout(self, request: typing.Any) -> typing.Any:
        name = request.source.name if hasattr(request, "source") else request.sink.name if hasattr(request, "sink") else request.name
        self.names.append(name)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        dropped = self.in_flight > QUEUE_DEPTH
        try:
            await asyncio.sleep(0.001)
            if dropped:
                return None
            if self.lose.get(name, 0) > 0:
                self.lose[name] -= 1
                return None
            return types.SimpleNamespace(success=name not in self.refuse, message="refused" if name in self.refuse else "", name=name)
        finally:
            self.in_flight -= 1


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def get_child(self, _name: str) -> _Logger:
        return self

    def __getattr__(self, level: str) -> typing.Callable[..., None]:
        return lambda msg, *a, **k: self.lines.append(f"{level}: {msg}")


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger


def _adapter(**clients: _Client) -> ArenaHumanSimulator:
    sim = ArenaHumanSimulator.__new__(ArenaHumanSimulator)
    sim._NodeInterface__node = _Node()  # pyright: ignore[reportAttributeAccessIssue]
    sim._realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_0"), {})
    sim._agent_type_paths = {}
    sim._known_regions = {}
    for name in ("add_source", "add_sink", "remove_source", "remove_sink"):
        setattr(sim, f"_{name}_client", clients.get(name, _Client()))
    return sim


def _twin() -> list[Region]:
    """The INFRA twin's 29 regions: 24 sources and 5 sinks, interleaved as a scenario lists them."""
    sources = [Region(name=f"src_{i:02d}", type="source", polygon=SQUARE, config={"agent": {"agent_type": "adult"}}) for i in range(24)]
    sinks = [Region(name=f"sink_{i}", type="sink", polygon=SQUARE) for i in range(5)]
    return [*sources[:3], sinks[0], *sources[3:6], sinks[1], *sources[6:9], *sinks[2:], *sources[9:]]


def _sources(regions: typing.Sequence[Region]) -> list[str]:
    return [r.name for r in regions if r.type == "source"]


def _lines(sim: ArenaHumanSimulator, level: str) -> list[str]:
    return [line for line in sim.node.logger.lines if line.startswith(f"{level}: ")]


def test_the_regions_are_sent_one_at_a_time_in_scenario_order() -> None:
    sources, sinks = _Client(), _Client()
    sim = _adapter(add_source=sources, add_sink=sinks)
    assert asyncio.run(sim.setup_regions(_twin())) is True
    assert sources.names == _sources(_twin())
    assert sinks.names == [f"sink_{i}" for i in range(5)]
    assert sources.max_in_flight == sinks.max_in_flight == 1
    assert sim.node.logger.lines == []


def test_a_timed_out_region_is_sent_again_in_place_before_the_next() -> None:
    sources = _Client(lose={"src_03": REGION_CALL_ATTEMPTS - 1})
    sim = _adapter(add_source=sources)
    assert asyncio.run(sim.setup_regions(_twin())) is True
    order = _sources(_twin())
    at = order.index("src_03")
    assert sources.names == [*order[:at], *["src_03"] * REGION_CALL_ATTEMPTS, *order[at + 1 :]]
    assert len(_lines(sim, "warning")) == REGION_CALL_ATTEMPTS - 1
    assert _lines(sim, "error") == []


def test_a_region_the_engine_never_answers_fails_the_setup_and_nothing_after_it_is_sent() -> None:
    sources = _Client(lose={"src_04": REGION_CALL_ATTEMPTS})
    sim = _adapter(add_source=sources)
    with pytest.raises(RegionSetupError, match=r"did not add region 'src_04' \(5 of 29 added\)"):
        asyncio.run(sim.setup_regions(_twin()))
    assert sources.names[-REGION_CALL_ATTEMPTS:] == ["src_04"] * REGION_CALL_ATTEMPTS
    assert "src_05" not in sources.names
    assert _lines(sim, "error") == [f"error: AddSource 'src_04' timed out {REGION_CALL_ATTEMPTS} times"]


def test_a_refused_region_fails_the_setup_without_being_sent_again() -> None:
    sinks = _Client(refuse=frozenset({"sink_1"}))
    sim = _adapter(add_sink=sinks)
    with pytest.raises(RegionSetupError, match="'sink_1'"):
        asyncio.run(sim.setup_regions(_twin()))
    assert sinks.names == ["sink_0", "sink_1"]
    assert _lines(sim, "error") == ["error: AddSink 'sink_1' failed: refused"]


def test_the_regions_leave_with_one_clear_all_per_service_sent_again_on_a_timeout() -> None:
    sources, sinks = _Client(lose={"": 1}), _Client()
    sim = _adapter(remove_source=sources, remove_sink=sinks)
    sim._known_regions = {region.name: region for region in _twin()}
    assert asyncio.run(sim.remove_all_regions()) is True
    assert sources.names == ["", ""]
    assert sinks.names == [""]
    assert sim._known_regions == {}


def test_a_clear_the_engine_never_confirms_fails_the_reset() -> None:
    sinks = _Client(lose={"": REGION_CALL_ATTEMPTS})
    sim = _adapter(remove_sink=sinks)
    sim._known_regions = {region.name: region for region in _twin()}
    with pytest.raises(RegionSetupError, match="did not clear its sinks"):
        asyncio.run(sim.remove_all_regions())
    assert sinks.names == [""] * REGION_CALL_ATTEMPTS


def test_no_regions_send_no_clear() -> None:
    sources, sinks = _Client(), _Client()
    sim = _adapter(remove_source=sources, remove_sink=sinks)
    assert asyncio.run(sim.remove_all_regions()) is True
    assert sources.names == sinks.names == []


class _Refusing(NoopHumanSimulator):
    async def _add_regions_impl(self, regions: typing.Sequence[Region]) -> bool:
        return False

    async def _remove_regions_impl(self, regions: typing.Sequence[Region]) -> bool:
        return False


def test_a_backend_that_reports_a_region_missing_fails_the_setup_and_the_removal() -> None:
    sim = _Refusing.__new__(_Refusing)
    sim._known_regions = {}
    with pytest.raises(RegionSetupError, match="did not add every one of the 29 regions"):
        asyncio.run(sim.setup_regions(_twin()))
    with pytest.raises(RegionSetupError, match="did not remove every one of the 29 regions"):
        asyncio.run(sim.remove_all_regions())


class _Recording(_Client):
    """A _Client that keeps every request it was sent."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[typing.Any] = []

    async def call_timeout(self, request: typing.Any) -> typing.Any:
        self.requests.append(request)
        return await super().call_timeout(request)


# The INFRA twin's Bras Basah 1 entrance: a 4 m square cut to the floor along an alcove wall. Most vertices lie on the cut
# edge, so their mean (155.65, 83.50) falls just outside the polygon.
CUT = [Position(x, y) for x, y in [(153.31, 81.48), (153.31, 85.48), (156.68, 85.48), (156.2, 85.11), (156.21, 84.49), (155.76, 84.43), (155.46, 83.99), (155.54, 83.6), (155.71, 83.4), (155.73, 83.17), (155.93, 82.83), (156.07, 82.72), (156.41, 82.64), (156.41, 82.13), (156.04, 81.51)]]


def _sent(request: typing.Any) -> tuple[shapely.Point, shapely.Polygon]:
    """The pose and the absolute polygon a source or sink request carries."""
    config = request.source if hasattr(request, "source") else request.sink
    pose = shapely.Point(config.pose.x, config.pose.y)
    return pose, shapely.Polygon([(config.pose.x + v.x, config.pose.y + v.y) for v in config.shape.vertices])


def test_a_concave_region_is_posed_inside_its_polygon_so_its_sink_can_remove_walkers() -> None:
    area = shapely.Polygon([(p.x, p.y) for p in CUT])
    mean = shapely.Point(sum(p.x for p in CUT) / len(CUT), sum(p.y for p in CUT) / len(CUT))
    assert not area.contains(mean)
    sources, sinks = _Recording(), _Recording()
    sim = _adapter(add_source=sources, add_sink=sinks)
    regions = [Region(name="bb1_src", type="source", polygon=CUT, config={"agent": {"agent_type": "adult"}}), Region(name="bb1_sink", type="sink", polygon=CUT)]
    assert asyncio.run(sim.setup_regions(regions)) is True
    for request in (*sources.requests, *sinks.requests):
        pose, sent = _sent(request)
        assert area.contains(pose)
        assert sent.symmetric_difference(area).area < 1e-9


def test_a_region_whose_vertex_mean_lies_inside_keeps_it() -> None:
    sinks = _Recording()
    sim = _adapter(add_sink=sinks)
    assert asyncio.run(sim.setup_regions([Region(name="square", type="sink", polygon=SQUARE)])) is True
    pose, _ = _sent(sinks.requests[0])
    assert (pose.x, pose.y) == (1.0, 1.0)
