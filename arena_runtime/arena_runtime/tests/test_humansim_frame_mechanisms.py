"""Under the input barrier the arena backend runs the world mechanisms once per engine frame, at the frame's time, on the
frame's walkers, and waits until what they changed is with the engine before it acknowledges the frame."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim_msgs.msg")

from arena_humansim_msgs.msg import AgentFrame
from arena_runtime.env.realizer import Realizer
from arena_runtime.humans import PED_RADIUS
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.humans.arena_humansim.engine_inputs import EngineInputs


def _adapter() -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._realizer = Realizer(Realizer._Configuration(x=10.0, y=-5.0, prefix="env_0"))
    adapter._agent_names = {1: "env_0/a", 2: "env_0/b"}
    adapter._engine_dt_ns = 50_000_000
    adapter._input_barrier = True
    adapter._mechanism_step = None
    adapter._mechanism_last_ns = None
    adapter._episode_epoch_ns = None
    adapter._mech_walls_syncing = False
    adapter._mech_walls_desired = {}
    adapter._semantic_draining = False
    adapter._semantic_backlog = __import__("collections").deque()
    adapter._engine_inputs = EngineInputs(lambda: 0)
    return adapter


def _frame(stamp_ns: int) -> AgentFrame:
    msg = AgentFrame()
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    msg.agent_id = [1, 2, 3]
    msg.x = [1.0, 2.0, 3.0]
    msg.y = [0.5, 0.5, 0.5]
    msg.radius = [0.3, 0.0, 0.3]
    return msg


def test_the_mechanisms_tick_on_the_frame_with_its_walkers() -> None:
    adapter = _adapter()
    calls: list[tuple[float, float, list]] = []

    async def step(now: float, dt: float, discs: list) -> None:
        calls.append((now, dt, discs))

    assert adapter.drives_mechanisms()
    adapter.set_mechanism_step(step)
    asyncio.run(adapter._step_mechanisms(12_350_000_000, _frame(12_350_000_000)))
    asyncio.run(adapter._step_mechanisms(12_400_000_000, _frame(12_400_000_000)))
    (now0, dt0, discs0), (now1, dt1, _) = calls
    assert (now0, now1) == (12.35, 12.4)
    assert dt0 == pytest.approx(0.05) and dt1 == pytest.approx(0.05)
    # agent 3's body is not spawned yet: it goes by the name the body will get; map frame (env offset); radius 0 = PED_RADIUS
    assert [name for name, _, _ in discs0] == ["env_0/a", "env_0/b", "env_0/flow_3"]
    assert discs0[0][1] == pytest.approx((11.0, -4.5))
    assert discs0[0][2] == pytest.approx(0.3) and discs0[1][2] == PED_RADIUS


def test_the_mechanisms_count_whole_ticks_from_the_episode_start() -> None:
    """The same door on two epochs: its clock and steps are the same floats, so it opens and closes on the same tick."""
    seen: list[list[tuple[float, float]]] = []
    for epoch_ns in (21_611_700_000, 20_546_100_000):
        adapter = _adapter()
        adapter._episode_epoch_ns = epoch_ns
        calls: list[tuple[float, float]] = []

        async def step(now: float, dt: float, discs: list, calls: list = calls) -> None:
            calls.append((now, dt))

        adapter.set_mechanism_step(step)
        for k in range(30):
            stamp = epoch_ns + k * 50_000_000
            asyncio.run(adapter._step_mechanisms(stamp, _frame(stamp)))
        seen.append(calls)
    assert seen[0] == seen[1]
    assert seen[0][20] == (1.0, 0.05)


def test_a_frame_waits_for_the_walls_its_mechanism_tick_set() -> None:
    adapter = _adapter()
    sent: list[str] = []

    async def sync() -> None:
        await asyncio.sleep(0.01)
        sent.append("walls")
        adapter._mech_walls_syncing = False

    async def step(now: float, dt: float, discs: list) -> None:
        adapter._mech_walls_syncing = True
        asyncio.get_running_loop().create_task(sync())

    adapter.set_mechanism_step(step)
    asyncio.run(adapter._step_mechanisms(1_000_000_000, _frame(1_000_000_000)))
    assert sent == ["walls"]


def test_without_a_step_nothing_runs() -> None:
    adapter = _adapter()
    asyncio.run(adapter._step_mechanisms(1_000_000_000, _frame(1_000_000_000)))
    adapter.set_mechanism_step(None)
    assert adapter._mechanism_step is None
