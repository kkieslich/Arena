"""Flatsim pedestrian discs take each pedestrian's own radius and keep it through updates that carry none."""

from __future__ import annotations

import asyncio
import typing

import pytest


async def _until(pred: typing.Callable[[], bool], what: str, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting for {what}")
        await asyncio.sleep(0.02)


def _peds(*peds: tuple[str, float, float]) -> typing.Any:
    from arena_people_msgs.msg import Pedestrian, Pedestrians

    msg = Pedestrians()
    for name, x, radius in peds:
        ped = Pedestrian(name=name, radius=radius)
        ped.pose.position.x = x
        ped.pose.orientation.w = 1.0
        msg.pedestrians.append(ped)
    return msg


def test_a_disc_takes_the_pedestrians_radius_ped_radius_without_one_and_keeps_its_size(on_flatsim):
    from arena_runtime.humans import PED_RADIUS

    async def body(sim, world):
        wheelchair, unknown = sim._name("wheelchair"), sim._name("unknown")

        def disc(name: str) -> tuple[float, float] | None:
            entity = world._entities.get(name)
            return None if entity is None else (entity.radius, entity.pose[0])

        assert list(await sim.pedestrian_update(_peds(("wheelchair", 1.0, 0.325), ("unknown", 3.0, 0.0)))) == [True, True]
        await _until(lambda: disc(wheelchair) is not None and disc(unknown) is not None, "both discs")
        assert disc(wheelchair) == pytest.approx((0.325, 1.0)) and disc(unknown) == pytest.approx((PED_RADIUS, 3.0))
        # a teleport or a possessed ped comes without a radius
        await sim.pedestrian_update(_peds(("wheelchair", 2.0, 0.0)))
        await _until(lambda: disc(wheelchair) == pytest.approx((0.325, 2.0)), "the moved disc at its own size")
        await sim.pedestrian_update(_peds(("unknown", 3.0, 0.18)))
        await _until(lambda: disc(unknown) == pytest.approx((0.18, 3.0)), "the disc resized once its radius is known")

    on_flatsim(body)


def test_removing_the_world_deletes_every_disc_and_a_later_pedestrian_of_the_same_name_gets_a_fresh_one(on_flatsim):
    async def body(sim, world):
        standing, walking = sim._name("flow_1220"), sim._name("flow_7")
        await sim.pedestrian_update(_peds(("flow_1220", 1.0, 0.33), ("flow_7", 3.0, 0.25)))
        await _until(lambda: standing in world._entities and walking in world._entities, "both discs")
        assert await sim.remove_world()
        # a disc no one deletes would stay where its pedestrian was, seen by the robots' sensors
        await _until(lambda: standing not in world._entities and walking not in world._entities, "no disc after the world went")
        await sim.pedestrian_update(_peds(("flow_7", 5.0, 0.22)))
        await _until(lambda: walking in world._entities, "the new pedestrian's disc")
        assert (world._entities[walking].radius, world._entities[walking].pose[0]) == pytest.approx((0.22, 5.0))
        assert standing not in world._entities

    on_flatsim(body)
