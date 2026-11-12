"""A scenario's teleport zones on the human simulator: realized into the env frame, sent with the elevators, dropped with the world."""

from __future__ import annotations

import asyncio
import math
import types
import typing

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.World.Scenario import TeleportZone

ORIGIN = (30.0, -12.0)


def _zone() -> TeleportZone:
    from arena_simulation_setup.tree.World.Scenario import TeleportZone

    return TeleportZone("up", ("down",), (1.0, 2.0, 0.0), (0.2, 2.0, 0.0), (2.5, 2.0, math.pi), ((1.0, 1.5), (3.0, 1.5), (3.0, 2.5)), True, 14.05, "escalator", 0.5)


def test_scenario_teleport_zones_are_realized_and_sent_with_the_elevators():
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.humans.utils import ObstacleLayer
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from arena_runtime.testing.env_host import env_host

    async def main() -> None:
        node = env_host()
        try:
            realizer = Realizer(Realizer._Configuration(x=ORIGIN[0], y=ORIGIN[1], prefix="env_1"))
            sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
            acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
            human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
            sent: list[tuple] = []

            async def record(elevators: typing.Sequence[object]) -> bool:
                sent.append((tuple(elevators), human._scenario_zones))
                return True

            human._set_elevators_impl = record
            env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
            zone = _zone()
            assert await env.setup_teleport_zones([zone])
            (realized,) = human._scenario_zones
            config = realizer.get_config()
            assert (realized.entry[0] - config.x, realized.entry[1] - config.y, realized.entry[2]) == zone.entry
            assert [(x - config.x, y - config.y) for x, y in realized.footprint] == list(zone.footprint)
            assert realized.hold[2] == zone.hold[2] and realized.destinations == zone.destinations
            assert (realized.delay, realized.belt, realized.kind) == (14.05, 0.5, "escalator")
            assert len(sent) == 1 and sent[0][1] == (realized,)
            assert await env.setup_teleport_zones([])
            assert human._scenario_zones == () and len(sent) == 2
            assert await env.setup_teleport_zones([])
            assert len(sent) == 2  # nothing to clear
            # a zone named like one of the world's: refused before anything is sent
            human._known_stairs = (types.SimpleNamespace(name="up"),)
            assert not await env.setup_teleport_zones([zone])
            assert human._scenario_zones == () and len(sent) == 2
            human._known_stairs = ()
            # the world's purge drops the scenario's zones with it
            assert await env.setup_teleport_zones([zone])
            assert len(human._scenario_zones) == 1 and len(sent) == 3
            await human.remove_obstacles(purge=ObstacleLayer.WORLD)
            assert human._scenario_zones == () and sent[-1] == ((), ())
        finally:
            node._heartbeat_stop.set()
            node.destroy_node()

    asyncio.run(main())


def test_arena_humansim_sends_a_scenario_zone_in_the_engine_frame():
    from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator

    stub = types.SimpleNamespace(_to_engine=lambda x, y: (x - ORIGIN[0], y - ORIGIN[1]))
    zone = _zone()
    realized = type(zone)(
        zone.name,
        zone.destinations,
        (zone.entry[0] + ORIGIN[0], zone.entry[1] + ORIGIN[1], zone.entry[2]),
        (zone.wait[0] + ORIGIN[0], zone.wait[1] + ORIGIN[1], zone.wait[2]),
        (zone.hold[0] + ORIGIN[0], zone.hold[1] + ORIGIN[1], zone.hold[2]),
        tuple((x + ORIGIN[0], y + ORIGIN[1]) for x, y in zone.footprint),
        True,
        zone.delay,
        zone.kind,
        zone.belt,
    )
    msg = ArenaHumanSimulator._scenario_zone_msg(stub, realized)
    # each half's flight time and belt reach HumanSim as the site config gives them
    assert (msg.name, list(msg.destinations), msg.kind, msg.open, msg.delay, msg.belt) == ("up", ["down"], "escalator", True, 14.05, 0.5)
    assert (msg.entry.x, msg.entry.y, msg.entry.theta) == (1.0, 2.0, 0.0)
    assert (msg.hold.x, msg.hold.y) == (2.5, 2.0) and math.isclose(msg.hold.theta, math.pi)
    assert [(p.x, p.y) for p in msg.footprint] == list(zone.footprint)
