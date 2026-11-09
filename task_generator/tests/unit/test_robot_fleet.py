from __future__ import annotations

import asyncio
import itertools
import typing
from collections.abc import Callable
from types import GenericAlias, SimpleNamespace

import attrs
import pytest
from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.shared import WORLD_ENTITY_PREFIX, Orientation, Pose, Position

if typing.TYPE_CHECKING:
    from arena_runtime.env.task_api import GoToPhase, TaskRequest
    from arena_simulation_setup.tree.World import Level

    from task_generator.tasks.robots.fleet.impl import TM_Fleet

HOMES = ["bay_1", "bay_2", "charger_1", "charger_2"]
PARAMS = {"homes": HOMES, "loading": "dock_a", "locker": "locker_1", "rate": 60.0}


class _Robot:
    def __init__(self, name: str) -> None:
        self.name = name
        self.safe_distance = 0.3
        self.requests: list[TaskRequest] = []
        self.done = False

    def namespace(self, *path: str) -> str:
        return "/".join((self.name, *path))

    def resolve_target(self, phase: GoToPhase) -> GoToPhase:
        return attrs.evolve(phase, pose=Pose(position=Position(float(HOMES.index(phase.target)), 0.0), orientation=Orientation.identity()))

    async def submit_task(self, request: TaskRequest) -> None:
        self.requests.append(request)
        self.done = False

    @property
    async def is_done(self) -> bool:
        return self.done

    def last(self) -> str:
        return self.requests[-1].phases[0].text

    def job(self) -> int:
        return int(self.last().split(":")[0].split()[-1])


class _Param:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = values

    def __getitem__(self, kind: type | GenericAlias) -> _Param:
        return self

    def __call__(self, name: str, default: object = None, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(value=self.values.get(str(name).rsplit(".", 1)[-1], default))


def _world() -> Level:
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.tree.World import Level, Zone

    def square(x: float) -> list[Position]:
        return [Position(x, 0.0), Position(x + 1.0, 0.0), Position(x + 1.0, 1.0), Position(x, 1.0)]

    def docked(name: str, x: float, charge_power_w: float | None) -> Obstacle:
        return Obstacle(name=f"{WORLD_ENTITY_PREFIX}{name}", model="charger", pose=Pose(position=Position(x, 3.0), orientation=Orientation.identity()), dock=[[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]], charge_power_w=charge_power_w)

    zones = [Zone(name="bay_1", polygon=square(0.0)), Zone(name="bay_2", polygon=square(2.0)), Zone(name="dock_a", polygon=square(4.0))]
    return Level(id="0", zones=zones, objects=[docked("charger_1", 0.0, 1620.0), docked("charger_2", 3.0, 1620.0), docked("locker_1", 6.0, None)])


def _mode(seed: int, **params: object) -> tuple[TM_Fleet, dict[str, _Robot], SimpleNamespace, dict[str, Callable[[SimpleNamespace], None]]]:
    from arena_rclpy_mixins.shared import Namespace

    from task_generator.tasks.robots.fleet.impl import TM_Fleet

    rng = EpisodeRng()
    rng.reseed(seed)
    robots = {name: _Robot(name) for name in ("r0", "r1", "r2", "r3")}
    world = _world()
    ctx = SimpleNamespace(robots=robots, world_manager=SimpleNamespace(world_compacted=lambda: world), conf=SimpleNamespace(RNG=rng))
    callbacks: dict[str, Callable[[SimpleNamespace], None]] = {}
    node = SimpleNamespace(
        ROSParam=_Param({**PARAMS, **params}),
        sim_time=SimpleNamespace(sec=100, nanosec=0),
        create_subscription=lambda kind, topic, callback, qos: callbacks.setdefault(topic, callback),
        destroy_subscription=lambda subscription: None,
        get_logger=lambda: SimpleNamespace(get_child=lambda name: SimpleNamespace(info=lambda message: None, debug=lambda message: None)),
    )
    mode = TM_Fleet(ctx=ctx, namespace=Namespace("task.fleet").ParamNamespace(), node=node)
    return mode, robots, node, callbacks


def test_releases_depend_on_the_seed_alone() -> None:
    from task_generator.tasks.robots.fleet.impl import releases

    def first(seed: int, n: int) -> list[object]:
        rng = EpisodeRng()
        rng.reseed(seed)
        return list(itertools.islice(releases(rng.stream("robots", "fleet"), 12.0, 0.3), n))

    assert first(3, 50) == first(3, 50)
    assert first(3, 50) != first(4, 50)
    jobs = first(3, 4000)
    assert abs(jobs[-1].release_s / len(jobs) - 300.0) < 15.0
    assert abs(sum(job.kind == "pickup" for job in jobs) / len(jobs) - 0.3) < 0.03


def test_fleet_charges_low_robots_once_a_dock_is_free_and_gives_jobs_to_the_others() -> None:
    pytest.importorskip("rclpy")
    mode, robots, node, callbacks = _mode(5)
    plan = asyncio.run(mode.reset())

    assert [plan.robots[name].start_pose.position.x for name in robots] == [0.0, 1.0, 2.0, 3.0]
    assert plan.forbidden_zones == []
    assert sorted(callbacks) == [f"{name}/power_publisher/energy" for name in robots]
    for name, soc in {"r0": 20.0, "r1": 90.0, "r2": 60.0, "r3": 25.0}.items():
        callbacks[f"{name}/power_publisher/energy"](SimpleNamespace(battery_soc_percent=soc))

    node.sim_time = SimpleNamespace(sec=100 + 3600, nanosec=0)
    asyncio.run(mode.tick({}, ()))

    assert robots["r0"].requests == [] and mode._state["r0"] == "parked"
    assert robots["r3"].requests == [] and mode._state["r3"] == "charge"
    assert [robots[name].job() for name in ("r1", "r2")] == [0, 1]

    robots["r1"].done = True
    asyncio.run(mode.tick({}, ()))
    assert robots["r0"].last() == "charge at charger_1"
    assert mode._home["r0"] == "charger_1" and mode._home["r2"] == "bay_1"
    assert robots["r1"].job() == 2

    callbacks["r0/power_publisher/energy"](SimpleNamespace(battery_soc_percent=100.0))
    asyncio.run(mode.tick({"r0": SimpleNamespace(outcome="met")}, ()))
    assert robots["r0"].job() == 3
    assert mode._state["r3"] == "charge"


@pytest.mark.parametrize("params", [{"loading": "dock_b"}, {"locker": "bay_1_marking"}, {"homes": ["bay_1", "bay_2", "charger_1", "charger_3"]}])
def test_fleet_reset_refuses_a_spot_the_world_does_not_have(params: dict[str, object]) -> None:
    pytest.importorskip("rclpy")
    mode, *_ = _mode(5, **params)
    with pytest.raises(ValueError, match="no zone or docked object"):
        asyncio.run(mode.reset())
