from __future__ import annotations

import asyncio
import itertools
import typing
from collections.abc import Callable
from types import GenericAlias, SimpleNamespace

import attrs
import pytest

if typing.TYPE_CHECKING:
    from arena_runtime.env.task_api import GoToPhase, TaskRequest
    from arena_simulation_setup.tree.World import Level

    from task_generator.tasks.robots.fleet.impl import Job, TM_Fleet

HOMES = ["bay_1", "bay_2", "charger_1", "charger_2"]
RATE = 60.0
PARAMS = {"homes": HOMES, "loading": "dock_a", "locker": "locker_1", "rate": RATE}
SEED = 5
T0 = 100


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_robots_msgs.msg")


class _Manager:
    safe_distance = 0.3

    def __init__(self, name: str) -> None:
        self.name = name
        self.done = False
        self.submitted: list[TaskRequest] = []

    @property
    async def is_done(self) -> bool:
        return self.done

    async def submit_task(self, request: TaskRequest) -> None:
        self.submitted.append(request)
        self.done = False

    def namespace(self, *path: str) -> str:
        return "/".join((self.name, *path))

    def resolve_target(self, phase: GoToPhase) -> GoToPhase:
        from arena_runtime.shared import Orientation, Pose, Position

        return attrs.evolve(phase, pose=Pose(position=Position(float(HOMES.index(phase.target)), 0.0), orientation=Orientation.identity()))

    def last(self) -> str:
        return self.submitted[-1].phases[0].text

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
    from arena_runtime.shared import WORLD_ENTITY_PREFIX, Orientation, Pose, Position
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.tree.World import Level, Zone

    def square(x: float) -> list[Position]:
        return [Position(x, 0.0), Position(x + 1.0, 0.0), Position(x + 1.0, 1.0), Position(x, 1.0)]

    def docked(name: str, x: float, charge_power_w: float | None) -> Obstacle:
        return Obstacle(name=f"{WORLD_ENTITY_PREFIX}{name}", model="charger", pose=Pose(position=Position(x, 3.0), orientation=Orientation.identity()), dock=[[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]], charge_power_w=charge_power_w)

    zones = [Zone(name="bay_1", polygon=square(0.0)), Zone(name="bay_2", polygon=square(2.0)), Zone(name="dock_a", polygon=square(4.0))]
    return Level(id="0", zones=zones, objects=[docked("charger_1", 0.0, 1620.0), docked("charger_2", 3.0, 1620.0), docked("locker_1", 6.0, None)])


def _jobs(seed: int, n: int, rate: float = 12.0) -> list[Job]:
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.fleet.impl import releases

    rng = EpisodeRng()
    rng.reseed(seed)
    return list(itertools.islice(releases(rng.stream("robots", "fleet"), rate, 0.3), n))


def _fleet(**params: object) -> tuple[TM_Fleet, dict[str, _Manager], SimpleNamespace, dict[str, Callable[[SimpleNamespace], None]]]:
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.Time import Time
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.fleet.impl import TM_Fleet

    rng = EpisodeRng()
    rng.reseed(SEED)
    managers = {name: _Manager(name) for name in ("r0", "r1", "r2", "r3")}
    world = _world()
    ctx = SimpleNamespace(robots=managers, world_manager=SimpleNamespace(world_compacted=lambda: world), conf=SimpleNamespace(RNG=rng))
    energy: dict[str, Callable[[SimpleNamespace], None]] = {}
    node = SimpleNamespace(
        ROSParam=_Param({**PARAMS, **params}),
        sim_time=Time(sec=T0),
        create_subscription=lambda kind, topic, callback, qos: energy.setdefault(topic, callback),
        destroy_subscription=lambda subscription: None,
        get_logger=lambda: SimpleNamespace(get_child=lambda name: SimpleNamespace(info=lambda message: None, debug=lambda message: None)),
    )
    return TM_Fleet(ctx=ctx, namespace=Namespace("task.fleet").ParamNamespace(), node=node), managers, node, energy


def _at(node: SimpleNamespace, release_s: float) -> None:
    from arena_rclpy_mixins.Time import Time

    node.sim_time = Time.from_float(T0 + release_s + 0.001)


def test_releases_depend_on_the_seed_alone() -> None:
    assert _jobs(3, 50) == _jobs(3, 50)
    assert _jobs(3, 50) != _jobs(4, 50)
    jobs = _jobs(3, 4000)
    assert abs(jobs[-1].release_s / len(jobs) - 300.0) < 15.0
    assert abs(sum(job.kind == "pickup" for job in jobs) / len(jobs) - 0.3) < 0.03


def test_fleet_charges_low_robots_once_a_charger_is_empty_and_gives_jobs_to_the_fullest() -> None:
    mode, managers, node, energy = _fleet()
    plan = asyncio.run(mode.reset())
    assert [plan.robots[name].start_pose.position.x for name in managers] == [0.0, 1.0, 2.0, 3.0]
    assert plan.forbidden_zones == []
    for name, soc in {"r0": 20.0, "r1": 90.0, "r2": 60.0, "r3": 25.0}.items():
        energy[f"{name}/power_publisher/energy"](SimpleNamespace(battery_soc_percent=soc))
    releases = [job.release_s for job in _jobs(SEED, 4, RATE)]

    _at(node, releases[2])
    asyncio.run(mode.tick({}, ()))
    assert [managers[name].job() for name in ("r1", "r2")] == [0, 1]
    assert managers["r0"].submitted == [] and managers["r3"].submitted == []

    managers["r1"].done = True
    asyncio.run(mode.tick({}, ()))
    assert managers["r0"].last() == "charge at charger_1"
    assert managers["r1"].job() == 2

    managers["r2"].done = True
    asyncio.run(mode.tick({}, ()))
    assert managers["r2"].last() == "park at bay_1"

    energy["r0/power_publisher/energy"](SimpleNamespace(battery_soc_percent=100.0))
    _at(node, releases[3])
    asyncio.run(mode.tick({"r0": SimpleNamespace(outcome="met")}, ()))
    assert managers["r0"].job() == 3
    assert managers["r3"].submitted == []


@pytest.mark.parametrize("params", [{"loading": "dock_b"}, {"locker": "bay_1_marking"}, {"homes": ["bay_1", "bay_2", "charger_1", "charger_3"]}])
def test_fleet_reset_refuses_a_spot_the_world_does_not_have(params: dict[str, object]) -> None:
    mode, *_ = _fleet(**params)
    with pytest.raises(ValueError, match="no zone or docked object"):
        asyncio.run(mode.reset())


def test_fleet_ticks_before_its_first_reset() -> None:
    mode, managers, _, _ = _fleet()
    asyncio.run(mode.tick({}, ()))
    assert all(manager.submitted == [] for manager in managers.values())
