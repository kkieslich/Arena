from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace

import pytest

TIMEOUT = 60
FLEET_HOMES = ["bay_1", "bay_2", "charger_1", "charger_2"]
FLEET_RATE = 60.0
FLEET_SEED = 5
FLEET_T0 = 100


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")


class _Manager:
    safe_distance = 0.5

    def __init__(self, name: str, done: bool) -> None:
        self.name = name
        self.done = done
        self.submitted: list[object] = []
        self.moved: list[object] = []

    @property
    async def is_done(self) -> bool:
        return self.done

    async def submit_task(self, request: object) -> None:
        self.submitted.append(request)
        self.done = False

    async def move(self, pose: object) -> None:
        self.moved.append(pose)

    def namespace(self, *path: str) -> str:
        return "/".join((self.name, *path))

    def resolve_target(self, phase: object) -> object:
        import attrs
        from arena_runtime.shared import Orientation, Pose, Position

        return attrs.evolve(phase, pose=Pose(position=Position(float(FLEET_HOMES.index(phase.target)), 0.0), orientation=Orientation.identity()))


class _World:
    def get_positions_on_map(self, n: int, safe_dist: float, forbidden_zones: object = None) -> list[object]:
        from arena_runtime.shared import Position

        return [Position(float(i), 0.0, 0.0) for i in range(n)]

    def get_position_on_map(self, safe_dist: float, forbid: bool) -> object:
        from arena_runtime.shared import Position

        return Position(7.0, 7.0, 0.0)


def _explore(managers: list[_Manager], *, reset: bool = True) -> tuple[object, SimpleNamespace]:
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.explore.impl import TM_Explore

    rng = EpisodeRng()
    rng.reseed(3)
    node = SimpleNamespace(sim_time=SimpleNamespace(sec=0))
    ctx = SimpleNamespace(
        robots={manager.name: manager for manager in managers},
        world_manager=_World(),
        conf=SimpleNamespace(RNG=rng, TIMEOUT=SimpleNamespace(value=TIMEOUT), GOAL_TOLERANCE_RADIUS=SimpleNamespace(value=0.3)),
    )
    mode = TM_Explore(ctx=ctx, namespace=Namespace("task.explore"), node=node)
    if reset:
        asyncio.run(mode.reset())
    return mode, node


def test_explore_gives_a_robot_whose_task_finished_a_fresh_goal() -> None:
    from arena_runtime.env.task_api import GoToPhase

    finished, running = _Manager("r0", done=False), _Manager("r1", done=False)
    mode, _ = _explore([finished, running])
    finished.done = True
    asyncio.run(mode.tick({"r0": object()}, []))
    (request,) = finished.submitted
    assert [type(phase) for phase in request.phases] == [GoToPhase]
    assert (request.phases[0].pose.position.x, request.phases[0].pose.position.y) == (7.0, 7.0)
    assert running.submitted == []
    assert running.moved == []


def test_explore_also_restarts_a_robot_stopped_without_a_phase_ending() -> None:
    stopped = _Manager("r0", done=False)
    mode, _ = _explore([stopped])
    stopped.done = True
    asyncio.run(mode.tick({}, []))
    assert len(stopped.submitted) == 1
    assert stopped.moved == []


def test_explore_teleports_a_robot_that_ran_into_the_timeout_and_restarts_its_clock() -> None:
    stuck, fresh = _Manager("r0", done=False), _Manager("r1", done=False)
    mode, node = _explore([stuck, fresh])
    node.sim_time = SimpleNamespace(sec=TIMEOUT - 1)
    asyncio.run(mode.tick({}, []))
    assert stuck.moved == []
    node.sim_time = SimpleNamespace(sec=TIMEOUT)
    asyncio.run(mode.tick({}, []))
    assert len(stuck.moved) == len(fresh.moved) == 1
    asyncio.run(mode.tick({}, []))
    assert len(stuck.moved) == 1


def test_explore_holds_a_robot_assigned_from_outside_while_its_task_runs_until_the_next_reset() -> None:
    assigned, free = _Manager("r0", done=False), _Manager("r1", done=True)
    mode, node = _explore([assigned, free])
    asyncio.run(mode.tick({}, ["r0"]))
    assert assigned.submitted == []
    assert len(free.submitted) == 1
    node.sim_time = SimpleNamespace(sec=TIMEOUT)
    asyncio.run(mode.tick({}, []))
    assert assigned.submitted == []
    assert assigned.moved == []
    asyncio.run(mode.reset())
    node.sim_time = SimpleNamespace(sec=2 * TIMEOUT)
    asyncio.run(mode.tick({}, []))
    assert len(assigned.moved) == 1


def test_explore_does_not_teleport_a_timed_out_robot_assigned_from_outside() -> None:
    assigned = _Manager("r0", done=False)
    mode, node = _explore([assigned])
    node.sim_time = SimpleNamespace(sec=TIMEOUT)
    asyncio.run(mode.tick({}, ["r0"]))
    assert assigned.moved == []
    assert assigned.submitted == []


def test_explore_ticks_before_its_first_reset() -> None:
    manager = _Manager("r0", done=True)
    mode, _ = _explore([manager], reset=False)
    asyncio.run(mode.tick({}, []))
    assert len(manager.submitted) == 1


def test_explore_is_never_done() -> None:
    mode, _ = _explore([_Manager("r0", done=True)])
    assert asyncio.run(mode.done) is False


def _composite(*, failing: frozenset[str] = frozenset()) -> tuple[object, dict[str, tuple[dict, list]]]:
    from arena_rclpy_mixins.shared import Namespace

    from task_generator.tasks.robots import TM_Robots
    from task_generator.tasks.robots.composite import TM_Composite

    seen: dict[str, tuple[dict, list]] = {}

    class _Recording(TM_Robots):
        async def tick(self, ended: dict, assigned: list) -> None:
            key = ",".join(self._ctx.robots)
            seen[key] = (ended, list(assigned))
            if key in failing:
                raise RuntimeError(key)

    node = SimpleNamespace()
    first = _Recording(ctx=SimpleNamespace(robots={"a": 1, "b": 2}), namespace=Namespace("task.first"), node=node)
    second = _Recording(ctx=SimpleNamespace(robots={"c": 3}), namespace=Namespace("task.second"), node=node)
    composite = TM_Composite(ctx=SimpleNamespace(robots={"a": 1, "b": 2, "c": 3}), namespace=Namespace("task.composite"), node=node, sub_modes=[first, second])
    return composite, seen


def test_composite_hands_each_sub_mode_only_the_phases_and_assignments_of_its_robots() -> None:
    composite, seen = _composite()
    ended = {"a": object(), "c": object(), "other": object()}
    asyncio.run(composite.tick(ended, ["b", "c", "other"]))
    assert seen == {"a,b": ({"a": ended["a"]}, ["b"]), "c": ({"c": ended["c"]}, ["c"])}


def test_composite_ticks_every_sub_mode_when_one_raises_then_raises_the_first_error() -> None:
    composite, seen = _composite(failing=frozenset({"a,b", "c"}))
    with pytest.raises(RuntimeError, match="a,b"):
        asyncio.run(composite.tick({}, []))
    assert set(seen) == {"a,b", "c"}


def test_a_scoped_robots_view_keeps_the_parents_robot_order() -> None:
    from task_generator.tasks.robots.composite import _ScopedRobotsView

    view = _ScopedRobotsView(SimpleNamespace(managers={"c": 3, "a": 1, "b": 2, "d": 4}), frozenset({"b", "c", "d"}))
    assert list(view.managers.items()) == [("c", 3), ("b", 2), ("d", 4)]


def _fleet_world() -> object:
    from arena_runtime.shared import WORLD_ENTITY_PREFIX, Orientation, Pose, Position
    from arena_simulation_setup.shared import Obstacle
    from arena_simulation_setup.tree.World import Level, Zone

    def square(x: float) -> list[Position]:
        return [Position(x, 0.0), Position(x + 1.0, 0.0), Position(x + 1.0, 1.0), Position(x, 1.0)]

    def docked(name: str, x: float, charge_power_w: float | None) -> Obstacle:
        return Obstacle(name=f"{WORLD_ENTITY_PREFIX}{name}", model="charger", pose=Pose(position=Position(x, 3.0), orientation=Orientation.identity()), dock=[[0.5, -0.5], [1.5, -0.5], [1.5, 0.5], [0.5, 0.5]], charge_power_w=charge_power_w)

    zones = [Zone(name="bay_1", polygon=square(0.0)), Zone(name="bay_2", polygon=square(2.0)), Zone(name="dock_a", polygon=square(4.0))]
    return Level(id="0", zones=zones, objects=[docked("charger_1", 0.0, 1620.0), docked("charger_2", 3.0, 1620.0), docked("locker_1", 6.0, None)])


def _fleet_jobs(seed: int, n: int, rate: float = 12.0) -> list[object]:
    pytest.importorskip("arena_robots_msgs.msg")
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.fleet.impl import releases

    rng = EpisodeRng()
    rng.reseed(seed)
    return list(itertools.islice(releases(rng.stream("robots", "fleet"), rate, 0.3), n))


def _fleet(**params: object) -> tuple[object, dict[str, _Manager], SimpleNamespace, dict[str, object]]:
    pytest.importorskip("arena_robots_msgs.msg")
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.Time import Time
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.fleet.impl import TM_Fleet

    values = {"homes": FLEET_HOMES, "loading": "dock_a", "locker": "locker_1", "rate": FLEET_RATE, **params}

    def param(name: object, default: object = None, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(value=values.get(str(name).rsplit(".", 1)[-1], default))

    rng = EpisodeRng()
    rng.reseed(FLEET_SEED)
    managers = {name: _Manager(name, done=False) for name in ("r0", "r1", "r2", "r3")}
    world = _fleet_world()
    ctx = SimpleNamespace(robots=managers, world_manager=SimpleNamespace(world_compacted=lambda: world), conf=SimpleNamespace(RNG=rng))
    energy: dict[str, object] = {}
    node = SimpleNamespace(
        ROSParam={str: param, float: param, list[str]: param},
        sim_time=Time(sec=FLEET_T0),
        create_subscription=lambda kind, topic, callback, qos: energy.setdefault(topic, callback),
        destroy_subscription=lambda subscription: None,
        get_logger=lambda: SimpleNamespace(get_child=lambda name: SimpleNamespace(info=lambda message: None, debug=lambda message: None)),
    )
    return TM_Fleet(ctx=ctx, namespace=Namespace("task.fleet").ParamNamespace(), node=node), managers, node, energy


def _at(node: SimpleNamespace, release_s: float) -> None:
    from arena_rclpy_mixins.Time import Time

    node.sim_time = Time.from_float(FLEET_T0 + release_s + 0.001)


def _phase_text(manager: _Manager) -> str:
    return manager.submitted[-1].phases[0].text


def _job_index(manager: _Manager) -> int:
    return int(_phase_text(manager).split(":")[0].split()[-1])


def test_fleet_releases_depend_on_the_seed_alone() -> None:
    assert _fleet_jobs(3, 50) == _fleet_jobs(3, 50)
    assert _fleet_jobs(3, 50) != _fleet_jobs(4, 50)
    jobs = _fleet_jobs(3, 4000)
    assert abs(jobs[-1].release_s / len(jobs) - 300.0) < 15.0
    assert abs(sum(job.kind == "pickup" for job in jobs) / len(jobs) - 0.3) < 0.03


def test_fleet_charges_low_robots_once_a_charger_is_empty_and_gives_jobs_to_the_fullest() -> None:
    mode, managers, node, energy = _fleet()
    plan = asyncio.run(mode.reset())
    assert [plan.robots[name].start_pose.position.x for name in managers] == [0.0, 1.0, 2.0, 3.0]
    assert plan.forbidden_zones == []
    for name, soc in {"r0": 20.0, "r1": 90.0, "r2": 60.0, "r3": 25.0}.items():
        energy[f"{name}/power_publisher/energy"](SimpleNamespace(battery_soc_percent=soc))
    releases = [job.release_s for job in _fleet_jobs(FLEET_SEED, 4, FLEET_RATE)]

    _at(node, releases[2])
    asyncio.run(mode.tick({}, ()))
    assert [_job_index(managers[name]) for name in ("r1", "r2")] == [0, 1]
    assert managers["r0"].submitted == [] and managers["r3"].submitted == []

    managers["r1"].done = True
    asyncio.run(mode.tick({}, ()))
    assert _phase_text(managers["r0"]) == "charge at charger_1"
    assert _job_index(managers["r1"]) == 2

    managers["r2"].done = True
    asyncio.run(mode.tick({}, ()))
    assert _phase_text(managers["r2"]) == "park at bay_1"

    energy["r0/power_publisher/energy"](SimpleNamespace(battery_soc_percent=100.0))
    _at(node, releases[3])
    asyncio.run(mode.tick({"r0": SimpleNamespace(outcome="met")}, ()))
    assert _job_index(managers["r0"]) == 3
    assert managers["r3"].submitted == []


@pytest.mark.parametrize("params", [{"loading": "dock_b"}, {"locker": "bay_1_marking"}, {"homes": ["bay_1", "bay_2", "charger_1", "charger_3"]}])
def test_fleet_reset_refuses_a_spot_the_world_does_not_have(params: dict[str, object]) -> None:
    mode, *_ = _fleet(**params)
    with pytest.raises(ValueError, match="no goto target"):
        asyncio.run(mode.reset())


def test_fleet_ticks_before_its_first_reset() -> None:
    mode, managers, _, _ = _fleet()
    asyncio.run(mode.tick({}, ()))
    assert all(manager.submitted == [] for manager in managers.values())
