from __future__ import annotations

import asyncio
import itertools
import math
from collections.abc import Iterator
from types import SimpleNamespace

import pytest

TIMEOUT = 60
FLEET_HOMES = ["bay_1", "bay_2", "charger_1", "charger_2"]
FLEET_SPOTS = [*FLEET_HOMES, "bay_3", "bay_4"]
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
        self.navigation_resets = 0
        self.soc: float | None = None
        self.battery_capacity_wh: float | None = None
        self.power_profile: dict[str, float] | None = None
        self.pose: object = None
        self.runner = SimpleNamespace(phases=[])

    @property
    async def is_done(self) -> bool:
        return self.done

    async def submit_task(self, request: object) -> None:
        self.submitted.append(request)
        self.runner.phases.extend(SimpleNamespace(outcome="met", reason="") for _ in request.phases)
        self.done = False

    async def move(self, pose: object) -> None:
        self.moved.append(pose)

    async def reset_navigation(self) -> None:
        self.navigation_resets += 1

    def namespace(self, *path: str) -> str:
        return "/".join((self.name, *path))

    def resolve_target(self, phase: object) -> object:
        import attrs
        from arena_runtime.shared import Orientation, Pose, Position

        return attrs.evolve(phase, pose=Pose(position=Position(float(FLEET_SPOTS.index(phase.target)), 0.0), orientation=Orientation.identity()))


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

    # zone centres bay_1 (0.5, 0.5), bay_2 (2.5, 0.5), dock_a (4.5, 0.5), bay_3 (8.5, 0.5), bay_4 (10.5, 0.5); dock centres charger_1 (1, 3), charger_2 (4, 3), locker_1 (7, 3)
    zones = [Zone(name="bay_1", polygon=square(0.0)), Zone(name="bay_2", polygon=square(2.0)), Zone(name="dock_a", polygon=square(4.0)), Zone(name="bay_3", polygon=square(8.0)), Zone(name="bay_4", polygon=square(10.0))]
    return Level(id="0", zones=zones, objects=[docked("charger_1", 0.0, 1620.0), docked("charger_2", 3.0, 1620.0), docked("locker_1", 6.0, None)])


LOADING_XY = (4.5, 0.5)
LOCKER_XY = (7.0, 3.0)
#: the charger nearest the locker, charger_2's dock centre
CHARGER_2_XY = (4.0, 3.0)
JACKAL = {"static_power_w": 42.2, "odometry_linear_w_per_mps": 50.0, "odometry_angular_w_per_radps": 17.0}


def _fleet_jobs(seed: int, n: int, rate: float = 12.0, pickup_share: float = 0.3) -> list[object]:
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.fleet.impl import releases

    rng = EpisodeRng()
    rng.reseed(seed)
    return list(itertools.islice(releases(rng.stream("robots", "fleet"), rate, pickup_share), n))


def _stream(kinds: str) -> Iterator[object]:
    """Jobs of the given kinds (`d`elivery, `p`ickup) all released at 1 s, then none."""
    from task_generator.tasks.robots.fleet.impl import Job

    yield from (Job(index, "pickup" if kind == "p" else "delivery", 1.0) for index, kind in enumerate(kinds))
    yield from (Job(index, "delivery", math.inf) for index in itertools.count(len(kinds)))


def _fleet(monkeypatch: pytest.MonkeyPatch | None = None, jobs: str | None = None, **params: object) -> tuple[object, dict[str, _Manager], SimpleNamespace, SimpleNamespace]:
    """The fleet mode on the test world with robots r0 to r3, the seed's job stream or the given `jobs`, and a record of its task log and log lines."""
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.Time import Time
    from arena_runtime.env.constants.rng import EpisodeRng

    from task_generator.tasks.robots.fleet import impl

    if jobs is not None:
        assert monkeypatch is not None
        monkeypatch.setattr(impl, "releases", lambda rng, rate, share: _stream(jobs))
    values = {"homes": FLEET_HOMES, "loading": "dock_a", "locker": "locker_1", "rate": FLEET_RATE, **params}

    def param(name: object, default: object = None, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(value=values.get(str(name).rsplit(".", 1)[-1], default))

    rng = EpisodeRng()
    rng.reseed(FLEET_SEED)
    managers = {name: _Manager(name, done=False) for name in ("r0", "r1", "r2", "r3")}
    world = _fleet_world()
    record = SimpleNamespace(log={}, lines=[])

    def publish(kind: str, fields: dict[str, str]) -> None:
        assert kind == "fleet"
        record.log = dict(fields)

    logger = SimpleNamespace(**{level: (lambda message, level=level: record.lines.append((level, message))) for level in ("debug", "info", "warning")})
    ctx = SimpleNamespace(
        robots=managers,
        world_manager=SimpleNamespace(world_compacted=lambda: world),
        environment_manager=SimpleNamespace(ezilear=lambda pose: pose),
        conf=SimpleNamespace(RNG=rng),
        episode=SimpleNamespace(publish_task_entity=publish),
    )
    node = SimpleNamespace(ROSParam={str: param, float: param, bool: param, list[str]: param}, sim_time=Time(sec=FLEET_T0), get_logger=lambda: SimpleNamespace(get_child=lambda name: logger))
    return impl.TM_Fleet(ctx=ctx, namespace=Namespace("task.fleet").ParamNamespace(), node=node), managers, node, record


def _at(node: SimpleNamespace, release_s: float) -> None:
    from arena_rclpy_mixins.Time import Time

    node.sim_time = Time.from_float(FLEET_T0 + release_s + 0.001)


def _tick(mode: object, ended: dict[str, object] | None = None, assigned: tuple[str, ...] = ()) -> None:
    asyncio.run(mode.tick(ended or {}, assigned))


def _met() -> SimpleNamespace:
    return SimpleNamespace(outcome="met", reason="")


def _socs(managers: dict[str, _Manager], **socs: float) -> None:
    for name, soc in socs.items():
        managers[name].soc = soc


def _phase(manager: _Manager) -> object:
    return manager.submitted[-1].phases[0]


def _phase_text(manager: _Manager) -> str:
    return _phase(manager).text


def _job_index(manager: _Manager) -> int:
    return int(_phase_text(manager).split(":")[0].split()[-1])


def _job_wh(start: tuple[float, float], per_m: float = 0.046) -> float:
    """The mode's estimate for a delivery from `start`: 1.2 times the straight line to the loading point, on to the locker and on to the charger nearest it, plus the jackal's 42.2 W over 25 s of loading and unloading."""
    return (math.dist(start, LOADING_XY) + math.dist(LOADING_XY, LOCKER_XY) + math.dist(LOCKER_XY, CHARGER_2_XY)) * 1.2 * per_m + 42.2 * 25.0 / 3600.0


def test_fleet_releases_depend_on_the_seed_alone() -> None:
    assert _fleet_jobs(3, 50) == _fleet_jobs(3, 50)
    assert _fleet_jobs(3, 50) != _fleet_jobs(4, 50)
    jobs = _fleet_jobs(3, 4000)
    assert abs(jobs[-1].release_s / len(jobs) - 300.0) < 15.0
    assert abs(sum(job.kind == "pickup" for job in jobs) / len(jobs) - 0.3) < 0.03


def test_fleet_charges_low_robots_once_a_charger_is_empty_and_gives_jobs_to_the_fullest_one_dock_at_a_time() -> None:
    mode, managers, node, record = _fleet(pickup_share=0.0)
    plan = asyncio.run(mode.reset())
    assert [plan.robots[name].start_pose.position.x for name in managers] == [0.0, 1.0, 2.0, 3.0]
    assert plan.forbidden_zones == []
    assert record.log == {"t0": "100.000"}
    _socs(managers, r0=20.0, r1=90.0, r2=60.0, r3=25.0)
    releases = [job.release_s for job in _fleet_jobs(FLEET_SEED, 3, FLEET_RATE, 0.0)]

    _at(node, releases[2])
    _tick(mode)
    # r1 and r2 take jobs 0 and 1, the two free robots over charge_below; job 2 waits. r2 holds charger_1 and is away,
    # so r0 takes charger_1 and r2 gets bay_1; r3 charges on its charger home
    assert _phase_text(managers["r1"]) == f"delivery 0: load at dock_a, released {releases[0]:.1f} s"
    assert _phase(managers["r1"]).on_failure == "stop_task"
    assert _phase_text(managers["r2"]) == f"delivery 1: load at dock_a, released {releases[1]:.1f} s"
    assert _phase_text(managers["r0"]) == "charge at charger_1"
    assert managers["r3"].submitted == []
    assert [record.log[f"job.{index}.released"] for index in range(3)] == [f"{release:.3f}" for release in releases]
    assert (record.log["job.0.robot"], record.log["job.0.assigned"]) == ("r1", f"{releases[2] + 0.001:.3f}")
    assert "job.2.robot" not in record.log
    assert (record.log["charge.0.robot"], record.log["charge.0.trigger"], record.log["charge.0.charger"]) == ("r0", "charge_below", "charger_1")
    assert (record.log["charge.1.robot"], record.log["charge.1.charger"], record.log["charge.1.soc_start"]) == ("r3", "charger_2", "25.00")
    assert record.log["charge.1.queued"] == record.log["charge.1.dispatched"] == record.log["charge.1.started"]

    # both reach the loading point's approach: r1 takes it, r2 holds in place before it
    _tick(mode)
    assert _phase_text(managers["r1"]).startswith("delivery 0: load at dock_a")
    assert _phase_text(managers["r2"]) == "delivery 1: wait for dock_a"
    assert _phase(managers["r2"]).hold and _phase(managers["r2"]).hold_time > 0

    # r1 loaded: it goes straight on to the locker and takes it, and r2 gets the loading point it freed
    _tick(mode, {"r1": _met()})
    assert _phase_text(managers["r1"]) == "delivery 0: unload at locker_1"
    assert _phase_text(managers["r2"]).startswith("delivery 1: load at dock_a")
    assert "job.0.loaded" in record.log and "job.0.timeout_1" in record.log

    # r1 unloaded on a tick the mode did not see: its task is done, so its last phase counts, and it takes job 2
    managers["r1"].done = True
    _tick(mode)
    assert (record.log["job.0.outcome"], record.log["job.0.ended"], record.log["job.0.wait_s"]) == ("completed", f"{releases[2] + 0.001:.3f}", "0.000")
    assert "job.0.reason" not in record.log
    assert _job_index(managers["r1"]) == 2

    _socs(managers, r0=100.0, r3=100.0)
    _tick(mode, {"r0": _met(), "r2": _met()})
    assert [record.log[f"charge.{index}.outcome"] for index in range(2)] == ["charged", "charged"]
    assert record.log["charge.0.soc_end"] == "100.00"
    assert _phase_text(managers["r2"]) == "delivery 1: unload at locker_1"

    _tick(mode, {"r2": _met()})
    assert _phase_text(managers["r2"]) == "park at bay_1"
    assert managers["r3"].submitted == []


def test_fleet_reads_the_charge_from_the_robot_manager_and_decides_no_charge_on_an_unknown_one(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="dp")
    asyncio.run(mode.reset())
    managers["r2"].soc = 50.0
    _at(node, 1.0)
    _tick(mode)
    # the one robot with a reading goes first, the others count neither as full nor as low
    assert _job_index(managers["r2"]) == 0
    assert _job_index(managers["r0"]) == 1
    assert managers["r1"].submitted == managers["r3"].submitted == []
    assert not any(field.startswith("charge.") for field in record.log)


def test_fleet_robots_queue_before_a_held_dock_longest_waiting_first_and_a_loaded_one_drives_straight_on(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="dpp")
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=80.0, r2=70.0, r3=60.0)
    _at(node, 1.0)
    _tick(mode)
    # no job waits for its first dock: every one is handed out at once
    assert _phase_text(managers["r0"]).startswith("delivery 0: load at dock_a")
    assert _phase_text(managers["r1"]).startswith("pickup 1: load at locker_1")
    assert _phase_text(managers["r2"]).startswith("pickup 2: load at locker_1")
    _tick(mode)
    assert _phase_text(managers["r2"]) == "pickup 2: wait for locker_1"

    # r0 loaded: it drives on to the locker and, r1 holding it, waits before it behind r2
    _at(node, 2.0)
    _tick(mode, {"r0": _met()})
    assert _phase_text(managers["r0"]) == "delivery 0: wait for locker_1"

    # r1 loaded and frees the locker: r2, waiting longest, gets it; r1 takes the loading point r0 freed
    _at(node, 3.0)
    _tick(mode, {"r1": _met()})
    assert _phase_text(managers["r2"]).startswith("pickup 2: load at locker_1")
    assert _phase_text(managers["r0"]) == "delivery 0: wait for locker_1"
    assert _phase_text(managers["r1"]) == "pickup 1: unload at dock_a"

    # r2 loaded: r0 gets the locker after 2 s of waiting, r2 waits before the loading point r1 holds
    _at(node, 4.0)
    _tick(mode, {"r2": _met()})
    assert _phase_text(managers["r0"]) == "delivery 0: unload at locker_1"
    assert _phase_text(managers["r2"]) == "pickup 2: wait for dock_a"
    _tick(mode, {"r0": _met()})
    assert (record.log["job.0.outcome"], record.log["job.0.wait_s"]) == ("completed", "2.000")
    assert managers["r3"].submitted == []


def test_fleet_takes_a_dock_only_within_approach_m_and_drives_on_while_it_is_further(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="pp")
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=80.0, r2=10.0, r3=10.0)
    managers["r0"].pose, managers["r1"].pose = _pose(*LOCKER_XY), _pose(LOCKER_XY[0], LOCKER_XY[1] + 9.0)
    _at(node, 1.0)
    _tick(mode)
    _tick(mode)
    # r0 at the locker holds it; r1, 9 m off, keeps driving
    assert _phase_text(managers["r1"]).startswith("pickup 1: load at locker_1")
    managers["r1"].pose = _pose(LOCKER_XY[0], LOCKER_XY[1] + 3.5)
    _tick(mode)
    assert _phase_text(managers["r1"]) == "pickup 1: wait for locker_1"
    _tick(mode, {"r0": _met()})
    assert _phase_text(managers["r1"]).startswith("pickup 1: load at locker_1")


def test_fleet_times_each_leg_from_its_start_without_its_waits_for_a_dock(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="pp", job_timeout=100.0, job_timeout_per_m=0.0)
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=80.0, r2=10.0, r3=10.0)
    _at(node, 1.0)
    _tick(mode)
    _tick(mode)
    assert record.log["job.1.timeout_0"] == "100.000"
    assert _phase_text(managers["r1"]) == "pickup 1: wait for locker_1"
    # r0 holds the locker 159 s without arriving: its leg times out and frees it; r1, waiting all along, is not timed out
    _at(node, 160.0)
    _tick(mode)
    assert (record.log["job.0.outcome"], record.log["job.0.reason"]) == ("abandoned", "timeout")
    assert "job.1.ended" not in record.log
    assert _phase_text(managers["r1"]).startswith("pickup 1: load at locker_1")
    # r1's own leg has 100 s of driving from there
    _at(node, 259.5)
    _tick(mode)
    assert "job.1.ended" not in record.log
    _at(node, 261.0)
    _tick(mode)
    assert (record.log["job.1.outcome"], record.log["job.1.reason"], record.log["job.1.wait_s"]) == ("abandoned", "timeout", "159.000")


def test_fleet_sends_a_robot_to_charge_before_a_job_that_would_leave_it_under_the_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d", charge_below=5.0)
    asyncio.run(mode.reset())
    for manager in managers.values():
        manager.battery_capacity_wh, manager.power_profile = 36.0, {"static_power_w": 42.2}
    far, near = _job_wh((0.5, 0.5)), _job_wh((2.5, 0.5))
    # r0 on bay_1 is fuller than r1 on bay_2, but the job would leave it just under 10 % of 36 Wh, and r1 just over
    _socs(managers, r0=10.0 + 100.0 * far / 36.0 - 0.1, r1=10.0 + 100.0 * near / 36.0 + 0.1, r2=4.0, r3=4.0)
    assert managers["r0"].soc > managers["r1"].soc
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r1"]) == 0
    assert record.log["job.0.energy_wh"] == f"{near:.3f}"
    assert (record.log["charge.0.robot"], record.log["charge.0.trigger"]) == ("r0", "job 0")
    # both chargers' holders charge at home, so r0 waits at its own
    assert managers["r0"].submitted == []
    assert [record.log[f"charge.{index}.robot"] for index in (1, 2)] == ["r2", "r3"]


def test_fleet_gives_a_job_to_a_robot_charged_to_charge_until_whatever_the_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d", wh_per_m=10.0)
    asyncio.run(mode.reset())
    for manager in managers.values():
        manager.battery_capacity_wh = 36.0
    _socs(managers, r0=100.0, r1=50.0, r2=50.0, r3=50.0)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r0"]) == 0
    assert not any(field.startswith("charge.") for field in record.log)


def test_the_energy_per_metre_follows_the_power_profile_when_wh_per_m_is_0(monkeypatch: pytest.MonkeyPatch) -> None:
    from task_generator.tasks.robots.fleet.impl import wh_per_m

    assert wh_per_m(JACKAL) == pytest.approx((42.2 + 50.0 * 0.4 + 17.0 * 0.2) / 0.4 / 3600.0)
    assert round(wh_per_m(JACKAL), 3) == 0.046
    assert wh_per_m(None) == 0.0
    mode, managers, node, record = _fleet(monkeypatch, jobs="d", wh_per_m=0.0)
    asyncio.run(mode.reset())
    managers["r0"].soc, managers["r0"].battery_capacity_wh, managers["r0"].power_profile = 90.0, 36.0, JACKAL
    _at(node, 1.0)
    _tick(mode)
    assert record.log["job.0.energy_wh"] == f"{_job_wh((0.5, 0.5), wh_per_m(JACKAL)):.3f}"


def test_fleet_logs_a_failed_job_as_failed_frees_its_dock_and_does_not_queue_it_again(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="dd")
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=80.0, r2=70.0, r3=60.0)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r0"]) == 0
    assert _job_index(managers["r1"]) == 1
    _tick(mode)
    assert _phase_text(managers["r1"]) == "delivery 1: wait for dock_a"

    _tick(mode, {"r0": SimpleNamespace(outcome="failed", reason="no path to dock_a")})
    assert (record.log["job.0.outcome"], record.log["job.0.reason"]) == ("failed", "no path to dock_a")
    assert "job.0.loaded" not in record.log
    # the loading point is free again and goes to r1, waiting for it; job 0 is not queued again
    assert _phase_text(managers["r1"]).startswith("delivery 1: load at dock_a")
    assert _phase_text(managers["r0"]) == "park at bay_1"
    assert not any("finished" in line for _, line in record.lines)
    assert any(level == "warning" and "failed delivery 0" in line for level, line in record.lines)
    for _ in range(3):
        _tick(mode)
    assert not any(_phase_text(manager).startswith("delivery 0") for manager in managers.values() if manager.submitted)


def test_fleet_abandons_a_job_past_its_timeout_and_sends_the_robot_home(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d", job_timeout=100.0, job_timeout_per_m=0.0)
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=50.0, r2=50.0, r3=50.0)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r0"]) == 0

    _at(node, 100.0)
    _tick(mode)
    assert "job.0.ended" not in record.log
    _at(node, 101.5)
    _tick(mode)
    assert (record.log["job.0.outcome"], record.log["job.0.reason"], record.log["job.0.ended"]) == ("abandoned", "timeout", "101.501")
    assert _phase_text(managers["r0"]) == "park at bay_1"


def test_fleet_abandons_a_trip_to_a_charger_past_the_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="", homes=["bay_1", "bay_2", "bay_3", "bay_4"], job_timeout=100.0, job_timeout_per_m=0.0)
    asyncio.run(mode.reset())
    _socs(managers, r0=20.0, r1=90.0, r2=90.0, r3=90.0)
    _at(node, 1.0)
    _tick(mode)
    assert _phase_text(managers["r0"]) == "charge at charger_1"
    _at(node, 102.0)
    _tick(mode)
    assert (record.log["charge.0.outcome"], record.log["charge.0.reason"]) == ("abandoned", "timeout")
    assert "charge.0.started" not in record.log
    # still under charge_below, it queues again and is sent to its charger home once more
    assert (record.log["charge.1.robot"], record.log["charge.1.dispatched"]) == ("r0", "102.001")


def test_fleet_leg_timeout_grows_with_the_route_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d", job_timeout=100.0, job_timeout_per_m=10.0)
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=50.0, r2=50.0, r3=50.0)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r0"]) == 0
    # r0 on bay_1: its first leg, 1.2 x bay_1 to the loading point
    timeout = 100.0 + 10.0 * 1.2 * math.dist((0.5, 0.5), LOADING_XY)
    assert record.log["job.0.timeout_0"] == f"{timeout:.3f}"
    _at(node, 1.0 + timeout - 1.0)
    _tick(mode)
    assert "job.0.ended" not in record.log
    _at(node, 1.0 + timeout + 0.5)
    _tick(mode)
    assert (record.log["job.0.outcome"], record.log["job.0.reason"]) == ("abandoned", "timeout")


def test_fleet_charger_trip_timeout_grows_with_the_distance_to_the_charger(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="", homes=["bay_1", "bay_2", "bay_3", "bay_4"], job_timeout=100.0, job_timeout_per_m=10.0)
    asyncio.run(mode.reset())
    # r3 on bay_4 (10.5, 0.5) goes to charger_1 (1, 3), the first one whose holder is away
    _socs(managers, r0=90.0, r1=90.0, r2=90.0, r3=20.0)
    _at(node, 1.0)
    _tick(mode)
    assert _phase_text(managers["r3"]) == "charge at charger_1"
    timeout = 100.0 + 10.0 * 1.2 * math.dist((10.5, 0.5), (1.0, 3.0))
    _at(node, 1.0 + timeout - 1.0)
    _tick(mode)
    assert "charge.0.ended" not in record.log
    _at(node, 1.0 + timeout + 0.5)
    _tick(mode)
    assert (record.log["charge.0.outcome"], record.log["charge.0.reason"]) == ("abandoned", "timeout")


def test_fleet_job_energy_counts_the_leg_to_the_charger_nearest_the_second_dock(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="dp", charge_below=5.0)
    asyncio.run(mode.reset())
    for manager in managers.values():
        manager.battery_capacity_wh, manager.power_profile = 36.0, {"static_power_w": 42.2}
    _socs(managers, r0=90.0, r1=80.0, r2=4.0, r3=4.0)
    _at(node, 1.0)
    _tick(mode)
    # the delivery unloads at the locker (7, 3), nearest charger_2 (4, 3); the pickup unloads at the loading point (4.5, 0.5), nearest charger_2 too
    assert record.log["job.0.energy_wh"] == f"{_job_wh((0.5, 0.5)):.3f}"
    pickup = (math.dist((2.5, 0.5), LOCKER_XY) + math.dist(LOCKER_XY, LOADING_XY) + math.dist(LOADING_XY, CHARGER_2_XY)) * 1.2 * 0.046 + 42.2 * 25.0 / 3600.0
    assert record.log["job.1.energy_wh"] == f"{pickup:.3f}"


def _pose(x: float, y: float) -> object:
    from arena_runtime.shared import Orientation, Pose, Position

    return Pose(position=Position(x, y), orientation=Orientation.identity())


def test_fleet_fails_a_jammed_job_and_resets_the_robot_to_its_home(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d")
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=50.0, r2=50.0, r3=50.0)
    for name, manager in managers.items():
        manager.pose = _pose(20.0 + 3 * int(name[1]), 9.0)
    managers["r0"].pose = _pose(1.0, 1.5)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r0"]) == 0
    # on its first leg from t 20, r0 creeps 0.15 m in 60 s, 3.4 m short of the loading point
    for t, x in ((20.0, 1.05), (50.0, 1.1), (80.5, 1.2)):
        managers["r0"].pose = _pose(x, 1.5)
        _at(node, t)
        _tick(mode)
    assert (record.log["job.0.outcome"], record.log["job.0.reason"], record.log["job.0.ended"]) == ("failed", "jam", "80.501")
    assert (record.log["jam.0.robot"], record.log["jam.0.job"], record.log["jam.0.target"]) == ("r0", "0", "dock_a")
    assert float(record.log["jam.0.robot_m"]) == pytest.approx(math.dist((1.2, 1.5), (23.0, 9.0)), abs=0.01)
    # reset to its home, bay_1 (the fake's dispatch pose of a spot), its navigation reset after the teleport, and parked there
    assert managers["r0"].moved and managers["r0"].moved[-1].position.x == pytest.approx(float(FLEET_SPOTS.index("bay_1")))
    assert managers["r0"].navigation_resets == len(managers["r0"].moved) == 1
    assert _phase_text(managers["r0"]) == "park at bay_1"
    assert any(level == "warning" and "r0 jammed" in line for level, line in record.lines)


def test_fleet_counts_no_jam_for_a_robot_that_moves_or_holds_at_its_dock(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d")
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=50.0, r2=50.0, r3=50.0)
    managers["r0"].pose = _pose(0.5, 2.0)
    _at(node, 1.0)
    _tick(mode)
    # 0.6 m in 60 s is progress
    for t, x in ((30.0, 0.8), (61.5, 1.1), (90.0, 1.4)):
        managers["r0"].pose = _pose(x, 2.0)
        _at(node, t)
        _tick(mode)
    # standing 0.5 m from the loading point for 2 minutes is docking, not a jam
    for t in (100.0, 160.0, 230.0):
        managers["r0"].pose = _pose(LOADING_XY[0] - 0.5, LOADING_XY[1])
        _at(node, t)
        _tick(mode)
    assert not any(field.startswith("jam.") for field in record.log) and "job.0.ended" not in record.log
    assert managers["r0"].moved == [] and managers["r0"].navigation_resets == 0


def test_fleet_abandons_a_jammed_charging_trip_and_the_rule_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for window, jams in ((60.0, True), (0.0, False)):
        mode, managers, node, record = _fleet(monkeypatch, jobs="", homes=["bay_1", "bay_2", "bay_3", "bay_4"], jam_window_s=window)
        asyncio.run(mode.reset())
        _socs(managers, r0=20.0, r1=90.0, r2=90.0, r3=90.0)
        managers["r0"].pose = _pose(8.0, 8.0)
        _at(node, 1.0)
        _tick(mode)
        assert _phase_text(managers["r0"]) == "charge at charger_1"
        for t in (62.0, 123.0):
            _at(node, t)
            _tick(mode)
        assert ("charge.0.reason" in record.log) == jams
        if jams:
            assert (record.log["charge.0.outcome"], record.log["charge.0.reason"], record.log["jam.0.charge"]) == ("abandoned", "jam", "0")
            # reset to the bay it had before the trip, not to the charger it was headed for, its navigation reset too
            assert managers["r0"].moved[-1].position.x == pytest.approx(float(FLEET_SPOTS.index("bay_1")))
            assert managers["r0"].navigation_resets == 1
            # still under charge_below, it queues for a charger again from there
            assert record.log["charge.1.robot"] == "r0"


def _top_up_fleet(monkeypatch: pytest.MonkeyPatch, jobs: str, **params: object) -> tuple[object, dict[str, _Manager], SimpleNamespace, SimpleNamespace]:
    """The fleet with the given jobs, charge below 30 % until 90 %, each robot standing at its home: r0 bay_1, r1 bay_2,
    r2 charger_1, r3 charger_2."""
    mode, managers, node, record = _fleet(monkeypatch, jobs=jobs, charge_below=30.0, charge_until=90.0, **params)
    asyncio.run(mode.reset())
    for name, xy in (("r0", (0.5, 0.5)), ("r1", (2.5, 0.5)), ("r2", (1.0, 3.0)), ("r3", (4.0, 3.0))):
        managers[name].pose = _pose(*xy)
    return mode, managers, node, record


def test_fleet_tops_up_a_free_robot_under_charge_until_at_a_charger_whose_holder_is_away_and_frees_it_once_docked(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="ddd")
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=100.0)
    _at(node, 1.0)
    _tick(mode)
    # the charger robots, fullest, take jobs 0 and 1, r1 job 2; r0 under charge_until takes charger_1, whose holder r2
    # is away, and r2 gets bay_1
    assert (_job_index(managers["r2"]), _job_index(managers["r3"]), _job_index(managers["r1"])) == (0, 1, 2)
    assert _phase_text(managers["r0"]) == "top up at charger_1"
    assert _phase(managers["r0"]).on_failure == "stop_task"
    assert (mode._home["r0"], mode._home["r2"]) == ("charger_1", "bay_1")
    assert (record.log["top_up.0.robot"], record.log["top_up.0.charger"], record.log["top_up.0.soc_dispatched"]) == ("r0", "charger_1", "60.00")
    # the top-up is no charge in the log's charge table, which the capacity reads as time out of service
    assert not any(field.startswith("charge.") for field in record.log)
    # while it drives there it is on a charging trip and takes no job
    assert "r0" not in mode._takers()

    # docked: it charges there and is free for jobs again
    managers["r0"].pose = _pose(1.0, 3.0)
    _socs(managers, r0=61.0)
    _tick(mode, {"r0": _met()})
    assert (record.log["top_up.0.outcome"], record.log["top_up.0.soc_start"]) == ("docked", "61.00")
    assert "top_up.0.started" in record.log and "top_up.0.ended" in record.log
    assert "r0" in mode._takers()


def test_fleet_tops_up_nobody_while_a_robot_waits_to_charge_or_no_holder_is_away_and_never_over_charge_until(monkeypatch: pytest.MonkeyPatch) -> None:
    # r0 must charge (under charge_below) and gets charger_1, whose holder is away; r1 under charge_until waits, no
    # charger being free for it: charger_2's holder r3 is parked on it
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d")
    _socs(managers, r0=20.0, r1=60.0, r2=100.0, r3=80.0)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r2"]) == 0
    assert _phase_text(managers["r0"]) == "charge at charger_1"
    assert managers["r1"].submitted == [] and "top_up.0.robot" not in record.log
    # a robot at charge_until or over takes no charger
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d")
    _socs(managers, r0=90.0, r1=95.0, r2=100.0, r3=80.0)
    _at(node, 1.0)
    _tick(mode)
    assert managers["r0"].submitted == managers["r1"].submitted == [] and "top_up.0.robot" not in record.log


def test_fleet_tops_up_the_lowest_first_one_robot_per_charger_and_the_rule_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    for top_up in (True, False):
        mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="dd", top_up=top_up)
        _socs(managers, r0=70.0, r1=50.0, r2=100.0, r3=99.0)
        _at(node, 1.0)
        _tick(mode)
        assert (_job_index(managers["r2"]), _job_index(managers["r3"])) == (0, 1)
        if top_up:
            # both holders away: r1, the lowest, takes charger_1, and r0 charger_2
            assert (_phase_text(managers["r1"]), _phase_text(managers["r0"])) == ("top up at charger_1", "top up at charger_2")
            assert (mode._home["r2"], mode._home["r3"]) == ("bay_2", "bay_1")
            assert [record.log[f"top_up.{index}.robot"] for index in range(2)] == ["r1", "r0"]
        else:
            assert managers["r0"].submitted == managers["r1"].submitted == [] and "top_up.0.robot" not in record.log


def test_fleet_a_jammed_top_up_gives_the_charger_back_and_resets_the_robot_to_its_old_home(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d")
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=95.0)
    managers["r0"].pose = _pose(8.0, 8.0)
    _at(node, 1.0)
    _tick(mode)
    assert _phase_text(managers["r0"]) == "top up at charger_1"
    for t in (62.0, 123.0):
        _at(node, t)
        _tick(mode)
    assert (record.log["top_up.0.outcome"], record.log["top_up.0.reason"], record.log["jam.0.top_up"]) == ("abandoned", "jam", "0")
    # the charger goes back to r2, and r0 to bay_1, where it is reset to
    assert (mode._home["r0"], mode._home["r2"]) == ("bay_1", "charger_1")
    assert managers["r0"].moved[-1].position.x == pytest.approx(float(FLEET_SPOTS.index("bay_1")))


def test_fleet_a_jammed_top_up_whose_holder_took_another_charger_since_leaves_one_robot_per_home(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="dd", jam_window_s=30.0)
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=99.0)
    _at(node, 1.0)
    _tick(mode)
    # r2 and r3 take the jobs, r0 tops up at charger_1 and r2 gets bay_1
    assert (_job_index(managers["r2"]), _job_index(managers["r3"]), _phase_text(managers["r0"])) == (0, 1, "top up at charger_1")
    managers["r2"].pose = _pose(*LOADING_XY)
    _at(node, 10.0)
    _tick(mode, {"r2": _met()})
    managers["r2"].pose = _pose(*LOCKER_XY)
    managers["r3"].pose = _pose(LOADING_XY[0], LOADING_XY[1] + 1.0)
    _socs(managers, r2=60.0)
    _at(node, 15.0)
    _tick(mode, {"r2": _met()})
    # r2, back at 60 %, tops up at charger_2 while its holder r3 is on job 1: r3 gets bay_1
    _at(node, 16.0)
    _tick(mode)
    assert _phase_text(managers["r2"]) == "top up at charger_2" and mode._home["r3"] == "bay_1"
    # r0 jams: back to bay_1, and charger_1 goes to r3, who has bay_1 now
    _at(node, 41.0)
    _tick(mode)
    assert (record.log["jam.0.robot"], record.log["top_up.0.reason"]) == ("r0", "jam")
    assert mode._home == {"r0": "bay_1", "r1": "bay_2", "r2": "charger_2", "r3": "charger_1"}
    assert managers["r0"].moved[-1].position.x == pytest.approx(float(FLEET_SPOTS.index("bay_1")))


@pytest.mark.parametrize("end", ["timeout", "failed"])
def test_fleet_a_top_up_that_times_out_or_fails_keeps_the_charger_as_the_robots_home(monkeypatch: pytest.MonkeyPatch, end: str) -> None:
    # only a jam gives the charger back: past its timeout, or on a failed trip, the robot parks at the charger it took
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d", jam_window_s=0.0, **({"job_timeout": 100.0, "job_timeout_per_m": 0.0} if end == "timeout" else {}))
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=95.0)
    _at(node, 1.0)
    _tick(mode)
    assert _phase_text(managers["r0"]) == "top up at charger_1"
    _at(node, 102.0)
    _tick(mode, {"r0": SimpleNamespace(outcome="failed", reason="planner")} if end == "failed" else {})
    assert (record.log["top_up.0.outcome"], record.log["top_up.0.reason"]) == ("abandoned", "timeout" if end == "timeout" else "planner")
    assert (mode._home["r0"], mode._home["r2"]) == ("charger_1", "bay_1")
    assert _phase_text(managers["r0"]) == "park at charger_1"


def _holder_back_low(mode: object, managers: dict[str, _Manager], node: SimpleNamespace, *, at: float = 40.0) -> None:
    """r2, away from charger_1 on job 0, loads at the loading point at 20 s and unloads at the locker at `at`, at 25 %:
    under charge_below, it queues to charge."""
    managers["r2"].pose = _pose(*LOADING_XY)
    _at(node, 20.0)
    _tick(mode, {"r2": _met()})
    managers["r2"].pose = _pose(*LOCKER_XY)
    _socs(managers, r2=25.0)
    _at(node, at)
    _tick(mode, {"r2": _met()})


@pytest.mark.parametrize("top_up", [True, False])
def test_fleet_a_holder_back_low_from_a_job_takes_its_charger_from_a_top_up_on_its_way_there(monkeypatch: pytest.MonkeyPatch, top_up: bool) -> None:
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d", top_up=top_up)
    # r3 on charger_2 under charge_until keeps it; r2, the fullest, takes the job; r0 tops up at charger_1 (with top_up)
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=85.0)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r2"]) == 0
    _holder_back_low(mode, managers, node)
    assert record.log["job.0.outcome"] == "completed"
    # with or without top-ups, r2 goes to charge at charger_1 at once
    assert _phase_text(managers["r2"]) == "charge at charger_1" and mode._home["r2"] == "charger_1" and "r2" not in mode._charge_queue
    assert record.log["charge.0.dispatched"] == "40.001"
    if not top_up:
        assert managers["r0"].submitted == [] and not any(field.startswith("top_up.") for field in record.log)
        return
    # r0, not docked yet, gives the charger up and goes back to its old home
    assert (record.log["top_up.0.outcome"], record.log["top_up.0.reason"], record.log["top_up.0.yielded_to"]) == ("abandoned", "yielded", "r2")
    assert (record.log["top_up.0.ended"], record.log["charge.0.yielded_by"]) == ("40.001", "r0")
    assert "top_up.0.started" not in record.log
    assert mode._home["r0"] == "bay_1" and _phase_text(managers["r0"]) == "park at bay_1"
    # no charger's holder is away: r0 takes no other one
    _at(node, 41.0)
    _tick(mode)
    assert _phase_text(managers["r0"]) == "park at bay_1" and "top_up.1.robot" not in record.log


def test_fleet_a_robot_queued_to_charge_takes_a_charger_from_a_top_up_on_its_way_there_its_own_first(monkeypatch: pytest.MonkeyPatch) -> None:
    # r1 at bay_2 falls under charge_below while r0 drives to top up at charger_1: r1 takes it, r0 gets bay_2
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d")
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=85.0)
    _at(node, 1.0)
    _tick(mode)
    assert _phase_text(managers["r0"]) == "top up at charger_1"
    _socs(managers, r1=25.0)
    _at(node, 10.0)
    _tick(mode)
    assert _phase_text(managers["r1"]) == "charge at charger_1"
    assert (mode._home["r0"], mode._home["r1"], mode._home["r2"]) == ("bay_2", "charger_1", "bay_1")
    assert (record.log["top_up.0.yielded_to"], record.log["charge.0.yielded_by"]) == ("r1", "r0")
    assert _phase_text(managers["r0"]) == "park at bay_2"

    # both holders away, both chargers topped up at: r3 back low takes back charger_2, its own, not charger_1
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="dd")
    _socs(managers, r0=70.0, r1=50.0, r2=100.0, r3=99.0)
    _at(node, 1.0)
    _tick(mode)
    assert (_phase_text(managers["r1"]), _phase_text(managers["r0"])) == ("top up at charger_1", "top up at charger_2")
    _socs(managers, r3=25.0)
    _at(node, 10.0)
    _tick(mode, {"r3": SimpleNamespace(outcome="failed", reason="planner")})
    assert record.log["job.1.outcome"] == "failed"
    assert _phase_text(managers["r3"]) == "charge at charger_2"
    assert (record.log["top_up.1.robot"], record.log["top_up.1.yielded_to"]) == ("r0", "r3")
    assert "top_up.0.outcome" not in record.log and mode._home["r1"] == "charger_1"
    assert (mode._home["r0"], mode._home["r3"]) == ("bay_1", "charger_2")


def test_fleet_a_docked_top_up_gives_its_charger_up_at_once_until_charged_there_to_charge_until(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d")
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=85.0)
    _at(node, 1.0)
    _tick(mode)
    # r0 docks at charger_1 at 40 %, free for jobs, before r2 is back
    managers["r0"].pose = _pose(1.0, 3.0)
    _socs(managers, r0=40.0)
    _at(node, 10.0)
    _tick(mode, {"r0": _met()})
    assert record.log["top_up.0.outcome"] == "docked"
    _holder_back_low(mode, managers, node)
    # r0, under charge_until, gives charger_1 up at once, and stays on it until r2 holds before it
    assert _phase_text(managers["r2"]) == "charge at charger_1" and "r2" not in mode._charge_queue
    assert (record.log["charge.0.dispatched"], record.log["charge.0.yielded_by"]) == ("40.001", "r0")
    assert (mode._home["r0"], mode._home["r2"]) == ("bay_1", "charger_1")
    assert _phase_text(managers["r0"]) == "top up at charger_1"
    managers["r2"].pose = _pose(4.0, 2.0)
    _at(node, 50.0)
    _tick(mode)
    assert _phase_text(managers["r2"]) == "wait for charger_1, r0 leaving it"
    assert _phase_text(managers["r0"]) == "park at bay_1"

    # charged to charge_until there, r0 holds charger_1 like any holder: under charge_until again it keeps it
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d")
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=85.0)
    _at(node, 1.0)
    _tick(mode)
    managers["r0"].pose = _pose(1.0, 3.0)
    _at(node, 5.0)
    _tick(mode, {"r0": _met()})
    _socs(managers, r0=90.0)
    _at(node, 10.0)
    _tick(mode)
    _socs(managers, r0=60.0)
    _holder_back_low(mode, managers, node)
    assert "r2" in mode._charge_queue and _phase_text(managers["r2"]) == "park at bay_1"
    assert mode._home["r0"] == "charger_1" and "charge.0.dispatched" not in record.log


def test_fleet_a_jammed_trip_to_a_charger_taken_from_a_top_up_gives_it_back_as_a_top_up(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _top_up_fleet(monkeypatch, jobs="d")
    _socs(managers, r0=60.0, r1=95.0, r2=100.0, r3=85.0)
    _at(node, 1.0)
    _tick(mode)
    _holder_back_low(mode, managers, node)
    assert _phase_text(managers["r2"]) == "charge at charger_1" and _phase_text(managers["r0"]) == "park at bay_1"
    # r2 stalls at the locker: jammed, it is reset to bay_1, and charger_1 goes back to r0, which drives back to it
    for t in (41.0, 102.0):
        _at(node, t)
        _tick(mode)
    assert (record.log["charge.0.outcome"], record.log["charge.0.reason"]) == ("abandoned", "jam")
    assert (mode._home["r0"], mode._home["r2"]) == ("charger_1", "bay_1")
    assert _phase_text(managers["r0"]) == "park at charger_1"
    assert record.log["charge.1.robot"] == "r2" and "charge.1.dispatched" not in record.log
    # back on charger_1 under charge_until, r0 holds it by its top-up still and gives it up to r2 at once
    managers["r0"].pose = _pose(1.0, 3.0)
    _at(node, 110.0)
    _tick(mode, {"r0": _met()})
    assert _phase_text(managers["r2"]) == "charge at charger_1"
    assert record.log["charge.1.yielded_by"] == "r0"


def _yield_fleet(monkeypatch: pytest.MonkeyPatch, **params: object) -> tuple[object, dict[str, _Manager], SimpleNamespace, SimpleNamespace]:
    """The fleet with no jobs, charge below 30 % until 90 %, approach 2 m, each robot standing at its home: r0 bay_1, r1 bay_2, r2 charger_1, r3 charger_2."""
    mode, managers, node, record = _fleet(monkeypatch, jobs="", charge_below=30.0, charge_until=90.0, approach_m=2.0, **params)
    asyncio.run(mode.reset())
    for name, xy in (("r0", (0.5, 0.5)), ("r1", (2.5, 0.5)), ("r2", (1.0, 3.0)), ("r3", (4.0, 3.0))):
        managers[name].pose = _pose(*xy)
    return mode, managers, node, record


def test_fleet_a_charged_holder_parked_on_its_charger_gives_it_up_and_leaves_once_the_queued_robot_holds_before_it(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _yield_fleet(monkeypatch)
    _socs(managers, r0=20.0, r1=95.0, r2=95.0, r3=60.0)
    _at(node, 1.0)
    _tick(mode)
    # r2 parked on charger_1 at 95 % gives it up and takes r0's bay; r3 at 60 % keeps charger_2
    assert _phase_text(managers["r0"]) == "charge at charger_1"
    assert (record.log["charge.0.robot"], record.log["charge.0.charger"], record.log["charge.0.yielded_by"]) == ("r0", "charger_1", "r2")
    assert (mode._home["r0"], mode._home["r2"], mode._home["r3"]) == ("charger_1", "bay_1", "charger_2")
    # r2 stays on the charger while r0 is on its way: they do not meet head-on along r0's way
    assert managers["r2"].submitted == []
    _at(node, 10.0)
    _tick(mode)
    assert managers["r2"].submitted == []

    # within approach_m (2 m) of charger_1 r0 holds in place, and only then r2 leaves for bay_1
    managers["r0"].pose = _pose(1.0, 1.5)
    _at(node, 20.0)
    _tick(mode)
    assert _phase_text(managers["r0"]) == "wait for charger_1, r2 leaving it"
    assert _phase(managers["r0"]).hold and _phase(managers["r0"]).target is None
    assert _phase_text(managers["r2"]) == "park at bay_1"

    # holding past the jam window is no jam; r2 still within approach_m of the charger keeps r0 holding
    managers["r2"].pose = _pose(1.5, 2.0)
    for t in (50.0, 90.0):
        _at(node, t)
        _tick(mode)
    assert _phase_text(managers["r0"]) == "wait for charger_1, r2 leaving it"
    assert not any(field.startswith("jam.") for field in record.log)

    # r2 is clear: r0 drives on to the charger and charges there
    managers["r2"].pose = _pose(0.5, 0.5)
    _at(node, 100.0)
    _tick(mode, {"r2": _met()})
    assert _phase_text(managers["r0"]) == "charge at charger_1"
    assert (record.log["charge.0.holder_left"], record.log["charge.0.held_s"]) == ("100.001", "80.000")
    _tick(mode, {"r0": _met()})
    assert record.log["charge.0.started"] == "100.001"
    assert managers["r1"].submitted == []


def test_fleet_a_holder_under_charge_until_or_charging_keeps_its_charger(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _yield_fleet(monkeypatch)
    # r3 at 25 % charges on its charger home; r2 parked on charger_1 just under charge_until keeps it
    _socs(managers, r0=20.0, r1=95.0, r2=89.9, r3=25.0)
    _at(node, 1.0)
    _tick(mode)
    assert (record.log["charge.1.robot"], record.log["charge.1.started"]) == ("r3", "1.001")
    assert managers["r0"].submitted == []
    assert not any(field.endswith(".yielded_by") for field in record.log)
    assert (mode._home["r0"], mode._home["r2"]) == ("bay_1", "charger_1")
    # r3 still charging (50 %, under charge_until) keeps charger_2; r2 at charge_until gives charger_1 up
    _socs(managers, r2=90.0, r3=50.0)
    _at(node, 2.0)
    _tick(mode)
    assert _phase_text(managers["r0"]) == "charge at charger_1"
    assert (record.log["charge.0.robot"], record.log["charge.0.yielded_by"]) == ("r0", "r2")
    assert mode._home["r3"] == "charger_2" and "charge.1.ended" not in record.log


def test_fleet_sends_the_lowest_battery_to_a_yielding_charger_first_ties_by_queue_time_then_name(monkeypatch: pytest.MonkeyPatch) -> None:
    for socs, first in (({"r0": 25.0, "r1": 15.0}, "r1"), ({"r0": 20.0, "r1": 20.0}, "r0")):
        mode, managers, node, record = _yield_fleet(monkeypatch)
        # r3 charging on charger_2 keeps it; r2 parked full on charger_1 gives it up to one of the two
        _socs(managers, r2=95.0, r3=25.0, **socs)
        _at(node, 1.0)
        _tick(mode)
        other = "r0" if first == "r1" else "r1"
        assert _phase_text(managers[first]) == "charge at charger_1"
        assert managers[other].submitted == []
        assert mode._home[first] == "charger_1" and mode._home["r2"] == ("bay_1" if first == "r0" else "bay_2")


def test_fleet_swaps_no_charger_back_once_both_robots_are_full(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _yield_fleet(monkeypatch)
    _socs(managers, r0=20.0, r1=95.0, r2=95.0, r3=60.0)
    _at(node, 1.0)
    _tick(mode)
    managers["r0"].pose = _pose(1.0, 1.5)
    _tick(mode)
    managers["r2"].pose = _pose(0.5, 0.5)
    _tick(mode, {"r2": _met()})
    _tick(mode, {"r0": _met()})
    _socs(managers, r0=95.0)
    _tick(mode)
    assert record.log["charge.0.outcome"] == "charged"
    sent = {name: len(manager.submitted) for name, manager in managers.items()}
    homes = dict(mode._home)
    # r0 parked full on charger_1 and r2 full on bay_1: nobody is queued, so nobody moves and the homes stay
    for t in (10.0, 100.0, 1000.0):
        _at(node, t)
        _tick(mode)
    assert {name: len(manager.submitted) for name, manager in managers.items()} == sent
    assert mode._home == homes == {"r0": "charger_1", "r1": "bay_2", "r2": "bay_1", "r3": "charger_2"}
    assert "charge.1.robot" not in record.log


def test_fleet_a_yield_trip_that_times_out_gives_the_charger_back_and_sends_the_holder_back_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _yield_fleet(monkeypatch, job_timeout=100.0, job_timeout_per_m=0.0)
    _socs(managers, r0=20.0, r1=95.0, r2=95.0, r3=60.0)
    _at(node, 1.0)
    _tick(mode)
    managers["r0"].pose = _pose(1.0, 1.5)
    _at(node, 2.0)
    _tick(mode)
    assert _phase_text(managers["r2"]) == "park at bay_1"
    # r2 stops halfway, within approach_m of the charger, and r0 holds until the trip's timeout
    managers["r2"].pose = _pose(1.2, 2.2)
    _at(node, 102.0)
    _tick(mode)
    assert (record.log["charge.0.outcome"], record.log["charge.0.reason"]) == ("abandoned", "timeout")
    assert (mode._home["r0"], mode._home["r2"]) == ("bay_1", "charger_1")
    # r2 heads back to charger_1, parked on its way, so r0, queued again, takes no charger until r2 is on it again
    assert _phase_text(managers["r2"]) == "park at charger_1"
    assert record.log["charge.1.robot"] == "r0" and "charge.1.dispatched" not in record.log
    assert _phase_text(managers["r0"]) == "park at bay_1"


def test_fleet_sends_a_second_leg_again_that_could_not_be_submitted_and_never_completes_the_job_without_it(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d", job_timeout=100.0, job_timeout_per_m=0.0)
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=50.0, r2=50.0, r3=50.0)
    _at(node, 1.0)
    _tick(mode)
    r0 = managers["r0"]
    real = r0.submit_task
    blocked = {"locker_1"}

    async def no_locker(request: object) -> None:
        if getattr(request.phases[0], "target", None) in blocked:
            raise ValueError("robot 'r0': goto target 'locker_1' has no free cell with 0.50 m clearance")
        await real(request)

    r0.submit_task = no_locker
    # the load leg ends met but the leg to the locker cannot be submitted: r0's task is done, its last phase the load leg
    _at(node, 30.0)
    _tick(mode, {"r0": _met()})
    r0.done = True
    _at(node, 31.0)
    _tick(mode)
    assert "job.0.ended" not in record.log and "job.0.loaded" in record.log
    # once the locker takes the robot again, the leg goes out on the next tick and the job ends only after it
    blocked.clear()
    _at(node, 32.0)
    _tick(mode)
    assert _phase_text(r0) == "delivery 0: unload at locker_1"
    _tick(mode, {"r0": _met()})
    assert record.log["job.0.outcome"] == "completed"


def test_fleet_abandons_a_second_leg_that_never_goes_out_once_its_timer_runs_out(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="d", job_timeout=100.0, job_timeout_per_m=0.0)
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=50.0, r2=50.0, r3=50.0)
    _at(node, 1.0)
    _tick(mode)
    r0 = managers["r0"]
    real = r0.submit_task

    async def no_locker(request: object) -> None:
        if getattr(request.phases[0], "target", None) == "locker_1":
            raise ValueError("no free cell")
        await real(request)

    r0.submit_task = no_locker
    _at(node, 30.0)
    _tick(mode, {"r0": _met()})
    r0.done = True
    for t in (31.0, 100.0, 129.0):
        _at(node, t)
        _tick(mode)
    assert "job.0.ended" not in record.log
    _at(node, 131.0)
    _tick(mode)
    assert (record.log["job.0.outcome"], record.log["job.0.reason"]) == ("abandoned", "timeout")


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"homes": ["bay_1", "bay_2", "bay_1", "charger_2"]}, r"homes names \['bay_1'\] more than once"),
        ({"homes": ["bay_1", "bay_2", "dock_a", "charger_2"]}, r"homes names \['dock_a'\], where jobs load and unload"),
        ({"charge_below": 60.0, "charge_until": 50.0}, r"charge_below \(60.0\) is above task.fleet.charge_until \(50.0\)"),
    ],
)
def test_fleet_reset_refuses_a_repeated_home_a_home_on_a_dock_and_charge_below_above_charge_until(params: dict[str, object], match: str) -> None:
    mode, *_ = _fleet(**params)
    with pytest.raises(ValueError, match=match):
        asyncio.run(mode.reset())


@pytest.mark.parametrize("params", [{"loading": "dock_b"}, {"locker": "bay_1_marking"}, {"homes": ["bay_1", "bay_2", "charger_1", "charger_3"]}])
def test_fleet_reset_refuses_a_spot_the_world_does_not_have(params: dict[str, object]) -> None:
    mode, *_ = _fleet(**params)
    with pytest.raises(ValueError, match="no goto target"):
        asyncio.run(mode.reset())


def test_fleet_abandons_a_job_whose_robot_is_assigned_from_outside_and_leaves_the_robot_to_that_task(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="dd")
    asyncio.run(mode.reset())
    _socs(managers, r0=90.0, r1=80.0, r2=70.0, r3=60.0)
    _at(node, 1.0)
    _tick(mode)
    assert _job_index(managers["r0"]) == 0

    _tick(mode, {}, ("r0",))
    assert (record.log["job.0.outcome"], record.log["job.0.reason"]) == ("abandoned", "assigned from outside")
    assert _job_index(managers["r1"]) == 1
    sent = len(managers["r0"].submitted)
    _tick(mode)
    assert len(managers["r0"].submitted) == sent
    managers["r0"].done = True
    _tick(mode)
    assert _phase_text(managers["r0"]) == "park at bay_1"

    asyncio.run(mode.teardown())
    assert record.log == {}


def test_fleet_logs_and_counts_each_time_a_battery_reaches_0(monkeypatch: pytest.MonkeyPatch) -> None:
    mode, managers, node, record = _fleet(monkeypatch, jobs="")
    asyncio.run(mode.reset())
    _socs(managers, r0=0.0, r1=50.0, r2=50.0, r3=50.0)
    _at(node, 5.0)
    _tick(mode)
    _tick(mode)
    assert (record.log["empty.0.robot"], record.log["empty.0.at"]) == ("r0", "5.001")
    assert "empty.1.robot" not in record.log
    managers["r0"].soc = 3.0
    _tick(mode)
    managers["r0"].soc = 0.0
    _at(node, 9.0)
    _tick(mode)
    assert (record.log["empty.1.robot"], record.log["empty.1.at"]) == ("r0", "9.001")
    assert [line for level, line in record.lines if "battery empty" in line] == ["r0: battery empty at 5.0 s, 1 this episode", "r0: battery empty at 9.0 s, 2 this episode"]


def test_fleet_ticks_before_its_first_reset() -> None:
    mode, managers, _, record = _fleet()
    asyncio.run(mode.tick({}, ()))
    assert all(manager.submitted == [] for manager in managers.values())
    assert record.log == {}
