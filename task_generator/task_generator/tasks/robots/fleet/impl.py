import enum
import itertools
import math
import typing
from collections import deque
from collections.abc import Collection, Iterator, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
import rclpy
import shapely
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_runtime.env.task_api import GoToPhase, PhaseState, TaskPhase, TaskRequest, authored_statics
from arena_runtime.env.task_layer import ResetPlan, RobotReset

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots

#: the semantic entity kind, and name, of the task log
LOG_KIND = "fleet"
#: how long a robot holds in place before a busy dock at most; the mode sends it on before, once the dock frees
WAIT_HOLD_S = 3600.0
#: mean speed and turn rate assumed while a robot drives, for the energy per metre derived from its power profile (Nav2 caps the jackal at 0.5 m/s and 1.5 rad/s)
DRIVE_SPEED_MPS = 0.4
DRIVE_TURN_RADPS = 0.2
#: when a robot goes to charge (`charge_rule`), and what a jam on a charging trip does (`charge_jam`)
CHARGE_RULES = ("threshold", "need")
CHARGE_JAMS = ("home", "retry")


@attrs.frozen
class Job:
    index: int
    kind: typing.Literal["delivery", "pickup"]
    release_s: float
    locker: str = ""  # the locker a job of a job list goes to; empty: the mode's one locker
    entrance: str = ""  # the recipient entrance it is addressed to, for the log


class _State(enum.Enum):
    PARKED = "parked"  # idle, at its home or on the way there
    FREE = "free"  # idle, to be sent home on this tick
    JOB = "job"  # on a job leg, its dock held once it is within approach_m of it
    DOCK_WAIT = "dock_wait"  # on a job leg, holding in place within approach_m of a dock another robot holds
    CHARGE = "charge"  # on the way to a charger, or charging in it
    OUTSIDE = "outside"  # running a task assigned from outside until it is done
    EMPTY = "empty"  # battery at 0 % with stop_empty: holds in place to the episode's end


@attrs.define
class _Run:
    """A job a robot has taken: the docks it loads and unloads at, when it took the job, the leg it is on, when that leg
    started and how long it may drive, whether it holds the leg's dock, and the time it waited for docks: on this leg (a
    leg's timer stops while it waits) and over the job."""

    job: Job
    docks: tuple[str, str]
    assigned_s: float
    leg: int = 0
    leg_started_s: float = 0.0
    leg_timeout_s: float = 0.0
    dispatched: bool = False
    reserved: bool = False
    waiting_since: float | None = None
    leg_waited_s: float = 0.0
    waited_s: float = 0.0

    def driving_s(self, now: float) -> float:
        """Seconds of the current leg spent other than waiting for its dock."""
        waiting = 0.0 if self.waiting_since is None else now - self.waiting_since
        return now - self.leg_started_s - self.leg_waited_s - waiting


@attrs.define
class _Charge:
    """One charge of one robot: its index in its table of the task log, where and when it went, how long the trip may take,
    when it arrived."""

    index: int
    charger: str
    dispatched_s: float
    timeout_s: float = 0.0
    table: str = "charge"  # the log table it writes to: `charge`, or `top_up` for an opportunistic charge
    started_s: float | None = None
    home_before: str | None = None  # the robot's home before the trip took the charger as its home
    swapped_with: str | None = None  # the charger's holder, who got that home
    swapped_top_up: bool = False  # that holder held the charger by a top-up: a revert gives it back as such
    leaving: str | None = None  # a holder parked on the charger that gives it up (`_yields`) and has not cleared it yet
    holding_since: float | None = None  # when the robot began to hold in place before the charger for `leaving`
    power_since: float | None = None  # once started: when the charge last rose by no_power_rise (or started), for `no_power_s`
    power_soc: float | None = None  # the robot's charge then, None while it was unknown


def releases(rng: np.random.Generator, rate_per_h: float, pickup_share: float) -> Iterator[Job]:
    """Jobs in release order, a Poisson stream drawn from rng alone."""
    release_s = 0.0
    for index in itertools.count():
        release_s += float(rng.exponential(3600.0 / rate_per_h))
        yield Job(index, "pickup" if rng.random() < pickup_share else "delivery", release_s)


def listed(entries: Sequence[str]) -> list[Job]:
    """The jobs of a job list (`task.fleet.jobs`): each entry `<release s> <delivery|pickup> <locker> [<recipient entrance>]`, in release order. ValueError for an entry that is none, or a list out of order."""
    jobs: list[Job] = []
    for index, entry in enumerate(entries):
        parts = str(entry).split()
        try:
            release_s = float(parts[0])
        except (IndexError, ValueError):
            release_s = math.nan
        if len(parts) not in (3, 4) or parts[1] not in ("delivery", "pickup") or not math.isfinite(release_s) or release_s < 0.0:
            raise ValueError(f"task.fleet.jobs entry {index} is '<release s> <delivery|pickup> <locker> [<recipient entrance>]', not {entry!r}")
        jobs.append(Job(index, typing.cast(typing.Literal["delivery", "pickup"], parts[1]), release_s, parts[2], parts[3] if len(parts) == 4 else ""))
    if any(a.release_s > b.release_s for a, b in itertools.pairwise(jobs)):
        raise ValueError("task.fleet.jobs is not in release order")
    return jobs


def _then_none(jobs: Sequence[Job]) -> Iterator[Job]:
    """`jobs`, then nothing more."""
    yield from jobs
    yield from (Job(index, "delivery", math.inf) for index in itertools.count(len(jobs)))


def wh_per_m(profile: Mapping[str, float] | None) -> float:
    """Energy in Wh per metre a robot with this power profile uses while driving at DRIVE_SPEED_MPS and DRIVE_TURN_RADPS, standby included, 0 without a profile."""
    if profile is None:
        return 0.0
    power_w = float(profile.get("static_power_w", 0.0)) + float(profile.get("odometry_linear_w_per_mps", 0.0)) * DRIVE_SPEED_MPS + float(profile.get("odometry_angular_w_per_radps", 0.0)) * DRIVE_TURN_RADPS
    return power_w / DRIVE_SPEED_MPS / 3600.0


class TM_Fleet(TM_Robots):
    """Deliveries and pickups released at a fixed rate from the episode seed, handed to free robots one dock at a time, with charging trips below a battery level or before a job the battery would not last, and opportunistic charging at chargers whose holders are away."""

    _homes: ROSParamT[list[str]]
    _loading: ROSParamT[str]
    _locker: ROSParamT[str]
    _rate: ROSParamT[float]
    _pickup_share: ROSParamT[float]
    _load_s: ROSParamT[float]
    _unload_s: ROSParamT[float]
    _charge_below: ROSParamT[float]
    _charge_until: ROSParamT[float]
    _reserve: ROSParamT[float]
    _wh_per_m: ROSParamT[float]
    _detour: ROSParamT[float]
    _job_timeout: ROSParamT[float]
    _job_timeout_per_m: ROSParamT[float]
    _jam_window_s: ROSParamT[float]
    _jam_progress_m: ROSParamT[float]
    _jam_goal_m: ROSParamT[float]
    _jam_home: ROSParamT[bool]
    _leg_timeout_moving: ROSParamT[bool]
    _approach_m: ROSParamT[float]
    _top_up: ROSParamT[bool]
    _lockers: ROSParamT[list[str]]
    _job_list: ROSParamT[list[str]]
    _charge_rule: ROSParamT[str]
    _nearest_charger: ROSParamT[bool]
    _charge_jam: ROSParamT[str]
    _soc_log: ROSParamT[str]
    _soc_log_s: ROSParamT[float]
    _idle_floor: ROSParamT[float]
    _one_need_charge: ROSParamT[bool]
    _charge_at_start: ROSParamT[bool]
    _stop_empty: ROSParamT[bool]
    _no_power_s: ROSParamT[float]
    _no_power_rise: ROSParamT[float]

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)

        self._homes = self.node.ROSParam[list[str]](self.namespace("homes"), [], type_=rclpy.Parameter.Type.STRING_ARRAY)
        self._loading = self.node.ROSParam[str](self.namespace("loading"), "")
        self._locker = self.node.ROSParam[str](self.namespace("locker"), "")
        self._rate = self.node.ROSParam[float](self.namespace("rate"), 12.0)
        self._pickup_share = self.node.ROSParam[float](self.namespace("pickup_share"), 0.3)
        self._load_s = self.node.ROSParam[float](self.namespace("load_s"), 15.0)
        self._unload_s = self.node.ROSParam[float](self.namespace("unload_s"), 10.0)
        self._charge_below = self.node.ROSParam[float](self.namespace("charge_below"), 30.0)
        self._charge_until = self.node.ROSParam[float](self.namespace("charge_until"), 100.0)
        self._reserve = self.node.ROSParam[float](self.namespace("reserve"), 10.0)
        self._wh_per_m = self.node.ROSParam[float](self.namespace("wh_per_m"), 0.046)
        self._detour = self.node.ROSParam[float](self.namespace("detour"), 1.2)
        self._job_timeout = self.node.ROSParam[float](self.namespace("job_timeout"), 300.0)
        self._job_timeout_per_m = self.node.ROSParam[float](self.namespace("job_timeout_per_m"), 5.0)
        self._jam_window_s = self.node.ROSParam[float](self.namespace("jam_window_s"), 60.0)
        self._jam_progress_m = self.node.ROSParam[float](self.namespace("jam_progress_m"), 0.5)
        self._jam_goal_m = self.node.ROSParam[float](self.namespace("jam_goal_m"), 1.0)
        self._jam_home = self.node.ROSParam[bool](self.namespace("jam_home"), True)
        self._leg_timeout_moving = self.node.ROSParam[bool](self.namespace("leg_timeout_moving"), True)
        self._approach_m = self.node.ROSParam[float](self.namespace("approach_m"), 5.0)
        self._top_up = self.node.ROSParam[bool](self.namespace("top_up"), True)
        self._lockers = self.node.ROSParam[list[str]](self.namespace("lockers"), [], type_=rclpy.Parameter.Type.STRING_ARRAY)
        self._job_list = self.node.ROSParam[list[str]](self.namespace("jobs"), [], type_=rclpy.Parameter.Type.STRING_ARRAY)
        self._charge_rule = self.node.ROSParam[str](self.namespace("charge_rule"), "threshold")
        self._nearest_charger = self.node.ROSParam[bool](self.namespace("nearest_charger"), False)
        self._charge_jam = self.node.ROSParam[str](self.namespace("charge_jam"), "home")
        self._soc_log = self.node.ROSParam[str](self.namespace("soc_log"), "")
        self._soc_log_s = self.node.ROSParam[float](self.namespace("soc_log_s"), 60.0)
        self._idle_floor = self.node.ROSParam[float](self.namespace("idle_floor"), 0.0)
        self._one_need_charge = self.node.ROSParam[bool](self.namespace("one_need_charge"), False)
        self._charge_at_start = self.node.ROSParam[bool](self.namespace("charge_at_start"), False)
        self._stop_empty = self.node.ROSParam[bool](self.namespace("stop_empty"), False)
        self._no_power_s = self.node.ROSParam[float](self.namespace("no_power_s"), 120.0)
        self._no_power_rise = self.node.ROSParam[float](self.namespace("no_power_rise"), 0.1)

        self._chargers: list[str] = []
        self._xy: dict[str, tuple[float, float]] = {}
        self._dock_area: dict[str, shapely.Polygon] = {}  # charger -> its dock polygon, where the runtime charges a robot
        self._no_power_at: dict[str, set[str]] = {}  # robot -> the chargers where a charge of it ended no_power, until one of its charges gets power
        self._stopped: dict[str, int] = {}  # robot stopped at 0 % (stop_empty) -> its index in the empty table
        self._in_dock_logged: set[str] = set()  # stopped robots whose stand in a charger's dock is logged
        self._docked_watch: dict[str, tuple[float, float]] = {}  # robot parked on its charger home under charge_until -> since when and lowest charge its charge has not risen
        self._home: dict[str, str] = {}
        self._spot: dict[str, str] = {}
        self._arrived: set[str] = set()
        self._state: dict[str, _State] = {}
        self._run: dict[str, _Run] = {}
        self._docks: dict[str, str] = {}
        self._dock_queue: dict[str, list[str]] = {}  # dock -> robots holding in place before it, longest waiting first
        self._queue: deque[Job] = deque()
        self._charge_queue: dict[str, tuple[int, float]] = {}  # robot -> its charge's log index, when it joined the queue
        self._charge: dict[str, _Charge] = {}
        self._charges = itertools.count()
        self._top_ups = itertools.count()
        self._topping: dict[str, str] = {}  # robot -> the charger it holds by a top-up, until it has charged there to charge_until
        self._empty: set[str] = set()
        self._empties = itertools.count()
        self._tracks: dict[str, deque[tuple[float, float, float]]] = {}
        self._home_tracks: set[str] = set()  # robots whose track in `_tracks` is of a trip home
        self._overdue: dict[str, tuple[tuple[int, int], deque[tuple[float, float, float]]]] = {}  # robot past its leg's timeout -> (job, leg), its positions since
        self._jams = itertools.count()
        self._log: dict[str, str] = {}
        self._log_changed = False
        self._jobs: Iterator[Job] = iter(())
        self._next = Job(0, "delivery", math.inf)  # nothing is released before the first reset
        self._t0 = 0.0
        self._soc_next = math.inf  # when the next line of soc_log is due
        self._need_for: dict[int, str] = {}  # job index -> the robot that charges because no free robot could take it (one_need_charge)
        self._start_check: set[str] = set()  # robots whose start charge charge_at_start has not looked at yet

    async def reset(self) -> ResetPlan:
        plan = await super().reset()
        robots = list(self._ctx.robots.values())
        homes = list(self._homes.value)
        if len(homes) < len(robots):
            raise ValueError(f"task.fleet.homes names {len(homes)} spots for {len(robots)} robots")
        repeated = sorted({home for home in homes if homes.count(home) > 1})
        if repeated:
            raise ValueError(f"task.fleet.homes names {repeated} more than once, a spot is the home of one robot")
        entries = list(self._job_list.value or [])
        lockers = list(dict.fromkeys(self._lockers.value or []))
        listed_jobs = listed(entries)
        if lockers or entries:
            # several lockers: every one a dock, with the mode's one locker where it is named too
            lockers = list(dict.fromkeys([*([self._locker.value] if self._locker.value else []), *lockers]))
            strays = sorted({job.locker for job in listed_jobs} - set(lockers))
            if strays:
                raise ValueError(f"task.fleet.jobs sends jobs to {strays}, which task.fleet.lockers does not name")
            docks: tuple[str, ...] = (self._loading.value, *lockers)
        else:
            docks = (self._loading.value, self._locker.value)
        if self._charge_rule.value not in CHARGE_RULES or self._charge_jam.value not in CHARGE_JAMS:
            raise ValueError(f"task.fleet.charge_rule is one of {', '.join(CHARGE_RULES)} and task.fleet.charge_jam one of {', '.join(CHARGE_JAMS)}, not {self._charge_rule.value!r} and {self._charge_jam.value!r}")
        if set(docks) & set(homes):
            raise ValueError(f"task.fleet.homes names {sorted(set(docks) & set(homes))}, where jobs load and unload")
        if self._charge_below.value > self._charge_until.value:
            raise ValueError(f"task.fleet.charge_below ({self._charge_below.value}) is above task.fleet.charge_until ({self._charge_until.value})")
        world = authored_statics(self._ctx.world_manager.world_compacted())
        spots = set(world.zone_ref_names())
        unknown = [spot for spot in (*homes, *docks) if spot not in spots]
        if unknown:
            raise ValueError(f"task.fleet names {unknown}, which are no goto target of the world")

        self._chargers = [entity.name for entity in world.objects if entity.charge_power_w is not None]
        self._xy, self._dock_area = {}, {}
        for spot in dict.fromkeys((*homes, *docks, *self._chargers)):
            corners = world.lookup_zone_polygon(spot)
            assert corners is not None, spot
            polygon = shapely.Polygon([(corner.x, corner.y) for corner in corners])
            centre = polygon.centroid
            self._xy[spot] = (centre.x, centre.y)
            if spot in self._chargers:
                self._dock_area[spot] = polygon
        self._no_power_at, self._stopped, self._in_dock_logged, self._docked_watch = {}, {}, set(), {}
        self._home = {robot.name: home for robot, home in zip(robots, homes, strict=False)}
        self._spot = dict(self._home)
        self._arrived = set(self._home)
        self._state = dict.fromkeys(self._home, _State.PARKED)
        self._run, self._docks, self._dock_queue, self._queue = {}, {}, {}, deque()
        self._charge_queue, self._charge, self._charges, self._top_ups, self._topping = {}, {}, itertools.count(), itertools.count(), {}
        self._empty, self._empties = set(), itertools.count()
        self._tracks, self._home_tracks, self._jams = {}, set(), itertools.count()
        self._overdue = {}
        self._need_for = {}
        self._start_check = {name for name, home in self._home.items() if home in self._chargers} if self._charge_at_start.value else set()
        for robot in robots:
            start = robot.resolve_target(GoToPhase(target=self._home[robot.name])).pose
            assert start is not None, f"{robot.name}'s home {self._home[robot.name]!r} has no dispatch pose"
            plan.robots[robot.name] = RobotReset(start_pose=start)
        self._jobs = _then_none(listed_jobs) if entries else releases(self._ctx.conf.RNG.stream("robots", "fleet"), self._rate.value, self._pickup_share.value)
        self._next = next(self._jobs)
        self._t0 = self.node.sim_time.to_seconds()
        self._log = {}
        self._note("t0", f"{self._t0:.3f}")
        self._soc_next = math.inf
        if self._soc_log.value:
            path = Path(self._soc_log.value)
            if path.is_file() and path.stat().st_size > 0:
                raise ValueError(f"task.fleet.soc_log {path} exists and is not empty: a charge log is written once, name another file")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as out:
                out.write(",".join(["t_s", *(f"soc:{name}" for name in self._home), *(f"home:{name}" for name in self._home)]) + "\n")
            self._soc_next = 0.0
        self._publish_log()
        return plan

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        """Release due jobs; follow each robot's leg, charge, timeout and jam; let robots that reach a dock's approach take it or hold in place before it, longest waiting first; give queued jobs to free robots, fullest first, send robots to chargers, lowest first, top up free robots at chargers whose holders are away, and send the rest home."""
        now = self.node.sim_time.to_seconds() - self._t0
        while self._next.release_s <= now:
            job = self._next
            self._logger.info(f"released {job.kind} {job.index} at {job.release_s:.1f} s")
            self._note(f"job.{job.index}.kind", job.kind)
            self._note(f"job.{job.index}.released", _s(job.release_s))
            if job.locker:
                self._note(f"job.{job.index}.locker", job.locker)
            if job.entrance:
                self._note(f"job.{job.index}.entrance", job.entrance)
            self._queue.append(job)
            self._next = next(self._jobs)

        for name in assigned:
            if name in self._state:
                self._hand_outside(name, now)
        await self._watch_batteries(now)
        self._watch_empty_docks(now)
        self._log_charge(now)
        self._check_start(now)
        self._watch_docked(now)

        for name, state in list(self._state.items()):
            last = await self._ended(name, ended)
            if state is _State.OUTSIDE:
                if await self._ctx.robots[name].is_done:
                    self._state[name] = _State.FREE
            elif state is _State.PARKED:
                if last is not None and last.outcome == "met":
                    self._arrived.add(name)
                elif last is not None:
                    self._state[name] = _State.FREE
            elif state in (_State.JOB, _State.DOCK_WAIT):
                await self._follow_job(name, state, last, now)
            elif state is _State.CHARGE:
                await self._follow_charge(name, last, now)
            elif state is _State.EMPTY and last is not None:
                # its hold in place ended: it holds on
                await self._submit(name, [GoToPhase(hold_time=WAIT_HOLD_S, text="battery empty: stopped")])
        await self._watch_jams(now)
        await self._watch_docks(now)

        await self._hand_out_jobs(now)
        await self._send_to_chargers(now)
        await self._top_up_robots(now)

        for name, state in self._state.items():
            if state is _State.FREE and await self._submit(name, [GoToPhase(target=self._home[name], text=f"park at {self._home[name]}")]):
                self._state[name] = _State.PARKED
                self._arrived.discard(name)
        self._publish_log()

    @property
    async def done(self) -> bool:
        return False

    async def teardown(self) -> None:
        self._log = {}
        self._ctx.episode.publish_task_entity(LOG_KIND, {})
        await super().teardown()

    # JOBS

    def _docks_of(self, job: Job) -> tuple[str, str]:
        """Where the job loads, then unloads."""
        locker = job.locker or self._locker.value
        return (self._loading.value, locker) if job.kind == "delivery" else (locker, self._loading.value)

    def _leg_phase(self, run: _Run) -> TaskPhase:
        job, dock = run.job, run.docks[run.leg]
        if run.leg == 0:
            return GoToPhase(target=dock, hold_time=self._load_s.value, on_failure="stop_task", text=f"{job.kind} {job.index}: load at {dock}, released {job.release_s:.1f} s")
        return GoToPhase(target=dock, hold_time=self._unload_s.value, on_failure="stop_task", text=f"{job.kind} {job.index}: unload at {dock}")

    async def _start_leg(self, name: str, now: float, *, retry: bool = False) -> bool:
        """Send the robot on its run's current leg, to the leg's dock, which it holds only once it is within `approach_m`
        of it (`_watch_docks`); the leg's timer starts (a retry keeps the timer of its first attempt), with a timeout from
        the leg's route estimate. False when the leg could not be submitted: the run is then not dispatched."""
        run = self._run[name]
        start = self._where(name)
        if not await self._submit(name, [self._leg_phase(run)]):
            run.dispatched = False
            return False
        dock = run.docks[run.leg]
        if not retry:
            run.leg_started_s, run.leg_waited_s = now, 0.0
        run.dispatched, run.reserved, run.waiting_since = True, False, None
        run.leg_timeout_s = self._timeout(math.dist(start, self._xy[dock]) * self._detour.value)
        self._note(f"job.{run.job.index}.timeout_{run.leg}", _s(run.leg_timeout_s))
        self._state[name] = _State.JOB
        return True

    async def _watch_docks(self, now: float) -> None:
        """Give each dock no robot holds to the robot holding in place before it longest, which drives on; then let every
        robot on a leg that came within `approach_m` of its dock take it if it is free and nobody waits for it, else hold
        in place and queue for it. A robot in the walkway before a busy dock is part of the traffic the study measures."""
        for dock, waiting in list(self._dock_queue.items()):
            while waiting and dock not in self._docks:
                name = waiting[0]
                run = self._run.get(name)
                if run is None or self._state.get(name) is not _State.DOCK_WAIT or run.docks[run.leg] != dock:
                    waiting.pop(0)
                    continue
                if not await self._submit(name, [self._leg_phase(run)]):
                    break
                waiting.pop(0)
                waited = now - typing.cast(float, run.waiting_since)
                run.leg_waited_s += waited
                run.waited_s += waited
                run.waiting_since, run.reserved = None, True
                self._docks[dock] = name
                self._state[name] = _State.JOB
                self._tracks.pop(name, None)
                self._logger.info(f"{name} takes {dock} after {waited:.1f} s, {now:.1f} s")
            if not waiting:
                del self._dock_queue[dock]
        for name, state in list(self._state.items()):
            if state is not _State.JOB or self._run[name].reserved or not self._run[name].dispatched:
                continue
            run = self._run[name]
            dock = run.docks[run.leg]
            if math.dist(self._where(name), self._xy[dock]) > self._approach_m.value:
                continue
            if dock not in self._docks and not self._dock_queue.get(dock):
                self._docks[dock] = name
                run.reserved = True
            elif await self._submit(name, [GoToPhase(hold_time=WAIT_HOLD_S, text=f"{run.job.kind} {run.job.index}: wait for {dock}")]):
                run.waiting_since = now
                self._dock_queue.setdefault(dock, []).append(name)
                self._state[name] = _State.DOCK_WAIT
                self._tracks.pop(name, None)
                self._logger.info(f"{name} waits for {dock}, held by {self._docks.get(dock, 'a robot before it')}, {now:.1f} s")

    async def _hand_out_jobs(self, now: float) -> None:
        """Give each queued job, oldest first, to the fullest free robot whose battery the job would leave at `reserve` or more; a fuller robot it would leave below goes to charge first, unless it is charged to `charge_until` already. With `one_need_charge` only the fullest robot that cannot take the job goes to charge, once per job, and only when no free robot can take it; the job waits."""
        for job in list(self._queue):
            docks = self._docks_of(job)
            short: list[tuple[str, float, float]] = []
            for name in self._takers():
                energy_wh, left = self._job_energy(name, job, docks)
                soc = self._soc(name)
                if left is not None and left < self._reserve.value and soc is not None and soc < self._charge_until.value:
                    if self._one_need_charge.value:
                        short.append((name, energy_wh, left))
                        continue
                    self._logger.info(f"{name} needs {energy_wh:.2f} Wh for {job.kind} {job.index} and would keep {left:.1f} %, it charges first")
                    self._queue_charge(name, f"job {job.index}", now)
                    continue
                self._run[name] = _Run(job=job, docks=docks, assigned_s=now)
                if not await self._start_leg(name, now):
                    del self._run[name]
                    break
                self._queue.remove(job)
                self._logger.info(f"{name} takes {job.kind} {job.index}, {energy_wh:.2f} Wh, {now:.1f} s")
                self._note(f"job.{job.index}.robot", name)
                self._note(f"job.{job.index}.assigned", _s(now))
                self._note(f"job.{job.index}.energy_wh", f"{energy_wh:.3f}")
                self._withdraw_need(job.index, name, now)
                break
            else:
                if short and job.index not in self._need_for:
                    name, energy_wh, left = short[0]
                    self._logger.info(f"no free robot can take {job.kind} {job.index}: the fullest, {name}, needs {energy_wh:.2f} Wh and would keep {left:.1f} %, it charges first")
                    self._queue_charge(name, f"job {job.index}", now)
                    self._need_for[job.index] = name

    def _withdraw_need(self, index: int, taker: str, now: float) -> None:
        """Job `index` has been taken: a robot queued to charge for it that has not left yet stays free (unless it is at or
        under the idle floor), so no robot waits for a charger for a job another robot does."""
        name = self._need_for.pop(index, None)
        if name is None or name == taker or name not in self._charge_queue:
            return
        soc = self._soc(name)
        if self._idle_floor.value > 0.0 and soc is not None and soc <= self._idle_floor.value:
            return
        charge, _ = self._charge_queue.pop(name)
        self._no_power_at.pop(name, None)
        self._note(f"charge.{charge}.ended", _s(now))
        self._note(f"charge.{charge}.outcome", "withdrawn")
        self._note(f"charge.{charge}.reason", f"job {index} taken by {taker}")

    def _takers(self) -> list[str]:
        """Free robots that may take a job, fullest battery first, those whose charge is unknown last."""
        idle = [name for name, state in self._state.items() if state in (_State.PARKED, _State.FREE) and name not in self._charge_queue]
        if not self._by_need:
            idle = [name for name in idle if (soc := self._soc(name)) is None or soc >= self._charge_below.value]
        return sorted(idle, key=lambda name: (self._soc(name) is None, -(self._soc(name) or 0.0)))

    @property
    def _by_need(self) -> bool:
        """Whether robots charge by need alone (`charge_rule` need): no threshold, no top-ups."""
        return self._charge_rule.value == "need"

    def _free_charger(self, name: str, chargers: list[str]) -> str | None:
        """The charger of `chargers` the robot takes: the nearest to where it is with `nearest_charger`, else the first."""
        if not chargers:
            return None
        if not self._nearest_charger.value:
            return chargers[0]
        where = self._where(name)
        return min(chargers, key=lambda charger: math.dist(where, self._xy[charger]))

    def _job_route_m(self, name: str, docks: tuple[str, str]) -> float:
        """The route estimate of a job in m: the straight line from where the robot is to both docks, times `detour`."""
        return (math.dist(self._where(name), self._xy[docks[0]]) + math.dist(self._xy[docks[0]], self._xy[docks[1]])) * self._detour.value

    def _to_charger_m(self, dock: str) -> float:
        """The route estimate in m from `dock` to the nearest charger (straight line times `detour`), 0 without chargers."""
        return min((math.dist(self._xy[dock], self._xy[charger]) for charger in self._chargers), default=0.0) * self._detour.value

    def _timeout(self, route_m: float) -> float:
        """Seconds a job or a charging trip on a route estimate of `route_m` may take, 0 for none."""
        if self._job_timeout.value <= 0.0:
            return 0.0
        return self._job_timeout.value + self._job_timeout_per_m.value * route_m

    def _job_energy(self, name: str, job: Job, docks: tuple[str, str]) -> tuple[float, float | None]:
        """The Wh the job takes the robot, by the route estimate from where it is to both docks and on to the charger nearest the second dock (straight lines times `detour`) at `wh_per_m`, plus its standby power over the load and unload time, and the battery percent it would leave, None while its charge or capacity is unknown. The leg to a charger keeps the reserve for after the job: a robot that takes it can still reach a charger."""
        robot = self._ctx.robots[name]
        profile = robot.power_profile
        route_m = self._job_route_m(name, docks) + self._to_charger_m(docks[1])
        per_m = self._wh_per_m.value if self._wh_per_m.value > 0.0 else wh_per_m(profile)
        standby_w = float(profile.get("static_power_w", 0.0)) if profile is not None else 0.0
        energy_wh = route_m * per_m + standby_w * (self._load_s.value + self._unload_s.value) / 3600.0
        soc, capacity_wh = robot.soc, robot.battery_capacity_wh
        if soc is None or capacity_wh is None:
            return energy_wh, None
        return energy_wh, soc - 100.0 * energy_wh / capacity_wh

    def _where(self, name: str) -> tuple[float, float]:
        """The robot's position in the world frame, else the spot it was last sent to."""
        pose = self._ctx.robots[name].pose
        if pose is None:
            return self._xy[self._spot[name]]
        position = self._ctx.environment_manager.ezilear(pose).position
        return (position.x, position.y)

    async def _follow_job(self, name: str, state: _State, last: PhaseState | None, now: float) -> None:
        """A leg that ended met frees its dock: after loading the robot goes straight on to its second dock, after
        unloading the job is done; a leg that ended otherwise fails the job; a leg that drove longer than its timeout
        (its waits for the dock apart) is abandoned. A robot holding in place for a dock ignores its hold's end."""
        run = self._run[name]
        if state is _State.JOB and not run.dispatched:
            # the leg could not be submitted: the robot's last phase is the previous leg's, so its end is no news; send
            # the leg again until it goes out or its timer runs out
            if not await self._start_leg(name, now, retry=True) and run.leg_timeout_s > 0.0 and run.driving_s(now) > run.leg_timeout_s:
                self._end_job(name, "abandoned", "timeout", now)
            return
        if state is _State.JOB and last is not None:
            if last.outcome != "met":
                self._end_job(name, "failed", str(last.reason) or last.outcome, now)
                return
            self._release(run.docks[run.leg], name)
            if run.leg == 1:
                self._end_job(name, "completed", "", now)
                return
            self._note(f"job.{run.job.index}.loaded", _s(now))
            run.leg = 1
            if not await self._start_leg(name, now):
                # not dispatched: the next ticks send the leg again (above), its timer running from now
                run.leg_started_s, run.leg_waited_s, run.reserved, run.waiting_since = now, 0.0, False, None
                run.leg_timeout_s = self._timeout(math.dist(self._where(name), self._xy[run.docks[1]]) * self._detour.value)
            return
        if run.leg_timeout_s > 0.0 and run.driving_s(now) > run.leg_timeout_s and (self._leg_timeout_moving.value or self._stalled_at_dock(name, run, now)):
            self._end_job(name, "abandoned", "timeout", now)

    def _stalled_at_dock(self, name: str, run: _Run, now: float) -> bool:
        """With `leg_timeout_moving` false, whether a leg past its timeout ends: only once the robot has stood (moved less
        than `jam_progress_m` over `jam_window_s`, watched from when the timeout passed) within `jam_goal_m` of its dock,
        where the jam rule does not look (docking that does not finish). A robot that still moves drives on, and one that
        stands further from its dock is the jam rule's. Without a pose, or with the jam rule off, the timeout applies."""
        window = self._jam_window_s.value
        if window <= 0.0 or self._ctx.robots[name].pose is None:
            return True
        key = (run.job.index, run.leg)
        if name not in self._overdue or self._overdue[name][0] != key:
            self._overdue[name] = (key, deque())
            self._note(f"job.{run.job.index}.overdue_{run.leg}", _s(now))
        track = self._overdue[name][1]
        x, y = self._where(name)
        track.append((now, x, y))
        while len(track) > 1 and now - track[1][0] >= window:
            track.popleft()
        t0, x0, y0 = track[0]
        if math.dist((x, y), self._xy[run.docks[run.leg]]) > self._jam_goal_m.value:
            return False
        return now - t0 >= window and all(math.dist((x0, y0), (px, py)) < self._jam_progress_m.value for _, px, py in track)

    def _end_job(self, name: str, outcome: str, reason: str, now: float) -> None:
        """Close the robot's job as completed, failed or abandoned, release its docks and free the robot; a job that did not complete is dropped, not queued again."""
        run = self._run.pop(name)
        for dock in run.docks:
            self._release(dock, name)
            if name in self._dock_queue.get(dock, []):
                self._dock_queue[dock].remove(name)
        job = run.job
        waited = run.waited_s + (0.0 if run.waiting_since is None else now - run.waiting_since)
        self._note(f"job.{job.index}.wait_s", _s(waited))
        self._note(f"job.{job.index}.ended", _s(now))
        self._note(f"job.{job.index}.outcome", outcome)
        if reason:
            self._note(f"job.{job.index}.reason", reason)
        if outcome == "completed":
            self._logger.info(f"{name} finished {job.kind} {job.index}: released {job.release_s:.1f} s, done {now:.1f} s")
        else:
            self._logger.warning(f"{name} {outcome} {job.kind} {job.index} ({reason}): released {job.release_s:.1f} s, given up {now:.1f} s")
        self._state[name] = _State.FREE

    def _release(self, dock: str, name: str) -> None:
        if self._docks.get(dock) == name:
            del self._docks[dock]

    @staticmethod
    def _timed_out(since_s: float, timeout_s: float, now: float) -> bool:
        return timeout_s > 0.0 and now - since_s > timeout_s

    # CHARGING

    def _queue_charge(self, name: str, trigger: str, now: float) -> None:
        if name in self._charge_queue:
            return
        index = next(self._charges)
        self._charge_queue[name] = (index, now)
        self._note(f"charge.{index}.robot", name)
        self._note(f"charge.{index}.trigger", trigger)
        self._note(f"charge.{index}.queued", _s(now))
        if self._idle_floor.value > 0.0 or self._one_need_charge.value or self._charge_at_start.value:
            self._note(f"charge.{index}.soc_queued", _pct(self._soc(name)))

    async def _send_to_chargers(self, now: float) -> None:
        """Queue free robots under `charge_below`, then send queued robots to a charger, lowest battery first (ties by
        queue time, then name): a robot parked at its charger home charges there, any other takes a charger whose holder
        is away, else one whose holder yields (`_yields`: parked on it charged to `charge_until`, or holding it by a
        top-up), else the charger of a robot still on its way to top up there (first the one that took it from this
        robot), and swaps homes with the holder. A holder that yields leaves for its new home only once the robot holds
        in place before the charger (`_follow_yield`); a robot on its way to top up abandons the top-up and heads for its
        new home at once."""
        for name in [name for name, charger in self._topping.items() if (soc := self._soc(name)) is not None and soc >= self._charge_until.value]:
            # charged to charge_until at the charger it took by a top-up: it holds it like any holder from now on
            del self._topping[name]
        for name, state in self._state.items():
            soc = self._soc(name)
            if not self._by_need and state in (_State.PARKED, _State.FREE) and soc is not None and soc < self._charge_below.value:
                self._queue_charge(name, "charge_below", now)
            elif self._idle_floor.value > 0.0 and state in (_State.PARKED, _State.FREE) and soc is not None and soc <= self._idle_floor.value and not self._docked_home(name):
                self._queue_charge(name, "floor", now)
        for name in sorted(self._charge_queue, key=lambda name: (self._soc(name) or 0.0, self._charge_queue[name][1], name)):
            if self._state[name] not in (_State.PARKED, _State.FREE):
                continue
            home = self._home[name]
            # a charger the robot may not use now (`_usable`: a robot stopped at 0 % stands in its dock, or a charge of
            # this robot had no power there and another robot stands in the dock) is passed over, its home too
            usable = [charger for charger in self._chargers if self._usable(name, charger)]
            home_ok = home in usable
            if home_ok and self._state[name] is _State.PARKED and name in self._arrived:
                self._start_charge(name, home, now)
                charge = self._charge[name]
                charge.started_s = now
                self._note(f"{charge.table}.{charge.index}.started", _s(now))
                self._note(f"{charge.table}.{charge.index}.soc_start", _pct(self._soc(name)))
                charge.power_since, charge.power_soc = now, self._soc(name)
            else:
                charger = home if home_ok else self._free_charger(name, [charger for charger in usable if self._owner_away(charger)])
                if charger is None:
                    charger = self._free_charger(name, [charger for charger in usable if self._yields(charger)])
                if charger is None:
                    on_the_way = [charger for charger in usable if self._topping_up(charger) is not None]
                    charger = next((charger for charger in on_the_way if self._charge[typing.cast(str, self._holder(charger))].swapped_with == name), next(iter(on_the_way), None))
                if charger is None:
                    continue
                trip_m = math.dist(self._where(name), self._xy[charger]) * self._detour.value
                if not await self._submit(name, [GoToPhase(target=charger, on_failure="stop_task", text=f"charge at {charger}")]):
                    continue
                owner = self._holder(charger)
                leaving = owner if owner is not None and owner != name and self._yields(charger) else None
                turning = self._topping_up(charger) if owner != name else None
                if turning is not None:
                    # on its way to top up there: it gives the charger up at once and goes to the queued robot's home
                    self._note(f"top_up.{self._charge[turning].index}.yielded_to", name)
                    self._end_charge(turning, "abandoned", "yielded", now)
                    self._tracks.pop(turning, None)
                    self._logger.info(f"{turning} gives {charger} up on its way to top up there to {name} ({_pct(self._soc(name))} %) and takes {home}, {now:.1f} s")
                swapped_top_up = owner is not None and owner != name and self._topping.pop(owner, None) == charger
                if owner is not None and owner != name:
                    self._home[owner] = home
                self._home[name] = charger
                self._topping.pop(name, None)
                self._start_charge(name, charger, now, trip_m)
                charge = self._charge[name]
                charge.home_before, charge.swapped_with, charge.leaving = home, owner if owner != name else None, leaving
                charge.swapped_top_up = swapped_top_up
                if leaving is not None or turning is not None:
                    self._note(f"{charge.table}.{charge.index}.yielded_by", typing.cast(str, leaving or turning))
                if leaving is not None:
                    self._logger.info(f"{leaving} gives {charger} up to {name} ({_pct(self._soc(leaving))} %) and takes {home}, {now:.1f} s")
            self._logger.info(f"{name} charges at {self._home[name]} from {self._soc(name):.1f} %, {now:.1f} s")

    async def _top_up_robots(self, now: float) -> None:
        """Opportunistic charging: once no robot waits to charge, a free robot (parked or on its way home, not queued to
        charge) whose battery is below `charge_until` and whose home is no charger takes a charger whose holder is away on
        a job as its home, lowest battery first, and the holder gets its old home. The trip is a charging trip: the jam
        rule and the trip timeout apply, and only a jam gives the charger back (a timeout, a failed trip or a task from
        outside leave it the robot's home). Once docked the robot charges there and is free to take jobs. Until it has
        charged there to `charge_until` it gives the charger up to a robot queued to charge at once, and on its way there
        it gives it up too (`_send_to_chargers`). Off with `top_up`."""
        if self._by_need or not self._top_up.value or self._charge_queue:
            return
        takers = [name for name, state in self._state.items() if state is _State.PARKED and self._home[name] not in self._chargers and (soc := self._soc(name)) is not None and soc < self._charge_until.value]
        takers.sort(key=lambda name: (self._soc(name) or 0.0, name))
        for charger in self._chargers:
            holder = self._holder(charger)
            if not takers or holder is None or self._state[holder] not in (_State.JOB, _State.DOCK_WAIT) or self._blocked(charger):
                continue
            name = takers[0]
            home = self._home[name]
            trip_m = math.dist(self._where(name), self._xy[charger]) * self._detour.value
            if not await self._submit(name, [GoToPhase(target=charger, on_failure="stop_task", text=f"top up at {charger}")]):
                continue
            takers.pop(0)
            swapped_top_up = self._topping.pop(holder, None) == charger
            self._home[holder], self._home[name] = home, charger
            self._topping[name] = charger
            self._arrived.discard(name)
            self._start_charge(name, charger, now, trip_m, top_up=True)
            charge = self._charge[name]
            charge.home_before, charge.swapped_with, charge.swapped_top_up = home, holder, swapped_top_up
            self._logger.info(f"{name} tops up at {charger} from {self._soc(name):.1f} %, {holder} away on a job takes {home}, {now:.1f} s")

    def _start_charge(self, name: str, charger: str, now: float, trip_m: float = 0.0, *, top_up: bool = False) -> None:
        """Start the robot's queued charge at `charger`, or with `top_up` an opportunistic one that no queue holds, a trip
        of `trip_m` by the route estimate (0 at its charger home)."""
        if top_up:
            index, table = next(self._top_ups), "top_up"
            self._note(f"top_up.{index}.robot", name)
            self._note(f"top_up.{index}.soc_dispatched", _pct(self._soc(name)))
        else:
            (index, _), table = self._charge_queue.pop(name), "charge"
        charge = _Charge(index=index, charger=charger, dispatched_s=now, timeout_s=self._timeout(trip_m), table=table)
        self._charge[name] = charge
        self._state[name] = _State.CHARGE
        self._note(f"{charge.table}.{charge.index}.charger", charger)
        self._note(f"{charge.table}.{charge.index}.dispatched", _s(now))

    async def _follow_charge(self, name: str, last: PhaseState | None, now: float) -> None:
        charge = self._charge[name]
        if charge.started_s is None and (charge.leaving is not None or charge.holding_since is not None):
            await self._follow_yield(name, charge, now)
            if charge.holding_since is not None:
                # holding in place before the charger: its hold's end is no news; the trip's timeout still runs
                if self._timed_out(charge.dispatched_s, charge.timeout_s, now):
                    await self._revert_swap(name, charge)
                    self._end_charge(name, "abandoned", "timeout", now)
                return
        if charge.started_s is None:
            if last is not None and last.outcome == "met":
                charge.started_s = now
                self._arrived.add(name)
                self._note(f"{charge.table}.{charge.index}.started", _s(now))
                self._note(f"{charge.table}.{charge.index}.soc_start", _pct(self._soc(name)))
                charge.power_since, charge.power_soc = now, self._soc(name)
                if charge.table == "top_up":
                    # docked at its new charger home: it charges there and is free for jobs
                    self._end_charge(name, "docked", "", now)
                    self._state[name] = _State.PARKED
                    return
            elif last is not None:
                self._end_charge(name, "abandoned", str(last.reason) or last.outcome, now)
                return
            elif self._timed_out(charge.dispatched_s, charge.timeout_s, now):
                self._end_charge(name, "abandoned", "timeout", now)
                return
        soc = self._soc(name)
        if charge.started_s is not None and soc is not None and soc >= self._charge_until.value:
            self._logger.info(f"{name} charged to {soc:.1f} % at {self._home[name]}, {now:.1f} s")
            self._end_charge(name, "charged", "", now)
            self._state[name] = _State.PARKED
        elif charge.started_s is not None:
            self._watch_power(name, charge, soc, now)

    def _watch_power(self, name: str, charge: _Charge, soc: float | None, now: float) -> None:
        """A started charge whose robot's charge has not risen by `no_power_rise` over its lowest since the window started
        within `no_power_s` (the window starts at the start, then again at each such rise) gets no power: the runtime
        charges one robot per dock, the first parked in it, so a robot that stops in a dock another robot holds stands at
        0 W. The charge ends `no_power` and the robot queues to charge again at once (trigger `no_power`, still the robot
        charging for the jobs it charged for); it keeps the charger as its home, and while another robot stands in that
        charger's dock it does not take it again (`_barred`) until the charge queued now ends or gets power."""
        timeout = self._no_power_s.value
        if timeout <= 0.0 or soc is None or charge.power_since is None:
            return
        if charge.power_soc is None:
            # the charge was unknown when the charge started: the window starts at the first reading
            charge.power_since, charge.power_soc = now, soc
            return
        # the rise counts from the lowest charge since the window started: power that comes late (the other robot left the
        # dock) is seen once it has made up the drain before it
        charge.power_soc = min(charge.power_soc, soc)
        if soc >= charge.power_soc + self._no_power_rise.value:
            charge.power_since, charge.power_soc = now, soc
            self._no_power_at.pop(name, None)
            return
        if now - charge.power_since <= timeout:
            return
        rise = soc - charge.power_soc
        reason = f"charge rose {rise:.2f} points in {now - charge.power_since:.0f} s"
        # `_end_charge` unlinks the robot from the jobs it charged for; the charge queued now is for them again
        jobs = [job for job, robot in self._need_for.items() if robot == name]
        self._no_power_at.setdefault(name, set()).add(charge.charger)
        self._end_charge(name, "no_power", reason, now)
        self._queue_charge(name, "no_power", now)
        for job in jobs:
            self._need_for[job] = name
        others = [other for other in self._state if other != name and self._in_dock(other, charge.charger)]
        self._logger.warning(f"{name} got no power at {charge.charger} ({reason}; {', '.join(others) or 'no other robot'} in its dock), queued to charge again, {now:.1f} s")

    async def _follow_yield(self, name: str, charge: _Charge, now: float) -> None:
        """A swap with a charged holder (`charge.leaving`): the robot drives to within `approach_m` of the charger and
        holds in place there; then the holder leaves for its new home, past the holding robot, so the two pass by the
        charger and not head-on somewhere along the robot's way; once the holder is clear of the charger, the robot drives
        on to it."""
        charger, holder = charge.charger, charge.leaving
        if holder is not None and self._cleared(holder, charger):
            self._note(f"{charge.table}.{charge.index}.holder_left", _s(now))
            charge.leaving = holder = None
        if holder is None:
            if charge.holding_since is not None and await self._submit(name, [GoToPhase(target=charger, on_failure="stop_task", text=f"charge at {charger}")]):
                self._note(f"{charge.table}.{charge.index}.held_s", _s(now - charge.holding_since))
                charge.holding_since = None
                self._tracks.pop(name, None)
            return
        if charge.holding_since is None:
            if math.dist(self._where(name), self._xy[charger]) > self._approach_m.value:
                return
            if not await self._submit(name, [GoToPhase(hold_time=WAIT_HOLD_S, text=f"wait for {charger}, {holder} leaving it")]):
                return
            charge.holding_since = now
            self._tracks.pop(name, None)
        # the holder leaves once the robot holds before the charger; a holder whose park could not be submitted tries again
        if self._state[holder] is _State.PARKED and holder in self._arrived and await self._submit(holder, [GoToPhase(target=self._home[holder], text=f"park at {self._home[holder]}")]):
            self._arrived.discard(holder)
            self._logger.info(f"{holder} leaves {charger} for {self._home[holder]}, {name} holding before it, {now:.1f} s")

    def _cleared(self, holder: str, charger: str) -> bool:
        """Whether a holder that gave its charger up is out of the way: off on a job, a charge or a task from outside, at
        its new home, or further than `approach_m` from the charger."""
        if self._state[holder] not in (_State.PARKED, _State.FREE):
            return True
        where = self._where(holder)
        return math.dist(where, self._xy[charger]) > self._approach_m.value or math.dist(where, self._xy[self._home[holder]]) <= self._jam_goal_m.value

    async def _revert_swap(self, name: str, charge: _Charge) -> None:
        """Give the charger of a trip that did not arrive back to its holder and the robot its home before the trip; a
        holder that has taken another home since (a charger it went to itself) leaves it to the robot that has the old
        home now, which gets the charger, so every home keeps one robot. A parked holder that is not on the charger (it
        left, or was parked elsewhere) is sent back to it, parked on its way, so the charger is not away for the next
        queued robot."""
        if charge.home_before is None or self._home[name] != charge.charger:
            return
        holder, top_up = charge.swapped_with, charge.swapped_top_up
        self._topping.pop(name, None)
        if holder is not None and self._home.get(holder) != charge.home_before:
            holder, top_up = self._holder(charge.home_before), False
        if holder is not None:
            self._home[holder] = charge.charger
            if top_up:
                self._topping[holder] = charge.charger
            if self._state[holder] is _State.PARKED and (holder not in self._arrived or math.dist(self._where(holder), self._xy[charge.charger]) > self._jam_goal_m.value):
                self._arrived.discard(holder)
                if not await self._submit(holder, [GoToPhase(target=charge.charger, text=f"park at {charge.charger}")]):
                    self._state[holder] = _State.FREE
        self._home[name] = charge.home_before

    def _end_charge(self, name: str, outcome: str, reason: str, now: float) -> None:
        charge = self._charge.pop(name)
        if outcome != "no_power":
            # a no-power record lasts until the charge queued after it ends, however it ends
            self._no_power_at.pop(name, None)
        for job, robot in list(self._need_for.items()):
            if robot == name:
                del self._need_for[job]
        self._note(f"{charge.table}.{charge.index}.ended", _s(now))
        self._note(f"{charge.table}.{charge.index}.outcome", outcome)
        self._note(f"{charge.table}.{charge.index}.soc_end", _pct(self._soc(name)))
        if reason:
            self._note(f"{charge.table}.{charge.index}.reason", reason)
            if outcome != "no_power":  # `_watch_power` logs its own line
                self._logger.warning(f"{name} {outcome} its charge at {charge.charger} ({reason}), {now:.1f} s")
        self._state[name] = _State.FREE

    def _docked_home(self, name: str) -> bool:
        """Whether the robot is parked on its charger home, where the dock charges it (not while a robot stopped at 0 % stands
        in that dock: then the dock may charge the stopped robot)."""
        return self._home[name] in self._chargers and self._state[name] is _State.PARKED and name in self._arrived and not self._blocked(self._home[name])

    def _in_dock(self, name: str, charger: str) -> bool:
        """Whether the robot's centre is in the charger's dock polygon, where the runtime charges a robot parked there; False
        while its pose is unknown (the runtime leaves such a robot out of its charging too), not the spot it was sent to."""
        if self._ctx.robots[name].pose is None:
            return False
        return bool(self._dock_area[charger].covers(shapely.Point(self._where(name))))

    def _blocked(self, charger: str) -> bool:
        """Whether a robot stopped at 0 % (`_stop`) stands in the charger's dock: the runtime charges the first robot parked
        in a dock, so another robot sent there would get no power. Such a charger is free for nobody."""
        return any(state is _State.EMPTY and self._in_dock(name, charger) for name, state in self._state.items())

    def _barred(self, name: str, charger: str) -> bool:
        """Whether a charge of the robot ended `no_power` at the charger (since its last charge that got power) and
        another robot stands in that charger's dock: the robot does not take the charger again until that robot has left
        it, so it is not sent back into a dock the runtime gives to another robot."""
        return charger in self._no_power_at.get(name, ()) and any(other != name and self._in_dock(other, charger) for other in self._state)

    def _usable(self, name: str, charger: str) -> bool:
        """Whether the robot may charge at the charger now: neither `_blocked` nor `_barred` for it."""
        return not self._blocked(charger) and not self._barred(name, charger)

    def _watch_docked(self, now: float) -> None:
        """A robot parked on its charger home (`_docked_home`) under `charge_until`, not queued to charge, is charged by its
        dock without a charge of the mode, so `_watch_power` does not look at it. If its charge has not risen by
        `no_power_rise` over its lowest within `no_power_s` (the window starts again at each such rise), the dock does not
        charge it (another robot holds it in the runtime): it queues to charge (trigger `no_power`) with a no-power record
        for its home, so it takes another charger while another robot stands in that dock (`_barred`)."""
        timeout, until = self._no_power_s.value, self._charge_until.value
        for name in self._state:
            soc = self._soc(name)
            if timeout <= 0.0 or soc is None or soc >= until or name in self._charge_queue or not self._docked_home(name):
                self._docked_watch.pop(name, None)
                continue
            since, lowest = self._docked_watch.get(name, (now, soc))
            lowest = min(lowest, soc)
            if soc >= lowest + self._no_power_rise.value:
                since, lowest = now, soc
            if now - since <= timeout:
                self._docked_watch[name] = (since, lowest)
                continue
            del self._docked_watch[name]
            home = self._home[name]
            reason = f"charge rose {soc - lowest:.2f} points in {now - since:.0f} s"
            self._no_power_at.setdefault(name, set()).add(home)
            self._queue_charge(name, "no_power", now)
            others = [other for other in self._state if other != name and self._in_dock(other, home)]
            self._logger.warning(f"{name} parked on its charger home {home} at {soc:.1f} % gets no power there ({reason}; {', '.join(others) or 'no other robot'} in its dock), queued to charge, {now:.1f} s")

    def _watch_empty_docks(self, now: float) -> None:
        """Log once for each robot stopped at 0 % that stands in a charger's dock (`empty.N.in_dock`): it blocks that
        charger (`_blocked`) while it stands there."""
        for name, index in self._stopped.items():
            if name in self._in_dock_logged:
                continue
            docks = [charger for charger in self._chargers if self._in_dock(name, charger)]
            if docks:
                self._in_dock_logged.add(name)
                self._note(f"empty.{index}.in_dock", ",".join(docks))
                self._logger.warning(f"{name}: stopped at 0 % in the dock of {', '.join(docks)}, which no robot is sent to while it stands there, {now:.1f} s")

    def _check_start(self, now: float) -> None:
        """`charge_at_start`: a robot whose home is a charger and whose start charge is under `charge_until` charges there
        before it takes a job (looked at once per robot, at the first tick its charge is known)."""
        for name in sorted(self._start_check):
            soc = self._soc(name)
            if soc is None:
                continue
            self._start_check.discard(name)
            if soc < self._charge_until.value and self._docked_home(name):
                self._queue_charge(name, "start", now)

    def _holder(self, spot: str) -> str | None:
        return next((name for name, home in self._home.items() if home == spot), None)

    def _owner_away(self, spot: str) -> bool:
        """Whether the spot's holder is away from it (or it has none). A holder stopped at 0 % counts as away: its charger is
        free for others, unless a robot stopped at 0 % stands in its dock (`_blocked`, which `_send_to_chargers` applies)."""
        holder = self._holder(spot)
        return holder is None or self._state[holder] in (_State.JOB, _State.DOCK_WAIT, _State.FREE, _State.EMPTY)

    def _yields(self, charger: str) -> bool:
        """Whether the charger's holder gives it up to a queued robot: parked on it (arrived), not queued itself, and charged
        to `charge_until`, or holding it by a top-up (it took the charger by a top-up and has not charged there to
        `charge_until` since)."""
        holder = self._holder(charger)
        if holder is None or self._state[holder] is not _State.PARKED or holder not in self._arrived or holder in self._charge_queue:
            return False
        soc = self._soc(holder)
        return soc is not None and (soc >= self._charge_until.value or self._topping.get(holder) == charger)

    def _topping_up(self, charger: str) -> str | None:
        """The charger's holder while it is on its way there to top up (not docked yet), else None."""
        holder = self._holder(charger)
        if holder is None or self._state[holder] is not _State.CHARGE:
            return None
        charge = self._charge[holder]
        return holder if charge.table == "top_up" and charge.started_s is None and charge.charger == charger else None

    def _soc(self, name: str) -> float | None:
        return self._ctx.robots[name].soc

    async def _watch_batteries(self, now: float) -> None:
        """Log and count each time a robot's battery reaches 0 %; with `stop_empty` the robot stops there for the rest of
        the episode (`_stop`)."""
        for name in list(self._state):
            soc = self._soc(name)
            if soc is None:
                continue
            if soc <= 0.0 and name not in self._empty:
                self._empty.add(name)
                index = next(self._empties)
                self._logger.warning(f"{name}: battery empty at {now:.1f} s, {index + 1} this episode")
                self._note(f"empty.{index}.robot", name)
                self._note(f"empty.{index}.at", _s(now))
                if self._stop_empty.value and self._state[name] is not _State.EMPTY:
                    await self._stop(name, index, now)
            elif soc > 0.0 and self._state[name] is not _State.EMPTY:
                self._empty.discard(name)

    async def _stop(self, name: str, index: int, now: float) -> None:
        """A robot at 0 %: its job is abandoned (reason empty), its charge or its place in the charge queue too; it holds in
        place where it is, an obstacle in the walkway, gets no job or trip, and its home is free for others, unless it
        stands in a charger's dock: that charger is then free for nobody (`_blocked`)."""
        self._stopped[name] = index
        state = self._state[name]
        if state in (_State.JOB, _State.DOCK_WAIT):
            self._note(f"empty.{index}.job", str(self._run[name].job.index))
            self._end_job(name, "abandoned", "empty", now)
        elif state is _State.CHARGE:
            charge = self._charge[name]
            self._note(f"empty.{index}.{charge.table}", str(charge.index))
            await self._revert_swap(name, charge)
            self._end_charge(name, "abandoned", "empty", now)
        if name in self._charge_queue:
            charge_index, _ = self._charge_queue.pop(name)
            self._note(f"charge.{charge_index}.ended", _s(now))
            self._note(f"charge.{charge_index}.outcome", "abandoned")
            self._note(f"charge.{charge_index}.reason", "empty")
        for job, robot in list(self._need_for.items()):
            if robot == name:
                del self._need_for[job]
        self._tracks.pop(name, None)
        self._state[name] = _State.EMPTY
        await self._ctx.robots[name].reset_navigation()
        await self._submit(name, [GoToPhase(hold_time=WAIT_HOLD_S, text="battery empty: stopped")])
        self._logger.warning(f"{name}: battery empty, stopped where it is for the rest of the episode, {now:.1f} s")

    def _log_charge(self, now: float) -> None:
        """Append the robots' battery charge and homes to `soc_log` once every `soc_log_s` seconds."""
        if now < self._soc_next:
            return
        self._soc_next += self._soc_log_s.value * (1 + math.floor((now - self._soc_next) / self._soc_log_s.value))
        socs = [self._soc(name) for name in self._home]
        with Path(self._soc_log.value).open("a") as out:
            out.write(",".join([_s(now), *("" if soc is None else f"{soc:.3f}" for soc in socs), *self._home.values()]) + "\n")

    # JAMS

    async def _watch_jams(self, now: float) -> None:
        """§8's jam rule: a robot on a job leg or a charging trip that moved less than `jam_progress_m` from where the
        last `jam_window_s` started fails its job (reason `jam`) or abandons its charge, and is reset to its home. Within
        `jam_goal_m` of its target it is docking or holding, and its window starts again. Each jam is logged with the
        distance to the nearest other robot, so robot-robot jams can be told apart. With `jam_home` a robot on its way
        home (parked, not arrived: after a job, a charge or a jam, or a holder leaving its charger) is watched the same
        way, its window started when the trip started; a jam there fails nothing and resets it to its home too."""
        window = self._jam_window_s.value
        for name, state in list(self._state.items()):
            target = self._jam_target(name, state)
            home_trip = target is not None and state is _State.PARKED
            if home_trip != (name in self._home_tracks):
                # a trip home after a job leg or a charging trip, or a leg after a trip home: the window starts again
                self._tracks.pop(name, None)
                if home_trip:
                    self._home_tracks.add(name)
                else:
                    self._home_tracks.discard(name)
            if window <= 0.0 or target is None or self._ctx.robots[name].pose is None:
                self._tracks.pop(name, None)
                continue
            x, y = self._where(name)
            if math.dist((x, y), self._xy[target]) <= self._jam_goal_m.value:
                self._tracks.pop(name, None)
                continue
            track = self._tracks.setdefault(name, deque())
            track.append((now, x, y))
            while len(track) > 1 and now - track[1][0] >= window:
                track.popleft()
            t0, x0, y0 = track[0]
            if now - t0 < window or any(math.dist((x0, y0), (px, py)) >= self._jam_progress_m.value for _, px, py in track):
                continue
            await self._unjam(name, state, target, now)

    def _jam_target(self, name: str, state: _State) -> str | None:
        """Where the robot is headed on a job leg, a charging trip or (with `jam_home`) a trip home, None otherwise: not
        for a robot holding before a dock or a charger, parked at its home, running a task from outside or stopped at 0 %."""
        if state is _State.JOB:
            run = self._run[name]
            return run.docks[run.leg] if run.dispatched else None
        if state is _State.CHARGE and self._charge[name].started_s is None and self._charge[name].holding_since is None:
            return self._charge[name].charger
        if state is _State.PARKED and self._jam_home.value and name not in self._arrived:
            return self._home[name]
        return None

    async def _unjam(self, name: str, state: _State, target: str, now: float) -> None:
        index = next(self._jams)
        others = [math.dist(self._where(name), self._where(other)) for other in self._state if other != name and self._ctx.robots[other].pose is not None]
        self._note(f"jam.{index}.robot", name)
        self._note(f"jam.{index}.at", _s(now))
        self._note(f"jam.{index}.target", target)
        if others:
            self._note(f"jam.{index}.robot_m", f"{min(others):.2f}")
        if state is _State.PARKED:
            await self._unjam_home(name, index, target, now)
            return
        if state is _State.JOB:
            self._note(f"jam.{index}.job", str(self._run[name].job.index))
            self._end_job(name, "failed", "jam", now)
        else:
            charge = self._charge[name]
            self._note(f"jam.{index}.{charge.table}", str(charge.index))
            if self._charge_jam.value == "retry":
                # the robot stays where it is and goes on to the same charger: no reset to a far home, which would
                # make it drive the whole way again; nav2's costmap is cleared of what it marked while it stalled
                self._note(f"jam.{index}.action", "retry")
                self._tracks.pop(name, None)
                await self._ctx.robots[name].reset_navigation()
                await self._submit(name, [GoToPhase(target=charge.charger, on_failure="stop_task", text=f"{'top up' if charge.table == 'top_up' else 'charge'} at {charge.charger}")])
                self._logger.warning(f"{name} jammed on its way to {target}: under {self._jam_progress_m.value:g} m in {self._jam_window_s.value:g} s, sent on to it, {now:.1f} s")
                return
            # §8: reset to its bay, the home it had before the trip; the charger goes back to its holder
            await self._revert_swap(name, charge)
            self._end_charge(name, "abandoned", "jam", now)
        self._tracks.pop(name, None)
        robot = self._ctx.robots[name]
        home = robot.resolve_target(GoToPhase(target=self._home[name])).pose
        if home is not None:
            await robot.move(home)
            # nav2 keeps its global costmap through a teleport: obstacles marked where the robot stalled would close its
            # way for every goal from home on
            await robot.reset_navigation()
        self._logger.warning(f"{name} jammed on its way to {target}: under {self._jam_progress_m.value:g} m in {self._jam_window_s.value:g} s, reset to {self._home[name]}, {now:.1f} s")

    async def _unjam_home(self, name: str, index: int, target: str, now: float) -> None:
        """A robot jammed on its way home (`jam.<k>.trip home`): no job or charge ends; it is teleported to its home and
        its navigation is reset, and its park there goes on (the runtime sends the goal again after the teleport). Where
        its home has no free cell for it (another robot stands there), only its navigation is reset where it stands
        (`jam.<k>.action reset`): nav2's costmaps keep what it marked while it stalled, which a goal sent again does not
        clear."""
        self._note(f"jam.{index}.trip", "home")
        self._tracks.pop(name, None)
        robot = self._ctx.robots[name]
        try:
            home = robot.resolve_target(GoToPhase(target=target)).pose
        except ValueError as e:
            self._logger.debug(f"{name}: {e}")
            home = None
        if home is not None:
            await robot.move(home)
        else:
            self._note(f"jam.{index}.action", "reset")
        await robot.reset_navigation()
        done = f"reset to {target}" if home is not None else "its navigation reset where it stands"
        self._logger.warning(f"{name} jammed on its way home to {target}: under {self._jam_progress_m.value:g} m in {self._jam_window_s.value:g} s, {done}, {now:.1f} s")

    # OUTSIDE TASKS AND PLUMBING

    def _hand_outside(self, name: str, now: float) -> None:
        """A task assigned from outside replaced whatever the robot did: its job is abandoned, its charge too, and the mode leaves it alone until that task is done."""
        state = self._state[name]
        if state in (_State.JOB, _State.DOCK_WAIT):
            self._end_job(name, "abandoned", "assigned from outside", now)
        elif state is _State.CHARGE:
            self._end_charge(name, "abandoned", "assigned from outside", now)
        self._charge_queue.pop(name, None)
        self._no_power_at.pop(name, None)
        self._arrived.discard(name)
        self._state[name] = _State.OUTSIDE
        self._logger.info(f"{name} runs a task assigned from outside, {now:.1f} s")

    async def _ended(self, name: str, ended: Mapping[str, PhaseState]) -> PhaseState | None:
        """The phase the robot ended on this tick, else, once its task is done, its last phase, for an end on a tick this mode did not see; None while the task runs."""
        last = ended.get(name)
        if last is not None:
            return last
        robot = self._ctx.robots[name]
        if not await robot.is_done:
            return None
        phases = robot.runner.phases
        return phases[-1] if phases else None

    async def _submit(self, name: str, phases: list[TaskPhase]) -> bool:
        try:
            await self._ctx.robots[name].submit_task(TaskRequest(phases=phases))
        except ValueError as e:
            # the spot has no free cell while another robot stands in it, so the robot keeps its state until the next tick
            self._logger.debug(f"{name}: {e}")
            return False
        target = phases[0].target if isinstance(phases[0], GoToPhase) else None
        if target is not None:
            self._spot[name] = target
        return True

    def _note(self, field: str, value: str) -> None:
        self._log[field] = value
        self._log_changed = True

    def _publish_log(self) -> None:
        if self._log_changed:
            self._ctx.episode.publish_task_entity(LOG_KIND, self._log)
            self._log_changed = False


def _s(t: float) -> str:
    return f"{t:.3f}"


def _pct(soc: float | None) -> str:
    return "" if soc is None else f"{soc:.2f}"
