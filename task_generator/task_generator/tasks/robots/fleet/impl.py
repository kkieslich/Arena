import enum
import itertools
import math
import typing
from collections import deque
from collections.abc import Collection, Iterator, Mapping

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
#: mean speed and turn rate assumed while a robot drives, for the energy per metre derived from its power profile (Nav2 caps the jackal at 0.5 m/s and 1.5 rad/s)
DRIVE_SPEED_MPS = 0.4
DRIVE_TURN_RADPS = 0.2


@attrs.frozen
class Job:
    index: int
    kind: typing.Literal["delivery", "pickup"]
    release_s: float


class _State(enum.Enum):
    PARKED = "parked"  # idle, at its home or on the way there
    FREE = "free"  # idle, to be sent home on this tick
    JOB = "job"  # on a job leg, holding that leg's dock
    WAIT = "wait"  # loaded, waiting at its home for the dock it unloads at
    CHARGE = "charge"  # on the way to a charger, or charging in it
    OUTSIDE = "outside"  # running a task assigned from outside until it is done


@attrs.define
class _Run:
    """A job a robot has taken: the docks it loads and unloads at, the leg it is on, when it took the job and, while it waits for its second dock, since when."""

    job: Job
    docks: tuple[str, str]
    assigned_s: float
    leg: int = 0
    waiting_s: float | None = None
    sent_home: bool = False


@attrs.define
class _Charge:
    """One charge of one robot: its index in the task log, where and when it went, when it arrived."""

    index: int
    charger: str
    dispatched_s: float
    started_s: float | None = None


def releases(rng: np.random.Generator, rate_per_h: float, pickup_share: float) -> Iterator[Job]:
    """Jobs in release order, a Poisson stream drawn from rng alone."""
    release_s = 0.0
    for index in itertools.count():
        release_s += float(rng.exponential(3600.0 / rate_per_h))
        yield Job(index, "pickup" if rng.random() < pickup_share else "delivery", release_s)


def wh_per_m(profile: Mapping[str, float] | None) -> float:
    """Energy in Wh per metre a robot with this power profile uses while driving at DRIVE_SPEED_MPS and DRIVE_TURN_RADPS, standby included, 0 without a profile."""
    if profile is None:
        return 0.0
    power_w = float(profile.get("static_power_w", 0.0)) + float(profile.get("odometry_linear_w_per_mps", 0.0)) * DRIVE_SPEED_MPS + float(profile.get("odometry_angular_w_per_radps", 0.0)) * DRIVE_TURN_RADPS
    return power_w / DRIVE_SPEED_MPS / 3600.0


class TM_Fleet(TM_Robots):
    """Deliveries and pickups released at a fixed rate from the episode seed, handed to free robots one dock at a time, with charging trips below a battery level or before a job the battery would not last."""

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
        self._job_timeout = self.node.ROSParam[float](self.namespace("job_timeout"), 1200.0)

        self._chargers: list[str] = []
        self._xy: dict[str, tuple[float, float]] = {}
        self._home: dict[str, str] = {}
        self._spot: dict[str, str] = {}
        self._arrived: set[str] = set()
        self._state: dict[str, _State] = {}
        self._run: dict[str, _Run] = {}
        self._docks: dict[str, str] = {}
        self._queue: deque[Job] = deque()
        self._charge_queue: dict[str, tuple[int, float]] = {}  # robot -> its charge's log index, when it joined the queue
        self._charge: dict[str, _Charge] = {}
        self._charges = itertools.count()
        self._empty: set[str] = set()
        self._empties = itertools.count()
        self._log: dict[str, str] = {}
        self._log_changed = False
        self._jobs: Iterator[Job] = iter(())
        self._next = Job(0, "delivery", math.inf)  # nothing is released before the first reset
        self._t0 = 0.0

    async def reset(self) -> ResetPlan:
        plan = await super().reset()
        robots = list(self._ctx.robots.values())
        homes = list(self._homes.value)
        if len(homes) < len(robots):
            raise ValueError(f"task.fleet.homes names {len(homes)} spots for {len(robots)} robots")
        repeated = sorted({home for home in homes if homes.count(home) > 1})
        if repeated:
            raise ValueError(f"task.fleet.homes names {repeated} more than once, a spot is the home of one robot")
        docks = (self._loading.value, self._locker.value)
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
        self._xy = {}
        for spot in dict.fromkeys((*homes, *docks, *self._chargers)):
            corners = world.lookup_zone_polygon(spot)
            assert corners is not None, spot
            centre = shapely.Polygon([(corner.x, corner.y) for corner in corners]).centroid
            self._xy[spot] = (centre.x, centre.y)
        self._home = {robot.name: home for robot, home in zip(robots, homes, strict=False)}
        self._spot = dict(self._home)
        self._arrived = set(self._home)
        self._state = dict.fromkeys(self._home, _State.PARKED)
        self._run, self._docks, self._queue = {}, {}, deque()
        self._charge_queue, self._charge, self._charges = {}, {}, itertools.count()
        self._empty, self._empties = set(), itertools.count()
        for robot in robots:
            start = robot.resolve_target(GoToPhase(target=self._home[robot.name])).pose
            assert start is not None, f"{robot.name}'s home {self._home[robot.name]!r} has no dispatch pose"
            plan.robots[robot.name] = RobotReset(start_pose=start)
        self._jobs = releases(self._ctx.conf.RNG.stream("robots", "fleet"), self._rate.value, self._pickup_share.value)
        self._next = next(self._jobs)
        self._t0 = self.node.sim_time.to_seconds()
        self._log = {}
        self._note("t0", f"{self._t0:.3f}")
        self._publish_log()
        return plan

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        """Release due jobs; follow each robot's leg, charge and timeout; then send loaded robots on to their second dock, give queued jobs to free robots, fullest first, send robots to chargers, lowest first, and the rest home."""
        now = self.node.sim_time.to_seconds() - self._t0
        while self._next.release_s <= now:
            job = self._next
            self._logger.info(f"released {job.kind} {job.index} at {job.release_s:.1f} s")
            self._note(f"job.{job.index}.kind", job.kind)
            self._note(f"job.{job.index}.released", _s(job.release_s))
            self._queue.append(job)
            self._next = next(self._jobs)

        for name in assigned:
            if name in self._state:
                self._hand_outside(name, now)
        self._watch_batteries(now)

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
            elif state in (_State.JOB, _State.WAIT):
                self._follow_job(name, state, last, now)
            elif state is _State.CHARGE:
                self._follow_charge(name, last, now)

        for name in sorted((name for name, state in self._state.items() if state is _State.WAIT), key=lambda name: typing.cast(float, self._run[name].waiting_s)):
            run = self._run[name]
            if await self._start_leg(name):
                self._logger.info(f"{name} goes on to unload {run.job.kind} {run.job.index} at {run.docks[1]}, {now:.1f} s")
            elif not run.sent_home and await self._submit(name, [GoToPhase(target=self._home[name], text=f"{run.job.kind} {run.job.index}: wait at {self._home[name]} for {run.docks[1]}")]):
                run.sent_home = True

        await self._hand_out_jobs(now)
        await self._send_to_chargers(now)

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
        return (self._loading.value, self._locker.value) if job.kind == "delivery" else (self._locker.value, self._loading.value)

    def _leg_phase(self, run: _Run) -> TaskPhase:
        job, dock = run.job, run.docks[run.leg]
        if run.leg == 0:
            return GoToPhase(target=dock, hold_time=self._load_s.value, on_failure="stop_task", text=f"{job.kind} {job.index}: load at {dock}, released {job.release_s:.1f} s")
        return GoToPhase(target=dock, hold_time=self._unload_s.value, on_failure="stop_task", text=f"{job.kind} {job.index}: unload at {dock}")

    async def _start_leg(self, name: str) -> bool:
        """Send the robot to its run's current dock if no other robot holds that dock, and hold it."""
        run = self._run[name]
        dock = run.docks[run.leg]
        if self._docks.get(dock, name) != name or not await self._submit(name, [self._leg_phase(run)]):
            return False
        self._docks[dock] = name
        self._state[name] = _State.JOB
        run.waiting_s = None
        return True

    async def _hand_out_jobs(self, now: float) -> None:
        """Give each queued job whose first dock no robot holds, oldest first, to the fullest free robot whose battery the job would leave at `reserve` or more; a fuller robot it would leave below goes to charge first, unless it is charged to `charge_until` already."""
        for job in list(self._queue):
            docks = self._docks_of(job)
            if docks[0] in self._docks:
                continue
            for name in self._takers():
                energy_wh, left = self._job_energy(name, job, docks)
                soc = self._soc(name)
                if left is not None and left < self._reserve.value and soc is not None and soc < self._charge_until.value:
                    self._logger.info(f"{name} needs {energy_wh:.2f} Wh for {job.kind} {job.index} and would keep {left:.1f} %, it charges first")
                    self._queue_charge(name, f"job {job.index}", now)
                    continue
                self._run[name] = _Run(job=job, docks=docks, assigned_s=now)
                if not await self._start_leg(name):
                    del self._run[name]
                    break
                self._queue.remove(job)
                self._logger.info(f"{name} takes {job.kind} {job.index}, {energy_wh:.2f} Wh, {now:.1f} s")
                self._note(f"job.{job.index}.robot", name)
                self._note(f"job.{job.index}.assigned", _s(now))
                self._note(f"job.{job.index}.energy_wh", f"{energy_wh:.3f}")
                break

    def _takers(self) -> list[str]:
        """Free robots that may take a job, fullest battery first, those whose charge is unknown last."""
        idle = [name for name, state in self._state.items() if state in (_State.PARKED, _State.FREE) and name not in self._charge_queue]
        idle = [name for name in idle if (soc := self._soc(name)) is None or soc >= self._charge_below.value]
        return sorted(idle, key=lambda name: (self._soc(name) is None, -(self._soc(name) or 0.0)))

    def _job_energy(self, name: str, job: Job, docks: tuple[str, str]) -> tuple[float, float | None]:
        """The Wh the job takes the robot, by straight-line distance from where it is to both docks times `detour` at `wh_per_m`, plus its standby power over the load and unload time, and the battery percent it would leave, None while its charge or capacity is unknown."""
        robot = self._ctx.robots[name]
        profile = robot.power_profile
        route_m = (math.dist(self._where(name), self._xy[docks[0]]) + math.dist(self._xy[docks[0]], self._xy[docks[1]])) * self._detour.value
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

    def _follow_job(self, name: str, state: _State, last: PhaseState | None, now: float) -> None:
        run = self._run[name]
        if state is _State.JOB and last is not None:
            if last.outcome != "met":
                self._end_job(name, "failed", str(last.reason) or last.outcome, now)
                return
            self._release(run.docks[run.leg], name)
            if run.leg == 1:
                self._end_job(name, "completed", "", now)
                return
            self._note(f"job.{run.job.index}.loaded", _s(now))
            run.leg, run.waiting_s, run.sent_home = 1, now, False
            self._state[name] = _State.WAIT
        elif self._timed_out(run.assigned_s, now):
            self._end_job(name, "abandoned", "timeout", now)

    def _end_job(self, name: str, outcome: str, reason: str, now: float) -> None:
        """Close the robot's job as completed, failed or abandoned, release its docks and free the robot; a job that did not complete is dropped, not queued again."""
        run = self._run.pop(name)
        for dock in run.docks:
            self._release(dock, name)
        job = run.job
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

    def _timed_out(self, since_s: float, now: float) -> bool:
        return self._job_timeout.value > 0.0 and now - since_s > self._job_timeout.value

    # CHARGING

    def _queue_charge(self, name: str, trigger: str, now: float) -> None:
        if name in self._charge_queue:
            return
        index = next(self._charges)
        self._charge_queue[name] = (index, now)
        self._note(f"charge.{index}.robot", name)
        self._note(f"charge.{index}.trigger", trigger)
        self._note(f"charge.{index}.queued", _s(now))

    async def _send_to_chargers(self, now: float) -> None:
        """Queue free robots under `charge_below`, then send queued robots to a charger, lowest battery first: a robot parked at its charger home charges there, any other takes a charger whose holder is away and swaps homes with it."""
        for name, state in self._state.items():
            soc = self._soc(name)
            if state in (_State.PARKED, _State.FREE) and soc is not None and soc < self._charge_below.value:
                self._queue_charge(name, "charge_below", now)
        for name in sorted(self._charge_queue, key=lambda name: (self._soc(name) or 0.0, self._charge_queue[name][1])):
            if self._state[name] not in (_State.PARKED, _State.FREE):
                continue
            home = self._home[name]
            if home in self._chargers and self._state[name] is _State.PARKED and name in self._arrived:
                self._start_charge(name, home, now)
                self._charge[name].started_s = now
                self._note(f"charge.{self._charge[name].index}.started", _s(now))
                self._note(f"charge.{self._charge[name].index}.soc_start", _pct(self._soc(name)))
            else:
                charger = home if home in self._chargers else next((charger for charger in self._chargers if self._owner_away(charger)), None)
                if charger is None or not await self._submit(name, [GoToPhase(target=charger, on_failure="stop_task", text=f"charge at {charger}")]):
                    continue
                owner = self._holder(charger)
                if owner is not None and owner != name:
                    self._home[owner] = home
                self._home[name] = charger
                self._start_charge(name, charger, now)
            self._logger.info(f"{name} charges at {self._home[name]} from {self._soc(name):.1f} %, {now:.1f} s")

    def _start_charge(self, name: str, charger: str, now: float) -> None:
        index, _ = self._charge_queue.pop(name)
        charge = _Charge(index=index, charger=charger, dispatched_s=now)
        self._charge[name] = charge
        self._state[name] = _State.CHARGE
        self._note(f"charge.{charge.index}.charger", charger)
        self._note(f"charge.{charge.index}.dispatched", _s(now))

    def _follow_charge(self, name: str, last: PhaseState | None, now: float) -> None:
        charge = self._charge[name]
        if charge.started_s is None:
            if last is not None and last.outcome == "met":
                charge.started_s = now
                self._arrived.add(name)
                self._note(f"charge.{charge.index}.started", _s(now))
                self._note(f"charge.{charge.index}.soc_start", _pct(self._soc(name)))
            elif last is not None:
                self._end_charge(name, "abandoned", str(last.reason) or last.outcome, now)
                return
            elif self._timed_out(charge.dispatched_s, now):
                self._end_charge(name, "abandoned", "timeout", now)
                return
        soc = self._soc(name)
        if charge.started_s is not None and soc is not None and soc >= self._charge_until.value:
            self._logger.info(f"{name} charged to {soc:.1f} % at {self._home[name]}, {now:.1f} s")
            self._end_charge(name, "charged", "", now)
            self._state[name] = _State.PARKED

    def _end_charge(self, name: str, outcome: str, reason: str, now: float) -> None:
        charge = self._charge.pop(name)
        self._note(f"charge.{charge.index}.ended", _s(now))
        self._note(f"charge.{charge.index}.outcome", outcome)
        self._note(f"charge.{charge.index}.soc_end", _pct(self._soc(name)))
        if reason:
            self._note(f"charge.{charge.index}.reason", reason)
            self._logger.warning(f"{name} {outcome} its charge at {charge.charger} ({reason}), {now:.1f} s")
        self._state[name] = _State.FREE

    def _holder(self, spot: str) -> str | None:
        return next((name for name, home in self._home.items() if home == spot), None)

    def _owner_away(self, spot: str) -> bool:
        holder = self._holder(spot)
        return holder is None or self._state[holder] in (_State.JOB, _State.FREE)

    def _soc(self, name: str) -> float | None:
        return self._ctx.robots[name].soc

    def _watch_batteries(self, now: float) -> None:
        """Log and count each time a robot's battery reaches 0 %."""
        for name in self._state:
            soc = self._soc(name)
            if soc is None:
                continue
            if soc <= 0.0 and name not in self._empty:
                self._empty.add(name)
                index = next(self._empties)
                self._logger.warning(f"{name}: battery empty at {now:.1f} s, {index + 1} this episode")
                self._note(f"empty.{index}.robot", name)
                self._note(f"empty.{index}.at", _s(now))
            elif soc > 0.0:
                self._empty.discard(name)

    # OUTSIDE TASKS AND PLUMBING

    def _hand_outside(self, name: str, now: float) -> None:
        """A task assigned from outside replaced whatever the robot did: its job is abandoned, its charge too, and the mode leaves it alone until that task is done."""
        state = self._state[name]
        if state in (_State.JOB, _State.WAIT):
            self._end_job(name, "abandoned", "assigned from outside", now)
        elif state is _State.CHARGE:
            self._end_charge(name, "abandoned", "assigned from outside", now)
        self._charge_queue.pop(name, None)
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
