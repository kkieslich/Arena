import enum
import functools
import itertools
import math
import typing
from collections import deque
from collections.abc import Collection, Iterator, Mapping

import attrs
import numpy as np
import rclpy
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_robots_msgs.msg import Energy
from arena_runtime.env.task_api import GoToPhase, PhaseState, TaskPhase, TaskRequest, authored_statics
from arena_runtime.env.task_layer import ResetPlan, RobotReset
from rclpy.subscription import Subscription

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots


@attrs.frozen
class Job:
    index: int
    kind: typing.Literal["delivery", "pickup"]
    release_s: float


class _State(enum.Enum):
    PARKED = "parked"
    FREE = "free"
    JOB = "job"
    CHARGE = "charge"


def releases(rng: np.random.Generator, rate_per_h: float, pickup_share: float) -> Iterator[Job]:
    """Jobs in release order, a Poisson stream drawn from rng alone."""
    release_s = 0.0
    for index in itertools.count():
        release_s += float(rng.exponential(3600.0 / rate_per_h))
        yield Job(index, "pickup" if rng.random() < pickup_share else "delivery", release_s)


class TM_Fleet(TM_Robots):
    """Deliveries and pickups released at a fixed rate from the episode seed, handed to free robots, with charging trips below a battery level."""

    _homes: ROSParamT[list[str]]
    _loading: ROSParamT[str]
    _locker: ROSParamT[str]
    _rate: ROSParamT[float]
    _pickup_share: ROSParamT[float]
    _load_s: ROSParamT[float]
    _unload_s: ROSParamT[float]
    _charge_below: ROSParamT[float]
    _charge_until: ROSParamT[float]

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

        self._subscriptions: dict[str, Subscription] = {}
        self._soc: dict[str, float] = {}
        self._chargers: list[str] = []
        self._home: dict[str, str] = {}
        self._state: dict[str, _State] = {}
        self._job: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._jobs: Iterator[Job] = iter(())
        self._next = Job(0, "delivery", math.inf)  # nothing is released before the first reset
        self._t0 = 0.0

    async def reset(self) -> ResetPlan:
        plan = await super().reset()
        robots = list(self._ctx.robots.values())
        homes = list(self._homes.value)
        if len(homes) < len(robots):
            raise ValueError(f"task.fleet.homes names {len(homes)} spots for {len(robots)} robots")
        world = authored_statics(self._ctx.world_manager.world_compacted())
        spots = {zone.name for zone in world.zones} | {entity.name for entity in world.objects if entity.dock is not None}
        unknown = [spot for spot in (*homes, self._loading.value, self._locker.value) if spot not in spots]
        if unknown:
            raise ValueError(f"task.fleet names {unknown}, which are no zone or docked object of the world")

        self._chargers = [entity.name for entity in world.objects if entity.charge_power_w is not None]
        self._home = {robot.name: home for robot, home in zip(robots, homes, strict=False)}
        self._state = dict.fromkeys(self._home, _State.PARKED)
        self._soc, self._job, self._queue = {}, {}, deque()
        for robot in robots:
            start = robot.resolve_target(GoToPhase(target=self._home[robot.name])).pose
            assert start is not None, f"{robot.name}'s home {self._home[robot.name]!r} has no dispatch pose"
            plan.robots[robot.name] = RobotReset(start_pose=start)
            if robot.name not in self._subscriptions:
                self._subscriptions[robot.name] = self.node.create_subscription(Energy, str(robot.namespace("power_publisher", "energy")), functools.partial(self._on_energy, robot.name), 10)
        self._jobs = releases(self._ctx.conf.RNG.stream("robots", "fleet"), self._rate.value, self._pickup_share.value)
        self._next = next(self._jobs)
        self._t0 = self.node.sim_time.to_seconds()
        return plan

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        """Release due jobs, then send free robots to charge, lowest battery first, to jobs, fullest first, or home."""
        now = self.node.sim_time.to_seconds() - self._t0
        while self._next.release_s <= now:
            self._logger.info(f"released {self._next.kind} {self._next.index} at {self._next.release_s:.1f} s")
            self._queue.append(self._next)
            self._next = next(self._jobs)

        for name, state in list(self._state.items()):
            if state is _State.JOB and await self._ctx.robots[name].is_done:
                job = self._job.pop(name)
                self._logger.info(f"{name} finished {job.kind} {job.index}: released {job.release_s:.1f} s, done {now:.1f} s")
                self._state[name] = _State.FREE
            elif state is _State.CHARGE:
                trip = ended.get(name)
                if trip is not None and trip.outcome != "met":
                    self._state[name] = _State.FREE
                elif self._battery(name) >= self._charge_until.value:
                    self._logger.info(f"{name} charged to {self._battery(name):.1f} % at {self._home[name]}, {now:.1f} s")
                    self._state[name] = _State.PARKED
        free = [name for name, state in self._state.items() if state in (_State.FREE, _State.PARKED)]

        for name in sorted(free, key=self._battery):
            if self._battery(name) >= self._charge_below.value:
                break
            home = self._home[name]
            charger = home if home in self._chargers else next((charger for charger in self._chargers if self._owner_away(charger)), None)
            if charger is None:
                continue
            if charger == home and self._state[name] is _State.PARKED:
                self._state[name] = _State.CHARGE
            elif await self._submit(name, [GoToPhase(target=charger, text=f"charge at {charger}")], _State.CHARGE):
                owner = self._holder(charger)
                if owner is not None and owner != name:
                    self._home[owner] = home
                self._home[name] = charger
            else:
                continue
            self._logger.info(f"{name} charges at {charger} from {self._battery(name):.1f} %, {now:.1f} s")

        for name in sorted(free, key=self._battery, reverse=True):
            if not self._queue:
                break
            if self._state[name] in (_State.FREE, _State.PARKED) and self._battery(name) >= self._charge_below.value:
                job = self._queue.popleft()
                if await self._submit(name, self._job_phases(job), _State.JOB):
                    self._job[name] = job
                else:
                    self._queue.appendleft(job)

        for name in free:
            if self._state[name] is _State.FREE:
                await self._submit(name, [GoToPhase(target=self._home[name], text=f"park at {self._home[name]}")], _State.PARKED)

    @property
    async def done(self) -> bool:
        return False

    async def teardown(self) -> None:
        for subscription in self._subscriptions.values():
            self.node.destroy_subscription(subscription)
        self._subscriptions = {}
        await super().teardown()

    def _job_phases(self, job: Job) -> list[TaskPhase]:
        load, unload = (self._loading.value, self._locker.value) if job.kind == "delivery" else (self._locker.value, self._loading.value)
        return [
            GoToPhase(target=load, hold_time=self._load_s.value, text=f"{job.kind} {job.index}: load at {load}, released {job.release_s:.1f} s"),
            GoToPhase(target=unload, hold_time=self._unload_s.value, text=f"{job.kind} {job.index}: unload at {unload}"),
        ]

    async def _submit(self, name: str, phases: list[TaskPhase], state: _State) -> bool:
        try:
            await self._ctx.robots[name].submit_task(TaskRequest(phases=phases))
        except ValueError as e:
            # the spot has no free cell while another robot stands in it, so the robot is free again until the next tick
            self._logger.debug(f"{name}: {e}")
            self._state[name] = _State.FREE
            return False
        self._state[name] = state
        return True

    def _holder(self, spot: str) -> str | None:
        return next((name for name, home in self._home.items() if home == spot), None)

    def _owner_away(self, spot: str) -> bool:
        holder = self._holder(spot)
        return holder is None or self._state[holder] in (_State.JOB, _State.FREE)

    def _on_energy(self, name: str, msg: Energy) -> None:
        self._soc[name] = msg.battery_soc_percent

    def _battery(self, name: str) -> float:
        return self._soc.get(name, 100.0)  # full until its power publisher reports
