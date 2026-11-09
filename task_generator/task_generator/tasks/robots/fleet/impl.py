import functools
import itertools
import typing
from collections import deque
from collections.abc import Collection, Iterator, Mapping

import attrs
import numpy as np
import rclpy
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


def releases(rng: np.random.Generator, rate_per_h: float, pickup_share: float) -> Iterator[Job]:
    """Jobs in release order, from rng alone: exponential gaps at the rate (a Poisson stream), each a pickup with pickup_share."""
    release_s = 0.0
    for index in itertools.count():
        release_s += float(rng.exponential(3600.0 / rate_per_h))
        yield Job(index, "pickup" if rng.random() < pickup_share else "delivery", release_s)


class TM_Fleet(TM_Robots):
    """Deliveries and pickups released at a fixed rate from the episode seed, handed to free robots, with charging trips below a battery level."""

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)
        param = self.node.ROSParam
        self._homes = param[list[str]](self.namespace("homes"), [], type_=rclpy.Parameter.Type.STRING_ARRAY)
        self._loading = param[str](self.namespace("loading"), "")
        self._locker = param[str](self.namespace("locker"), "")
        self._rate = param[float](self.namespace("rate"), 12.0)
        self._pickup_share = param[float](self.namespace("pickup_share"), 0.3)
        self._load_s = param[float](self.namespace("load_s"), 15.0)
        self._unload_s = param[float](self.namespace("unload_s"), 10.0)
        self._charge_below = param[float](self.namespace("charge_below"), 30.0)
        self._charge_until = param[float](self.namespace("charge_until"), 100.0)
        self._soc: dict[str, float] = {}
        self._subscriptions: dict[str, Subscription] = {}
        self._home: dict[str, str] = {}
        self._state: dict[str, str] = {}
        self._job: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._chargers: list[str] = []

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
        self._soc, self._job, self._queue = {}, {}, deque()
        self._home = {robot.name: home for robot, home in zip(robots, homes, strict=False)}
        self._state = {robot.name: "parked" for robot in robots}
        for robot in robots:
            start = robot.resolve_target(GoToPhase(target=self._home[robot.name])).pose
            assert start is not None
            # no forbidden zone at the start: it would stay in the map all episode and leave the home no free cell to return to
            plan.robots[robot.name] = RobotReset(start_pose=start)
            if robot.name not in self._subscriptions:
                topic = str(robot.namespace("power_publisher", "energy"))
                self._subscriptions[robot.name] = self.node.create_subscription(Energy, topic, functools.partial(self._on_energy, robot.name), 10)
        self._jobs = releases(self._ctx.conf.RNG.stream("robots", "fleet"), self._rate.value, self._pickup_share.value)
        self._next = next(self._jobs)
        self._t0 = self._now()
        return plan

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        now = self._now() - self._t0
        while self._next.release_s <= now:
            self._queue.append(self._next)
            self._logger.info(f"released {self._next.kind} {self._next.index} at {self._next.release_s:.1f} s")
            self._next = next(self._jobs)

        for name, manager in self._ctx.robots.items():
            state = self._state[name]
            if state == "job" and await manager.is_done:
                job = self._job.pop(name)
                self._logger.info(f"{name} finished {job.kind} {job.index}: released {job.release_s:.1f} s, done {now:.1f} s")
                self._state[name] = "free"
            elif state == "charge":
                trip = ended.get(name)
                if trip is not None and trip.outcome != "met":
                    self._state[name] = "free"
                elif self._battery(name) >= self._charge_until.value:
                    self._logger.info(f"{name} charged to {self._battery(name):.1f} % at {self._home[name]}, {now:.1f} s")
                    self._state[name] = "parked"
        free = [name for name, state in self._state.items() if state in ("free", "parked")]

        for name in sorted(free, key=self._battery):
            if self._battery(name) >= self._charge_below.value:
                break
            # a charger whose owner is away (on a job, or done with one), so its dock is empty
            charger = self._home[name] if self._home[name] in self._chargers else next((c for c in self._chargers if self._owner_away(c)), None)
            if charger is None:
                continue
            if self._home[name] == charger and self._state[name] == "parked":
                self._state[name] = "charge"  # parked in its own charger, or on its way there
            elif await self._submit(name, [GoToPhase(target=charger, text=f"charge at {charger}")], "charge"):
                other = self._holder(charger)
                if other not in (None, name):
                    # swap homes, so every spot keeps one owner and the charger's owner parks in the spot this robot left
                    self._home[other] = self._home[name]
                self._home[name] = charger
            else:
                continue
            self._logger.info(f"{name} charges at {charger} from {self._battery(name):.1f} %, {now:.1f} s")

        for name in sorted(free, key=self._battery, reverse=True):
            if not self._queue:
                break
            if self._state[name] in ("free", "parked") and self._battery(name) >= self._charge_below.value:
                job = self._queue.popleft()
                if await self._submit(name, self._job_phases(job), "job"):
                    self._job[name] = job
                else:
                    self._queue.appendleft(job)

        for name in free:
            if self._state[name] == "free":
                await self._submit(name, [GoToPhase(target=self._home[name], text=f"park at {self._home[name]}")], "parked")

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

    async def _submit(self, name: str, phases: list[TaskPhase], state: str) -> bool:
        try:
            await self._ctx.robots[name].submit_task(TaskRequest(phases=phases))
        except ValueError as e:
            # every name is a spot of the world (checked at reset), so this one has no free cell while another robot stands in it: free again, retried next tick
            self._logger.debug(f"{name}: {e}")
            self._state[name] = "free"
            return False
        self._state[name] = state
        return True

    def _holder(self, spot: str) -> str | None:
        return next((name for name, home in self._home.items() if home == spot), None)

    def _owner_away(self, spot: str) -> bool:
        holder = self._holder(spot)
        return holder is None or self._state[holder] in ("job", "free")

    def _on_energy(self, name: str, msg: Energy) -> None:
        self._soc[name] = msg.battery_soc_percent

    def _battery(self, name: str) -> float:
        # a robot whose power publisher has not reported yet counts as full
        return self._soc.get(name, 100.0)

    def _now(self) -> float:
        time = self.node.sim_time
        return time.sec + time.nanosec * 1e-9
