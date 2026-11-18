import typing

from arena_rclpy_mixins.params import Param, ParamGroup, Range
from rclpy import Parameter

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("fleet")


class FleetParams(ParamGroup):
    HOMES = Param[list[str]]("homes", [], "Zone or docked object each robot starts and parks at, one per robot in robot order.", type_=Parameter.Type.STRING_ARRAY)
    LOADING = Param[str]("loading", "", "Zone or docked object where deliveries load and pickups unload, one robot at a time.")
    LOCKER = Param[str]("locker", "", "Docked object where deliveries unload and pickups load, one robot at a time.")
    RATE = Param[float]("rate", 12.0, "Jobs released per hour of sim time.", constraint=Range(0.1, 3600.0))
    PICKUP_SHARE = Param[float]("pickup_share", 0.3, "Share of the jobs that are pickups, the rest are deliveries.", constraint=Range(0.0, 1.0))
    LOAD_S = Param[float]("load_s", 15.0, "Seconds a robot holds where it loads.", constraint=Range(0.0, 3600.0))
    UNLOAD_S = Param[float]("unload_s", 10.0, "Seconds a robot holds where it unloads.", constraint=Range(0.0, 3600.0))
    CHARGE_BELOW = Param[float]("charge_below", 30.0, "Battery percent under which a free robot goes to a charger instead of taking a job.", constraint=Range(0.0, 100.0))
    CHARGE_UNTIL = Param[float]("charge_until", 100.0, "Battery percent at which a charging robot is free again, at least charge_below; with top_up, a free robot under it takes a charger whose holder is away.", constraint=Range(0.0, 100.0))
    RESERVE = Param[float]("reserve", 10.0, "Battery percent a job must leave by its energy estimate, else the robot charges first.", constraint=Range(0.0, 100.0))
    WH_PER_M = Param[float]("wh_per_m", 0.046, "Wh a robot uses per metre driven, standby while driving included, for the job energy estimate; 0 derives it from the robot's power profile at 0.4 m/s and 0.2 rad/s.", constraint=Range(0.0, 10.0))
    DETOUR = Param[float]("detour", 1.2, "Route length over straight-line distance assumed for the job energy estimate.", constraint=Range(1.0, 10.0))
    JOB_TIMEOUT = Param[float]("job_timeout", 300.0, "Seconds a job leg may drive from its start (its waits before a busy dock apart), and a trip to a charger from its dispatch, before the job or charge is abandoned and the robot goes home, on top of job_timeout_per_m for each metre of its route estimate; 0 for none.", constraint=Range(0.0, 86400.0))
    APPROACH_M = Param[float]("approach_m", 5.0, "Metres from the loading point or locker within which a robot on a job leg takes that dock, or holds in place before it while another robot has it; it holds the dock until its hold there ends.", constraint=Range(0.0, 1000.0))
    JAM_WINDOW_S = Param[float]("jam_window_s", 60.0, "A robot on a job leg or a charging trip that moves less than jam_progress_m in this many seconds is jammed: its job fails (reason jam), its charge is abandoned, and it is reset to its home; 0 for no jam rule.", constraint=Range(0.0, 3600.0))
    JAM_PROGRESS_M = Param[float]("jam_progress_m", 0.5, "Metres a robot must move within jam_window_s, away from where the window started, not to count as jammed.", constraint=Range(0.0, 100.0))
    JAM_GOAL_M = Param[float]("jam_goal_m", 1.0, "Within this many metres of its leg's dock or its charger a robot is docking or holding, not jammed, and its jam window starts again.", constraint=Range(0.0, 100.0))
    TOP_UP = Param[bool]("top_up", True, "Opportunistic charging: once no robot waits to charge, a free robot under charge_until whose home is no charger takes a charger whose holder is away on a job as its home; a robot queued to charge takes it from it on the way, or docked until it has charged to charge_until.")
    JOB_TIMEOUT_PER_M = Param[float]("job_timeout_per_m", 5.0, "Seconds the timeout grows by per metre of the route estimate (straight line times detour): from where a job leg starts to its dock, to the charger for a charging trip.", constraint=Range(0.0, 3600.0))


@ROBOTS_MODES.register(Modes.TM_Robots.FLEET, namespace=_NS, schema=FleetParams)
def _load_fleet() -> type["TM_Robots"]:
    from .impl import TM_Fleet

    return TM_Fleet
