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
    JOB_TIMEOUT = Param[float](
        "job_timeout",
        300.0,
        "Seconds a job leg may drive from its start (its waits before a busy dock apart), and a trip to a charger from its dispatch, before the job or charge is abandoned and the robot goes home, on top of job_timeout_per_m for each metre of its route estimate; 0 for none.",
        constraint=Range(0.0, 86400.0),
    )
    APPROACH_M = Param[float]("approach_m", 5.0, "Metres from the loading point or locker within which a robot on a job leg takes that dock, or holds in place before it while another robot has it; it holds the dock until its hold there ends.", constraint=Range(0.0, 1000.0))
    JAM_WINDOW_S = Param[float](
        "jam_window_s",
        60.0,
        "A robot on a job leg or a charging trip that moves less than jam_progress_m in this many seconds is jammed: its job fails (reason jam), its charge is abandoned, and it is reset to its home; with jam_home a robot on its way home too (see jam_home); 0 for no jam rule.",
        constraint=Range(0.0, 3600.0),
    )
    JAM_PROGRESS_M = Param[float]("jam_progress_m", 0.5, "Metres a robot must move within jam_window_s, away from where the window started, not to count as jammed.", constraint=Range(0.0, 100.0))
    JAM_GOAL_M = Param[float]("jam_goal_m", 1.0, "Within this many metres of its leg's dock, its charger or (jam_home) its home a robot is docking or holding, not jammed, and its jam window starts again.", constraint=Range(0.0, 100.0))
    JAM_HOME = Param[bool](
        "jam_home",
        True,
        "The jam rule also watches a robot on its way home (after a job, a charge or a jam, also one whose job was abandoned on its timeout): one that moves less than jam_progress_m in jam_window_s, further than jam_goal_m from its home, is teleported home and its navigation is reset (logged jam.N.trip home; no job or charge ends), once per trip unless it moves jam_progress_m on from where it was reset. Robots parked at their home, holding before a dock or a charger, or stopped at 0 % are not watched.",
    )
    TOP_UP = Param[bool]("top_up", True, "Opportunistic charging: once no robot waits to charge, a free robot under charge_until whose home is no charger takes a charger whose holder is away on a job as its home; a robot queued to charge takes it from it on the way, or docked until it has charged to charge_until.")
    LOCKERS = Param[list[str]]("lockers", [], "Docked objects where deliveries unload and pickups load when a site has several lockers: a job of the job list names the one it goes to. Empty: the one locker of `locker`.", type_=Parameter.Type.STRING_ARRAY)
    JOBS = Param[list[str]](
        "jobs", [], "The job list of the episode, one entry per job in release order: '<release s> <delivery|pickup> <locker> [<recipient entrance>]', the locker one of `lockers` (or `locker`). Empty: jobs are drawn from the episode seed at `rate` with `pickup_share`.", type_=Parameter.Type.STRING_ARRAY
    )
    CHARGE_RULE = Param[str](
        "charge_rule", "threshold", "When a robot goes to charge. threshold: a free robot under charge_below, a robot whose next job would leave it under reserve, and opportunistic top-ups (top_up). need: only a robot whose next job would leave it under reserve by the job's energy estimate; no threshold, no top-ups."
    )
    NEAREST_CHARGER = Param[bool]("nearest_charger", False, "A robot sent to charge whose home is no charger takes the nearest free charger (straight line from where it is) instead of the first free one in the world's order.")
    CHARGE_JAM = Param[str](
        "charge_jam", "home", "What a jam on a charging trip does. home: the charge is abandoned and the robot is reset to the home it had before the trip. retry: the jam is logged, the robot's navigation is reset where it stands and it is sent on to the same charger; the trip's timeout keeps running."
    )
    IDLE_FLOOR = Param[float]("idle_floor", 0.0, "Battery percent at or under which a free robot not docked on a charger goes to charge, whatever the charge rule (the floor of charging by need: the reserve plus the longest trip to a charger); 0 for none.", constraint=Range(0.0, 100.0))
    ONE_NEED_CHARGE = Param[bool](
        "one_need_charge", False, "When a queued job is one no free robot can take without ending under the reserve, only the fullest of them goes to charge, and only if no robot already charges for that job; the others stay free. Off: every free robot the job is offered to and cannot take goes to charge."
    )
    CHARGE_AT_START = Param[bool]("charge_at_start", False, "A robot whose home is a charger and that starts the episode under charge_until charges there before it takes a job.")
    STOP_EMPTY = Param[bool](
        "stop_empty",
        False,
        "A robot whose battery reaches 0 % stops where it is for the rest of the episode: its job is abandoned (reason empty), its charge too, and it gets no job or trip. Its home is free for others, unless it stopped in a charger's dock (logged once, empty.N.in_dock): no robot is sent to that charger while it stands there.",
    )
    NO_POWER_S = Param[float](
        "no_power_s",
        120.0,
        "Seconds a started charge may go without the robot's charge rising by no_power_rise over its lowest since the window started (the window starts at the charge's start, then again at each such rise) before it ends with outcome no_power: the runtime charges one robot per dock, the first parked in it. The robot queues to charge again at once (trigger no_power) and keeps the charger as its home, but takes that charger again only once no other robot stands in its dock (until the charge queued then ends or gets power). A robot parked on its charger home under charge_until, not queued, is watched the same way: if its dock does not charge it, it queues to charge (trigger no_power) and takes another charger while another robot stands in that dock. 0 for no check.",
        constraint=Range(0.0, 86400.0),
    )
    NO_POWER_RISE = Param[float]("no_power_rise", 0.1, "Battery percentage points a started charge must rise by within no_power_s to count as getting power.", constraint=Range(0.0, 100.0))
    SOC_LOG = Param[str]("soc_log", "", "File the mode writes the robots' battery charge and homes to every soc_log_s seconds (CSV with a header line); empty for none. A file that exists and is not empty is refused at the reset: logs are written once.")
    SOC_LOG_S = Param[float]("soc_log_s", 60.0, "Seconds between two lines of soc_log.", constraint=Range(1.0, 86400.0))
    LEG_TIMEOUT_MOVING = Param[bool](
        "leg_timeout_moving",
        True,
        "A job leg past its timeout is abandoned even while the robot still moves. false: past its timeout a leg drives on while the robot moves (logged job.N.overdue_K when the timeout passed) and is abandoned (reason timeout) only once the robot has moved less than jam_progress_m over jam_window_s within jam_goal_m of its dock, where the jam rule does not look; a robot that stands further from its dock is the jam rule's. A jam of a leg past its timeout ends the job abandoned with reason timeout (jam.N.overdue logged), not failed with reason jam. At leg_timeout_cap times its timeout a leg ends whatever the robot does. A tick without a pose or while the robot waits for a busy dock decides nothing; a leg that cannot be submitted and every leg with jam_window_s 0 keep the plain timeout; charging trips are not affected.",
    )
    LEG_TIMEOUT_CAP = Param[float](
        "leg_timeout_cap", 2.0, "With leg_timeout_moving false: a leg that has driven this many times its timeout ends (abandoned, reason timeout) whatever the robot does, so a robot that keeps moving and never arrives cannot hold its dock for good; 0 for no cap.", constraint=Range(0.0, 100.0)
    )
    KEEP_CROWD_OFF = Param[bool](
        "keep_crowd_off",
        True,
        "Walkers route round the robots' parking spots: the dock of each robot's home and of every charger is walled off for the crowd simulator alone (the robots' map and planners stay as they are), so a walker never walks into a parking spot or stops at a robot parked there. false: the crowd treats them as free floor.",
    )
    JOB_TIMEOUT_PER_M = Param[float]("job_timeout_per_m", 5.0, "Seconds the timeout grows by per metre of the route estimate (straight line times detour): from where a job leg starts to its dock, to the charger for a charging trip.", constraint=Range(0.0, 3600.0))


@ROBOTS_MODES.register(Modes.TM_Robots.FLEET, namespace=_NS, schema=FleetParams)
def _load_fleet() -> type["TM_Robots"]:
    from .impl import TM_Fleet

    return TM_Fleet
