import typing

from arena_rclpy_mixins.declarations import declare_double, declare_string
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace
from rcl_interfaces.msg import ParameterDescriptor
from rclpy import Parameter

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, ROBOTS_MODES

if typing.TYPE_CHECKING:
    from task_generator.tasks.robots import TM_Robots

_NS = _REGISTRY_NAMESPACE("fleet")


def _declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    node.rosparam.declare_forward(ns("homes"), [], descriptor=ParameterDescriptor(type=Parameter.Type.STRING_ARRAY.value, description="Zone or docked object each robot starts and parks at, one per robot in robot order."))
    declare_string(node, ns("loading"), "", label="Loading point", description="Zone or docked object where deliveries load and pickups unload, one robot at a time.")
    declare_string(node, ns("locker"), "", label="Locker", description="Docked object where deliveries unload and pickups load, one robot at a time.")
    declare_double(node, ns("rate"), 12.0, label="Rate", description="Jobs released per hour of sim time.", lo=0.1, hi=3600.0)
    declare_double(node, ns("pickup_share"), 0.3, label="Pickup share", description="Share of the jobs that are pickups, the rest are deliveries.", lo=0.0, hi=1.0)
    declare_double(node, ns("load_s"), 15.0, label="Load time", description="Seconds a robot holds where it loads.", lo=0.0, hi=3600.0)
    declare_double(node, ns("unload_s"), 10.0, label="Unload time", description="Seconds a robot holds where it unloads.", lo=0.0, hi=3600.0)
    declare_double(node, ns("charge_below"), 30.0, label="Charge below", description="Battery percent under which a free robot goes to a charger instead of taking a job.", lo=0.0, hi=100.0)
    declare_double(node, ns("charge_until"), 100.0, label="Charge until", description="Battery percent at which a charging robot is free again, at least charge_below.", lo=0.0, hi=100.0)
    declare_double(node, ns("reserve"), 10.0, label="Reserve", description="Battery percent a job must leave by its energy estimate, else the robot charges first.", lo=0.0, hi=100.0)
    declare_double(node, ns("wh_per_m"), 0.046, label="Energy per metre", description="Wh a robot uses per metre driven, standby while driving included, for the job energy estimate; 0 derives it from the robot's power profile at 0.4 m/s and 0.2 rad/s.", lo=0.0, hi=10.0)
    declare_double(node, ns("detour"), 1.2, label="Detour factor", description="Route length over straight-line distance assumed for the job energy estimate.", lo=1.0, hi=10.0)
    declare_double(node, ns("job_timeout"), 300.0, label="Job timeout", description="Seconds a job leg may drive from its start (its waits before a busy dock apart), and a trip to a charger from its dispatch, before the job or charge is abandoned and the robot goes home, on top of job_timeout_per_m for each metre of its route estimate; 0 for none.", lo=0.0, hi=86400.0)
    declare_double(node, ns("approach_m"), 5.0, label="Dock approach", description="Metres from the loading point or locker within which a robot on a job leg takes that dock, or holds in place before it while another robot has it; it holds the dock until its hold there ends.", lo=0.0, hi=1000.0)
    declare_double(node, ns("jam_window_s"), 60.0, label="Jam window", description="A robot on a job leg or a charging trip that moves less than jam_progress_m in this many seconds is jammed: its job fails (reason jam), its charge is abandoned, and it is reset to its home; 0 for no jam rule.", lo=0.0, hi=3600.0)
    declare_double(node, ns("jam_progress_m"), 0.5, label="Jam progress", description="Metres a robot must move within jam_window_s, away from where the window started, not to count as jammed.", lo=0.0, hi=100.0)
    declare_double(node, ns("jam_goal_m"), 1.0, label="Jam goal radius", description="Within this many metres of its leg's dock or its charger a robot is docking or holding, not jammed, and its jam window starts again.", lo=0.0, hi=100.0)
    declare_double(node, ns("job_timeout_per_m"), 5.0, label="Job timeout per metre", description="Seconds the timeout grows by per metre of the route estimate (straight line times detour): from where a job leg starts to its dock, to the charger for a charging trip.", lo=0.0, hi=3600.0)


@ROBOTS_MODES.register(Modes.TM_Robots.FLEET, namespace=_NS, schema=_declare_schema)
def _load_fleet() -> type["TM_Robots"]:
    from .impl import TM_Fleet

    return TM_Fleet
