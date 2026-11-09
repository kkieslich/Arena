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
    node.rosparam.declare_forward(ns("homes"), [], descriptor=ParameterDescriptor(type=Parameter.Type.STRING_ARRAY.value, additional_constraints="label:Homes", description="One spot per robot, in robot order: the zone or docked object it starts at and parks at, waiting bays and chargers."))
    declare_string(node, ns("loading"), "", label="Loading point", description="Zone deliveries load at and pickups unload at.")
    declare_string(node, ns("locker"), "", label="Locker", description="Docked object deliveries unload at and pickups load at.")
    declare_double(node, ns("rate"), 12.0, lo=0.1, hi=3600.0, label="Rate", description="Jobs released per hour of sim time, a Poisson stream drawn from the episode seed.")
    declare_double(node, ns("pickup_share"), 0.3, lo=0.0, hi=1.0, label="Pickup share", description="Share of the jobs that are pickups, the rest are deliveries.")
    declare_double(node, ns("load_s"), 15.0, lo=0.0, hi=3600.0, label="Load time", description="Seconds a robot holds where it loads.")
    declare_double(node, ns("unload_s"), 10.0, lo=0.0, hi=3600.0, label="Unload time", description="Seconds a robot holds where it unloads.")
    declare_double(node, ns("charge_below"), 30.0, lo=0.0, hi=100.0, label="Charge below", description="Battery percent under which a free robot goes to a charger instead of taking a job.")
    declare_double(node, ns("charge_until"), 100.0, lo=0.0, hi=100.0, label="Charge until", description="Battery percent at which a charging robot is free again.")


@ROBOTS_MODES.register(Modes.TM_Robots.FLEET, namespace=_NS, schema=_declare_schema)
def _load_fleet() -> type["TM_Robots"]:
    from .impl import TM_Fleet

    return TM_Fleet
