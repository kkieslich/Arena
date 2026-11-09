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
    declare_string(node, ns("loading"), "", label="Loading point", description="Zone where deliveries load and pickups unload.")
    declare_string(node, ns("locker"), "", label="Locker", description="Docked object where deliveries unload and pickups load.")
    declare_double(node, ns("rate"), 12.0, label="Rate", description="Jobs released per hour of sim time.", lo=0.1, hi=3600.0)
    declare_double(node, ns("pickup_share"), 0.3, label="Pickup share", description="Share of the jobs that are pickups, the rest are deliveries.", lo=0.0, hi=1.0)
    declare_double(node, ns("load_s"), 15.0, label="Load time", description="Seconds a robot holds where it loads.", lo=0.0, hi=3600.0)
    declare_double(node, ns("unload_s"), 10.0, label="Unload time", description="Seconds a robot holds where it unloads.", lo=0.0, hi=3600.0)
    declare_double(node, ns("charge_below"), 30.0, label="Charge below", description="Battery percent under which a free robot goes to a charger instead of taking a job.", lo=0.0, hi=100.0)
    declare_double(node, ns("charge_until"), 100.0, label="Charge until", description="Battery percent at which a charging robot is free again.", lo=0.0, hi=100.0)


@ROBOTS_MODES.register(Modes.TM_Robots.FLEET, namespace=_NS, schema=_declare_schema)
def _load_fleet() -> type["TM_Robots"]:
    from .impl import TM_Fleet

    return TM_Fleet
