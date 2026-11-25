from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

import arena_people_msgs.msg
import arena_robots_msgs.msg
import geometry_msgs.msg
import nav2_msgs.msg
import numpy as np
import shapely
import shapely.affinity
import std_msgs.msg
import visualization_msgs.msg as viz
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.shared import Namespace
from arena_rclpy_mixins.Time import Time, TimeNode
from arena_robots.caps import PolygonSpec
from builtin_interfaces.msg import Duration
from rclpy.parameter import Parameter

from arena_runtime.env.collision_grid import Cell
from arena_runtime.humans.devices import body_discs, device_outline

if TYPE_CHECKING:
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.robot_manager.robot_manager import RobotManager

_ACTION_CODES: dict[str | None, int] = {
    None: 0,
    'stop': 1,
    'slowdown': 2,
    'approach': 3,
    'limit': 4,
}

_RED = std_msgs.msg.ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0)
_GREY = std_msgs.msg.ColorRGBA(r=0.55, g=0.6, b=0.65, a=1.0)
_DEVICE = std_msgs.msg.ColorRGBA(r=0.1, g=0.45, b=0.95, a=1.0)
_CONTACT = std_msgs.msg.ColorRGBA(r=0.95, g=0.1, b=0.1, a=0.6)

_CELL_KINDS: dict[Cell, str] = {
    Cell.MAP: arena_robots_msgs.msg.CollisionEvent.KIND_STATIC,
    Cell.WALL: arena_robots_msgs.msg.CollisionEvent.KIND_WALL,
    Cell.STATIC: arena_robots_msgs.msg.CollisionEvent.KIND_STATIC,
}


class PedestrianIndex:
    """Body discs and held devices on the ground of one roster in a spatial index.

    Built once per roster and shared by the trackers of every robot of the env: per tick a tracker queries the
    index for what its polygons touch instead of testing every pedestrian, so its cost stays flat as the crowd grows.
    """

    def __init__(self, peds: Iterable[arena_people_msgs.msg.Pedestrian], default_radius: float):
        self.peds = list(peds)
        self.bodies = body_discs(self.peds, default_radius)
        self.devices = [device_outline(p) for p in self.peds]
        body_at = np.array([i for i, body in enumerate(self.bodies) if body is not None], dtype=np.intp)
        device_at = np.array([i for i, device in enumerate(self.devices) if device is not None], dtype=np.intp)
        self._owner = np.concatenate([body_at, device_at])
        self._is_device = np.concatenate([np.zeros(len(body_at), dtype=bool), np.ones(len(device_at), dtype=bool)])
        geometries = np.empty(len(self._owner), dtype=object)
        geometries[: len(body_at)] = self.bodies[body_at]
        geometries[len(body_at) :] = [self.devices[i] for i in device_at]
        self._tree = shapely.STRtree(geometries)

    def touching(self, geometry: shapely.Geometry) -> tuple[set[int], set[int]]:
        """Roster indices of the pedestrians whose body, and of those whose held device, intersects `geometry`."""
        hit = self._tree.query(geometry, predicate='intersects')
        device = self._is_device[hit]
        return set(self._owner[hit[~device]].tolist()), set(self._owner[hit[device]].tolist())

    def touches_any(self, geometry: shapely.Geometry) -> bool:
        return self._tree.query(geometry, predicate='intersects').size > 0

    def held_devices(self) -> Iterator[tuple[arena_people_msgs.msg.Pedestrian, shapely.Polygon]]:
        """Each held device on the ground with its holder, in roster order."""
        for p, device in zip(self.peds, self.devices, strict=True):
            if device is not None:
                yield p, device


class _RosterIndexes:
    """The index of the latest roster, the env hands every tracker the same message object."""

    def __init__(self) -> None:
        self._latest: tuple[arena_people_msgs.msg.Pedestrians | None, float, PedestrianIndex] | None = None

    def of(self, msg: arena_people_msgs.msg.Pedestrians | None, default_radius: float) -> PedestrianIndex:
        latest = self._latest
        if latest is not None and latest[0] is msg and latest[1] == default_radius:
            return latest[2]
        index = PedestrianIndex(msg.pedestrians if msg is not None else [], default_radius)
        self._latest = (msg, default_radius, index)
        return index


_ROSTER_INDEXES = _RosterIndexes()


class CollisionTrackerNode(TimeNode):
    """Separate Node, same process, same executor as the env node.

    Reads RobotManager.pose + EnvironmentManager.collision_grid. The footprint
    defines a collision: it feeds `collision_events` and the robot manager's
    collision callback. Pedestrians are their body disc and held device on
    the ground. The cap polygons only feed `collision_monitor_state`. Tick
    timer uses sim time, so it is gated during pauses.
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        polygons: dict[str, PolygonSpec],
        *,
        rate_hz: float = 10.0,
        default_pedestrian_radius: float = 0.3,
        footprint: list[list[float]] | None = None,
    ):
        for name, spec in polygons.items():
            if spec.type == 'circle' and spec.radius is None:
                raise ValueError(f'collision_tracker: polygon {name!r}: circle without a radius')
        super().__init__(
            'collision_tracker',
            namespace=str(robot_manager.namespace),
            use_global_arguments=False,
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )

        self._rm = robot_manager
        self._env: EnvironmentManager = robot_manager._environment_manager
        self._default_peds_radius = default_pedestrian_radius
        self._peds_msg: arena_people_msgs.msg.Pedestrians | None = None

        self._poly_cache: dict[str, dict] = {}
        for name, spec in polygons.items():
            if spec.type == 'polygon':
                if spec.points is None or len(spec.points) < 3:
                    self.get_logger().warning(f'collision_tracker: skipping polygon {name!r}: fewer than 3 points')
                    continue
                self._poly_cache[name] = {
                    'type': 'polygon',
                    'shapely': shapely.Polygon(spec.points),
                    'action_code': _ACTION_CODES.get(spec.action_type, 0),
                }
            else:
                self._poly_cache[name] = {
                    'type': 'circle',
                    'R_c': spec.radius,
                    'action_code': _ACTION_CODES.get(spec.action_type, 0),
                }

        self._footprint_base: shapely.Polygon | None = None
        if footprint is not None and len(footprint) >= 3:
            self._footprint_base = shapely.Polygon(footprint)
        # None while a reset is in flight
        self._valid_after: Time | None = None

        self._pub_state: LazyPublisher[nav2_msgs.msg.CollisionMonitorState] = LazyPublisher(self.create_publisher(nav2_msgs.msg.CollisionMonitorState, 'collision_monitor_state', 10))
        self._pub_events = None if self._footprint_base is None else self.create_publisher(arena_robots_msgs.msg.CollisionEvents, 'collision_events', 10)
        self._pub_markers: LazyPublisher[viz.MarkerArray] = LazyPublisher(self.create_publisher(viz.MarkerArray, 'collision_markers', 10))
        self._marker_ids: dict[tuple[str, str], int] = {}
        env_ns = Namespace(robot_manager.node.get_namespace())
        # the env's own roster, handed over in-process: a subscription would decode every arena_peds again per robot
        self._unsubscribe_peds = self._env.on_arena_peds(self._on_peds)
        self._sub_reset_start = self.create_subscription(std_msgs.msg.Empty, str(env_ns('reset_start')), self._on_reset_start, 1)
        self._sub_reset_end = self.create_subscription(std_msgs.msg.Empty, str(env_ns('reset_end')), self._on_reset_end, 1)
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

    def _on_peds(self, msg: arena_people_msgs.msg.Pedestrians):
        self._peds_msg = msg

    def _on_reset_start(self, _msg: std_msgs.msg.Empty):
        self._valid_after = None

    def _on_reset_end(self, _msg: std_msgs.msg.Empty):
        self._peds_msg = None
        self._marker_ids.clear()
        self._valid_after = self.sim_time

    def _robot_polygon(self, entry: dict, rx: float, ry: float, rth: float) -> shapely.Polygon:
        if entry['type'] == 'polygon':
            poly = shapely.affinity.rotate(entry['shapely'], rth, origin=(0, 0), use_radians=True)
            return shapely.affinity.translate(poly, xoff=rx, yoff=ry)
        return shapely.Point(rx, ry).buffer(entry['R_c'])

    def _monitor_state(self, rx: float, ry: float, rth: float, peds: PedestrianIndex | None = None) -> nav2_msgs.msg.CollisionMonitorState:
        if peds is None:
            peds = self._pedestrian_index()
        grid = self._env.collision_grid
        polygons_hit: dict[str, int] = {}
        for name, entry in self._poly_cache.items():
            robot_poly = self._robot_polygon(entry, rx, ry, rth)
            if (grid is not None and grid.hit(robot_poly) is not None) or peds.touches_any(robot_poly):
                polygons_hit[name] = entry['action_code']

        state = nav2_msgs.msg.CollisionMonitorState()
        if polygons_hit:
            name, code = min(polygons_hit.items(), key=lambda kv: (1 if kv[1] == 0 else 0, kv[1]))
            state.polygon_name = name
            state.action_type = code
        else:
            state.polygon_name = ''
            state.action_type = 0
        return state

    def _tick(self):
        stamped = self._rm.pose_stamped
        if stamped is None:
            return
        pose, stamp = stamped
        if self._valid_after is None or Time.from_rclpy(stamp) <= self._valid_after:
            return

        rx, ry, rth = pose.to_2d()

        grid = self._env.collision_grid
        statics = self._env.static_polygons
        peds = self._pedestrian_index()

        self._pub_state.publish(lambda: self._monitor_state(rx, ry, rth, peds))

        if self._footprint_base is None or self._pub_events is None:
            return

        footprint = shapely.affinity.translate(
            shapely.affinity.rotate(self._footprint_base, rth, origin=(0, 0), use_radians=True),
            xoff=rx,
            yoff=ry,
        )
        events: list[arena_robots_msgs.msg.CollisionEvent] = []

        hit = grid.hit(footprint) if grid is not None else None
        if hit is not None:
            cell, obstacle_id = hit
            static = statics.get(obstacle_id)
            ev = arena_robots_msgs.msg.CollisionEvent()
            ev.kind = _CELL_KINDS[cell]
            ev.obstacle_id = obstacle_id
            ev.distance = 0.0
            ev.obstacle_position = geometry_msgs.msg.Point(x=rx, y=ry, z=0.0) if static is None else geometry_msgs.msg.Point(x=float(static.centroid.x), y=float(static.centroid.y), z=0.0)
            events.append(ev)

        bodies, devices = peds.touching(footprint)
        for i in sorted(bodies | devices):
            p = peds.peds[i]
            if i in bodies:
                events.append(self._event(arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN, p, p.pose.position.x, p.pose.position.y))
            if i in devices and (device := peds.devices[i]) is not None:
                touch = footprint.intersection(device).centroid
                events.append(self._event(arena_robots_msgs.msg.CollisionEvent.KIND_DEVICE, p, touch.x, touch.y, p.device.kind))

        self._pub_markers.publish(lambda: self._markers(footprint, events, peds))

        evmsg = arena_robots_msgs.msg.CollisionEvents()
        evmsg.header.stamp = self.sim_time.to_msg()
        evmsg.header.frame_id = 'map'
        evmsg.events = events
        self._pub_events.publish(evmsg)

        if events:
            self._rm.report_collision()

    def _markers(self, footprint: shapely.Polygon, events: list[arena_robots_msgs.msg.CollisionEvent], peds: PedestrianIndex) -> viz.MarkerArray:
        """The robot footprint and the held devices on the ground, red while touched, and a red column at each contact."""
        stamp = self.sim_time.to_msg()
        touched = {ev.obstacle_id for ev in events if ev.kind == arena_robots_msgs.msg.CollisionEvent.KIND_DEVICE}

        def marker(ns: str, mid: int, kind: int, scale: tuple[float, float, float]) -> viz.Marker:
            m = viz.Marker(ns=ns, id=mid, type=kind, action=viz.Marker.ADD)
            m.header.frame_id, m.header.stamp = 'map', stamp
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = scale
            return m

        def ring(geom: shapely.Polygon) -> list[geometry_msgs.msg.Point]:
            coords = list(geom.exterior.coords)
            return [geometry_msgs.msg.Point(x=float(a[0]), y=float(a[1]), z=0.03) for pair in zip(coords, coords[1:], strict=False) for a in pair]

        robot = marker('footprint', 0, viz.Marker.LINE_LIST, (0.03, 0.0, 0.0))
        robot.points = ring(footprint)
        robot.color = _RED if events else _GREY
        devices = marker('devices', 0, viz.Marker.LINE_LIST, (0.025, 0.0, 0.0))
        points: list[geometry_msgs.msg.Point] = []
        colors: list[std_msgs.msg.ColorRGBA] = []
        for p, device in peds.held_devices():
            outline = ring(device)
            points.extend(outline)
            colors.extend([_RED if p.name in touched else _DEVICE] * len(outline))
        devices.points, devices.colors, devices.color = points, colors, _DEVICE
        if not devices.points:
            devices.action = viz.Marker.DELETE
        out = [robot, devices]
        for ev in events:
            if ev.kind not in (arena_robots_msgs.msg.CollisionEvent.KIND_DEVICE, arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN):
                continue
            mid = self._marker_ids.setdefault((ev.kind, ev.obstacle_id), len(self._marker_ids))
            column = marker('contacts', mid, viz.Marker.CYLINDER, (0.16, 0.16, 1.2))
            column.pose.position = geometry_msgs.msg.Point(x=ev.obstacle_position.x, y=ev.obstacle_position.y, z=0.6)
            column.color = _CONTACT
            column.lifetime = Duration(sec=1)
            out.append(column)
        return viz.MarkerArray(markers=out)

    @staticmethod
    def _event(kind: str, ped: arena_people_msgs.msg.Pedestrian, x: float, y: float, device: str = "") -> arena_robots_msgs.msg.CollisionEvent:
        return arena_robots_msgs.msg.CollisionEvent(kind=kind, obstacle_id=ped.name, distance=0.0, obstacle_position=geometry_msgs.msg.Point(x=float(x), y=float(y), z=0.0), device=device)

    def _pedestrian_index(self) -> PedestrianIndex:
        """The index of the latest roster, built once per roster for every tracker of the env."""
        return _ROSTER_INDEXES.of(self._peds_msg, self._default_peds_radius)

    def shutdown(self):
        self._unsubscribe_peds()
        self.destroy_node()
