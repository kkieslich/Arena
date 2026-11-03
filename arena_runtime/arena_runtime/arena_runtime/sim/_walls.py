from __future__ import annotations

import math
from collections.abc import Collection

import attrs
from arena_simulation_setup.shared import Obstacle, Wall
from arena_simulation_setup.tree.Wall import WallSegment
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position
from arena_simulation_setup.utils.models import Model, ModelType

WALL_LEAN = 0.01


async def realize_renderable(
    wall: Wall,
    asset_types: Collection[ModelType],
) -> tuple[list[WallSegment], list[tuple[Obstacle, Model]]]:
    """Realize a wall into segments plus the obstacles whose model resolves to one of asset_types.

    Each simulator passes the formats it can render (USD for Isaac, SDF for Gazebo), so the
    realize-and-skip path is identical and only the accepted asset format differs. Obstacles
    with no acceptable model format are dropped (the loader decides).
    """
    segments, obstacles = await wall.assets()
    resolved: list[tuple[Obstacle, Model]] = []
    for obstacle in obstacles:
        try:
            model = await (await obstacle.model.resolve()).model.get(asset_types)
        except FileNotFoundError:
            continue
        if model.type is ModelType.UNKNOWN:
            continue
        resolved.append((obstacle, model))
    return [strip for segment in segments for strip in upright_strips(segment)], resolved


def sloped_box(segment: WallSegment) -> tuple[tuple[float, float, float], Pose] | None:
    """Box size and pose of a segment whose end lies higher or lower than its start, pitched along it and `height` tall measured vertically, None for a level segment or one shorter than it is tall, such as a post, which stands upright."""
    dx, dy, dz = segment.end.x - segment.start.x, segment.end.y - segment.start.y, segment.end.z - segment.start.z
    run = math.hypot(dx, dy)
    if abs(dz) < 1e-9 or math.hypot(run, dz) < segment.height:
        return None
    pitch = math.atan2(dz, run)
    center = Position((segment.start.x + segment.end.x) / 2.0, (segment.start.y + segment.end.y) / 2.0, (segment.start.z + segment.end.z) / 2.0 + segment.height / 2.0)
    return (math.hypot(run, dz), segment.width, segment.height * math.cos(pitch)), Pose(position=center, orientation=Orientation.from_euler((0.0, -pitch, math.atan2(dy, dx))))


def upright_strips(segment: WallSegment) -> list[WallSegment]:
    """A sloped segment as strips stacked up its height, each thin enough that the pitched ends of its box stand within `WALL_LEAN` of vertical, and an upright cap over the teeth at each end."""
    run = math.hypot(segment.end.x - segment.start.x, segment.end.y - segment.start.y)
    dz = segment.end.z - segment.start.z
    if abs(dz) < 1e-9 or math.hypot(run, dz) < segment.height:
        return [segment]
    pitch = math.atan2(abs(dz), run)
    count = math.ceil(segment.height * math.cos(pitch) * math.sin(pitch) / 2.0 / WALL_LEAN - 1e-9)
    step = segment.height / count
    strips = [attrs.evolve(segment, start=Position(segment.start.x, segment.start.y, segment.start.z + k * step), end=Position(segment.end.x, segment.end.y, segment.end.z + k * step), height=step) for k in range(count)]
    if count == 1:
        return strips
    lean = step * math.cos(pitch) * math.sin(pitch) / 2.0
    ux, uy = (segment.end.x - segment.start.x) / run * lean, (segment.end.y - segment.start.y) / run * lean
    caps = [attrs.evolve(segment, start=Position(end.x - ux, end.y - uy, end.z), end=Position(end.x + ux, end.y + uy, end.z)) for end in (segment.start, segment.end)]
    return [*strips, *caps]
