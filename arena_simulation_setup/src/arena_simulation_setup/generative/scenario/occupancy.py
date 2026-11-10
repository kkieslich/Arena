"""The level occupancy grid at the runtime map's resolution with world furniture stamped on."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .formats import is_world, load_world

if TYPE_CHECKING:
    import numpy as np
    import shapely

    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.model import Level, World
    from arena_simulation_setup.tree.World.view import WorldView

RESOLUTION = 0.05
UNKNOWN_FOOTPRINT_RADIUS = 1.0


def level_of(world: World) -> Level | None:
    """The level the generator works on: the first one with zones, else the first."""
    return next((level for level in world.levels.values() if level.zones), next(iter(world.levels.values()), None))


def _footprints(view: WorldView, world: World, objects: list[Obstacle]) -> list[shapely.Polygon | None]:
    """Each object's `Obstacle.footprint()` with the world's own and linked assets resolvable, as the runtime stamps it."""
    import asyncio

    from arena_simulation_setup.tree import Identifier
    from arena_simulation_setup.tree.World.view import world_assets

    async def one(obj: Obstacle) -> shapely.Polygon | None:
        import shapely

        footprint = await obj.footprint()
        # a floor marking (a waiting bay) takes no floor, as the runtime's world manager leaves it out
        return shapely.Polygon() if footprint is None and await obj.floor_marking() else footprint

    async def collect() -> list[shapely.Polygon | None]:
        return list(await asyncio.gather(*(one(obj) for obj in objects)))

    with world_assets(view, world):
        return Identifier._run_sync(collect())


def world_statics(world_root: str | Path, world: World | None = None) -> list[dict[str, Any]]:
    """Every object of the level with its footprint `polygon`, a covering `radius` and whether its asset gave the footprint;
    floor markings (no box in their annotation, such as a waiting bay) are left out, as the runtime leaves them out."""
    import shapely

    from arena_simulation_setup.tree.World.view import WorldView

    view = WorldView(Path(world_root).resolve())
    if world is None:
        world = load_world(world_root)
    level = level_of(world)
    if level is None:
        return []
    out: list[dict[str, Any]] = []
    for obj, footprint in zip(level.objects, _footprints(view, world, level.objects), strict=True):
        if footprint is not None and footprint.is_empty:
            continue
        x, y = obj.pose.position.x, obj.pose.position.y
        resolved = footprint is not None
        if footprint is None:
            r = UNKNOWN_FOOTPRINT_RADIUS
            footprint = shapely.box(x - r, y - r, x + r, y + r)
        polygon = list(footprint.exterior.coords)[:-1]
        out.append({"name": obj.name, "model": obj.model.name, "pose": [x, y], "polygon": polygon, "radius": max(math.hypot(px - x, py - y) for px, py in polygon), "resolved": resolved})
    return out


def _stamp(obj: Mapping[str, Any]) -> shapely.Polygon | None:
    """The polygon a static is stamped as: its footprint, else the box of its radius around its pose."""
    import shapely

    poly = obj.get("polygon")
    if poly is not None:
        return shapely.Polygon(poly)
    pose = obj.get("pose")
    if not isinstance(pose, (list, tuple)) or len(pose) < 2:
        return None
    radius = float(obj.get("radius") or obj.get("interaction_radius") or 0.5)
    x, y = float(pose[0]), float(pose[1])
    return shapely.box(x - radius, y - radius, x + radius, y + radius)


def level_grid(world_root: str | Path, statics: Sequence[Mapping[str, Any]] = (), world: World | None = None) -> tuple[np.ndarray, float, float, float] | None:
    """The level's walls at `RESOLUTION` with `statics` stamped, as (occupied, resolution, origin x, origin y), or None."""
    root = Path(world_root)
    if not is_world(root):
        return None
    level = level_of(world if world is not None else load_world(root))
    if level is None:
        return None
    footprints = [(str(obj.get("name", "")), poly) for obj in statics if (poly := _stamp(obj)) is not None]
    if footprints:
        grid, origin = level.render_grid(resolution=RESOLUTION, asset_color="black", asset_name_color=None, static_objects=footprints)
    else:
        grid, origin = level.render_grid(resolution=RESOLUTION)
    return grid != 255, RESOLUTION, float(origin[0]), float(origin[1])
