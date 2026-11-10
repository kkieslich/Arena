"""A site as the forecaster reads it: the base world's flow grid, the edited world's free space, the people and the demand.

It mirrors the [reader](../read.py) without writing a scenario or running PEDS:
- **people** come from the unedited base world, its graph and crowd settings, through populate's `crowd` and the reader's
  `_people` (start point, idle stops, speed), drawn at `crowd_seeds` and weighted 1 / their number: the expected crowd,
  not the crowd of any recorded run. The default seeds lie far from the split's seeds, so a forecast never sees the
  people of the runs it is scored on.
- **demand** comes from the edited graph through the reader's `plan_needs`: each need, the objects that serve it, its
  dwell, and each object's pose, capacity and formation. People use an object from its front: its annotated `face`, else
  behind its yaw, as the runtime queues it.
- **free space** is the edited world's occupancy grid as the reader samples it (walls, world objects, the graph's modelled
  objects stamped), at `NAV_SUB` cells per flow cell and side, a cell free when some point of it keeps `CLEAR_M` from
  everything. A locked door (`gated true`) is closed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .maps import FlowGrid
from .model import Demand, Need, Person, Scene

if TYPE_CHECKING:
    import numpy as np

    from ..edit import Site
    from ..read import Floor, Graph

#: crowd seeds of the expected crowd: 16 draws, none of them a split's seed
CROWD_SEEDS = tuple(range(1000, 1016))
#: nav cells per flow cell and side (0.25 m)
NAV_SUB = 2
#: clearance a walking person keeps (m): HumanSim's navmesh inflation radius
CLEAR_M = 0.38
#: samples per nav cell side when testing its clearance
_PROBES = 5
#: half-width of a locked door's closed band (m)
_DOOR_BAND_M = 0.3
#: where people stand to use an object without a formation, and their queue spacing (m): the fountain's line
FRONT_M = 0.8
STEP_M = 1.0


def base_people(base: Site, settings: Mapping[str, Any], crowd_seeds: Sequence[int] = CROWD_SEEDS) -> list[Person]:
    """The people of the unedited `base` site at every crowd seed, as the reader draws them (`read._base`, `read._people`)."""
    from ...scenario import populate as P
    from .. import read as R

    graph = R.read_graph(base.graph)
    floor = R._floor(base.root, base.world, graph)
    robot = R._robot(floor, settings)
    homes = R._homes(floor, robot)
    people = []
    for seed in crowd_seeds:
        crowd = P.crowd(
            floor.zones,
            overrides=dict(graph.purposes),
            homes=homes,
            forbidden=P.robot_zones(robot or {}),
            seed=seed,
            world=base.base,
            count=settings["count"],
            scale=float(settings["scale"]),
            tag="flowedit",
        )
        humans, types = R._people(R.Base(base.base, floor, graph, homes, crowd), settings, {}, seed)
        for human in humans:
            steps = types[human["agent"]["agent_type"]]["sequences"]["idle"]["steps"].values()
            stops = tuple((float(s["target_pose"]["x"]), float(s["target_pose"]["y"])) for s in steps)
            pose = human["pose"]
            start = (float(pose[0]), float(pose[1])) if not isinstance(pose, str) else (stops[0] if stops else None)
            if start is None:
                continue
            people.append(Person(start=start, stops=stops, speed=float(human["agent"]["desired_velocity"]), weight=1.0 / len(crowd_seeds)))
    return people


def _free(grid: FlowGrid, floor: Floor) -> np.ndarray:
    """Nav cells of `grid` (NAV_SUB per cell and side, row 0 at y0) where some probe point keeps CLEAR_M from walls and objects."""
    import numpy as np

    res = grid.cell / NAV_SUB
    xs, ys = grid.centres(NAV_SUB)
    occ = floor.grid
    if occ is None:
        return grid.in_rooms(xs, ys)
    dist = occ.dist
    rows, cols = dist.shape
    best = np.zeros(xs.shape)
    offsets = (np.arange(_PROBES) - (_PROBES - 1) / 2) * (res / _PROBES)
    for dx in offsets:
        for dy in offsets:
            lx, ly = xs + dx - floor.origin[0], ys + dy - floor.origin[1]
            r = rows - 1 - np.floor((ly - occ.oy) / occ.res).astype(np.int64)
            c = np.floor((lx - occ.ox) / occ.res).astype(np.int64)
            inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            best = np.maximum(best, np.where(inside, dist[np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)], 0.0))
    return best >= CLEAR_M


def _close_locked(free: np.ndarray, grid: FlowGrid, site: Site, floor: Floor, notes: list[str]) -> np.ndarray:
    """`free` with every door the graph gates as locked closed across its span."""
    import shapely

    locked = {str(f.subject).partition(":")[2] for f in site.graph.facts if f.verb == "gated" and f.object is True}
    if not locked:
        return free
    xs, ys = grid.centres(NAV_SUB)
    level = site.world.levels.get(floor.level_id)
    for door in level.doors if level is not None else []:
        if door.name in locked:
            band = shapely.LineString([(door.start.x + floor.origin[0], door.start.y + floor.origin[1]), (door.end.x + floor.origin[0], door.end.y + floor.origin[1])]).buffer(_DOOR_BAND_M)
            free = free & ~shapely.contains_xy(band, xs, ys)
            notes.append(f"door:{door.name}: locked, closed to everyone")
    return free


def _facing(site: Site, model: str, yaw: float) -> float:
    """The direction (rad) from an object to where people use it: its annotated front, else behind its yaw, as the runtime queues it."""
    from ..generate import front_angle, resolve_footprints

    if not model:
        return yaw + math.pi
    resolve_footprints(site, [model])
    return yaw + front_angle(site.root, model)


def demands(site: Site, graph: Graph, floor: Floor, settings: Mapping[str, Any], notes: list[str]) -> tuple[list[Demand], dict[str, Need]]:
    """The objects people seek per need on the people's floor, as the reader plans them (`read.plan_needs`), and each need's rates."""
    from .. import read as R
    from .. import settings as S

    plans = R.plan_needs(graph, settings, floor.level_id, notes)
    origins = R._origins(site.root, site.world)
    out, needs = [], {}
    for need, plan in plans.items():
        rates = S.need(settings, need)
        needs[need] = Need(initial=rates["initial"], decay=rates["decay_rate"], below=float(rates["below"]), amount=float(plan.amount))
        for name in plan.objects:
            t = graph.objects[name]
            ox, oy = origins.get(t.floor, (0.0, 0.0))
            params = (t.formation or {}).get("params") or {}
            front = float(params.get("front_offset", t.interaction_radius or FRONT_M))
            out.append(
                Demand(
                    name=name,
                    need=need,
                    at=(t.x + ox, t.y + oy),
                    facing=_facing(site, t.model, t.yaw),
                    dwell=float(plan.dwell["mean"]),
                    capacity=max(1, int(t.capacity or 1)),
                    front=front,
                    step=float(params.get("base_step", STEP_M)),
                )
            )
    return out, needs


def scene_of(site: Site, settings: Mapping[str, Any], duration: float, *, grid: FlowGrid, people: list[Person]) -> Scene:
    """The scene of `site` on its base world's `grid`, with the base world's `people` (`base_people`)."""
    from .. import read as R

    notes: list[str] = []
    graph = R.read_graph(site.graph)
    floor = R._floor(site.root, site.world, graph)
    free = _close_locked(_free(grid, floor), grid, site, floor, notes)
    found, needs = demands(site, graph, floor, settings, notes)
    if settings.get("entrances", {}).get("rate"):
        notes.append("entrances: visitors through exterior doors are not forecast yet, only the placed people")
    return Scene(grid=grid, free=free, sub=NAV_SUB, people=people, demands=found, needs=needs, duration=float(duration), stand=float(settings["idle"]["stand"]["mean"]), name=site.name, notes=notes)


__all__ = ["CLEAR_M", "CROWD_SEEDS", "NAV_SUB", "base_people", "demands", "scene_of"]
