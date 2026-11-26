"""INFRA's walkway rule (study design §3): the width a layout leaves the crowd where the crowd walks, by its flow.

**The walker floor** (`WalkerFloor`) is the floor HumanSim's walkers walk on, built from what the runtime hands HumanSim,
not from the robot map (which inflates obstacles, holds lidar-only ones and has neither the crowd walls nor the zone
edges):

- the level's zones, the floor; HumanSim itself knows no floor edges, so where a zone edge has no wall the two differ
  (`WalkerFloor.bare_edges`);
- the walls the simulator builds (`Level.closed_wall_segments`: paths, zone walls, stair walls and the stairs' open
  walls) and the crowd's walls across exterior doors (`Level.outside_walls`);
- every object whose model resolves with a bounding box, its box at its pose, whatever its height: HumanSim walls every
  static obstacle's box off (`AddObstacles`) and skips one whose model has none;
- the docks of docked objects, and in a layout the elements' floor (`occupied`): footprints, and docks.

**Occupancy.** The fleet walls the docks of the robots' homes and chargers off for the crowd while a robot parks there
(`keep_crowd_off`), most of a run: those, and every footprint, are permanent floor. A locker's dock holds a robot only
while it delivers (`infra.UNLOAD_S` at the dock, a few deliveries an hour: about 1 % of the time, `dock_share`), the
loading point's slot a robot while it loads: those are intermittent, no obstacle to the walkers' routes or to the
walkway widths. They get their own check (`narrowings(docked=True)`): while a robot stands in the dock, every stretch
it narrows keeps `ONE_WAY_M`, and enough of it that the walkers of both directions get through taking turns
(`docked_width`). Walkers meeting at such a short neck wait for each other (HumanSim's turn-taking, with its door
deadlock fixed); the robot leaves after its hold.

**Routes** (`Routes`) are the crowd's shortest paths on the walker floor at HumanSim's planning clearance
(`ROUTE_CLEARANCE_M`), between the sources and sinks of the site's crowd scenario (`WalkerDemand`: each source's rate
split over its sinks by their `sink_affinity` weights), wheelchair users without stairs and escalators
(`NO_CLIMB_TYPES`), the other walkers also over the scenario's teleport zones and the world's own stairs and escalators
(`links_of`, priced as HumanSim prices a ride). Each carries its flow [ped/s].

**Narrowings** (`narrowings`). Within `WINDOW_M` of an element, the crowd's route pieces start and end at the
window's openings: where they cross its edge (`PORT_BIN_M` bins along it) and where they end inside it (a source or
sink, a stair's mouth; `PORT_BIN_M` bins of the floor). Between two openings the walkers take the shortest way and
every other way within `ROUTE_SLACK` of it that passes another side of an obstacle (HumanSim weighs its routes by
congestion, so a way a few per cent longer carries walkers too), on the floor before the layout and after it. A way's
width is the shortest distance between the obstacles on its left and those on its right: every line from one side to
the other crosses the way, so this is the narrowest cross-section it passes, a neck on a diagonal (between a dock's
corner and a plant's) as narrow as it is, and a corner the way turns round is none. Only gaps within `MEASURE_M` of the
element count: the others it cannot change. A neck needs `TWO_WAY_M` where `FLOW_MIN_PED_H` or more walkers cross it
each way, `ONE_WAY_M` where they cross it one way: every walker whose way crosses the neck counts, whichever openings
they walk between, so walkers from two openings who meet head-on in one corridor count as two-way. A layout narrows a
stretch when, after it,

- a way is narrower than its neck's requirement and than the same way before (`WIDTH_TOL_M`), or
- the narrowest way is narrower than its requirement and than the narrowest way before: the layout moved the walkers
  into a narrow passage of the site, or
- no way is left between two openings: it closes the walkway.

A stretch that was narrower than its requirement before and that the layout leaves as it was is the site's, not the
layout's. Two openings the routes join with no way between them in the window cannot be measured: that fails, never
passes. A crowd some of whose walkers find no route at all is refused (`place.crowd_routes`).

**Why these widths.** Two-way: 1.8 m, "a clear width of 1800mm will allow two wheelchair users to pass each other"
(Singapore BCA Code on Accessibility in the Built Environment 2019, 4.2.1.2 and Fig. 15; the twin is in Singapore). It
is also HumanSim's own two-way width for a wheelchair (`wheelchair_manual` passage block: body plus the widest body plus
0.1 m, 1.7 m, rounded to the code): below it walkers meeting in the stretch take turns. A permanent neck on a two-way
route makes every meeting a wait, all run long; a robot in a dock does so only while it stands there, so the docked
check asks for a lane, not for passing. One-way: 0.9 m, the clear width of an accessible route (2010 ADA 403.5.1, 36 in =
0.915 m), one wheelchair. HumanSim routes through gaps of twice its planning clearance (0.8 m) and wider, so a gap between
0.8 m and 1.8 m on a two-way route is where its walkers take turns.
"""

from __future__ import annotations

import math
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse
    from shapely.geometry.base import BaseGeometry

    from .edit import Site

Point = tuple[float, float]

#: A stretch carrying walkers both ways keeps this clear width: two wheelchair users pass (BCA 2019 4.2.1.2) [m]
TWO_WAY_M = 1.8
#: A stretch carrying walkers one way keeps this: one wheelchair (2010 ADA 403.5.1, 36 in) [m]
ONE_WAY_M = 0.9
#: A direction of a stretch carries walkers when its flow is at least this: one walker every 10 minutes [ped/h]
FLOW_MIN_PED_H = 6.0
#: HumanSim's planning clearance: its global planner keeps paths `global_planner.inflation_radius` (0.38 m) from walls,
#: rounded up to its 0.2 m cells; walkers of every type plan with it (their radii are 0.25-0.325 m) [m]
ROUTE_CLEARANCE_M = 0.4
#: A way within this share of the shortest carries walkers too (HumanSim's congestion-weighted route costs)
ROUTE_SLACK = 0.10
#: ... and within this absolute length, so a short stretch's ways are not split by a cell [m]
ROUTE_SLACK_M = 0.5
#: The routes within this of an element's occupied floor are what it is checked against [m]
WINDOW_M = 6.0
#: A width counts as narrowed when it is at least this much under the width before [m]
WIDTH_TOL_M = 0.1
#: Cell sizes: the whole level's routes, the window's ways and widths [m]
ROUTE_RES_M, WINDOW_RES_M = 0.1, 0.05
#: Ways tried beside the shortest between two openings, each round another side of an obstacle
ALTERNATIVES = 2
#: Gaps an element narrows lie within this of its floor: a gap under `TWO_WAY_M` with it at one end, and the passage
#: beside it its walkers may move into [m]
MEASURE_M = TWO_WAY_M + 1.0
#: Where routes cross the window's edge, binned this far apart; the walkers' ways start within a bin of it [m]
PORT_BIN_M = 1.0
#: A path end in the window runs on to a wall; the wall this near that point lies on both of its sides [m]
END_M = 1.0
#: Walls are this thick on the walker floor (`generate.WALL_GAP_M` either side of the line) [m]
WALL_M = 0.1
#: Walker types that take no stairs or escalators (HumanSim's `climbs: []`): wheelchair users take the ramps
NO_CLIMB_TYPES = ("wheelchair_manual", "wheelchair_electric")
#: Walking speed a teleport zone's delay is counted at, and the step on and off it [m/s], [m]
LINK_SPEED_MPS, LINK_STEP_M = 1.1, 2.0
#: How fast walkers climb a still stair (HumanSim's `teleport_zones.STAIR_SPEED`) [m/s]
STAIR_SPEED_MPS = 0.7
#: A link's ends snap to the routable cell nearest them within this, `links.MOUTH_SNAP_M`; a link the routes cannot
#: reach is refused [m]
LINK_SNAP_M = 1.5
#: The elements whose docks the fleet walls off for the crowd while a robot parks there: its homes and chargers
PARKED = ("charger_pair", "bay_pair")
#: The most walkers a metre of walkway width passes per second (Weidmann 1993's maximal specific flow, 1.22 ped/(m s)),
#: and the share of it a neck keeps when its two directions take turns (one direction at a time: half, at best)
SPECIFIC_FLOW_PED_MS, TURNS_SHARE = 1.22, 0.5


# --- the crowd's demand ------------------------------------------------------------------------------------------------


@attrs.frozen(cache_hash=True)
class Flow:
    """Walkers from region `source` to region `sink` at `rate` [ped/s]; `climbs`: they may take stairs and escalators."""

    source: str
    sink: str
    rate: float
    climbs: bool = True


@attrs.frozen(cache_hash=True)
class Link:
    """A teleport zone half walkers board at `entry` and leave at each of `exits` (its destinations' entries), costing
    `cost_m` of walking."""

    name: str
    entry: Point
    exits: tuple[Point, ...]
    cost_m: float


@attrs.frozen(cache_hash=True)
class WalkerDemand:
    """Who walks where: the crowd scenario's source and sink `regions` (name, world-frame polygon), its `flows` and the
    teleport zones (`links`) climbing walkers may take."""

    regions: tuple[tuple[str, tuple[Point, ...]], ...]
    flows: tuple[Flow, ...]
    links: tuple[Link, ...] = ()

    @classmethod
    def from_scenario(cls, doc: Mapping[str, Any]) -> WalkerDemand:
        """The demand of an arena-scenario/2 document: each source region's mean rate split over its `sink_affinity`
        sinks by weight (a source without one feeds every sink alike), by its agent type's climbing."""
        regions: dict[str, tuple[Point, ...]] = {}
        for name, region in sorted((doc.get("regions") or {}).items()):
            if region.get("type") in ("source", "sink") and len(region.get("polygon") or []) >= 3:
                regions[name] = tuple((float(p["x"]), float(p["y"])) for p in region["polygon"])
        sinks = [name for name, r in sorted((doc.get("regions") or {}).items()) if r.get("type") == "sink" and name in regions]
        flows: dict[tuple[str, str, bool], float] = {}
        for name, region in sorted((doc.get("regions") or {}).items()):
            if region.get("type") != "source" or name not in regions:
                continue
            config = region.get("config") or {}
            profile = config.get("rate_profile") or []
            rate = sum(float(p.get("rate", 0.0)) for p in profile) / len(profile) if profile else 0.0
            agent = config.get("agent") or {}
            climbs = str(agent.get("agent_type", "")).rsplit("/", 1)[-1].removesuffix(".yaml") not in NO_CLIMB_TYPES
            affinity = [(a["sink"], float(a.get("weight", 1.0))) for a in agent.get("sink_affinity") or [] if a.get("sink") in regions] or [(s, 1.0) for s in sinks]
            total = sum(w for _, w in affinity)
            for sink, weight in affinity:
                if rate > 0.0 and weight > 0.0 and total > 0.0:
                    key = (name, sink, climbs)
                    flows[key] = flows.get(key, 0.0) + rate * weight / total
        return cls(tuple(regions.items()), tuple(Flow(s, t, round(q, 9), c) for (s, t, c), q in sorted(flows.items())), links_of(doc.get("teleport_zones") or []))

    def with_links(self, zones: Iterable[Mapping[str, Any]]) -> WalkerDemand:
        """The demand with the teleport zones `zones` (HumanSim's form, e.g. the world's stairs, `links.world_links`) as
        links too."""
        return attrs.evolve(self, links=(*self.links, *links_of(zones)))

    @classmethod
    def load(cls, path: Path) -> WalkerDemand:
        """The demand of a scenario file or directory (its scenario.yaml)."""
        import yaml

        path = Path(path)
        path = path / "scenario.yaml" if path.is_dir() else path
        return cls.from_scenario(yaml.safe_load(path.read_text()) or {})

    @property
    def total(self) -> float:
        return sum(f.rate for f in self.flows)


def links_of(zones: Iterable[Mapping[str, Any]]) -> tuple[Link, ...]:
    """Teleport zones in HumanSim's form as links, each half to each of its destinations, costing what HumanSim prices
    the ride at (`teleport_zones._ride_time`), as walking at `LINK_SPEED_MPS`: the boarding half from its entry to its
    hold and the landing half from its hold to its entry, at belt speed or on a still stair at `STAIR_SPEED_MPS`, plus
    the boarding half's `delay`; a zone without a hold steps on and off over `LINK_STEP_M`."""
    by_name = {str(z["name"]): z for z in zones if "entry" in z}

    def point(p: Mapping[str, Any]) -> Point:
        return (float(p["x"]), float(p["y"]))

    def run(z: Mapping[str, Any]) -> float:
        return math.dist(point(z["entry"]), point(z["hold"])) if z.get("hold") else LINK_STEP_M / 2.0

    out = []
    for name, z in sorted(by_name.items()):
        for d in z.get("destinations") or []:
            if d not in by_name:
                continue
            far = by_name[d]
            belt = abs(float(z.get("belt") or 0.0))
            ride = (run(z) + run(far)) / (belt if belt > 0.0 else STAIR_SPEED_MPS) + float(z.get("delay") or 0.0)
            out.append(Link(f"{name}>{d}", point(z["entry"]), (point(far["entry"]),), round(ride * LINK_SPEED_MPS, 3)))
    return tuple(out)


# --- the walker floor --------------------------------------------------------------------------------------------------


@attrs.frozen(eq=False)
class Occupied:
    """Floor the crowd cannot walk on: `name`, its polygon, and whether it is occupied only at times (`intermittent`)."""

    name: str
    geometry: BaseGeometry
    intermittent: bool = False


@attrs.frozen(eq=False)
class WalkerFloor:
    """The floor of one level the crowd walks on (see the module doc): the zones' `floor`, the wall `segments`, the
    `obstacles` (objects and docks)."""

    level_id: str
    floor: BaseGeometry
    segments: tuple[tuple[Point, Point], ...]
    obstacles: tuple[Occupied, ...]
    _free: BaseGeometry | None = attrs.field(default=None, init=False)
    _free_docked: BaseGeometry | None = attrs.field(default=None, init=False)

    @property
    def walls(self) -> BaseGeometry:
        import shapely

        lines = [shapely.LineString(s) for s in self.segments if s[0] != s[1]]
        return shapely.union_all(lines).buffer(WALL_M / 2.0) if lines else shapely.Polygon()

    @property
    def free(self) -> BaseGeometry:
        """The free floor: the zones less walls and permanent obstacles (the crowd's routes run on it)."""
        free = self._free
        if free is None:
            import shapely

            blocked = shapely.union_all([self.walls, *(o.geometry for o in self.obstacles if not o.intermittent)])
            free = self.floor.difference(blocked)
            object.__setattr__(self, "_free", free)
        return free

    @property
    def free_docked(self) -> BaseGeometry:
        """The free floor while every intermittent dock is occupied."""
        free = self._free_docked
        if free is None:
            import shapely

            docks = [o.geometry for o in self.obstacles if o.intermittent]
            free = self.free.difference(shapely.union_all(docks)) if docks else self.free
            object.__setattr__(self, "_free_docked", free)
        return free

    def bare_edges(self, within: float = 0.15) -> BaseGeometry:
        """The floor's edges with no wall within `within`: where HumanSim, which knows walls only, could walk off it."""
        import shapely

        edges = self.floor.boundary.difference(self.walls.buffer(within))
        return shapely.union_all([g for g in getattr(edges, "geoms", [edges]) if g.length >= 0.3]) if not edges.is_empty else edges

    def export(self) -> dict[str, Any]:
        """The floor in HumanSim's terms: wall segments (AddWalls), the obstacles' outlines (AddObstacles walls their
        boxes off; crowd walls for docks), the floor's outline and its bare edges."""
        import shapely

        def rings(g: BaseGeometry) -> list[list[list[float]]]:
            polys = [p for p in getattr(g, "geoms", [g]) if isinstance(p, shapely.Polygon) and not p.is_empty]
            return [[[round(x, 3), round(y, 3)] for x, y in p.exterior.coords] for p in polys]

        bare = self.bare_edges()
        return {
            "level": self.level_id,
            "walls": [[[round(a[0], 3), round(a[1], 3)], [round(b[0], 3), round(b[1], 3)]] for a, b in self.segments],
            "obstacles": [{"name": o.name, "intermittent": o.intermittent, "outline": rings(o.geometry)} for o in self.obstacles],
            "floor": rings(self.floor),
            "bare_edges_m": round(float(bare.length), 2),
        }


def walker_floor(site: Site, level_id: str, ignore: Iterable[str] = ()) -> WalkerFloor:
    """The walker floor of `site`'s level `level_id` (see the module doc), without the objects `ignore` (graph ids,
    `obj:<name>`, e.g. a layout's own elements, which it places anew)."""
    import shapely

    from . import generate as gen

    ignore = frozenset(ignore)
    level = site.world.levels[level_id]
    shapes = [z.shape() for z in level.zones if len(z.polygon) >= 3]
    floor = shapely.union_all([s.buffer(0) for s in shapes]) if shapes else shapely.Polygon()
    segments = [tuple(map(tuple, s)) for s in level.closed_wall_segments()]
    segments += [((w.start.x, w.start.y), (w.end.x, w.end.y)) for w in level.outside_walls()]
    here: dict[str, tuple[dict[str, Any], str]] = {}
    for obj in gen.objects(site):
        placed = gen._at(site, obj)
        model = gen._model(site, obj)
        if placed is None or placed[0] != level_id or model is None or obj in ignore:
            continue
        here[obj] = (placed[1], model)
    from .place import resolve_models

    resolve_models(site, sorted({model for _, model in here.values()}))
    obstacles = [Occupied(obj, gen.footprint(site.root, model, float(pose["x"]), float(pose["y"]), float(pose["yaw"]))) for obj, (pose, model) in sorted(here.items()) if gen._FOOTPRINTS.get((site.root, model)) is not None]
    for o in level.objects:
        corners = o.dock_polygon()
        if corners is not None and f"obj:{o.name}" not in ignore:
            obstacles.append(Occupied(f"obj:{o.name}#dock", shapely.Polygon([(p.x, p.y) for p in corners]), intermittent=not _parked_member(o.name)))
    return WalkerFloor(level_id, floor, tuple(segments), tuple(obstacles))  # type: ignore[arg-type]


def _parked_member(name: str) -> bool:
    from .place import CATALOG

    return any(name in CATALOG[e].members for e in PARKED if e in CATALOG)


def occupied(spots: Iterable[Any]) -> list[Occupied]:
    """The floor `spots` (place.Spot) take from the crowd: footprints, permanent; docks, permanent at the fleet's homes
    and chargers (`PARKED`), intermittent at a locker and the loading point."""
    out = []
    for s in spots:
        parked = s.element.name in PARKED
        for o in s.objects:
            if s.element.blocks_floor:
                out.append(Occupied(f"{s.element.name}:{o.name}", o.footprint))
            out.append(Occupied(f"{s.element.name}:{o.name}#dock", o.dock, intermittent=not parked))
    return out


# --- rasters -----------------------------------------------------------------------------------------------------------


@attrs.frozen(eq=False)
class _Grid:
    """A raster over `bounds` at `res`: cell (row, col) has its centre at (x0 + (col + 0.5) res, y0 + (row + 0.5) res)."""

    x0: float
    y0: float
    res: float
    shape: tuple[int, int]

    @classmethod
    def over(cls, bounds: Sequence[float], res: float) -> _Grid:
        x0, y0, x1, y1 = bounds
        return cls(float(x0), float(y0), res, (max(1, math.ceil((y1 - y0) / res)), max(1, math.ceil((x1 - x0) / res))))

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        xs = self.x0 + (np.arange(self.shape[1]) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.shape[0]) + 0.5) * self.res
        xx, yy = np.meshgrid(xs, ys)
        return xx, yy

    def cells(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.floor((xy[:, 1] - self.y0) / self.res).astype(int), np.floor((xy[:, 0] - self.x0) / self.res).astype(int)

    def xy(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        import numpy as np

        return np.column_stack([self.x0 + (cols + 0.5) * self.res, self.y0 + (rows + 0.5) * self.res])

    def inside(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return (rows >= 0) & (rows < self.shape[0]) & (cols >= 0) & (cols < self.shape[1])

    def raster(self, geometry: BaseGeometry) -> np.ndarray:
        """True at the cells whose centre `geometry` covers."""
        import numpy as np
        import shapely

        if geometry.is_empty:
            return np.zeros(self.shape, dtype=bool)
        xx, yy = self.centres()
        clipped = shapely.clip_by_rect(geometry, self.x0 - self.res, self.y0 - self.res, self.x0 + (self.shape[1] + 1) * self.res, self.y0 + (self.shape[0] + 1) * self.res)
        shapely.prepare(clipped)
        return shapely.contains_xy(clipped, xx, yy)


#: The 16-neighbourhood: steps (drow, dcol) and the cells each must keep free on its way
_STEPS = (((0, 1), ()), ((1, 0), ()), ((1, 1), ((0, 1), (1, 0))), ((1, -1), ((0, -1), (1, 0))), ((1, 2), ((0, 1), (1, 1))), ((2, 1), ((1, 0), (1, 1))), ((1, -2), ((0, -1), (1, -1))), ((2, -1), ((1, 0), (1, -1))))


def _graph(routable: np.ndarray, res: float, extra: Sequence[tuple[int, int, float]] = ()) -> tuple[Any, np.ndarray]:
    """The 16-connected graph of the `routable` cells (both ways), plus directed `extra` edges (node, node, length), and
    the node index of every cell (-1 off it)."""
    import numpy as np
    import scipy.sparse

    h, w = routable.shape
    index = np.full(routable.shape, -1, dtype=np.int64)
    index[routable] = np.arange(int(routable.sum()))
    rows, cols, lengths = [], [], []
    for (dr, dc), via in _STEPS:
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        ok = routable[r0:r1, c0:c1] & routable[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        for vr, vc in via:
            ok &= routable[r0 + vr : r1 + vr, c0 + vc : c1 + vc]
        a = index[r0:r1, c0:c1][ok]
        b = index[r0 + dr : r1 + dr, c0 + dc : c1 + dc][ok]
        step = res * math.hypot(dr, dc)
        rows += [a, b]
        cols += [b, a]
        lengths += [np.full(len(a), step), np.full(len(a), step)]
    for a, b, length in extra:
        rows.append(np.array([a]))
        cols.append(np.array([b]))
        lengths.append(np.array([length]))
    n = int(routable.sum())
    if not rows:
        return scipy.sparse.csr_matrix((n, n)), index
    graph = scipy.sparse.csr_matrix((np.concatenate(lengths), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return graph, index


def _edt(mask: np.ndarray) -> np.ndarray:
    """Each cell's distance to the nearest False cell of `mask`, in cells."""
    import scipy.ndimage

    return typing.cast("np.ndarray", scipy.ndimage.distance_transform_edt(mask))


def _edt_nearest(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`_edt`, and the row and the column of each cell's nearest False cell."""
    import scipy.ndimage

    dist, nearest = typing.cast("tuple[np.ndarray, np.ndarray]", scipy.ndimage.distance_transform_edt(mask, return_indices=True))
    return dist, nearest[0], nearest[1]


def _dilate(mask: np.ndarray) -> np.ndarray:
    """`mask` grown by a cell in all eight directions, as a bool array."""
    import numpy as np
    import scipy.ndimage

    return np.asarray(scipy.ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool)), dtype=bool)


def _clearance(free: np.ndarray, res: float) -> np.ndarray:
    """Each cell's distance to the nearest blocked cell [m] (0 on blocked cells)."""
    return _edt(free) * res


# --- the crowd's routes ------------------------------------------------------------------------------------------------


@attrs.frozen(eq=False)
class Route:
    """One route piece: its points (world frame), its flow [ped/s] and its flows' source and sink."""

    xy: np.ndarray
    rate: float
    source: str
    sink: str


@attrs.frozen(eq=False)
class Routes:
    """The crowd's routes on one level's walker floor; `missing` lists the flows without one (no path at the clearance)."""

    routes: tuple[Route, ...]
    missing: tuple[Flow, ...]
    _tree: Any = attrs.field(default=None, init=False)

    @property
    def lines(self) -> list[Any]:
        import shapely

        return [shapely.LineString(r.xy) if len(r.xy) > 1 else shapely.Point(r.xy[0]).buffer(1e-3) for r in self.routes]

    def require_all(self, what: str = "the crowd") -> None:
        """ValueError naming the flows that found no route: a rule measured against the others would pass every spot
        the lost walkers would have walked past."""
        if self.missing:
            lost = sorted({(f.source, f.sink, f.climbs) for f in self.missing})
            rate = sum(f.rate for f in self.missing) * 3600.0
            listed = ", ".join(f"{a} -> {b}{'' if c else ' (no stairs)'}" for a, b, c in lost[:12]) + (f" and {len(lost) - 12} more" if len(lost) > 12 else "")
            raise ValueError(f"{what}: {len(lost)} flows ({rate:.0f} ped/h) find no route on the walker floor at {ROUTE_CLEARANCE_M} m clearance: {listed}")

    def near(self, area: BaseGeometry) -> list[Route]:
        """The routes that cross `area`."""
        import shapely

        tree = self._tree
        if tree is None:
            tree = shapely.STRtree(self.lines)
            object.__setattr__(self, "_tree", tree)
        return [self.routes[int(i)] for i in tree.query(area, predicate="intersects")]

    @classmethod
    def build(cls, floor: WalkerFloor, demand: WalkerDemand, res: float = ROUTE_RES_M, clearance: float = ROUTE_CLEARANCE_M) -> Routes:
        """Shortest paths at `clearance` on `floor` for every flow of `demand` (see the module doc), on a `res` grid."""
        import numpy as np
        import scipy.sparse.csgraph
        import shapely

        free_poly = floor.free
        grid = _Grid.over(shapely.union_all([free_poly, *(shapely.Polygon(p) for _, p in demand.regions)]).buffer(clearance + res).bounds, res)
        free = grid.raster(free_poly)
        routable = _clearance(free, res) >= clearance - 1e-9
        base, _ = _graph(routable, res)
        rr, cc = np.nonzero(routable)
        coords = grid.xy(rr, cc)
        import scipy.spatial

        tree = scipy.spatial.KDTree(coords) if len(coords) else None

        def snap(p: Point, reach: float = LINK_SNAP_M) -> int | None:
            if tree is None:
                return None
            d, i = tree.query(p)
            return int(i) if d <= reach else None

        # the links as directed edges between the floor cells nearest their ends, the cheapest of any repeats
        extra: dict[tuple[int, int], float] = {}
        for link in demand.links:
            a = snap(link.entry)
            for exit_ in link.exits:
                b = snap(exit_)
                if a is None or b is None:
                    # a link the routes cannot take would leave its walkers on a detour no one walks: refuse it
                    raise ValueError(f"link {link.name!r}: an end lies over {LINK_SNAP_M:g} m from the walker floor's routable cells (a stair half on a landing: compose it over the landing, `links.floor_links`)")
                if a != b:
                    extra[(a, b)] = min(link.cost_m, extra.get((a, b), math.inf))
        climbing = base
        if extra:
            import scipy.sparse

            (heads, tails), lengths = zip(*extra, strict=True), list(extra.values())
            climbing = base + scipy.sparse.csr_matrix((lengths, (heads, tails)), shape=base.shape)
        graphs = {False: base, True: climbing}
        regions = dict(demand.regions)
        cells: dict[str, np.ndarray] = {}
        for name, points in regions.items():
            poly = shapely.Polygon(points).buffer(0)
            inside = np.nonzero(shapely.contains_xy(poly, coords[:, 0], coords[:, 1]))[0] if len(coords) else np.zeros(0, dtype=int)
            if not len(inside) and tree is not None:
                near = tree.query_ball_point(np.asarray(poly.centroid.coords[0]), r=max(1.0, math.sqrt(poly.area)))
                inside = np.asarray(sorted(near), dtype=int)
            cells[name] = inside
        routes: list[Route] = []
        missing: list[Flow] = []
        by_source: dict[tuple[str, bool], list[Flow]] = {}
        for f in demand.flows:
            by_source.setdefault((f.source, f.climbs), []).append(f)
        for (source, climbs), flows in sorted(by_source.items()):
            start = cells.get(source, np.zeros(0, dtype=int))
            if not len(start):
                missing += flows
                continue
            dist, pred, _ = scipy.sparse.csgraph.dijkstra(graphs[climbs], indices=start, min_only=True, return_predecessors=True)
            for f in flows:
                goal = cells.get(f.sink, np.zeros(0, dtype=int))
                if not len(goal) or not np.isfinite(dist[goal]).any():
                    missing.append(f)
                    continue
                node = int(goal[np.argmin(dist[goal])])
                path = [node]
                while pred[node] >= 0:
                    node = int(pred[node])
                    path.append(node)
                xy = coords[np.asarray(path[::-1])]
                # a teleport zone is a jump: each piece walks on the floor
                jumps = np.nonzero(np.hypot(*np.diff(xy, axis=0).T) > 3.0 * res)[0]
                for piece in np.split(xy, jumps + 1):
                    if len(piece):
                        routes.append(Route(piece, f.rate, f.source, f.sink))
        return cls(tuple(_merge(routes)), tuple(missing))


def _merge(routes: list[Route]) -> list[Route]:
    """Routes along the same cells as one, their flows added."""
    out: dict[tuple[Any, ...], Route] = {}
    for r in routes:
        key = (r.xy.shape, r.xy.tobytes())
        if key in out:
            old = out[key]
            out[key] = Route(old.xy, old.rate + r.rate, old.source, old.sink)
        else:
            out[key] = r
    return list(out.values())


# --- narrowings --------------------------------------------------------------------------------------------------------


@attrs.frozen
class Narrowing:
    """A walkway a layout narrows, between openings `ports` (window edge bins `edge:<run>:<bin>`, route ends inside the
    window `end:<x>:<y>`): `required` and the widths `before` and `after` [m], `at` (the neck's middle), the flow that
    crosses the neck each way [ped/h], what makes it (`by`: occupied floor names), whether it was measured with a robot in
    an intermittent dock (`docked`), its `kind` and the neck's two ends (`gap`). Kinds: `narrowed` (a way narrower than
    before and than its requirement), `moved` (the narrowest way narrower than the narrowest before: walkers moved into a
    narrow passage), `closed` (no way left), `unmeasured` (two openings the crowd's routes join but no way between them
    in the window, or a way whose sides the window does not part: the check cannot say, and fails)."""

    ports: tuple[str, str]
    required: float
    before: float
    after: float
    at: Point
    flow_ab: float
    flow_ba: float
    by: tuple[str, ...]
    docked: bool = False
    kind: str = "narrowed"
    gap: tuple[Point, Point] | None = None

    @property
    def two_way(self) -> bool:
        return self.required >= TWO_WAY_M

    @property
    def moved(self) -> bool:
        return self.kind == "moved"

    def describe(self) -> str:
        what = "a docked robot's neck in a" if self.docked else ("a two-way" if self.two_way else "a one-way")
        if self.kind == "unmeasured":
            return f"cannot measure {what} walkway between {self.ports[0]} and {self.ports[1]} ({self.flow_ab:.0f}/{self.flow_ba:.0f} ped/h): no way between them in the window"
        if self.kind == "closed":
            return f"closes {what} walkway ({self.flow_ab:.0f}/{self.flow_ba:.0f} ped/h, before {self.before:.2f} m) at ({self.at[0]:.1f}, {self.at[1]:.1f}) by {', '.join(self.by)}"
        how = "moves the walkers into" if self.kind == "moved" else "narrows"
        return f"{how} {what} walkway ({self.flow_ab:.0f}/{self.flow_ba:.0f} ped/h) to {self.after:.2f} m (before {self.before:.2f} m, needs {self.required:.2f} m) at ({self.at[0]:.1f}, {self.at[1]:.1f}) by {', '.join(self.by) or 'the site'}"


@attrs.frozen
class Stretch:
    """What `narrowings` measured for one opening pair: the requirement and flows at its narrowest way's neck, and the
    narrowest width before and after (None: no way after), with a robot in the intermittent docks or not (`docked`)."""

    ports: tuple[str, str]
    required: float
    before: float
    after: float | None
    flow_ab: float
    flow_ba: float
    at: Point | None
    docked: bool = False


def _ports(grid: _Grid, routable: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Labels of the window boundary's openings, runs of routable boundary cells along the ring (0 elsewhere), and each
    ring cell's position along the ring [m] (NaN elsewhere)."""
    import numpy as np

    h, w = routable.shape
    ring = [(0, c) for c in range(w)] + [(r, w - 1) for r in range(1, h)] + [(h - 1, c) for c in range(w - 2, -1, -1)] + [(r, 0) for r in range(h - 2, 0, -1)]
    labels = np.zeros(routable.shape, dtype=np.int32)
    arc = np.full(routable.shape, np.nan)
    on = [bool(routable[r, c]) for r, c in ring]
    start = on.index(False) if not all(on) else 0
    n = 0
    prev = False
    for k in range(len(ring)):
        r, c = ring[(start + k) % len(ring)]
        cur = on[(start + k) % len(ring)]
        if cur and not prev:
            n += 1
        if cur:
            labels[r, c] = n
            arc[r, c] = k * grid.res
        prev = cur
    return labels, arc


def _path(graph: scipy.sparse.csr_matrix, index: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float] | None:
    """The shortest path from cells `a` to cells `b` (node ids) as node ids, and its length; None without one."""
    import numpy as np
    import scipy.sparse.csgraph

    if not len(a) or not len(b):
        return None
    dist, pred, _ = scipy.sparse.csgraph.dijkstra(graph, indices=a, min_only=True, return_predecessors=True)
    if not np.isfinite(dist[b]).any():
        return None
    node = int(b[np.argmin(dist[b])])
    length = float(dist[node])
    path = [node]
    while pred[node] >= 0:
        node = int(pred[node])
        path.append(node)
    return np.asarray(path[::-1]), length


@attrs.frozen
class _Width:
    """The narrowest gap between the obstacles left and right of a path near the element [m] and its two ends; no ends
    (`left`, `right` None) where no gap lies within `MEASURE_M` of it, the width then `WIDE_M`. Not `measured` where the
    path and its run-ons part no left side of the window from a right one and the path comes within `MEASURE_M` of the
    element: no width can be read (a path that parts none and keeps beyond `MEASURE_M`, along the window's edge, is
    `WIDE_M`)."""

    width: float
    left: Point | None = None
    right: Point | None = None
    measured: bool = True

    @property
    def middle(self) -> Point | None:
        if self.left is None or self.right is None:
            return None
        return ((self.left[0] + self.right[0]) / 2.0, (self.left[1] + self.right[1]) / 2.0)

    @property
    def gap(self) -> tuple[Point, Point] | None:
        return None if self.left is None or self.right is None else (self.left, self.right)


#: The width of a way with no gap within `MEASURE_M` of the element: wider than any requirement [m]
WIDE_M = 2.0 * MEASURE_M


def _extended(free: np.ndarray, pts: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """`pts` with each end that lies inside the window run on straight, in the path's direction there, to the window's
    edge, through whatever stands in the way (so the run-on and the path part the window into two sides even where the
    floor wraps round a free-standing obstacle); and the first blocked cell each run-on meets."""
    h, w = free.shape
    hits: list[tuple[int, int]] = []
    for at_start in (True, False):
        # the end and the point a few cells back, read from the path as it stands now
        end, prev = (pts[0], pts[min(len(pts) - 1, 5)]) if at_start else (pts[-1], pts[max(0, len(pts) - 6)])
        r, c = end
        if r in (0, h - 1) or c in (0, w - 1):
            continue
        dr, dc = r - prev[0], c - prev[1]
        norm = math.hypot(dr, dc) or 1.0
        ext: list[tuple[int, int]] = []
        hit = False
        for k in range(1, 4 * (h + w)):
            rr, cc = int(round(r + dr / norm * k * 0.5)), int(round(c + dc / norm * k * 0.5))
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if not free[rr, cc] and not hit:
                hits.append((rr, cc))
                hit = True
            if not ext or ext[-1] != (rr, cc):
                ext.append((rr, cc))
        pts = ext[::-1] + pts if at_start else pts + ext
    return pts, hits


def _width(free: np.ndarray, grid: _Grid, rows: np.ndarray, cols: np.ndarray, focus: np.ndarray | None = None) -> _Width:
    """The width of the walkway a path (cells `rows`, `cols`) takes through the window: the shortest distance between the
    blocked cells on its left and those on its right. Every line from one side to the other crosses the path, so this is
    the narrowest cross-section the path passes, on a diagonal too, and a corner it turns round is no cross-section. A
    path end inside the window runs on straight to the window's edge, which parts the sides; the blocked cells within
    `END_M` of the first it meets are on both sides and left out, and a gap across that run-on is none of the way's
    (only blocked cells nearer the walked path than the run-on count). With `focus` (a cell mask), only blocked cells
    there count: the gaps an element can narrow have it at one end. `WIDE_M` without a gap there."""
    import numpy as np
    import scipy.ndimage
    import scipy.spatial

    res = grid.res
    pts = list(zip(rows.tolist(), cols.tolist(), strict=True))
    if len(pts) < 2:
        return _Width(WIDE_M)
    walked = np.zeros(free.shape, dtype=bool)
    walked[rows, cols] = True
    pts, hits = _extended(free, pts)
    # the path's cells and the cells halfway between neighbours (16-connected steps skip at most one), a solid line
    # once grown by a cell
    cells = np.asarray(pts, dtype=float)
    halves = np.rint((cells[1:] + cells[:-1]) / 2.0)
    marked = np.vstack([cells, halves]).astype(int)
    barrier = np.zeros(free.shape, dtype=bool)
    barrier[marked[:, 0], marked[:, 1]] = True
    grow = np.ones((3, 3), dtype=bool)
    barrier = _dilate(barrier) & free
    labelled = typing.cast("tuple[np.ndarray, int]", scipy.ndimage.label(free & ~barrier, structure=grow))
    sides, count = labelled
    path = np.asarray(pts, dtype=float)
    tangent = np.gradient(path, axis=0)
    tree = scipy.spatial.KDTree(path)
    touching = _dilate(barrier)
    left = np.zeros(free.shape, dtype=bool)
    right = np.zeros(free.shape, dtype=bool)
    for label in range(1, count + 1):
        region = sides == label
        edge = region & touching
        if not edge.any():
            continue  # no side of the path: a pocket off it, beyond an obstacle
        er, ec = np.nonzero(edge)
        _, k = tree.query(np.column_stack([er, ec]).astype(float))
        cross = tangent[k, 0] * (ec - path[k, 1]) - tangent[k, 1] * (er - path[k, 0])
        (left if float(np.sum(np.sign(cross))) > 0 else right)[region] = True
    if not left.any() or not right.any():
        # the path parts no left side from a right one: where it keeps beyond `MEASURE_M` of the element (along the
        # window's edge, `WINDOW_M` off), no neck can have the element at one end; nearer, nothing can be read
        if focus is not None and not focus[rows, cols].any():
            return _Width(WIDE_M)
        return _Width(0.0, measured=False)
    blocked = ~free
    near_left = blocked & _dilate(left)
    near_right = blocked & _dilate(right)
    if focus is not None:
        near_left &= focus
        near_right &= focus
    if hits:
        spot = np.zeros(free.shape, dtype=bool)
        for r, c in hits:
            spot[r, c] = True
        away = _edt(~spot) * res > END_M
        near_left &= away
        near_right &= away
    run_on = np.zeros(free.shape, dtype=bool)
    run_on[cells[:, 0].astype(int), cells[:, 1].astype(int)] = True
    run_on &= ~walked
    if run_on.any():
        # a gap across the run-on beyond a path's end is no neck of the way: only the cells nearer the walked path count
        nearer = _edt(~walked) <= _edt(~run_on)
        near_left &= nearer
        near_right &= nearer
    if not near_left.any() or not near_right.any():
        return _Width(WIDE_M)
    dist, ir, ic = _edt_nearest(~near_left)
    rr, cc = np.nonzero(near_right)
    k = int(np.argmin(dist[rr, cc]))
    r, c = int(rr[k]), int(cc[k])
    lr, lc = int(ir[r, c]), int(ic[r, c])
    # cell centres: the gap between the cells' facing edges is a cell less
    width = max(0.0, float(dist[r, c]) * res - res)
    to_xy = lambda a, b: (round(grid.x0 + (b + 0.5) * res, 2), round(grid.y0 + (a + 0.5) * res, 2))  # noqa: E731
    return _Width(round(min(width, WIDE_M), 2), to_xy(lr, lc), to_xy(r, c))


def _across(free: np.ndarray, rows: np.ndarray, cols: np.ndarray, focus: np.ndarray) -> np.ndarray:
    """A cut across a path that has no neck near the element, to send the next search round another side of the
    obstacle it passes nearest there: from that obstacle's cell nearest the path (blocked, within `focus`), straight
    through the path to the next blocked cell or the window's edge, as a cell mask."""
    import numpy as np

    h, w = free.shape
    on_path = np.zeros(free.shape, dtype=bool)
    on_path[rows, cols] = True
    near = ~free & focus
    cut = np.zeros(free.shape, dtype=bool)
    if not near.any():
        cut[rows[len(rows) // 2], cols[len(cols) // 2]] = True
        return cut
    dist, pr, pc = _edt_nearest(~on_path)
    br, bc = np.nonzero(near)
    k = int(np.argmin(dist[br, bc]))
    r0, c0 = int(br[k]), int(bc[k])
    r1, c1 = int(pr[r0, c0]), int(pc[r0, c0])
    dr, dc = float(r1 - r0), float(c1 - c0)
    norm = math.hypot(dr, dc) or 1.0
    for k in range(1, 4 * (h + w)):
        rr, cc = int(round(r0 + dr / norm * k * 0.5)), int(round(c0 + dc / norm * k * 0.5))
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if not free[rr, cc]:
            if math.hypot(rr - r0, cc - c0) > 1.5:
                break
            continue  # still in the obstacle it starts from
        cut[rr, cc] = True
    return cut


@attrs.frozen(eq=False)
class _Way:
    """One way between two openings: its width, length [m] and cells (rows, cols)."""

    width: _Width
    length: float
    cells: np.ndarray


#: Elements this far apart leave no neck between them under any requirement (the two-way width; a lane beside a docked
#: robot needs more only for over 3900 walkers an hour through one neck) [m]
JOINT_M = TWO_WAY_M

#: The length of a graph edge a search has closed: no way within the slack takes it [m]
CLOSED_M = 1e9


@attrs.frozen(eq=False)
class _Routing:
    """A window floor's routing graph, built once: the graph, the node of each cell (-1 off it), each node's cell and
    each stored edge's tail node (so a search can close edges by their ends)."""

    graph: Any
    index: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    tails: np.ndarray

    @classmethod
    def of(cls, routable: np.ndarray, res: float) -> _Routing:
        import numpy as np

        graph, index = _graph(routable, res)
        graph = graph.tocsr()
        graph.sum_duplicates()
        tails = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
        rows, cols = np.nonzero(routable)
        return cls(graph, index, rows, cols, tails)


def _ways(routing: _Routing, grid: _Grid, free: np.ndarray, a_mask: np.ndarray, b_mask: np.ndarray, focus: np.ndarray) -> list[_Way]:
    """The walkers' ways between two openings (cell masks): the shortest path, and every other within the route slack of
    it that passes another side of an obstacle, at most `ALTERNATIVES` more. Each round closes the last way for the next
    search: across its neck, or where it has no neck near the element, from the obstacle it passes nearest there
    (`_across`); a closed cell's edges get `CLOSED_M`."""
    import numpy as np
    import scipy.ndimage
    import scipy.sparse

    out: list[_Way] = []
    shortest = None
    index = routing.index
    a_nodes, b_nodes = index[a_mask & (index >= 0)], index[b_mask & (index >= 0)]
    graph = routing.graph
    data = None
    for _ in range(1 + ALTERNATIVES):
        found = _path(graph, index, a_nodes, b_nodes)
        if found is None or found[1] >= CLOSED_M:
            break
        nodes, length = found
        if shortest is None:
            shortest = length
        elif length > shortest * (1.0 + ROUTE_SLACK) + ROUTE_SLACK_M:
            break
        rows, cols = routing.rows[nodes], routing.cols[nodes]
        width = _width(free, grid, rows, cols, focus)
        out.append(_Way(width, length, np.column_stack([rows, cols])))
        if not width.measured:
            break  # no neck to close for the next search: the stretch is unmeasured anyway
        gap = width.gap
        if gap is not None:
            (lr, lc), (r_, c_) = grid.cells(np.array([gap[0]])), grid.cells(np.array([gap[1]]))
            n = int(max(abs(r_[0] - lr[0]), abs(c_[0] - lc[0]))) * 2 + 3
            cut = np.zeros(free.shape, dtype=bool)
            cut[np.clip(np.round(np.linspace(lr[0], r_[0], n)).astype(int), 0, free.shape[0] - 1), np.clip(np.round(np.linspace(lc[0], c_[0], n)).astype(int), 0, free.shape[1] - 1)] = True
        else:
            cut = _across(free, rows, cols, focus)
        closed = index[_dilate(cut) & (index >= 0)]
        if not len(closed):
            break
        mark = np.zeros(len(routing.rows), dtype=bool)
        mark[closed] = True
        data = (routing.graph.data if data is None else data).copy()
        data[mark[routing.tails] | mark[routing.graph.indices]] = CLOSED_M
        graph = scipy.sparse.csr_matrix((data, routing.graph.indices, routing.graph.indptr), shape=routing.graph.shape)
    return out


def _crossing(way: _Way, gap: tuple[Point, Point], grid: _Grid) -> int:
    """Which way `way` crosses the neck `gap` (+1 left to right of the gap's direction, -1 back), 0 where it does not."""
    import numpy as np
    import shapely

    xy = grid.xy(way.cells[:, 0], way.cells[:, 1])
    if len(xy) < 2:
        return 0
    line = shapely.LineString(xy)
    seg = shapely.LineString(gap)
    hit = line.intersection(seg)
    if hit.is_empty:
        return 0
    p = hit.representative_point()
    i = int(np.argmin(np.hypot(xy[:, 0] - p.x, xy[:, 1] - p.y)))
    j0, j1 = max(0, i - 2), min(len(xy) - 1, i + 2)
    (gx, gy), (hx, hy) = gap
    dx, dy = xy[j1, 0] - xy[j0, 0], xy[j1, 1] - xy[j0, 1]
    return 1 if (hx - gx) * dy - (hy - gy) * dx >= 0 else -1


def window(occupied_floor: BaseGeometry) -> BaseGeometry:
    """The floor `narrowings` reads around `occupied_floor`, as a box."""
    import shapely

    return shapely.box(*occupied_floor.buffer(WINDOW_M).bounds)


def docked_width(flow_ab: float, flow_ba: float) -> float:
    """The width a stretch keeps while a robot stands in an intermittent dock [m]: `ONE_WAY_M`, one wheelchair (2010 ADA
    403.5.1; ADA 403.5.3 allows a route narrower than two-way where passing spaces lie within 61 m: here the walkway either
    side of a dock), and what its walkers of both directions (ped/h) need taking turns: their flow over
    `TURNS_SHARE` of `SPECIFIC_FLOW_PED_MS`."""
    return max(ONE_WAY_M, (flow_ab + flow_ba) / 3600.0 / (TURNS_SHARE * SPECIFIC_FLOW_PED_MS))


def dock_share(deliveries_per_h: float, hold_s: float) -> float:
    """The share of the time a robot stands in a dock: deliveries an hour times its hold there."""
    return deliveries_per_h * hold_s / 3600.0


def _requirement(forth: float, back: float, docked: bool) -> float | None:
    """The width a neck crossed by `forth` and `back` walkers an hour keeps; None where neither way reaches
    `FLOW_MIN_PED_H`."""
    if docked:
        return docked_width(forth, back) if max(forth, back) >= FLOW_MIN_PED_H else None
    if min(forth, back) >= FLOW_MIN_PED_H:
        return TWO_WAY_M
    return ONE_WAY_M if max(forth, back) >= FLOW_MIN_PED_H else None


def narrowings(floor: WalkerFloor, routes: Routes, mine: Sequence[Occupied], others: Sequence[Occupied] = (), *, docked: bool = False, stretches: list[Stretch] | None = None) -> list[Narrowing]:
    """The walkway stretches the floor `mine` (one element's) narrows on `floor`, with `others` (other elements' floor)
    occupied too, against the crowd's `routes` (see the module doc), narrowest first. Without `docked`, on the permanent
    floor (intermittent docks free) against the two-way or one-way width; with it, every intermittent dock occupied,
    against `docked_width`, and only where an intermittent dock makes the neck. The requirement at a neck goes by every
    walker crossing it, each way, whichever openings they walk between (`_crossing`). A walkway the floor closes within
    the window (no way left) counts as narrowed to nothing; two openings the routes join with no way between them in the
    window as `unmeasured`, a failure. `stretches`, when given, gets what was measured for every opening pair."""
    import numpy as np
    import shapely

    if docked:
        if not any(o.intermittent for o in (*mine, *others)):
            return []
    else:
        mine = [o for o in mine if not o.intermittent]
        others = [o for o in others if not o.intermittent]
    if not mine:
        return []
    mine_geom = shapely.union_all([o.geometry for o in mine])
    if others and mine_geom.distance(shapely.union_all([o.geometry for o in others])) >= JOINT_M:
        # a neck between this element and the others is at least as wide as the floor between them: none under any
        # requirement, and what this element narrows alone is its own verdict's
        return []
    box = window(mine_geom)
    res = WINDOW_RES_M
    grid = _Grid.over(box.bounds, res)
    near = routes.near(box)
    if not near:
        return []
    all_occ = [*mine, *others]
    # the level's free floor in the window only (a cell beyond it reads as free, as the window's edge is no wall)
    bx0, by0, bx1, by1 = box.bounds
    base = shapely.clip_by_rect(floor.free_docked if docked else floor.free, bx0 - res, by0 - res, bx1 + res, by1 + res)
    after_poly = base.difference(shapely.union_all([o.geometry for o in all_occ]))
    before_poly = base.difference(shapely.union_all([o.geometry for o in others])) if others else base
    free_b, free_a = grid.raster(before_poly), grid.raster(after_poly)
    route_b = _clearance(free_b, res) >= ROUTE_CLEARANCE_M - 1e-9
    route_a = _clearance(free_a, res) >= ROUTE_CLEARANCE_M - 1e-9
    ports, arc = _ports(grid, route_b)
    xx, yy = grid.centres()
    toward = grid.raster(mine_geom)
    focus = _edt(~toward) * res <= MEASURE_M
    # the openings the crowd's route pieces start and end at, and the walkers an hour from each to each
    flows: dict[tuple[str, str], float] = {}
    ends: dict[str, list[tuple[float, float]]] = {}
    for r in near:
        if len(r.xy) < 2:
            continue
        clipped = shapely.clip_by_rect(shapely.LineString(r.xy), *box.bounds)
        for piece in getattr(clipped, "geoms", [clipped]):
            if piece.is_empty or piece.length < res:
                continue
            a, b = _port_at(piece.coords[0], grid, ports, arc, box), _port_at(piece.coords[-1], grid, ports, arc, box)
            if a == b:
                continue  # a piece that leaves and comes back by one opening crosses no stretch between two
            for port, end in ((a, piece.coords[0]), (b, piece.coords[-1])):
                if port.startswith("end:"):
                    ends.setdefault(port, []).append((float(end[0]), float(end[1])))
            flows[(a, b)] = flows.get((a, b), 0.0) + r.rate * 3600.0
    masks: dict[str, np.ndarray] = {}

    def mask_of(port: str) -> np.ndarray:
        """An opening's cells: a window edge bin's routable cells, or the routable cells within a bin of where route
        pieces end inside the window (a source or sink, a stair's mouth), the nearest ones within two bins if none."""
        if port not in masks:
            if port.startswith("edge:"):
                _, run, at = port.split(":")
                masks[port] = (ports == int(run)) & (np.abs(arc - int(at) * PORT_BIN_M) <= PORT_BIN_M)
            else:
                cx, cy = np.mean(np.asarray(ends[port]), axis=0)
                d = np.hypot(xx - cx, yy - cy)
                cells = route_b & (d <= PORT_BIN_M)
                masks[port] = cells if cells.any() else route_b & (d <= 2.0 * PORT_BIN_M)
        return masks[port]

    # pairs whose walkers could not reach `FLOW_MIN_PED_H` at a neck even ten of them together are not measured
    pairs: list[tuple[str, str]] = sorted(k for k in {(min(a, b), max(a, b)) for a, b in flows} if flows.get(k, 0.0) + flows.get((k[1], k[0]), 0.0) >= FLOW_MIN_PED_H / 10.0)
    routing_b, routing_a = _Routing.of(route_b, res), _Routing.of(route_a, res)
    measured: dict[tuple[str, str], tuple[list[_Way], list[_Way]]] = {}
    for a, b in pairs:
        measured[(a, b)] = (_ways(routing_b, grid, free_b, mask_of(a), mask_of(b), focus), _ways(routing_a, grid, free_a, mask_of(a), mask_of(b), focus))

    def crossing_flows(gap: tuple[Point, Point], after: bool) -> tuple[float, float]:
        """The walkers an hour that cross the neck `gap` each way, over every opening pair's ways (after the layout, or
        before), a pair's whole flow on each way it may take."""
        forth = back = 0.0
        for (a, b), (ways_b, ways_a) in measured.items():
            signs = {_crossing(w, gap, grid) for w in (ways_a if after else ways_b)} - {0}
            if not signs:
                continue
            q_ab, q_ba = flows.get((a, b), 0.0), flows.get((b, a), 0.0)
            if len(signs) > 1:
                # its ways cross the neck both ways: each of its flows may come either way
                forth, back = forth + q_ab + q_ba, back + q_ab + q_ba
            else:
                forth, back = (forth + q_ab, back + q_ba) if signs.pop() > 0 else (forth + q_ba, back + q_ab)
        return forth, back

    mine_names = {o.name for o in mine}
    out: list[Narrowing] = []
    for (a, b), (ways_b, ways_a) in measured.items():
        q_ab, q_ba = flows.get((a, b), 0.0), flows.get((b, a), 0.0)
        if not ways_b or not all(w.width.measured for w in (*ways_b, *ways_a)):
            # the routes join the two openings, the window shows no way between them, or a way whose width cannot be
            # read: the check cannot say
            if _requirement(q_ab, q_ba, docked) is not None:
                c = mine_geom.centroid
                out.append(Narrowing((a, b), _requirement(q_ab, q_ba, docked) or ONE_WAY_M, 0.0, 0.0, (round(c.x, 2), round(c.y, 2)), round(q_ab, 1), round(q_ba, 1), tuple(sorted({o.name.split("#")[0] for o in mine})), docked, "unmeasured"))
            continue
        before_min = min(w.width.width for w in ways_b)
        if not ways_a:
            required = _requirement(q_ab, q_ba, docked)
            if stretches is not None and required is not None:
                stretches.append(Stretch((a, b), required, before_min, None, round(q_ab, 1), round(q_ba, 1), None, docked))
            closing = [o for o in mine if o.intermittent or not docked]
            if required is not None and closing:
                c = shapely.union_all([o.geometry for o in closing]).centroid
                out.append(Narrowing((a, b), required, round(before_min, 2), 0.0, (round(c.x, 2), round(c.y, 2)), round(q_ab, 1), round(q_ba, 1), tuple(sorted({o.name.split("#")[0] for o in closing})), docked, "closed"))
            continue
        # each way after the layout against the same way before, at the requirement of its neck's crossing flow
        checks = []
        for w in ways_a:
            if w.width.gap is None:
                continue
            forth, back = crossing_flows(w.width.gap, after=True)
            required = _requirement(forth, back, docked)
            same = _width(free_b, grid, w.cells[:, 0], w.cells[:, 1], focus).width
            checks.append((w, same, required, forth, back))
        narrowest = min(ways_a, key=lambda w: w.width.width)
        if stretches is not None:
            req = next((c[2] for c in checks if c[0] is narrowest), None)
            fb = next(((c[3], c[4]) for c in checks if c[0] is narrowest), (q_ab, q_ba))
            stretches.append(Stretch((a, b), req if req is not None else _requirement(q_ab, q_ba, docked) or ONE_WAY_M, before_min, narrowest.width.width, round(fb[0], 1), round(fb[1], 1), narrowest.width.middle, docked))
        found = None
        narrowed = [c for c in checks if c[2] is not None and c[0].width.width < c[2] - 1e-9 and c[0].width.width < c[1] - WIDTH_TOL_M]
        if narrowed:
            w, same, required, forth, back = min(narrowed, key=lambda c: c[0].width.width - c[2])
            found = (w, same, required, forth, back, "narrowed")
        else:
            moved = [c for c in checks if c[0] is narrowest and c[2] is not None and c[0].width.width < c[2] - 1e-9 and c[0].width.width < before_min - WIDTH_TOL_M]
            if moved:
                w, _, required, forth, back = moved[0]
                found = (w, before_min, required, forth, back, "moved")
        if found is None:
            continue
        w, before, required, forth, back, kind = found
        gap = shapely.LineString(w.width.gap).buffer(2.0 * res)
        by = [o for o in all_occ if o.geometry.intersects(gap)]
        if kind == "moved":
            on_ways = shapely.MultiPoint(grid.xy(*np.vstack([v.cells for v in ways_b]).T)).buffer(ROUTE_CLEARANCE_M)
            by = [o for o in all_occ if o.geometry.intersects(on_ways)] or by
        if not any(o.name in mine_names for o in by):
            continue  # another element's narrowing (its own check) or the site's
        if docked and not any(o.intermittent for o in by):
            continue  # the permanent floor's (the check without `docked`)
        names_by = tuple(sorted({o.name.split("#")[0] for o in by}))
        out.append(Narrowing((a, b), required, round(before, 2), w.width.width, w.width.middle, round(forth, 1), round(back, 1), names_by, docked, kind, w.width.gap))
    return sorted(out, key=lambda n: (n.after - n.required, n.ports))


def _port_at(end: Sequence[float], grid: _Grid, ports: np.ndarray, arc: np.ndarray, box: BaseGeometry) -> str:
    """The opening a route piece's end lies at: where it crosses a window boundary run (`PORT_BIN_M` steps along it),
    else where it ends inside the window (`PORT_BIN_M` bins of the world frame: a source or sink, a stair's mouth)."""
    import numpy as np

    x0, y0, x1, y1 = box.bounds
    x, y = float(end[0]), float(end[1])
    if min(x - x0, x1 - x, y - y0, y1 - y) <= 2.0 * grid.res:
        r, c = grid.cells(np.array([[x, y]]))
        r0, c0 = int(np.clip(r[0], 0, grid.shape[0] - 1)), int(np.clip(c[0], 0, grid.shape[1] - 1))
        rows, cols = np.nonzero(ports[max(0, r0 - 4) : r0 + 5, max(0, c0 - 4) : c0 + 5] > 0)
        if not len(rows):
            # no routable cell of the edge here (a passage narrower than the walkers' clearance): an end like any other
            return f"end:{round(x / PORT_BIN_M)}:{round(y / PORT_BIN_M)}"
        rows, cols = rows + max(0, r0 - 4), cols + max(0, c0 - 4)
        k = int(np.argmin((rows - r0) ** 2 + (cols - c0) ** 2))
        return f"edge:{int(ports[rows[k], cols[k]])}:{round(float(arc[rows[k], cols[k]]) / PORT_BIN_M)}"
    return f"end:{round(x / PORT_BIN_M)}:{round(y / PORT_BIN_M)}"


__all__ = [
    "FLOW_MIN_PED_H",
    "ONE_WAY_M",
    "ROUTE_CLEARANCE_M",
    "TWO_WAY_M",
    "WINDOW_M",
    "Flow",
    "Link",
    "Narrowing",
    "Occupied",
    "Route",
    "Routes",
    "Stretch",
    "JOINT_M",
    "WIDE_M",
    "WalkerDemand",
    "WalkerFloor",
    "dock_share",
    "links_of",
    "docked_width",
    "narrowings",
    "occupied",
    "walker_floor",
    "window",
]
