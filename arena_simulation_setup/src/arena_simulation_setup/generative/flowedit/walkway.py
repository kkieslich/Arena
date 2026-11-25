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
(`keep_crowd_off`), most of a run: those are permanent. A locker's dock holds a robot only while it delivers and a
visitor while it fetches a parcel, the loading point's slot a robot while it loads: those are intermittent. Both count:
HumanSim's counterflow jams at a narrow passage do not clear once formed (PILEUP_FINDING), so a jam that starts while a
robot delivers outlasts the delivery. `Narrowing.intermittent` says which narrowings come from intermittent floor alone.

**Routes** (`Routes`) are the crowd's shortest paths on the walker floor at HumanSim's planning clearance
(`ROUTE_CLEARANCE_M`), between the sources and sinks of the site's crowd scenario (`WalkerDemand`: each source's rate
split over its sinks by their `sink_affinity` weights), wheelchair users without stairs and escalators
(`NO_CLIMB_TYPES`), the other walkers also through the scenario's teleport zones. Each carries its flow [ped/s].

**Narrowings** (`narrowings`). Within `WINDOW_M` of an element, the routes that cross the window give the directional
flow between the window's openings (its boundary where free floor crosses it, and the sources and sinks inside it). An
opening pair whose lesser direction carries `FLOW_MIN_PED_H` or more is a two-way stretch and needs `TWO_WAY_M`; one
that carries it one way only needs `ONE_WAY_M`. Between the two openings the walkers take the shortest way and every
other way within `ROUTE_SLACK` of it that passes another side of an obstacle (HumanSim weighs its routes by congestion,
so a way a few per cent longer carries walkers too), on the floor before the layout and after it. A way's width is the
shortest distance between the obstacles on its left and those on its right: every line from one side to the other
crosses the way, so this is the narrowest cross-section it passes, a neck on a diagonal (between a dock's corner and a
plant's) as narrow as it is, and a corner the way turns round is none. Only gaps within `MEASURE_M` of the element
count: the others it cannot change. A layout narrows the stretch when, after it,

- a way is narrower than the requirement and than the same way before (`WIDTH_TOL_M`), or
- the narrowest way is narrower than the requirement and than the narrowest way before: the layout moved the walkers
  into a narrow passage of the site.

A stretch that was narrower than its requirement before and that the layout leaves as it was is the site's, not the
layout's. A way the layout cuts within the window leaves the window: its detour is the study's walking delay, and NAV@r
checks that the floor stays connected.

**Why these widths.** Two-way: 1.8 m, "a clear width of 1800mm will allow two wheelchair users to pass each other"
(Singapore BCA Code on Accessibility in the Built Environment 2019, 4.2.1.2 and Fig. 15; the twin is in Singapore). It
is also HumanSim's own two-way width for a wheelchair (`wheelchair_manual` passage block: body plus the widest body plus
0.1 m, 1.7 m, rounded to the code): below it walkers meeting in the stretch take turns, and under sustained counterflow
they jam for good (1.0-1.2 m doors in PILEUP_FINDING). One-way: 0.9 m, the clear width of an accessible route (2010 ADA
403.5.1, 36 in = 0.915 m), one wheelchair. HumanSim routes through gaps of twice its planning clearance (0.8 m) and wider,
so a gap between 0.8 m and 1.8 m on a two-way route is the jam case.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    import numpy as np
    import shapely
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
#: A path end in the window runs on to a wall; the wall this near that point lies on both of its sides [m]
END_M = 1.0
#: Walls are this thick on the walker floor (`generate.WALL_GAP_M` either side of the line) [m]
WALL_M = 0.1
#: Walker types that take no stairs or escalators (HumanSim's `climbs: []`): wheelchair users take the ramps
NO_CLIMB_TYPES = ("wheelchair_manual", "wheelchair_electric")
#: Walking speed a teleport zone's delay is counted at, and the step on and off it [m/s], [m]
LINK_SPEED_MPS, LINK_STEP_M = 1.1, 2.0
#: The elements whose docks the fleet walls off for the crowd while a robot parks there: its homes and chargers
PARKED = ("charger_pair", "bay_pair")


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
        zones = {z["name"]: z for z in doc.get("teleport_zones") or [] if "entry" in z}
        links = []
        for name, z in sorted(zones.items()):
            exits = tuple((float(zones[d]["entry"]["x"]), float(zones[d]["entry"]["y"])) for d in z.get("destinations") or [] if d in zones)
            if exits:
                links.append(Link(name, (float(z["entry"]["x"]), float(z["entry"]["y"])), exits, float(z.get("delay", 0.0)) * LINK_SPEED_MPS + LINK_STEP_M))
        return cls(tuple(regions.items()), tuple(Flow(s, t, round(q, 9), c) for (s, t, c), q in sorted(flows.items())), tuple(links))

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
    _free: Any = attrs.field(default=None, init=False)

    @property
    def walls(self) -> BaseGeometry:
        import shapely

        lines = [shapely.LineString(s) for s in self.segments if s[0] != s[1]]
        return shapely.union_all(lines).buffer(WALL_M / 2.0) if lines else shapely.Polygon()

    @property
    def free(self) -> BaseGeometry:
        """The free floor: the zones less walls and obstacles."""
        if self._free is None:
            import shapely

            blocked = shapely.union_all([self.walls, *(o.geometry for o in self.obstacles)])
            object.__setattr__(self, "_free", self.floor.difference(blocked))
        return self._free

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
    gen.resolve_footprints(site, sorted({model for _, model in here.values()}))
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
        return np.meshgrid(xs, ys)

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


def _clearance(free: np.ndarray, res: float) -> np.ndarray:
    """Each cell's distance to the nearest blocked cell [m] (0 on blocked cells)."""
    import scipy.ndimage

    return scipy.ndimage.distance_transform_edt(free) * res


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
    regions: tuple[tuple[str, tuple[Point, ...]], ...] = ()
    _tree: Any = attrs.field(default=None, init=False)

    @property
    def lines(self) -> list[Any]:
        import shapely

        return [shapely.LineString(r.xy) if len(r.xy) > 1 else shapely.Point(r.xy[0]).buffer(1e-3) for r in self.routes]

    def near(self, area: BaseGeometry) -> list[Route]:
        """The routes that cross `area`."""
        import shapely

        if self._tree is None:
            object.__setattr__(self, "_tree", shapely.STRtree(self.lines))
        return [self.routes[int(i)] for i in self._tree.query(area, predicate="intersects")]

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
        _, index = _graph(routable, res)
        rr, cc = np.nonzero(routable)
        coords = grid.xy(rr, cc)
        import scipy.spatial

        tree = scipy.spatial.cKDTree(coords) if len(coords) else None

        def snap(p: Point, reach: float = 1.5) -> int | None:
            if tree is None:
                return None
            d, i = tree.query(p)
            return int(i) if d <= reach else None

        extra = []
        for link in demand.links:
            a = snap(link.entry)
            for exit_ in link.exits:
                b = snap(exit_)
                if a is not None and b is not None:
                    extra.append((a, b, link.cost_m))
        graphs = {False: _graph(routable, res)[0], True: _graph(routable, res, extra)[0]}
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
        return cls(tuple(_merge(routes)), tuple(missing), demand.regions)


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
    """A stretch a layout narrows: between openings `ports` (window boundary runs or regions), `required` and the
    widths `before` and `after` [m], `at` (where it is narrowest after), the flows both ways [ped/h], whether it is the
    way's narrowest cell moved into a narrow passage (`moved`), what makes it (`by`: occupied floor names) and whether
    that floor is intermittent alone; `gap` the narrowest cross-section's two ends."""

    ports: tuple[str, str]
    required: float
    before: float
    after: float
    at: Point
    flow_ab: float
    flow_ba: float
    by: tuple[str, ...]
    intermittent: bool
    moved: bool = False
    gap: tuple[Point, Point] | None = None

    @property
    def two_way(self) -> bool:
        return self.required >= TWO_WAY_M

    def describe(self) -> str:
        what = "two-way" if self.two_way else "one-way"
        how = "moves the walkers into" if self.moved else "narrows"
        return f"{how} a {what} walkway ({self.flow_ab:.0f}/{self.flow_ba:.0f} ped/h) to {self.after:.2f} m (before {self.before:.2f} m, needs {self.required:.1f} m) at ({self.at[0]:.1f}, {self.at[1]:.1f}) by {', '.join(self.by) or 'the site'}" + (" (intermittent floor only)" if self.intermittent else "")


@attrs.frozen
class Stretch:
    """What `narrowings` measured for one opening pair: its requirement, flows and the narrowest width before and after."""

    ports: tuple[str, str]
    required: float
    before: float
    after: float | None
    flow_ab: float
    flow_ba: float
    at: Point | None


def _ports(grid: _Grid, routable: np.ndarray) -> np.ndarray:
    """Labels of the window boundary's openings: runs of routable boundary cells along the ring (0 elsewhere)."""
    import numpy as np

    h, w = routable.shape
    ring = [(0, c) for c in range(w)] + [(r, w - 1) for r in range(1, h)] + [(h - 1, c) for c in range(w - 2, -1, -1)] + [(r, 0) for r in range(h - 2, 0, -1)]
    labels = np.zeros(routable.shape, dtype=np.int32)
    on = [bool(routable[r, c]) for r, c in ring]
    if all(on):
        for r, c in ring:
            labels[r, c] = 1
        return labels
    start = on.index(False)
    n = 0
    prev = False
    for k in range(len(ring)):
        r, c = ring[(start + k) % len(ring)]
        cur = on[(start + k) % len(ring)]
        if cur and not prev:
            n += 1
        if cur:
            labels[r, c] = n
        prev = cur
    return labels


def _path(graph: Any, index: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float] | None:
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
    """The narrowest gap between the obstacles left and right of a path in the window [m], and its two ends."""

    width: float
    left: Point
    right: Point

    @property
    def middle(self) -> Point:
        return ((self.left[0] + self.right[0]) / 2.0, (self.left[1] + self.right[1]) / 2.0)


def _width(free: np.ndarray, grid: _Grid, rows: np.ndarray, cols: np.ndarray, focus: np.ndarray | None = None) -> _Width | None:
    """The width of the walkway a path (cells `rows`, `cols`) takes through the window: the shortest distance between the
    blocked cells on its left and those on its right. Every line from one side to the other crosses the path, so this is
    the narrowest cross-section the path passes, on a diagonal too, and a corner it turns round is no cross-section. A
    path end inside the window (a source or sink there) runs on straight to the first blocked cell, which parts the
    sides; the blocked cells within `END_M` of where it meets them are on both sides and left out. With `focus` (a cell
    mask), only blocked cells there count: the gaps an element can narrow have it at one end."""
    import numpy as np
    import scipy.ndimage

    h, w = free.shape
    res = grid.res
    pts = list(zip(rows.tolist(), cols.tolist(), strict=True))
    if len(pts) < 2:
        return None
    hits = []
    for end, prev in ((0, min(len(pts) - 1, 5)), (len(pts) - 1, max(0, len(pts) - 6))):
        r, c = pts[end]
        if r in (0, h - 1) or c in (0, w - 1):
            continue
        dr, dc = r - pts[prev][0], c - pts[prev][1]
        norm = math.hypot(dr, dc) or 1.0
        ext = []
        for k in range(1, 4 * (h + w)):
            rr, cc = int(round(r + dr / norm * k * 0.5)), int(round(c + dc / norm * k * 0.5))
            if not (0 <= rr < h and 0 <= cc < w):
                break
            if not free[rr, cc]:
                hits.append((rr, cc))
                break
            ext.append((rr, cc))
        pts = ext[::-1] + pts if end == 0 else pts + ext
    barrier = np.zeros(free.shape, dtype=bool)
    for (r0, c0), (r1, c1) in itertools.pairwise(pts):
        n = max(abs(r1 - r0), abs(c1 - c0)) * 2 + 1
        barrier[np.round(np.linspace(r0, r1, n)).astype(int), np.round(np.linspace(c0, c1, n)).astype(int)] = True
    barrier = scipy.ndimage.binary_dilation(barrier, structure=np.ones((3, 3), dtype=bool)) & free
    sides, count = scipy.ndimage.label(free & ~barrier, structure=np.ones((3, 3), dtype=bool))
    if count < 2:
        return None
    # each side region by which side of the path it lies on: the cross product at the path point nearest to it
    path = np.asarray(pts, dtype=float)
    tangent = np.gradient(path, axis=0)
    import scipy.spatial

    tree = scipy.spatial.cKDTree(path)
    touching = scipy.ndimage.binary_dilation(barrier, structure=np.ones((3, 3), dtype=bool))
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
    blocked = ~free
    grow = np.ones((3, 3), dtype=bool)
    near_left = blocked & scipy.ndimage.binary_dilation(left, structure=grow)
    near_right = blocked & scipy.ndimage.binary_dilation(right, structure=grow)
    if focus is not None:
        near_left &= focus
        near_right &= focus
    if hits:
        spot = np.zeros(free.shape, dtype=bool)
        for r, c in hits:
            spot[r, c] = True
        away = scipy.ndimage.distance_transform_edt(~spot) * res > END_M
        near_left &= away
        near_right &= away
    if not near_left.any() or not near_right.any():
        return None
    dist, (ir, ic) = scipy.ndimage.distance_transform_edt(~near_left, return_indices=True)
    rr, cc = np.nonzero(near_right)
    k = int(np.argmin(dist[rr, cc]))
    r, c = int(rr[k]), int(cc[k])
    lr, lc = int(ir[r, c]), int(ic[r, c])
    # cell centres: the gap between the cells' facing edges is a cell less
    width = max(0.0, float(dist[r, c]) * res - res)
    to_xy = lambda a, b: (round(grid.x0 + (b + 0.5) * res, 2), round(grid.y0 + (a + 0.5) * res, 2))  # noqa: E731
    return _Width(round(width, 2), to_xy(lr, lc), to_xy(r, c))


def _ways(routable: np.ndarray, grid: _Grid, free: np.ndarray, a_mask: np.ndarray, b_mask: np.ndarray, focus: np.ndarray) -> list[tuple[_Width, float, np.ndarray]]:
    """The walkers' ways between two openings (cell masks): the shortest path, and those within the route slack of it
    that pass another side of an obstacle (found by closing each way's narrowest gap in turn), at most
    `ALTERNATIVES` more; each with its width, length and cells."""
    import numpy as np

    out: list[tuple[_Width, float, np.ndarray]] = []
    shortest = None
    mask = routable.copy()
    for _ in range(1 + ALTERNATIVES):
        graph, index = _graph(mask, grid.res)
        found = _path(graph, index, index[a_mask & (index >= 0)], index[b_mask & (index >= 0)])
        if found is None:
            break
        nodes, length = found
        if shortest is None:
            shortest = length
        elif length > shortest * (1.0 + ROUTE_SLACK) + ROUTE_SLACK_M:
            break
        rr, cc = np.nonzero(mask)
        rows, cols = rr[nodes], cc[nodes]
        width = _width(free, grid, rows, cols, focus)
        if width is None:
            # no gap near the element on this way: it keeps the way as wide as the window shows
            width = _Width(2.0 * MEASURE_M, (0.0, 0.0), (0.0, 0.0))
            out.append((width, length, np.column_stack([rows, cols])))
            break
        out.append((width, length, np.column_stack([rows, cols])))
        # close the gap for the next way: the segment between its two ends, a cell wide
        (lr, lc), (r_, c_) = grid.cells(np.array([width.left])), grid.cells(np.array([width.right]))
        n = int(max(abs(r_[0] - lr[0]), abs(c_[0] - lc[0]))) * 2 + 3
        sr = np.clip(np.round(np.linspace(lr[0], r_[0], n)).astype(int), 0, mask.shape[0] - 1)
        sc = np.clip(np.round(np.linspace(lc[0], c_[0], n)).astype(int), 0, mask.shape[1] - 1)
        cut = np.zeros(mask.shape, dtype=bool)
        cut[sr, sc] = True
        import scipy.ndimage

        mask &= ~scipy.ndimage.binary_dilation(cut, structure=np.ones((3, 3), dtype=bool))
    return out


def window(occupied_floor: BaseGeometry) -> BaseGeometry:
    """The floor `narrowings` reads around `occupied_floor`, as a box."""
    import shapely

    return shapely.box(*occupied_floor.buffer(WINDOW_M).bounds)


def narrowings(floor: WalkerFloor, routes: Routes, mine: Sequence[Occupied], others: Sequence[Occupied] = (), *, stretches: list[Stretch] | None = None) -> list[Narrowing]:
    """The walkway stretches the floor `mine` (one element's) narrows on `floor`, with `others` (other elements' floor)
    occupied too, against the crowd's `routes` (see the module doc), narrowest first. `stretches`, when given, gets what
    was measured for every opening pair that carries walkers."""
    import numpy as np
    import shapely

    if not mine:
        return []
    mine_geom = shapely.union_all([o.geometry for o in mine])
    box = window(mine_geom)
    res = WINDOW_RES_M
    grid = _Grid.over(box.bounds, res)
    near = routes.near(box)
    if not near:
        return []
    all_occ = [*mine, *others]
    after_poly = floor.free.difference(shapely.union_all([o.geometry for o in all_occ]))
    before_poly = floor.free.difference(shapely.union_all([o.geometry for o in others])) if others else floor.free
    free_b, free_a = grid.raster(before_poly), grid.raster(after_poly)
    route_b = _clearance(free_b, res) >= ROUTE_CLEARANCE_M - 1e-9
    route_a = _clearance(free_a, res) >= ROUTE_CLEARANCE_M - 1e-9
    ports = _ports(grid, route_b)
    xx, yy = grid.centres()
    import scipy.ndimage

    focus = scipy.ndimage.distance_transform_edt(~grid.raster(mine_geom)) * res <= MEASURE_M
    canonical = _canonical(routes)
    # flows between openings, from the routes' pieces in the window
    flows: dict[tuple[str, str], float] = {}
    for r in near:
        if len(r.xy) < 2:
            continue
        clipped = shapely.clip_by_rect(shapely.LineString(r.xy), *box.bounds)
        for piece in getattr(clipped, "geoms", [clipped]):
            if piece.is_empty or piece.length < res:
                continue
            a = _port_at(piece.coords[0], grid, ports, box, canonical.get(r.source, r.source))
            b = _port_at(piece.coords[-1], grid, ports, box, canonical.get(r.sink, r.sink))
            if a is None or b is None or a == b:
                continue
            flows[(a, b)] = flows.get((a, b), 0.0) + r.rate * 3600.0
    masks: dict[str, np.ndarray] = {}

    def mask_of(port: str) -> np.ndarray:
        if port not in masks:
            if port.startswith("region:"):
                region = _region_polygon(routes, port.removeprefix("region:"))
                masks[port] = shapely.contains_xy(region.buffer(res), xx, yy) if region is not None else np.zeros(grid.shape, dtype=bool)
            else:
                masks[port] = ports == int(port.removeprefix("edge:"))
        return masks[port]

    mine_names = {o.name for o in mine}
    out: list[Narrowing] = []
    for a, b in sorted({tuple(sorted(k)) for k in flows}):
        q_ab, q_ba = flows.get((a, b), 0.0), flows.get((b, a), 0.0)
        if max(q_ab, q_ba) < FLOW_MIN_PED_H:
            continue
        required = TWO_WAY_M if min(q_ab, q_ba) >= FLOW_MIN_PED_H else ONE_WAY_M
        ways_b = _ways(route_b, grid, free_b, mask_of(a), mask_of(b), focus)
        if not ways_b:
            continue
        before_min = min(w.width for w, _, _ in ways_b)
        # a way the elements do not touch keeps its width: only the floor's change narrows (and closes) ways
        ways_a = _ways(route_a, grid, free_a, mask_of(a), mask_of(b), focus)
        if not ways_a:
            if stretches is not None:
                stretches.append(Stretch((a, b), required, before_min, None, round(q_ab, 1), round(q_ba, 1), None))
            continue
        widths = []
        for w, _, cells in ways_a:
            same = _width(free_b, grid, cells[:, 0], cells[:, 1], focus)
            widths.append((w, same.width if same is not None else w.width))
        after = min(widths, key=lambda t: t[0].width)
        if stretches is not None:
            stretches.append(Stretch((a, b), required, before_min, after[0].width, round(q_ab, 1), round(q_ba, 1), after[0].middle))
        found = None
        narrowed = [(w, wb) for w, wb in widths if w.width < required - 1e-9 and w.width < wb - WIDTH_TOL_M]
        if narrowed:
            w, wb = min(narrowed, key=lambda t: t[0].width)
            found = (w, wb, False)
        elif after[0].width < required - 1e-9 and after[0].width < before_min - WIDTH_TOL_M:
            found = (after[0], before_min, True)
        if found is None:
            continue
        w, before, moved = found
        gap = shapely.LineString([w.left, w.right]).buffer(2.0 * res)
        by = [o for o in all_occ if o.geometry.intersects(gap)]
        if moved:
            ways = shapely.MultiPoint(grid.xy(*np.vstack([c for _, _, c in ways_b]).T)).buffer(ROUTE_CLEARANCE_M)
            by = [o for o in all_occ if o.geometry.intersects(ways)] or by
        if not any(o.name in mine_names for o in by):
            continue  # another element's narrowing (its own check) or the site's
        names_by = tuple(sorted({o.name.split("#")[0] for o in by}))
        out.append(Narrowing((a, b), required, round(before, 2), w.width, w.middle, round(q_ab, 1), round(q_ba, 1), names_by, all(o.intermittent for o in by), moved, (w.left, w.right)))
    return sorted(out, key=lambda n: (n.after - n.required, n.ports))



def _canonical(routes: Routes) -> dict[str, str]:
    """Each region's opening: regions on the same polygon (a source per walker group and the sink of an entrance) are one."""
    first: dict[tuple[Point, ...], str] = {}
    out = {}
    for name, points in routes.regions:
        key = tuple((round(x, 2), round(y, 2)) for x, y in points)
        out[name] = first.setdefault(key, name)
    return out


def _region_polygon(routes: Routes, name: str) -> BaseGeometry | None:
    import shapely

    points = dict(routes.regions).get(name)
    return shapely.Polygon(points).buffer(0) if points else None


def _port_at(end: Sequence[float], grid: _Grid, ports: np.ndarray, box: BaseGeometry, region: str) -> str | None:
    """The opening a route piece's end lies at: a window boundary run within two cells, else its route's region."""
    import numpy as np

    x0, y0, x1, y1 = box.bounds
    x, y = float(end[0]), float(end[1])
    if min(x - x0, x1 - x, y - y0, y1 - y) <= 2.0 * grid.res:
        r, c = grid.cells(np.array([[x, y]]))
        r0, c0 = int(np.clip(r[0], 0, grid.shape[0] - 1)), int(np.clip(c[0], 0, grid.shape[1] - 1))
        patch = ports[max(0, r0 - 4) : r0 + 5, max(0, c0 - 4) : c0 + 5]
        labels = patch[patch > 0]
        if len(labels):
            return f"edge:{int(np.bincount(labels).argmax())}"
        return None
    return f"region:{region}"


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
    "WalkerDemand",
    "WalkerFloor",
    "narrowings",
    "occupied",
    "walker_floor",
    "window",
]
