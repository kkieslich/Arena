"""A site as the learned forecaster reads it: the FLOWEDIT graph and the map geometry on the flow grid, plus the graph itself.

Nothing in it comes from a recorded run. Per world (`raster`):
- **channels** (`CHANNELS`, one (ny, nx) layer each, on the base world's 0.5 m flow grid): the rooms, the free space people
  walk (HumanSim's clearance, `scene._free`), its clearance, the open and the locked doors (`gated`), the entrances (doors
  in one room only, weighted by the settings' entrance rate), one layer per room purpose, and the people's start points
  and idle stops as the reader draws them at the expected crowd's seeds (`scene.CROWD_SEEDS`, none of them a run's).
- **need channels** (`NEED_CHANNELS`, one stack per need the graph's objects satisfy): the fronts of its objects weighted
  by capacity and by dwell, their queue spots, the walking distance to the nearest front (as exp(-d / NEAR_M)), and the
  need's expected trips per person, spread over the rooms.
- **graph**: nodes for rooms, doors, demand objects and needs with their features, typed edges with the walking distance
  between their anchors, and each node's cells (a room's cells, a door's span, an object's front and queue), where the
  network writes its embedding.

`demand=False` leaves out the need channels and the demand and need nodes: the geometry-only prior of the demand ablation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import attrs

from .maps import FlowGrid

if TYPE_CHECKING:
    import numpy as np

    from ..edit import Site
    from ..read import Floor, Graph
    from .model import Scene

PURPOSES = ("work", "break", "transit", "private", "unknown")
CHANNELS = ("rooms", "free", "clearance", "door", "locked", "entrance", *(f"purpose_{p}" for p in PURPOSES), "starts", "stops")
NEED_CHANNELS = ("front_capacity", "front_dwell", "queue", "near", "trips")
NODE_KINDS = ("room", "door", "demand", "need")
#: edge kinds: (name, source kind, target kind)
EDGE_KINDS = (("room_door", "room", "door"), ("demand_room", "demand", "room"), ("demand_need", "demand", "need"), ("room_need", "room", "need"), ("demand_demand", "demand", "demand"))
#: node features per kind
NODE_FEATURES = {
    "room": (*(f"purpose_{p}" for p in PURPOSES), "area", "starts", "stops", "doors"),
    "door": ("open", "locked", "exterior", "width"),
    "demand": ("capacity", "dwell", "front", "step"),
    "need": ("trips", "amount", "objects"),
}
#: walking distance scale of the `near` channel and the edges (m)
NEAR_M = 10.0
#: clearance (m) the clearance channel saturates at
CLEARANCE_M = 2.0
#: queue spots behind the user's spot marked in the `queue` channel
QUEUE_SPOTS = 3
#: a trip's walking and waiting time the trips estimate assumes (s)
TRIP_S = 60.0


@attrs.frozen
class Nodes:
    """Nodes of one kind: names, features (n, len(NODE_FEATURES[kind])) and the flat grid cells each one covers."""

    names: tuple[str, ...]
    x: Any
    cells: tuple[Any, ...]


@attrs.frozen
class Edges:
    """Typed edges: indices into the source and target kinds' nodes, and the walking distance between them (/ NEAR_M)."""

    src: Any
    dst: Any
    dist: Any


@attrs.frozen
class Raster:
    """What the learned forecaster reads of one world."""

    world: str
    grid: FlowGrid
    channels: Any
    needs: tuple[str, ...]
    need_channels: Any
    nodes: Mapping[str, Nodes]
    edges: Mapping[str, Edges]
    #: (ny, nx) cells a forecast may hold time in
    mask: Any
    #: the crowd's expected pedestrian-seconds per room cell: people x duration / room cells
    scale: float
    people: float
    demand: bool = True


def _pool(a: np.ndarray, sub: int) -> np.ndarray:
    """Mean of each sub x sub block: nav cells to flow cells."""
    ny, nx = a.shape[0] // sub, a.shape[1] // sub
    return a[: ny * sub, : nx * sub].reshape(ny, sub, nx, sub).mean(axis=(1, 3))


def _cell(grid: FlowGrid, p: Sequence[float]) -> int | None:
    i, j, ok = grid.index(p[0], p[1])
    return int(i) * grid.nx + int(j) if bool(ok) else None


def _splat(grid: FlowGrid, points: Sequence[Sequence[float]], weights: Sequence[float]) -> np.ndarray:
    import numpy as np

    out = np.zeros(grid.ny * grid.nx)
    for p, w in zip(points, weights, strict=True):
        c = _cell(grid, p)
        if c is not None:
            out[c] += w
    return out.reshape(grid.ny, grid.nx)


class _Walk:
    """Walking distances on a scene's free nav cells (HumanSim's comfort cost, as the demand model walks)."""

    def __init__(self, scene: Scene) -> None:
        import numpy as np

        from .model import _nav_graph, _nearest_free

        free = np.asarray(scene.free, bool)
        self.shape = free.shape
        self.sub = scene.sub
        self.grid = scene.grid
        self.res = scene.nav_res
        self.graph, _ = _nav_graph(free, self.res)
        self.nearest = _nearest_free(free)

    def node(self, p: Sequence[float]) -> int:
        g = self.grid
        i = min(max(int((p[1] - g.y0) / self.res), 0), self.shape[0] - 1)
        j = min(max(int((p[0] - g.x0) / self.res), 0), self.shape[1] - 1)
        return int(self.nearest[i * self.shape[1] + j])

    def field(self, sources: Sequence[Sequence[float]]) -> np.ndarray:
        """(ny, nx) flow cells: walking distance (m) from the nearest of `sources`, the cell's nearest nav cell; inf unreachable."""
        import numpy as np
        import scipy.sparse.csgraph

        if not sources:
            return np.full((self.grid.ny, self.grid.nx), np.inf)
        d = scipy.sparse.csgraph.dijkstra(self.graph, directed=False, indices=sorted({self.node(p) for p in sources}), min_only=True)
        d = d[self.nearest].reshape(self.shape)
        ny, nx = self.grid.ny, self.grid.nx
        return d[: ny * self.sub, : nx * self.sub].reshape(ny, self.sub, nx, self.sub).min(axis=(1, 3))

    def between(self, a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> np.ndarray:
        """(len(a), len(b)) walking distances (m), inf where unreachable."""
        import numpy as np
        import scipy.sparse.csgraph

        if not a or not b:
            return np.zeros((len(a), len(b)))
        d = scipy.sparse.csgraph.dijkstra(self.graph, directed=False, indices=[self.node(p) for p in a])
        return d[:, [self.node(p) for p in b]]


def expected_trips(need: Any, duration: float) -> float:  # noqa: ANN401 - model.Need
    """A rough count of a person's trips for `need` over `duration` from the means of its settings: the first once it decays
    below its threshold, then one per refill's decay plus TRIP_S."""
    initial, decay = float(need.initial["mean"]), max(float(need.decay["mean"]), 1e-6)
    first = max(0.0, (initial - need.below) / decay)
    if first >= duration:
        return 0.0
    refill = max(1e-6, min(100.0, need.below + need.amount) - need.below)
    return 1.0 + (duration - first) / (refill / decay + TRIP_S)


def _doors(site: Site, floor: Floor, graph: Graph) -> list[tuple[str, tuple[float, float], tuple[float, float], bool, bool]]:
    """(name, start, end, locked, exterior) of each door of the people's level, map frame."""
    locked = {str(f.subject).partition(":")[2] for f in site.graph.facts if f.verb == "gated" and f.object is True}
    exterior = set(graph.exterior_doors())
    level = site.world.levels.get(floor.level_id)
    ox, oy = floor.origin
    return [(d.name, (d.start.x + ox, d.start.y + oy), (d.end.x + ox, d.end.y + oy), d.name in locked, d.name in exterior) for d in (level.doors if level is not None else [])]


def _band(grid: FlowGrid, a: Sequence[float], b: Sequence[float], width: float) -> np.ndarray:
    """(ny, nx) cells whose centre lies within `width` of segment a-b."""
    import shapely

    xs, ys = grid.centres()
    band = shapely.LineString([a, b]).buffer(width)
    return shapely.contains_xy(band, xs, ys)


def raster(site: Site, scene: Scene, settings: Mapping[str, Any], *, demand: bool = True) -> Raster:
    """The channels and graph of `site` on its scene's grid (the base world's), from its graph and geometry only."""
    import numpy as np
    import scipy.ndimage
    import shapely
    import shapely.affinity

    from .. import read as R

    grid = scene.grid
    graph = R.read_graph(site.graph)
    floor = R._floor(site.root, site.world, graph)
    walk = _Walk(scene)
    rooms_mask = grid.room_cells()
    free = np.asarray(scene.free, bool)
    xs, ys = grid.centres()

    layers: dict[str, np.ndarray] = {"rooms": rooms_mask.astype(float), "free": _pool(free.astype(float), scene.sub)}
    clearance = scipy.ndimage.distance_transform_edt(free) * scene.nav_res
    layers["clearance"] = np.minimum(_pool(clearance, scene.sub), CLEARANCE_M) / CLEARANCE_M

    doors = _doors(site, floor, graph)
    rate = float((settings.get("entrances") or {}).get("rate") or 0.0)
    layers["door"] = np.zeros((grid.ny, grid.nx))
    layers["locked"] = np.zeros((grid.ny, grid.nx))
    layers["entrance"] = np.zeros((grid.ny, grid.nx))
    door_cells: list[np.ndarray] = []
    for _, a, b, locked, exterior in doors:
        band = _band(grid, a, b, grid.cell)
        door_cells.append(np.flatnonzero(band))
        layers["locked" if locked else "door"] += band
        if exterior and not locked:
            layers["entrance"] += band * rate

    # rooms: the zones of the people's level, by cell centre
    zone_shapes = {z["name"]: floor.shape(z["name"]) for z in floor.zones}
    ox, oy = floor.origin
    room_names = sorted(n for n, s in zone_shapes.items() if s is not None and not s.is_empty)
    room_cells: list[np.ndarray] = []
    for p in PURPOSES:
        layers[f"purpose_{p}"] = np.zeros((grid.ny, grid.nx))
    room_x = []
    for name in room_names:
        shape = shapely.affinity.translate(zone_shapes[name], ox, oy)
        inside = shapely.contains_xy(shape, xs, ys)
        room_cells.append(np.flatnonzero(inside))
        purpose = graph.purposes.get(name, "unknown")
        purpose = purpose if purpose in PURPOSES else "unknown"
        layers[f"purpose_{purpose}"] += inside
        room_x.append((purpose, shape))

    people = scene.people
    layers["starts"] = _splat(grid, [p.start for p in people], [p.weight for p in people])
    stop_pts = [s for p in people for s in p.stops]
    stop_w = [p.weight / len(p.stops) for p in people for _ in p.stops]
    layers["stops"] = _splat(grid, stop_pts, stop_w)
    channels = np.stack([layers[c] for c in CHANNELS]).astype(np.float32)

    # anchors: a room's free point nearest its representative point, a door's middle, a demand's front
    room_anchor = [tuple(shapely.get_coordinates(s.representative_point())[0]) for _, s in room_x]
    door_anchor = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for _, a, b, _, _ in doors]
    starts_per_room = [float(layers["starts"].ravel()[c].sum()) for c in room_cells]
    stops_per_room = [float(layers["stops"].ravel()[c].sum()) for c in room_cells]
    room_doors = {n: [] for n in room_names}
    for di, (dname, *_rest) in enumerate(doors):
        for r in graph.doors.get(dname, []):
            if r in room_doors:
                room_doors[r].append(di)
    nodes: dict[str, Nodes] = {
        "room": Nodes(
            tuple(room_names),
            np.array([[*(float(p == q) for q in PURPOSES), s.area / 100.0, st / 10.0, sp / 10.0, len(room_doors[n])] for n, (p, s), st, sp in zip(room_names, room_x, starts_per_room, stops_per_room, strict=True)], np.float32).reshape(len(room_names), len(NODE_FEATURES["room"])),
            tuple(room_cells),
        ),
        "door": Nodes(
            tuple(d[0] for d in doors),
            np.array([[float(not locked), float(locked), float(exterior), math.dist(a, b)] for _, a, b, locked, exterior in doors], np.float32).reshape(len(doors), len(NODE_FEATURES["door"])),
            tuple(door_cells),
        ),
    }
    room_door = [(ri, di) for ri, n in enumerate(room_names) for di in room_doors[n]]
    rd = walk.between(room_anchor, door_anchor)
    edges: dict[str, Edges] = {"room_door": _edges(room_door, rd)}

    needs: tuple[str, ...] = ()
    need_channels = np.zeros((0, len(NEED_CHANNELS), grid.ny, grid.nx), np.float32)
    demands = [d for d in scene.demands if d.need in scene.needs] if demand else []
    if demand:
        needs = tuple(sorted({d.need for d in demands}))
        stacks = []
        for need in needs:
            mine = [d for d in demands if d.need == need]
            fronts = [d.spot(0) for d in mine]
            queue = [d.spot(k) for d in mine for k in range(1, QUEUE_SPOTS + 1)]
            near = walk.field(fronts)
            trips = expected_trips(scene.needs[need], scene.duration)
            stacks.append(
                np.stack(
                    [
                        _splat(grid, fronts, [d.capacity for d in mine]),
                        _splat(grid, fronts, [d.dwell / 60.0 for d in mine]),
                        _splat(grid, queue, [1.0] * len(queue)),
                        np.where(np.isfinite(near), np.exp(-near / NEAR_M), 0.0),
                        rooms_mask * trips,
                    ]
                )
            )
        if stacks:
            need_channels = np.stack(stacks).astype(np.float32)
        fronts = [d.spot(0) for d in demands]
        demand_cells = tuple(np.array(sorted({c for k in range(QUEUE_SPOTS + 1) if (c := _cell(grid, d.spot(k))) is not None}), np.int64) for d in demands)
        nodes["demand"] = Nodes(tuple(d.name for d in demands), np.array([[d.capacity, d.dwell / 60.0, d.front, d.step] for d in demands], np.float32).reshape(len(demands), len(NODE_FEATURES["demand"])), demand_cells)
        nodes["need"] = Nodes(needs, np.array([[expected_trips(scene.needs[n], scene.duration), scene.needs[n].amount / 100.0, sum(d.need == n for d in demands)] for n in needs], np.float32).reshape(len(needs), len(NODE_FEATURES["need"])), tuple(np.zeros(0, np.int64) for _ in needs))
        # a demand's room: the room whose cells hold its front
        front_cell = [_cell(grid, f) for f in fronts]
        demand_room = [(k, ri) for k, c in enumerate(front_cell) if c is not None for ri, cells in enumerate(room_cells) if c in set(cells.tolist())]
        dr = walk.between(fronts, room_anchor)
        edges["demand_room"] = _edges(demand_room, dr)
        edges["demand_need"] = _edges([(k, needs.index(d.need)) for k, d in enumerate(demands)], np.zeros((len(demands), len(needs))))
        rn = np.full((len(room_names), len(needs)), np.inf)
        for k, d in enumerate(demands):
            n = needs.index(d.need)
            rn[:, n] = np.minimum(rn[:, n], dr[k, :] if len(room_names) else rn[:, n])
        edges["room_need"] = _edges([(ri, n) for ri in range(len(room_names)) for n in range(len(needs))], rn)
        dd = walk.between(fronts, fronts)
        edges["demand_demand"] = _edges([(a, b) for a in range(len(demands)) for b in range(len(demands)) if a != b and demands[a].need == demands[b].need], dd)
    else:
        nodes["demand"] = Nodes((), np.zeros((0, len(NODE_FEATURES["demand"])), np.float32), ())
        nodes["need"] = Nodes((), np.zeros((0, len(NODE_FEATURES["need"])), np.float32), ())
        for name, _, _ in EDGE_KINDS[1:]:
            edges[name] = _edges([], np.zeros((0, 0)))

    people_n = float(sum(p.weight for p in people))
    room_n = max(1, int(rooms_mask.sum()))
    return Raster(
        world=site.name,
        grid=grid,
        channels=channels,
        needs=needs,
        need_channels=need_channels,
        nodes=nodes,
        edges=edges,
        mask=rooms_mask,
        scale=people_n * scene.duration / room_n,
        people=people_n,
        demand=demand,
    )


def _edges(pairs: Sequence[tuple[int, int]], dist: np.ndarray) -> Edges:
    import numpy as np

    src = np.array([a for a, _ in pairs], np.int64)
    dst = np.array([b for _, b in pairs], np.int64)
    d = np.array([dist[a, b] for a, b in pairs], np.float64) if pairs else np.zeros(0)
    # an unreachable pair keeps its edge, at a distance far past the scale
    d = np.where(np.isfinite(d), d, 10 * NEAR_M) / NEAR_M
    return Edges(src, dst, d.astype(np.float32).reshape(-1, 1))


__all__ = ["CHANNELS", "EDGE_KINDS", "NEED_CHANNELS", "NODE_FEATURES", "NODE_KINDS", "PURPOSES", "Edges", "Nodes", "Raster", "expected_trips", "raster"]
