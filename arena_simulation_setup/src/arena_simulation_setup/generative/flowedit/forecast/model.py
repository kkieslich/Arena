"""The demand-driven flow forecaster: a site's expected flow map from its graph and geometry, no PEDS run.

A `Scene` is what the forecaster reads off a site (`scene.py` builds it the way the reader builds a scenario): the flow grid,
the free space people walk, the people (start, idle stops, speed), the demand objects (where each is used, its dwell and
capacity, the need it serves) and each need's rates. `layers` lays every walk the people make along a shortest path on the
free space, once per scene; `predict` weighs those layers with the few `Params` and spreads them with a small kernel.

Per person, over the run's `duration` T (`schedule`):
- **idle**: the person walks from its start to its first stop, then cycles through its stops in order (the reader's
  `idle` sequence), standing `stand` s at each.
- **need trips**: a need falls from its initial value at its decay rate and interrupts the idle timeline once below
  `below`, for a trip from where the person is to the nearest object satisfying it: walk to the object's front, wait in
  its queue, use it for its dwell, walk back to the first stop, where `idle` starts again. A use adds `amount`. The
  seconds in each idle segment, the trips and where they leave from are averaged over draws of the settings' initial
  values and decay rates. Objects of one need share the trips by a softmin of the path length (temperature `tau`;
  HumanSim takes the nearest).
- **queues**: the objects' arrival rates give each a utilisation rho = lambda * dwell / capacity and a wait
  `queue` * dwell * rho^c / (2 c (1 - rho)) (M/D/c, rho at most 0.9), stood on the queue's spots behind the front.

Paths are shortest on the free space with HumanSim's comfort radius as a cost. Walking time goes to each step's heading,
averaged over HEADING_WINDOW steps and spread over the heading bins by `heading` rad; standing time goes to the standing
bin. A person whose start is off the free space or cut off from its stops stands there the whole run, as HumanSim's
router leaves it. The map is spread with a Gaussian of width `sigma`, kept to the cells of the rooms people can reach, and
scaled to the crowd's pedestrian-seconds (people times T), so it compares with a recorded map cell by cell.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import attrs

from .maps import BINS, HEADINGS, STANDING, FlowGrid

if TYPE_CHECKING:
    import numpy as np

Point = tuple[float, float]

#: clearance at which walking costs no more than its length (m): HumanSim's navmesh comfort radius
COMFORT_M = 0.6
#: queue spots behind the user's spot that take the wait
QUEUE_SPOTS = 4
#: utilisation above which a queue counts as saturated
MAX_RHO = 0.9
#: draws of the needs' initial values and decay rates a schedule averages over
NEED_DRAWS = 64
#: people whose shortest-path trees are held at once (a few stops each), to keep memory small
PEOPLE_BATCH = 16
#: path steps a heading is averaged over: a grid path's zigzag read as the straight line it stands for
HEADING_WINDOW = 5
#: directions a path heading is kept at (0.5 degrees)
HEADING_STEPS = 720


@attrs.frozen
class Person:
    """One walker: where it starts, the stops its idle cycle walks through, its desired speed, and its weight (1 / crowd draws)."""

    start: Point
    stops: tuple[Point, ...]
    speed: float
    weight: float = 1.0


@attrs.frozen
class Demand:
    """An object people use for `need`: at `at`, used from `front` m along `facing` (rad), queued at `step` m spacing beyond."""

    name: str
    need: str
    at: Point
    facing: float
    dwell: float
    capacity: int = 1
    front: float = 0.8
    step: float = 1.0

    def spot(self, k: int) -> Point:
        """The k-th spot in front of the object: 0 is the user's, 1... the queue's."""
        d = self.front + k * self.step
        return (self.at[0] + d * math.cos(self.facing), self.at[1] + d * math.sin(self.facing))


@attrs.frozen
class Need:
    """A need's settings (`settings.need`): initial value and decay rate (distributions), the threshold, and what a use adds."""

    initial: Mapping[str, float]
    decay: Mapping[str, float]
    below: float
    amount: float = 100.0


@attrs.define
class Scene:
    """What the forecaster reads off a site. `free` marks the walkable nav cells, `sub` of them per grid cell and side, row 0 at y0."""

    grid: FlowGrid
    free: Any
    sub: int
    people: list[Person]
    demands: list[Demand]
    needs: dict[str, Need]
    duration: float
    stand: float = 8.0
    name: str = ""
    notes: list[str] = attrs.Factory(list)

    @property
    def nav_res(self) -> float:
        return self.grid.cell / self.sub


PARAM_FIELDS = ("sigma", "heading", "walk", "need", "tau", "queue")
#: bounds the fit keeps the params to
PARAM_BOUNDS: Mapping[str, tuple[float, float]] = {"sigma": (0.05, 3.0), "heading": (0.02, 1.5), "walk": (0.05, 20.0), "need": (0.01, 50.0), "tau": (0.1, 50.0), "queue": (0.01, 20.0)}


@attrs.frozen
class Params:
    """The forecaster's fitted parameters."""

    #: kernel width (m)
    sigma: float = 0.5
    #: spread (rad) of a walker's heading about its path's direction
    heading: float = 0.3
    #: walking mass per second walked, relative to standing
    walk: float = 1.0
    #: need trips per expected trip of the needs model
    need: float = 1.0
    #: temperature (m) of the softmin over path length among the objects of one need
    tau: float = 2.0
    #: queue wait per M/D/c wait
    queue: float = 1.0

    def vector(self) -> list[float]:
        return [float(getattr(self, f)) for f in PARAM_FIELDS]

    @classmethod
    def from_vector(cls, values: Sequence[float]) -> Params:
        return cls(**{f: min(max(float(v), PARAM_BOUNDS[f][0]), PARAM_BOUNDS[f][1]) for f, v in zip(PARAM_FIELDS, values, strict=True)})

    def dump(self) -> dict[str, float]:
        return {f: round(float(getattr(self, f)), 6) for f in PARAM_FIELDS}

    @classmethod
    def load(cls, doc: Mapping[str, Any]) -> Params:
        unknown = set(doc) - set(PARAM_FIELDS)
        if unknown:
            raise ValueError(f"forecast params: unknown {sorted(unknown)}, known {list(PARAM_FIELDS)}")
        return cls(**{k: float(v) for k, v in doc.items()})


def _draw(rng: np.random.Generator, dist: Mapping[str, float], n: int) -> np.ndarray:
    """`n` draws of a settings distribution (mean, std, clipped), as HumanSim draws a need's parameters."""
    import numpy as np

    mean, std = float(dist["mean"]), float(dist.get("std", 0.0))
    values = rng.normal(mean, std, n) if std > 0 else np.full(n, mean)
    return np.clip(values, float(dist.get("clip_low", -np.inf)), float(dist.get("clip_high", np.inf)))


def _advance(cover: np.ndarray, durs: np.ndarray, first: float, at_first: bool, length: float) -> tuple[int, float]:
    """Add `length` s of the idle timeline to `cover`, starting with the first walk (`at_first`) or at the cycle's first
    stop; where it ends: (stop, share of the walk on from that stop)."""
    if length <= 0:
        return (0, 0.0)
    if at_first:
        part = min(length, first)
        cover[0] += part
        length -= part
        if length <= 0:
            return (0, 0.0)
    cycle = float(durs.sum())
    if cycle <= 0:
        return (0, 0.0)
    full = math.floor(length / cycle)
    cover[1:] += full * durs
    rest = length - full * cycle
    for s, d in enumerate(durs):
        part = min(rest, float(d))
        cover[1 + s] += part
        rest -= part
        if rest <= 0:
            return (s // 2, 0.0) if s % 2 == 0 else (s // 2, part / d if d > 0 else 1.0)
    return (0, 0.0)


def schedule(first: float, segments: Sequence[float], trips_s: Sequence[tuple[Need, Sequence[float]]], duration: float, draws: int = NEED_DRAWS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A person's expected seconds in each segment of its idle timeline, its expected trips per need, and where they start,
    over `draws` draws of the needs' initial values and decay rates.

    The timeline runs as the reader's `idle` sequence: the walk to the first stop (`first` s), then the cycle `segments`:
    stand at stop 0, walk to stop 1, stand at stop 1, ..., walk back to stop 0 (s each). A need falling below its threshold
    interrupts it for a trip of `trips_s[n][1][i]` s from stop i (from a walk, between its two stops by how far it got). The
    trip ends back at stop 0, where the cycle starts again, and refills the need by its amount (to at most 100). A trip the
    run's end cuts counts by its share. Returns the seconds per segment (the first walk first), the trips per need, and per
    need the share of its trips that leave from each stop.
    """
    import numpy as np

    durs = np.asarray(segments, float)
    k = max(1, len(durs) // 2)
    cover = np.zeros(1 + len(durs))
    trips = np.zeros(len(trips_s))
    origin = np.zeros((len(trips_s), k))
    draws = draws if trips_s else 1
    rng = np.random.default_rng(0)
    level = [_draw(rng, need.initial, draws) for need, _ in trips_s]
    rate = [np.maximum(_draw(rng, need.decay, draws), 1e-6) for need, _ in trips_s]
    refill = [max(0.0, min(100.0, need.below + need.amount) - need.below) for need, _ in trips_s]
    for d in range(draws):
        due = [max(0.0, (level[n][d] - need.below) / rate[n][d]) for n, (need, _) in enumerate(trips_s)]
        t, at_first = 0.0, True
        while True:
            n = int(np.argmin(due)) if due else -1
            until = min(due[n], duration) if n >= 0 else duration
            stop, u = _advance(cover, durs, first, at_first, until - t)
            if n < 0 or due[n] >= duration:
                break
            secs = trips_s[n][1]
            nxt = (stop + 1) % k
            trip = (1 - u) * secs[stop] + u * secs[nxt]
            share = min(1.0, (duration - due[n]) / max(trip, 1e-6))
            trips[n] += share
            origin[n, stop] += share * (1 - u)
            origin[n, nxt] += share * u
            t = due[n] + trip
            if t >= duration:
                break
            due[n] = t + refill[n] / rate[n][d]
            due = [max(x, t) for x in due]
            at_first = False
    return cover / draws, trips / draws, origin / np.maximum(origin.sum(axis=1, keepdims=True), 1e-12)


# --- layers: every walk on shortest paths, once per scene -------------------------------------------------------------

IDLE_WALK, IDLE_STAND, FIRST_WALK, TRIP_WALK, STUCK = range(5)


@attrs.define
class Layers:
    """A scene's walks as sparse deposits (cell, heading, seconds at the person's speed) with what each belongs to, plus the
    per-person and per-object numbers `predict` weighs them with."""

    scene: Scene
    #: flat cell-and-bin index of a standing deposit, flat cell index times bins of a walking one
    flat: Any
    #: walking deposits: path heading (index of HEADING_STEPS from -pi), -1 standing
    heading: Any
    seconds: Any
    person: Any
    kind: Any
    demand: Any
    #: idle deposits: their segment of the person's timeline (0 the first walk, 1 + 2i the stand at stop i, 2 + 2i the walk on)
    segment: Any
    #: (ny, nx) cells a forecast may hold time in: in the rooms and reachable
    mask: Any
    #: per person and segment: expected seconds in it over the run (`schedule`), and the seconds one pass takes
    cover: Any
    passes: Any
    weight: Any
    #: per person and demand: path length to and from (m, inf when unreachable), and walking time (s), by where trips start
    trip_m: Any
    trip_walk_s: Any
    #: per person and need (in `needs` order): expected trips
    trips: Any
    needs: list[str]
    #: per demand: flat cell-and-bin index of standing at the user's spot and the queue spots (-1 off the rooms)
    spots: Any


def _nav_graph(free: np.ndarray, res: float) -> tuple[Any, np.ndarray]:
    """The 8-connected graph of the free nav cells (no corner cutting), each edge its length times the mean comfort cost of its ends."""
    import numpy as np
    import scipy.ndimage
    import scipy.sparse

    ny, nx = free.shape
    clearance = scipy.ndimage.distance_transform_edt(free) * res
    cost = 1.0 + np.clip((COMFORT_M - clearance) / COMFORT_M, 0.0, 1.0)
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, vals = [], [], []
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        i0, i1 = 0, ny - di
        j0, j1 = (0, nx - dj) if dj >= 0 else (-dj, nx)
        ok = free[i0:i1, j0:j1] & free[i0 + di : i1 + di, j0 + dj : j1 + dj]
        if di and dj:
            ok &= free[i0 + di : i1 + di, j0:j1] & free[i0:i1, j0 + dj : j1 + dj]
        length = res * math.hypot(di, dj)
        src = idx[i0:i1, j0:j1][ok]
        dst = idx[i0 + di : i1 + di, j0 + dj : j1 + dj][ok]
        rows.append(src)
        cols.append(dst)
        vals.append(length * (cost.ravel()[src] + cost.ravel()[dst]) / 2)
    n = ny * nx
    graph = scipy.sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return graph, cost


def _nearest_free(free: np.ndarray) -> np.ndarray:
    """For every nav cell, the flat index of the nearest free one."""
    import numpy as np
    import scipy.ndimage

    if free.all():
        return np.arange(free.size)
    _, (ii, jj) = scipy.ndimage.distance_transform_edt(~free, return_indices=True)
    return (ii * free.shape[1] + jj).ravel()


class _Paths:
    """Shortest paths on a scene's nav graph from a set of sources, computed in batches."""

    BATCH = 64

    def __init__(self, scene: Scene) -> None:
        import numpy as np

        self.scene = scene
        free = np.asarray(scene.free, bool)
        self.ny, self.nx = free.shape
        self.res = scene.nav_res
        self.graph, _ = _nav_graph(free, self.res)
        self.nearest = _nearest_free(free)
        g = scene.grid
        xs = g.x0 + (np.arange(self.nx) + 0.5) * self.res
        ys = g.y0 + (np.arange(self.ny) + 0.5) * self.res
        self.x = np.tile(xs, self.ny)
        self.y = np.repeat(ys, self.nx)
        inside = g.in_rooms(self.x, self.y) & free.ravel()
        cell = (np.arange(self.ny)[:, None] // scene.sub * g.nx + np.arange(self.nx)[None, :] // scene.sub).ravel()
        #: flat grid cell of each nav cell, -1 where a sample would fall outside the rooms
        self.cell = np.where(inside, cell, -1)
        self.free = free.ravel()

    def raw(self, p: Point) -> int:
        """The nav cell of point `p` (clamped onto the grid), free or not."""
        g = self.scene.grid
        i = min(max(int((p[1] - g.y0) / self.res), 0), self.ny - 1)
        j = min(max(int((p[0] - g.x0) / self.res), 0), self.nx - 1)
        return i * self.nx + j

    def node(self, p: Point) -> int:
        """The free nav cell nearest point `p`, as HumanSim snaps a target onto its navmesh."""
        return int(self.nearest[self.raw(p)])

    def trees(self, sources: Sequence[int]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Distance and predecessor rows of every source; call per batch to keep memory small."""
        import scipy.sparse.csgraph

        out = {}
        unique = sorted(set(sources))
        for k in range(0, len(unique), self.BATCH):
            batch = unique[k : k + self.BATCH]
            dist, pred = scipy.sparse.csgraph.dijkstra(self.graph, directed=False, indices=batch, return_predecessors=True)
            for s, d, p in zip(batch, dist, pred, strict=True):
                out[s] = (d, p)
        return out

    @staticmethod
    def trace(pred: np.ndarray, source: int, target: int) -> list[int] | None:
        """The nodes from `source` to `target` on a predecessor row of `source`, None when unreachable."""
        if source == target:
            return [source]
        path = [target]
        while path[-1] != source:
            nxt = int(pred[path[-1]])
            if nxt < 0:
                return None
            path.append(nxt)
        return path[::-1]

    def deposit(self, nodes: Sequence[int], speed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """(flat cell index times bins, heading index, seconds) of walking `nodes` at `speed`, half of each step's time at
        either end, its heading averaged over HEADING_WINDOW steps, and the path's length (m) and time (s)."""
        import numpy as np

        if len(nodes) < 2:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), 0.0, 0.0
        n = np.asarray(nodes)
        dx, dy = np.diff(self.x[n]), np.diff(self.y[n])
        step = np.hypot(dx, dy)
        dt = step / max(speed, 1e-3)
        window, start = np.ones(HEADING_WINDOW), (HEADING_WINDOW - 1) // 2
        theta = np.arctan2(np.convolve(dy / step, window)[start : start + len(step)], np.convolve(dx / step, window)[start : start + len(step)])
        h = np.floor((theta + math.pi) / (2 * math.pi) * HEADING_STEPS).astype(np.int64) % HEADING_STEPS
        cells = np.concatenate([self.cell[n[:-1]], self.cell[n[1:]]])
        heads = np.concatenate([h, h])
        secs = np.concatenate([dt / 2, dt / 2])
        ok = cells >= 0
        return cells[ok] * BINS, heads[ok], secs[ok], float(step.sum()), float(dt.sum())

    def stand(self, node: int) -> int:
        """Flat cell-and-bin index of standing at `node`, -1 off the rooms."""
        c = int(self.cell[node])
        return -1 if c < 0 else c * BINS + STANDING

    def stand_at(self, p: Point) -> int:
        """Flat cell-and-bin index of standing at point `p`, free or not, -1 off the rooms."""
        g = self.scene.grid
        i, j, ok = g.index(p[0], p[1])
        return int(i * g.nx + j) * BINS + STANDING if bool(ok) and bool(g.in_rooms(p[0], p[1])) else -1


def layers(scene: Scene) -> Layers:
    """Every walk of `scene` on shortest paths, as sparse deposits `predict` weighs, and each person's schedule."""
    import numpy as np

    paths = _Paths(scene)
    people, demands = scene.people, scene.demands
    needs = sorted({d.need for d in demands} & set(scene.needs))
    flat: list[np.ndarray] = []
    heading: list[np.ndarray] = []
    seconds: list[np.ndarray] = []
    meta: list[tuple[int, int, int, int, int]] = []  # (person, kind, demand, segment, length)

    def add(f: Sequence[int] | np.ndarray, s: Sequence[float] | np.ndarray, person: int, kind: int, demand: int = -1, segment: int = -1, h: np.ndarray | None = None) -> None:
        f, s = np.asarray(f, np.int64), np.asarray(s, float)
        keep = f >= 0
        if keep.any():
            flat.append(f[keep])
            heading.append(np.full(int(keep.sum()), -1, np.int64) if h is None else np.asarray(h)[keep])
            seconds.append(s[keep])
            meta.append((person, kind, demand, segment, int(keep.sum())))

    spot_nodes = [[paths.node(d.spot(k)) for k in range(QUEUE_SPOTS + 1)] for d in demands]
    fronts = paths.trees([s[0] for s in spot_nodes])
    stop_nodes = [[paths.node(p) for p in person.stops] for person in people]
    start_nodes = [paths.node(person.start) for person in people]
    n_people, n_demands = len(people), len(demands)
    width = 1 + 2 * max((len(s) for s in stop_nodes), default=0)
    passes, cover = np.zeros((n_people, width)), np.zeros((n_people, width))
    stuck = np.zeros(n_people, bool)

    # idle: the walk to the first stop, the stand at each stop and the walk on to the next
    for k in range(0, n_people, PEOPLE_BATCH):
        chunk = range(k, min(n_people, k + PEOPLE_BATCH))
        trees = paths.trees([n for p in chunk for n in [start_nodes[p], *stop_nodes[p]]])
        for p in chunk:
            person, stops = people[p], stop_nodes[p]
            route = paths.trace(trees[start_nodes[p]][1], start_nodes[p], stops[0]) if stops and paths.free[paths.raw(person.start)] else None
            if route is None:
                # a start off the free space or cut off from the stops: HumanSim's router finds no way, the person stands there
                stuck[p] = True
                add([paths.stand_at(person.start)], [scene.duration], p, STUCK)
                continue
            f, h, s, _, t = paths.deposit(route, person.speed)
            add(f, s, p, FIRST_WALK, segment=0, h=h)
            passes[p, 0] = t
            for i, a in enumerate(stops):
                add([paths.stand(a)], [scene.stand], p, IDLE_STAND, segment=1 + 2 * i)
                passes[p, 1 + 2 * i] = scene.stand
                b = stops[(i + 1) % len(stops)]
                if len(stops) > 1 and (route := paths.trace(trees[a][1], a, b)) is not None:
                    f, h, s, _, t = paths.deposit(route, person.speed)
                    add(f, s, p, IDLE_WALK, segment=2 + 2 * i, h=h)
                    passes[p, 2 + 2 * i] = t

    # trips: per person and object, the walk there from each stop and the walk back to the first stop
    parts: dict[tuple[int, int], tuple[tuple, list[tuple | None]]] = {}
    for o in range(n_demands):
        front = spot_nodes[o][0]
        dist, pred = fronts[front]
        for p, person in enumerate(people):
            if stuck[p] or not np.isfinite(dist[stop_nodes[p][0]]):
                continue
            back = paths.deposit(paths.trace(pred, front, stop_nodes[p][0]), person.speed)
            there = [paths.deposit(path[::-1], person.speed) if (path := paths.trace(pred, front, s)) is not None else None for s in stop_nodes[p]]
            parts[(p, o)] = (back, there)

    # schedules: each need's trips go to its nearest object from each stop
    trips = np.zeros((n_people, len(needs)))
    origin: dict[tuple[int, int], np.ndarray] = {}
    for p in range(n_people):
        if stuck[p]:
            continue
        k = len(stop_nodes[p])
        per_need, kept = [], []
        for n, need in enumerate(needs):
            objs = [o for o, d in enumerate(demands) if d.need == need and (p, o) in parts]
            secs = [min((parts[(p, o)][1][i][4] + parts[(p, o)][0][4] + demands[o].dwell for o in objs if parts[(p, o)][1][i] is not None), default=math.inf) for i in range(k)]
            finite = [x for x in secs if math.isfinite(x)]
            if finite:
                per_need.append((scene.needs[need], [x if math.isfinite(x) else max(finite) for x in secs]))
                kept.append(n)
        c, t, w = schedule(passes[p, 0], passes[p, 1 : 1 + 2 * k], per_need, scene.duration)
        cover[p, : 1 + 2 * k] = c
        for j, n in enumerate(kept):
            trips[p, n] = t[j]
            origin[(p, n)] = w[j]

    # trip deposits: the walk back, and the walks there by where the trips leave from
    trip_m = np.full((n_people, n_demands), np.inf)
    trip_walk_s = np.zeros((n_people, n_demands))
    for (p, o), (back, there) in parts.items():
        reach = [i for i, x in enumerate(there) if x is not None]
        if not reach:
            continue
        w = origin.get((p, needs.index(demands[o].need)) if demands[o].need in needs else (p, -1))
        ws = np.array([1.0 if w is None else float(w[i]) for i in reach])
        ws = ws / ws.sum() if ws.sum() > 0 else np.full(len(reach), 1.0 / len(reach))
        add(
            np.concatenate([back[0], *(there[i][0] for i in reach)]),
            np.concatenate([back[2], *(there[i][2] * wi for i, wi in zip(reach, ws, strict=True))]),
            p,
            TRIP_WALK,
            o,
            h=np.concatenate([back[1], *(there[i][1] for i in reach)]),
        )
        trip_m[p, o] = back[3] + float(sum(there[i][3] * wi for i, wi in zip(reach, ws, strict=True)))
        trip_walk_s[p, o] = back[4] + float(sum(there[i][4] * wi for i, wi in zip(reach, ws, strict=True)))

    spots = np.array([[paths.stand(n) for n in nodes] for nodes in spot_nodes], np.int64).reshape(n_demands, QUEUE_SPOTS + 1)
    reach_cells = np.zeros(scene.grid.nx * scene.grid.ny, bool)
    reach_cells[paths.cell[paths.cell >= 0]] = True
    mask = reach_cells.reshape(scene.grid.ny, scene.grid.nx) & scene.grid.room_cells()
    sizes = [m[4] for m in meta]

    def column(i: int) -> np.ndarray:
        return np.repeat(np.array([m[i] for m in meta], np.int64), sizes) if meta else np.zeros(0, np.int64)

    return Layers(
        scene=scene,
        flat=np.concatenate(flat) if flat else np.zeros(0, np.int64),
        heading=np.concatenate(heading) if heading else np.zeros(0, np.int64),
        seconds=np.concatenate(seconds) if seconds else np.zeros(0),
        person=column(0),
        kind=column(1),
        demand=column(2),
        segment=column(3),
        mask=mask,
        cover=cover,
        passes=passes,
        weight=np.array([p.weight for p in people]),
        trip_m=trip_m,
        trip_walk_s=trip_walk_s,
        trips=trips,
        needs=needs,
        spots=spots,
    )


# --- predict: weigh the layers with the params -----------------------------------------------------------------------


def heading_table(spread: float) -> np.ndarray:
    """(HEADING_STEPS, HEADINGS): the share of each heading bin for a walker whose path heads at each direction, its heading
    spread normally by `spread` rad (wrapped): a path along a bin edge, as a corridor along an axis is, splits it evenly."""
    import numpy as np
    import scipy.special

    theta = -math.pi + (np.arange(HEADING_STEPS) + 0.5) * 2 * math.pi / HEADING_STEPS
    lo = -math.pi + np.arange(HEADINGS) * 2 * math.pi / HEADINGS
    hi = lo + 2 * math.pi / HEADINGS
    w = np.zeros((HEADING_STEPS, HEADINGS))
    for k in (-1, 0, 1):
        w += scipy.special.ndtr((hi[None, :] - theta[:, None] + k * 2 * math.pi) / spread) - scipy.special.ndtr((lo[None, :] - theta[:, None] + k * 2 * math.pi) / spread)
    return w / w.sum(axis=1, keepdims=True)


def trip_shares(layers: Layers, params: Params) -> np.ndarray:
    """Per person and demand: expected trips there, the person's trips for the need shared by a softmin of path length."""
    import numpy as np

    demands = layers.scene.demands
    shares = np.zeros(layers.trip_m.shape)
    for n, need in enumerate(layers.needs):
        objs = [o for o, d in enumerate(demands) if d.need == need]
        dist = layers.trip_m[:, objs]
        finite = np.isfinite(dist)
        d = np.where(finite, dist, 0.0)
        nearest = np.where(finite, d, np.inf).min(axis=1, keepdims=True)
        w = np.where(finite, np.exp(-(d - np.where(np.isfinite(nearest), nearest, 0.0)) / params.tau), 0.0)
        total = w.sum(axis=1, keepdims=True)
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        shares[:, objs] = params.need * layers.trips[:, [n]] * w
    return shares


def waits(layers: Layers, shares: np.ndarray, params: Params) -> tuple[np.ndarray, np.ndarray]:
    """Per demand: the queue wait (s) and its utilisation."""
    import numpy as np

    scene = layers.scene
    wait, rho = np.zeros(len(scene.demands)), np.zeros(len(scene.demands))
    for o, d in enumerate(scene.demands):
        rate = float((layers.weight * shares[:, o]).sum()) / max(scene.duration, 1e-6)
        c = max(1, int(d.capacity))
        rho[o] = min(MAX_RHO, rate * d.dwell / c)
        wait[o] = params.queue * d.dwell * rho[o] ** c / (2 * c * (1 - rho[o]))
    return wait, rho


def predict(layers: Layers, params: Params) -> np.ndarray:
    """The forecast map (ny, nx, bins) in pedestrian-seconds: people times the duration, over the reachable cells of the rooms."""
    import numpy as np
    import scipy.ndimage

    scene = layers.scene
    T = scene.duration
    shares = trip_shares(layers, params)
    wait, rho = waits(layers, shares, params)
    dwell = np.array([d.dwell for d in scene.demands])
    on_trips = (shares * (layers.trip_walk_s + dwell + wait)).sum(axis=1) if len(scene.demands) else np.zeros(len(layers.weight))
    # the idle time the params leave, spread over the timeline as the schedule spreads it
    nominal = layers.cover.sum(axis=1)
    idle = np.maximum(0.0, T - on_trips)
    factor = np.divide(idle, nominal, out=np.zeros_like(idle), where=nominal > 0)
    per_pass = np.divide(layers.cover, layers.passes, out=np.zeros_like(layers.cover), where=layers.passes > 0) * factor[:, None]

    p, kind = layers.person, layers.kind
    scale = np.zeros(len(layers.seconds))
    idle_kind = (kind == IDLE_WALK) | (kind == IDLE_STAND) | (kind == FIRST_WALK)
    scale[idle_kind] = per_pass[p[idle_kind], layers.segment[idle_kind]]
    scale[(kind == IDLE_WALK) | (kind == FIRST_WALK)] *= params.walk
    scale[kind == STUCK] = 1.0
    trip = kind == TRIP_WALK
    scale[trip] = shares[p[trip], layers.demand[trip]] * params.walk
    scale *= layers.weight[p]
    g = scene.grid
    size = g.ny * g.nx * BINS
    mass = layers.seconds * scale
    walking = layers.heading >= 0
    out = np.bincount(layers.flat[~walking], weights=mass[~walking], minlength=size)[:size]
    shares_of = heading_table(params.heading)[layers.heading[walking]]
    out += np.bincount((layers.flat[walking][:, None] + np.arange(HEADINGS)).ravel(), weights=(mass[walking][:, None] * shares_of).ravel(), minlength=size)[:size]

    # standing at the objects: the use at the user's spot, the wait on the queue spots by how often the queue reaches them
    for o, d in enumerate(scene.demands):
        visits = float((layers.weight * shares[:, o]).sum())
        spots = layers.spots[o]
        if spots[0] >= 0:
            out[spots[0]] += visits * d.dwell
        reach = np.array([rho[o] ** k for k in range(1, QUEUE_SPOTS + 1)])
        if reach.sum() > 0:
            for k, f in enumerate(spots[1:]):
                if f >= 0:
                    out[f] += visits * wait[o] * reach[k] / reach.sum()

    out = out.reshape(g.ny, g.nx, BINS)
    if params.sigma >= 0.05:
        s = params.sigma / g.cell
        out = scipy.ndimage.gaussian_filter(out, sigma=(s, s, 0), mode="constant", truncate=3.0)
    out *= layers.mask[:, :, None]
    total = float(out.sum())
    people_s = float(layers.weight.sum()) * T
    return out * (people_s / total) if total > 0 else out


__all__ = ["PARAM_BOUNDS", "PARAM_FIELDS", "Demand", "Layers", "Need", "Params", "Person", "Scene", "heading_table", "layers", "predict", "schedule", "trip_shares", "waits"]
