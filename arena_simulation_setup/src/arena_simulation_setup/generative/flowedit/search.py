"""INFRA layout searches (study design §4, §5): the valid layouts of the catalog's elements on a site, the robot-only
pick (the shortest expected robot travel for the §8 task stream and the robots' top-up trips, as in Kundu and Saha's
charger placement), the joint search's ranking by predicted conflict exposure per walker group (`joint_exposure`: the
time a group's member spends near the robots' routes or on the elements' floor, from the site-as-is reference runs;
the older occupancy fallback `joint` beside it), the guideline placement (`guideline`) and random valid layouts.

A layout places the charger pair, the bay pair and the locker on valid candidate spots (`place.candidate_spots`), with
the loading point pinned to one spot (route end A on the twin, §8). Each element may go where its function puts it
(`Placement`: world zones, world-frame areas or a pinned point; the INFRA site config gives them, see `infra.SiteConfig`).
`Search` holds one site's search space:

- the valid spots of each element in its placement, less those that touch a keep-out polygon (a PEDS source or sink)
  or clash with the loading point;
- which spot pairs `place.check_layout` passes. Its checks look at most a corridor window away from a spot, so a pair
  further apart always passes. A layout of pairwise compatible spots is a candidate; picks get the full check;
- a NAV@r base of the site without the elements (`metrics.navr_filter.NavrBase`). A layout is valid when
  `check_layout` passes and `navr_check` accepts it;
- robot path lengths between the dock goto points on the jackal's floor at r 0.267 m (`RobotPaths`).

The joint search (§5) ranks only layouts whose expected robot travel per job is within `TRAVEL_SLACK` of the robot-only
pick's (`travel_cap`), so its top layouts can meet the equal-service constraint (§6: T within 5 % of the robot-only
pick's). The premise check (§10) runs the two most different layouts, all elements at the narrowest and at the widest
section (`premise_pair`).

Layouts keep `Layout.layout_id`, a hash of their spots, so the same layout has the same id in every search and run.
The searches are deterministic: the same site, arguments and seed give the same layouts."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from . import place
from .place import EXCLUDED_ZONES, Layout, Spot

if TYPE_CHECKING:
    import numpy as np
    import shapely

    from arena_simulation_setup.metrics.navr_filter import NavrBase, NavrVerdict

    from .edit import Site

Point = tuple[float, float]

#: The elements the searches place, in catalog order; the loading point is pinned, not searched (§8)
SEARCHED = ("charger_pair", "bay_pair", "locker")
LOADING = "loading_point"
#: Route end A of the twin smu_link_full, the south end of lower_concourse: the loading point of the twin agent's layout
TWIN_LOADING_XY = (152.44, 19.61)
#: How far a pinned element (the loading point) may lie from the goto point of the valid spot it snaps to, or from that
#: spot's bay (a site marks its courier access point on the wall behind the bay) [m]
PIN_SNAP_M = 0.5
#: Each robot's home, one robot each, in robot order (the fleet mode's `homes`)
HOMES = ("bay_1", "bay_2", "charger_1", "charger_2")
#: The cell size of the robot path raster: NAV@r's, so a dock the NAV@r filter reaches has a path
PATH_RESOLUTION_M = 0.05
#: A point further than this from the path raster's floor is unreachable (a dock in a cut-off zone)
PATH_SNAP_M = 1.0
#: The joint search's sample size (§4, placeholder), how many layouts it ranks on (§5), and the random control's count
JOINT_SAMPLES = 2000
JOINT_TOP = 10
RANDOM_LAYOUTS = 5
#: Pairwise compatible layouts a search holds at most (index arrays of 12 bytes each)
MAX_CANDIDATES = 50_000_000
#: The joint search's travel pre-filter: a layout's expected robot travel per job at most this times the robot-only
#: pick's, the modelled side of the equal-service constraint (§6: T <= 1.05 T(robot-only))
TRAVEL_SLACK = 1.05
#: Seed streams, so the joint sample and the random control are independent draws for the same seed
_JOINT, _RANDOM = 1, 2
#: The walker groups the joint ranking scores (§6, §9; arena-evaluation's `infra.GROUPS`)
GROUPS = ("adult", "wheelchair", "long_cane")
#: The joint ranking's conflict band: a walker within this distance of a driving robot counts as exposed to it, a band
#: about the robots' routes, since nav2 does not drive the exact shortest line [m]
CONFLICT_BAND_M = 2.0
#: The robots' speed while they drive a route, for the time they spend on it (the jackal's planner limit) [m/s]
ROBOT_SPEED_MPS = 0.5
#: The holds at a job's ends, at the loading point and at the locker (the fleet mode's load_s and unload_s) [s]
HOLD_S = (15.0, 10.0)
#: The joint top 10 are distinct layouts: one enters only if one of its elements sits at least this far from that
#: element's spot in every better-ranked layout [m]
DISTINCT_M = 5.0
#: Candidates the exposure ranking checks at most before it gives up filling its top (each takes the full check)
MAX_RANKED_CHECKS = 50_000


@attrs.frozen
class Placement:
    """Where one element may go, by its function (study design §3, §4; the INFRA site config): `zones` of the world (names
    or `fnmatch` patterns; None: every zone the generator allows, less the excluded ones), `areas` (world-frame polygons
    every object of a spot must stand in; none: no limit), or a `pin` (a level point the element snaps to, `PIN_SNAP_M`)."""

    zones: tuple[str, ...] | None = None
    areas: tuple[Any, ...] = ()
    pin: Point | None = None

    def holds(self, spot: Spot) -> bool:
        """Whether every object of `spot` stands in one of `areas` (True without areas)."""
        if not self.areas:
            return True
        import shapely

        return all(any(area.covers(shapely.Point(o.x, o.y)) for area in self.areas) for o in spot.objects)


@attrs.frozen
class TaskMix:
    """The §8 task stream the robot-only search weighs robot travel with.

    Jobs come at `rate_per_h`, a share `pickup_share` of them pickups (locker to loading point), the rest deliveries
    (loading point to locker). The fleet mode sends a free robot home, so a job runs home, load end, unload end, home.
    Homes are the 2 bay slots and the 2 chargers (`HOMES`), one robot each, so jobs start and end at each home equally
    often. After `charges_per_job` of the jobs the robot drives to a charger instead of home and keeps it as its home
    (the fleet mode's swap). Each robot charges `charges_per_robot` times per episode of `episode_s` (§8 sizes the battery
    for about once), so `charges_per_job` is robots x charges over the jobs of an episode, at most 1. Chained jobs (a
    robot that takes the next job where the last ended) are not modelled: the home legs are an upper bound.

    `top_ups_per_job` adds the charging trips idle robots make between a bay slot and a charger (the fleet mode's
    top-ups, from standby drain while parked): that many round trips bay - charger per job, at the mean bay-charger
    path length (`Search.travel`). The study takes it from the task-rate run's fleet log: top-ups dispatched after the
    warm-up per completed job released after it."""

    rate_per_h: float = 12.0
    pickup_share: float = 0.3
    robots: int = len(HOMES)
    episode_s: float = 1500.0
    charges_per_robot: float = 1.0
    top_ups_per_job: float = 0.0

    def __attrs_post_init__(self) -> None:
        if self.top_ups_per_job < 0:
            raise ValueError(f"top_ups_per_job must not be negative, got {self.top_ups_per_job}")

    @property
    def charges_per_job(self) -> float:
        jobs = self.rate_per_h * self.episode_s / 3600.0
        return min(1.0, self.robots * self.charges_per_robot / jobs) if jobs > 0 else 0.0

    def weights(self) -> dict[str, float]:
        """Per job, the weight of the mean path length from the loading point (`L`) and from the locker (`K`) to the bay
        slots and to the chargers; the loading point to locker leg counts once.

        A delivery (1 - p) drives home to L, L to K, then K home (1 - q) or K to a charger (q); a pickup (p) home to K,
        K to L, then L home (1 - q) or L to a charger (q). A home is a bay slot or a charger, half and half."""
        p, q = self.pickup_share, self.charges_per_job
        return {"L_bay": (1 - p * q) / 2, "L_charger": (1 + p * q) / 2, "K_bay": (1 - q + p * q) / 2, "K_charger": (1 + q - p * q) / 2}


@attrs.frozen(eq=False)
class RobotPaths:
    """Shortest robot path lengths (m) between level points on a NAV@r base's floor at one radius: the base's
    traversable cells, their largest 8-connected component, edges between neighbouring cells (a diagonal sqrt(2) cells
    long), so a path runs through the floor NAV@r calls reachable. A coarser `resolution` keeps every k-th cell (its
    block's middle cell): it cannot join floor across an obstacle, which keeps the radius between them on both sides,
    but it can drop a passage narrower than a block. A point snaps to the nearest cell of that component and the snap
    distance counts; beyond `PATH_SNAP_M` it is unreachable (inf). The elements of a layout are no obstacles here:
    paths are those of the site as is. 8-connected paths run up to 8 % longer than straight ones off the grid axes."""

    resolution: float
    centres: np.ndarray
    graph: Any
    tree: Any

    @classmethod
    def from_base(cls, base: NavrBase, radius: float | None = None, resolution: float = PATH_RESOLUTION_M) -> RobotPaths:
        import numpy as np
        import scipy.sparse
        from scipy import ndimage
        from scipy.spatial import cKDTree

        from arena_simulation_setup.metrics.navr_filter import JACKAL_RADIUS

        fine = base.layer(JACKAL_RADIUS if radius is None else radius).traversable
        res = base.resolution
        k = max(1, round(resolution / res))
        height = fine.shape[0]
        coarse = fine[k // 2 :: k, k // 2 :: k]
        rows, cols = coarse.shape
        labels, count = ndimage.label(coarse, structure=np.ones((3, 3), dtype=bool))
        if count == 0:
            raise ValueError("the base has no traversable floor at the path radius")
        sizes = np.bincount(labels.ravel())[1:]
        floor = labels == int(np.argmax(sizes)) + 1
        del labels
        ids = np.full(floor.shape, -1, dtype=np.int64)
        r, c = np.nonzero(floor)
        ids[r, c] = np.arange(len(r))
        ox, oy = base.origin
        fr, fc = r * k + k // 2, c * k + k // 2
        centres = np.column_stack([ox + (fc + 0.5) * res, oy + (height - fr - 0.5) * res])
        step = k * res
        heads, tails, weights = [], [], []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a = ids[: rows - dr, max(0, -dc) : cols - max(0, dc)]
            b = ids[dr:, max(0, dc) : cols - max(0, -dc)]
            both = (a >= 0) & (b >= 0)
            heads.append(a[both])
            tails.append(b[both])
            weights.append(np.full(int(both.sum()), step * math.hypot(dr, dc)))
        n = len(r)
        graph = scipy.sparse.csr_matrix((np.concatenate(weights), (np.concatenate(heads), np.concatenate(tails))), shape=(n, n))
        return cls(step, centres, graph, cKDTree(centres))

    @property
    def cells(self) -> int:
        return len(self.centres)

    def lengths(self, sources: Sequence[Point], targets: Sequence[Point], chunk: int = 16) -> np.ndarray:
        """The path length from each of `sources` to each of `targets`, (len(sources), len(targets)), inf where none."""
        import numpy as np
        from scipy.sparse.csgraph import dijkstra

        src_d, src_i = self.tree.query(np.asarray(sources, dtype=float).reshape(-1, 2))
        dst_d, dst_i = self.tree.query(np.asarray(targets, dtype=float).reshape(-1, 2))
        out = np.full((len(src_i), len(dst_i)), np.inf)
        for start in range(0, len(src_i), chunk):
            block = dijkstra(self.graph, directed=False, indices=src_i[start : start + chunk])
            out[start : start + chunk] = block[:, dst_i]
        out += src_d[:, None] + dst_d[None, :]
        out[src_d > PATH_SNAP_M, :] = np.inf
        out[:, dst_d > PATH_SNAP_M] = np.inf
        return out

    def integrals(self, sources: Sequence[Point], targets: Sequence[Point], values: np.ndarray, chunk: int = 8) -> tuple[np.ndarray, np.ndarray]:
        """Along the shortest path from each of `sources` to each of `targets` (the paths `lengths` measures): its length
        and the path integral of each row of `values` (one value per path cell, shape (K, cells)): the sum over the
        path's steps of the step's length times the value at the cell it enters, plus each snap distance times the value
        at the snapped cell. Returns (lengths (S, T), integrals (K, S, T)), inf where no path. Each source's shortest
        path tree is summed by pointer jumping (log2 of the path's cells passes over every cell)."""
        import numpy as np
        from scipy.sparse.csgraph import dijkstra

        values = np.atleast_2d(np.asarray(values, dtype=float))
        if values.shape[1] != self.cells:
            raise ValueError(f"values hold {values.shape[1]} cells per row, the paths have {self.cells}")
        src_d, src_i = self.tree.query(np.asarray(sources, dtype=float).reshape(-1, 2))
        dst_d, dst_i = self.tree.query(np.asarray(targets, dtype=float).reshape(-1, 2))
        src_d, src_i, dst_d, dst_i = np.atleast_1d(src_d), np.atleast_1d(src_i), np.atleast_1d(dst_d), np.atleast_1d(dst_i)
        lengths = np.full((len(src_i), len(dst_i)), np.inf)
        out = np.full((values.shape[0], len(src_i), len(dst_i)), np.inf)
        every = np.arange(self.cells)
        for start in range(0, len(src_i), chunk):
            dist, pred = dijkstra(self.graph, directed=False, indices=src_i[start : start + chunk], return_predecessors=True)
            dist, pred = np.atleast_2d(dist), np.atleast_2d(pred)
            for row in range(dist.shape[0]):
                parent = pred[row].astype(np.int64)
                root = parent < 0  # the source and the cells no path reaches point at themselves
                parent[root] = every[root]
                acc = values * np.linalg.norm(self.centres - self.centres[parent], axis=1)[None, :]
                pointer = parent
                for _ in range(64):
                    acc = acc + acc[:, pointer]
                    pointer = pointer[pointer]
                    if np.array_equal(pointer[pointer], pointer):
                        break
                s = start + row
                lengths[s] = dist[row, dst_i] + src_d[s] + dst_d
                out[:, s, :] = acc[:, dst_i] + values[:, [src_i[s]]] * src_d[s] + values[:, dst_i] * dst_d[None, :]
        far = (src_d > PATH_SNAP_M)[:, None] | (dst_d > PATH_SNAP_M)[None, :]
        lengths[far] = np.inf
        out[:, ~np.isfinite(lengths)] = np.inf
        return lengths, out


@attrs.frozen(eq=False)
class Occupancy:
    """Pedestrian occupancy of a site: the time-averaged density (pedestrians per m^2) on square cells, `density[i, j]`
    for the cell from (`x0` + j `cell_m`, `y0` + i `cell_m`), world frame."""

    x0: float
    y0: float
    cell_m: float
    density: np.ndarray

    @classmethod
    def from_cells(cls, x: Sequence[float], y: Sequence[float], ped_seconds: Sequence[float], cell_m: float, seconds: float) -> Occupancy:
        """The occupancy of flow-map cells (centres `x`, `y`, `cell_m` wide) holding `ped_seconds` each, over `seconds` of
        recording in all (runs x episode length): the pedestrian-seconds summed per cell, over `seconds` and the cell's area.
        A cell listed twice (per heading bin, per run) adds up."""
        import numpy as np

        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        x, y, w = (np.asarray(v, dtype=float) for v in (x, y, ped_seconds))
        if not len(x):
            raise ValueError("no flow cells")
        x0, y0 = float(x.min()) - cell_m / 2.0, float(y.min()) - cell_m / 2.0
        j = np.floor((x - x0) / cell_m).astype(np.int64)
        i = np.floor((y - y0) / cell_m).astype(np.int64)
        density = np.zeros((int(i.max()) + 1, int(j.max()) + 1))
        np.add.at(density, (i, j), w)
        return cls(x0, y0, float(cell_m), density / (seconds * cell_m * cell_m))

    @classmethod
    def from_flow_cells(cls, path: Path, world: str, episode_s: float, planner: str | None = None) -> Occupancy:
        """The occupancy of `world`'s runs in arena_evaluation's flow cells (`flow_cells.parquet`, or a benchmark run
        directory holding it): every run (stage and planner, all seeds) of `world`, each `episode_s` long, warm-up
        included. `planner` keeps one contestant's runs."""
        import pandas as pd

        path = Path(path)
        if path.is_dir():
            path = path / "flow_cells.parquet"
        cells = pd.read_parquet(path)
        cells = cells[cells["flow_world"] == world]
        if planner is not None:
            cells = cells[cells["planner"] == planner]
        if cells.empty:
            raise ValueError(f"{path} has no flow cells of world {world!r}" + (f" and planner {planner!r}" if planner else ""))
        sizes = sorted(set(cells["cell_m"]))
        if len(sizes) != 1:
            raise ValueError(f"{path}: flow cells of {world!r} come in sizes {sizes}")
        runs = len(cells[["planner", "stage"]].drop_duplicates())
        return cls.from_cells(cells["x"], cells["y"], cells["ped_seconds"], sizes[0], runs * episode_s)

    @classmethod
    def from_group_cells(cls, paths: Path | Sequence[Path], world: str, episode_s: float) -> dict[str, Occupancy]:
        """The occupancy of each walker group of `world`'s runs in arena-evaluation's group cells (`evaluation infra
        occupancy`: `infra_occupancy.parquet`, or a benchmark run directory holding it; several are read as one): per
        group every run (stage and planner) of `world`, each `episode_s` long, warm-up included, as `from_flow_cells`.
        Groups by name (`GROUPS`); a group with no walker time gets no entry."""
        import pandas as pd

        frames = []
        for path in [paths] if isinstance(paths, (str, Path)) else list(paths):
            path = Path(path)
            if path.is_dir():
                path = path / "infra_occupancy.parquet"
            frames.append(pd.read_parquet(path))
        cells = pd.concat(frames, ignore_index=True)
        cells = cells[cells["world"] == world]
        if cells.empty:
            raise ValueError(f"{paths}: no group cells of world {world!r}")
        sizes = sorted(set(cells["cell_m"]))
        if len(sizes) != 1:
            raise ValueError(f"{paths}: group cells of {world!r} come in sizes {sizes}")
        runs = len(cells[["planner", "stage"]].drop_duplicates())
        out = {}
        for group, rows in cells.groupby("group", sort=True):
            if float(rows["ped_seconds"].sum()) > 0:
                out[str(group)] = cls.from_cells(rows["x"], rows["y"], rows["ped_seconds"], sizes[0], runs * episode_s)
        return out

    @property
    def mean_density(self) -> float:
        """The mean density over the cells that hold any pedestrian time: the walked floor's."""
        import numpy as np

        walked = self.density[self.density > 0]
        return float(walked.mean()) if len(walked) else 0.0

    @property
    def present(self) -> float:
        """The mean number of pedestrians on the site at a time: the density integrated over every cell."""
        return float(self.density.sum()) * self.cell_m * self.cell_m

    def band(self, radius: float) -> np.ndarray:
        """Per cell, the mean number of pedestrians within `radius` of its centre: the density times the cell area,
        summed over the cells whose centres lie in that disc, scaled so the disc counts its exact area pi r^2."""
        import numpy as np
        from scipy import ndimage

        k = int(math.ceil(radius / self.cell_m))
        yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
        disc = ((xx * self.cell_m) ** 2 + (yy * self.cell_m) ** 2 <= radius * radius).astype(float)
        disc *= math.pi * radius * radius / (disc.sum() * self.cell_m * self.cell_m)
        return ndimage.convolve(self.density * self.cell_m * self.cell_m, disc, mode="constant")

    def sample(self, field: np.ndarray, points: np.ndarray) -> np.ndarray:
        """`field` (one value per cell of this grid) at world-frame `points` (N, 2): each point's cell, 0 off the grid."""
        import numpy as np

        points = np.asarray(points, dtype=float).reshape(-1, 2)
        j = np.floor((points[:, 0] - self.x0) / self.cell_m).astype(np.int64)
        i = np.floor((points[:, 1] - self.y0) / self.cell_m).astype(np.int64)
        inside = (i >= 0) & (i < field.shape[0]) & (j >= 0) & (j < field.shape[1])
        out = np.zeros(len(points))
        out[inside] = field[i[inside], j[inside]]
        return out

    def people(self, geometry: shapely.Geometry) -> float:
        """The mean number of pedestrians on `geometry` at a time: the density integrated over it, cell by cell."""
        import numpy as np
        import shapely

        if geometry.is_empty:
            return 0.0
        ny, nx = self.density.shape
        x0, y0, x1, y1 = geometry.bounds
        j0, j1 = max(0, math.floor((x0 - self.x0) / self.cell_m)), min(nx, math.ceil((x1 - self.x0) / self.cell_m))
        i0, i1 = max(0, math.floor((y0 - self.y0) / self.cell_m)), min(ny, math.ceil((y1 - self.y0) / self.cell_m))
        if j1 <= j0 or i1 <= i0:
            return 0.0
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        boxes = shapely.box(self.x0 + jj * self.cell_m, self.y0 + ii * self.cell_m, self.x0 + (jj + 1) * self.cell_m, self.y0 + (ii + 1) * self.cell_m)
        return float(np.sum(shapely.area(shapely.intersection(boxes, geometry)) * self.density[ii, jj]))


def approach_area(spot: Spot) -> shapely.Geometry:
    """The floor of `spot` the fallback ranking reads occupancy under: the footprints, the docks and the clear zone."""
    import shapely

    return shapely.union_all([spot.taken, spot.clear])


@attrs.frozen(eq=False)
class Checked:
    """The full check of one layout: `check_layout`'s problems and the NAV@r filter's verdict."""

    problems: tuple[str, ...]
    navr: NavrVerdict

    @property
    def valid(self) -> bool:
        return not self.problems and self.navr.ok

    @property
    def reasons(self) -> tuple[str, ...]:
        return (*self.problems, *self.navr.reasons)


@attrs.frozen(eq=False)
class Scored:
    """One layout a search returns: its candidate `index` in `Search.candidates`, the expected robot travel per job
    (`travel_m`, under the search's task mix), the pedestrians under its elements (`occupancy`, without an occupancy
    None), its NAV@r floor loss, the `score` its search ranked it by (lower is better), and, where the exposure ranking
    scored it, its predicted conflict exposure per walker group (`exposure`, seconds per hour per member)."""

    layout: Layout
    index: int
    travel_m: float
    occupancy: float | None
    floor_loss_m2: float
    score: float
    navr: NavrVerdict
    exposure: Mapping[str, float] | None = None

    @property
    def layout_id(self) -> str:
        return self.layout.layout_id

    def record(self) -> dict[str, Any]:
        """The layout as plain data: its id, scores and each object's spot and pose."""
        return {
            "layout_id": self.layout_id,
            "score": _num(self.score),
            "travel_m": _num(self.travel_m),
            "occupancy": None if self.occupancy is None else _num(self.occupancy),
            "floor_loss_m2": _num(self.floor_loss_m2),
            **({"exposure": {group: _num(value) for group, value in self.exposure.items()}} if self.exposure is not None else {}),
            "spots": [
                {"element": s.element.name, "spot": s.id, "zone": s.zone, "piece": s.piece, "step": s.step, "objects": [{"name": o.name, "model": o.model, "x": o.x, "y": o.y, "yaw": o.yaw, "goto": list(o.goto)} for o in s.objects]}
                for s in self.layout.spots
            ],
        }


def _num(value: float) -> float | None:
    return None if not math.isfinite(value) else round(float(value), 3)


def _overlaps(a: shapely.Geometry, b: shapely.Geometry) -> bool:
    return bool(a.intersects(b)) and a.intersection(b).area > place.AREA_EPS


def _reach(spot: Spot) -> shapely.Polygon:
    """The floor `check_layout` reads around `spot` (its keep-clear floor with the generator's margin and the corridor
    window in front), as a box: a spot whose box misses another's passes `check_layout` with it."""
    import shapely

    from . import generate as gen

    lo, hi, _, front = spot.frame.extent(shapely.union_all([spot.taken, spot.clear]))
    return shapely.box(*shapely.union_all([spot.keep_clear.buffer(gen.MARGIN_M + 0.01), place._window(spot.frame, (lo, hi), front)]).bounds)


class Search:
    """The search space of one site (see the module doc): built once, then `robot_only`, `joint` and `random_valid`
    read it. `loading` pins the loading point: a `Spot`, or a level point that snaps to the nearest valid loading spot's
    goto point (within `PIN_SNAP_M` of it or of the spot's bay). `pins` pins searched elements the same way, by element name (e.g. the locker,
    whose robot-only optimum is otherwise next to the loading point). `zones` and `exclude` choose the candidate zones
    as `candidate_spots` does.
    A spot whose floor (`Spot.keep_clear`) overlaps a `keep_out` polygon is no candidate: the study's PEDS sources and
    sinks, given as polygons where the §3 rule needs zone names, and the links (stair and escalator halves) with the
    apron in front of their mouths. Spots on other levels than the loading point's are left
    out. `mix` is the task mix robot travel is weighed with. `placements` gives an element its zones, areas or pin
    (`Placement`); `zones` is the default for the others. `loading` and `pins` take precedence over the placements' pins.
    `facts` are the site's (`place.SiteFacts`): the catalog's rules that read them filter every element's spots, so the
    robot-only, joint and random picks draw from the same rule-valid spots."""

    def __init__(
        self,
        site: Site,
        *,
        loading: Spot | Point | None = None,
        pins: Mapping[str, Spot | Point] | None = None,
        zones: Sequence[str] | None = None,
        exclude: Sequence[str] = EXCLUDED_ZONES,
        placements: Mapping[str, Placement] | None = None,
        keep_out: Iterable[shapely.Geometry] = (),
        mix: TaskMix | None = None,
        path_resolution: float = PATH_RESOLUTION_M,
        base: NavrBase | None = None,
        facts: place.SiteFacts | None = None,
    ) -> None:
        import shapely

        self.site = site
        self.mix = TaskMix() if mix is None else mix
        self.path_resolution = path_resolution
        self.keep_out = tuple(keep_out)
        self.placements: dict[str, Placement] = dict(placements or {})
        unknown = sorted(set(self.placements) - {*SEARCHED, LOADING})
        if unknown:
            raise ValueError(f"placements name {unknown}, not elements of a layout ({', '.join((*SEARCHED, LOADING))})")
        blocked = shapely.union_all(self.keep_out) if self.keep_out else None
        self.facts = facts
        found = {name: place.candidate_spots(site, name, zones=self._zones(name, zones), exclude=exclude, facts=facts) for name in (*SEARCHED, LOADING)}
        if loading is None:
            loading = self.placements.get(LOADING, Placement()).pin
            if loading is None:
                raise ValueError("no loading point: pass `loading` or a placement with a pin for it")
        self.loading = loading if isinstance(loading, Spot) else self._snap(found[LOADING], loading)
        pins = {**{name: p.pin for name, p in self.placements.items() if name in SEARCHED and p.pin is not None}, **dict(pins or {})}
        unknown = sorted(set(pins) - set(SEARCHED))
        if unknown:
            raise ValueError(f"pins name {unknown}, not elements the search places ({', '.join(SEARCHED)})")
        self.pins = {name: at if isinstance(at, Spot) else self._snap(found[name], at) for name, at in pins.items()}
        # §3: no element in a PEDS source or sink, the pinned ones included: say so rather than search an empty space
        for name, spot in [(LOADING, self.loading), *self.pins.items()]:
            if blocked is not None and _overlaps(spot.keep_clear, blocked):
                raise ValueError(f"the pinned {name} ({spot.zone} piece {spot.piece} step {spot.step}, goto {spot.objects[0].goto}) overlaps a PEDS source or sink or a link of the keep-out")
        level = self.loading.level
        self.pair_checks = 0
        self.dropped: dict[str, dict[str, int]] = {}
        spots: dict[str, tuple[Spot, ...]] = {}
        for name in SEARCHED:
            pool = [self.pins[name]] if name in self.pins else found[name]
            valid = [s for s in pool if s.valid]
            within = [s for s in valid if name in self.pins or self.placements.get(name, Placement()).holds(s)]
            here = [s for s in within if s.level == level]
            free = [s for s in here if blocked is None or not _overlaps(s.keep_clear, blocked)]
            ok = [s for s in free if self._compatible(s, self.loading)]
            spots[name] = tuple(ok)
            self.dropped[name] = {"spots": len(pool), "invalid": len(pool) - len(valid), "outside_areas": len(valid) - len(within), "other_level": len(within) - len(here), "keep_out": len(here) - len(free), "loading_point": len(free) - len(ok)}
        self.spots: Mapping[str, tuple[Spot, ...]] = spots
        self._pairs = {(a, b): self._pair_matrix(a, b) for i, a in enumerate(SEARCHED) for b in SEARCHED[i + 1 :]}
        self._base = base
        self._candidates: np.ndarray | None = None
        self._paths: RobotPaths | None = None
        self._lengths: dict[str, np.ndarray] | None = None
        self._checked: dict[int, Checked] = {}
        self._people: dict[Occupancy, tuple[dict[str, list[float]], float]] = {}
        self._bay_charger: np.ndarray | None = None
        self._exposure: dict[Any, dict[str, np.ndarray]] = {}

    # --- spots and candidates --------------------------------------------------------------------------------------

    def _zones(self, name: str, default: Sequence[str] | None) -> list[str] | None:
        """The world zones element `name` may go to: its placement's zones (patterns matched against every level's zone
        names), else `default`. ValueError for a pattern no zone matches."""
        import fnmatch

        placement = self.placements.get(name)
        if placement is None or placement.zones is None:
            return None if default is None else list(default)
        names = sorted({z.name for level in self.site.world.levels.values() for z in level.zones})
        out, missing = [], []
        for pattern in placement.zones:
            hits = [z for z in names if fnmatch.fnmatchcase(z, pattern)]
            if not hits:
                missing.append(pattern)
            out += [z for z in hits if z not in out]
        if missing:
            raise ValueError(f"{name}: no zone of {self.site.key} matches {missing}")
        return out

    @staticmethod
    def _snap(spots: Sequence[Spot], at: Point) -> Spot:
        """The valid spot whose first object's goto point is nearest to `at`, within `PIN_SNAP_M` of that point or of the
        object's footprint."""
        import shapely

        valid = [s for s in spots if s.valid]
        if not valid:
            raise ValueError(f"no valid {spots[0].element.name if spots else 'element'} spot to pin")
        best = min(valid, key=lambda s: (math.dist(s.objects[0].goto, at), s.id))
        off = min(math.dist(best.objects[0].goto, at), best.objects[0].footprint.distance(shapely.Point(at)))
        if off > PIN_SNAP_M:
            raise ValueError(f"the nearest valid {best.element.name} spot ({best.zone} piece {best.piece} step {best.step}) is {off:.2f} m from {at}, over {PIN_SNAP_M} m")
        return best

    def _compatible(self, a: Spot, b: Spot) -> bool:
        """Whether `check_layout` passes the layout of `a` and `b` alone; spots whose reach misses pass without it."""
        if not _reach(a).intersects(_reach(b)):
            return True
        self.pair_checks += 1
        return not place.check_layout(self.site, Layout.of([a, b]))

    def _pair_matrix(self, first: str, second: str) -> np.ndarray:
        import numpy as np
        import shapely

        a, b = self.spots[first], self.spots[second]
        out = np.ones((len(a), len(b)), dtype=bool)
        if not a or not b:
            return out
        tree = shapely.STRtree([_reach(s) for s in b])
        for i, s in enumerate(a):
            for j in tree.query(_reach(s)):
                self.pair_checks += 1
                out[i, j] = not place.check_layout(self.site, Layout.of([s, b[int(j)]]))
        return out

    @property
    def candidates(self) -> np.ndarray:
        """Every layout of pairwise compatible spots as rows of spot indices (charger pair, bay pair, locker), in
        locker, then charger pair, then bay pair order. Valid layouts are among them."""
        import numpy as np

        if self._candidates is None:
            cb, ck, bk = self._pairs[(SEARCHED[0], SEARCHED[1])], self._pairs[(SEARCHED[0], SEARCHED[2])], self._pairs[(SEARCHED[1], SEARCHED[2])]
            total = int(sum(int((cb & ck[:, k : k + 1] & bk[:, k][None, :]).sum()) for k in range(ck.shape[1])))
            if total > MAX_CANDIDATES:
                raise ValueError(f"{total} candidate layouts, over {MAX_CANDIDATES}: narrow the zones or coarsen the step")
            rows = np.empty((total, 3), dtype=np.int32)
            at = 0
            for k in range(ck.shape[1]):
                c, b = np.nonzero(cb & ck[:, k : k + 1] & bk[:, k][None, :])
                rows[at : at + len(c)] = np.column_stack([c, b, np.full(len(c), k)])
                at += len(c)
            self._candidates = rows
        return self._candidates

    def layout(self, index: int) -> Layout:
        """Candidate `index` with the loading point, as a `Layout`."""
        c, b, k = (int(v) for v in self.candidates[index])
        return Layout.of([self.loading, self.spots[SEARCHED[0]][c], self.spots[SEARCHED[1]][b], self.spots[SEARCHED[2]][k]])

    # --- validity --------------------------------------------------------------------------------------------------

    @property
    def base(self) -> NavrBase:
        """The NAV@r base of the loading point's level, without any object of the catalog's elements."""
        if self._base is None:
            from arena_simulation_setup.metrics.navr_filter import NavrBase
            from arena_simulation_setup.tree.World import WorldView

            members = {name for element in place.CATALOG.values() for name in element.members}
            level = self.site.world.levels[self.loading.level]
            self._base = NavrBase(level, view=WorldView(self.site.root), world=self.site.world, exclude=members & {o.name for o in level.objects})
        return self._base

    def check(self, index: int) -> Checked:
        """The full check of candidate `index`: `check_layout` and the NAV@r filter, cached."""
        if index not in self._checked:
            from arena_simulation_setup.metrics.navr_filter import navr_check

            layout = self.layout(index)
            self._checked[index] = Checked(tuple(place.check_layout(self.site, layout)), navr_check(self.base, place.placed_elements(layout)))
        return self._checked[index]

    def valid(self, index: int) -> bool:
        return self.check(index).valid

    def sample(self, n: int, seed: int, stream: int = 0, exclude: Iterable[str] = (), among: np.ndarray | None = None) -> list[int]:
        """Up to `n` valid candidates drawn uniformly without replacement, as candidate indices in draw order: the
        candidates (or those in `among`, candidate indices) in a random order from (`seed`, `stream`), the first `n`
        valid ones whose layout id is not in `exclude`. Fewer when the site has fewer."""
        import numpy as np

        skip = set(exclude)
        pool = np.arange(len(self.candidates)) if among is None else np.asarray(among, dtype=np.int64)
        out: list[int] = []
        for index in pool[np.random.default_rng([int(seed), int(stream)]).permutation(len(pool))]:
            if len(out) >= n:
                break
            index = int(index)
            if self.valid(index) and self.layout(index).layout_id not in skip:
                out.append(index)
        return out

    def within_travel(self, cap_m: float) -> np.ndarray:
        """The candidate indices whose expected robot travel per job is at most `cap_m`, in index order."""
        import numpy as np

        return np.flatnonzero(self.travel() <= cap_m)

    # --- robot travel ----------------------------------------------------------------------------------------------

    @property
    def paths(self) -> RobotPaths:
        if self._paths is None:
            self._paths = RobotPaths.from_base(self.base, resolution=self.path_resolution)
        return self._paths

    @property
    def lengths(self) -> dict[str, np.ndarray]:
        """Mean path lengths (m) by spot index: `L_<element>` from the loading point to each spot's goto points, `K_<element>`
        from each locker spot (rows) to them, `LK` from the loading point to each locker spot."""
        if self._lengths is None:
            import numpy as np

            lockers = [s.objects[0].goto for s in self.spots[SEARCHED[2]]]
            sources = [self.loading.objects[0].goto, *lockers]
            members = {name: [o.goto for s in self.spots[name] for o in s.objects] for name in SEARCHED[:2]}
            targets = [p for name in SEARCHED[:2] for p in members[name]] + lockers
            d = self.paths.lengths(sources, targets) if targets else np.zeros((len(sources), 0))
            out: dict[str, np.ndarray] = {}
            at = 0
            for name in SEARCHED[:2]:
                width = len(place.CATALOG[name].members)
                block = d[:, at : at + len(members[name])].reshape(len(sources), len(self.spots[name]), width).mean(axis=2)
                out[f"L_{name}"], out[f"K_{name}"] = block[0], block[1:]
                at += len(members[name])
            out["LK"] = d[0, at:]
            self._lengths = out
        return self._lengths

    def travel(self, index: int | np.ndarray | None = None) -> np.ndarray:
        """The expected robot travel per job (m) under `mix` of the candidates `index` (default all)."""
        import numpy as np

        rows = self.candidates if index is None else self.candidates[np.atleast_1d(index)]
        c, b, k = rows[:, 0], rows[:, 1], rows[:, 2]
        d, w = self.lengths, self.mix.weights()
        ch, bay = SEARCHED[0], SEARCHED[1]
        out = d["LK"][k] + w["L_bay"] * d[f"L_{bay}"][b] + w["L_charger"] * d[f"L_{ch}"][c] + w["K_bay"] * d[f"K_{bay}"][k, b] + w["K_charger"] * d[f"K_{ch}"][k, c]
        if self.mix.top_ups_per_job > 0:
            out = out + 2.0 * self.mix.top_ups_per_job * self.bay_charger[c, b]
        return out

    @property
    def bay_charger(self) -> np.ndarray:
        """Mean path length (m) between each charger spot's goto points (rows) and each bay spot's (columns): the
        top-up round trip's leg (`TaskMix.top_ups_per_job`)."""
        if self._bay_charger is None:
            import numpy as np

            chargers = [o.goto for s in self.spots[SEARCHED[0]] for o in s.objects]
            bays = [o.goto for s in self.spots[SEARCHED[1]] for o in s.objects]
            width_c, width_b = len(place.CATALOG[SEARCHED[0]].members), len(place.CATALOG[SEARCHED[1]].members)
            if not chargers or not bays:
                self._bay_charger = np.zeros((len(self.spots[SEARCHED[0]]), len(self.spots[SEARCHED[1]])))
            else:
                d = self.paths.lengths(chargers, bays)
                self._bay_charger = d.reshape(len(self.spots[SEARCHED[0]]), width_c, len(self.spots[SEARCHED[1]]), width_b).mean(axis=(1, 3))
        return self._bay_charger

    # --- conflict exposure (the joint ranking) -----------------------------------------------------------------------

    def leg_exposure(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """The path integrals of each row of `values` (one value per path cell, shape (K, cells)) over the legs of the
        job cycle, by spot index as `lengths` (each with a leading axis of length K): `L_<element>` from the loading
        point, `K_<element>` from each locker spot (axis 1), `LK` from the loading point to each locker spot, and `BC`
        from each charger spot to each bay spot (as `bay_charger`)."""
        import numpy as np

        values = np.atleast_2d(values)
        lockers = [s.objects[0].goto for s in self.spots[SEARCHED[2]]]
        sources = [self.loading.objects[0].goto, *lockers]
        members = {name: [o.goto for s in self.spots[name] for o in s.objects] for name in SEARCHED[:2]}
        targets = [p for name in SEARCHED[:2] for p in members[name]] + lockers
        n = values.shape[0]
        _, e = self.paths.integrals(sources, targets, values) if targets else (None, np.zeros((n, len(sources), 0)))
        out: dict[str, np.ndarray] = {}
        at = 0
        for name in SEARCHED[:2]:
            width = len(place.CATALOG[name].members)
            block = e[:, :, at : at + len(members[name])].reshape(n, len(sources), len(self.spots[name]), width).mean(axis=3)
            out[f"L_{name}"], out[f"K_{name}"] = block[:, 0], block[:, 1:]
            at += len(members[name])
        out["LK"] = e[:, 0, at:]
        if members[SEARCHED[0]] and members[SEARCHED[1]]:
            _, bc = self.paths.integrals(members[SEARCHED[0]], members[SEARCHED[1]], values)
            wc, wb = len(place.CATALOG[SEARCHED[0]].members), len(place.CATALOG[SEARCHED[1]].members)
            out["BC"] = bc.reshape(n, len(self.spots[SEARCHED[0]]), wc, len(self.spots[SEARCHED[1]]), wb).mean(axis=(2, 4))
        else:
            out["BC"] = np.zeros((n, len(self.spots[SEARCHED[0]]), len(self.spots[SEARCHED[1]])))
        return out

    def job_exposure(self, groups: Mapping[str, Occupancy], band_m: float = CONFLICT_BAND_M, speed_mps: float = ROBOT_SPEED_MPS, hold_s: tuple[float, float] = HOLD_S) -> dict[str, np.ndarray]:
        """Per walker group and candidate, the walker-seconds of the group within `band_m` of a robot per job: the
        job-cycle legs weighed as in `travel` (the top-up round trips included), driven at `speed_mps`, plus the holds
        at the loading point and the locker (`hold_s`). Walkers per path cell come from each group's occupancy (the
        site as is). Cached per groups, band, speed and holds."""
        import numpy as np

        key = (tuple((name, id(occ)) for name, occ in groups.items()), float(band_m), float(speed_mps), tuple(hold_s), self.mix)
        if key in self._exposure:
            return self._exposure[key]
        names = list(groups)
        bands = {name: groups[name].band(band_m) for name in names}
        legs = self.leg_exposure(np.stack([groups[name].sample(bands[name], self.paths.centres) for name in names]))
        rows = self.candidates
        c, b, k = rows[:, 0], rows[:, 1], rows[:, 2]
        w = self.mix.weights()
        ch, bay = SEARCHED[0], SEARCHED[1]
        lockers = np.array([s.objects[0].goto for s in self.spots[SEARCHED[2]]], dtype=float).reshape(-1, 2)
        out: dict[str, np.ndarray] = {}
        for i, name in enumerate(names):
            driven = legs["LK"][i][k] + w["L_bay"] * legs[f"L_{bay}"][i][b] + w["L_charger"] * legs[f"L_{ch}"][i][c] + w["K_bay"] * legs[f"K_{bay}"][i][k, b] + w["K_charger"] * legs[f"K_{ch}"][i][k, c]
            if self.mix.top_ups_per_job > 0:
                driven = driven + 2.0 * self.mix.top_ups_per_job * legs["BC"][i][c, b]
            at_loading = float(groups[name].sample(bands[name], np.asarray([self.loading.objects[0].goto]))[0])
            at_locker = groups[name].sample(bands[name], lockers) if len(lockers) else np.zeros(0)
            out[name] = driven / speed_mps + hold_s[0] * at_loading + hold_s[1] * at_locker[k]
        self._exposure[key] = out
        return out

    def element_people(self, occupancy: Occupancy) -> np.ndarray:
        """Per candidate, the pedestrians under its elements and the loading point (`occupancy` of every candidate)."""
        import numpy as np

        per, loading = self._spot_people(occupancy)
        rows = self.candidates
        return loading + np.asarray(per[SEARCHED[0]])[rows[:, 0]] + np.asarray(per[SEARCHED[1]])[rows[:, 1]] + np.asarray(per[SEARCHED[2]])[rows[:, 2]]

    def conflict(self, groups: Mapping[str, Occupancy], rate_per_h: float, band_m: float = CONFLICT_BAND_M, speed_mps: float = ROBOT_SPEED_MPS, hold_s: tuple[float, float] = HOLD_S) -> dict[str, np.ndarray]:
        """The joint ranking's predicted conflict exposure (§5) per walker group and candidate: the seconds per hour an
        average member of the group spends within `band_m` of a driving robot (`job_exposure` times `rate_per_h` jobs
        per hour) or on an element's footprint, dock or approach zone (`element_people` times 3600), over the group's
        walkers present. Lower is better. The NAV@r floor loss is added per checked layout (`conflict_floor`)."""
        jobs = self.job_exposure(groups, band_m, speed_mps, hold_s)
        out = {}
        for name, occ in groups.items():
            present = occ.present
            if present <= 0:
                raise ValueError(f"group {name!r} has no walker time in its occupancy")
            out[name] = (rate_per_h * jobs[name] + 3600.0 * self.element_people(occ)) / present
        return out

    @staticmethod
    def conflict_floor(groups: Mapping[str, Occupancy], floor_loss_m2: float) -> dict[str, float]:
        """Per group, the seconds per hour a member would spend on the floor a layout cuts off (the NAV@r floor loss at
        the group's mean walked density, over its walkers present), the floor-loss term of `conflict`."""
        return {name: 3600.0 * occ.mean_density * floor_loss_m2 / occ.present for name, occ in groups.items()}

    def spot_centres(self) -> dict[str, np.ndarray]:
        """Per searched element, each spot's centre (the mean of its objects' goto points), by spot index."""
        import numpy as np

        return {name: np.array([np.mean([o.goto for o in s.objects], axis=0) for s in self.spots[name]], dtype=float).reshape(-1, 2) for name in SEARCHED}

    # --- scoring ---------------------------------------------------------------------------------------------------

    def occupancy(self, occupancy: Occupancy, index: int) -> float:
        """The pedestrians under candidate `index`'s elements and the loading point (`approach_area`)."""
        per, loading = self._spot_people(occupancy)
        c, b, k = (int(v) for v in self.candidates[index])
        return loading + per[SEARCHED[0]][c] + per[SEARCHED[1]][b] + per[SEARCHED[2]][k]

    def _spot_people(self, occupancy: Occupancy) -> tuple[dict[str, list[float]], float]:
        """The pedestrians under each spot's approach area by element, and under the loading point's; cached."""
        if occupancy not in self._people:
            self._people[occupancy] = ({name: [occupancy.people(approach_area(s)) for s in spots] for name, spots in self.spots.items()}, occupancy.people(approach_area(self.loading)))
        return self._people[occupancy]

    def scored(self, index: int, score: float, occupancy: Occupancy | None = None, exposure: Mapping[str, float] | None = None) -> Scored:
        checked = self.check(index)
        return Scored(self.layout(index), index, float(self.travel(index)[0]), None if occupancy is None else self.occupancy(occupancy, index), checked.navr.floor_loss_m2, float(score), checked.navr, None if exposure is None else dict(exposure))

    def counts(self) -> dict[str, Any]:
        """The search space's size: spots per element (and why others dropped), pair checks run, candidates."""
        return {"spots": {name: len(s) for name, s in self.spots.items()}, "dropped": self.dropped, "pair_checks": self.pair_checks, "candidates": len(self.candidates)}

    def corridor(self) -> tuple[np.ndarray, np.ndarray]:
        """Per candidate, the narrowest and the widest remaining corridor (`Spot.corridor_m`) at its searched elements."""
        import numpy as np

        rows = self.candidates
        widths = np.column_stack([np.array([s.corridor_m for s in self.spots[name]], dtype=float)[rows[:, i]] if len(rows) else np.zeros(0) for i, name in enumerate(SEARCHED)])
        return widths.min(axis=1), widths.max(axis=1)


def _search(site: Site | Search, kwargs: Mapping[str, Any]) -> Search:
    if isinstance(site, Search):
        if kwargs:
            raise TypeError(f"a built Search takes no search arguments, got {sorted(kwargs)}")
        return site
    return Search(site, **kwargs)


def robot_only(site: Site | Search, **kwargs: Any) -> Scored:
    """The robot-only pick (§5): the valid layout with the shortest expected robot travel per job for the task mix,
    over every valid layout (the candidates by travel, the first that passes the full check). `kwargs` build the
    `Search` when `site` is a site. ValueError without a valid layout."""
    import numpy as np

    search = _search(site, kwargs)
    travel = search.travel()
    for index in np.argsort(travel, kind="stable"):
        index = int(index)
        if not math.isfinite(travel[index]):
            break
        if search.valid(index):
            return search.scored(index, travel[index])
    raise ValueError("no valid layout with a finite robot travel")


def travel_cap(robot: Scored, slack: float = TRAVEL_SLACK) -> float:
    """The joint search's travel pre-filter (§5, §6): at most `slack` times the robot-only pick's travel per job."""
    return slack * robot.travel_m


def joint(site: Site | Search, occupancy: Occupancy, *, n: int = JOINT_SAMPLES, top: int = JOINT_TOP, seed: int = 0, floor_weight: float | None = None, max_travel_m: float | None = None, **kwargs: Any) -> list[Scored]:
    """The joint search's ranking with the fallback (§5): `n` valid layouts drawn at random (`Search.sample`) among
    those whose expected robot travel per job is at most `max_travel_m` (the travel pre-filter, `travel_cap`; None: all),
    each scored by the pedestrians under its elements' footprints, docks and approach zones (`occupancy`, from the
    site-as-is reference runs) plus its NAV@r floor loss times `floor_weight` (default the occupancy's mean density on
    walked floor, so both count pedestrians); the `top` lowest, best first, ties by layout id."""
    search = _search(site, kwargs)
    weight = occupancy.mean_density if floor_weight is None else floor_weight
    among = None if max_travel_m is None else search.within_travel(max_travel_m)
    ranked = [search.scored(i, ranking_score(search, occupancy, i, weight), occupancy) for i in search.sample(n, seed, _JOINT, among=among)]
    return sorted(ranked, key=lambda s: (s.score, s.layout_id))[:top]


def ranking_score(search: Search, occupancy: Occupancy, index: int, floor_weight: float | None = None) -> float:
    """The fallback ranking's score of candidate `index` (§5, lower is better): pedestrians under its elements plus its
    NAV@r floor loss times `floor_weight` (default the occupancy's mean density on walked floor)."""
    weight = occupancy.mean_density if floor_weight is None else floor_weight
    return search.occupancy(occupancy, index) + weight * search.check(index).navr.floor_loss_m2


def distinct(search: Search, ranked: Sequence[int], top: int, distinct_m: float = DISTINCT_M) -> list[int]:
    """The first `top` of `ranked` (candidate indices, best first) that are distinct layouts: one enters only if one of
    its elements sits at least `distinct_m` from that element's spot in every layout already in (spot centres)."""
    import numpy as np

    centres = search.spot_centres()
    rows = search.candidates
    picked: list[int] = []
    for index in ranked:
        if len(picked) >= top:
            break
        here = [centres[name][rows[index, i]] for i, name in enumerate(SEARCHED)]
        if all(max(float(np.linalg.norm(here[i] - centres[name][rows[other, i]])) for i, name in enumerate(SEARCHED)) >= distinct_m for other in picked):
            picked.append(int(index))
    return picked


def joint_exposure(
    site: Site | Search,
    groups: Mapping[str, Occupancy],
    *,
    rate_per_h: float,
    top: int = JOINT_TOP,
    band_m: float = CONFLICT_BAND_M,
    distinct_m: float = DISTINCT_M,
    max_travel_m: float | None = None,
    speed_mps: float = ROBOT_SPEED_MPS,
    hold_s: tuple[float, float] = HOLD_S,
    max_checks: int = MAX_RANKED_CHECKS,
    **kwargs: Any,
) -> list[Scored]:
    """The joint search's ranking by predicted conflict exposure (§5): every candidate whose expected robot travel per
    job is at most `max_travel_m` (the travel pre-filter, `travel_cap`; None: all), scored per walker group by
    `Search.conflict` plus the floor-loss term of its full check (`Search.conflict_floor`); the ranking score is the
    largest group's value (the worst-off group, as C_worst). Returns the `top` best valid layouts that are distinct
    (`distinct`), best first, ties by robot travel then layout id; each carries its per-group exposure. Candidates are
    checked in order of their score without the floor term, a lower bound, until no unchecked one can enter the top."""
    import numpy as np

    search = _search(site, kwargs)
    scores = search.conflict(groups, rate_per_h, band_m, speed_mps, hold_s)
    names = list(scores)
    worst = np.max(np.stack([scores[name] for name in names]), axis=0)
    travel = search.travel()
    among = np.arange(len(search.candidates)) if max_travel_m is None else search.within_travel(max_travel_m)
    order = among[np.lexsort((travel[among], worst[among]))]
    centres = search.spot_centres()
    rows = search.candidates
    full: dict[int, tuple[float, dict[str, float]]] = {}

    def near(index: int, other: int) -> bool:
        return all(float(np.linalg.norm(centres[name][rows[index, i]] - centres[name][rows[other, i]])) < distinct_m for i, name in enumerate(SEARCHED))

    def key(index: int) -> tuple[float, float, str]:
        return (full[index][0], float(travel[index]), search.layout(index).layout_id)

    picked: list[int] = []
    for checks, index in enumerate(order):
        index = int(index)
        if not math.isfinite(worst[index]) or checks >= max_checks:
            break
        # the score without the floor term is a lower bound: nothing later can enter a full top
        if len(picked) >= top and worst[index] >= max(full[i][0] for i in picked) - 1e-12:
            break
        # a layout near one already in can only enter by displacing it, which takes a floor loss on that one
        if any(near(index, other) and full[other][0] <= worst[index] for other in picked):
            continue
        if not search.valid(index):
            continue
        floor = Search.conflict_floor(groups, search.check(index).navr.floor_loss_m2)
        exposure = {name: float(scores[name][index] + floor[name]) for name in names}
        full[index] = (max(exposure.values()), exposure)
        picked = distinct(search, sorted([*picked, index], key=key), top, distinct_m)
    return [search.scored(i, full[i][0], exposure=full[i][1]) for i in picked]


def exposure_of(search: Search, groups: Mapping[str, Occupancy], index: int, *, rate_per_h: float, band_m: float = CONFLICT_BAND_M, speed_mps: float = ROBOT_SPEED_MPS, hold_s: tuple[float, float] = HOLD_S) -> dict[str, float]:
    """The predicted conflict exposure per group of candidate `index` (`joint_exposure`'s, the floor term included)."""
    scores = search.conflict(groups, rate_per_h, band_m, speed_mps, hold_s)
    floor = Search.conflict_floor(groups, search.check(index).navr.floor_loss_m2)
    return {name: float(scores[name][index] + floor[name]) for name in scores}


def guideline(site: Site | Search, **kwargs: Any) -> Scored:
    """The guideline placement (the practice baseline): on the same rule-valid spots and with the same pinned loading
    point as the searches, by walking distance on the floor a wheelchair reaches (`place.walking_paths`, what the
    catalog's distance rules measure):

    - G2 the waiting bays at the bay spot nearest the loading point (LTA and URA, Last-Mile Delivery Guide for
      Developments, 2026, p. 10: delivery waiting bays close to delivery routes, where couriers park);
    - G3 the parcel locker at the locker spot nearest a recipient entrance (same guide, p. 9: lockers near entrances);
    - G4 the charger dock at the charger spot nearest the waiting bays, one robot station (Restroom Association
      Singapore, Designing Future-Ready Restrooms, 2025, §2.4 and §4.1: docking zones adjacent to the facilities they
      serve, at power points, out of the way; the catalog's clearance rules keep docks off the walkways).

    In that order each element takes the best spot by its rule that is compatible with the ones placed before it, and
    the layout must pass the full check, else the next spot of the last element, then of the one before (lexicographic
    in the rule order). Scored by travel. ValueError without recipient entrances in the site facts or a valid layout."""
    import numpy as np

    search = _search(site, kwargs)
    level = search.loading.level
    recipients = [e for e in (search.facts.recipient_entrances if search.facts is not None else ()) if e.level in (None, level)]
    if not recipients:
        raise ValueError("the guideline placement needs the site facts' recipient entrances (the locker's rule)")
    walk = place.walking_paths(search.site, level)
    centres = search.spot_centres()
    loading = search.loading.objects[0].goto
    bay_order = np.argsort(walk.lengths([loading], centres[SEARCHED[1]])[0], kind="stable") if len(centres[SEARCHED[1]]) else np.zeros(0, dtype=int)
    lockers = np.array([s.objects[0].goto for s in search.spots[SEARCHED[2]]], dtype=float).reshape(-1, 2)
    locker_order = np.argsort(walk.lengths([e.anchor for e in recipients], lockers).min(axis=0), kind="stable") if len(lockers) else np.zeros(0, dtype=int)
    rows = search.candidates
    for b in bay_order:
        charger_order = np.argsort(walk.lengths([tuple(centres[SEARCHED[1]][b])], centres[SEARCHED[0]])[0], kind="stable")
        rank = np.empty(len(charger_order), dtype=np.int64)
        rank[charger_order] = np.arange(len(charger_order))
        for k in locker_order:
            matches = np.flatnonzero((rows[:, 1] == b) & (rows[:, 2] == k))
            for index in matches[np.argsort(rank[rows[matches, 0]], kind="stable")]:
                if search.valid(int(index)):
                    return search.scored(int(index), float(search.travel(int(index))[0]))
    raise ValueError("no valid guideline layout")


def premise_pair(site: Site | Search, **kwargs: Any) -> tuple[Scored, Scored]:
    """The premise check's two most different layouts (§10): the valid layout with every searched element at the
    narrowest section (the smallest widest remaining corridor over its elements) and the one with every element at the
    widest (the largest narrowest corridor); ties go to the shorter robot travel, then the layout id. Scored by travel.
    ValueError without a valid layout."""
    import numpy as np

    search = _search(site, kwargs)
    narrowest, widest = search.corridor()
    travel = search.travel()
    def first(order: np.ndarray) -> Scored:
        for index in order:
            index = int(index)
            if math.isfinite(travel[index]) and search.valid(index):
                return search.scored(index, float(travel[index]))
        raise ValueError("no valid layout with a finite robot travel")

    narrow = first(np.lexsort((travel, widest)))
    wide = first(np.lexsort((travel, -narrowest)))
    return narrow, wide


def random_valid(site: Site | Search, k: int = RANDOM_LAYOUTS, seed: int = 0, *, exclude: Iterable[str] = (), occupancy: Occupancy | None = None, **kwargs: Any) -> list[Scored]:
    """`k` random valid layouts (the §5 control), in draw order, none with a layout id in `exclude`; scored by travel."""
    search = _search(site, kwargs)
    return [search.scored(i, float(search.travel(i)[0]), occupancy) for i in search.sample(k, seed, _RANDOM, exclude)]


__all__ = [
    "CONFLICT_BAND_M",
    "DISTINCT_M",
    "GROUPS",
    "HOMES",
    "JOINT_SAMPLES",
    "JOINT_TOP",
    "LOADING",
    "PATH_RESOLUTION_M",
    "RANDOM_LAYOUTS",
    "SEARCHED",
    "TRAVEL_SLACK",
    "TWIN_LOADING_XY",
    "Checked",
    "Occupancy",
    "Placement",
    "RobotPaths",
    "Scored",
    "Search",
    "TaskMix",
    "approach_area",
    "distinct",
    "exposure_of",
    "guideline",
    "joint",
    "joint_exposure",
    "premise_pair",
    "random_valid",
    "ranking_score",
    "robot_only",
    "travel_cap",
]
