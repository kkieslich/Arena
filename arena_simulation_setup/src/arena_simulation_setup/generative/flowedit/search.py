"""INFRA layout searches (study design §4, §5): the valid layouts of the catalog's elements on a site, the robot-only
pick (the shortest expected robot travel for the §8 task stream, as in Kundu and Saha's charger placement), the joint
search's fallback ranking (pedestrian occupancy under each element's footprint and approach zone, from the site-as-is
reference runs, plus the NAV@r floor loss) and random valid layouts.

A layout places the charger pair, the bay pair and the locker on valid candidate spots (`place.candidate_spots`), with
the loading point pinned to one spot (route end A on the twin, §8). `Search` holds one site's search space:

- the valid spots of each element, less those that touch a keep-out polygon (a PEDS source or sink) or clash with the
  loading point;
- which spot pairs `place.check_layout` passes. Its checks look at most a corridor window away from a spot, so a pair
  further apart always passes. A layout of pairwise compatible spots is a candidate; picks get the full check;
- a NAV@r base of the site without the elements (`metrics.navr_filter.NavrBase`). A layout is valid when
  `check_layout` passes and `navr_check` accepts it;
- robot path lengths between the dock goto points on the jackal's floor at r 0.267 m (`RobotPaths`).

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
#: How far a pinned element (the loading point) may lie from the goto point of the valid spot it snaps to
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
#: Seed streams, so the joint sample and the random control are independent draws for the same seed
_JOINT, _RANDOM = 1, 2


@attrs.frozen
class TaskMix:
    """The §8 task stream the robot-only search weighs robot travel with.

    Jobs come at `rate_per_h`, a share `pickup_share` of them pickups (locker to loading point), the rest deliveries
    (loading point to locker). The fleet mode sends a free robot home, so a job runs home, load end, unload end, home.
    Homes are the 2 bay slots and the 2 chargers (`HOMES`), one robot each, so jobs start and end at each home equally
    often. After `charges_per_job` of the jobs the robot drives to a charger instead of home and keeps it as its home
    (the fleet mode's swap). Each robot charges `charges_per_robot` times per episode of `episode_s` (§8 sizes the battery
    for about once), so `charges_per_job` is robots x charges over the jobs of an episode, at most 1. Chained jobs (a
    robot that takes the next job where the last ended) are not modelled: the home legs are an upper bound."""

    rate_per_h: float = 12.0
    pickup_share: float = 0.3
    robots: int = len(HOMES)
    episode_s: float = 1500.0
    charges_per_robot: float = 1.0

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

    @property
    def mean_density(self) -> float:
        """The mean density over the cells that hold any pedestrian time: the walked floor's."""
        import numpy as np

        walked = self.density[self.density > 0]
        return float(walked.mean()) if len(walked) else 0.0

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
    None), its NAV@r floor loss, and the `score` its search ranked it by (lower is better)."""

    layout: Layout
    index: int
    travel_m: float
    occupancy: float | None
    floor_loss_m2: float
    score: float
    navr: NavrVerdict

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
    goto point (within `PIN_SNAP_M`). `pins` pins searched elements the same way, by element name (e.g. the locker,
    whose robot-only optimum is otherwise next to the loading point). `zones` and `exclude` choose the candidate zones
    as `candidate_spots` does.
    A spot whose floor (`Spot.keep_clear`) overlaps a `keep_out` polygon is no candidate: the study's PEDS sources and
    sinks, given as polygons where the §3 rule needs zone names. Spots on other levels than the loading point's are left
    out. `mix` is the task mix robot travel is weighed with."""

    def __init__(
        self,
        site: Site,
        *,
        loading: Spot | Point,
        pins: Mapping[str, Spot | Point] | None = None,
        zones: Sequence[str] | None = None,
        exclude: Sequence[str] = EXCLUDED_ZONES,
        keep_out: Iterable[shapely.Geometry] = (),
        mix: TaskMix | None = None,
        path_resolution: float = PATH_RESOLUTION_M,
        base: NavrBase | None = None,
    ) -> None:
        import shapely

        self.site = site
        self.mix = TaskMix() if mix is None else mix
        self.path_resolution = path_resolution
        self.keep_out = tuple(keep_out)
        blocked = shapely.union_all(self.keep_out) if self.keep_out else None
        found = {name: place.candidate_spots(site, name, zones=zones, exclude=exclude) for name in (*SEARCHED, LOADING)}
        self.loading = loading if isinstance(loading, Spot) else self._snap(found[LOADING], loading)
        unknown = sorted(set(pins or {}) - set(SEARCHED))
        if unknown:
            raise ValueError(f"pins name {unknown}, not elements the search places ({', '.join(SEARCHED)})")
        self.pins = {name: at if isinstance(at, Spot) else self._snap(found[name], at) for name, at in (pins or {}).items()}
        level = self.loading.level
        self.pair_checks = 0
        self.dropped: dict[str, dict[str, int]] = {}
        spots: dict[str, tuple[Spot, ...]] = {}
        for name in SEARCHED:
            pool = [self.pins[name]] if name in self.pins else found[name]
            valid = [s for s in pool if s.valid]
            here = [s for s in valid if s.level == level]
            free = [s for s in here if blocked is None or not _overlaps(s.keep_clear, blocked)]
            ok = [s for s in free if self._compatible(s, self.loading)]
            spots[name] = tuple(ok)
            self.dropped[name] = {"spots": len(pool), "invalid": len(pool) - len(valid), "other_level": len(valid) - len(here), "keep_out": len(here) - len(free), "loading_point": len(free) - len(ok)}
        self.spots: Mapping[str, tuple[Spot, ...]] = spots
        self._pairs = {(a, b): self._pair_matrix(a, b) for i, a in enumerate(SEARCHED) for b in SEARCHED[i + 1 :]}
        self._base = base
        self._candidates: np.ndarray | None = None
        self._paths: RobotPaths | None = None
        self._lengths: dict[str, np.ndarray] | None = None
        self._checked: dict[int, Checked] = {}
        self._people: dict[Occupancy, tuple[dict[str, list[float]], float]] = {}

    # --- spots and candidates --------------------------------------------------------------------------------------

    @staticmethod
    def _snap(spots: Sequence[Spot], at: Point) -> Spot:
        """The valid spot whose first object's goto point is nearest to `at`, within `PIN_SNAP_M`."""
        valid = [s for s in spots if s.valid]
        if not valid:
            raise ValueError(f"no valid {spots[0].element.name if spots else 'element'} spot to pin")
        best = min(valid, key=lambda s: (math.dist(s.objects[0].goto, at), s.id))
        off = math.dist(best.objects[0].goto, at)
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

    def sample(self, n: int, seed: int, stream: int = 0, exclude: Iterable[str] = ()) -> list[int]:
        """Up to `n` valid candidates drawn uniformly without replacement, as candidate indices in draw order: the
        candidates in a random order from (`seed`, `stream`), the first `n` valid ones whose layout id is not in
        `exclude`. Fewer when the site has fewer."""
        import numpy as np

        skip = set(exclude)
        out: list[int] = []
        for index in np.random.default_rng([int(seed), int(stream)]).permutation(len(self.candidates)):
            if len(out) >= n:
                break
            index = int(index)
            if self.valid(index) and self.layout(index).layout_id not in skip:
                out.append(index)
        return out

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
        return d["LK"][k] + w["L_bay"] * d[f"L_{bay}"][b] + w["L_charger"] * d[f"L_{ch}"][c] + w["K_bay"] * d[f"K_{bay}"][k, b] + w["K_charger"] * d[f"K_{ch}"][k, c]

    # --- scoring ---------------------------------------------------------------------------------------------------

    def occupancy(self, occupancy: Occupancy, index: int) -> float:
        """The pedestrians under candidate `index`'s elements and the loading point (`approach_area`)."""
        if occupancy not in self._people:
            self._people[occupancy] = ({name: [occupancy.people(approach_area(s)) for s in spots] for name, spots in self.spots.items()}, occupancy.people(approach_area(self.loading)))
        per, loading = self._people[occupancy]
        c, b, k = (int(v) for v in self.candidates[index])
        return loading + per[SEARCHED[0]][c] + per[SEARCHED[1]][b] + per[SEARCHED[2]][k]

    def scored(self, index: int, score: float, occupancy: Occupancy | None = None) -> Scored:
        checked = self.check(index)
        return Scored(self.layout(index), index, float(self.travel(index)[0]), None if occupancy is None else self.occupancy(occupancy, index), checked.navr.floor_loss_m2, float(score), checked.navr)

    def counts(self) -> dict[str, Any]:
        """The search space's size: spots per element (and why others dropped), pair checks run, candidates."""
        return {"spots": {name: len(s) for name, s in self.spots.items()}, "dropped": self.dropped, "pair_checks": self.pair_checks, "candidates": len(self.candidates)}


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


def joint(site: Site | Search, occupancy: Occupancy, *, n: int = JOINT_SAMPLES, top: int = JOINT_TOP, seed: int = 0, floor_weight: float | None = None, **kwargs: Any) -> list[Scored]:
    """The joint search's ranking with the fallback (§5): `n` valid layouts drawn at random (`Search.sample`), each
    scored by the pedestrians under its elements' footprints, docks and approach zones (`occupancy`, from the site-as-is
    reference runs) plus its NAV@r floor loss times `floor_weight` (default the occupancy's mean density on walked floor,
    so both count pedestrians); the `top` lowest, best first, ties by layout id."""
    search = _search(site, kwargs)
    weight = occupancy.mean_density if floor_weight is None else floor_weight
    ranked = [search.scored(i, search.occupancy(occupancy, i) + weight * search.check(i).navr.floor_loss_m2, occupancy) for i in search.sample(n, seed, _JOINT)]
    return sorted(ranked, key=lambda s: (s.score, s.layout_id))[:top]


def random_valid(site: Site | Search, k: int = RANDOM_LAYOUTS, seed: int = 0, *, exclude: Iterable[str] = (), occupancy: Occupancy | None = None, **kwargs: Any) -> list[Scored]:
    """`k` random valid layouts (the §5 control), in draw order, none with a layout id in `exclude`; scored by travel."""
    search = _search(site, kwargs)
    return [search.scored(i, float(search.travel(i)[0]), occupancy) for i in search.sample(k, seed, _RANDOM, exclude)]


__all__ = [
    "HOMES",
    "JOINT_SAMPLES",
    "JOINT_TOP",
    "LOADING",
    "PATH_RESOLUTION_M",
    "RANDOM_LAYOUTS",
    "SEARCHED",
    "TWIN_LOADING_XY",
    "Checked",
    "Occupancy",
    "RobotPaths",
    "Scored",
    "Search",
    "TaskMix",
    "approach_area",
    "joint",
    "random_valid",
    "robot_only",
]
