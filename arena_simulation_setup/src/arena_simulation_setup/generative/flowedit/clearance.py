"""INFRA's clearance rule (study design §3): the width every element leaves the crowd beside it, against every wall,
pillar and fixture of the site and every other element, by the site's directional base flow.

The walkway rule (`walkway`) measures the necks the crowd's shortest routes cross. A stream along a wall carries no
shortest route, so a gap between a parked robot and a pillar there passed at any width (0.77 m beside a bay pair on the
twin, about 300 walkers an hour through it). This rule measures every gap, routes or not; both rules apply.

**Floor.** What a layout takes from the crowd (`Piece`):

- every element footprint (an element that `blocks_floor`), permanent;
- the occupied slot of every robot home, a charger's or a waiting bay's dock (`PARKED`), where a robot parks most of a
  run: the dock grown to at least `SLOT_W_M` x `SLOT_L_M` about its centre (`slot_polygon`), permanent;
- the dock of a locker and of the loading point, where a robot stands only while it delivers or loads: intermittent.

Pieces of the layout within `JOIN_M` of each other are one. The site's obstacles (`Obstacles`) are its walls joined
where they touch (each pillar ring one piece), and its fixtures.

**Gaps.** A gap is the shortest distance between a layout piece and a part of an obstacle (each wall segment, each
fixture) or another layout piece. Measured to each wall segment, not to the nearest point of the whole connected wall,
the wall an element stands against cannot hide the far side of its corridor when both belong to one wall. Of the gaps
between a layout piece and one obstacle piece, the counted one that falls shortest of its requirement is the pair's.
Where two faces run alongside each other at the gap's width, the gap sits at the middle of that stretch. A gap counts
when

- it is at least `SEALED_M` wide: a narrower one is sealed flush, the piece joins that obstacle and nobody walks
  through it;
- it is under `MEASURE_M` (`TWO_WAY_M`): a wider one meets every requirement;
- it lies between two elements, or an element and the site: a dock and its own element's footprint are one design
  (the robot at its locker), no gap; a gap between two elements is measured once;
- its segment runs through no third piece (a dock's distance to the wall behind its own locker is none);
- no other part of the same obstacle piece lies nearer its end on the layout piece, in sight: then it is no neck,
  only the facet beside one that comes nearer (a pillar's next facet beside the one the piece is sealed to);
- it is a through-passage: probes `PROBE_M` either side of its middle, along its normal, reach free floor (on the floor,
  `PROBE_CLEAR_M` from every piece) on both sides. A recess between two wall-standing elements, or the side of a
  locker, is no passage. Nor is a gap that leads nowhere: with the gap plugged, no floor for a body meets it from one
  side (`SIDE_M` from every piece: a wedge beside a seal); or no other route a wheelchair takes (`ROUTE_HALF_M` from
  every piece) joins its two sides. The latter holds with a base flow only where no base walker crossed the gap's line
  before the layout (walkers who did cross it are walkers the layout sends into the gap); without one, only for a
  recess, a pocket (within `POCKET_WINDOW_M`) whose floor lies within `RECESS_M` of the gap, and every deeper pocket
  counts (conservative).

**Requirement.** A counted gap keeps

- `TWO_WAY_M` (1.8 m, two wheelchair users pass, BCA 2019 4.2.1.2) where the site's base flow across it is two-way: its
  minor direction carries at least `TWO_WAY_SHARE` of the walkers crossing it, and at least `FLOW_MIN_PED_H` walkers an
  hour cross it;
- `ONE_WAY_M` (0.9 m, one wheelchair, 2010 ADA 403.5.1) where it is one-way, or fewer than `FLOW_MIN_PED_H` walkers an
  hour cross it;
- `ONE_WAY_M` beside a robot in an intermittent dock, whatever the flow: a lane taking turns while it stands there.

The base flow (`BaseFlow`) is the site as is, without the layout: arena_evaluation's flow map of its PEDS runs
(pedestrian-seconds per 0.5 m cell and heading bin), and where given the same runs' walker tracks (`Tracks`).

- Direction (two-way or one-way): the map's split, the moving walkers' time in the cells within `BAND_M` of the gap,
  each heading bin's projected on the gap's normal.
- Walkers an hour across the gap: the walkers whose tracks cross its segment; without tracks, the map's flux at
  `WALK_SPEED_MPS` times `MAP_CALIBRATION` (the tracks' count per the map's estimate on the twin; it agrees with the
  tracks on the `FLOW_MIN_PED_H` threshold for about 9 in 10 segments, so give tracks where they exist).

**Without a base flow** every gap is two-way (`TWO_WAY_M`), the conservative default, and each verdict says so
(`CONSERVATIVE`). The rule never says unknown.

The geometry needs no ROS and no site: `check_record` takes INFRA-FlowNet's layout records and site geometry (the
generator's schema), the placement rules take spots (`spot_pieces`, `place.clearance_gaps`). Both use `gaps`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from .walkway import LINK_SPEED_MPS, ONE_WAY_M, TWO_WAY_M

if TYPE_CHECKING:
    import numpy as np
    from shapely.geometry.base import BaseGeometry

Point = tuple[float, float]
Segment = tuple[Point, Point]

#: A gap under this is sealed flush: the piece joins that wall, pillar or element, and nobody walks through [m]
SEALED_M = 0.15
#: Gaps this wide and wider meet every requirement and are not measured [m]
MEASURE_M = TWO_WAY_M
#: A gap is two-way only where at least this many walkers an hour cross it [ped/h]
FLOW_MIN_PED_H = 10.0
#: ... and its minor direction carries at least this share of them
TWO_WAY_SHARE = 0.10
#: The occupied slot of a robot home is at least this wide and this long: a parked jackal with its margins [m]
SLOT_W_M, SLOT_L_M = 1.0, 1.2
#: Layout pieces this close are one; walls are this thick [m]
JOIN_M = 0.05
#: Another part of the same obstacle piece is nearer a gap's end when it is nearer by this: facets meeting at a vertex,
#: their thick outlines' round ends, differ by less [m]
NEARER_M = JOIN_M
#: A gap is a through-passage where free floor lies at one of these distances either side of its middle [m]
PROBE_M = (0.6, 1.0, 1.5)
#: ... at least this far from every piece [m]
PROBE_CLEAR_M = 0.25
#: A route needs twice this clear (a wheelchair, 0.65 m): the dead-end test's other routes keep it from every piece [m]
ROUTE_HALF_M = 0.325
#: The floor that meets a gap keeps this from every piece (a body's half width less a margin) [m]
SIDE_M = 0.1
#: The dead-end test reads the floor within this of a gap; a side that reaches beyond is open floor [m]
POCKET_WINDOW_M = 15.0
#: A pocket whose floor lies within this of the gap is a recess (a nook beside an element), no passage [m]
RECESS_M = 2.5
#: The base flow across a gap is read from the cells whose centres lie within this of its segment [m]
BAND_M = 0.5
#: A heading bin crosses a gap where its direction's cosine with the gap's normal is above this
CROSS_COS = 0.2
#: The walking speed the base flow's walker time crosses a gap at: HumanSim's adult desired speed (1.1 m/s), an upper
#: bound for a crowd's moving walkers, so the walkers an hour are not underestimated [m/s]
WALK_SPEED_MPS = LINK_SPEED_MPS
#: Walkers whose tracks cross a segment per the map's flux estimate (at `WALK_SPEED_MPS`), on the v3 twin's site-as-is
#: run (seed 299, 2700 s): 3000 random segments 0.6-1.8 m long on walked floor, those with >= 10 walkers/h by tracks:
#: median 0.57, quartiles 0.38-0.74 (at necks, where walkers slow down, 0.25-0.31)
MAP_CALIBRATION = 0.57
#: Consecutive track samples further apart than this belong to no step [s]
TRACK_STEP_S = 1.5
#: The elements whose docks hold a parked robot most of a run: the fleet's homes and chargers
PARKED = ("charger_pair", "bay_pair")
#: Heading bins of the base flow, as arena_evaluation bins a velocity (`forecast.maps`): 8 headings and a standing bin
HEADINGS = 8
BINS = HEADINGS + 1
#: What a verdict without a base flow says
CONSERVATIVE = "no base flow: two-way assumed (conservative)"


# --- the base flow ------------------------------------------------------------------------------------------------------


@attrs.frozen(eq=False)
class Tracks:
    """Walker tracks of the site as is, as steps between consecutive samples (`TRACK_STEP_S` apart at most): from
    (`px`, `py`) to (`qx`, `qy`) by walker `walker`, sorted by `xmin` (the step's smaller x), over `hours` of recording."""

    px: np.ndarray
    py: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    walker: np.ndarray
    xmin: np.ndarray
    hours: float

    @classmethod
    def from_samples(cls, key: Any, t: Any, x: Any, y: Any, hours: float) -> Tracks:  # noqa: ANN401 - arrays
        """The tracks of samples (walker `key`, time `t` [s], position `x`, `y`), in any order, over `hours`."""
        import numpy as np

        if hours <= 0:
            raise ValueError(f"tracks over {hours} h: give the recorded hours")
        _, k = np.unique(np.asarray(key), return_inverse=True)
        t, x, y = (np.asarray(v, dtype=float) for v in (t, x, y))
        order = np.lexsort((t, k))
        k, t, x, y = k[order], t[order], x[order], y[order]
        step = (k[1:] == k[:-1]) & (np.diff(t) < TRACK_STEP_S)
        px, py, qx, qy, w = x[:-1][step], y[:-1][step], x[1:][step], y[1:][step], k[:-1][step]
        xmin = np.minimum(px, qx)
        by_x = np.argsort(xmin, kind="stable")
        return cls(px[by_x], py[by_x], qx[by_x], qy[by_x], w[by_x], xmin[by_x], float(hours))

    @classmethod
    def from_parquet(cls, paths: Sequence[Path], hours: float) -> Tracks:
        """The tracks of arena_evaluation's walker track files (`ped_tracks.v2`: key, t, x, y), each file its own
        walkers, over `hours` in all."""
        import numpy as np
        import pandas as pd

        frames = []
        for n, path in enumerate(paths):
            f = pd.read_parquet(path, columns=["key", "t", "x", "y"])
            f["key"] = f"{n}:" + f["key"].astype(str)
            frames.append(f)
        if not frames:
            raise ValueError("no track files")
        f = pd.concat(frames, ignore_index=True)
        return cls.from_samples(f["key"].to_numpy(), f["t"].to_numpy(), f["x"].to_numpy(dtype=np.float64), f["y"].to_numpy(dtype=np.float64), hours)

    def crossing(self, segment: Segment) -> tuple[float, float, float]:
        """The walkers an hour whose tracks cross `segment` (a -> b): (forth, back, either way), forth along its left
        normal; a walker crossing it several times counts once."""
        import numpy as np

        (ax, ay), (bx, by) = segment
        span = max(TRACK_STEP_S * 3.0, 3.0)  # no step is longer: walkers stay under 2 m/s
        lo, hi = np.searchsorted(self.xmin, min(ax, bx) - span), np.searchsorted(self.xmin, max(ax, bx), side="right")
        px, py, qx, qy, w = self.px[lo:hi], self.py[lo:hi], self.qx[lo:hi], self.qy[lo:hi], self.walker[lo:hi]
        dx, dy = bx - ax, by - ay
        s1, s2 = dx * (py - ay) - dy * (px - ax), dx * (qy - ay) - dy * (qx - ax)
        ex, ey = qx - px, qy - py
        s3, s4 = ex * (ay - py) - ey * (ax - px), ex * (by - py) - ey * (bx - px)
        hit = (s1 * s2 < 0) & (s3 * s4 < 0)
        forth, back = len(np.unique(w[hit & (s1 < 0)])), len(np.unique(w[hit & (s1 > 0)]))
        return forth / self.hours, back / self.hours, len(np.unique(w[hit])) / self.hours


@attrs.frozen(eq=False)
class BaseFlow:
    """The site's directional base flow: pedestrian-seconds per square cell and heading bin, `time[i, j, b]` for the cell
    from (`x0` + j `cell`, `y0` + i `cell`) and bin b (`HEADINGS` headings from -pi, then standing), over `seconds` of
    recording in all (runs x episode length); and, where given, the same runs' walker `tracks`."""

    x0: float
    y0: float
    cell: float
    seconds: float
    time: np.ndarray
    tracks: Tracks | None = None

    def __attrs_post_init__(self) -> None:
        if self.seconds <= 0:
            raise ValueError(f"a base flow over {self.seconds} s: give the recorded seconds (runs x episode length)")
        if self.time.ndim != 3 or self.time.shape[2] != BINS:
            raise ValueError(f"a base flow is (ny, nx, {BINS}) pedestrian-seconds, got shape {self.time.shape}")

    @classmethod
    def from_cells(cls, x: Any, y: Any, heading_rad: Any, ped_seconds: Any, cell_m: float, seconds: float) -> BaseFlow:  # noqa: ANN401 - arrays
        """The base flow of flow-map rows (cell centres `x`, `y`, `cell_m` wide; heading bin centre `heading_rad`, NaN or
        None standing), `ped_seconds` each, over `seconds` of recording. A row listed twice (per run) adds up."""
        import numpy as np

        xs, ys, w = (np.asarray(v, dtype=float) for v in (x, y, ped_seconds))
        if not len(xs):
            raise ValueError("no flow cells: no base flow")
        h = np.asarray([math.nan if v is None else float(v) for v in np.asarray(heading_rad, dtype=object)], dtype=float)
        x0, y0 = float(xs.min()) - cell_m / 2.0, float(ys.min()) - cell_m / 2.0
        j = np.floor((xs - x0) / cell_m).astype(np.int64)
        i = np.floor((ys - y0) / cell_m).astype(np.int64)
        b = np.where(np.isnan(h), HEADINGS, np.floor((np.nan_to_num(h) + math.pi) / (2 * math.pi) * HEADINGS).astype(np.int64) % HEADINGS)
        time = np.zeros((int(i.max()) + 1, int(j.max()) + 1, BINS))
        np.add.at(time, (i, j, b), w)
        return cls(x0, y0, float(cell_m), float(seconds), time)

    @classmethod
    def from_flow_cells(cls, path: Path, world: str, episode_s: float, planner: str | None = None, tracks: Sequence[Path] = ()) -> BaseFlow:
        """The base flow of `world`'s runs in arena_evaluation's flow cells (`flow_cells.parquet`, or a benchmark run
        directory holding it): every run (stage and planner) of `world`, each `episode_s` long, warm-up included.
        `planner` keeps one contestant's runs. `world` is the site as is, never a layout world. `tracks`: the same runs'
        walker track files (`ped_tracks.v2`), all of them."""
        import pandas as pd

        path = Path(path)
        if path.is_dir():
            path = path / "flow_cells.parquet"
        cells = pd.read_parquet(path)
        if "flow_world" in cells:
            cells = cells[cells["flow_world"] == world]
        if planner is not None:
            cells = cells[cells["planner"] == planner]
        if cells.empty:
            raise ValueError(f"{path} has no flow cells of world {world!r}" + (f" and planner {planner!r}" if planner else ""))
        sizes = sorted(set(cells["cell_m"]))
        if len(sizes) != 1:
            raise ValueError(f"{path}: flow cells of {world!r} come in sizes {sizes}")
        keys = [k for k in ("planner", "stage") if k in cells]
        runs = len(cells[keys].drop_duplicates()) if keys else 1
        flow = cls.from_cells(cells["x"], cells["y"], cells["heading_rad"], cells["ped_seconds"], sizes[0], runs * episode_s)
        return attrs.evolve(flow, tracks=Tracks.from_parquet(list(tracks), runs * episode_s / 3600.0)) if tracks else flow

    @classmethod
    def from_bins_first(cls, dirmap: Any, x0: float, y0: float, seconds: float, cell: float = 0.5) -> BaseFlow:  # noqa: ANN401 - an array
        """The base flow of a (BINS, ny, nx) map (INFRA-FlowNet's layout)."""
        import numpy as np

        return cls(float(x0), float(y0), float(cell), float(seconds), np.moveaxis(np.asarray(dirmap, dtype=float), 0, -1))

    def walkers(self, segment: Segment) -> tuple[float, str]:
        """The walkers an hour crossing `segment` either way, and what counted them: the tracks where given, else the
        map (`crossing`)."""
        if self.tracks is not None:
            return self.tracks.crossing(segment)[2], "tracks"
        return sum(self.crossing(segment)), "map"

    def crossing(self, segment: Segment) -> tuple[float, float]:
        """The map's walkers an hour crossing the gap `segment` (a -> b) each way: (forth, back), forth along its left
        normal. The moving walkers' time in the cells whose centres lie within `BAND_M` of it, each heading bin's
        projected on the normal (bins within `CROSS_COS` of along the gap count for neither), as a flux through a band
        `2 BAND_M` thick at `WALK_SPEED_MPS`, times `MAP_CALIBRATION`."""
        import numpy as np
        import shapely

        (ax, ay), (bx, by) = segment
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-9:
            return 0.0, 0.0
        nx_, ny_ = -(by - ay) / length, (bx - ax) / length
        band = shapely.LineString([(ax, ay), (bx, by)]).buffer(BAND_M)
        xa, ya, xb, yb = band.bounds
        rows, cols = self.time.shape[:2]
        i0, i1 = max(0, int((ya - self.y0) / self.cell)), min(rows, int((yb - self.y0) / self.cell) + 1)
        j0, j1 = max(0, int((xa - self.x0) / self.cell)), min(cols, int((xb - self.x0) / self.cell) + 1)
        if i0 >= i1 or j0 >= j1:
            return 0.0, 0.0
        jj, ii = np.meshgrid(np.arange(j0, j1), np.arange(i0, i1))
        inside = shapely.contains_xy(band, self.x0 + (jj + 0.5) * self.cell, self.y0 + (ii + 0.5) * self.cell)
        if not inside.any():
            return 0.0, 0.0
        per_bin = self.time[ii[inside], jj[inside], :HEADINGS].sum(axis=0)
        theta = -math.pi + (np.arange(HEADINGS) + 0.5) * 2 * math.pi / HEADINGS
        c = np.cos(theta) * nx_ + np.sin(theta) * ny_
        to_ph = WALK_SPEED_MPS / (2.0 * BAND_M) / self.seconds * 3600.0 * MAP_CALIBRATION
        forth = float(np.sum(per_bin[c > CROSS_COS] * c[c > CROSS_COS])) * to_ph
        back = float(np.sum(per_bin[c < -CROSS_COS] * -c[c < -CROSS_COS])) * to_ph
        return forth, back


def requirement(segment: Segment, base_flow: BaseFlow | None, docked: bool = False) -> tuple[float, str, float | None]:
    """(required width, why, walkers an hour across it) of a gap (see the module doc); the walkers None without a base
    flow or beside a docked robot."""
    if docked:
        return ONE_WAY_M, "beside a docked robot (intermittent): a lane taking turns", None
    if base_flow is None:
        return TWO_WAY_M, CONSERVATIVE, None
    forth, back = base_flow.crossing(segment)
    walkers, source = base_flow.walkers(segment)
    if forth + back <= 0.0:
        return ONE_WAY_M, f"one-way: no base walker crosses it ({walkers:.1f} walkers/h by {source})", walkers
    minor = min(forth, back) / (forth + back)
    if minor < TWO_WAY_SHARE:
        return ONE_WAY_M, f"one-way: minor direction {minor:.0%}, {walkers:.1f} walkers/h by {source}", walkers
    if walkers < FLOW_MIN_PED_H:
        return ONE_WAY_M, f"one-way: minor direction {minor:.0%} but {walkers:.1f} walkers/h by {source}, under {FLOW_MIN_PED_H:g}", walkers
    return TWO_WAY_M, f"two-way: minor direction {minor:.0%}, {walkers:.1f} walkers/h by {source}", walkers


# --- pieces and obstacles -----------------------------------------------------------------------------------------------


@attrs.frozen(eq=False)
class Piece:
    """Floor a layout takes from the crowd: `name`, its polygon, and whether a robot stands there only at times
    (`intermittent`: a locker's or the loading point's dock)."""

    name: str
    geometry: BaseGeometry
    intermittent: bool = False


@attrs.frozen
class Gap:
    """A counted gap (see the module doc): `what` it lies between, its `width`, the `required` width and `why`, its
    `segment` (piece end first), whether it lies beside an intermittent dock (`docked`) and the base walkers an hour
    crossing it (None without a base flow or beside a docked robot)."""

    what: str
    width: float
    required: float
    why: str
    segment: Segment
    docked: bool = False
    walkers_ph: float | None = None

    @property
    def ok(self) -> bool:
        return self.width >= self.required - 1e-9

    def describe(self) -> str:
        (ax, ay), (bx, by) = self.segment
        return f"{self.width:.2f} m gap {self.what} at ({(ax + bx) / 2:.1f}, {(ay + by) / 2:.1f}), needs {self.required:.2f} m ({self.why})"

    def record(self) -> dict[str, Any]:
        """The gap as INFRA-FlowNet's guard reports a problem."""
        out: dict[str, Any] = {"what": self.what, "gap_m": round(self.width, 3), "required_m": self.required, "why": self.why, "segment": [[round(v, 3) for v in p] for p in self.segment], "docked": self.docked}
        if self.walkers_ph is not None:
            out["walkers_ph"] = round(self.walkers_ph, 1)
        return out


@attrs.frozen(eq=False)
class Obstacles:
    """The site's obstacles of one level: connected `pieces` (name, polygon: walls joined where they touch, each pillar
    ring one piece; each fixture), the `parts` gaps are measured to (piece index, polygon: each wall segment, each
    fixture), and its `floor` (the zones)."""

    pieces: tuple[tuple[str, BaseGeometry], ...]
    parts: tuple[tuple[int, BaseGeometry], ...]
    floor: BaseGeometry
    _tree: Any = attrs.field(default=None, init=False)

    @classmethod
    def of(cls, segments: Iterable[Segment], fixtures: Mapping[str, BaseGeometry], floor: BaseGeometry, *, close: bool = True) -> Obstacles:
        """Walls (`segments`, `JOIN_M` thick) and fixture polygons; the floor closed over gaps under 0.2 m between its
        zones (`close`)."""
        import shapely

        lines = [shapely.LineString(s) for s in segments if tuple(s[0]) != tuple(s[1])]
        thick = [w.buffer(JOIN_M / 2.0) for w in lines]
        walls = shapely.union_all(thick) if thick else shapely.Polygon()
        comps = [g for g in getattr(walls, "geoms", [walls]) if not g.is_empty]
        pieces = [(f"wall_piece_{k}", c) for k, c in enumerate(comps)]
        tree = shapely.STRtree(comps) if comps else None
        parts: list[tuple[int, BaseGeometry]] = []
        for w in thick:
            hit = [int(k) for k in tree.query(w.representative_point(), predicate="intersects")] if tree is not None else []
            parts.append((hit[0] if hit else int(tree.nearest(w)), w))  # type: ignore[union-attr]
        for name, g in sorted(fixtures.items()):
            if not g.is_empty and g.area > 0:
                parts.append((len(pieces), g))
                pieces.append((f"site:{name}", g))
        closed = floor.buffer(0.1, join_style="mitre").buffer(-0.1, join_style="mitre") if close and not floor.is_empty else floor
        shapely.prepare(closed)
        return cls(tuple(pieces), tuple(parts), closed)

    @property
    def tree(self) -> Any:  # noqa: ANN401 - a shapely STRtree
        if self._tree is None:
            import shapely

            object.__setattr__(self, "_tree", shapely.STRtree([g for _, g in self.parts]))
        return self._tree

    def near(self, geometry: BaseGeometry, distance: float) -> list[int]:
        """The parts within `distance` of `geometry`, by index."""
        if not self.parts:
            return []
        return sorted(int(k) for k in self.tree.query(geometry, predicate="dwithin", distance=distance))


def slot_polygon(dock: BaseGeometry | Sequence[Sequence[float]]) -> BaseGeometry:
    """The occupied slot of a robot home: its dock, grown to at least `SLOT_W_M` x `SLOT_L_M` about its centre along
    its own axes (the dock itself when it is that big)."""
    import numpy as np
    import shapely

    p = dock if hasattr(dock, "geom_type") else shapely.Polygon(dock)
    r = shapely.minimum_rotated_rectangle(p)
    c = list(r.exterior.coords)[:4]
    e1, e2 = np.subtract(c[1], c[0]), np.subtract(c[2], c[1])
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    if l1 < 1e-9 or l2 < 1e-9:
        return p
    short, long_ = (e1, e2) if l1 <= l2 else (e2, e1)
    ls, ll = min(l1, l2), max(l1, l2)
    ws, wl = max(ls, SLOT_W_M), max(ll, SLOT_L_M)
    if ws == ls and wl == ll:
        return p
    u, v = short / ls, long_ / ll
    cx, cy = r.centroid.x, r.centroid.y
    return shapely.Polygon([(cx + a * ws / 2 * u[0] + b * wl / 2 * v[0], cy + a * ws / 2 * u[1] + b * wl / 2 * v[1]) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def spot_pieces(spots: Iterable[Any]) -> list[Piece]:
    """The floor `spots` (place.Spot) take: footprints of elements that block floor; at the fleet's homes and chargers
    (`PARKED`) each dock's occupied slot, permanent; at a locker and the loading point each dock, intermittent."""
    out = []
    for s in spots:
        parked = s.element.name in PARKED
        for o in s.objects:
            name = f"{s.element.name}:{o.name}"
            if s.element.blocks_floor:
                out.append(Piece(f"{name}#footprint", o.footprint))
            out.append(Piece(f"{name}#slot", slot_polygon(o.dock)) if parked else Piece(f"{name}#dock", o.dock, intermittent=True))
    return out


def record_pieces(layout: Mapping[str, Any]) -> list[Piece]:
    """The floor of a layout record (INFRA-FlowNet's layouts.json: elements, each with objects holding `footprint`,
    `dock` and `blocks_floor`), as `spot_pieces` takes it. A bay pair's and the loading point's footprint is no floor
    (floor markings); an object's `blocks_floor` false drops it too."""
    import shapely

    out = []
    for el in layout["elements"]:
        element = str(el["element"])
        parked = element in PARKED
        marking = element == "bay_pair" or element.startswith("loading")
        for o in el["objects"]:
            name = f"{element}:{o['name']}"
            if not marking and o.get("blocks_floor", True):
                out.append(Piece(f"{name}#footprint", shapely.Polygon(o["footprint"])))
            out.append(Piece(f"{name}#slot", slot_polygon(o["dock"])) if parked else Piece(f"{name}#dock", shapely.Polygon(o["dock"]), intermittent=True))
    return out


def merge(pieces: Sequence[Piece], join: float = JOIN_M) -> list[Piece]:
    """Permanent pieces within `join` of each other merged into one (names joined with '+'), in order."""
    import shapely

    groups: list[tuple[list[str], BaseGeometry]] = []
    for p in pieces:
        hit = [k for k, (_, g) in enumerate(groups) if g.distance(p.geometry) <= join]
        names, geo = [p.name], p.geometry
        for k in sorted(hit, reverse=True):
            n2, g = groups.pop(k)
            names, geo = n2 + names, shapely.union_all([g, geo])
        groups.append((names, geo))
    return [Piece("+".join(n), g) for n, g in groups]


# --- gaps ---------------------------------------------------------------------------------------------------------------


def _segment(a: BaseGeometry, b: BaseGeometry) -> Segment:
    """The gap between `a` and `b` as a segment, `a`'s end first. Where `a` runs alongside `b` at the gap's width (two
    parallel faces), at the middle of that stretch: the through test then reads the channel's two ends, not a mouth."""
    from shapely.ops import nearest_points

    pa, pb = nearest_points(a, b)
    width = pa.distance(pb)
    near = a.boundary.intersection(b.buffer(width + 2e-3))
    runs = [g for g in getattr(near, "geoms", [near]) if g.geom_type == "LineString" and g.length > 0.05]
    if runs:
        m = max(runs, key=lambda g: g.length).interpolate(0.5, normalized=True)
        q = nearest_points(m, b)[1]
        if m.distance(q) <= width + 2e-3:
            pa, pb = m, q
    return (float(pa.x), float(pa.y)), (float(pb.x), float(pb.y))


def gaps(layout: Sequence[Piece], site: Obstacles, base_flow: BaseFlow | None = None, *, all_gaps: bool = False, only: Iterable[str] | None = None) -> list[Gap]:
    """The counted gaps of the floor `layout` takes on `site` (see the module doc) that are narrower than they must be,
    narrowest first: between each layout piece and each other piece (a wall piece, a fixture, another element), the
    counted gap that falls shortest of its requirement. With `all_gaps`, that gap for every such pair (the dead-end
    test, the costly one, runs only for gaps narrower than they must be). `only` (piece names) keeps the gaps of the
    merged pieces that hold one of them (the others are still obstacles)."""
    import shapely
    from shapely.ops import substring

    mine = merge([p for p in layout if not p.intermittent])
    docks = [p for p in layout if p.intermittent]
    wanted = None if only is None else set(only)

    def free(x: float, y: float) -> bool:
        q = shapely.Point(x, y)
        if not site.floor.contains(q) or site.near(q, PROBE_CLEAR_M - 1e-9):
            return False
        return all(q.distance(p.geometry) >= PROBE_CLEAR_M for p in mine)

    def through(seg: Segment) -> bool:
        (ax, ay), (bx, by) = seg
        length = math.hypot(bx - ax, by - ay)
        if length < 1e-9:
            return False
        mx, my = (ax + bx) / 2, (ay + by) / 2
        nx_, ny_ = -(by - ay) / length, (bx - ax) / length
        return all(any(free(mx + s * d * nx_, my + s * d * ny_) for d in PROBE_M) for s in (1, -1))

    def dead_end(seg: Segment, ends: Sequence[BaseGeometry]) -> bool:
        """Whether the gap `seg` between `ends` leads nowhere (see the module doc)."""
        (ax, ay), (bx, by) = seg
        length = math.hypot(bx - ax, by - ay)
        mx, my, nx_, ny_ = (ax + bx) / 2, (ay + by) / 2, -(by - ay) / length, (bx - ax) / length
        ux, uy = (bx - ax) / length, (by - ay) / length
        box = shapely.box(mx - POCKET_WINDOW_M, my - POCKET_WINDOW_M, mx + POCKET_WINDOW_M, my + POCKET_WINDOW_M)
        blocks = shapely.union_all([site.parts[k][1] for k in site.near(box, 0.0)] + [p.geometry for p in mine if p.geometry.intersects(box)] + list(ends))
        floor_here = shapely.clip_by_rect(site.floor, *box.bounds)
        line = shapely.LineString(seg)

        def free(margin: float) -> list[BaseGeometry]:
            # the floor a body `margin` from every piece reaches, the gap plugged
            out = floor_here.difference(shapely.union_all([blocks.buffer(margin), line.buffer(margin + 0.02)]))
            return [c for c in getattr(out, "geoms", [out]) if not c.is_empty and c.area > 0]

        bodies, routes = free(SIDE_M), free(ROUTE_HALF_M)
        reach, half = SIDE_M + 0.12, length / 2
        side_bodies, side_routes = [], []
        for side in (1, -1):
            # the floor that meets the plugged gap from this side: just beyond the plug, between the gap's ends
            far = (mx + side * reach * nx_, my + side * reach * ny_)
            rect = shapely.Polygon([(mx - half * ux, my - half * uy), (mx + half * ux, my + half * uy), (far[0] + half * ux, far[1] + half * uy), (far[0] - half * ux, far[1] - half * uy)])
            here = [c for c in bodies if c.intersects(rect)]
            side_bodies.append(here)
            reached = shapely.union_all(here) if here else None
            side_routes.append({k for k, c in enumerate(routes) if reached is not None and c.intersects(reached)})
        if not side_bodies[0] or not side_bodies[1]:
            return True  # no floor for a body on one side (a wedge beside a seal)
        if base_flow is not None and base_flow.walkers(seg)[0] > 0.0:
            return False  # walkers crossed its line before the layout: the layout sends them into the gap
        if side_routes[0] & side_routes[1]:
            return False  # another route joins its sides: a passage
        edge = box.exterior
        open_ = [any(routes[k].distance(edge) < 1e-6 for k in side) for side in side_routes]
        if all(open_):
            return False  # open floor on both sides beyond the window: not shown to be a dead end
        if base_flow is not None:
            return True  # a pocket no base walker went into
        p = open_.index(False) if any(open_) else min((0, 1), key=lambda s_: sum(c.area for c in side_bodies[s_]))
        bodies_p = shapely.union_all(side_bodies[p])
        rings = [g.exterior for g in getattr(bodies_p, "geoms", [bodies_p]) if g.geom_type == "Polygon"]
        # without a base flow: a nook within reach of the gap is a recess; a deeper pocket counts
        return max(line.distance(shapely.Point(xy)) for r in rings for xy in r.coords) <= RECESS_M

    def elements(name: str) -> set[str]:
        return {n.split(":")[0] for n in name.split("+") if ":" in n}

    def nearer_part(geometry: BaseGeometry, seg: Segment, width: float, same: Sequence[BaseGeometry]) -> bool:
        """Whether another part of the same obstacle piece (`same`) lies nearer than `width` (by `NEARER_M`) to the
        gap's end on the layout piece, in sight of it (not behind the piece): the gap is then no neck of the free floor, only a part
        beside one that comes nearer (a pillar's next facet beside the one it is sealed to)."""
        from shapely.ops import nearest_points

        a = shapely.Point(seg[0])
        for g in same:
            if g.distance(a) >= width - NEARER_M:
                continue
            q = nearest_points(a, g)[1]
            sight = shapely.LineString([a, q])
            if sight.length < 1e-9 or sight.intersection(geometry).length < 1e-6:
                return True
        return False

    out: list[Gap] = []

    def scan(piece: Piece, others: list[Piece], docks: bool, measured: set[int]) -> None:
        """The gaps of `piece` to the site's parts and to `others` (the layout's permanent pieces), those at `measured`
        indices only (each gap between two layout pieces is measured once)."""
        geometry = piece.geometry
        mine_of = elements(piece.name)
        # (name of the other piece, the part a gap is measured to, whether it counts): every other piece blocks a gap's
        # segment; a gap lies between `piece` and another element or the site (a dock and its own element's footprint
        # are one design, the robot at its locker)
        near = [(o.name, o.geometry, j in measured and not (mine_of & elements(o.name))) for j, o in enumerate(others) if o.geometry is not geometry]
        site_parts = site.near(geometry, MEASURE_M)
        near += [(site.pieces[site.parts[k][0]][0], site.parts[k][1], True) for k in site_parts]
        group = [None] * (len(near) - len(site_parts)) + [site.parts[k][0] for k in site_parts]
        best: dict[str, Gap] = {}
        for k, (oname, other, counted) in enumerate(near):
            if not counted:
                continue
            width = float(geometry.distance(other))
            if width >= MEASURE_M or width < SEALED_M:
                continue  # wide enough, or sealed flush (the piece joins that obstacle there)
            seg = _segment(geometry, other)
            line = shapely.LineString(seg)
            if line.length > 4 * JOIN_M:
                inner = substring(line, 2 * JOIN_M, line.length - 2 * JOIN_M)
                if any(inner.intersects(g) for j, (_, g, _) in enumerate(near) if j != k and g is not other):
                    continue  # the segment runs through a third piece: no gap between these two
            if group[k] is not None and nearer_part(geometry, seg, width, [g for j, (_, g, _) in enumerate(near) if j != k and group[j] == group[k]]):
                continue  # the same wall comes nearer beside it (a pillar's next facet): no neck there
            if not through(seg):
                continue  # a recess, no passage
            required, why, walkers = requirement(seg, base_flow, docks)
            gap = Gap(f"{piece.name} - {oname}", round(width, 3), required, why, seg, docks, walkers)
            if not gap.ok and dead_end(seg, (geometry, other)):
                continue  # a dead-end pocket nobody walks in, no passage
            if oname not in best or gap.width - gap.required < best[oname].width - best[oname].required:
                best[oname] = gap
        out.extend(g for g in best.values() if all_gaps or not g.ok)

    def keep(piece: Piece) -> bool:
        return wanted is None or any(n in wanted for n in piece.name.split("+"))

    for k, piece in enumerate(mine):
        # a gap between two permanent pieces once: from the earlier one, or from the one `only` keeps
        if keep(piece):
            scan(piece, mine, False, {j for j, o in enumerate(mine) if j != k and (j > k or not keep(o))})
    for dock in docks:
        if keep(dock):
            scan(dock, mine, True, set(range(len(mine))))
    return sorted(out, key=lambda g: (g.width - g.required, g.what))


# --- INFRA-FlowNet's schema ---------------------------------------------------------------------------------------------


def site_obstacles(geom: Mapping[str, Any]) -> Obstacles:
    """The obstacles of a site geometry record (INFRA-FlowNet's meta/geom json: `walls` as segments, pillar rings among
    them; `obstacles` as polygons by name; `floor` as polygons)."""
    import shapely

    floor = shapely.union_all([shapely.Polygon(p) for p in geom["floor"]])
    fixtures = {str(k): shapely.Polygon(p) for k, p in (geom.get("obstacles") or {}).items()}
    # the record's floor is the generator's own: no closing over zone gaps
    return Obstacles.of([tuple(map(tuple, w)) for w in geom.get("walls", []) if len(w) >= 2], fixtures, floor, close=False)


def check_record(layout: Mapping[str, Any], geom: Mapping[str, Any], base_flow: BaseFlow | None = None) -> list[dict[str, Any]]:
    """Every gap a layout record leaves narrower than it must be on a site geometry record (see `site_obstacles`), as
    INFRA-FlowNet's guard reports them: [] when the layout passes."""
    return [g.record() for g in gaps(record_pieces(layout), site_obstacles(geom), base_flow)]


__all__ = [
    "BAND_M",
    "CONSERVATIVE",
    "FLOW_MIN_PED_H",
    "MEASURE_M",
    "PARKED",
    "SEALED_M",
    "SLOT_L_M",
    "SLOT_W_M",
    "TWO_WAY_SHARE",
    "WALK_SPEED_MPS",
    "BaseFlow",
    "Gap",
    "Obstacles",
    "Piece",
    "check_record",
    "gaps",
    "merge",
    "record_pieces",
    "requirement",
    "site_obstacles",
    "slot_polygon",
    "spot_pieces",
]
