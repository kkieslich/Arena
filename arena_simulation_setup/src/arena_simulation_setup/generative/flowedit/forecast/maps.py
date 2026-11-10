"""Flow maps as arena_evaluation's flow measures hold them: pedestrian-seconds per 0.5 m cell over a world's rooms, 8 headings
plus a standing bin, and the Jensen-Shannon divergence (bits) between two of them.

The grid, the heading bins, `jsd` and `pooled` mirror `arena_evaluation.processing.flow` (`grid.py`, `maps.py`), which
imports this package, not the other way round; `test_generative_flowedit_forecast.py` checks they agree.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

if TYPE_CHECKING:
    import numpy as np
    import shapely

CELL_M = 0.5
HEADINGS = 8
#: the last bin: speed under arena_evaluation's STANDING_MPS (0.2 m/s)
STANDING = HEADINGS
BINS = HEADINGS + 1

Polygon = Sequence[tuple[float, float]]


def zone_polygons(root: Path) -> list[list[tuple[float, float]]]:
    """Every zone outline of the world at `root` in the world frame, each moved by its level's origin, as the flow grid takes them."""
    from arena_simulation_setup.tree.World.view import WorldView

    view = WorldView(Path(root).resolve())
    world = view.load(validate=False)
    origins = view.level_origins() or {}
    polygons = []
    for level_id, level in world.levels.items():
        ox, oy = origins.get(level_id, (0.0, 0.0))
        polygons.extend([(c.x + ox, c.y + oy) for c in zone.polygon] for zone in level.zones)
    return polygons


def heading_bin(theta: Any) -> Any:  # noqa: ANN401 - float or array
    """The heading bin of direction `theta` (rad, world frame), as arena_evaluation bins a velocity."""
    import numpy as np

    return np.floor((np.asarray(theta) + math.pi) / (2 * math.pi) * HEADINGS).astype(np.int64) % HEADINGS


def bin_centre(b: int) -> float:
    """The direction (rad) at the middle of heading bin `b`."""
    return -math.pi + (b + 0.5) * 2 * math.pi / HEADINGS


@attrs.frozen
class FlowGrid:
    """Square cells from (x0, y0), row i covering y0 + i * cell, each with HEADINGS heading bins and a standing bin."""

    x0: float
    y0: float
    nx: int
    ny: int
    cell: float = CELL_M
    rooms: shapely.Geometry | None = attrs.field(default=None, eq=False, repr=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.ny, self.nx, BINS)

    def empty(self) -> np.ndarray:
        import numpy as np

        return np.zeros(self.shape)

    @classmethod
    def from_polygons(cls, polygons: Iterable[Polygon], cell: float = CELL_M) -> FlowGrid:
        """The grid over the union of `polygons` (world frame); ValueError without one of at least three corners."""
        import shapely

        shapes = [shapely.Polygon(p) for p in polygons if len(p) >= 3]
        shapes = [s if s.is_valid else s.buffer(0) for s in shapes]
        rooms = shapely.union_all(shapes) if shapes else None
        if rooms is None or rooms.is_empty:
            raise ValueError("no zone to lay a flow grid over")
        x0, y0, x1, y1 = rooms.bounds
        shapely.prepare(rooms)
        return cls(x0=float(x0), y0=float(y0), nx=max(1, math.ceil((x1 - x0) / cell)), ny=max(1, math.ceil((y1 - y0) / cell)), cell=cell, rooms=rooms)

    @classmethod
    def from_world(cls, root: Path, cell: float = CELL_M) -> FlowGrid:
        """The grid over every zone of the world at `root`: the grid of its flow maps when it is a base world."""
        return cls.from_polygons(zone_polygons(root), cell)

    def centres(self, sub: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """The world-frame centres (x, y) of the grid's cells split `sub` times per side, as (ny * sub, nx * sub) arrays."""
        import numpy as np

        step = self.cell / sub
        xs = self.x0 + (np.arange(self.nx * sub) + 0.5) * step
        ys = self.y0 + (np.arange(self.ny * sub) + 0.5) * step
        return np.meshgrid(xs, ys)

    def index(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: ANN401 - floats or arrays
        """Row, column and whether the point lies on the grid, elementwise."""
        import numpy as np

        i = np.floor((np.asarray(y, dtype=float) - self.y0) / self.cell).astype(np.int64)
        j = np.floor((np.asarray(x, dtype=float) - self.x0) / self.cell).astype(np.int64)
        return i, j, (i >= 0) & (i < self.ny) & (j >= 0) & (j < self.nx)

    def in_rooms(self, x: Any, y: Any) -> np.ndarray:  # noqa: ANN401 - floats or arrays
        """Whether each point lies inside the rooms, as arena_evaluation keeps a sample."""
        import numpy as np
        import shapely

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return np.ones(np.broadcast(x, y).shape, bool) if self.rooms is None else shapely.contains_xy(self.rooms, x, y)

    def room_cells(self) -> np.ndarray:
        """(ny, nx) mask of the cells that overlap the rooms: the only cells a flow map holds time in."""
        import numpy as np
        import shapely

        if self.rooms is None:
            return np.ones((self.ny, self.nx), bool)
        i, j = np.mgrid[0 : self.ny, 0 : self.nx]
        boxes = shapely.box(self.x0 + j * self.cell, self.y0 + i * self.cell, self.x0 + (j + 1) * self.cell, self.y0 + (i + 1) * self.cell)
        return shapely.intersects(self.rooms, boxes)

    def dense(self, x: Any, y: Any, heading_rad: Any, ped_seconds: Any) -> np.ndarray:  # noqa: ANN401 - columns
        """A map from cell rows (arena_evaluation's CELL_SCHEMA columns), a NaN or None heading the standing bin."""
        import numpy as np

        h = self.empty()
        theta = np.array([np.nan if v is None else v for v in heading_rad], dtype=float) if not isinstance(heading_rad, np.ndarray) else heading_rad.astype(float)
        i, j, ok = self.index(x, y)
        b = np.where(np.isnan(theta), STANDING, heading_bin(np.nan_to_num(theta)))
        np.add.at(h, (i[ok], j[ok], b[ok]), np.asarray(ped_seconds, dtype=float)[ok])
        return h

    def rows(self, counts: np.ndarray) -> dict[str, list[Any]]:
        """The bins of a map that hold time as CELL_SCHEMA columns: cell centre, cell size, heading bin centre (None standing), time."""
        import numpy as np

        i, j, b = np.nonzero(counts)
        return {
            "x": [float(v) for v in self.x0 + (j + 0.5) * self.cell],
            "y": [float(v) for v in self.y0 + (i + 0.5) * self.cell],
            "cell_m": [self.cell] * len(b),
            "heading_rad": [None if k == STANDING else bin_centre(int(k)) for k in b],
            "ped_seconds": [float(v) for v in counts[i, j, b]],
        }


#: arena_evaluation's STANDING_MPS and MAX_STEP_FACTOR: below this speed a frame is standing; a frame lasts until the
#: pedestrian's next one, at most this many of its median steps
STANDING_MPS = 0.2
MAX_STEP_FACTOR = 4.0


def counts(tracks: Any, grid: FlowGrid) -> np.ndarray:  # noqa: ANN401 - a polars track table
    """Pedestrian-seconds per cell and bin of a track table (arena_evaluation's `ped_track_store` columns key, t, x, y, vx,
    vy), inside the rooms only, as arena_evaluation's `FlowGrid.counts` bins them."""
    import numpy as np
    import polars as pl

    h = grid.empty()
    if tracks is None or len(tracks) == 0:
        return h
    df = tracks.select("key", "t", "x", "y", "vx", "vy").sort("key", "t")
    step = (pl.col("t").shift(-1) - pl.col("t")).over("key")
    df = df.with_columns(step.alias("dt"))
    df = df.with_columns(pl.col("dt").median().over("key").alias("median_dt"))
    df = df.with_columns(pl.when(pl.col("dt").is_null()).then(pl.col("median_dt")).otherwise(pl.min_horizontal("dt", MAX_STEP_FACTOR * pl.col("median_dt"))).fill_null(0.0).alias("w"))
    derived_x = ((pl.col("x").shift(-1) - pl.col("x")) / pl.col("dt")).over("key")
    derived_y = ((pl.col("y").shift(-1) - pl.col("y")) / pl.col("dt")).over("key")
    df = df.with_columns(
        pl.when(pl.col("vx").is_nan() | pl.col("vx").is_null()).then(derived_x).otherwise(pl.col("vx")).alias("vx"),
        pl.when(pl.col("vy").is_nan() | pl.col("vy").is_null()).then(derived_y).otherwise(pl.col("vy")).alias("vy"),
    ).with_columns(pl.col("vx").forward_fill().over("key").fill_null(0.0), pl.col("vy").forward_fill().over("key").fill_null(0.0))
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    vx, vy, w = df["vx"].to_numpy(), df["vy"].to_numpy(), df["w"].to_numpy()
    i, j, keep = grid.index(x, y)
    keep &= w > 0
    keep &= grid.in_rooms(x, y)
    speed = np.hypot(vx, vy)
    b = np.where(speed < STANDING_MPS, STANDING, heading_bin(np.arctan2(vy, vx)))
    np.add.at(h, (i[keep], j[keep], b[keep]), w[keep])
    return h


def occupancy(counts: np.ndarray) -> np.ndarray:
    """The map with its headings summed: pedestrian-seconds per cell."""
    return counts.sum(axis=-1, keepdims=True)


def normalized(counts: np.ndarray) -> np.ndarray | None:
    """The map as a distribution over cells and bins, None when it holds no time."""
    total = float(counts.sum())
    return None if total <= 0 else counts / total


def jsd(p: np.ndarray, q: np.ndarray) -> float | None:
    """Jensen-Shannon divergence (base 2) of two maps of one grid, None when either holds no time."""
    import numpy as np

    a, b = normalized(p), normalized(q)
    if a is None or b is None:
        return None
    a, b = a.ravel(), b.ravel()
    total = a + b

    def kl(x: np.ndarray) -> float:
        # KL(x || (a + b) / 2) as sum x log2(2x / (a + b)): a + b >= x > 0, where halving a denormal x could round to 0
        k = x > 0
        return float(np.sum(x[k] * np.log2(2 * x[k] / total[k])))

    return max(0.0, (kl(a) + kl(b)) / 2)


def pooled(maps: Sequence[np.ndarray]) -> np.ndarray | None:
    """The mean of the normalized maps (one per seed): the flow the crowd settles to, None when none holds time."""
    import numpy as np

    shares = [n for m in maps if (n := normalized(m)) is not None]
    return None if not shares else np.mean(shares, axis=0)


__all__ = ["BINS", "CELL_M", "HEADINGS", "STANDING", "FlowGrid", "bin_centre", "counts", "heading_bin", "jsd", "normalized", "occupancy", "pooled", "zone_polygons"]
