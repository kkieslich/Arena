"""NAV@r filter for placement layouts: rejects a layout whose elements strand floor or cut a dock off, checked in memory
against a base level rasterized once, each layout only around its elements.

Kolja's rule (9 Nov 2026): a layout is rejected if floor beyond the elements' own footprints and clearance leaves the
dominant traversable component at r = 0.325 m (the wheelchair) by more than 0.5 m^2, or if a dock goto point is outside
the dominant component at r = 0.267 m (the jackal). Each waiting-bay slot counts as occupied by a parked jackal.

Usage: build a `NavrBase` once per base level (the level without the elements), then call `navr_check(base, elements)`
per layout. `elements` are `PlacedElement`s; a waiting bay is a non-blocking element with its slot centre as dock point,
and the filter parks a jackal on it.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Iterable, Sequence
from contextlib import nullcontext

import attrs
import numpy as np
import shapely
from scipy import ndimage

from arena_simulation_setup.metrics.nav_at_r import _CONNECTIVITY, _FREE, CUT_OFF_AREA, RESOLUTION, SPAWN_CLEARANCE, components, expectation, occupancy
from arena_simulation_setup.tree.World import Level, World, WorldView, world_assets

#: The wheelchair's radius: the floor-loss radius
WHEELCHAIR_RADIUS = 0.325
#: The jackal's collision radius: the dock radius and the parked robot's footprint
JACKAL_RADIUS = 0.267
#: (floor-loss radius, dock radius)
RADII = (WHEELCHAIR_RADIUS, JACKAL_RADIUS)
#: Stranded floor up to this area (m^2) is slivers, not a fault
FLOOR_TOLERANCE_M2 = 0.5


class UnresolvedFootprintError(ValueError):
    """A placed element or a base entity whose footprint is unknown, which NAV@r would otherwise count as free floor."""


@dataclasses.dataclass(frozen=True)
class PlacedElement:
    """One placed object as the NAV@r filter takes it.

    - `name`: the object's name in the layout (`charger_1`, `bay_1`, ...), as `NavrVerdict.unreachable_docks` reports it.
    - `footprint`: its floor polygon in the level (world) frame, from its resolved asset. Never a fallback box: resolve
      the models with `resolve_footprints(..., strict=True)`. None or an empty polygon raises `UnresolvedFootprintError`.
    - `dock_point`: the (x, y) a robot is sent to, the dock's goto point; None for an object without a dock.
    - `blocks_floor`: whether the footprint is an obstacle. A charger or a locker blocks. A waiting-bay slot (also the
      loading point) is a floor marking and does not: `navr_check` parks a jackal (a disc of `park_radius`) on the dock
      point of every non-blocking element instead. That jackal blocks in every check but its own slot's goto check (the
      robot sent there is the one that parks). A blocking footprint over a dock point makes that dock unreachable.

    `generative.flowedit.place.placed_elements` builds them from a placement layout.
    """

    name: str
    footprint: shapely.Polygon
    dock_point: tuple[float, float] | None
    blocks_floor: bool


def parked_robot(element: PlacedElement, radius: float = JACKAL_RADIUS) -> shapely.Polygon:
    """The footprint of the jackal `navr_check` parks on a non-blocking element's dock point: a disc of `radius`."""
    if element.dock_point is None:
        raise ValueError(f'{element.name!r} has no dock point to park a robot on')
    return shapely.Point(element.dock_point).buffer(radius, quad_segs=16)


@attrs.frozen
class NavrVerdict:
    """The filter's verdict on one layout.

    `ok` unless `floor_loss_m2` (floor of the base's dominant component at the floor radius that is still traversable
    but no longer in the dominant component, i.e. beyond the elements' footprints and their clearance) exceeds the
    tolerance, or a dock point is in `unreachable_docks` (outside the dominant component at the dock radius).
    `new_cutoff_zones` are zones NAV@r reports cut off at the floor radius now but not on the base; they are reported,
    not filtered on. Per radius: the layout's NAV@r value, its dominant component's area (m^2) and, with a start, the
    start component's share. `relabelled` names the radii whose components were recomputed in full.
    """

    ok: bool
    floor_loss_m2: float
    new_cutoff_zones: tuple[str, ...]
    unreachable_docks: tuple[str, ...]
    nav_at_r: dict[float, float]
    dominant_m2: dict[float, float]
    start_share: dict[float, float] | None
    relabelled: tuple[float, ...]
    reasons: tuple[str, ...]


@attrs.frozen(eq=False)
class _Zone:
    """A zone's window on the grid and the cells whose centres it holds."""

    name: str
    area: float
    rows: slice
    cols: slice
    inside: np.ndarray


@attrs.frozen(eq=False)
class _Layer:
    """The base at one radius: traversable and spawn cells, their components, the dominant label and per-zone data."""

    radius: float
    spawn_radius: float
    traversable: np.ndarray
    spawnable: np.ndarray
    labels: np.ndarray
    cells: np.ndarray
    spawns: np.ndarray
    dominant: int
    #: Per zone of the base: whether it holds a cell traversable on the walls alone, and its dominant cells
    walls_ok: np.ndarray
    reach: np.ndarray
    cut_off: frozenset[str]
    starts: dict[tuple[float, float], tuple[int, int] | None] = attrs.field(factory=dict)


class NavrBase:
    """One level rasterized once for `navr_check`, with per radius its traversable cells and components.

    `level` is the base layout without the placed elements; `exclude` drops objects by name first (e.g. elements the
    layout moves). Objects whose model resolves without a bounding box (floor markings, scans) are free floor, as in
    NAV@r; an object whose model does not resolve at all raises `UnresolvedFootprintError`, because NAV@r would skip it
    silently (e.g. bucket models with ARENA_ASSETS_NET=off). Pass `view` (and the loaded `world`) for world-local assets.
    """

    def __init__(
        self,
        level: Level,
        *,
        radii: Iterable[float] = RADII,
        resolution: float = RESOLUTION,
        spawn_clearance: float = SPAWN_CLEARANCE,
        exclude: Collection[str] = (),
        view: WorldView | None = None,
        world: World | None = None,
    ) -> None:
        if exclude:
            level = attrs.evolve(level, objects=[o for o in level.objects if o.name not in exclude])
        with world_assets(view, world) if view is not None else nullcontext():
            occ = occupancy(level, resolution=resolution)
        if occ.missing:
            raise UnresolvedFootprintError(f'base objects {list(occ.missing)} have models that do not resolve, so NAV@r would count their floor as free: check the asset buckets and ARENA_ASSETS_NET')
        self.level = level
        self.resolution = resolution
        self.spawn_clearance = spawn_clearance
        self.origin = occ.origin
        self.frame = occ.frame
        self.occupied = occ.grid != _FREE
        self.walls_free = occ.walls == _FREE
        self.unannotated = occ.unresolved
        self.zones = tuple(self._zone(zone.name, zone.shape()) for zone in level.zones if len(zone.polygon) >= 3)
        self._layers: dict[float, _Layer] = {}
        for radius in radii:
            self.layer(radius)

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape  # type: ignore[return-value]

    def layer(self, radius: float) -> _Layer:
        """The base at `radius`, computed on first use."""
        radius = float(radius)
        if radius in self._layers:
            return self._layers[radius]
        res = self.resolution
        clearance = _clearance(~self.occupied, res)
        spawn_radius = max(self.spawn_clearance, radius)
        traversable = clearance >= radius
        spawnable = traversable & (clearance >= self.spawn_clearance)
        del clearance
        labels, cells, spawns = components(traversable, spawnable)
        dominant = int(np.argmax(cells)) + 1 if len(cells) else 0
        walls_traversable = _clearance(self.walls_free, res) >= radius
        walls_ok = np.array([bool(np.any(z.inside & walls_traversable[z.rows, z.cols])) for z in self.zones], dtype=bool)
        del walls_traversable
        reach = np.array([np.count_nonzero(z.inside & (labels[z.rows, z.cols] == dominant)) for z in self.zones], dtype=np.int64)
        cut_off = frozenset(z.name for z, ok, n in zip(self.zones, walls_ok, reach, strict=True) if _cut_off(z, ok, n))
        layer = _Layer(radius, spawn_radius, traversable, spawnable, labels, cells, spawns, dominant, walls_ok, reach, cut_off)
        self._layers[radius] = layer
        return layer

    def _zone(self, name: str, polygon: shapely.Polygon) -> _Zone:
        height, width = self.shape
        res, (ox, oy) = self.resolution, self.origin
        x_min, y_min, x_max, y_max = polygon.bounds
        cols = slice(max(0, math.floor((x_min - ox) / res)), min(width, math.ceil((x_max - ox) / res)))
        rows = slice(max(0, height - math.ceil((y_max - oy) / res)), min(height, height - math.floor((y_min - oy) / res)))
        if cols.stop <= cols.start or rows.stop <= rows.start:
            return _Zone(name, polygon.area, slice(0, 0), slice(0, 0), np.zeros((0, 0), dtype=bool))
        xs = ox + (np.arange(cols.start, cols.stop) + 0.5) * res
        ys = oy + (height - 1 - np.arange(rows.start, rows.stop) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys)
        inside = shapely.contains_xy(polygon, xx, yy)
        return _Zone(name, polygon.area, rows, cols, inside)

    def cell(self, point: Sequence[float]) -> tuple[int, int] | None:
        """The (row, col) of the cell holding a level-frame point, None off the grid."""
        height, width = self.shape
        row = height - 1 - math.floor((point[1] - self.origin[1]) / self.resolution)
        col = math.floor((point[0] - self.origin[0]) / self.resolution)
        return (row, col) if 0 <= row < height and 0 <= col < width else None

    def nav_at_r(self, radius: float) -> float:
        layer = self.layer(radius)
        return expectation(layer.cells, layer.spawns)[0]

    def dominant_m2(self, radius: float) -> float:
        layer = self.layer(radius)
        return float(layer.cells[layer.dominant - 1]) * self.resolution**2 if layer.dominant else 0.0

    def cut_off_zones(self, radius: float) -> tuple[str, ...]:
        layer = self.layer(radius)
        return tuple(z.name for z in self.zones if z.name in layer.cut_off)

    def _snap(self, layer: _Layer, start: tuple[float, float]) -> tuple[int, int] | None:
        """The traversable base cell nearest to `start`, as NAV@r snaps it (first in row-major order among ties)."""
        if start not in layer.starts:
            layer.starts[start] = _nearest(layer.traversable, self, start)
        return layer.starts[start]


def _nearest(traversable: np.ndarray, base: NavrBase, start: tuple[float, float]) -> tuple[int, int] | None:
    cell = base.cell(start)
    if cell is not None and traversable[cell]:
        return cell
    cells = np.argwhere(traversable)
    if not len(cells):
        return None
    height = traversable.shape[0]
    distances = np.hypot(base.origin[0] + (cells[:, 1] + 0.5) * base.resolution - start[0], base.origin[1] + (height - 1 - cells[:, 0] + 0.5) * base.resolution - start[1])
    row, col = cells[int(distances.argmin())]
    return int(row), int(col)


def _cut_off(zone: _Zone, walls_ok: bool, reachable: int) -> bool:
    """NAV@r's cut-off test: a zone of at least CUT_OFF_AREA, traversable on the walls alone, with no dominant cell."""
    return zone.area >= CUT_OFF_AREA and reachable == 0 and bool(walls_ok)


def _clearance(free: np.ndarray, resolution: float) -> np.ndarray:
    """`clearance_of` on a bool free mask: meters from each cell centre to the nearest occupied cell centre."""
    distance = ndimage.distance_transform_edt(free)
    assert isinstance(distance, np.ndarray)
    return distance * resolution


@attrs.frozen(eq=False)
class _Mask:
    """The cells one element's footprint fills: the top-left (row, col) and a bool mask."""

    row: int
    col: int
    mask: np.ndarray

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.row, self.col, self.row + self.mask.shape[0], self.col + self.mask.shape[1]


def _paint(masks: Iterable[_Mask], rows: slice, cols: slice) -> np.ndarray:
    """The union of `masks` on the window `rows`, `cols`."""
    out = np.zeros((rows.stop - rows.start, cols.stop - cols.start), dtype=bool)
    for m in masks:
        r0, c0, r1, c1 = m.box
        a0, a1, b0, b1 = max(r0, rows.start), min(r1, rows.stop), max(c0, cols.start), min(c1, cols.stop)
        if a0 < a1 and b0 < b1:
            out[a0 - rows.start : a1 - rows.start, b0 - cols.start : b1 - cols.start] |= m.mask[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0]
    return out


def _windows(masks: Sequence[_Mask], margin: int, shape: tuple[int, int]) -> list[tuple[slice, slice, list[_Mask]]]:
    """The masks grouped into disjoint windows, each its masks' bounding box grown by `margin` cells and clipped to the grid."""
    groups = [[(max(0, r0 - margin), max(0, c0 - margin), min(shape[0], r1 + margin), min(shape[1], c1 + margin)), [m]] for m in masks for r0, c0, r1, c1 in [m.box]]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                (a0, b0, a1, b1), (c0, d0, c1, d1) = groups[i][0], groups[j][0]
                if a0 < c1 and c0 < a1 and b0 < d1 and d0 < b1:
                    groups[i] = [(min(a0, c0), min(b0, d0), max(a1, c1), max(b1, d1)), groups[i][1] + groups[j][1]]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    return [(slice(r0, r1), slice(c0, c1), ms) for (r0, c0, r1, c1), ms in groups]


@attrs.frozen(eq=False)
class _Window:
    """One window of a layout at one radius: the base's traversable and spawn cells the layout removes, and the new
    labels of the pieces split off inside it (pockets, 0 elsewhere), None without any."""

    rows: slice
    cols: slice
    removed: np.ndarray
    removed_spawn: np.ndarray
    pockets: np.ndarray | None = None

    def clip(self, rows: slice, cols: slice) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
        """The overlap with `rows`, `cols` as (slices into this window, slices into theirs), None when disjoint."""
        a0, a1, b0, b1 = max(rows.start, self.rows.start), min(rows.stop, self.rows.stop), max(cols.start, self.cols.start), min(cols.stop, self.cols.stop)
        if a0 >= a1 or b0 >= b1:
            return None
        return (slice(a0 - self.rows.start, a1 - self.rows.start), slice(b0 - self.cols.start, b1 - self.cols.start)), (slice(a0 - rows.start, a1 - rows.start), slice(b0 - cols.start, b1 - cols.start))


@attrs.define(eq=False)
class _State:
    """A layout at one radius. In local mode the components are the base's with `cells`/`spawns` reduced plus the
    pockets the windows split off (labels after the base's), `lost` the base dominant's cells in those pockets; in full
    mode `labels` holds the whole relabelling."""

    layer: _Layer
    windows: list[_Window]
    cells: np.ndarray
    spawns: np.ndarray
    dominant: int
    labels: np.ndarray | None = None
    lost: int = 0

    @property
    def full(self) -> bool:
        return self.labels is not None

    def label_window(self, rows: slice, cols: slice) -> np.ndarray:
        """The layout's component labels on a window, 0 off its traversable cells."""
        if self.labels is not None:
            return self.labels[rows, cols]
        out = self.layer.labels[rows, cols].copy()
        for w in self.windows:
            overlap = w.clip(rows, cols)
            if overlap is None:
                continue
            mine, theirs = overlap
            out[theirs][w.removed[mine]] = 0
            if w.pockets is not None:
                pockets = w.pockets[mine]
                out[theirs][pockets > 0] = pockets[pockets > 0]
        return out

    def label_at(self, cell: tuple[int, int]) -> int:
        row, col = cell
        return int(self.label_window(slice(row, row + 1), slice(col, col + 1))[0, 0])


def _update(base: NavrBase, layer: _Layer, masks: Sequence[_Mask], force_full: bool = False) -> _State:
    """The layout's components at `layer`'s radius: the base's, with the cells within the radius of the new footprints
    removed, each footprint group in its own disjoint window that holds every removed cell away from its ring.

    In a window, a piece of what is left that does not touch the ring is a pocket, a component of its own. If each base
    component that loses cells keeps at most one piece on the ring of each window, the rest of it stays connected: any
    path through a window enters and leaves on the ring and can be rerouted through that piece. Then the base labels
    stand, less the removed cells, plus the pockets. Otherwise, or when the dominant component changes, it relabels all."""
    res = base.resolution
    margin = math.ceil(layer.spawn_radius / res) + 2
    count = len(layer.cells)
    windows: list[_Window] = []
    removed_cells = np.zeros(count + 1, dtype=np.int64)
    removed_spawns = np.zeros(count + 1, dtype=np.int64)
    pocket_cells: list[int] = []
    pocket_spawns: list[int] = []
    lost = 0
    split = force_full
    for rows, cols, group in _windows(masks, margin, base.shape):
        new = _paint(group, rows, cols)
        if not new.any():
            continue
        distance = ndimage.distance_transform_edt(~new)
        assert isinstance(distance, np.ndarray)
        clearance = distance * res
        traversable = layer.traversable[rows, cols]
        removed = traversable & ~(clearance >= layer.radius)
        removed_spawn = layer.spawnable[rows, cols] & ~(clearance >= layer.spawn_radius)
        base_labels = layer.labels[rows, cols]
        if split or not removed.any():
            windows.append(_Window(rows, cols, removed, removed_spawn))
            if not split:
                removed_spawns += np.bincount(base_labels[removed_spawn], minlength=count + 1)  # spawn cells only, no component changes
            continue
        kept = traversable & ~removed
        local, pieces = ndimage.label(kept, structure=_CONNECTIVITY)
        piece_label = np.zeros(pieces + 1, dtype=np.int64)
        piece_label[local[kept]] = base_labels[kept]  # each piece lies in one base component
        ring = np.zeros(kept.shape, dtype=bool)
        ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
        on_ring = np.zeros(pieces + 1, dtype=bool)
        on_ring[local[ring & kept]] = True
        on_ring[0] = False
        affected = np.zeros(count + 1, dtype=bool)
        affected[base_labels[removed]] = True
        ring_pieces = np.bincount(piece_label[on_ring], minlength=count + 1)
        if np.any(ring_pieces[affected] > 1):
            split = True
            windows.append(_Window(rows, cols, removed, removed_spawn))
            continue
        removed_cells += np.bincount(base_labels[removed], minlength=count + 1)
        removed_spawns += np.bincount(base_labels[removed_spawn], minlength=count + 1)
        pocket_ids = np.flatnonzero(~on_ring & affected[piece_label])
        pocket_ids = pocket_ids[pocket_ids > 0]
        pockets = None
        if len(pocket_ids):
            sizes = np.bincount(local[kept], minlength=pieces + 1)
            spawn_sizes = np.bincount(local[layer.spawnable[rows, cols] & ~removed_spawn], minlength=pieces + 1)
            relabel = np.zeros(pieces + 1, dtype=np.int64)
            for piece in pocket_ids:
                relabel[piece] = count + 1 + len(pocket_cells)
                pocket_cells.append(int(sizes[piece]))
                pocket_spawns.append(int(spawn_sizes[piece]))
                removed_cells[piece_label[piece]] += sizes[piece]
                removed_spawns[piece_label[piece]] += spawn_sizes[piece]
                lost += int(sizes[piece]) if piece_label[piece] == layer.dominant else 0
            pockets = relabel[local]
        windows.append(_Window(rows, cols, removed, removed_spawn, pockets))
    if not split:
        cells = np.concatenate([layer.cells - removed_cells[1:], np.array(pocket_cells, dtype=np.int64)])
        spawns = np.concatenate([layer.spawns - removed_spawns[1:], np.array(pocket_spawns, dtype=np.int64)])
        dominant = int(np.argmax(cells)) + 1 if len(cells) else 0
        if dominant == layer.dominant:
            return _State(layer, windows, cells, spawns, dominant, lost=lost)
    traversable = layer.traversable.copy()
    spawnable = layer.spawnable.copy()
    for w in windows:
        traversable[w.rows, w.cols] &= ~w.removed
        spawnable[w.rows, w.cols] &= ~w.removed_spawn
    labels, cells, spawns = components(traversable, spawnable)
    dominant = int(np.argmax(cells)) + 1 if len(cells) else 0
    return _State(layer, windows, cells, spawns, dominant, labels)


def _floor_loss(layer: _Layer, state: _State) -> int:
    """Cells of the base's dominant component still traversable but outside the layout's dominant component."""
    if not state.full:
        return state.lost
    assert state.labels is not None
    return int(np.count_nonzero((layer.labels == layer.dominant) & (state.labels > 0) & (state.labels != state.dominant)))


def _cut_off_zones(base: NavrBase, layer: _Layer, state: _State) -> tuple[str, ...]:
    out = []
    for i, zone in enumerate(base.zones):
        if not zone.inside.size:
            continue
        if state.full:
            reach = int(np.count_nonzero(zone.inside & (state.label_window(zone.rows, zone.cols) == state.dominant)))
        else:
            reach = int(layer.reach[i])
            for w in state.windows:
                overlap = w.clip(zone.rows, zone.cols)
                if overlap is None:
                    continue
                mine, theirs = overlap
                gone = w.removed[mine] if w.pockets is None else w.removed[mine] | (w.pockets[mine] > 0)
                dominant = layer.labels[w.rows, w.cols][mine] == layer.dominant
                reach -= int(np.count_nonzero(gone & dominant & zone.inside[theirs]))
        if _cut_off(zone, layer.walls_ok[i], reach):
            out.append(zone.name)
    return tuple(out)


def _dock_reachable(base: NavrBase, state: _State, cell: tuple[int, int], lifted: Sequence[_Mask], kept: Sequence[_Mask]) -> bool:
    """Whether the dock `cell` is in the layout's dominant component with the `lifted` footprints (its parked robot)
    gone. The lift frees cells only within the radius of the lifted footprints, inside the window around them; so the
    dock is reachable iff its piece of the window, joined with the window's other pieces through the layout's own
    components (which agree with it outside the window), holds a cell of the dominant component."""
    if not lifted:
        return state.dominant > 0 and state.label_at(cell) == state.dominant
    res, radius = base.resolution, state.layer.radius
    margin = math.ceil(radius / res) + 2
    height, width = base.shape
    r0 = min(m.row for m in lifted) - margin
    c0 = min(m.col for m in lifted) - margin
    r1 = max(m.row + m.mask.shape[0] for m in lifted) + margin
    c1 = max(m.col + m.mask.shape[1] for m in lifted) + margin
    rows, cols = slice(max(0, r0), min(height, r1)), slice(max(0, c0), min(width, c1))
    p_rows, p_cols = slice(max(0, r0 - margin), min(height, r1 + margin)), slice(max(0, c0 - margin), min(width, c1 + margin))
    occupied = base.occupied[p_rows, p_cols] | _paint(kept, p_rows, p_cols)
    inner = (slice(rows.start - p_rows.start, rows.stop - p_rows.start), slice(cols.start - p_cols.start, cols.stop - p_cols.start))
    if occupied.any():
        distance = ndimage.distance_transform_edt(~occupied)
        assert isinstance(distance, np.ndarray)
        traversable = (distance * res)[inner] >= radius
    else:  # open floor all around: no obstacle within the radius of the window
        traversable = np.ones(occupied.shape, dtype=bool)[inner]
    local, _ = ndimage.label(traversable, structure=_CONNECTIVITY)
    piece = int(local[cell[0] - rows.start, cell[1] - cols.start])
    if not piece:
        return False
    labels = state.label_window(rows, cols)
    both = (local > 0) & (labels > 0)
    edges = np.unique(np.stack([local[both], labels[both]]), axis=1)
    # union-find over the window's pieces (p) and the layout's components (-g) they touch
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, g in edges.T:
        parent[find(int(p))] = find(-int(g))
    return find(piece) == find(-state.dominant)


def navr_check(
    base: NavrBase | Level,
    elements: Iterable[PlacedElement],
    *,
    radii: tuple[float, float] = RADII,
    tolerance_m2: float = FLOOR_TOLERANCE_M2,
    start: tuple[float, float] | None = None,
    park_radius: float | None = JACKAL_RADIUS,
    relabel: bool = False,
) -> NavrVerdict:
    """Check one layout's `elements` against `base` (a `NavrBase`, or a level to precompute one for).

    `radii` is (floor-loss radius, dock radius): the floor loss and the new cut-off zones at the first, the dock goto
    points at the second, NAV@r at both. A jackal of `park_radius` is parked on the dock point of each non-blocking
    element (None parks none). Raises `UnresolvedFootprintError` for an element without a footprint. `relabel`
    recomputes every component in full (the reference the incremental update must match)."""
    if not isinstance(base, NavrBase):
        base = NavrBase(base, radii=radii)
    elements = list(elements)
    bad = [e.name for e in elements if not isinstance(e.footprint, shapely.Polygon) or e.footprint.is_empty or e.footprint.area <= 0.0]
    if bad:
        raise UnresolvedFootprintError(f'elements {bad} have no footprint: resolve their models (resolve_footprints(..., strict=True)) instead of letting NAV@r skip them')
    masks: dict[int, _Mask] = {}  # blocking footprints by element index
    parked: dict[int, _Mask] = {}  # jackals parked on non-blocking elements' dock points
    for i, e in enumerate(elements):
        footprint = e.footprint if e.blocks_floor else parked_robot(e, park_radius) if e.dock_point is not None and park_radius else None
        placed = None if footprint is None else base.frame.polygon_mask(footprint)
        if placed is not None:
            (masks if e.blocks_floor else parked)[i] = _Mask(*placed)
    obstacles = [*masks.values(), *parked.values()]
    floor_radius, dock_radius = (float(r) for r in radii)
    states = {r: _update(base, base.layer(r), obstacles, force_full=relabel) for r in dict.fromkeys((floor_radius, dock_radius))}

    floor = states[floor_radius]
    loss = _floor_loss(base.layer(floor_radius), floor) * base.resolution**2
    new_cut = tuple(z for z in _cut_off_zones(base, base.layer(floor_radius), floor) if z not in base.layer(floor_radius).cut_off)

    dock = states[dock_radius]
    unreachable = []
    for i, e in enumerate(elements):
        if e.dock_point is None:
            continue
        cell = base.cell(e.dock_point)
        if cell is None:
            unreachable.append(e.name)
            continue
        lifted = [parked[i]] if i in parked else []
        kept = [*masks.values(), *(m for j, m in parked.items() if j != i)]
        if not _dock_reachable(base, dock, cell, lifted, kept):
            unreachable.append(e.name)

    values = {r: expectation(s.cells, s.spawns)[0] for r, s in states.items()}
    dominant = {r: float(s.cells[s.dominant - 1]) * base.resolution**2 if s.dominant else 0.0 for r, s in states.items()}
    shares = None
    if start is not None:
        shares = {r: _start_share(base, s, start) for r, s in states.items()}
    reasons = []
    if loss > tolerance_m2:
        reasons.append(f'{loss:.2f} m2 of floor leaves the dominant component at r {floor_radius:g} (tolerance {tolerance_m2:g})')
    if unreachable:
        reasons.append(f'docks {unreachable} are outside the dominant component at r {dock_radius:g}')
    return NavrVerdict(
        ok=not reasons,
        floor_loss_m2=loss,
        new_cutoff_zones=new_cut,
        unreachable_docks=tuple(unreachable),
        nav_at_r=values,
        dominant_m2=dominant,
        start_share=shares,
        relabelled=tuple(r for r, s in states.items() if s.full),
        reasons=tuple(reasons),
    )


def _start_share(base: NavrBase, state: _State, start: tuple[float, float]) -> float:
    """The start component's share of the traversable cells, the start snapped to the nearest traversable cell as NAV@r does."""
    layer = state.layer
    total = state.cells.sum()
    if not total:
        return 0.0
    snapped = base._snap(layer, start)
    if snapped is not None and state.label_at(snapped):
        return float(state.cells[state.label_at(snapped) - 1] / total)
    labels = state.labels if state.labels is not None else state.label_window(slice(0, base.shape[0]), slice(0, base.shape[1]))
    snapped = _nearest(labels > 0, base, start)
    return 0.0 if snapped is None else float(state.cells[labels[snapped] - 1] / total)
