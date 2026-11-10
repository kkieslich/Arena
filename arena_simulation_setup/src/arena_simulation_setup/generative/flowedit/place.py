"""INFRA placement operators: the MVP elements (study design §3), their deterministic wall-adjacent candidate spots every
1 m (§4) with a verdict per §3 rule, and layouts of one spot per element inserted, moved or removed through FLOWEDIT edits.

A spot stands against a straight wall piece of an allowed zone (`generate.wall_pieces`), its objects' fronts (the
models' annotated `face`) turned into the room, their backs at the generator's wall gap. A pair (2 chargers, 2 bays)
is one spot: its members stand `spacing_m` apart, centre to centre, on the same piece. The catalog holds generic rules;
a site's facts (`SiteFacts`: entrances, the courier access point, power outlets) are what some of them read. Rules with
data say pass or fail; rules without site data (outlets while a site names none, tactile paths, exit width, dead ends,
recipient entrances or a courier access point a site does not name, and doors or lifts on a level without any) say
unknown and filter nothing. NAV@r is not a rule here: `placed_elements` hands a layout to the NAV@r filter."""

from __future__ import annotations

import collections
import enum
import fnmatch
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from arena_simulation_setup.metrics.navr_filter import PlacedElement

from .edit import SEPARATOR, EditType, Site, apply, save, world_object

if TYPE_CHECKING:
    import numpy as np
    import shapely

    from arena_simulation_setup.tree.World.model import Level

    from .generate import WallPiece

#: Spots lie this far apart along a wall piece (§4)
STEP_M = 1.0
#: The members of a pair stand this far apart, centre to centre, on one wall piece
PAIR_SPACING_M = 1.0
#: The remaining clear corridor width beyond an element, its docks and its clear zone (§3, placeholder)
CORRIDOR_MIN_M = 1.8
#: Corridor widths are measured up to this width; a wider corridor reports it
CORRIDOR_CAP_M = 5.0
#: The wheelchair turning circle's diameter that stays free in front of each element (§3)
TURNING_CIRCLE_M = 1.5
#: Chargers and waiting bays keep this far from door spans and lift cabins (§3)
DOOR_DISTANCE_M = 2.0
#: The fleet study's charger power (Kolja: 22 Wh batteries, 175 W chargers)
CHARGE_POWER_W = 175.0
#: The floor raster's cell size, the one NAV@r and the nav map use
RESOLUTION_M = 0.05
#: Zones no element goes to besides the cut-off ones, as `fnmatch` patterns: none by default. The zones NAV@r cuts off
#: at the wheelchair radius are left out on every site (`cut_off_zones`); a site names others where it has a reason
#: NAV@r does not see (the twin's 4 % ramp, in its INFRA site config).
EXCLUDED_ZONES: tuple[str, ...] = ()
#: Overlaps below this area (m^2) are shared edges, not overlaps: under a mm along a pair's 1.2 m dock edge, which
#: poses to 0.1 mm (`POSITION_DECIMALS`, as world.yaml keeps them) can leave between docks that meet
AREA_EPS = 1e-3

#: The jackal dock, 1.2 m deep and 1.0 m wide, in front of a charger whose origin is its middle (face +x)
CHARGER_DOCK = ((0.25, -0.5), (1.45, -0.5), (1.45, 0.5), (0.25, 0.5))
#: The jackal dock in front of a parcel locker whose origin is a back corner (face +x): from 0.84 m, just beyond the
#: 0.79 m deep box its doors sweep open (the model's swept bounding box), so a docked robot leaves the doors free
LOCKER_DOCK = ((0.84, 0.02), (2.04, 0.02), (2.04, 1.02), (0.84, 1.02))
#: A waiting bay's slot around its origin, 1.2 m deep and 1.0 m wide: its footprint and its dock (face +x)
SLOT = ((-0.6, -0.5), (0.6, -0.5), (0.6, 0.5), (-0.6, 0.5))
#: Placeholder: a parcel locker stands within this walking distance of an entrance parcel recipients use
RECIPIENT_DISTANCE_M = 30.0
#: A charger or a locker stands within this cable run (walked on the floor) of a power outlet (§3)
OUTLET_CABLE_M = 10.0
#: The courier handoff point stands at the courier access point: its goto point within this distance of it
COURIER_ACCESS_M = 0.5
#: The kinds of entrance a site names (`Entrance.type`)
ENTRANCE_TYPES = ("mrt", "building", "shop", "escalator", "stairs", "lift", "other")


class Verdict(enum.StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class Rule(enum.StrEnum):
    """The §3 rules a spot gets a verdict on, in report order."""

    AGAINST_WALL = "against_wall"  # every object backs onto the wall piece (§3: against a wall)
    NO_OVERLAP = "no_overlap"  # footprints on the floor, clear of walls, door spans and objects
    CLEAR_ZONE = "clear_zone"  # the approach or access zone in front is free floor
    TURNING_CIRCLE = "turning_circle"  # a free 1.5 m circle in front
    CORRIDOR_WIDTH = "corridor_width"  # remaining clear corridor width
    DOCKS_ON_FLOOR = "docks_on_floor"  # every dock free floor, its goto point on a free cell
    PEDS_ZONES = "peds_zones"  # outside PEDS source and sink zones; unknown until they are given
    DOOR_DISTANCE = "door_distance"  # >= 2 m from door spans; unknown on a level without doors
    LIFT_DISTANCE = "lift_distance"  # >= 2 m from lift cabins; unknown on a level without lifts
    OUTLET_CABLE = "outlet_cable"  # cable run <= 10 m to a site outlet; unknown while the site names no outlets
    RECIPIENT_DISTANCE = "recipient_distance"  # within walking distance of a recipient entrance; unknown without one
    COURIER_ACCESS = "courier_access"  # at the site's courier access point; unknown without one
    TACTILE_PATH = "tactile_path"  # tactile paths clear with a 0.6 m margin; no data
    EXIT_WIDTH = "exit_width"  # required exit width clear; no data
    DEAD_END_TURNING = "dead_end_turning"  # a turning circle at every dead end the element makes; not computed


#: Rules every element gets a verdict on
COMMON_RULES = (Rule.AGAINST_WALL, Rule.NO_OVERLAP, Rule.TURNING_CIRCLE, Rule.CORRIDOR_WIDTH, Rule.DOCKS_ON_FLOOR, Rule.PEDS_ZONES, Rule.TACTILE_PATH, Rule.EXIT_WIDTH, Rule.DEAD_END_TURNING)
#: Rules without site data: always unknown
NO_DATA_RULES = (Rule.TACTILE_PATH, Rule.EXIT_WIDTH, Rule.DEAD_END_TURNING)
#: Rules that read the site's facts (`SiteFacts`): unknown while the site gives none for them
FACT_RULES = (Rule.OUTLET_CABLE, Rule.RECIPIENT_DISTANCE, Rule.COURIER_ACCESS)

Point = tuple[float, float]


@attrs.frozen
class Element:
    """One placeable element of the catalog: `members` objects of `model`, `spacing_m` apart along the wall, each with
    `dock` (object frame) and, for a model without a bounding box, `footprint` (object frame); `clear_m` of clear
    approach or access in front of the footprints; a robot in a dock charges with `charge_power_w`. `blocks_floor` is
    False for floor markings, which only a parked robot blocks. `recipient_distance_m` is the walking distance to a
    recipient entrance it keeps within, `courier_access` whether it stands at the courier access point. `rules` are the
    rules it gets a verdict on."""

    name: str
    model: str
    members: tuple[str, ...]
    dock: tuple[Point, ...]
    spacing_m: float = 0.0
    footprint: tuple[Point, ...] | None = None
    clear_m: float = 0.0
    turning_circle_m: float = TURNING_CIRCLE_M
    charge_power_w: float | None = None
    blocks_floor: bool = True
    door_distance_m: float | None = None
    recipient_distance_m: float | None = None
    courier_access: bool = False
    rules: tuple[Rule, ...] = COMMON_RULES


def _rules(*extra: Rule) -> tuple[Rule, ...]:
    return tuple(rule for rule in Rule if rule in COMMON_RULES or rule in extra)


#: The MVP catalog (§3, §4: 1 charger dock of 2 chargers, 1 waiting bay of 2 slots, 1 parcel locker) and the courier
#: handoff point, where couriers hand parcels to the fleet and its deliveries start (§8; element and object
#: `loading_point`), by element name. Object names as the fleet mode refers to them. Each element's placement rules are
#: generic; what they read about a site comes from its facts (`SiteFacts`), and the search decides.
CATALOG: Mapping[str, Element] = {
    element.name: element
    for element in (
        Element("charger_pair", "Common/Object/robot_charger", ("charger_1", "charger_2"), CHARGER_DOCK, PAIR_SPACING_M, clear_m=1.2, charge_power_w=CHARGE_POWER_W, door_distance_m=DOOR_DISTANCE_M, rules=_rules(Rule.CLEAR_ZONE, Rule.DOOR_DISTANCE, Rule.LIFT_DISTANCE, Rule.OUTLET_CABLE)),
        Element("bay_pair", "Common/Object/waiting_bay", ("bay_1", "bay_2"), SLOT, PAIR_SPACING_M, footprint=SLOT, blocks_floor=False, door_distance_m=DOOR_DISTANCE_M, rules=_rules(Rule.DOOR_DISTANCE, Rule.LIFT_DISTANCE)),
        Element("locker", "Common/Object/parcel_locker", ("locker_1",), LOCKER_DOCK, clear_m=1.5, recipient_distance_m=RECIPIENT_DISTANCE_M, rules=_rules(Rule.CLEAR_ZONE, Rule.OUTLET_CABLE, Rule.RECIPIENT_DISTANCE)),
        Element("loading_point", "Common/Object/waiting_bay", ("loading_point",), SLOT, footprint=SLOT, blocks_floor=False, courier_access=True, rules=_rules(Rule.COURIER_ACCESS)),
    )
}


@attrs.frozen
class Entrance:
    """An entrance of a site, a fact the catalog's rules read: its `name`, `type` (one of `ENTRANCE_TYPES`), where it
    is (`points`: one world-frame point, or the corners of a polygon), whether parcel recipients come in through it
    (`recipients`), its `level` (None: every level), and whether its position is a placeholder. Walking distances run
    from its `anchor`."""

    name: str
    type: str
    points: tuple[Point, ...]
    recipients: bool = False
    level: str | None = None
    placeholder: bool = False

    def __attrs_post_init__(self) -> None:
        if self.type not in ENTRANCE_TYPES:
            raise ValueError(f"entrance {self.name!r}: type {self.type!r} is not one of {', '.join(ENTRANCE_TYPES)}")
        if len(self.points) == 2 or not self.points:
            raise ValueError(f"entrance {self.name!r}: a point or a polygon of 3 or more corners")

    @property
    def anchor(self) -> Point:
        """The point itself, or the polygon's centroid."""
        if len(self.points) == 1:
            return self.points[0]
        import shapely

        c = shapely.Polygon(self.points).centroid
        return (float(c.x), float(c.y))


@attrs.frozen
class SiteFacts:
    """What a site is, apart from the catalog: its `entrances`, the `courier_access` point where couriers arrive with
    parcels, and its power `outlets` (world-frame points). The catalog's rules that read them (`FACT_RULES`) are unknown
    while a site gives none for them; the courier handoff point is pinned at the access point (`SiteFacts.pins`)."""

    entrances: tuple[Entrance, ...] = ()
    courier_access: Point | None = None
    outlets: tuple[Point, ...] = ()

    @property
    def recipient_entrances(self) -> tuple[Entrance, ...]:
        return tuple(e for e in self.entrances if e.recipients)

    def pins(self) -> dict[str, Point]:
        """The elements the facts pin by rule, by name: an element at the courier access point, there (one access point,
        one spot)."""
        if self.courier_access is None:
            return {}
        return {name: self.courier_access for name, element in CATALOG.items() if element.courier_access}


@attrs.frozen(eq=False)
class PlacedObject:
    """One object of a spot: its pose, level-frame footprint and dock, and the dock's goto point (the deepest point of
    the dock, where the runtime sends a robot)."""

    name: str
    model: str
    x: float
    y: float
    yaw: float
    footprint: shapely.Polygon
    dock: shapely.Polygon
    goto: Point


@attrs.frozen(eq=False)
class Spot:
    """A candidate spot of `element`: step `step` along wall piece `piece` (ring order, `piece_m` long) of `zone` on
    `level`, its centre `along_m` from the piece's start, the piece's `frame`, its `objects`, the `clear` (approach or
    access) zone and the `turning` circle in front, the remaining `corridor_m`, and a verdict per rule of the element."""

    element: Element
    level: str
    zone: str
    piece: int
    step: int
    along_m: float
    piece_m: float
    frame: _Frame
    objects: tuple[PlacedObject, ...]
    clear: shapely.Polygon
    turning: shapely.Polygon | None
    corridor_m: float
    verdicts: tuple[tuple[Rule, Verdict], ...]

    @property
    def id(self) -> str:
        """Short hash of the element and its objects' poses, to the mm: the same spot on the same site keeps it."""
        record = {"element": self.element.name, "level": self.level, "objects": [[o.name, o.model, round(o.x, 3), round(o.y, 3), round(o.yaw, 4)] for o in self.objects]}
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:8]

    @property
    def verdict(self) -> dict[Rule, Verdict]:
        return dict(self.verdicts)

    @property
    def failed(self) -> tuple[Rule, ...]:
        return tuple(rule for rule, verdict in self.verdicts if verdict is Verdict.FAIL)

    @property
    def valid(self) -> bool:
        """Whether no rule fails; unknown rules do not filter."""
        return not self.failed

    @property
    def taken(self) -> shapely.Polygon:
        """The floor the element and the robots in its docks stand on."""
        import shapely

        return shapely.union_all([g for o in self.objects for g in (o.footprint, o.dock)])

    @property
    def keep_clear(self) -> shapely.Polygon:
        """The floor no other element may take: `taken`, the clear zone and the turning circle."""
        import shapely

        return shapely.union_all([self.taken, self.clear, *([self.turning] if self.turning is not None else [])])


@attrs.frozen(eq=False)
class Layout:
    """One spot per element. Its `layout_id` is a short hash of its spots, in any order."""

    spots: tuple[Spot, ...]

    @classmethod
    def of(cls, spots: Iterable[Spot]) -> Layout:
        spots = tuple(spots)
        names = [s.element.name for s in spots]
        if len(set(names)) != len(names):
            raise ValueError(f"a layout takes one spot per element, got {names}")
        order = list(CATALOG)
        return cls(tuple(sorted(spots, key=lambda s: (order.index(s.element.name) if s.element.name in order else len(order), s.element.name))))

    @property
    def layout_id(self) -> str:
        return hashlib.sha256(json.dumps(sorted(s.id for s in self.spots)).encode()).hexdigest()[:8]

    @property
    def objects(self) -> tuple[PlacedObject, ...]:
        return tuple(o for s in self.spots for o in s.objects)


def placed_elements(layout: Layout) -> list[PlacedElement]:
    """The objects of `layout` as the NAV@r filter takes them (`metrics.navr_filter.PlacedElement`), in layout order."""
    return [PlacedElement(o.name, o.footprint, o.goto, s.element.blocks_floor) for s in layout.spots for o in s.objects]


# --- the site's geometry ---------------------------------------------------------------------------------------------


@attrs.define
class _Context:
    """The geometry of one level of a site that spots are checked against."""

    root: Path
    level_id: str
    level: Level
    wall_segments: list[tuple[Point, Point]]
    walls: shapely.Geometry
    door_spans: shapely.Geometry
    doors: shapely.Geometry
    lifts: shapely.Geometry
    shapes: dict[str, shapely.Polygon]
    floor: shapely.Geometry
    floor_inner: shapely.Geometry
    obstacles: dict[str, shapely.Polygon]
    obstacle_union: shapely.Geometry
    _grid: tuple[np.ndarray, Point] | None = attrs.field(default=None, init=False)

    @property
    def grid(self) -> tuple[np.ndarray, Point]:
        """The floor raster: True where a cell is floor, not wall and no obstacle; and its origin."""
        if self._grid is None:
            grid, origin = attrs.evolve(self.level, objects=[]).render_grid(resolution=RESOLUTION_M, asset_color="black", asset_name_color=None, static_objects=sorted(self.obstacles.items()))
            self._grid = (grid == 255, (float(origin[0]), float(origin[1])))
        return self._grid


#: How many sites keep their geometry (a floor raster each) and spots cached, the least recently used dropped first
CACHED_SITES = 8
_CONTEXTS: dict[tuple[Any, ...], _Context] = {}
_SPOTS: dict[tuple[Any, ...], tuple[Spot, ...]] = {}
_CUT_OFF: dict[tuple[Any, ...], tuple[str, ...]] = {}
_BASES: dict[tuple[Any, ...], Any] = {}
_WALKS: dict[tuple[Any, ...], Any] = {}
#: Walking distances run on this grid (8-connected, up to 8 % longer than straight off the grid axes)
WALK_RESOLUTION_M = 0.25
_SITES: collections.OrderedDict[tuple[Any, ...], None] = collections.OrderedDict()


def clear_cache() -> None:
    """Forget the cached site geometry and spots."""
    _CONTEXTS.clear()
    _SPOTS.clear()
    _CUT_OFF.clear()
    _BASES.clear()
    _WALKS.clear()
    _SITES.clear()


def _site_key(site: Site) -> tuple[Any, ...]:
    """The cache key of `site`, marked as used just now: the caches keep the `CACHED_SITES` sites used last."""
    key = (str(site.root.resolve()), site.key, site.scenario, site.seed)
    _SITES[key] = None
    _SITES.move_to_end(key)
    while len(_SITES) > CACHED_SITES:
        old, _ = _SITES.popitem(last=False)
        for cache in (_CONTEXTS, _SPOTS, _CUT_OFF, _BASES, _WALKS):
            for stale in [k for k in cache if k[: len(old)] == old]:
                del cache[stale]
    return key


def _obstacles(site: Site, level_id: str, ignore: frozenset[str]) -> dict[str, shapely.Polygon]:
    """The footprints of the objects on the level that block floor, by id, except `ignore`. A model that resolves
    without a bounding box blocks nothing (a floor marking, a scan of the whole site such as the twin's `link_scan`);
    one that does not resolve keeps the generator's fallback box; an object over the passage clearance is skipped. The
    docks of docked objects are obstacles too, as `<id>#dock`."""
    import shapely

    from arena_simulation_setup.tree.World.model import _PASSAGE_CLEARANCE

    from . import generate as gen

    raised = {f"obj:{o.name}" for o in site.world.levels[level_id].objects if o.pose.position.z > _PASSAGE_CLEARANCE}
    here: dict[str, tuple[dict[str, Any], str]] = {}
    for obj in gen.objects(site):
        placed = gen._at(site, obj)
        model = gen._model(site, obj)
        if placed is None or placed[0] != level_id or model is None or obj in ignore or obj in raised:
            continue
        here[obj] = (placed[1], model)
    gen.resolve_footprints(site, sorted({model for _, model in here.values()}))
    out = {}
    for obj, (pose, model) in sorted(here.items()):
        if gen._FOOTPRINTS.get((site.root, model)) is None and (site.root, model) not in gen._UNRESOLVED:
            continue
        out[obj] = gen.footprint(site.root, model, float(pose["x"]), float(pose["y"]), float(pose["yaw"]))
    # a dock is floor a robot stands on, whether its object blocks floor or not (a waiting bay)
    for o in site.world.levels[level_id].objects:
        corners = o.dock_polygon()
        if corners is not None and f"obj:{o.name}" not in ignore:
            out[f"obj:{o.name}#dock"] = shapely.Polygon([(p.x, p.y) for p in corners])
    return out


def _context(site: Site, level_id: str, ignore: frozenset[str]) -> _Context:
    import shapely

    from . import generate as gen

    present = frozenset(obj for obj in ignore if obj in site.graph.entities)
    key = (*_site_key(site), level_id, present)
    if key in _CONTEXTS:
        return _CONTEXTS[key]
    level = site.world.levels[level_id]
    segments = [((w.start.x, w.start.y), (w.end.x, w.end.y)) for w in level.all_walls]
    lines = [shapely.LineString(s) for s in segments]
    spans = [shapely.LineString([(d.start.x, d.start.y), (d.end.x, d.end.y)]) for d in level.doors]
    door_spans = shapely.union_all(spans) if spans else shapely.GeometryCollection()
    lifts = [shapely.Polygon([(c.x, c.y) for c in e.cabin_corners()]) for e in level.elevators]
    shapes = {z.name: z.shape() for z in level.zones if len(z.polygon) >= 3}
    floor = shapely.union_all([s.buffer(0) for s in shapes.values()]).buffer(1e-3)
    obstacles = _obstacles(site, level_id, present)
    context = _Context(
        root=site.root,
        level_id=level_id,
        level=level,
        wall_segments=segments,
        walls=shapely.union_all(lines).buffer(gen.WALL_GAP_M) if lines else shapely.Polygon(),
        door_spans=door_spans,
        doors=door_spans.buffer(gen.DOOR_CLEARANCE_M) if spans else shapely.Polygon(),
        lifts=shapely.union_all(lifts) if lifts else shapely.GeometryCollection(),
        shapes=shapes,
        floor=floor,
        floor_inner=floor.buffer(-gen.WALL_GAP_M - 1e-3),
        obstacles=obstacles,
        obstacle_union=shapely.union_all(list(obstacles.values())) if obstacles else shapely.Polygon(),
    )
    _CONTEXTS[key] = context
    return context


def resolve_element(site: Site, element: Element) -> shapely.Polygon:
    """The footprint of one of `element`'s objects in its own frame. ValueError when its model does not resolve, when it
    has no annotated `face` (its front, and so the side its dock is on, would be a guess), or when neither the element
    nor the model's bounding box gives a footprint."""
    import shapely

    from arena_simulation_setup.utils.face import front_angle as annotated_front

    from . import generate as gen

    gen.resolve_footprints(site, [element.model], strict=True)
    face = gen._FACES.get((site.root, element.model))
    if annotated_front(face) is None:
        raise ValueError(f"model {element.model!r} has no face annotation ({face!r}): its front, and so the side its dock is on, is unknown")
    if element.footprint is not None:
        return shapely.Polygon(element.footprint)
    local = gen._FOOTPRINTS.get((site.root, element.model))
    if local is None:
        raise ValueError(f"model {element.model!r} has no bounding box: give element {element.name!r} a footprint")
    return local


def element_named(element: str | Element) -> Element:
    if isinstance(element, Element):
        return element
    if element not in CATALOG:
        raise ValueError(f"no element {element!r} (catalog: {', '.join(CATALOG)})")
    return CATALOG[element]


def cut_off_zones(site: Site, level_id: str, radius: float | None = None) -> tuple[str, ...]:
    """The zones of a level NAV@r cuts off from its main floor at `radius` (default the wheelchair's, 0.325 m): zones of
    at least 2 m^2, traversable on the walls alone, with no cell in the dominant component (`NavrBase.cut_off_zones`),
    on the walls and doors only, so a room behind a narrow passage or up a stair counts whatever furniture stands in
    it. Cached per site."""
    from arena_simulation_setup.metrics.navr_filter import WHEELCHAIR_RADIUS, NavrBase

    radius = WHEELCHAIR_RADIUS if radius is None else float(radius)
    key = (*_site_key(site), level_id, radius)
    if key not in _CUT_OFF:
        _CUT_OFF[key] = _navr_base(site, level_id, radius).cut_off_zones(radius)
    return _CUT_OFF[key]


def _navr_base(site: Site, level_id: str, radius: float) -> Any:  # noqa: ANN401 - a NavrBase
    """The NAV@r base of a level on its walls and doors alone, at `radius`. Cached per site."""
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    key = (*_site_key(site), level_id, radius)
    if key not in _BASES:
        level = site.world.levels[level_id]
        _BASES[key] = NavrBase(level, radii=(radius,), exclude={o.name for o in level.objects})
    return _BASES[key]


def walking_paths(site: Site, level_id: str, radius: float | None = None) -> Any:  # noqa: ANN401 - a search.RobotPaths
    """Walking distances on a level's floor (`search.RobotPaths` on the walls and doors alone, at the wheelchair radius by
    default, on a `WALK_RESOLUTION_M` grid): what the catalog's distance rules measure. Cached per site."""
    from arena_simulation_setup.metrics.navr_filter import WHEELCHAIR_RADIUS

    from .search import RobotPaths

    radius = WHEELCHAIR_RADIUS if radius is None else float(radius)
    key = (*_site_key(site), level_id, radius)
    if key not in _WALKS:
        _WALKS[key] = RobotPaths.from_base(_navr_base(site, level_id, radius), radius, WALK_RESOLUTION_M)
    return _WALKS[key]


def candidate_zones(level: Level, zones: Sequence[str] | None = None, exclude: Sequence[str] = EXCLUDED_ZONES) -> list[str]:
    """The zones of `level` elements may go to: `zones` (default: the generator's allowed zones), less those matching a
    pattern of `exclude`."""
    from . import generate as gen

    allowed = [z.name for z in gen.allowed_zones(level)]
    if zones is not None:
        unknown = sorted(set(zones) - {z.name for z in level.zones})
        if unknown:
            raise ValueError(f"no zones {unknown} on level {level.id!r}")
        allowed = [z for z in allowed if z in zones]
    return [z for z in allowed if not any(fnmatch.fnmatchcase(z, pattern) for pattern in exclude)]


# --- spots -------------------------------------------------------------------------------------------------------------


@attrs.frozen
class _Frame:
    """A wall piece's frame: `start` on the zone edge's line, `u` along it, `n` into the zone."""

    start: Point
    u: Point
    n: Point

    def at(self, along: float, depth: float) -> Point:
        return (self.start[0] + self.u[0] * along + self.n[0] * depth, self.start[1] + self.u[1] * along + self.n[1] * depth)

    def along(self, x: float, y: float) -> float:
        return (x - self.start[0]) * self.u[0] + (y - self.start[1]) * self.u[1]

    def depth(self, x: float, y: float) -> float:
        return (x - self.start[0]) * self.n[0] + (y - self.start[1]) * self.n[1]

    def extent(self, geometry: shapely.Geometry) -> tuple[float, float, float, float]:
        """`geometry`'s along and depth ranges: (along_lo, along_hi, depth_lo, depth_hi)."""
        import shapely

        coords = shapely.get_coordinates(geometry)
        along = [self.along(x, y) for x, y in coords]
        depth = [self.depth(x, y) for x, y in coords]
        return min(along), max(along), min(depth), max(depth)

    def rect(self, along_lo: float, along_hi: float, depth_lo: float, depth_hi: float) -> shapely.Polygon:
        import shapely

        return shapely.Polygon([self.at(along_lo, depth_lo), self.at(along_hi, depth_lo), self.at(along_hi, depth_hi), self.at(along_lo, depth_hi)])


def _placed(polygon: shapely.Polygon, x: float, y: float, yaw: float) -> shapely.Polygon:
    import shapely.affinity

    return shapely.affinity.translate(shapely.affinity.rotate(polygon, yaw, origin=(0, 0), use_radians=True), xoff=x, yoff=y)


def _overlaps(a: shapely.Geometry, b: shapely.Geometry) -> bool:
    return bool(a.intersects(b)) and a.intersection(b).area > AREA_EPS


def _span(local: shapely.Polygon, dock: shapely.Polygon, element: Element) -> float:
    """The length of wall `element` takes: its members' spacing plus the wider of a footprint and a dock, both turned
    so their front faces along the wall's normal (their along-wall extent is their extent across the front)."""
    import shapely

    out = 0.0
    for geometry in (local, dock):
        ys = shapely.get_coordinates(geometry)[:, 1]
        out = max(out, float(ys.max() - ys.min()))
    return (len(element.members) - 1) * element.spacing_m + out


def _members(piece: WallPiece, frame: _Frame, local: shapely.Polygon, front: float, centre: float, count: int, spacing: float) -> list[tuple[float, float, float, bool]]:
    """The poses of a row of `count` objects whose footprints' middles stand `spacing` apart, the row's middle `centre`
    along the piece: turned so their fronts point along the piece's normal, each back at the wall gap from its wall or
    the zone edge, whichever lies further in, as `generate.Spots.against_wall` places an object; and whether a wall
    stands behind each (`generate.backs_onto_wall`). The row's middle is placed once and the objects `spacing` apart
    along their own frame, all to the 0.1 mm world.yaml keeps (`POSITION_DECIMALS`), so a pair's docks meet without
    overlapping on a slanted wall too and a saved layout stands where its spots say."""
    import shapely.affinity

    from arena_simulation_setup.tree.World.write import POSITION_DECIMALS

    from . import generate as gen

    nx, ny = frame.n
    yaw = math.atan2(ny, nx) - front
    yaw = round(math.atan2(math.sin(yaw), math.cos(yaw)), 4) + 0.0
    turned = shapely.affinity.rotate(local, yaw, origin=(0, 0), use_radians=True)
    coords = list(turned.exterior.coords)
    lo = min(vx * frame.u[0] + vy * frame.u[1] for vx, vy in coords)
    hi = max(vx * frame.u[0] + vy * frame.u[1] for vx, vy in coords)
    back = max(-(vx * nx + vy * ny) for vx, vy in coords)
    # the objects' own front and along-wall axes, from the rounded yaw, the latter pointing the way the piece runs
    fx, fy = math.cos(yaw + front), math.sin(yaw + front)
    ax, ay = (-fy, fx) if -fy * frame.u[0] + fx * frame.u[1] >= 0 else (fy, -fx)
    shifts = [(i - (count - 1) / 2.0) * spacing for i in range(count)]
    offsets = [piece.offset(*frame.at(centre + shift - (lo + hi) / 2.0, 0.0)) for shift in shifts]
    depths = [max(offset or 0.0, 0.0) + gen.WALL_GAP_M + gen.WALL_SLACK_M + back for offset in offsets]
    middle = max(depths)
    rx, ry = frame.at(centre - (lo + hi) / 2.0, middle)
    out = []
    for shift, offset, depth in zip(shifts, offsets, depths, strict=True):
        x = round(rx + ax * shift + fx * (depth - middle), POSITION_DECIMALS) + 0.0
        y = round(ry + ay * shift + fy * (depth - middle), POSITION_DECIMALS) + 0.0
        out.append((x, y, yaw, offset is not None and gen.backs_onto_wall(piece, turned, x, y)))
    return out


def _corridor_width(ctx: _Context, frame: _Frame, blocked: shapely.Geometry, along: tuple[float, float], front: float, others: shapely.Geometry | None = None, zone: shapely.Geometry | None = None) -> float:
    """The clear corridor width left in front of `blocked` (and `others`, painted as obstacles too), up to
    `CORRIDOR_CAP_M`, to about a raster cell: the free run in front of it, on rays from `blocked`'s front along the
    normal every 0.25 m across its width, each running to the first obstacle on the floor raster or out of `zone` (the
    room or corridor the element stands in, so a ray through a doorway stops at the door); the lower median ray's, a run
    more than half of the rays find. An obstacle beside the element, or in front of a few of its rays, leaves the
    passage in front of it as wide as the other rays find it; one across half its width or more narrows it."""
    import numpy as np
    import shapely

    grid, (ox, oy) = ctx.grid
    height, width = grid.shape
    res = RESOLUTION_M
    cap = CORRIDOR_CAP_M
    lo, hi = along
    x0, y0, x1, y1 = _window(frame, along, front).bounds
    c0, c1 = max(0, math.floor((x0 - ox) / res)), min(width, math.ceil((x1 - ox) / res))
    r0, r1 = max(0, height - math.ceil((y1 - oy) / res)), min(height, height - math.floor((y0 - oy) / res))
    if c1 <= c0 or r1 <= r0:
        return 0.0
    free = grid[r0:r1, c0:c1].copy()
    if zone is not None:
        xs = ox + (np.arange(c0, c1) + 0.5) * res
        ys = oy + (height - 1 - np.arange(r0, r1) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys)
        free &= shapely.contains_xy(zone.buffer(res / 2.0), xx, yy)
    paint = blocked if others is None or others.is_empty else shapely.union_all([blocked, others])
    bx0, by0, bx1, by1 = paint.bounds
    pc0, pc1 = max(c0, math.floor((bx0 - ox) / res)), min(c1, math.ceil((bx1 - ox) / res))
    pr0, pr1 = max(r0, height - math.ceil((by1 - oy) / res)), min(r1, height - math.floor((by0 - oy) / res))
    if pc1 > pc0 and pr1 > pr0:
        xs = ox + (np.arange(pc0, pc1) + 0.5) * res
        ys = oy + (height - 1 - np.arange(pr0, pr1) + 0.5) * res
        xx, yy = np.meshgrid(xs, ys)
        free[pr0 - r0 : pr1 - r0, pc0 - c0 : pc1 - c0] &= ~shapely.contains_xy(paint, xx, yy)
    rays = np.linspace(lo + res, hi - res, max(2, math.ceil((hi - lo) / 0.25) + 1))
    depths = np.arange(front + res, front + cap + res, res)
    runs = []
    for a in rays:
        px = frame.start[0] + frame.u[0] * a + frame.n[0] * depths
        py = frame.start[1] + frame.u[1] * a + frame.n[1] * depths
        cols = np.floor((px - ox) / res).astype(int) - c0
        rows = height - 1 - np.floor((py - oy) / res).astype(int) - r0
        inside = (rows >= 0) & (rows < free.shape[0]) & (cols >= 0) & (cols < free.shape[1])
        open_ = np.zeros(len(depths), dtype=bool)
        open_[inside] = free[rows[inside], cols[inside]]
        run = len(depths) if open_.all() else int(np.argmin(open_))
        # the first blocked sample's cell starts half a cell before its centre, on average
        runs.append(cap if run == len(depths) else run * res + res / 2.0)
    # the lower median: a run more than half of the rays find, never the mean of a short and a long one
    return round(min(float(np.sort(runs)[(len(runs) - 1) // 2]), cap), 3)


def _window(frame: _Frame, along: tuple[float, float], front: float) -> shapely.Polygon:
    """The floor `_corridor_width` reads for an element spanning `along` with its front at `front`: wide enough that
    every distance under `CORRIDOR_CAP_M / 2` from a ray cell is measured, not cut off by the window."""
    res = RESOLUTION_M
    return frame.rect(along[0] - CORRIDOR_CAP_M / 2.0 - res, along[1] + CORRIDOR_CAP_M / 2.0 + res, front - CORRIDOR_CAP_M / 2.0, front + 1.5 * CORRIDOR_CAP_M + res)


def _verdicts(ctx: _Context, element: Element, objects: Sequence[PlacedObject], against: bool, clear: shapely.Polygon, turning: shapely.Polygon | None, corridor: float, corridor_min_m: float, peds: Sequence[shapely.Polygon] | None) -> tuple[tuple[Rule, Verdict], ...]:
    import shapely

    from . import generate as gen

    def ok(condition: bool) -> Verdict:
        return Verdict.PASS if condition else Verdict.FAIL

    def free_floor(geometry: shapely.Geometry) -> bool:
        return bool(ctx.floor.covers(geometry)) and not ctx.walls.intersects(geometry) and not ctx.obstacle_union.intersects(geometry)

    footprints = [o.footprint for o in objects]
    docks = [o.dock for o in objects]
    out: dict[Rule, Verdict] = {}
    out[Rule.AGAINST_WALL] = ok(against)
    out[Rule.NO_OVERLAP] = ok(
        all(ctx.floor_inner.covers(f) and not ctx.walls.intersects(f) and not ctx.doors.intersects(f) and not ctx.obstacle_union.intersects(f.buffer(gen.MARGIN_M)) for f in footprints)
        and not any(_overlaps(a, b) for i, a in enumerate(footprints) for b in footprints[i + 1 :])
    )
    out[Rule.CLEAR_ZONE] = ok(clear.is_empty or (free_floor(clear) and not ctx.doors.intersects(clear)))
    out[Rule.TURNING_CIRCLE] = ok(turning is None or free_floor(turning))
    out[Rule.CORRIDOR_WIDTH] = ok(corridor >= corridor_min_m)
    grid, (ox, oy) = ctx.grid

    def on_free_cell(point: Point) -> bool:
        row = grid.shape[0] - 1 - math.floor((point[1] - oy) / RESOLUTION_M)
        col = math.floor((point[0] - ox) / RESOLUTION_M)
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1] and bool(grid[row, col])

    out[Rule.DOCKS_ON_FLOOR] = ok(all(free_floor(d) for d in docks) and all(on_free_cell(o.goto) for o in objects))
    taken = shapely.union_all([*footprints, *docks, clear])
    out[Rule.PEDS_ZONES] = Verdict.UNKNOWN if peds is None else ok(not any(_overlaps(taken, zone) for zone in peds))
    near = shapely.union_all([*footprints, *docks])
    distance = element.door_distance_m if element.door_distance_m is not None else DOOR_DISTANCE_M
    out[Rule.DOOR_DISTANCE] = Verdict.UNKNOWN if ctx.door_spans.is_empty else ok(near.distance(ctx.door_spans) >= distance)
    out[Rule.LIFT_DISTANCE] = Verdict.UNKNOWN if ctx.lifts.is_empty else ok(near.distance(ctx.lifts) >= distance)
    for rule in (*NO_DATA_RULES, *FACT_RULES):
        out[rule] = Verdict.UNKNOWN  # the fact rules get theirs from the site's facts (`_with_facts`)
    return tuple((rule, out[rule]) for rule in Rule if rule in element.rules)


def _with_facts(site: Site, level_id: str, element: Element, spots: list[Spot], facts: SiteFacts | None) -> list[Spot]:
    """`spots` with the verdicts of the rules that read the site's facts: the courier access point (a spot's goto point
    within `COURIER_ACCESS_M` of it), recipient entrances (walking distance from one, on this level, to a spot's goto
    point within the element's `recipient_distance_m`) and outlets (every object's footprint centre within
    `OUTLET_CABLE_M` walked from an outlet). Unknown where the facts give nothing for a rule."""
    import numpy as np

    rules = [rule for rule in FACT_RULES if rule in element.rules]
    if facts is None or not spots or not rules:
        return spots
    verdicts: dict[Rule, list[Verdict]] = {}
    if Rule.COURIER_ACCESS in rules and facts.courier_access is not None:
        ax, ay = facts.courier_access
        verdicts[Rule.COURIER_ACCESS] = [Verdict.PASS if math.hypot(s.objects[0].goto[0] - ax, s.objects[0].goto[1] - ay) <= COURIER_ACCESS_M else Verdict.FAIL for s in spots]
    entrances = [e.anchor for e in facts.recipient_entrances if e.level in (None, level_id)]
    if Rule.RECIPIENT_DISTANCE in rules and element.recipient_distance_m is not None and entrances:
        lengths = walking_paths(site, level_id).lengths(entrances, [s.objects[0].goto for s in spots]).min(axis=0)
        verdicts[Rule.RECIPIENT_DISTANCE] = [Verdict.PASS if d <= element.recipient_distance_m else Verdict.FAIL for d in lengths.tolist()]
    if Rule.OUTLET_CABLE in rules and facts.outlets:
        centres = [(float(o.footprint.centroid.x), float(o.footprint.centroid.y)) for s in spots for o in s.objects]
        runs = walking_paths(site, level_id).lengths(list(facts.outlets), centres).min(axis=0).reshape(len(spots), -1).max(axis=1)
        verdicts[Rule.OUTLET_CABLE] = [Verdict.PASS if d <= OUTLET_CABLE_M else Verdict.FAIL for d in np.asarray(runs).tolist()]
    if not verdicts:
        return spots
    return [attrs.evolve(s, verdicts=tuple((rule, verdicts[rule][i] if rule in verdicts else v) for rule, v in s.verdicts)) for i, s in enumerate(spots)]


def _spot(ctx: _Context, element: Element, local: shapely.Polygon, front: float, zone: str, index: int, piece: WallPiece, step: int, centre: float, corridor_min_m: float, peds: Sequence[shapely.Polygon] | None) -> Spot:
    import shapely
    from shapely.ops import polylabel

    frame = _Frame(tuple(piece.line.coords[0]), piece.direction, piece.normal)  # type: ignore[arg-type]
    dock_local = shapely.Polygon(element.dock)
    objects: list[PlacedObject] = []
    against = True
    poses = _members(piece, frame, local, front, centre, len(element.members), element.spacing_m)
    for name, (x, y, yaw, backed) in zip(element.members, poses, strict=True):
        against = against and backed
        dock = _placed(dock_local, x, y, yaw)
        goto = polylabel(dock, tolerance=0.01)
        objects.append(PlacedObject(name, element.model, x, y, yaw, _placed(local, x, y, yaw), dock, (round(goto.x, 3), round(goto.y, 3))))
    footprints = shapely.union_all([o.footprint for o in objects])
    docks = shapely.union_all([o.dock for o in objects])
    lo, hi, _, _ = frame.extent(shapely.union_all([footprints, docks]))
    fp_lo, fp_hi, _, fp_front = frame.extent(footprints)
    dock_front = frame.extent(docks)[3]
    # the clear zone runs from the footprints' front over the clear depth and the docks, as wide as both
    far = max(fp_front + element.clear_m, dock_front) if element.clear_m > 0 else fp_front
    clear = frame.rect(lo, hi, fp_front, far) if far > fp_front + 1e-6 else shapely.Polygon()
    turning = None
    if element.turning_circle_m > 0:
        # beyond the docks: a wheelchair turns where no robot stands in a dock
        radius = element.turning_circle_m / 2.0
        turning = shapely.Point(frame.at((fp_lo + fp_hi) / 2.0, max(fp_front, dock_front) + radius)).buffer(radius, quad_segs=16)
    blocked = shapely.union_all([footprints, docks, clear])
    corridor = _corridor_width(ctx, frame, blocked, (lo, hi), max(far, dock_front, fp_front), zone=ctx.shapes.get(zone))
    verdicts = _verdicts(ctx, element, objects, against, clear, turning, corridor, corridor_min_m, peds)
    return Spot(element, ctx.level_id, zone, index, step, round(centre, 3), round(piece.line.length, 3), frame, tuple(objects), clear, turning, corridor, verdicts)


def candidate_spots(
    site: Site,
    element: str | Element,
    *,
    zones: Sequence[str] | None = None,
    exclude: Sequence[str] = EXCLUDED_ZONES,
    cut_off: bool = True,
    peds_zones: Sequence[str] | None = None,
    step_m: float = STEP_M,
    corridor_min_m: float = CORRIDOR_MIN_M,
    facts: SiteFacts | None = None,
) -> tuple[Spot, ...]:
    """Every candidate spot of `element` on `site`, the same ones for the same site and arguments (and cached): on each
    level, in each zone of `candidate_zones(zones, exclude)` by name, less the zones NAV@r cuts off at the wheelchair
    radius (`cut_off_zones`; `cut_off=False` keeps them), along each wall piece in ring order, every
    `step_m` where the element fits on the piece, the steps centred on it. Spots that fail a rule are kept, with their
    verdicts; `Spot.valid` tells them apart. `peds_zones` names the PEDS source and sink zones; until given, that rule
    is unknown. `facts` are the site's (`SiteFacts`); the rules that read them are unknown without. The element's own
    objects, if the site has them, are no obstacles to it (a move)."""
    from . import generate as gen

    element = element_named(element)
    key = (*_site_key(site), element, None if zones is None else tuple(zones), tuple(exclude), cut_off, None if peds_zones is None else tuple(peds_zones), step_m, corridor_min_m, facts)
    if key in _SPOTS:
        return _SPOTS[key]
    local = resolve_element(site, element)
    front = gen.front_angle(site.root, element.model)
    turned_dock = _turned(element.dock, -front)
    span = _span(_turned(tuple(local.exterior.coords)[:-1], -front), turned_dock, element)
    ignore = frozenset(f"obj:{name}" for name in element.members)
    if peds_zones is not None:
        unknown = sorted(set(peds_zones) - {z.name for level in site.world.levels.values() for z in level.zones})
        if unknown:
            raise ValueError(f"no zones {unknown} for the PEDS zones")
    out: list[Spot] = []
    for level_id, level in site.world.levels.items():
        names = candidate_zones(level, zones, exclude)
        if names and cut_off:
            gone = set(cut_off_zones(site, level_id))
            names = [z for z in names if z not in gone]
        if not names:
            continue
        ctx = _context(site, level_id, ignore)
        peds = None if peds_zones is None else [ctx.shapes[name] for name in peds_zones if name in ctx.shapes]
        here: list[Spot] = []
        for zone in names:
            for index, piece in enumerate(gen.wall_pieces(ctx.shapes[zone], ctx.wall_segments)):
                length = piece.line.length
                if length + 1e-9 < span:
                    continue
                count = math.floor((length - span) / step_m + 1e-9) + 1
                first = span / 2.0 + (length - span - (count - 1) * step_m) / 2.0
                here += [_spot(ctx, element, local, front, zone, index, piece, k, first + k * step_m, corridor_min_m, peds) for k in range(count)]
        out += _with_facts(site, level_id, element, here, facts)
    spots = tuple(out)
    _SPOTS[key] = spots
    return spots


def _turned(points: Sequence[Point], angle: float) -> shapely.Polygon:
    """`points` as a polygon turned by `angle` about the origin, so a model's front (at `-angle`) points along +x."""
    import shapely
    import shapely.affinity

    return shapely.affinity.rotate(shapely.Polygon(points), angle, origin=(0, 0), use_radians=True)


# --- layouts -----------------------------------------------------------------------------------------------------------


def check_layout(site: Site, layout: Layout, *, corridor_min_m: float = CORRIDOR_MIN_M) -> list[str]:
    """Why `layout` is no valid layout on `site`, empty when it is: a spot with a failing rule, two elements whose
    floor overlaps (`Spot.taken` of one in `Spot.keep_clear` of another, or a footprint within the generator's margin
    of the other's footprints and docks, as `no_overlap` keeps it from objects), an object name taken twice, or a
    corridor that the other elements narrow below `corridor_min_m`. Objects of the layout's elements that `site`
    already has are no obstacles here: the layout places them anew."""
    import shapely

    from . import generate as gen

    def footprints(spot: Spot) -> shapely.Geometry:
        return shapely.union_all([o.footprint for o in spot.objects]).buffer(gen.MARGIN_M)

    problems = [f"{s.element.name} at {s.zone} piece {s.piece} step {s.step}: {', '.join(s.failed)} fails" for s in layout.spots if s.failed]
    names = [o.name for o in layout.objects]
    problems += [f"object {name!r} is placed twice" for name in sorted({n for n in names if names.count(n) > 1})]
    for i, a in enumerate(layout.spots):
        for b in layout.spots[i + 1 :]:
            if a.level == b.level and (_overlaps(a.taken, b.keep_clear) or _overlaps(b.taken, a.keep_clear) or _overlaps(footprints(a), b.taken) or _overlaps(footprints(b), a.taken)):
                problems.append(f"{a.element.name} and {b.element.name} overlap")
    placed = frozenset(f"obj:{o.name}" for o in layout.objects)
    for s in layout.spots:
        others = [o for o in layout.spots if o is not s and o.level == s.level]
        if not others:
            continue
        blocked = shapely.union_all([s.taken, s.clear])
        lo, hi, _, front = s.frame.extent(blocked)
        near = shapely.union_all([shapely.union_all([o.taken, o.clear]) for o in others])
        if not near.intersects(_window(s.frame, (lo, hi), front)):
            continue
        ctx = _context(site, s.level, placed)
        width = _corridor_width(ctx, s.frame, blocked, (lo, hi), front, near, zone=ctx.shapes.get(s.zone))
        if width < corridor_min_m:
            problems.append(f"{s.element.name}: the other elements leave a {width:.2f} m corridor, under {corridor_min_m} m")
    return problems


def insert_params(spot: Spot, obj: PlacedObject) -> dict[str, Any]:
    """The `insert` edit params of one object of `spot`: model, pose, level, dock and a charger's charge power."""
    element = spot.element
    params: dict[str, Any] = {"model": element.model, "x": obj.x, "y": obj.y, "yaw": obj.yaw, "level": spot.level, "dock": [[float(x), float(y)] for x, y in element.dock]}
    if element.charge_power_w is not None:
        params["charge_power_w"] = float(element.charge_power_w)
    return params


def _as_catalog(site: Site, obj: str, element: Element) -> bool:
    """Whether the world object `obj` of `site` has `element`'s model, dock and charge power."""
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World.write import short_identifier

    found = world_object(site.world, obj.partition(":")[2])
    if found is None:
        return False
    placed = found[0].objects[found[1]]
    model = Obstacle.parse({"name": "probe", "model": element.model, "pose": [0.0, 0.0, 0.0]}).model
    if short_identifier(placed.model) != short_identifier(model) or placed.charge_power_w != element.charge_power_w:
        return False
    dock = [(p.x, p.y) for p in placed.dock or []]
    return len(dock) == len(element.dock) and all(math.dist(a, b) < 1e-6 for a, b in zip(dock, element.dock, strict=True))


def apply_layout(site: Site, layout: Layout, *, check: bool = True) -> Site:
    """`site` with every object of `layout` in place, in memory: a `move` edit for an object the site has as the
    catalog has it (model, dock, charge power), else an `insert` edit with the catalog's (after a `remove` edit of the
    site's object of that name, if any). ValueError for an invalid layout (`check_layout`) unless not `check`."""
    if check:
        problems = check_layout(site, layout)
        if problems:
            raise ValueError(f"layout {layout.layout_id}: " + "; ".join(problems))
    for spot in layout.spots:
        for obj in spot.objects:
            target = f"obj:{obj.name}"
            if target in site.graph.entities and _as_catalog(site, target, spot.element):
                site = apply(site, EditType.MOVE, target, {"x": obj.x, "y": obj.y, "yaw": obj.yaw, "level": spot.level})
                continue
            if target in site.graph.entities:
                site = apply(site, EditType.REMOVE, target)
            site = apply(site, EditType.INSERT, target, insert_params(spot, obj))
    return site


def remove_element(site: Site, element: str | Element) -> Site:
    """`site` without the objects of `element` it has, one `remove` edit each."""
    for name in element_named(element).members:
        if f"obj:{name}" in site.graph.entities:
            site = apply(site, EditType.REMOVE, f"obj:{name}")
    return site


@attrs.define
class _NamedSite(Site):
    """A site that `edit.save` writes under `world_name` instead of the name its edits give it."""

    world_name: str = ""

    @property
    def name(self) -> str:
        return self.world_name


def layout_prefix(site: Site) -> str:
    """How the world name of every layout copy of `site` starts: `<site key>__layout_`."""
    return f"{site.key}{SEPARATOR}layout_"


def layout_name(site: Site, layout: Layout) -> str:
    """The default world name of `layout` applied to `site`: `<site key>__layout_<layout id>`."""
    return f"{layout_prefix(site)}{layout.layout_id}"


def save_layout(site: Site, layout: Layout, name: str | None = None, *, check: bool = True) -> Path:
    """Apply `layout` to `site` (`apply_layout`) and save the world copy through `edit.save` under `name` (default
    `layout_name`), its assets hardlinked; the path it was written to. `edit.save` replaces a world of the same name in
    the writable local root, so `name` must start with `layout_prefix(site)`: a layout copy only ever replaces another
    layout copy of the same site, never the site itself or another world."""
    name = name or layout_name(site, layout)
    prefix = layout_prefix(site)
    if not name.startswith(prefix) or name == prefix:
        raise ValueError(f"a layout copy of {site.key} is named {prefix}<suffix>, not {name!r}: saving replaces any world of that name")
    edited = apply_layout(site, layout, check=check)
    fields = {f.name: getattr(edited, f.name) for f in attrs.fields(Site)}
    return save(_NamedSite(**fields, world_name=name))


__all__ = [
    "CATALOG",
    "CHARGE_POWER_W",
    "CORRIDOR_MIN_M",
    "COURIER_ACCESS_M",
    "ENTRANCE_TYPES",
    "EXCLUDED_ZONES",
    "FACT_RULES",
    "OUTLET_CABLE_M",
    "RECIPIENT_DISTANCE_M",
    "STEP_M",
    "Element",
    "Entrance",
    "Layout",
    "PlacedElement",
    "PlacedObject",
    "Rule",
    "SiteFacts",
    "Spot",
    "Verdict",
    "apply_layout",
    "candidate_spots",
    "candidate_zones",
    "cut_off_zones",
    "check_layout",
    "clear_cache",
    "insert_params",
    "layout_name",
    "layout_prefix",
    "placed_elements",
    "remove_element",
    "resolve_element",
    "save_layout",
    "walking_paths",
]
