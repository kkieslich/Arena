"""INFRA layout searches for demand at the recipient entrances (study design Part A: A3, A5, A6).

Deliveries are addressed to recipient entrances: every entrance of a site of type building or lift
(`recipient_entrances`), each with the same demand. A layout places the charger pair, the bay pair and several lockers:
a locker spot is a candidate when the catalog's rules hold (`locker_element`) and it lies within the reach of at least
one recipient entrance as people walk, and a layout has the fewest lockers that give every recipient entrance a locker
within the reach (`EntranceSearch.cover`). Each entrance is served by the locker of the layout nearest to it as people
walk. A delivery runs from the loading point to that locker, a pickup back the same way.

Two distances run from a recipient entrance to a locker spot (`Walks`): as people walk, on the floor and over the
site's stairs and escalators (the site config's `links`), and step-free, on the floor alone, which is what wheelchair
users take.

`EntranceSearch` holds the search space on top of a `search.Search` (its spots, pair checks, robot paths and
occupancy terms) and scores a layout by

- robot travel per job (`Demand`): the loading point to locker leg of the entrance the job is addressed to, the drives
  from and back to the robot's waiting spot (a bay slot or a charger), and charging trips between a bay slot and a
  charger at a rate given as an input;
- predicted exposure per walker group, the first study's term (`search.Search.conflict_parts`): the time a member
  spends within the conflict band of a driving robot, over the same legs, plus the time on an element's floor;
- the walk to the locker per walker group: over the recipient entrances, the mean and the longest path to the locker
  that serves the entrance, as the group walks.

The picks: `travel_optimal` (least robot travel per job over all valid layouts), `pedestrian_aware` (among the valid
layouts within a travel cap, the least predicted cost of the worst-off group, `Cost`) and `practice_based` (waiting
bays nearest the loading point, a locker nearest each group of recipient entrances, chargers nearest the bays).

With several lockers the layouts are too many to list (charger spots x bay spots x locker spots to the power of the
lockers). The search stays exact without listing them, because every score is a sum over the lockers: a term of the
charger and bay spots plus, per locker, terms of that locker's spot with the bay spot, with the charger spot and with
the entrances it serves (`_Channel`). The lockers of a layout split the entrances into groups, one per locker
(`Cover.partitions`: the few ways to split them so that one spot can serve each group). For one way to split them, one
bay spot and every charger spot at once, the least value of each locker group gives a lower bound; only charger and
bay spots whose bound beats the best layout found so far are solved exactly, over the locker spots that can still
beat it (`EntranceSearch.best`). Every pick passes the full check (`place.check_layout` and the NAV@r filter)."""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import attrs

from . import place
from .place import Entrance, Layout, Spot
from .search import CONFLICT_BAND_M, GROUPS, PATH_SNAP_M, ROBOT_SPEED_MPS, SEARCHED, Checked, Occupancy, RobotPaths, Search, _num, _reach, approach_area

if TYPE_CHECKING:
    import numpy as np

    from arena_simulation_setup.metrics.navr_filter import NavrVerdict

CHARGERS, BAYS, LOCKER = SEARCHED
#: The entrance types parcel recipients come in by (A3): every building door and every lift of the site zone map
RECIPIENT_TYPES = ("building", "lift")
#: The reach: every recipient entrance has a locker within this walk as people walk [m]. No source fixes it: the
#: reference value, and the values the study sweeps (A3)
REACH_M = 30.0
REACHES_M = (20.0, 25.0, 30.0, 40.0, 50.0)
#: The travel margins of the pedestrian-aware search: robot travel per job at most this times the travel-optimal
#: layout's (A2)
MARGINS = (1.05, 1.10, 1.25)
#: The link kinds with steps: walkers take them, wheelchair users do not (HumanSim's `climbs`)
STEP_KINDS = ("stairs", "escalator")
#: The speed on a link with steps, for the walk in seconds: on a stair flight and on an escalator's belt (the twin's
#: site config) [m/s]
LINK_SPEED_MPS = {"stairs": 0.7, "escalator": 0.5}
#: The walking speed of each walker group on the floor, for the walk in seconds: HumanSim's desired speeds of the agent
#: types adult, wheelchair_manual and blind_white_cane [m/s]
WALK_SPEED_MPS = {"adult": 1.1, "wheelchair": 0.9, "long_cane": 0.75}
#: Seconds of delay per second of exposure at the 2 m conflict band, per walker group, for the pooled cost
#: (`Cost.pooled`): the first study's calibration (relative delay per share of time within 1 m of a robot: adult 1.22,
#: wheelchair 2.19, long cane 1.29) times the 1 m exposure over the 2 m exposure of the twin's travel-optimal layout at
#: a reach of 40 m (0.38, 0.31, 0.42). A weak calibration and a default only: an input, not a constant of the method
DELAY_PER_EXPOSURE = {"adult": 0.47, "wheelchair": 0.68, "long_cane": 0.55}
#: The walker groups that take step-free floor only
STEP_FREE = ("wheelchair",)
#: The share of the robots whose waiting spot is a bay slot, the others wait at a charger (`search.HOMES`: 2 and 2)
BAY_SHARE = 0.5
#: Locker spot combinations one exact solve looks at in one array at most
MAX_PRODUCT = 2_000_000
#: Locker sets the practice-based placement tries per bay spot before it moves to the next bay spot
MAX_SETS = 2_000
_EPS = 1e-9


def recipient_entrances(facts: place.SiteFacts, types: Sequence[str] = RECIPIENT_TYPES, level: str | None = None) -> tuple[Entrance, ...]:
    """The recipient entrances of a site by rule (A3): every entrance whose type is one of `types`, in the facts' order;
    with `level`, those on that level or on every level."""
    return tuple(e for e in facts.entrances if e.type in types and (level is None or e.level in (None, level)))


def locker_element(*, outlet: bool = True) -> place.Element:
    """The catalog's locker for demand at the recipient entrances: its rules without the first study's recipient
    distance rule (one fixed distance on the step-free floor to an entrance marked `recipients`), which the reach as
    people walk replaces (`EntranceSearch.cover`). `outlet=False` also drops the outlet cable rule, for a site whose
    outlet marks are in doubt; the default keeps every other rule."""
    locker = place.CATALOG[LOCKER]
    drop = {place.Rule.RECIPIENT_DISTANCE} | (set() if outlet else {place.Rule.OUTLET_CABLE})
    return attrs.evolve(locker, rules=tuple(rule for rule in locker.rules if rule not in drop), recipient_distance_m=None)


def numbered(spot: Spot, number: int) -> Spot:
    """A locker spot as locker `number` of a layout (from 1): the first keeps the catalog's names (element `locker`,
    object `locker_1`, so a layout with one locker has the id the first study's search gives it), the others are
    element and object `locker_<number>`."""
    if number == 1:
        return spot
    name = f"{LOCKER}_{number}"
    return attrs.evolve(spot, element=attrs.evolve(spot.element, name=name, members=(name,)), objects=tuple(attrs.evolve(o, name=name) for o in spot.objects))


@attrs.frozen
class Demand:
    """The job stream robot travel and exposure are weighed with (A3, A5): `rate_per_h` jobs per hour in all, the same
    share addressed to every recipient entrance, a share `pickup_share` of them pickups (locker to loading point, the
    same legs as a delivery the other way). A free robot waits at its waiting spot, a bay slot for a share `bay_share`
    of the robots and a charger for the others, so a job drives from the waiting spot to the first end, between the
    ends and back from the second end. `charging_trips_per_job` drives between a bay slot and a charger are added per
    job, one way each (the rate of the fleet-only run: charging trips dispatched per completed job)."""

    rate_per_h: float = 12.0
    pickup_share: float = 0.3
    bay_share: float = BAY_SHARE
    charging_trips_per_job: float = 0.0

    def __attrs_post_init__(self) -> None:
        if self.rate_per_h <= 0:
            raise ValueError(f"rate_per_h must be positive, got {self.rate_per_h}")
        if not 0.0 <= self.pickup_share <= 1.0 or not 0.0 <= self.bay_share <= 1.0:
            raise ValueError(f"pickup_share {self.pickup_share} and bay_share {self.bay_share} must lie in [0, 1]")
        if self.charging_trips_per_job < 0:
            raise ValueError(f"charging_trips_per_job must not be negative, got {self.charging_trips_per_job}")

    @property
    def deliveries_per_h(self) -> float:
        """Deliveries per hour: each brings one recipient, who walks from the entrance to the locker and back (A3)."""
        return self.rate_per_h * (1.0 - self.pickup_share)


@attrs.frozen(eq=False)
class Walks:
    """Walking distances on one level: `floor` (the step-free walking floor, `place.walking_paths`) and the site's
    links as directed edges between the floor cells nearest their entries, (from cell, to cell, metres, kind, delay):
    a link half leads to each of its `destinations`, its length the distance between the two entries. Walkers take
    every link; a link whose kind has steps (`STEP_KINDS`) is left out of the step-free distances."""

    floor: RobotPaths
    links: tuple[tuple[int, int, float, str, float], ...] = ()
    _graphs: dict[Any, Any] = attrs.field(factory=dict, init=False, repr=False)

    @classmethod
    def of(cls, floor: RobotPaths, links: Iterable[Mapping[str, Any]] = ()) -> Walks:
        """The walks on `floor` with `links` (teleport zones in HumanSim's form, as the site config's `links`).
        ValueError for a destination that is no link, or an entry further than `PATH_SNAP_M` from the floor."""
        links = [dict(z) for z in links]
        by_name = {str(z.get("name")): z for z in links}
        edges = []
        for z in links:
            for dest in z.get("destinations") or []:
                if dest not in by_name:
                    raise ValueError(f"link {z.get('name')!r} leads to {dest!r}, which is no link")
                a = (float(z["entry"]["x"]), float(z["entry"]["y"]))
                b = (float(by_name[dest]["entry"]["x"]), float(by_name[dest]["entry"]["y"]))
                (da, ia), (db, ib) = floor.tree.query(a), floor.tree.query(b)
                if max(da, db) > PATH_SNAP_M:
                    raise ValueError(f"link {z.get('name')!r} to {dest!r}: an entry lies {max(da, db):.2f} m from the walking floor")
                edges.append((int(ia), int(ib), math.dist(a, b) + float(da) + float(db), str(z.get("kind") or ""), float(z.get("delay") or 0.0)))
        return cls(floor, tuple(edges))

    def _graph(self, climbs: bool, speed_mps: float | None) -> Any:  # noqa: ANN401 - a scipy sparse matrix
        key = (climbs, speed_mps)
        if key not in self._graphs:
            import scipy.sparse

            graph = (self.floor.graph + self.floor.graph.T).tocsr()
            if speed_mps is not None:
                graph = graph / speed_mps
            edges: dict[tuple[int, int], float] = {}
            for head, tail, metres, kind, delay in self.links:
                if kind in STEP_KINDS and not climbs:
                    continue
                if speed_mps is None:
                    weight = metres
                else:
                    weight = metres / LINK_SPEED_MPS[kind] if kind in LINK_SPEED_MPS else (delay if delay > 0 else metres / speed_mps)
                edges[(head, tail)] = min(weight, edges.get((head, tail), math.inf))
            if edges:
                heads, tails = zip(*edges, strict=True)
                graph = graph + scipy.sparse.csr_matrix((list(edges.values()), (heads, tails)), shape=graph.shape)
            self._graphs[key] = graph
        return self._graphs[key]

    def _run(self, sources: Sequence[place.Point], targets: Sequence[place.Point], climbs: bool, speed_mps: float | None) -> np.ndarray:
        import numpy as np
        from scipy.sparse.csgraph import dijkstra

        src_d, src_i = self.floor.tree.query(np.asarray(sources, dtype=float).reshape(-1, 2))
        dst_d, dst_i = self.floor.tree.query(np.asarray(targets, dtype=float).reshape(-1, 2))
        if not len(src_i) or not len(dst_i):
            return np.zeros((len(src_i), len(dst_i)))
        scale = 1.0 if speed_mps is None else 1.0 / speed_mps
        out = np.atleast_2d(dijkstra(self._graph(climbs, speed_mps), directed=True, indices=src_i))[:, dst_i] + scale * (src_d[:, None] + dst_d[None, :])
        out[src_d > PATH_SNAP_M, :] = np.inf
        out[:, dst_d > PATH_SNAP_M] = np.inf
        return out

    def lengths(self, sources: Sequence[place.Point], targets: Sequence[place.Point], *, climbs: bool) -> np.ndarray:
        """The shortest walk from each of `sources` to each of `targets` [m], (len(sources), len(targets)), inf where
        none: over the links with steps too when `climbs`, else step-free."""
        return self._run(sources, targets, climbs, None)

    def seconds(self, sources: Sequence[place.Point], targets: Sequence[place.Point], speed_mps: float, *, climbs: bool) -> np.ndarray:
        """The quickest walk from each of `sources` to each of `targets` [s] at `speed_mps` on the floor, a link with
        steps at its kind's speed (`LINK_SPEED_MPS`), another link at its `delay`."""
        if speed_mps <= 0:
            raise ValueError(f"speed_mps must be positive, got {speed_mps}")
        return self._run(sources, targets, climbs, float(speed_mps))


def leg_integrals(search: Search, lockers: Sequence[Spot], values: np.ndarray) -> dict[str, np.ndarray]:
    """The legs of the job cycle for the locker spots `lockers`, on the robots' paths (`search.paths`): path lengths
    `m_<leg>` and the path integrals `x_<leg>` of each row of `values` (one value per path cell, shape (rows, cells);
    a leading axis of that length). Legs: `L_charger` (C,) and `L_bay` (B,) from the loading point to each charger and
    bay spot (the mean over the spot's objects), `LK` (K,) from the loading point to each locker spot, `K_charger`
    (K, C) and `K_bay` (K, B) from each locker spot, `BC` (C, B) between each charger spot and each bay spot."""
    import numpy as np

    values = np.atleast_2d(np.asarray(values, dtype=float))
    chargers = [o.goto for s in search.spots[CHARGERS] for o in s.objects]
    bays = [o.goto for s in search.spots[BAYS] for o in s.objects]
    if not chargers or not bays:
        raise ValueError("the search has no charger spot or no bay spot")
    gotos = [s.objects[0].goto for s in lockers]
    lengths, integrals = search.paths.integrals([search.loading.objects[0].goto, *gotos], [*chargers, *bays, *gotos], values)
    between, between_x = search.paths.integrals(chargers, bays, values)
    return assemble_legs(search, lengths, integrals, between, between_x)


def assemble_legs(search: Search, lengths: np.ndarray, integrals: np.ndarray, between: np.ndarray, between_x: np.ndarray) -> dict[str, np.ndarray]:
    """`leg_integrals`' result from the raw path arrays: `lengths` (1 + K, targets) and `integrals` (rows, 1 + K,
    targets) from the loading point and each locker spot to every charger object, bay object and locker spot, in that
    order; `between` and `between_x` from every charger object to every bay object."""
    nc, nb = len(search.spots[CHARGERS]), len(search.spots[BAYS])
    wc, wb = len(place.CATALOG[CHARGERS].members), len(place.CATALOG[BAYS].members)
    a, b = nc * wc, nc * wc + nb * wb

    def spots(block: np.ndarray, count: int, width: int) -> np.ndarray:
        return block.reshape(*block.shape[:-1], count, width).mean(axis=-1)

    m_c, m_b = spots(lengths[:, :a], nc, wc), spots(lengths[:, a:b], nb, wb)
    x_c, x_b = spots(integrals[:, :, :a], nc, wc), spots(integrals[:, :, a:b], nb, wb)
    rows = integrals.shape[0]
    return {
        "m_L_charger": m_c[0],
        "m_K_charger": m_c[1:],
        "m_L_bay": m_b[0],
        "m_K_bay": m_b[1:],
        "m_LK": lengths[0, b:],
        "m_BC": between.reshape(nc, wc, nb, wb).mean(axis=(1, 3)),
        "x_L_charger": x_c[:, 0],
        "x_K_charger": x_c[:, 1:],
        "x_L_bay": x_b[:, 0],
        "x_K_bay": x_b[:, 1:],
        "x_LK": integrals[:, 0, b:],
        "x_BC": between_x.reshape(rows, nc, wc, nb, wb).mean(axis=(2, 4)),
    }


@attrs.frozen(eq=False)
class _Channel:
    """One score of a layout that is a sum over its lockers. With charger spot c, bay spot b, and locker j at spot k_j
    serving the entrances P_j, a share q_j of all entrances:

        pair[c, b] + sum over j of ( unit[k_j] + q_j (leg[k_j] + bay[b, k_j] + charger[c, k_j]) + sum over e in P_j of entrance[e, k_j] )

    Shapes: pair (C, B), unit and leg (K,), bay (B, K), charger (C, K), entrance (E, K)."""

    pair: np.ndarray
    unit: np.ndarray
    leg: np.ndarray
    bay: np.ndarray
    charger: np.ndarray
    entrance: np.ndarray

    def times(self, factor: float) -> _Channel:
        return _Channel(*(factor * part for part in attrs.astuple(self, recurse=False)))

    def plus(self, other: _Channel) -> _Channel:
        return _Channel(*(a + b for a, b in zip(attrs.astuple(self, recurse=False), attrs.astuple(other, recurse=False), strict=True)))


@attrs.frozen
class Cost:
    """The predicted pedestrian cost a layout is ranked by (A5, A6), per walker group: `exposure[g]` times the group's
    predicted exposure (seconds per hour per member, at `band_m`) plus `walk[g]` times its mean walk to the locker (m,
    as the group walks). The ranking value is the largest group's cost: the worst-off group. A group named in neither
    is not scored."""

    exposure: Mapping[str, float] = attrs.field(factory=dict, converter=dict)
    walk: Mapping[str, float] = attrs.field(factory=dict, converter=dict)
    band_m: float = CONFLICT_BAND_M

    def __attrs_post_init__(self) -> None:
        if not self.groups:
            raise ValueError("a cost scores at least one walker group")
        if any(v < 0 for v in (*self.exposure.values(), *self.walk.values())):
            raise ValueError("cost weights must not be negative")

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*self.exposure, *self.walk]))

    @classmethod
    def pooled(cls, delay_per_exposure: Mapping[str, float], recipients_per_h: float, present: float, speeds: Mapping[str, float] | None = None, band_m: float = CONFLICT_BAND_M) -> Cost:
        """Both parts as time a group loses per hour its members spend on the site, in seconds per hour per member: the
        exposure times `delay_per_exposure[g]` (seconds of delay per second of exposure, a calibration), and the
        recipients' walk to the locker and back, 2 W / speed, times the recipients per hour per walker on the site,
        `recipients_per_h / present` (recipients come in the crowd's mix, so the share is the same in every group)."""
        if present <= 0 or recipients_per_h < 0:
            raise ValueError(f"present must be positive and recipients_per_h not negative, got {present} and {recipients_per_h}")
        speeds = dict(WALK_SPEED_MPS if speeds is None else speeds)
        return cls(dict(delay_per_exposure), {g: recipients_per_h * 2.0 / (present * speeds[g]) for g in delay_per_exposure}, band_m)

    def of(self, exposure: Mapping[str, float], walk_m: Mapping[str, float]) -> dict[str, float]:
        """The cost per group from a layout's exposure and mean walk per group."""
        return {g: (self.exposure[g] * exposure[g] if self.exposure.get(g) else 0.0) + (self.walk[g] * walk_m[g] if self.walk.get(g) else 0.0) for g in self.groups}


@attrs.frozen(eq=False)
class Cover:
    """The lockers of the layouts at one reach: `within[e, k]`, whether locker spot k lies within `reach_m` of recipient
    entrance e as people walk; `lockers`, the spots within the reach of at least one entrance (the candidates); `count`,
    the fewest lockers that give every entrance one within the reach; and `partitions`, every way to split the
    entrances into `count` groups so that one spot lies within the reach of a whole group (each a tuple of groups of
    entrance indices, the groups in order of their first entrance)."""

    reach_m: float
    within: np.ndarray
    lockers: np.ndarray
    count: int
    partitions: tuple[tuple[tuple[int, ...], ...], ...]


def _fewest(masks: Sequence[int], full: int) -> int:
    """The fewest of `masks` (bit sets) whose union is `full`."""
    reached = {0}
    for count in range(1, bin(full).count("1") + 1):
        reached = {m | u for m in reached for u in masks}
        if full in reached:
            return count
    raise ValueError("the masks do not cover the full set")


def _partitions(full: int, count: int, coverable: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """Every way to split the bit set `full` into `count` non-empty parts, each one of `coverable`; the parts in order
    of their lowest bit."""
    if full == 0:
        if count == 0:
            yield ()
        return
    if count == 0:
        return
    lowest = full & -full
    for part in coverable:
        if part & lowest and not part & ~full:
            for rest in _partitions(full ^ part, count - 1, coverable):
                yield (part, *rest)


def _ascending(values: Sequence[np.ndarray]) -> Iterator[tuple[float, tuple[int, ...]]]:
    """One position per array of `values` (each sorted ascending), every combination in order of the sum of the values
    at its positions, ties by position: (sum, positions)."""
    if any(len(v) == 0 for v in values):
        return
    start = (0,) * len(values)
    heap = [(float(sum(v[0] for v in values)), start)]
    seen = {start}
    while heap:
        total, at = heapq.heappop(heap)
        yield total, at
        for j in range(len(at)):
            if at[j] + 1 < len(values[j]):
                step = (*at[:j], at[j] + 1, *at[j + 1 :])
                if step not in seen:
                    seen.add(step)
                    heapq.heappush(heap, (float(sum(v[i] for v, i in zip(values, step, strict=True))), step))


@attrs.frozen(eq=False)
class Placed:
    """One layout a search returns: its charger spot, bay spot and locker spots (indices into the search's spots; the
    lockers in the layout's order, locker 1 first), the entrances each locker serves, the `value` its search ranked
    it by (lower is better), its robot travel per job and the NAV@r verdict of its full check."""

    layout: Layout
    reach_m: float
    charger: int
    bay: int
    lockers: tuple[int, ...]
    serves: tuple[tuple[int, ...], ...]
    value: float
    travel_m: float
    navr: NavrVerdict

    @property
    def layout_id(self) -> str:
        return self.layout.layout_id

    @property
    def key(self) -> tuple[int, int, tuple[int, ...]]:
        return (self.charger, self.bay, self.lockers)


class EntranceSearch:
    """The search space for demand at the recipient entrances (see the module doc), on top of `search`: a
    `search.Search` built with `elements={"locker": locker_element()}`, so its locker spots are the rule-valid ones
    of the whole site. `entrances` are the recipient entrances (`recipient_entrances`), `links` the site's stairs and
    escalators (the site config's), `reach_max_m` the largest reach that will be asked for: locker spots further from
    every entrance are dropped. `groups` are the walker groups' occupancy (`search.Occupancy`, by name; none: no
    exposure terms) and `bands` the conflict bands the exposure is kept at. `legs` takes `leg_integrals`' result for
    these lockers, groups and bands (rows in `rows` order), or a function with `leg_integrals`' arguments that gives
    it (one that reads a cache, or spreads the paths over processes). With `screen` a spot that fails the full
    check on its own (with the loading point alone) is left out: a layout with it is taken as invalid, so the search
    does not walk through every layout that holds it."""

    def __init__(
        self,
        search: Search,
        entrances: Sequence[Entrance],
        *,
        links: Iterable[Mapping[str, Any]] = (),
        reach_max_m: float = max(REACHES_M),
        demand: Demand | None = None,
        groups: Mapping[str, Occupancy] | None = None,
        bands: Sequence[float] = (CONFLICT_BAND_M,),
        speed_mps: float = ROBOT_SPEED_MPS,
        walk_speeds: Mapping[str, float] | None = None,
        legs: Mapping[str, np.ndarray] | Callable[[Search, Sequence[Spot], np.ndarray], Mapping[str, np.ndarray]] | None = None,
        screen: bool = True,
    ) -> None:
        import numpy as np

        self.search = search
        self.demand = Demand() if demand is None else demand
        self.entrances = tuple(entrances)
        level = search.loading.level
        if not self.entrances:
            raise ValueError("no recipient entrances: the site names no entrance of a recipient type")
        elsewhere = [e.name for e in self.entrances if e.level not in (None, level)]
        if elsewhere:
            raise ValueError(f"recipient entrances {elsewhere} are not on the loading point's level {level!r}")
        if len(self.entrances) > 20:
            raise ValueError(f"{len(self.entrances)} recipient entrances, over the 20 the cover is worked out for")
        if not search.spots[CHARGERS] or not search.spots[BAYS]:
            raise ValueError("the search has no charger spot or no bay spot")
        self.groups: dict[str, Occupancy] = dict(groups or {})
        self.bands = tuple(float(b) for b in bands)
        self.speed_mps = float(speed_mps)
        self.walk_speeds = dict(WALK_SPEED_MPS if walk_speeds is None else walk_speeds)
        self.reach_max_m = float(reach_max_m)
        self.walks = Walks.of(place.walking_paths(search.site, level), links)
        anchors = [e.anchor for e in self.entrances]
        every = search.spots[LOCKER]
        walk = self.walks.lengths(anchors, [s.objects[0].goto for s in every], climbs=True)
        near = np.flatnonzero((walk <= self.reach_max_m + _EPS).any(axis=0))
        #: Per element, the spots left out because they fail the full check on their own
        self.alone_invalid: dict[str, int] = dict.fromkeys(SEARCHED, 0)
        if screen:
            alone = np.array([self._alone(every[int(i)]) for i in near], dtype=bool)
            self.alone_invalid[LOCKER] = int((~alone).sum())
            near = near[alone]
        #: The locker spots within the largest reach of at least one entrance, as indices into `search.spots["locker"]`
        self.kept = near
        self.lockers: tuple[Spot, ...] = tuple(every[int(i)] for i in self.kept)
        gotos = [s.objects[0].goto for s in self.lockers]
        #: Per entrance and locker spot, the walk as people walk and step-free [m]
        self.walk_m = walk[:, self.kept]
        self.free_m = self.walks.lengths(anchors, gotos, climbs=False)
        self._walk_s: dict[str, np.ndarray] = {}
        self._gotos = gotos
        self._cb = search.compatible(CHARGERS, BAYS).copy()
        if screen:
            chargers = np.array([self._alone(spot) for spot in search.spots[CHARGERS]], dtype=bool)
            bays = np.array([self._alone(spot) for spot in search.spots[BAYS]], dtype=bool)
            self.alone_invalid[CHARGERS], self.alone_invalid[BAYS] = int((~chargers).sum()), int((~bays).sum())
            self._cb &= chargers[:, None] & bays[None, :]
        self._ck = search.compatible(CHARGERS, LOCKER)[:, self.kept]
        self._bk = search.compatible(BAYS, LOCKER)[:, self.kept]
        self._kk: np.ndarray | None = None
        #: The rows of the leg integrals: one per walker group and conflict band
        self.rows = [(name, band) for name in self.groups for band in self.bands]
        if legs is None or callable(legs):
            fields = [self.groups[name].sample(self.groups[name].band(band), search.paths.centres) for name, band in self.rows]
            legs = (leg_integrals if legs is None else legs)(search, self.lockers, np.stack(fields) if fields else np.zeros((0, search.paths.cells)))
        self.legs = dict(legs)
        if self.legs["m_LK"].shape != (len(self.lockers),) or self.legs["x_LK"].shape != (len(self.rows), len(self.lockers)):
            raise ValueError(f"the leg integrals hold {self.legs['m_LK'].shape[0]} locker spots and {self.legs['x_LK'].shape[0]} rows, the search {len(self.lockers)} and {len(self.rows)}")
        self._covers: dict[float, Cover] = {}
        self._checked: dict[tuple[int, int, tuple[int, ...]], Checked] = {}
        self._element: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.solves = 0

    def _alone(self, spot: Spot) -> bool:
        """Whether `spot` passes the full check with the loading point alone."""
        from arena_simulation_setup.metrics.navr_filter import navr_check

        layout = Layout.of([self.search.loading, spot])
        return not place.check_layout(self.search.site, layout) and navr_check(self.search.base, place.placed_elements(layout)).ok

    # --- the walk ----------------------------------------------------------------------------------------------------

    def climbs(self, group: str) -> bool:
        return group not in STEP_FREE

    def walk_of(self, group: str) -> np.ndarray:
        """Per entrance and locker spot, the walk of `group` [m]: step-free for the groups of `STEP_FREE`."""
        return self.walk_m if self.climbs(group) else self.free_m

    def walk_seconds(self, group: str) -> np.ndarray:
        """Per entrance and locker spot, the quickest walk of `group` [s] at its walking speed (`Walks.seconds`)."""
        if group not in self._walk_s:
            if group not in self.walk_speeds:
                raise ValueError(f"no walking speed for group {group!r}")
            self._walk_s[group] = self.walks.seconds([e.anchor for e in self.entrances], self._gotos, self.walk_speeds[group], climbs=self.climbs(group))
        return self._walk_s[group]

    # --- the lockers of a layout -------------------------------------------------------------------------------------

    def cover(self, reach_m: float) -> Cover:
        """The lockers of the layouts at `reach_m` (`Cover`). ValueError when an entrance has no locker spot within the
        reach, or the reach is over the search's largest."""
        import numpy as np

        reach_m = float(reach_m)
        if reach_m > self.reach_max_m + _EPS:
            raise ValueError(f"the reach {reach_m:g} m is over the search's largest, {self.reach_max_m:g} m")
        if reach_m not in self._covers:
            within = self.walk_m <= reach_m + _EPS
            missing = [e.name for e, row in zip(self.entrances, within, strict=True) if not row.any()]
            if missing:
                nearest = ", ".join(f"{e.name} {self.walk_m[i].min() if len(self.lockers) else math.inf:.1f} m" for i, e in enumerate(self.entrances) if e.name in missing)
                raise ValueError(f"no locker spot within {reach_m:g} m as people walk of {len(missing)} recipient entrances (nearest valid spot: {nearest})")
            n = len(self.entrances)
            masks = (within * (1 << np.arange(n))[:, None]).sum(axis=0)
            largest = [m for m in sorted({int(m) for m in masks if m}) if not any(m != o and m & o == m for o in {int(v) for v in masks})]
            full = (1 << n) - 1
            count = _fewest(largest, full)
            coverable: set[int] = set()
            for mask in largest:
                sub = mask
                while sub:
                    coverable.add(sub)
                    sub = (sub - 1) & mask
            parts = [tuple(tuple(e for e in range(n) if part >> e & 1) for part in split) for split in _partitions(full, count, sorted(coverable))]
            self._covers[reach_m] = Cover(reach_m, within, np.flatnonzero(within.any(axis=0)), count, tuple(sorted(parts)))
        return self._covers[reach_m]

    @property
    def locker_pairs(self) -> np.ndarray:
        """Which two locker spots `check_layout` passes as a pair, (K, K); a spot with itself does not."""
        if self._kk is None:
            import numpy as np
            import shapely

            n = len(self.lockers)
            out = np.ones((n, n), dtype=bool)
            np.fill_diagonal(out, False)
            boxes = [_reach(s) for s in self.lockers]
            tree = shapely.STRtree(boxes) if boxes else None
            for i, spot in enumerate(self.lockers):
                for j in tree.query(boxes[i]) if tree is not None else ():
                    j = int(j)
                    if j > i:
                        out[i, j] = out[j, i] = not place.check_layout(self.search.site, Layout.of([spot, numbered(self.lockers[j], 2)]))
            self._kk = out
        return self._kk

    def served(self, lockers: Iterable[int]) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        """The lockers of a layout in its order and the entrances each serves: every entrance goes to the locker
        nearest to it as people walk (ties to the lower spot index); the lockers in order of their first entrance."""
        import numpy as np

        spots = sorted({int(k) for k in lockers})
        nearest = np.argmin(self.walk_m[:, spots], axis=1)
        groups = {k: tuple(e for e in range(len(self.entrances)) if spots[int(nearest[e])] == k) for k in spots}
        order = sorted(spots, key=lambda k: (groups[k][0] if groups[k] else len(self.entrances), k))
        return tuple(order), tuple(groups[k] for k in order)

    def layout(self, charger: int, bay: int, lockers: Sequence[int]) -> Layout:
        """The layout of charger spot `charger`, bay spot `bay` and the locker spots `lockers` (in the layout's order)
        with the loading point."""
        s = self.search
        return Layout.of([s.loading, s.spots[CHARGERS][charger], s.spots[BAYS][bay], *(numbered(self.lockers[k], i + 1) for i, k in enumerate(lockers))])

    def check(self, charger: int, bay: int, lockers: Sequence[int]) -> Checked:
        """The full check of a layout: `check_layout` and the NAV@r filter, cached."""
        key = (int(charger), int(bay), tuple(int(k) for k in lockers))
        if key not in self._checked:
            from arena_simulation_setup.metrics.navr_filter import navr_check

            layout = self.layout(*key)
            self._checked[key] = Checked(tuple(place.check_layout(self.search.site, layout)), navr_check(self.search.base, place.placed_elements(layout)))
        return self._checked[key]

    # --- the scores --------------------------------------------------------------------------------------------------

    def _zeros(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import numpy as np

        c, b, k, e = len(self.search.spots[CHARGERS]), len(self.search.spots[BAYS]), len(self.lockers), len(self.entrances)
        return np.zeros((c, b)), np.zeros(k), np.zeros(k), np.zeros((b, k)), np.zeros((c, k)), np.zeros((e, k))

    def travel_channel(self) -> _Channel:
        """Robot travel per job [m] (`Demand`)."""
        d, m = self.demand, self.legs
        wb, wc = d.bay_share, 1.0 - d.bay_share
        zeros = self._zeros()
        pair = wb * m["m_L_bay"][None, :] + wc * m["m_L_charger"][:, None] + d.charging_trips_per_job * m["m_BC"]
        return _Channel(pair, zeros[1], m["m_LK"], wb * m["m_K_bay"].T, wc * m["m_K_charger"].T, zeros[5])

    def _elements(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        """The element term of `group` [s per h per member]: per charger and bay spot (with the loading point's) and
        per locker spot, the time on the footprint, dock and approach zone."""
        if group not in self._element:
            import numpy as np

            occ = self.groups[group]
            per, loading = self.search._spot_people(occ)
            scale = 3600.0 / occ.present
            pair = scale * (loading + np.asarray(per[CHARGERS], dtype=float)[:, None] + np.asarray(per[BAYS], dtype=float)[None, :])
            self._element[group] = (pair, scale * np.asarray(per[LOCKER], dtype=float)[self.kept])
        return self._element[group]

    def exposure_channel(self, group: str, band_m: float = CONFLICT_BAND_M, *, element: bool = True) -> _Channel:
        """The predicted exposure of `group` [s per h per member]: the time within `band_m` of a driving robot over the
        legs robot travel counts (the traffic term), plus the time on an element's floor with `element`."""
        if (group, float(band_m)) not in self.rows:
            raise ValueError(f"no occupancy of group {group!r} at the {band_m:g} m band (groups {sorted(self.groups)}, bands {list(self.bands)})")
        occ = self.groups[group]
        if occ.present <= 0:
            raise ValueError(f"group {group!r} has no walker time in its occupancy")
        row = self.rows.index((group, float(band_m)))
        d, x = self.demand, self.legs
        wb, wc = d.bay_share, 1.0 - d.bay_share
        scale = d.rate_per_h / (self.speed_mps * occ.present)
        zeros = self._zeros()
        pair = scale * (wb * x["x_L_bay"][row][None, :] + wc * x["x_L_charger"][row][:, None] + d.charging_trips_per_job * x["x_BC"][row])
        unit = zeros[1]
        if element:
            floor, unit = self._elements(group)
            pair = pair + floor
        return _Channel(pair, unit, scale * x["x_LK"][row], scale * wb * x["x_K_bay"][row].T, scale * wc * x["x_K_charger"][row].T, zeros[5])

    def walk_channel(self, group: str) -> _Channel:
        """The mean walk to the locker of `group` over the recipient entrances [m]."""
        zeros = self._zeros()
        return _Channel(*zeros[:5], self.walk_of(group) / len(self.entrances))

    def cost_channels(self, cost: Cost) -> dict[str, _Channel]:
        """`cost` per walker group."""
        out = {}
        for group in cost.groups:
            parts = []
            if cost.exposure.get(group):
                parts.append(self.exposure_channel(group, cost.band_m).times(cost.exposure[group]))
            if cost.walk.get(group):
                parts.append(self.walk_channel(group).times(cost.walk[group]))
            if not parts:
                parts.append(self.walk_channel(group).times(0.0))
            out[group] = parts[0] if len(parts) == 1 else parts[0].plus(parts[1])
        return out

    def value(self, channel: _Channel, charger: int, bay: int, lockers: Sequence[int]) -> float:
        """`channel`'s value of a layout (lockers in any order)."""
        order, serves = self.served(lockers)
        n = len(self.entrances)
        total = float(channel.pair[charger, bay])
        for k, group in zip(order, serves, strict=True):
            total += float(channel.unit[k] + len(group) / n * (channel.leg[k] + channel.bay[bay, k] + channel.charger[charger, k]) + sum(channel.entrance[e, k] for e in group))
        return total

    def travel(self, charger: int, bay: int, lockers: Sequence[int]) -> float:
        return self.value(self.travel_channel(), charger, bay, lockers)

    def exposure(self, charger: int, bay: int, lockers: Sequence[int], band_m: float = CONFLICT_BAND_M) -> tuple[dict[str, float], dict[str, float]]:
        """The predicted exposure per group of a layout at `band_m` and its traffic term [s per h per member]."""
        return (
            {g: self.value(self.exposure_channel(g, band_m), charger, bay, lockers) for g in self.groups},
            {g: self.value(self.exposure_channel(g, band_m, element=False), charger, bay, lockers) for g in self.groups},
        )

    def walk(self, lockers: Sequence[int], groups: Sequence[str] = GROUPS) -> dict[str, dict[str, float]]:
        """The walk to the locker per group of a layout: the mean and the longest over the recipient entrances, in
        metres (the shortest walk) and seconds (the quickest), each entrance to the locker that serves it."""
        order, serves = self.served(lockers)
        pairs = [(e, k) for k, group in zip(order, serves, strict=True) for e in group]
        out = {}
        for group in groups:
            metres = [float(self.walk_of(group)[e, k]) for e, k in pairs]
            seconds = [float(self.walk_seconds(group)[e, k]) for e, k in pairs]
            out[group] = {"mean_m": sum(metres) / len(metres), "longest_m": max(metres), "mean_s": sum(seconds) / len(seconds), "longest_s": max(seconds)}
        return out

    def placed(self, reach_m: float, charger: int, bay: int, lockers: Sequence[int], value: float | None = None) -> Placed:
        order, serves = self.served(lockers)
        travel = self.travel(charger, bay, order)
        return Placed(self.layout(charger, bay, order), float(reach_m), int(charger), int(bay), order, serves, travel if value is None else float(value), travel, self.check(charger, bay, order).navr)

    def record(self, placed: Placed, cost: Cost | None = None) -> dict[str, Any]:
        """A layout as plain data: its id, reach, robot travel, the entrances each locker serves, the walk per group
        and per entrance, the predicted exposure per group at each band (and its traffic term), the cost per group
        under `cost` with the worst-off group, and each object's spot and pose."""
        lockers = {k: f"{LOCKER}_{i + 1}" for i, k in enumerate(placed.lockers)}
        out: dict[str, Any] = {
            "layout_id": placed.layout_id,
            "reach_m": placed.reach_m,
            "lockers": len(placed.lockers),
            "value": _num(placed.value),
            "travel_m": _num(placed.travel_m),
            "floor_loss_m2": _num(placed.navr.floor_loss_m2),
            "serves": {lockers[k]: [self.entrances[e].name for e in group] for k, group in zip(placed.lockers, placed.serves, strict=True)},
            "walk": {g: {key: _num(v) for key, v in values.items()} for g, values in self.walk(placed.lockers).items()},
            "walk_by_entrance": {self.entrances[e].name: {"locker": lockers[k], "walk_m": _num(float(self.walk_m[e, k])), "step_free_m": _num(float(self.free_m[e, k]))} for k, group in zip(placed.lockers, placed.serves, strict=True) for e in group},
        }
        if self.groups:
            out["exposure"], out["exposure_traffic"] = {}, {}
            for band in self.bands:
                total, traffic = self.exposure(placed.charger, placed.bay, placed.lockers, band)
                out["exposure"][f"{band:g}"] = {g: _num(v) for g, v in total.items()}
                out["exposure_traffic"][f"{band:g}"] = {g: _num(v) for g, v in traffic.items()}
        if cost is not None:
            values = {g: self.value(ch, placed.charger, placed.bay, placed.lockers) for g, ch in self.cost_channels(cost).items()}
            out["cost"] = {g: round(v, 6) for g, v in values.items()}
            out["worst_group"] = max(values, key=lambda g: values[g])
        out["spots"] = [
            {"element": s.element.name, "spot": s.id, "zone": s.zone, "piece": s.piece, "step": s.step, "objects": [{"name": o.name, "model": o.model, "x": o.x, "y": o.y, "yaw": o.yaw, "goto": list(o.goto)} for o in s.objects]}
            for s in placed.layout.spots
        ]
        return out

    def counts(self) -> dict[str, Any]:
        s = self.search
        return {"entrances": [e.name for e in self.entrances], "spots": {CHARGERS: len(s.spots[CHARGERS]), BAYS: len(s.spots[BAYS]), LOCKER: len(self.lockers)}, "locker_spots_of_the_site": len(s.spots[LOCKER]), "invalid_alone": dict(self.alone_invalid), "reach_max_m": self.reach_max_m}

    # --- the exact search --------------------------------------------------------------------------------------------

    def _pair_ok(self, first: Sequence[int], spots_a: np.ndarray, second: Sequence[int], spots_b: np.ndarray) -> np.ndarray:
        """Which spot of `spots_a` (rows) and spot of `spots_b` (columns) can be two lockers of one layout that serve
        the entrance groups `first` and `second`: the pair passes `check_layout`, and every entrance of a group is
        nearer to its own locker than to the other (`served`)."""
        ok = self.locker_pairs[spots_a][:, spots_b].copy()
        w = self.walk_m
        for e in first:
            a, b = w[e, spots_a][:, None], w[e, spots_b][None, :]
            ok &= (a < b) | ((a == b) & (spots_a[:, None] < spots_b[None, :]))
        for e in second:
            a, b = w[e, spots_b][None, :], w[e, spots_a][:, None]
            ok &= (a < b) | ((a == b) & (spots_b[None, :] < spots_a[:, None]))
        return ok

    def best(
        self,
        cover: Cover,
        objective: Sequence[_Channel],
        caps: Sequence[tuple[_Channel, float]] = (),
        *,
        tie: _Channel | None = None,
        start: Sequence[tuple[int, int, Sequence[int]]] = (),
        bays: Sequence[int] | None = None,
    ) -> tuple[float, tuple[int, int, tuple[int, ...]]] | None:
        """The valid layout at `cover`'s reach with the least value of the largest of the `objective` channels, among
        those where every channel of `caps` is at most its cap: (value, (charger, bay, lockers)), None when there is
        none. Ties go to the lower `tie` channel, then to the lower spot indices. `start` are layouts to begin from
        (those that hold the caps and pass the full check bound the search from the first step); `bays` keeps the
        search to those bay spots. Exact: see the module doc."""
        import numpy as np

        n = len(self.entrances)
        chans = [ch for ch, _ in caps] + list(objective) + ([tie] if tie is not None else [])
        ncap, nobj = len(caps), len(objective)
        limits = [float(cap) + _EPS * max(1.0, abs(float(cap))) for _, cap in caps]
        best_key: tuple[float, float, tuple[int, int, tuple[int, ...]]] | None = None

        def consider(charger: int, bay: int, lockers: Sequence[int]) -> None:
            nonlocal best_key
            order, _ = self.served(lockers)
            values = [self.value(ch, charger, bay, order) for ch in chans]
            if any(values[h] > limits[h] for h in range(ncap)):
                return
            key = (max(values[ncap : ncap + nobj]), values[-1] if tie is not None else 0.0, (int(charger), int(bay), order))
            if (best_key is None or key < best_key) and self.check(charger, bay, order).valid:
                best_key = key

        for charger, bay, lockers in start:
            if bays is None or bay in bays:
                consider(charger, bay, lockers)
        for parts in cover.partitions:
            lists = [np.flatnonzero(cover.within[list(part)].all(axis=0)) for part in parts]
            if any(len(spots) == 0 for spots in lists):
                continue
            share = [len(part) / n for part in parts]
            pair_ok = {(i, j): self._pair_ok(parts[i], lists[i], parts[j], lists[j]) for i in range(len(parts)) for j in range(i + 1, len(parts))}
            fixed = [[ch.unit[spots] + q * ch.leg[spots] + ch.entrance[list(part)][:, spots].sum(axis=0) for part, spots, q in zip(parts, lists, share, strict=True)] for ch in chans]
            for bay in range(self._cb.shape[1]) if bays is None else bays:
                alive = self._cb[:, bay].copy()
                if not alive.any():
                    continue
                ok = [self._ck[:, spots] & self._bk[bay, spots][None, :] for spots in lists]
                table = [[fixed[h][j][None, :] + share[j] * (ch.bay[bay, spots][None, :] + ch.charger[:, spots]) for j, spots in enumerate(lists)] for h, ch in enumerate(chans)]
                for h in range(ncap):
                    least = [np.where(ok[j], table[h][j], np.inf).min(axis=1) for j in range(len(parts))]
                    total = chans[h].pair[:, bay] + sum(least)
                    alive &= total <= limits[h]
                    for j in range(len(parts)):
                        with np.errstate(invalid="ignore"):
                            ok[j] &= table[h][j] <= (limits[h] - (total - least[j]))[:, None]
                if not alive.any():
                    continue
                bound = np.full(len(alive), -np.inf)
                for h in range(ncap, ncap + nobj):
                    bound = np.maximum(bound, chans[h].pair[:, bay] + sum(np.where(ok[j], table[h][j], np.inf).min(axis=1) for j in range(len(parts))))
                alive &= np.isfinite(bound)
                chargers = np.flatnonzero(alive)
                for charger in chargers[np.argsort(bound[chargers], kind="stable")]:
                    if best_key is not None and bound[charger] > best_key[0]:
                        break
                    found = self._exact(int(charger), int(bay), parts, lists, pair_ok, [ok[j][charger] for j in range(len(parts))], [[table[h][j][charger] for j in range(len(parts))] for h in range(len(chans))], [float(ch.pair[charger, bay]) for ch in chans], limits, nobj, tie is not None, best_key)
                    if found is not None:
                        best_key = found
        return None if best_key is None else (best_key[0], best_key[2])

    def _exact(
        self,
        charger: int,
        bay: int,
        parts: Sequence[Sequence[int]],
        lists: Sequence[np.ndarray],
        pair_ok: Mapping[tuple[int, int], np.ndarray],
        ok: Sequence[np.ndarray],
        table: Sequence[Sequence[np.ndarray]],
        pair: Sequence[float],
        limits: Sequence[float],
        nobj: int,
        tied: bool,
        best_key: tuple[float, float, tuple[int, int, tuple[int, ...]]] | None,
    ) -> tuple[float, float, tuple[int, int, tuple[int, ...]]] | None:
        """The best valid layout of one charger spot, bay spot and way to split the entrances that beats `best_key`
        (None: any), else None. `ok[j]` are the locker spots of group j still allowed, `table[h][j]` channel h's term
        per spot of group j, `pair[h]` its term of the charger and bay spot."""
        import numpy as np

        self.solves += 1
        count, ncap = len(parts), len(limits)
        objective = range(ncap, ncap + nobj)
        at = [np.flatnonzero(ok[j]) for j in range(count)]
        found = None

        def narrow() -> bool:
            for _ in range(3):
                before = sum(len(a) for a in at)
                for h in [*range(ncap), *(objective if best_key is not None or found is not None else ())]:
                    if any(len(a) == 0 for a in at):
                        return False
                    least = [float(table[h][j][at[j]].min()) for j in range(count)]
                    limit = limits[h] if h < ncap else (found or best_key)[0]
                    for j in range(count):
                        at[j] = at[j][pair[h] + table[h][j][at[j]] + (sum(least) - least[j]) <= limit + _EPS * max(1.0, abs(limit))]
                if sum(len(a) for a in at) == before:
                    break
            return all(len(a) for a in at)

        def look(positions: Sequence[np.ndarray]) -> None:
            nonlocal found
            keep = np.ones(len(positions[0]), dtype=bool)
            for (i, j), allowed in pair_ok.items():
                keep &= allowed[at[i][positions[i]], at[j][positions[j]]]
            totals = [pair[h] + sum(table[h][j][at[j][positions[j]]] for j in range(count)) for h in range(len(table))]
            for h in range(ncap):
                keep &= totals[h] <= limits[h]
            value = np.max(np.stack([totals[h] for h in objective]), axis=0)
            rank = totals[-1] if tied else np.zeros(len(value))
            current = found or best_key
            if current is not None:
                keep &= value <= current[0]
            rows = np.flatnonzero(keep)
            for row in rows[np.lexsort((rank[rows], value[rows]))]:
                lockers = tuple(int(lists[j][at[j][positions[j][row]]]) for j in range(count))
                key = (float(value[row]), float(rank[row]), (charger, bay, lockers))
                current = found or best_key
                if current is not None and key >= current:
                    break
                if self.check(charger, bay, lockers).valid:
                    found = key
                    break

        if not narrow():
            return None
        if best_key is None:
            # a first layout to bound the search with: per group the spot with the least objective terms
            first = [np.array([int(np.argmin(sum(table[h][j][at[j]] for h in objective)))]) for j in range(count)]
            look(first)
            if found is not None and not narrow():
                return found
        sizes = [len(a) for a in at]
        total = math.prod(sizes)
        for begin in range(0, total, MAX_PRODUCT):
            positions = np.unravel_index(np.arange(begin, min(total, begin + MAX_PRODUCT)), sizes)
            look(positions)
        return found

    def nearest_sets(self, cover: Cover, bay: int) -> Iterator[tuple[int, ...]]:
        """The locker sets of the layouts at `cover`'s reach that go with bay spot `bay`, nearest first: in order of
        the entrances' mean walk to the locker that serves them, as people walk; each in the layout's order."""
        import numpy as np

        def stream(parts: Sequence[Sequence[int]]) -> Iterator[tuple[float, tuple[int, ...]]]:
            lists = [np.flatnonzero(cover.within[list(part)].all(axis=0) & self._bk[bay]) for part in parts]
            walks = [self.walk_m[list(part)][:, spots].sum(axis=0) for part, spots in zip(parts, lists, strict=True)]
            order = [np.argsort(w, kind="stable") for w in walks]
            pair_ok = {(i, j): self._pair_ok(parts[i], lists[i], parts[j], lists[j]) for i in range(len(parts)) for j in range(i + 1, len(parts))}
            for total, at in _ascending([w[o] for w, o in zip(walks, order, strict=True)]):
                local = [int(order[j][at[j]]) for j in range(len(parts))]
                if all(allowed[local[i], local[j]] for (i, j), allowed in pair_ok.items()):
                    yield total, tuple(int(lists[j][local[j]]) for j in range(len(parts)))

        for _, lockers in heapq.merge(*(stream(parts) for parts in cover.partitions)):
            yield lockers


def travel_optimal(space: EntranceSearch, reach_m: float = REACH_M) -> Placed:
    """The travel-optimal layout at `reach_m` (A5): the least robot travel per job over all valid layouts. ValueError
    without a valid layout."""
    found = space.best(space.cover(reach_m), [space.travel_channel()])
    if found is None:
        raise ValueError(f"no valid layout at a reach of {reach_m:g} m")
    return space.placed(reach_m, *found[1], value=found[0])


def pedestrian_aware(space: EntranceSearch, reach_m: float, cost: Cost | Sequence[Cost], max_travel_m: float, *, start: Sequence[Placed] = ()) -> Placed:
    """The pedestrian-aware pick at `reach_m` (A5): among the valid layouts whose robot travel per job is at most
    `max_travel_m`, the one with the least cost of the worst-off group (`Cost`); ties go to the shorter robot travel.
    Several costs rank in order: the first decides, the next breaks its ties. `start` are layouts already known (the
    travel-optimal layout), which bound the search. ValueError without a valid layout within the travel."""
    cover = space.cover(reach_m)
    travel = space.travel_channel()
    caps = [(travel, float(max_travel_m))]
    begin = [p.key for p in start]
    found = None
    for one in [cost] if isinstance(cost, Cost) else list(cost):
        channels = list(space.cost_channels(one).values())
        found = space.best(cover, channels, caps, tie=travel, start=begin)
        if found is None:
            raise ValueError(f"no valid layout within {max_travel_m:.1f} m of robot travel per job at a reach of {reach_m:g} m")
        caps = [*caps, *((ch, found[0]) for ch in channels)]
        begin = [found[1]]
    assert found is not None
    return space.placed(reach_m, *found[1], value=found[0])


def practice_based(space: EntranceSearch, reach_m: float = REACH_M) -> Placed:
    """The practice-based layout at `reach_m` (A5; the first study's guideline placement on the new demand), by
    walking distance:

    - the waiting bays at the bay spot nearest the loading point (on the step-free floor, as `search.guideline`);
    - a locker nearest each group of recipient entrances: the lockers with the least mean walk from the entrances to
      the locker that serves them, as people walk;
    - the chargers at the charger spot nearest the waiting bays.

    In that order each takes the best spots by its rule that go with the ones placed before, and the layout must pass
    the full check, else the next charger spot, then the next locker set (`MAX_SETS` per bay spot), then the next bay
    spot. ValueError without a valid layout."""
    import numpy as np

    cover = space.cover(reach_m)
    s = space.search
    walk = place.walking_paths(s.site, s.loading.level)
    centres = s.spot_centres()
    bay_order = np.argsort(walk.lengths([s.loading.objects[0].goto], centres[BAYS])[0], kind="stable")
    for bay in (int(b) for b in bay_order):
        charger_order = np.argsort(walk.lengths([tuple(centres[BAYS][bay])], centres[CHARGERS])[0], kind="stable")
        for lockers in itertools.islice(space.nearest_sets(cover, bay), MAX_SETS):
            for charger in (int(c) for c in charger_order):
                if space._cb[charger, bay] and space._ck[charger, list(lockers)].all() and space.check(charger, bay, lockers).valid:
                    return space.placed(reach_m, charger, bay, lockers)
    raise ValueError(f"no valid practice-based layout at a reach of {reach_m:g} m")


__all__ = [
    "BAY_SHARE",
    "DELAY_PER_EXPOSURE",
    "LINK_SPEED_MPS",
    "MARGINS",
    "REACHES_M",
    "REACH_M",
    "RECIPIENT_TYPES",
    "STEP_FREE",
    "STEP_KINDS",
    "WALK_SPEED_MPS",
    "Cost",
    "Cover",
    "Demand",
    "EntranceSearch",
    "Placed",
    "Walks",
    "assemble_legs",
    "leg_integrals",
    "locker_element",
    "numbered",
    "pedestrian_aware",
    "practice_based",
    "recipient_entrances",
    "travel_optimal",
]
