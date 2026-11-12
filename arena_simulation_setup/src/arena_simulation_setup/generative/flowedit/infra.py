"""INFRA Stage 1 suites (study design §5, §8, §10): seed-paired arena_evaluation benchmark suites on a site (the twin
smu_link_full), with `flowedit suite`'s stage names `<world>__s<seed>`.

The study runs in this order, each suite its own benchmark run:

1. `reference_suite` (`infra_h1_site`): the site as is without robots on the selection seeds. Its flow cells
   (`flow_cells.parquet`) give the occupancy the joint search's fallback ranks with (`search.Occupancy.from_flow_cells`),
   and its runs are the reference every layout's element part pairs with (§6).
2. the premise check (`infra_h1_premise`, §10): the two most different layouts (`search.premise_pair`) on the premise
   seeds, with the fleet and without robots. With the site's runs at those seeds it is §10's 15 episodes. It checks
   that layouts differ by more than the seed spread and measures the task rate: 75 % of the fleet's capacity in the
   wide layout (the narrow one's cost and failed jobs are layout cost).
3. the selection (`infra_h1`, §5, §10): the robot-only pick, the joint top 10 (ranked among the layouts the travel
   pre-filter keeps) and 5 random valid layouts, each with the fleet and without robots, on the selection seeds.
4. the evaluation (`infra_h1_eval` and `infra_h1_eval_site`, §10, §12): the joint pick and the robot-only pick, with
   the fleet and without robots, and the site as is, on fresh evaluation seeds.

A layout's run without robots is the runner's `unhindered_peds` reference run of its stage (`references:
[unhindered_peds]`): the same copy and seed with the robots parked off the map. Layout copies are saved through
`place.save_layout` (assets hardlinked) with the crowd scenario copied in from the site, so every copy has the site's
through-walker sources and pairs with it; a copy also gets the scenario's locker visitors (`locker_visitors.yaml`
beside it), aimed at its own locker. Every suite spawns the same robots with the same launch arguments and episode
length, so a layout's run without robots differs from the site's only by the elements and their visitors. The warm-up
(§8) is cut in the analysis, not here.

Where each element may go comes from the site config (`SiteConfig`, `infra_sites/<site>.yaml`): zones by function, so
a site zone map replaces them without code changes."""

from __future__ import annotations

import copy
import itertools
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from . import place
from . import suite as SU
from .search import HOMES, LOADING, SEARCHED, Placement, Scored

if TYPE_CHECKING:
    import shapely

    from .edit import Site

#: The twin and its study crowd (§9: sources at both route ends, 70/15/15; locker visitors in layout copies)
SITE = "smu_link_full"
SCENARIO = "study_mix_v2"
#: Stage 1 seeds (Kolja, 10 Nov: the design doc's method with less data): selection 1-5 (doc 1-10), evaluation 11-20
#: (doc 11-30), premise check 1-3 (§10)
SELECTION_SEEDS = (1, 2, 3, 4, 5)
EVALUATION_SEEDS = tuple(range(11, 21))
PREMISE_SEEDS = (1, 2, 3)
SEEDS = SELECTION_SEEDS
#: Episode length in sim seconds, and the warm-up the analysis leaves out (Kolja, 9 Nov: 25 min with 7 min warm-up)
EPISODE_S = 1500.0
WARMUP_S = 420.0
#: The fleet: one jackal per home (§8: 4 jackals)
ROBOT = f"jackal[{len(HOMES)}]"
#: Launch args of every suite: lockstep for seed pairing, people see the robots (§8: PEDS robot visibility on), and no
#: pedestrian skeletons on `arena_peds` (`human.skeleton`: no INFRA measure reads them, the crowd and what the robots see
#: stay the same, and it takes most of the env node's per-pedestrian cost off a run at the study density)
LAUNCH = {"sim": "flatsim", "headless": "true", "lockstep": "true", "human.robots.visible": "true", "human.skeleton": "false"}
#: The reference runs the layout suites ask the runner for: each stage without robots (the traffic part's reference,
#: §6); the fleet without pedestrians is never read
REFERENCES = ["unhindered_peds"]
#: What the recorder keeps of every INFRA run (kkieslich/arena-evaluation's suite `record` block): the INFRA measures
#: read the engine's pedestrian tracks (`agent_states`), the robots and the semantics (the fleet's task log), not the
#: pedestrians' skeletons and bodies on /tf, their named and numbered frames, `arena_peds`, their animation states or the
#: humans' tf topic (its only subscriber in a headless run is the recorder, and `hri_producer` idles without one); they
#: grow with the crowd: 94 % of a pilot recording at 15-20 people, 93 % of a twin smoke at up to ~210 walkers
RECORD = {"exclude": ["peds", "animation_states", "tf_humans"], "tf_frames": [[r"_agent_\d+$", -1.0], [r"^env_\d+/[^/]+$", -1.0], [r"^\d+$", -1.0]]}
#: The study battery: the longest job of either premise layout on the twin (28.6 Wh by the fleet's own energy estimate,
#: the narrow layout's; the robot-only pick's 25.6 Wh) fits in the 80 % between `CHARGE_UNTIL` and the fleet's 10 %
#: reserve (28.6 / 0.8 = 35.7 Wh). The chargers' power is `place.CHARGE_POWER_W` (the catalog writes it into the
#: world). And the start charge each robot draws per episode seed (§8: 40-100 %)
CAPACITY_WH = 36.0
SOC_MIN = 40.0
SOC_MAX = 100.0
#: The fleet mode's charging thresholds (§8: 30 %; charge until 90 %)
CHARGE_BELOW = 30.0
CHARGE_UNTIL = 90.0
#: The fleet mode's defaults for the job mix and the hold times at each end
PICKUP_SHARE = 0.3
LOAD_S = 15.0
UNLOAD_S = 10.0
#: The fleet's planner: Nav2 DWB with the same parameters in every layout (§8)
PLANNER = {"driver": "nav2", "local_planner": "dwb", "inter_planner": "navigate_w_replanning_time"}
FLEET_CONTESTANT = "fleet_dwb"
#: The jackal's planner speed limit (`velocity_limits` in its caps/mobile.yaml, Nav2 DWB's `max_vel_x`), for the rate
#: estimate: an upper bound, since the crowd slows the robots
SPEED_MPS = 0.5
#: The share of the fleet's capacity in the wide premise layout the rate asks for (§8: near 75 %)
UTILISATION = 0.75
#: Suite names; `arena evaluation benchmark --suite <name>` finds them under <imported assets root>/suites
REFERENCE_SUITE = "infra_h1_site"
PREMISE_SUITE = "infra_h1_premise"
LAYOUT_SUITE = "infra_h1"
EVALUATION_SUITE = "infra_h1_eval"
EVALUATION_SITE_SUITE = "infra_h1_eval_site"
MARKER = "# generated by flowedit infra"
#: The site configs shipped with the code, `<site>.yaml`
SITES_DIR = Path(__file__).with_name("infra_sites")
#: The locker visitors beside a crowd scenario: merged into each layout copy's scenario, never read by the runtime
VISITORS = "locker_visitors.yaml"
#: The sides a site's crowd keeps to (HumanSim's global_planner.lane_side); unset: HumanSim's default
CROWD_SIDES = ("right", "left", "none")
#: The walker groups of the study crowd (§6, #93): name, HumanSim agent type, share of every source's walkers
GROUPS = (("adult", "adult", 0.70), ("wheelchair", "wheelchair_manual", 0.15), ("long_cane", "blind_white_cane", 0.15))
#: The study crowd's inflow in all without a crowd density in the site config (both route ends of study_mix_v2: 0.3
#: ped/m2 x 4.04 m x 1.0 m/s) [ped/s]
CROWD_RATE = 1.2
#: A point entrance's source and sink: a square this big around it, cut to the floor a walker reaches, which keeps this far
#: from the walls (HumanSim's planner keeps its paths 0.38 m from them, in whole 0.2 m cells) [m]
ENTRANCE_SQUARE_M, ENTRANCE_WALL_M = 4.0, 0.4
#: An entrance whose area holds less floor than this on the floor a walker reaches (its stairs or doorway walled off at that
#: clearance) takes a larger square around its mark, the next of these sides that holds it [m2], [m]
ENTRANCE_MIN_M2, ENTRANCE_GROW_M = 2.0, (6.0, 8.0, 10.0)
#: The parcel locker's object, which the visitors' target stands in front of
LOCKER = "locker"


# --- the site config -------------------------------------------------------------------------------------------------


@attrs.frozen
class CrowdDensity:
    """The crowd's design density and what turns it into an inflow (§9): `density` walkers per m^2 on `area_m2` of floor,
    so `density x area_m2` walkers present, and `mean_time_s`, the mean time a walker spends on the site from its source
    to its sink, measured in the site's crowd. By Little's law (walkers present = inflow x mean time) the inflow in all
    is `density x area_m2 / mean_time_s`."""

    density: float
    area_m2: float
    mean_time_s: float

    KEYS = ("density", "area_m2", "mean_time_s")

    def present(self, density: float | None = None) -> float:
        """The walkers present at `density` (default the design density)."""
        return (self.density if density is None else float(density)) * self.area_m2

    def rate(self, density: float | None = None) -> float:
        """The inflow in all [ped/s] that keeps `density` (default the design density) on the area: Little's law."""
        return self.present(density) / self.mean_time_s


@attrs.frozen
class SiteConfig:
    """A site's facts for the INFRA study and its crowd: `facts` (`place.SiteFacts`: entrances, the courier access
    point, power outlets), the zones no element goes to (`exclude`, `fnmatch` patterns), the crowd scenario every stage
    runs, and `flow_split`, the share of the inflow each group of entrance types gets (split evenly within a group, at
    the crowd's total rate), for the crowd built from the entrances, and `crowd_density`, which sets that rate. No element
    names: the catalog's rules read the facts (`place.CATALOG`) and the search decides. `path` is the file it came from."""

    site: str
    scenario: str = SCENARIO
    exclude: tuple[str, ...] = place.EXCLUDED_ZONES
    facts: place.SiteFacts = attrs.field(factory=place.SiteFacts)
    flow_split: tuple[tuple[tuple[str, ...], float], ...] = ()
    crowd_side: str | None = None
    crowd_density: CrowdDensity | None = None
    path: Path | None = None

    KEYS = ("site", "scenario", "exclude", "entrances", "courier_access", "outlets", "flow_split", "crowd_side", "crowd_density")

    @property
    def placements(self) -> dict[str, Placement]:
        """What the facts compile to for the search: the elements a rule pins (the courier handoff point at the courier
        access point), by name. The other rules filter the spots themselves (`Search(facts=...)`)."""
        return {name: Placement(pin=at) for name, at in self.facts.pins().items()}

    @classmethod
    def parse(cls, doc: Mapping[str, Any], path: Path | None = None) -> SiteConfig:
        where = f"{path}: " if path is not None else ""
        if "elements" in doc:
            raise ValueError(f"{where}a site config holds the site's facts (entrances, courier_access, outlets), not elements: the catalog's rules place them")
        unknown = sorted(set(doc) - set(cls.KEYS))
        if unknown:
            raise ValueError(f"{where}unknown keys {unknown} ({', '.join(cls.KEYS)})")
        if not doc.get("site"):
            raise ValueError(f"{where}no site")

        def point(value: Any, what: str) -> tuple[float, float]:  # noqa: ANN401 - yaml
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"{where}{what} must be [x, y]")
            return (float(value[0]), float(value[1]))

        entrances = []
        for i, entry in enumerate(doc.get("entrances") or []):
            name = (entry or {}).get("name") or f"#{i}"
            bad = sorted(set(entry or {}) - {"name", "type", "at", "polygon", "recipients", "level", "placeholder"})
            if bad:
                raise ValueError(f"{where}entrance {name}: unknown keys {bad} (name, type, at, polygon, recipients, level, placeholder)")
            if ("at" in entry) == ("polygon" in entry):
                raise ValueError(f"{where}entrance {name}: give its position as `at` [x, y] or as a `polygon`")
            points = (point(entry["at"], f"entrance {name}: at"),) if "at" in entry else tuple(point(q, f"entrance {name}: a polygon corner") for q in entry["polygon"])
            try:
                entrances.append(place.Entrance(str(name), str(entry.get("type", "")), points, bool(entry.get("recipients", False)), None if entry.get("level") is None else str(entry["level"]), bool(entry.get("placeholder", False))))
            except ValueError as e:
                raise ValueError(f"{where}{e}") from None
        names = [e.name for e in entrances]
        if len(set(names)) != len(names):
            raise ValueError(f"{where}entrance names repeat: {sorted({n for n in names if names.count(n) > 1})}")
        if doc.get("courier_access") is None:
            raise ValueError(f"{where}no courier_access: the courier handoff point stands there (§8)")
        courier = point(doc["courier_access"], "courier_access")
        outlets = tuple(point(q, "an outlet") for q in doc.get("outlets") or [])
        split: list[tuple[tuple[str, ...], float]] = []
        for group in doc.get("flow_split") or []:
            types = tuple(str(t) for t in (group or {}).get("types") or [])
            share = float((group or {}).get("share", 0.0))
            wrong = sorted(set(types) - set(place.ENTRANCE_TYPES))
            if not types or wrong or share <= 0.0:
                raise ValueError(f"{where}flow_split: each group names entrance types ({', '.join(place.ENTRANCE_TYPES)}) and a share > 0, not {group}")
            split.append((types, share))
        if split:
            seen = [t for types, _ in split for t in types]
            if len(set(seen)) != len(seen):
                raise ValueError(f"{where}flow_split: an entrance type in two groups")
            if abs(sum(share for _, share in split) - 1.0) > 1e-6:
                raise ValueError(f"{where}flow_split: the shares add up to {sum(share for _, share in split):g}, not 1")
        side = doc.get("crowd_side")
        if side is not None and side not in CROWD_SIDES:
            raise ValueError(f"{where}crowd_side must be one of {', '.join(CROWD_SIDES)}, not {side!r}")
        density = None
        if doc.get("crowd_density") is not None:
            block = doc["crowd_density"]
            if not isinstance(block, Mapping) or set(block) != set(CrowdDensity.KEYS):
                raise ValueError(f"{where}crowd_density needs exactly {', '.join(CrowdDensity.KEYS)}, not {block}")
            values = {k: float(block[k]) for k in CrowdDensity.KEYS}
            if min(values.values()) <= 0.0:
                raise ValueError(f"{where}crowd_density: every value must be > 0, not {block}")
            density = CrowdDensity(**values)
        facts = place.SiteFacts(tuple(entrances), courier, outlets)
        return cls(str(doc["site"]), str(doc.get("scenario") or SCENARIO), tuple(doc.get("exclude", place.EXCLUDED_ZONES)), facts, tuple(split), side, density, path)

    @classmethod
    def load(cls, path: Path) -> SiteConfig:
        import yaml

        path = Path(path)
        return cls.parse(yaml.safe_load(path.read_text()) or {}, path)

    def record(self) -> dict[str, Any]:
        """The config as plain data, for a manifest."""
        return {
            "site": self.site,
            "path": None if self.path is None else str(self.path),
            "scenario": self.scenario,
            "exclude": list(self.exclude),
            "entrances": [{"name": e.name, "type": e.type, "points": [list(q) for q in e.points], "recipients": e.recipients, "level": e.level, "placeholder": e.placeholder} for e in self.facts.entrances],
            "courier_access": None if self.facts.courier_access is None else list(self.facts.courier_access),
            "outlets": [list(q) for q in self.facts.outlets],
            "flow_split": [{"types": list(types), "share": share} for types, share in self.flow_split],
            "crowd_side": self.crowd_side,
            "crowd_density": None if self.crowd_density is None else attrs.asdict(self.crowd_density),
            "pins": {name: list(p.pin) for name, p in self.placements.items() if p.pin is not None},
        }


def site_config(site: str, path: Path | None = None) -> SiteConfig:
    """The site config at `path`, else the one shipped for `site` (`SITES_DIR/<site>.yaml`). FileNotFoundError when
    neither exists."""
    path = Path(path) if path is not None else SITES_DIR / f"{site}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no INFRA site config {path}: pass --site-config, or add {SITES_DIR / f'{site}.yaml'}")
    return SiteConfig.load(path)


# --- the crowd ---------------------------------------------------------------------------------------------------------


def _scenario_file(scenario: Path) -> Path:
    scenario = Path(scenario)
    return scenario / "scenario.yaml" if scenario.is_dir() else scenario


def visitors(scenario: Path) -> dict[str, Any] | None:
    """The locker visitors beside a scenario (`VISITORS` in its directory), None without them."""
    import yaml

    path = _scenario_file(scenario).parent / VISITORS
    if not path.is_file():
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc.get("regions"), dict) or not isinstance(doc.get("target"), dict) or not doc["target"].get("name"):
        raise ValueError(f"{path}: needs `target` (with a name) and `regions`")
    return doc


def entrance_shares(config: SiteConfig) -> dict[str, float]:
    """Each entrance's share of the crowd's inflow: its type group's `flow_split` share, split evenly within the group,
    the shares of groups without an entrance spread over the others in proportion (the total stays). Without a split,
    every entrance the same."""
    entrances = config.facts.entrances
    if not entrances:
        raise ValueError(f"{config.site}: the site config names no entrances")
    if not config.flow_split:
        return {e.name: 1.0 / len(entrances) for e in entrances}
    members = {i: [e.name for e in entrances if e.type in types] for i, (types, _) in enumerate(config.flow_split)}
    loose = sorted(e.name for e in entrances if not any(e.name in names for names in members.values()))
    if loose:
        raise ValueError(f"{config.site}: entrances {loose} have a type no flow_split group names")
    present = {i: share for i, (_, share) in enumerate(config.flow_split) if members[i]}
    total = sum(present.values())
    return {name: present[i] / total / len(members[i]) for i in present for name in members[i]}


def walk_area(floor: shapely.Geometry, walls: shapely.Geometry | None = None) -> shapely.Polygon:
    """The floor a walker reaches: the floor kept `ENTRANCE_WALL_M` from its edges and from every wall, its largest
    connected piece. A pocket the walls close off at that clearance (between pillars, behind a glass front) is left out:
    a walker spawned there could not get out, and one bound for a sink there could not reach it."""
    import shapely

    free = floor.buffer(-ENTRANCE_WALL_M)
    if walls is not None and not walls.is_empty:
        free = free.difference(walls.buffer(ENTRANCE_WALL_M))
    pieces = [g for g in getattr(free, "geoms", [free]) if isinstance(g, shapely.Polygon) and not g.is_empty]
    if not pieces:
        raise ValueError(f"no floor left {ENTRANCE_WALL_M} m clear of the walls")
    return max(pieces, key=lambda g: g.area)


def _entrance_polygon(entrance: place.Entrance, walk: shapely.Polygon) -> list[tuple[float, float]]:
    """The source and sink area of an entrance on the floor a walker reaches (`walk_area`): its polygon, or a square around
    its point, cut to that floor (its largest piece). A polygon that lies on it stays as given. Where that leaves less than
    `ENTRANCE_MIN_M2` (stairs or a doorway the walls close off at the walkers' clearance), a larger square around the mark
    (`ENTRANCE_GROW_M`) puts the area on the nearest floor walkers reach."""
    import shapely

    def largest(area: shapely.Geometry) -> shapely.Polygon | None:
        cut = area.intersection(walk)
        pieces = [g for g in getattr(cut, "geoms", [cut]) if isinstance(g, shapely.Polygon) and not g.is_empty]
        return _without_holes(max(pieces, key=lambda g: g.area)) if pieces else None

    if len(entrance.points) > 1:
        area = shapely.Polygon(entrance.points)
        if walk.covers(area):
            return [(float(x), float(y)) for x, y in entrance.points]
        mark = area.centroid
    else:
        mark = shapely.Point(entrance.points[0])
        half = ENTRANCE_SQUARE_M / 2.0
        area = shapely.box(mark.x - half, mark.y - half, mark.x + half, mark.y + half)
    best = largest(area)
    for side in ENTRANCE_GROW_M:
        if best is not None and best.area >= ENTRANCE_MIN_M2:
            break
        best = largest(shapely.box(mark.x - side / 2.0, mark.y - side / 2.0, mark.x + side / 2.0, mark.y + side / 2.0))
    if best is None or best.area < ENTRANCE_MIN_M2:
        raise ValueError(f"entrance {entrance.name}: under {ENTRANCE_MIN_M2:g} m2 of floor a walker reaches within {ENTRANCE_GROW_M[-1] / 2:g} m of {entrance.points[0]}")
    best = best.simplify(0.05)
    return [(round(float(px), 2), round(float(py), 2)) for px, py in list(best.exterior.coords)[:-1]]


def _without_holes(area: shapely.Polygon) -> shapely.Polygon:
    """`area` without holes: a region is sent as its outer ring, so a pillar inside it would count as its floor. Each hole
    is cut through, along x or along y, whichever keeps the larger piece, and the largest piece stays."""
    import shapely
    from shapely.ops import split

    while area.interiors:
        hole = shapely.Polygon(area.interiors[0]).centroid
        x0, y0, x1, y1 = area.bounds
        best = None
        for line in (shapely.LineString([(hole.x, y0 - 1.0), (hole.x, y1 + 1.0)]), shapely.LineString([(x0 - 1.0, hole.y), (x1 + 1.0, hole.y)])):
            parts = [g for g in split(area, line).geoms if isinstance(g, shapely.Polygon)]
            piece = max(parts, key=lambda g: g.area)
            if best is None or piece.area > best.area:
                best = piece
        area = best
    return area


def entrance_crowd(config: SiteConfig, floor: shapely.Geometry, *, walls: shapely.Geometry | None = None, rate: float = CROWD_RATE, visitor_rate_per_h: float = 12.0, robots: Sequence[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """The study crowd from the site's entrances (§2, §9): every entrance a PEDS source and sink. Its walkers come in at
    `rate` in all, each entrance its `entrance_shares` share, one source per walker group (`GROUPS`, 70/15/15), and
    leave by another entrance, picked in proportion to the other entrances' shares (more walkers come in where more
    leave). Parcel recipients (`visitor_rate_per_h` in all, split evenly over the recipient entrances) come in at a
    recipient entrance, use the locker and leave by the entrance they came in by. Returns the scenario document, the
    locker visitors document (None without a recipient entrance) and the visitors' agent types by file name."""
    shares = entrance_shares(config)
    entrances = config.facts.entrances
    if len(entrances) < 2:
        raise ValueError(f"{config.site}: a crowd between entrances needs two of them")
    walk = walk_area(floor, walls)
    areas = {e.name: [{"x": x, "y": y} for x, y in _entrance_polygon(e, walk)] for e in entrances}
    regions: dict[str, Any] = {}
    for e in entrances:
        others = [o for o in entrances if o.name != e.name]
        weight = sum(shares[o.name] for o in others)
        affinity = [{"sink": f"{o.name}_sink", "weight": round(shares[o.name] / weight, 6)} for o in others]
        for group, agent_type, part in GROUPS:
            regions[f"{e.name}_{group}"] = {
                "type": "source",
                "polygon": areas[e.name],
                "config": {"rate_profile": [{"t": 0.0, "rate": round(rate * shares[e.name] * part, 6)}], "max_concurrent": -1, "max_total": -1, "agent": {"agent_type": agent_type, "sink_affinity": affinity}},
            }
    for e in entrances:
        regions[f"{e.name}_sink"] = {"type": "sink", "polygon": areas[e.name], "config": {"capacity": -1}}
    scenario = {"format": "arena-scenario/2", "robots": [dict(r) for r in robots], "regions": regions}
    recipients = list(config.facts.recipient_entrances)
    if not recipients:
        return scenario, None, {}
    types: dict[str, dict[str, Any]] = {}
    visitor_regions: dict[str, Any] = {}
    for e in recipients:
        centre = place.Entrance(e.name, e.type, tuple((p["x"], p["y"]) for p in areas[e.name])).anchor if len(areas[e.name]) > 2 else areas[e.name][0]
        for group, agent_type, part in GROUPS:
            name = f"locker_visitor__{e.name}_{group}"
            types[f"{name}.yaml"] = {
                "name": name,
                "mode": "behavior_tree",
                "extends": agent_type,
                "initial_sequence": "visit",
                "sequences": {
                    "visit": {"steps": {"pickup": {"target": "locker_visit", "interaction": "USE", "duration": {"mean": 30.0, "std": 10.0, "clip_low": 15.0, "clip_high": 60.0}, "patience": {"mean": 1200.0, "std": 0.0}}}, "then": "leave", "on_failure": "leave"},
                    "leave": {"steps": {"exit": {"kind": "go_to", "target_pose": {"x": round(float(centre[0]), 2), "y": round(float(centre[1]), 2), "theta": 0.0}}}, "then": "leave", "on_failure": "leave"},
                },
            }
            visitor_regions[f"{e.name}_locker_{group}"] = {
                "type": "source",
                "polygon": areas[e.name],
                "config": {"rate_profile": [{"t": 0.0, "rate": round(visitor_rate_per_h / 3600.0 / len(recipients) * part, 8)}], "max_concurrent": -1, "max_total": -1, "agent": {"agent_type": f"./types/{name}.yaml", "sink_affinity": [{"sink": f"{e.name}_sink", "weight": 1.0}]}},
            }
    visitors_doc = {
        "rate_per_h": visitor_rate_per_h,
        "target": {"name": "locker_visit", "humansim": {"type": "parcel_locker", "capacity": 1, "interaction_radius": 0.8, "formation": {"type": "line", "params": {"base_step": 0.9}}}, "clearance_m": 0.35},
        "regions": visitor_regions,
    }
    return scenario, visitors_doc, types


def write_crowd(folder: Path, scenario: Mapping[str, Any], visitors_doc: Mapping[str, Any] | None, types: Mapping[str, Mapping[str, Any]], header: str) -> Path:
    """Write a crowd as a scenario directory: scenario.yaml (with `header` as comment lines), the locker visitors and
    their agent types. Refuses a directory that already has a scenario.yaml."""
    import yaml

    folder = Path(folder)
    if (folder / "scenario.yaml").exists():
        raise FileExistsError(f"{folder / 'scenario.yaml'} exists: pick another name")
    folder.mkdir(parents=True, exist_ok=True)
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    (folder / "scenario.yaml").write_text(comment + yaml.safe_dump(dict(scenario), sort_keys=False))
    if visitors_doc is not None:
        (folder / VISITORS).write_text(comment + yaml.safe_dump(dict(visitors_doc), sort_keys=False))
        (folder / "types").mkdir(exist_ok=True)
        for name, doc in types.items():
            (folder / "types" / name).write_text(yaml.safe_dump(dict(doc), sort_keys=False))
    return folder


def keep_out(scenario: Path) -> list[shapely.Polygon]:
    """The source and sink polygons of a scenario (its file or directory, world frame), and of its locker visitors: the
    floor no element may take (§3)."""
    import shapely
    import yaml

    path = _scenario_file(scenario)
    doc = yaml.safe_load(path.read_text()) or {}
    regions = dict(doc.get("regions") or {})
    extra = visitors(path)
    if extra is not None:
        regions.update(extra["regions"])
    out = []
    for name, region in regions.items():
        if region.get("type") in ("source", "sink"):
            points = [(float(p["x"]), float(p["y"])) for p in region.get("polygon") or []]
            if len(points) < 3:
                raise ValueError(f"{path}: region {name!r} has no polygon")
            out.append(shapely.Polygon(points))
    return out


def visitor_sources(scenario: Path) -> list[str]:
    """The names of the locker visitors' sources of a scenario, [] without visitors."""
    extra = visitors(scenario)
    return [] if extra is None else [name for name, r in extra["regions"].items() if r.get("type") == "source"]


def locker_front(layout: place.Layout, clearance_m: float) -> tuple[float, float, float, float]:
    """Where locker visitors queue in front of the layout's locker: the anchor (the footprint's centroid), the heading a
    visitor faces (towards the locker, away from its dock), and the front offset that puts the first visitor
    `clearance_m` outside the footprint. The line extends from there away from the locker (HumanSim's line formation)."""
    import shapely

    spot = next((s for s in layout.spots if s.element.name == LOCKER), None)
    if spot is None:
        raise ValueError(f"layout {layout.layout_id} has no {LOCKER}")
    (obj,) = spot.objects
    centre, dock = obj.footprint.centroid, obj.dock.centroid
    dx, dy = dock.x - centre.x, dock.y - centre.y
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        raise ValueError(f"{obj.name}: its dock sits on its footprint's centroid, no front")
    ux, uy = dx / norm, dy / norm
    depth = max(float((x - centre.x) * ux + (y - centre.y) * uy) for x, y in shapely.get_coordinates(obj.footprint.exterior))
    return round(float(centre.x), 3), round(float(centre.y), 3), round(math.atan2(-uy, -ux), 4), round(depth + float(clearance_m), 3)


def crowd(layout: place.Layout, scenario: Path, visitor_rate_per_h: float | None = None) -> tuple[dict[str, Any], str] | None:
    """The scenario document of a layout copy: the site's scenario with its locker visitors merged in, their target posed
    in front of the layout's locker (`locker_front`) and every visitor source scaled to `visitor_rate_per_h` in all
    (default the file's `rate_per_h`), with a header line; None when the scenario has no visitors."""
    import yaml

    path = _scenario_file(scenario)
    extra = visitors(path)
    if extra is None:
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    target = extra["target"]
    x, y, theta, front = locker_front(layout, float(target.get("clearance_m", 0.35)))
    block = copy.deepcopy(dict(target.get("humansim") or {}))
    formation = block.setdefault("formation", {"type": "line", "params": {}})
    formation.setdefault("params", {})["front_offset"] = front
    objects = [o for o in doc.get("objects") or [] if o.get("name") != target["name"]]
    objects.append({"name": target["name"], "model": "", "pose": [x, y, theta], "humansim": block})
    doc["objects"] = objects
    regions = dict(doc.get("regions") or {})
    clash = sorted(set(regions) & set(extra["regions"]))
    if clash:
        raise ValueError(f"{path}: the locker visitors' regions {clash} are regions of the scenario too")
    base_rate = float(extra.get("rate_per_h") or 0.0)
    scale = 1.0 if visitor_rate_per_h is None else (float(visitor_rate_per_h) / base_rate if base_rate > 0 else math.nan)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"{path.parent / VISITORS}: needs a positive rate_per_h to scale the visitors to {visitor_rate_per_h}")
    for name, region in extra["regions"].items():
        region = copy.deepcopy(region)
        for frame in (region.get("config") or {}).get("rate_profile") or []:
            frame["rate"] = round(float(frame.get("rate", 0.0)) * scale, 10)
        regions[name] = region
    doc["regions"] = regions
    rate = base_rate * scale
    header = f"locker visitors of {VISITORS} merged in: {len(extra['regions'])} regions, {rate:g} visitors per hour in all, target {target['name']} at ({x}, {y}), facing {theta} rad, front offset {front} m"
    return doc, header


def scenario_dir(site: Site, scenario: str, source: Path | None = None) -> Path:
    """The directory of `scenario` to copy into layout copies: under `source` (a world directory), else the site's,
    else that of the world the site's name resolves to (the twin's draft crowd is local only)."""
    from ..scenario.cli import world_root

    roots = [Path(source)] if source is not None else [site.root]
    if source is None:
        try:
            roots.append(world_root(None, site.key))
        except (FileNotFoundError, ValueError):
            pass
    for root in roots:
        if (root / "scenarios" / scenario / "scenario.yaml").is_file():
            return root / "scenarios" / scenario
    raise FileNotFoundError(f"no scenario {scenario!r} under {', '.join(str(r / 'scenarios') for r in roots)}")


def save_layouts(site: Site, layouts: Iterable[place.Layout], scenario: Path, *, visitor_rate_per_h: float | None = None) -> dict[str, Path]:
    """Save each layout copy (`place.save_layout`, assets hardlinked) with `scenario` (a scenario directory) copied into
    its `scenarios/`, the locker visitors merged in (`crowd`); the copies by layout id. ValueError when the assets were
    copied, not hardlinked (the copy and the site on different mounts)."""
    import yaml

    out: dict[str, Path] = {}
    scenario = Path(scenario)
    for layout in layouts:
        if layout.layout_id in out:
            continue
        _check_links(site, place.layout_name(site, layout))
        merged = crowd(layout, scenario, visitor_rate_per_h)
        path = place.save_layout(site, layout)
        target = path / "scenarios" / scenario.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(scenario, target, ignore=shutil.ignore_patterns(VISITORS) if merged is not None else None)
        if merged is not None:
            doc, header = merged
            lead = itertools.takewhile(lambda line: line.startswith("#") or not line.strip(), (scenario / "scenario.yaml").read_text().splitlines())
            comments = [line for line in lead if line.startswith("#")]
            (target / "scenario.yaml").write_text("\n".join([f"{MARKER}: {header}", *comments]) + "\n" + yaml.safe_dump(doc, sort_keys=False, width=10_000))
        if not hardlinked(site.root, path):
            raise ValueError(f"{path}: the assets of {site.root} were copied, not hardlinked; save on the site's mount")
        out[layout.layout_id] = path
    return out


def _check_links(site: Site, name: str) -> None:
    """ValueError when the copy `name` of `site` would get its assets copied, not hardlinked: its root is on another
    file system or mount than the site's (`edit.save` copies then, gigabytes for the twin's splats)."""
    import os

    from arena_simulation_setup.tree.World import WorldIdentifier

    files = [p for p in (site.root / "assets").rglob("*") if p.is_file()]
    if not files:
        return
    target = WorldIdentifier(name).resolve_write_sync().path.parent
    target.mkdir(parents=True, exist_ok=True)
    probe = target / f".linkcheck-{os.getpid()}"
    try:
        os.link(files[0], probe)
    except OSError as exc:
        raise ValueError(f"{target} cannot hardlink the assets of {site.root} ({exc.strerror}): set ARENA_ASSETS_DIR_LOCAL to a root on the site's mount") from None
    probe.unlink()


def hardlinked(source: Path, copy: Path) -> bool:
    """Whether every file under `source/assets` is the same file under `copy/assets` (True without assets)."""
    files = [p for p in (Path(source) / "assets").rglob("*") if p.is_file()]
    return all((Path(copy) / "assets" / f.relative_to(Path(source) / "assets")).stat().st_ino == f.stat().st_ino for f in files)


# --- the fleet and the suites ------------------------------------------------------------------------------------------


def fleet_config(rate_per_h: float, *, pickup_share: float = PICKUP_SHARE, load_s: float = LOAD_S, unload_s: float = UNLOAD_S, charge_below: float = CHARGE_BELOW, charge_until: float = CHARGE_UNTIL) -> dict[str, Any]:
    """The `fleet` mode block of a layout stage (`task.fleet.*`): homes on the bay slots and chargers, the layout's
    loading point and locker by object name (`place.CATALOG`)."""
    if not 0.0 <= charge_below <= charge_until <= 100.0:
        raise ValueError(f"charge_below {charge_below} and charge_until {charge_until} must satisfy 0 <= below <= until <= 100")
    if rate_per_h <= 0:
        raise ValueError(f"rate must be positive, got {rate_per_h}")
    (loading,) = place.CATALOG["loading_point"].members
    (locker,) = place.CATALOG["locker"].members
    return {
        "homes": list(HOMES),
        "loading": loading,
        "locker": locker,
        "rate": float(rate_per_h),
        "pickup_share": float(pickup_share),
        "load_s": float(load_s),
        "unload_s": float(unload_s),
        "charge_below": float(charge_below),
        "charge_until": float(charge_until),
    }


def fleet_rate(travel_m: float, *, robots: int = len(HOMES), speed_mps: float = SPEED_MPS, load_s: float = LOAD_S, unload_s: float = UNLOAD_S, utilisation: float = UTILISATION) -> float:
    """A rate (jobs per hour) at `utilisation` of the fleet's capacity in the wide premise layout (every element at the
    widest section, `search.premise_pair`), whose expected robot travel per job is `travel_m`: a job takes its travel at
    `speed_mps` plus the holds. The same rate runs in every layout (§8); a layout slower than the wide one may build a
    backlog, which the restricted mean task time T shows. Charging time and the crowd are not counted, so this is an
    upper bound: the premise check measures the capacity in the same layout (`evaluation infra premise`)."""
    if not math.isfinite(travel_m):
        raise ValueError("the layout has no finite robot travel")
    job_s = travel_m / speed_mps + load_s + unload_s
    return round(utilisation * robots * 3600.0 / job_s, 2)


def fleet_stage(world: str, seed: int, fleet: Mapping[str, Any], *, scenario: str = SCENARIO, duration: float = EPISODE_S, robot: str = ROBOT) -> dict[str, Any]:
    """The stage of layout copy `world` at episode seed `seed` with the fleet in `fleet` mode (`fleet_config`)."""
    return {
        "name": SU.stage_name(world, seed),
        "map": world,
        "robot": robot,
        "modes": {"robots": "fleet", "obstacles": "scenario"},
        "episodes": 1,
        "seed": int(seed),
        "timeout": float(duration),
        "config": {"scenario": {"file": scenario}, "fleet": copy.deepcopy(dict(fleet))},
    }


def _doc(stages: Sequence[Mapping[str, Any]], references: bool | list[str], side: str | None = None) -> dict[str, Any]:
    names = [s["name"] for s in stages]
    if dup := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"stages {dup} appear twice")
    launch = dict(LAUNCH)
    if side is not None:
        launch["humansim.global_planner.lane_side"] = side  # the site's crowd side (SiteConfig.crowd_side)
    return {"references": copy.copy(references), "launch": launch, "record": copy.deepcopy(RECORD), "stages": [dict(s) for s in stages]}


def reference_suite(site: str = SITE, seeds: Iterable[int] = SELECTION_SEEDS, *, scenario: str = SCENARIO, duration: float = EPISODE_S, robot: str = ROBOT, side: str | None = None) -> dict[str, Any]:
    """The site as is without robots at each seed: `<site>__s<seed>`, the robots parked off the map."""
    return _doc([SU.stage(site, seed, scenario, duration, robot) for seed in seeds], references=False, side=side)


def layout_suite(worlds: Iterable[str], fleet: Mapping[str, Any], seeds: Iterable[int] = SELECTION_SEEDS, *, scenario: str = SCENARIO, duration: float = EPISODE_S, robot: str = ROBOT, side: str | None = None) -> dict[str, Any]:
    """Each layout copy in `worlds` at each seed with the fleet (`fleet_stage`), and per stage the runner's run without
    robots (`REFERENCES`). The premise check, the selection and the evaluation are such suites."""
    seeds = list(seeds)
    return _doc([fleet_stage(world, seed, fleet, scenario=scenario, duration=duration, robot=robot) for world in dict.fromkeys(worlds) for seed in seeds], references=list(REFERENCES), side=side)


def _runs_per_episode(doc: Mapping[str, Any]) -> int:
    """How many runs each episode of a suite's stage makes with one contestant: itself and the reference runs."""
    references = doc.get("references")
    if references is True:
        return 3
    return 1 + len(set(references)) if isinstance(references, list) else 1


def episodes(doc: Mapping[str, Any]) -> int:
    """The episodes a suite runs with one contestant: each stage's, and its reference runs'."""
    return sum(int(s.get("episodes", 1)) for s in doc.get("stages") or []) * _runs_per_episode(doc)


def sim_hours(doc: Mapping[str, Any]) -> float:
    """The sim hours a suite runs with one contestant: its episodes times their timeouts."""
    return round(sum(int(s.get("episodes", 1)) * float(s.get("timeout", 0.0)) for s in doc.get("stages") or []) * _runs_per_episode(doc) / 3600.0, 2)


def contest(capacity_wh: float = CAPACITY_WH, planner: Mapping[str, str] | None = None, *, soc_min: float = SOC_MIN, soc_max: float = SOC_MAX) -> list[dict[str, Any]]:
    """The layout suites' contest: the fleet's planner, every robot's battery scaled to `capacity_wh`, and the start
    charge each robot draws per episode seed from [`soc_min`, `soc_max`] % (§8)."""
    if not 0.0 <= soc_min <= soc_max <= 100.0:
        raise ValueError(f"soc_min {soc_min} and soc_max {soc_max} must satisfy 0 <= min <= max <= 100")
    return [
        {
            "name": FLEET_CONTESTANT,
            "mobile": dict(PLANNER if planner is None else planner),
            "task.episode.battery.capacity_wh": float(capacity_wh),
            "task.episode.battery.soc.min": float(soc_min),
            "task.episode.battery.soc.max": float(soc_max),
        }
    ]


def contest_literal(entries: Sequence[Mapping[str, Any]]) -> str:
    """`entries` as the one-line YAML `--contest` takes."""
    import yaml

    return yaml.safe_dump(list(entries), default_flow_style=True, width=10_000, sort_keys=False).strip()


def write(doc: Mapping[str, Any], out: Path, header: Sequence[str] = (), *, force: bool = False) -> Path:
    """Write a suite to `out`, replacing only a suite this wrote unless `force`."""
    import yaml

    out = Path(out)
    if out.is_file() and not force and not out.read_text().startswith(MARKER):
        raise ValueError(f"{out} was not written by flowedit infra, pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([MARKER, *(f"# {line}" for line in header)]) + "\n" + yaml.safe_dump(dict(doc), sort_keys=False))
    return out


def default_out(name: str) -> Path:
    """`<imported assets root>/suites/<name>.yaml`, where `arena evaluation benchmark --suite <name>` finds it."""
    from arena_simulation_setup.tree import imported_root

    return imported_root() / SU.SUITES / f"{name}.yaml"


def split_parts(items: Iterable[str], weights: Sequence[int]) -> list[list[str]]:
    """`items` (layout copies) over one part per weight, in proportion to the weights (e.g. the lanes of the machine
    each part runs on): in sorted order, each item to the part furthest below its share (the fewest items per weight,
    the lower part on ties), an empty part first so none stays empty. The same items and weights always give the same
    parts."""
    weights = [int(w) for w in weights]
    if len(weights) < 2 or any(w <= 0 for w in weights):
        raise ValueError(f"parts need two or more positive weights, not {weights}")
    items = sorted(dict.fromkeys(items))
    if len(items) < len(weights):
        raise ValueError(f"{len(items)} layout copies cannot fill {len(weights)} parts")
    parts: list[list[str]] = [[] for _ in weights]
    for item in items:
        best = min(range(len(weights)), key=lambda i: (bool(parts[i]), (len(parts[i]) + 1) / weights[i], i))
        parts[best].append(item)
    return parts


def part_name(suite: str, index: int) -> str:
    """The suite name of part `index` (from 1) of `suite`: `<suite>_p<index>`."""
    return f"{suite}_p{index}"


def manifest_path(suite: Path) -> Path:
    """The manifest a layout suite's writer leaves beside it: `<suite>.layouts.json`."""
    suite = Path(suite)
    return suite.with_name(f"{suite.stem}.layouts.json")


def picks_record(picks: Mapping[str, Scored | Sequence[Scored]], ranking: Mapping[str, float] | None = None) -> list[dict[str, Any]]:
    """The layouts by role (`robot_only`, `joint`, `random`, ...) as plain data, each with its role and rank, and its
    fallback ranking score (`ranking`, by layout id) where given."""
    out = []
    for role, chosen in picks.items():
        for rank, scored in enumerate([chosen] if isinstance(chosen, Scored) else chosen, start=1):
            record = {"role": role, "rank": rank, **scored.record()}
            if ranking is not None and scored.layout_id in ranking:
                value = ranking[scored.layout_id]
                record["ranking_score"] = round(float(value), 4) if math.isfinite(value) else None
            out.append(record)
    return out


def runs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The layouts a manifest's suite runs, one entry per layout id in run order (a layout with two roles, such as the
    robot-only pick in the joint top 10, keeps both in `roles`)."""
    out: dict[str, dict[str, Any]] = {}
    for record in manifest.get("runs") or []:
        entry = out.setdefault(record["layout_id"], {**record, "roles": []})
        entry["roles"].append(record["role"])
    return list(out.values())


__all__ = [
    "entrance_crowd",
    "walk_area",
    "entrance_shares",
    "write_crowd",
    "CAPACITY_WH",
    "CHARGE_BELOW",
    "CHARGE_UNTIL",
    "EPISODE_S",
    "EVALUATION_SEEDS",
    "EVALUATION_SITE_SUITE",
    "EVALUATION_SUITE",
    "CROWD_RATE",
    "CrowdDensity",
    "CROWD_SIDES",
    "GROUPS",
    "LAUNCH",
    "LAYOUT_SUITE",
    "PREMISE_SEEDS",
    "PREMISE_SUITE",
    "RECORD",
    "REFERENCES",
    "REFERENCE_SUITE",
    "ROBOT",
    "SCENARIO",
    "SEEDS",
    "SELECTION_SEEDS",
    "SITE",
    "SITES_DIR",
    "SOC_MAX",
    "SOC_MIN",
    "VISITORS",
    "WARMUP_S",
    "SiteConfig",
    "contest",
    "contest_literal",
    "crowd",
    "default_out",
    "episodes",
    "fleet_config",
    "fleet_rate",
    "fleet_stage",
    "hardlinked",
    "keep_out",
    "layout_suite",
    "locker_front",
    "manifest_path",
    "part_name",
    "picks_record",
    "reference_suite",
    "runs",
    "save_layouts",
    "split_parts",
    "scenario_dir",
    "sim_hours",
    "site_config",
    "visitor_sources",
    "visitors",
    "write",
]
