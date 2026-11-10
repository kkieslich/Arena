"""INFRA Stage 1 suites (study design §5, §8, §10): seed-paired arena_evaluation benchmark suites on a site (the twin
smu_link_full), with `flowedit suite`'s stage names `<world>__s<seed>`.

The study runs in this order, each suite its own benchmark run:

1. `reference_suite` (`infra_h1_site`): the site as is without robots on the selection seeds. Its flow cells
   (`flow_cells.parquet`) give the occupancy the joint search's fallback ranks with (`search.Occupancy.from_flow_cells`),
   and its runs are the reference every layout's element part pairs with (§6).
2. the premise check (`infra_h1_premise`, §10): the two most different layouts (`search.premise_pair`) on the premise
   seeds, with the fleet and without robots. With the site's runs at those seeds it is §10's 15 episodes. It checks
   that layouts differ by more than the seed spread and measures the task rate.
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
#: Launch args of every suite: lockstep for seed pairing, people see the robots (§8: PEDS robot visibility on)
LAUNCH = {"sim": "flatsim", "headless": "true", "lockstep": "true", "human.robots.visible": "true"}
#: The reference runs the layout suites ask the runner for: each stage without robots (the traffic part's reference,
#: §6); the fleet without pedestrians is never read
REFERENCES = ["unhindered_peds"]
#: What the recorder keeps of every INFRA run (kkieslich/arena-evaluation's suite `record` block): the INFRA measures
#: read the engine's pedestrian tracks (`agent_states`), the robots and the semantics (the fleet's task log), not the
#: pedestrians' skeletons and bodies on /tf, their named frames, `arena_peds` or their animation states, which grow
#: with the crowd (about 94 % of a pilot recording at 15-20 people)
RECORD = {"exclude": ["peds", "animation_states"], "tf_frames": [[r"_agent_\d+$", -1.0], [r"^env_\d+/[^/]+$", -1.0]]}
#: The study battery and charger (Kolja, 9 Nov: 22 Wh, 175 W; the catalog writes the charger power into the world), and
#: the start charge each robot draws per episode seed (§8: 40-100 %)
CAPACITY_WH = 22.0
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
#: The share of the fleet's capacity the rate asks for (§8: near 75 %)
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
#: The parcel locker's object, which the visitors' target stands in front of
LOCKER = "locker"


# --- the site config -------------------------------------------------------------------------------------------------


@attrs.frozen
class SiteConfig:
    """Where each element may go on a site, by its function (`search.Placement` per element name), the zones no element
    goes to (`exclude`, `fnmatch` patterns), and the crowd scenario every stage runs. `path` is the file it came from."""

    site: str
    scenario: str = SCENARIO
    exclude: tuple[str, ...] = place.EXCLUDED_ZONES
    placements: Mapping[str, Placement] = attrs.field(factory=dict)
    path: Path | None = None

    @classmethod
    def parse(cls, doc: Mapping[str, Any], path: Path | None = None) -> SiteConfig:
        import shapely

        where = f"{path}: " if path is not None else ""
        unknown = sorted(set(doc) - {"site", "scenario", "exclude", "elements"})
        if unknown:
            raise ValueError(f"{where}unknown keys {unknown} (site, scenario, exclude, elements)")
        if not doc.get("site"):
            raise ValueError(f"{where}no site")
        placements: dict[str, Placement] = {}
        for name, entry in (doc.get("elements") or {}).items():
            if name not in (*SEARCHED, LOADING):
                raise ValueError(f"{where}element {name!r} is not one of {', '.join((*SEARCHED, LOADING))}")
            entry = entry or {}
            bad = sorted(set(entry) - {"zones", "areas", "pin"})
            if bad:
                raise ValueError(f"{where}{name}: unknown keys {bad} (zones, areas, pin)")
            zones = entry.get("zones")
            if zones is not None and (isinstance(zones, str) or not all(isinstance(z, str) for z in zones)):
                raise ValueError(f"{where}{name}: zones must be a list of zone names or patterns")
            areas = []
            for area in entry.get("areas") or []:
                points = [(float(x), float(y)) for x, y in area]
                polygon = shapely.Polygon(points) if len(points) >= 3 else None
                if polygon is None or not polygon.is_valid or polygon.area <= 0:
                    raise ValueError(f"{where}{name}: area {area} is no valid polygon")
                areas.append(polygon)
            pin = entry.get("pin")
            if pin is not None:
                if len(pin) != 2:
                    raise ValueError(f"{where}{name}: pin must be [x, y]")
                pin = (float(pin[0]), float(pin[1]))
            placements[name] = Placement(None if zones is None else tuple(zones), tuple(areas), pin)
        if LOADING not in placements or placements[LOADING].pin is None:
            raise ValueError(f"{where}the loading point needs a pin (§8: it is not searched)")
        return cls(str(doc["site"]), str(doc.get("scenario") or SCENARIO), tuple(doc.get("exclude", place.EXCLUDED_ZONES)), placements, path)

    @classmethod
    def load(cls, path: Path) -> SiteConfig:
        import yaml

        path = Path(path)
        return cls.parse(yaml.safe_load(path.read_text()) or {}, path)

    def record(self) -> dict[str, Any]:
        """The config as plain data, for a manifest."""
        import shapely

        return {
            "site": self.site,
            "path": None if self.path is None else str(self.path),
            "scenario": self.scenario,
            "exclude": list(self.exclude),
            "elements": {
                name: {
                    "zones": None if p.zones is None else list(p.zones),
                    "areas": [shapely.get_coordinates(a.exterior).tolist() for a in p.areas],
                    "pin": None if p.pin is None else list(p.pin),
                }
                for name, p in self.placements.items()
            },
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


def fleet_rate(travel_m: Iterable[float], *, robots: int = len(HOMES), speed_mps: float = SPEED_MPS, load_s: float = LOAD_S, unload_s: float = UNLOAD_S, utilisation: float = UTILISATION) -> float:
    """A rate (jobs per hour) at `utilisation` of the fleet's capacity in the slowest of the layouts, whose expected
    robot travel per job is `travel_m`: a job takes its travel at `speed_mps` plus the holds. The same rate runs in
    every layout (§8), so the slowest one sets it and no layout builds a backlog. Charging time is not counted: the
    premise check measures the real one."""
    slowest = max(travel_m)
    if not math.isfinite(slowest):
        raise ValueError("a layout has no finite robot travel")
    job_s = slowest / speed_mps + load_s + unload_s
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


def _doc(stages: Sequence[Mapping[str, Any]], references: bool | list[str]) -> dict[str, Any]:
    names = [s["name"] for s in stages]
    if dup := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"stages {dup} appear twice")
    return {"references": copy.copy(references), "launch": dict(LAUNCH), "record": copy.deepcopy(RECORD), "stages": [dict(s) for s in stages]}


def reference_suite(site: str = SITE, seeds: Iterable[int] = SELECTION_SEEDS, *, scenario: str = SCENARIO, duration: float = EPISODE_S, robot: str = ROBOT) -> dict[str, Any]:
    """The site as is without robots at each seed: `<site>__s<seed>`, the robots parked off the map."""
    return _doc([SU.stage(site, seed, scenario, duration, robot) for seed in seeds], references=False)


def layout_suite(worlds: Iterable[str], fleet: Mapping[str, Any], seeds: Iterable[int] = SELECTION_SEEDS, *, scenario: str = SCENARIO, duration: float = EPISODE_S, robot: str = ROBOT) -> dict[str, Any]:
    """Each layout copy in `worlds` at each seed with the fleet (`fleet_stage`), and per stage the runner's run without
    robots (`REFERENCES`). The premise check, the selection and the evaluation are such suites."""
    seeds = list(seeds)
    return _doc([fleet_stage(world, seed, fleet, scenario=scenario, duration=duration, robot=robot) for world in dict.fromkeys(worlds) for seed in seeds], references=list(REFERENCES))


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
    "CAPACITY_WH",
    "CHARGE_BELOW",
    "CHARGE_UNTIL",
    "EPISODE_S",
    "EVALUATION_SEEDS",
    "EVALUATION_SITE_SUITE",
    "EVALUATION_SUITE",
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
    "picks_record",
    "reference_suite",
    "runs",
    "save_layouts",
    "scenario_dir",
    "sim_hours",
    "site_config",
    "visitor_sources",
    "visitors",
    "write",
]
