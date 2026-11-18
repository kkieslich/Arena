"""Layouts of the entrance search as worlds, with their parcel recipients (study design Part A: A3, A6, A8).

`flowedit infra search` (`demand.py`) reports a layout as a plain record (`EntranceSearch.record`): the charger pair,
the bay pair, the loading point and several lockers with their poses, and the recipient entrances each locker serves.
This module reads such records (`layout_records`), turns one back into a `place.Layout` on the search space it came
from (`layout_of`), and saves the world copy with one crowd scenario per episode seed (`save_layout`), each the site's
crowd plus the seed's parcel recipients.

The jobs of a seed (`jobs`) are drawn here from the seed alone: release times of a Poisson stream at the job rate,
each job a delivery or a pickup, each addressed to a recipient entrance drawn with equal weights. The layout does not
enter, so every layout has the same jobs at the same release times (A2), and three streams of the seed (release
times, kinds, entrances) keep a change of one setting from moving the others.

The recipients of a seed (`recipients`) follow that job list, never what a fleet delivered: each delivery brings one
recipient (A3), who comes in at the job's recipient entrance `offset_s` after the job's release, walks to the locker
that serves this entrance, stands there and walks back to the entrance. A layout's run with the fleet and its run
without robots at one seed run the same scenario, so they have the same recipients; the site as is has none. A
recipient's walker group is drawn with the crowd's mix (`crowd_mix`) from a stream of the seed and the job, so it is
the same in every layout. Its agent type extends the group's type of the crowd: a wheelchair user takes step-free
floor only, as every walker of that type does, the others may take stairs and escalators.

In the scenario (`recipient_crowd`) each locker gets a target in front of it (the first study's locker visitors'
target, `infra.spot_front`), and each recipient is a source of its own on its entrance's area that releases exactly
one walker at its time: its rate is 0 until then and `RELEASE_RATE` from `RELEASE_RAMP_S` later, capped at one walker
(`max_total`). The engine draws each source from a stream of its own name, so the site's sources release the same
walkers with or without the recipients. A recipient's source is named `recipient_<job>__<entrance>_locker_<group>`:
the recorded tracks carry the source name, the manifest lists the names as `visitor_sources`, and the analysis leaves
those walkers out of the paired delay and scores their walk apart (kkieslich/arena-evaluation, `processing/infra.py`)."""

from __future__ import annotations

import copy
import itertools
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from . import infra as IN
from . import place
from .demand import numbered
from .search import LOADING, SEARCHED

if TYPE_CHECKING:
    from .edit import Site
    from .place import Layout, Spot
    from .search import Search

CHARGERS, BAYS, LOCKER = SEARCHED
#: Seconds from a job's release to its recipient's arrival at the recipient entrance. No source fixes it and the first
#: study had none (its visitors came at a rate of their own). The default is the warm-up: the largest offset at which
#: the recipients' stream is complete from the first scored second on. A placeholder until the capacity run gives the
#: time a delivery takes
OFFSET_S = IN.WARMUP_S
#: How long a recipient stands at the locker [s]: the first study's locker visitors' (`infra.entrance_crowd`)
STAND_S: Mapping[str, float] = {"mean": 30.0, "std": 10.0, "clip_low": 15.0, "clip_high": 60.0}
#: How long a recipient waits for the locker before it gives up and walks back [s]: the first study's
PATIENCE_S = 1200.0
#: The target in front of a locker as HumanSim takes it, and how far the first in line stands from the locker's
#: footprint [m]: the first study's (`infra.entrance_crowd`)
TARGET: Mapping[str, Any] = {"type": "parcel_locker", "capacity": 1, "interaction_radius": 0.8, "formation": {"type": "line", "params": {"base_step": 0.9}}}
CLEARANCE_M = 0.35
#: The name of locker 1's target (the first study's) and of the others'
TARGET_NAME = "locker_visit"
#: The job kinds that bring a recipient (A3: each delivered parcel)
RECIPIENT_KINDS = ("delivery",)
#: A recipient's source: the rate it jumps to at the recipient's time and how long the jump takes. The engine releases
#: Poisson(rate x tick) walkers per tick, capped at one in all: at 2000 per s and a 0.05 s tick the first tick after
#: the time releases it but for a chance of e^-100 [1/s], [s]
RELEASE_RATE = 2000.0
RELEASE_RAMP_S = 0.05
#: Recipients' times are whole multiples of this, on the engine's tick grid [s]
TIME_STEP_S = 0.1
#: The fleet's battery as the manufacturer gives it (study design A4: the jackal's 270 Wh, charged in 4 h; the robot
#: model's power.yaml holds the 270 Wh and no charge time) and a parked jackal's standby draw (the first study's
#: 42.2 W, `place.CHARGE_POWER_W`). The chargers of a layout copy charge with the power that fills an empty parked
#: robot in the charge time: 270 / 4 + 42.2 = 109.7 W (the first study's catalog: 290 W for its 36 Wh battery)
BATTERY_WH = 270.0
CHARGE_TIME_H = 4.0
STANDBY_W = 42.2
CHARGE_POWER_W = round(BATTERY_WH / CHARGE_TIME_H + STANDBY_W, 1)
#: The seed sequence key of the job streams, and each stream's
_JOBS_KEY = 0x1AF4A
_STREAMS = {"release": 1, "kind": 2, "entrance": 3, "group": 4}
PREFIX = "recipient"


@attrs.frozen
class Job:
    """One job of a seed: its index in release order, `delivery` (loading point to the locker that serves the recipient
    entrance) or `pickup` (the same way back), its release time from the episode start and the recipient entrance it
    is addressed to."""

    index: int
    kind: str
    release_s: float
    entrance: str

    def record(self) -> dict[str, Any]:
        return attrs.asdict(self)


@attrs.frozen
class Recipient:
    """The recipient of a job: the recipient entrance it comes in and leaves by, the locker (object name) that serves
    this entrance, its walker group and that group's agent type, the time it comes in, and its source's name."""

    job: int
    entrance: str
    locker: str
    group: str
    agent_type: str
    t_s: float

    @property
    def source(self) -> str:
        return f"{PREFIX}_{self.job:03d}__{self.entrance}_locker_{self.group}"

    def record(self) -> dict[str, Any]:
        return {**attrs.asdict(self), "source": self.source}


# --- layouts from the search's report ----------------------------------------------------------------------------------


def _placement(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (tuple(sorted((str(s["element"]), str(s["spot"])) for s in record["spots"])), tuple(sorted((str(k), tuple(v)) for k, v in record["serves"].items())))


def layout_records(report: Any) -> dict[str, dict[str, Any]]:  # noqa: ANN401 - parsed JSON
    """Every layout record (`demand.EntranceSearch.record`) in a search report, by layout id: any mapping with a
    `layout_id`, `spots` and `serves`, wherever it sits (`flowedit infra search --out` keeps them under
    `reaches[].layouts[]`). A layout reported more than once (two roles, two reaches) must have the same spots and
    serve the same entrances each time."""
    out: dict[str, dict[str, Any]] = {}

    def walk(node: Any) -> None:  # noqa: ANN401
        if isinstance(node, Mapping):
            if isinstance(node.get("layout_id"), str) and isinstance(node.get("spots"), list) and isinstance(node.get("serves"), Mapping):
                known = out.setdefault(node["layout_id"], dict(node))
                if _placement(known) != _placement(node):
                    raise ValueError(f"layout {node['layout_id']} is reported twice with other spots or other entrances per locker")
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(report)
    return out


def read_report(path: Path) -> dict[str, dict[str, Any]]:
    """`layout_records` of the report file at `path`."""
    return layout_records(json.loads(Path(path).read_text()))


def _locker_number(element: str) -> int | None:
    """Which locker of a layout the element `locker` or `locker_<n>` is (`demand.numbered`), None for another element."""
    if element == LOCKER:
        return 1
    head, _, tail = element.rpartition("_")
    return int(tail) if head == LOCKER and tail.isdigit() and int(tail) >= 2 else None


def layout_of(search: Search, record: Mapping[str, Any]) -> Layout:
    """The layout of a record on `search`, the search space the record's search ran on (the same site, site config,
    crowd scenario and locker rules): each spot is found by its id, which holds the element's and objects' names and
    poses to the mm, the lockers numbered as the record numbers them. ValueError for a spot the search space does not
    have, a loading point other than the search's, or a layout whose id is not the record's."""
    layout_id = record["layout_id"]
    chosen: list[Spot] = []
    for entry in record["spots"]:
        name, spot_id = str(entry["element"]), str(entry["spot"])
        number = _locker_number(name)
        if name == LOADING:
            pool: Sequence[Spot] = (search.loading,)
        elif name in (CHARGERS, BAYS):
            pool = search.spots[name]
        elif number is not None:
            pool = [numbered(spot, number) for spot in search.spots[LOCKER]]
        else:
            raise ValueError(f"layout {layout_id}: {name!r} is no element of a layout")
        found = next((spot for spot in pool if spot.id == spot_id), None)
        if found is None:
            at = ", ".join(f"{o['name']} ({o['x']:.2f}, {o['y']:.2f})" for o in entry.get("objects") or [])
            raise ValueError(f"layout {layout_id}: the search space has no {name} spot {spot_id} ({at}) among its {len(pool)}: the record comes from another site, site config, crowd scenario, loading point or locker rules")
        chosen.append(found)
    layout = place.Layout.of(chosen)
    if layout.layout_id != layout_id:
        raise ValueError(f"the spots of record {layout_id} give layout {layout.layout_id}")
    return layout


def with_charge_power(layout: Layout, power_w: float) -> Layout:
    """`layout` with every charger charging at `power_w`. The layout's id stays: it holds the poses, not the power."""
    if power_w <= 0:
        raise ValueError(f"the charge power must be positive, got {power_w}")
    return place.Layout.of([attrs.evolve(s, element=attrs.evolve(s.element, charge_power_w=float(power_w))) if s.element.charge_power_w is not None else s for s in layout.spots])


def serves_of(record: Mapping[str, Any], layout: Layout | None = None) -> dict[str, tuple[str, ...]]:
    """The recipient entrances each locker of a record serves, by the locker's object name (`locker_1`, `locker_2`,
    ...). ValueError when an entrance has two lockers, or (with `layout`) a name is no locker of the layout."""
    serves = {str(locker): tuple(str(e) for e in entrances) for locker, entrances in record["serves"].items()}
    every = [e for entrances in serves.values() for e in entrances]
    twice = sorted({e for e in every if every.count(e) > 1})
    if twice:
        raise ValueError(f"layout {record.get('layout_id')}: entrances {twice} are served by two lockers")
    if layout is not None:
        lockers = {o.name for s in layout.spots if _locker_number(s.element.name) is not None for o in s.objects}
        unknown = sorted(set(serves) - lockers)
        if unknown:
            raise ValueError(f"layout {layout.layout_id}: {unknown} serve entrances but are no lockers of the layout ({sorted(lockers)})")
    return serves


# --- jobs and recipients of a seed -------------------------------------------------------------------------------------


def _rng(seed: int, stream: str, *more: int) -> Any:  # noqa: ANN401 - a numpy Generator
    import numpy as np

    if int(seed) < 0:
        raise ValueError(f"seed must not be negative, got {seed}")
    return np.random.default_rng(np.random.SeedSequence([int(seed), _JOBS_KEY, _STREAMS[stream], *(int(m) for m in more)]))


def jobs(seed: int, entrances: Sequence[str], *, rate_per_h: float, pickup_share: float = IN.PICKUP_SHARE, episode_s: float = IN.EPISODE_S) -> tuple[Job, ...]:
    """The jobs of episode seed `seed` released before `episode_s`, in release order: a Poisson stream at `rate_per_h`
    jobs per hour in all, a share `pickup_share` of them pickups, each addressed to one of `entrances` with equal
    weights (the same job rate per recipient entrance, A4). Drawn from the seed alone, so the same in every layout:
    the release times, the kinds and the entrances each from a stream of their own. Another rate moves the same jobs
    in time, another episode length cuts the same list elsewhere."""
    entrances = list(entrances)
    if not entrances:
        raise ValueError("no recipient entrances to address jobs to")
    if rate_per_h <= 0 or not 0.0 <= pickup_share <= 1.0 or episode_s <= 0:
        raise ValueError(f"rate_per_h {rate_per_h} and episode_s {episode_s} must be positive and pickup_share {pickup_share} lie in [0, 1]")
    release, kind, where = (_rng(seed, stream) for stream in ("release", "kind", "entrance"))
    out: list[Job] = []
    now = 0.0
    while True:
        now += float(release.exponential(1.0)) * 3600.0 / float(rate_per_h)
        if now >= episode_s:
            return tuple(out)
        out.append(Job(len(out), "pickup" if float(kind.random()) < pickup_share else "delivery", round(now, 3), entrances[int(where.integers(len(entrances)))]))


def crowd_mix(scenario: Mapping[str, Any]) -> dict[str, tuple[str, float]]:
    """The walker groups of a crowd scenario (its document): per group (`infra.GROUPS`) its agent type and its share of
    the inflow, from the sources' rates at their first keyframe. ValueError for a source whose agent type is no
    group's, or a scenario without inflow."""
    group_of = {agent_type: group for group, agent_type, _ in IN.GROUPS}
    rates: dict[str, float] = {}
    for name, region in (scenario.get("regions") or {}).items():
        if region.get("type") != "source":
            continue
        config = region.get("config") or {}
        agent_type = str((config.get("agent") or {}).get("agent_type", "adult"))
        if agent_type not in group_of:
            raise ValueError(f"source {name!r}: agent type {agent_type!r} is no walker group's ({', '.join(group_of)}); give the mix")
        profile = config.get("rate_profile") or []
        rates[group_of[agent_type]] = rates.get(group_of[agent_type], 0.0) + (float(profile[0].get("rate", 0.0)) if profile else 0.0)
    total = sum(rates.values())
    if total <= 0:
        raise ValueError("the scenario's sources have no inflow to take the walker groups' mix from")
    agent_types = {group: agent_type for group, agent_type, _ in IN.GROUPS}
    return {group: (agent_types[group], rates[group] / total) for group, _, _ in IN.GROUPS if rates.get(group, 0.0) > 0}


def recipients(
    seed: int,
    stream: Iterable[Job],
    serves: Mapping[str, Sequence[str]],
    mix: Mapping[str, tuple[str, float]],
    *,
    offset_s: float = OFFSET_S,
    episode_s: float = IN.EPISODE_S,
    kinds: Sequence[str] = RECIPIENT_KINDS,
) -> tuple[Recipient, ...]:
    """The recipients of a seed's jobs in a layout whose lockers serve the entrances as `serves` says (locker object
    name -> entrances): one per job of a kind in `kinds`, at the job's recipient entrance `offset_s` after its release
    (to `TIME_STEP_S`), bound for the locker that serves the entrance. A recipient due at or after `episode_s` is left
    out. The group is drawn with `mix` (`crowd_mix`) from a stream of the seed and the job's index: the same in every
    layout, whatever the other jobs are. ValueError for a job addressed to an entrance no locker serves."""
    import numpy as np

    locker_of = {entrance: locker for locker, entrances in serves.items() for entrance in entrances}
    groups = list(mix)
    shares = np.array([mix[g][1] for g in groups], dtype=float)
    if not groups or shares.min() < 0 or shares.sum() <= 0:
        raise ValueError(f"the walker groups' mix needs positive shares, not {dict(mix)}")
    shares = shares / shares.sum()
    out = []
    for job in stream:
        if job.kind not in kinds:
            continue
        if job.entrance not in locker_of:
            raise ValueError(f"job {job.index} is addressed to {job.entrance!r}, which no locker serves ({sorted(locker_of)})")
        t_s = round(round((job.release_s + offset_s) / TIME_STEP_S) * TIME_STEP_S, 3)
        if t_s >= episode_s:
            continue
        group = groups[int(_rng(seed, "group", job.index).choice(len(groups), p=shares))]
        out.append(Recipient(job.index, job.entrance, locker_of[job.entrance], group, mix[group][0], t_s))
    return tuple(out)


# --- the crowd scenario of a layout and seed ---------------------------------------------------------------------------


def target_name(locker: str) -> str:
    """The name of the target in front of the locker object `locker`: the first study's for `locker_1`, numbered as the
    locker for the others."""
    number = locker.rpartition("_")[2]
    return TARGET_NAME if number == "1" else f"{TARGET_NAME}_{number}"


def type_name(entrance: str, group: str) -> str:
    return f"{PREFIX}__{entrance}_{group}"


def scenario_name(scenario: str, seed: int, tag: str = "") -> str:
    """The name of a layout copy's crowd scenario at episode seed `seed`: `<scenario>__s<seed>`, with a tag
    `<scenario>__<tag>__s<seed>`."""
    return f"{scenario}__{tag}__s{int(seed)}" if tag else f"{scenario}__s{int(seed)}"


def _entrance_area(scenario: Mapping[str, Any], entrance: str) -> tuple[list[dict[str, float]], str, tuple[float, float]]:
    """The area walkers of `entrance` come in on and leave by, its sink's name and the point a leaving walker heads
    for, from the crowd scenario: the sink `<entrance>_sink` (`infra.entrance_crowd` writes one per entrance)."""
    from arena_simulation_setup.metrics.use_points import region_point

    sink = f"{entrance}_sink"
    region = (scenario.get("regions") or {}).get(sink)
    polygon = [{"x": float(p["x"]), "y": float(p["y"])} for p in (region or {}).get("polygon") or []]
    if region is None or region.get("type") != "sink" or len(polygon) < 3:
        raise ValueError(f"the crowd scenario has no sink {sink!r} with a polygon: recipients of {entrance} come in and leave on its area")
    x, y = region_point([(p["x"], p["y"]) for p in polygon])
    return polygon, sink, (round(float(x), 2), round(float(y), 2))


def recipient_crowd(
    scenario: Mapping[str, Any],
    layout: Layout,
    serves: Mapping[str, Sequence[str]],
    who: Sequence[Recipient],
    mix: Mapping[str, tuple[str, float]],
    *,
    stand_s: Mapping[str, float] = STAND_S,
    patience_s: float = PATIENCE_S,
    target: Mapping[str, Any] = TARGET,
    clearance_m: float = CLEARANCE_M,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], str]:
    """The crowd scenario of a layout copy with the recipients `who`: the site's scenario (its document) with a target
    in front of every locker that serves an entrance, one source per recipient (see the module doc) and nothing else
    changed; the recipients' agent types by file name, one per served entrance and walker group of `mix` (the same
    files at every seed); and a header line. ValueError for a name the scenario already has."""
    doc = copy.deepcopy(dict(scenario))
    spots = {o.name: s for s in layout.spots if _locker_number(s.element.name) is not None for o in s.objects}
    targets: dict[str, dict[str, Any]] = {}
    for locker in serves:
        if locker not in spots:
            raise ValueError(f"layout {layout.layout_id} has no locker {locker!r} ({sorted(spots)})")
        x, y, theta, front = IN.spot_front(spots[locker], clearance_m)
        block = copy.deepcopy(dict(target))
        block.setdefault("formation", {"type": "line", "params": {}}).setdefault("params", {})["front_offset"] = front
        targets[locker] = {"name": target_name(locker), "model": "", "pose": [x, y, theta], "humansim": block}
    names = {t["name"] for t in targets.values()}
    doc["objects"] = [o for o in doc.get("objects") or [] if o.get("name") not in names] + list(targets.values())
    regions = dict(doc.get("regions") or {})
    areas = {entrance: _entrance_area(scenario, entrance) for entrances in serves.values() for entrance in entrances}
    types: dict[str, dict[str, Any]] = {}
    for locker, entrances in serves.items():
        for entrance, (group, (agent_type, _)) in itertools.product(entrances, mix.items()):
            _, _, (cx, cy) = areas[entrance]
            name = type_name(entrance, group)
            types[f"{name}.yaml"] = {
                "name": name,
                "mode": "behavior_tree",
                "extends": agent_type,
                "initial_sequence": "visit",
                "sequences": {
                    "visit": {"steps": {"pickup": {"target": targets[locker]["name"], "interaction": "USE", "duration": {k: float(v) for k, v in stand_s.items()}, "patience": {"mean": float(patience_s), "std": 0.0}}}, "then": "leave", "on_failure": "leave"},
                    "leave": {"steps": {"exit": {"kind": "go_to", "target_pose": {"x": cx, "y": cy, "theta": 0.0}}}, "then": "leave", "on_failure": "leave"},
                },
            }
    for r in who:
        if r.entrance not in areas or f"{type_name(r.entrance, r.group)}.yaml" not in types:
            raise ValueError(f"recipient of job {r.job}: entrance {r.entrance!r} or group {r.group!r} is not served by the layout or not in the mix")
        if r.source in regions:
            raise ValueError(f"the crowd scenario already has a region {r.source!r}")
        polygon, sink, _ = areas[r.entrance]
        profile = [{"t": r.t_s, "rate": 0.0}, {"t": round(r.t_s + RELEASE_RAMP_S, 3), "rate": RELEASE_RATE}]
        regions[r.source] = {
            "type": "source",
            "polygon": copy.deepcopy(polygon),
            "config": {"rate_profile": profile, "max_concurrent": -1, "max_total": 1, "agent": {"agent_type": f"./types/{type_name(r.entrance, r.group)}.yaml", "sink_affinity": [{"sink": sink, "weight": 1.0}]}},
        }
    doc["regions"] = regions
    counts = {group: sum(r.group == group for r in who) for group in mix}
    header = f"parcel recipients: {len(who)} ({', '.join(f'{n} {g}' for g, n in counts.items())}), one source each ({PREFIX}_<job>__<entrance>_locker_<group>); targets " + ", ".join(f"{t['name']} at ({t['pose'][0]}, {t['pose'][1]}) for {locker}" for locker, t in targets.items())
    return doc, types, header


def _world_path(name: str) -> Path:
    from arena_simulation_setup.tree.World import WorldIdentifier

    return WorldIdentifier(name).resolve_write_sync().path


def save_layout(site: Site, layout: Layout, scenario: Path, crowds: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]], str]]) -> Path:
    """Save the copy of `layout` (`place.save_layout`, assets hardlinked) and write its crowd scenarios `crowds`
    (scenario name -> `recipient_crowd`'s result), each a copy of the site's scenario directory `scenario` (without the
    first study's locker visitors file) with its document and the recipients' agent types written in. Nothing recorded
    is replaced: a copy that exists is kept when it holds this layout as this code places it (its `edits.yaml`), and a
    scenario that exists must be the one that would be written; ValueError otherwise. Returns the copy's path."""
    import yaml

    from arena_simulation_setup.tree.World.write import to_yaml

    from .edit import EDITS, dump_edits

    scenario = Path(scenario)
    name = place.layout_name(site, layout)
    path = _world_path(name)
    if path.exists():
        expected = to_yaml(dump_edits(place.apply_layout(site, layout)))
        if not (path / EDITS).is_file() or (path / EDITS).read_text() != expected:
            raise ValueError(f"{path} exists and is not the copy of layout {layout.layout_id} this would write: nothing recorded is replaced, move it away first")
    else:
        IN._check_links(site, name)
        path = place.save_layout(site, layout)
    if not IN.hardlinked(site.root, path):
        raise ValueError(f"{path}: the assets of {site.root} were copied, not hardlinked; save on the site's mount")
    lead = [line for line in itertools.takewhile(lambda line: line.startswith("#") or not line.strip(), (scenario / "scenario.yaml").read_text().splitlines()) if line.startswith("#")]
    for target, (doc, types, header) in crowds.items():
        folder = path / "scenarios" / target
        text = "\n".join([f"{IN.MARKER}: {header}", *lead]) + "\n" + yaml.safe_dump(dict(doc), sort_keys=False, width=10_000)
        files = {f"types/{file}": yaml.safe_dump(dict(body), sort_keys=False) for file, body in types.items()}
        if folder.exists():
            differ = [f for f, body in {"scenario.yaml": text, **files}.items() if not (folder / f).is_file() or (folder / f).read_text() != body]
            if differ:
                raise ValueError(f"{folder} exists with another {', '.join(differ[:3])}: a written scenario is not replaced, pass another tag")
            continue
        shutil.copytree(scenario, folder, ignore=shutil.ignore_patterns(IN.VISITORS))
        (folder / "scenario.yaml").write_text(text)
        (folder / "types").mkdir(exist_ok=True)
        for file, body in files.items():
            (folder / file).write_text(body)
    return path


__all__ = [
    "CHARGE_POWER_W",
    "CLEARANCE_M",
    "OFFSET_S",
    "PATIENCE_S",
    "RECIPIENT_KINDS",
    "STAND_S",
    "TARGET",
    "Job",
    "Recipient",
    "crowd_mix",
    "jobs",
    "layout_of",
    "layout_records",
    "read_report",
    "recipient_crowd",
    "recipients",
    "save_layout",
    "scenario_name",
    "serves_of",
    "target_name",
    "type_name",
    "with_charge_power",
]
