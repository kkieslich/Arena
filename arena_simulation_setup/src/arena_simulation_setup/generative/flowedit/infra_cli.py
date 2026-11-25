"""`flowedit infra`: the INFRA Stage 1 suites (`infra.py`), in run order.

    flowedit infra reference                    the site as is without robots, selection seeds   -> infra_h1_site
    flowedit infra premise                      the 2 most different layouts, premise seeds      -> infra_h1_premise
    flowedit infra layouts --occupancy <run>    robot-only, joint top 10, 5 random, selection    -> infra_h1
    flowedit infra evaluation --pick <json>     the two picks and the site, evaluation seeds     -> infra_h1_eval(_site)
    flowedit infra evaluation --report <json> --layouts <ids> --seeds <seeds> --rate <jobs/h>
                                                layouts of `flowedit infra search` (study design Part A): their world
                                                copies with the seeds' parcel recipients, the site as is and each
                                                layout without robots as a crowd-only stage -> infra_rd_site, _without
    flowedit infra search --reach 30,40         demand at the recipient entrances (`demand.py`): the travel-optimal
                                                layout, the pedestrian-aware pick per travel margin and the
                                                practice-based layout per reach, as a report; writes no world or suite
    flowedit infra site-walls                   the walls the site config removes (`removed_walls`, objects that are
                                                not part of the site) out of the site's world and its layout copies,
                                                and the walls it adds (`added_walls`, gaps the site does not have) in

Each writes its suite under <imported assets root>/suites (where `arena evaluation benchmark --suite` finds it), the
layout suites a `layouts.json` manifest in its bundle (`suites/<suite>/suite.yaml`), and prints the benchmark call that runs it."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

from ..scenario.cli import world_root
from . import infra as IN
from . import place
from . import search as SE
from . import suite as SU
from . import walkway as WK
from .edit import open_site


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _seeds(text: str) -> list[int]:
    """`1-5`, `1,3,5` or a mix, as seeds in order."""
    out: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.strip().partition("-")
        out += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    if not out:
        raise argparse.ArgumentTypeError(f"no seeds in {text!r}")
    return list(dict.fromkeys(out))


def _weights(text: str) -> list[int]:
    """`4,6`: one positive weight per part."""
    try:
        out = [int(w) for w in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"parts are comma-separated whole numbers, not {text!r}") from None
    if len(out) < 2 or any(w <= 0 for w in out):
        raise argparse.ArgumentTypeError(f"parts need two or more positive weights, not {text!r}")
    return out


def _point(text: str) -> tuple[float, float]:
    x, _, y = text.partition(",")
    return float(x), float(y)


def _pin(text: str) -> tuple[str, tuple[float, float]]:
    name, _, at = text.partition("=")
    return name, _point(at)


def _site_root(text: str) -> Path:
    """A world directory, or a world name resolved as `flowedit suite` reads worlds."""
    path = Path(text)
    return path if path.is_dir() else world_root(None, text)


def _config(args: argparse.Namespace, key: str) -> IN.SiteConfig | None:
    """The site config of `--site-config`, else the one shipped for the site, None when there is neither."""
    if args.site_config is not None:
        return IN.site_config(key, args.site_config)
    try:
        return IN.site_config(key)
    except FileNotFoundError:
        return None


def _scenario(args: argparse.Namespace, config: IN.SiteConfig | None) -> str:
    return args.scenario or (config.scenario if config is not None else IN.SCENARIO)


def _out(args: argparse.Namespace, name: str) -> Path:
    """The suite file `--out` names, as a bundle (`SU.bundle_file`), else suite `name`'s default bundle."""
    return SU.bundle_file(args.out) if args.out else IN.default_out(name)


def _summary(doc: Mapping[str, Any]) -> str:
    return f"{len(doc['stages'])} stages, {IN.episodes(doc)} episodes, {IN.sim_hours(doc):g} sim-hours"


def _checked(config: IN.SiteConfig | None, root: Path) -> IN.SiteConfig | None:
    """`config`, after its check that the site's world at `root` holds none of the walls it removes."""
    if config is not None:
        config.check_site(root)
    return config


def _cmd_reference(args: argparse.Namespace) -> int:
    site = open_site(_site_root(args.site))
    config = _checked(_config(args, site.key), site.root)
    scenario = _scenario(args, config)
    IN.scenario_dir(site, scenario, args.scenarios_from)
    seeds = args.seeds or list(IN.SELECTION_SEEDS)
    doc = IN.reference_suite(site.key, seeds, scenario=scenario, duration=args.duration, side=config.crowd_side if config is not None else None)
    name = SU.name_of(_out(args, IN.REFERENCE_SUITE))
    if args.dry_run:
        from arena_simulation_setup.tree.World.write import to_yaml

        print(to_yaml(doc), end="")
        _log(f"(dry run - {_summary(doc)}; nothing written)")
        return 0
    header = [
        f"INFRA Stage 1 reference: {site.key} as is without robots, scenario {scenario} (through-walkers only), seeds {', '.join(map(str, seeds))}, {args.duration:g} s each",
        "run it first: its flow cells (flow_cells.parquet in the benchmark dir) are the joint search's occupancy (`flowedit infra layouts --occupancy`),",
        "and each layout's run without robots pairs with its run at the same seed (element part, §6)",
    ]
    out = IN.write(doc, _out(args, IN.REFERENCE_SUITE), header, force=args.force)
    print(f"wrote {out}: {_summary(doc)}")
    print(SU.benchmark_command(name, args.lanes))
    return 0


class _Space:
    """The search space of a layout command: the site, its config, the crowd scenario and the `Search`."""

    def __init__(self, args: argparse.Namespace, elements: dict[str, Any] | None = None, *, count: bool = True) -> None:
        self.site = open_site(_site_root(args.site))
        self.config = _checked(_config(args, self.site.key), self.site.root)
        self.scenario_name = _scenario(args, self.config)
        self.side = self.config.crowd_side if self.config is not None else None
        self.scenario = IN.scenario_dir(self.site, self.scenario_name, args.scenarios_from)
        placements = dict(self.config.placements) if self.config is not None else {}
        exclude = self.config.exclude if self.config is not None else SE.EXCLUDED_ZONES
        loading = args.loading if args.loading is not None else (None if SE.LOADING in placements else SE.TWIN_LOADING_XY)
        self.mix = SE.TaskMix(rate_per_h=args.rate or SE.TaskMix().rate_per_h, pickup_share=args.pickup_share, episode_s=args.duration, top_ups_per_job=getattr(args, "top_ups", 0.0))
        self.rate_from: dict[str, Any] | None = None
        facts = self.config.facts if self.config is not None else None
        if facts is not None:
            # the walkway rule measures against the crowd of the scenario every stage runs
            facts = attrs.evolve(facts, walkers=WK.WalkerDemand.load(self.scenario))
        self.search = SE.Search(self.site, loading=loading, pins=dict(getattr(args, "pin", None) or []), exclude=exclude, placements=placements, keep_out=IN.keep_out(self.scenario / "scenario.yaml", self.config.links if self.config is not None else ()), mix=self.mix, facts=facts, elements=elements)
        _log(f"site config: {self.config.path if self.config is not None else '(none, the search defaults)'}")
        if count:
            _log(f"search space: {json.dumps(self.search.counts())}")
        else:
            # the first study's candidate list (charger x bay x locker triples) is no part of the entrance search
            _log(f"spots: {json.dumps({name: len(found) for name, found in self.search.spots.items()})}")
        loading_spot = self.search.loading
        _log(f"loading point: {loading_spot.zone} piece {loading_spot.piece} step {loading_spot.step}, goto {loading_spot.objects[0].goto}")

    def loading_record(self) -> dict[str, Any]:
        s = self.search.loading
        return {"spot": s.id, "zone": s.zone, "piece": s.piece, "step": s.step, "goto": list(s.objects[0].goto)}

    def settle_rate(self, args: argparse.Namespace, pick: Any) -> tuple[float, Any]:  # noqa: ANN401
        """The rate `--rate` gives, else 75 % of the fleet's capacity in the wide premise layout (`search.premise_pair`),
        the layout the premise check measures the rate in; and the layouts `pick` returns at that rate. The rate sets
        how many jobs a charge is spread over, so the travel it is estimated from, so both settle together. Sets
        `rate_from`, what the rate came from, for the manifest."""
        import attrs

        if args.rate is not None:
            self.search.mix = attrs.evolve(self.mix, rate_per_h=args.rate)
            self.rate_from = {"given": "--rate"}
            return float(args.rate), pick()
        rate = self.mix.rate_per_h
        self.search.mix = attrs.evolve(self.mix, rate_per_h=rate)
        tries = 0
        while True:
            _, wide = SE.premise_pair(self.search)
            estimate = IN.fleet_rate(wide.travel_m, speed_mps=args.speed, load_s=IN.LOAD_S, unload_s=IN.UNLOAD_S)
            if estimate == rate:
                break
            rate = estimate
            self.search.mix = attrs.evolve(self.mix, rate_per_h=rate)
            tries += 1
            if tries == 8:
                _log(f"the rate estimate did not settle, last {rate:g} jobs/h")
                break
        self.rate_from = {"estimate": "wide premise layout", "layout_id": wide.layout_id, "travel_m": round(wide.travel_m, 3), "speed_mps": args.speed, "utilisation": IN.UTILISATION}
        _log(f"rate {rate:g} jobs/h: {IN.UTILISATION:.0%} of the fleet's capacity in the wide premise layout {wide.layout_id} ({wide.travel_m:.1f} m per job) at {args.speed:g} m/s, no charging, no crowd (an upper bound; `evaluation infra premise` measures it in the same layout, pass that as --rate)")
        return rate, pick()


def _print_runs(runs: list[tuple[str, SE.Scored]], speed: float, ranking: dict[str, float] | None = None, exposure: dict[str, dict[str, float]] | None = None) -> None:
    for role, s in runs:
        job_s = s.travel_m / speed + IN.LOAD_S + IN.UNLOAD_S
        occ = "" if s.occupancy is None else f", {s.occupancy:.3f} pedestrians under it"
        rank = "" if not ranking or s.layout_id not in ranking else f", ranking score {ranking[s.layout_id]:.3f}"
        groups = (exposure or {}).get(s.layout_id)
        exp = "" if not groups else ", exposure s/h " + " ".join(f"{g} {v:.2f}" for g, v in groups.items())
        _log(f"{role:10s} {s.layout_id}: {s.travel_m:7.1f} m per job (about {job_s / 60:.1f} min at {speed:g} m/s){occ}{rank}{exp}, floor loss {s.floor_loss_m2:.2f} m2, {' / '.join(sp.zone for sp in s.layout.spots)}")


def _plan_parts(args: argparse.Namespace, layout_ids: Iterable[str]) -> list[list[str]] | None:
    """With `--parts`, the layout copies of each part (`IN.split_parts`), worked out before anything is written and in a
    dry run too, so a split that cannot be made stops the command before it writes a suite; None without."""
    if not getattr(args, "parts", None):
        return None
    ids = list(dict.fromkeys(layout_ids))
    if len(ids) < len(args.parts):
        hint = ": the joint pick is the robot-only pick, run it without --parts" if len(ids) == 1 else ""
        raise ValueError(f"--parts {','.join(map(str, args.parts))} splits by layout copy into {len(args.parts)} parts, but the suite has {len(ids)} layout {'copy' if len(ids) == 1 else 'copies'}{hint}")
    parts = IN.split_parts(ids, args.parts)
    for index, (part, weight) in enumerate(zip(parts, args.parts, strict=True), start=1):
        _log(f"part {index} (weight {weight}): {len(part)} layouts, {', '.join(part)}")
    return parts


def _drop_stale_parts(out: Path, keep: Collection[str]) -> None:
    """Remove the part suites an earlier run left beside `out` (bundles `<suite>_p<N>`, written by flowedit infra) that
    this run did not write, so a suite's parts are always the ones its manifest lists."""
    name = SU.name_of(out)
    pattern = re.compile(rf"^{re.escape(name)}_p\d+$")
    for path in sorted(SU.suites_dir(out).glob(f"{name}_p*/{SU.SUITE_FILE}")):
        part = SU.name_of(path)
        if pattern.match(part) and part not in keep and path.read_text().startswith(IN.MARKER):
            path.unlink()
            with contextlib.suppress(OSError):
                path.parent.rmdir()
            _log(f"removed {path}: a part of an earlier run of {name}")


def _write_parts(args: argparse.Namespace, out: Path, plan: list[list[str]] | None, worlds: dict[str, str], doc_for: Callable[[list[str]], dict[str, Any]], contest: str) -> list[dict[str, Any]]:
    """The suite of each part of `plan` (`_plan_parts`; `worlds`: layout id -> world name) beside `out`, each runnable on
    its own machine as its own run, and no part suite of an earlier run left; the parts as the manifest records them,
    [] without."""
    records: list[dict[str, Any]] = []
    plan = plan or []
    for index, (ids, weight) in enumerate(zip(plan, getattr(args, "parts", None) or [], strict=True), start=1):
        name = IN.part_name(SU.name_of(out), index)
        doc = doc_for([worlds[layout_id] for layout_id in ids])
        header = [
            f"part {index} of {len(plan)} of suite {SU.name_of(out)} (weight {weight}): layouts {', '.join(ids)}",
            f"{_summary(doc)}; every part runs as its own run on its own machine, `evaluation infra` reads their run dirs together with {IN.manifest_path(out).name}",
            f"run: {SU.benchmark_command(name, weight, contest)}",
        ]
        path = IN.write(doc, SU.sibling(out, name), header, force=args.force)
        records.append({"suite": name, "weight": weight, "layouts": ids, "worlds": [worlds[layout_id] for layout_id in ids], "episodes": IN.episodes(doc), "sim_hours": IN.sim_hours(doc)})
        print(f"part {index}: {path} ({len(ids)} layouts, {_summary(doc)}): {SU.benchmark_command(name, weight, contest)}")
    _drop_stale_parts(out, {r["suite"] for r in records})
    return records


def _write_layout_suite(args: argparse.Namespace, space: _Space, name: str, runs: list[tuple[str, SE.Scored]], seeds: list[int], rate: float, header: list[str], extra: dict[str, Any], kept: dict[str, tuple[SE.Scored, dict[str, Any]]] | None = None) -> int:
    """Save the copies of `runs`, write their fleet suite `name` on `seeds` and its manifest. `kept` (by role: the layout
    and its manifest record) are saved as copies too and recorded in the manifest under their role, but run in no stage
    of this suite (the guideline layout, which only the evaluation runs)."""
    import attrs

    from . import place

    unique = list({s.layout_id: s for _, s in runs}.values())
    plan = _plan_parts(args, [s.layout_id for s in unique])
    also = [scored for scored, _ in (kept or {}).values() if scored.layout_id not in {s.layout_id for s in unique}]
    worlds = IN.save_layouts(space.site, [s.layout for s in [*unique, *also]], space.scenario, visitor_rate_per_h=args.visitor_rate)
    if space.config is not None:
        for path in worlds.values():
            space.config.check_site(path)
    fleet = IN.fleet_config(rate, pickup_share=args.pickup_share, charge_below=args.charge_below, charge_until=args.charge_until)
    doc = IN.layout_suite([worlds[s.layout_id].name for s in unique], fleet, seeds, scenario=space.scenario_name, duration=args.duration, side=space.side)
    contest = IN.contest_literal(IN.contest(args.capacity_wh, soc_min=args.soc_min, soc_max=args.soc_max))
    out = _out(args, name)
    header = [
        *header,
        f"fleet: {IN.ROBOT}, rate {rate:g} jobs/h, charge below {args.charge_below:g} % until {args.charge_until:g} %; battery {args.capacity_wh:g} Wh and start charge {args.soc_min:g}-{args.soc_max:g} % through the contest, chargers {place.CHARGE_POWER_W:g} W in the worlds",
        f"references: {IN.REFERENCES} adds per stage the layout without robots, the traffic part's reference (§6)",
        f"{_summary(doc)}",
        f"run: {SU.benchmark_command(SU.name_of(out), args.lanes, contest)}",
    ]
    out = IN.write(doc, out, header, force=args.force)
    parts = _write_parts(args, out, plan, {s.layout_id: worlds[s.layout_id].name for s in unique}, lambda names: IN.layout_suite(names, fleet, seeds, scenario=space.scenario_name, duration=args.duration, side=space.side), contest)
    record = {
        "suite": SU.name_of(out),
        "site": space.site.key,
        "site_root": str(space.site.root),
        "site_config": None if space.config is None else space.config.record(),
        "scenario": str(space.scenario),
        "visitor_sources": IN.visitor_sources(space.scenario),
        "visitor_rate_per_h": args.visitor_rate,
        "seeds": seeds,
        "episode_s": args.duration,
        "warmup_s": IN.WARMUP_S,
        "rate_per_h": rate,
        "rate_from": space.rate_from,
        "speed_mps": args.speed,
        "fleet": fleet,
        "contest": contest,
        "mix": attrs.asdict(space.search.mix),
        "search": space.search.counts(),
        "loading": space.loading_record(),
        "episodes": IN.episodes(doc),
        "sim_hours": IN.sim_hours(doc),
        **extra,
        **{role: {**record, "world": str(worlds[scored.layout_id])} for role, (scored, record) in (kept or {}).items()},
        "worlds": {layout_id: str(path) for layout_id, path in worlds.items() if layout_id in {s.layout_id for s in unique}},
        **({"parts": parts} if parts else {}),
    }
    manifest = IN.manifest_path(out)
    manifest.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}: {_summary(doc)}; {manifest}; worlds {', '.join(str(p) for p in worlds.values())}")
    print(SU.benchmark_command(SU.name_of(out), args.lanes, contest))
    return 0


def _cmd_premise(args: argparse.Namespace) -> int:
    space = _Space(args)
    seeds: list[int] = args.seeds or list(IN.PREMISE_SEEDS)
    rate, (narrow, wide) = space.settle_rate(args, lambda: SE.premise_pair(space.search))
    runs = [("narrow", narrow), ("wide", wide)]
    _print_runs(runs, args.speed)
    for role, s in runs:
        _log(f"{role:10s} {s.layout_id}: remaining corridor {', '.join(f'{sp.element.name} {sp.corridor_m:.2f} m' for sp in s.layout.spots if sp.element.name in SE.SEARCHED)}")
    if narrow.layout_id == wide.layout_id:
        raise ValueError("the narrowest and the widest layout are the same: the site has one section width")
    doc = IN.layout_suite([narrow.layout_id, wide.layout_id], IN.fleet_config(rate), seeds, scenario=space.scenario_name, duration=args.duration, side=space.side)
    _log(f"premise suite: {_summary(doc)}; with the site as is at seeds {', '.join(map(str, seeds))} from {IN.REFERENCE_SUITE}: {IN.episodes(doc) + len(seeds)} episodes (§10: 15)")
    _plan_parts(args, [narrow.layout_id, wide.layout_id])
    if args.dry_run:
        _log("(dry run - no world or suite written)")
        return 0
    header = [
        f"INFRA Stage 1 premise check (§10) on {space.site.key}: all elements at the narrowest section ({narrow.layout_id}) and at the widest ({wide.layout_id}), seeds {', '.join(map(str, seeds))}, {args.duration:g} s each, warm-up {IN.WARMUP_S:g} s cut in the analysis",
        f"with the site as is at the same seeds from {IN.REFERENCE_SUITE}: C_worst of the two against the seed spread (raise the density if it is smaller), and the task rate measured in the wide layout (`evaluation infra premise`)",
    ]
    extra = {"roles": ["narrow", "wide"], "runs": IN.picks_record({"narrow": narrow, "wide": wide})}
    return _write_layout_suite(args, space, IN.PREMISE_SUITE, runs, seeds, rate, header, extra)


def _groups(args: argparse.Namespace, space: _Space, occupancy: SE.Occupancy | None) -> dict[str, SE.Occupancy] | None:
    """The walker groups' occupancy the joint ranking scores: `--group-occupancy` (arena-evaluation's group cells),
    else the flow cells' occupancy of all walkers as one group, else None (a dry run without either)."""
    if args.group_occupancy:
        groups = SE.Occupancy.from_group_cells(args.group_occupancy, space.site.key, args.duration)
        missing = [g for g in SE.GROUPS if g not in groups]
        if missing:
            _log(f"the group cells hold no walker time of {missing}: the joint ranking scores {sorted(groups)}")
        return groups
    if occupancy is not None:
        _log("no --group-occupancy: the joint ranking scores all walkers as one group (the flow cells' occupancy)")
        return {"all": occupancy}
    return None


def _cmd_layouts(args: argparse.Namespace) -> int:
    space = _Space(args)
    search = space.search
    if args.occupancy is None and not args.group_occupancy and not args.dry_run:
        raise ValueError("--group-occupancy (the reference runs' group cells, `evaluation infra occupancy`) or --occupancy is needed for the joint ranking; --dry-run searches without them")
    occupancy = None if args.occupancy is None else SE.Occupancy.from_flow_cells(args.occupancy, space.site.key, args.duration)
    groups = _groups(args, space, occupancy)
    seeds: list[int] = args.seeds or list(IN.SELECTION_SEEDS)
    state: dict[str, Any] = {}

    def pick() -> list[Any]:
        robot = SE.robot_only(search)
        cap = SE.travel_cap(robot, args.travel_slack)
        within = search.within_travel(cap)
        rate = search.mix.rate_per_h
        joint = [] if groups is None else SE.joint_exposure(search, groups, rate_per_h=rate, top=args.top, band_m=args.band, distinct_m=args.distinct, max_travel_m=cap, speed_mps=args.speed)
        chosen = {robot.layout_id, *(s.layout_id for s in joint)}
        random = SE.random_valid(search, args.random, seed=args.seed, exclude=chosen, occupancy=occupancy)
        state.update(cap=cap, within=len(within))
        return [robot, joint, random]

    rate, (robot, joint, random) = space.settle_rate(args, pick)
    _log(
        f"travel model: {search.mix.top_ups_per_job:g} top-up drives bay - charger per job; travel pre-filter: at most {args.travel_slack:g} x the robot-only pick's {robot.travel_m:.1f} m = {state['cap']:.1f} m per job: {state['within']} of {len(search.candidates)} candidate layouts (valid or not); the joint search ranks every one of them"
    )
    guide = SE.guideline(search) if args.guideline else None
    runs = [("robot_only", robot), *(("joint", s) for s in joint), *(("random", s) for s in random)]
    exposure: dict[str, dict[str, float]] = {}
    traffic: dict[str, dict[str, float]] = {}
    if groups is None:
        _log("no group or flow-cell occupancy: the joint top 10 is left out of this dry run (the occupancy comes from the reference runs)")
    else:
        if len(joint) < args.top:
            _log(f"the travel pre-filter keeps only {len(joint)} distinct valid layouts, fewer than the top {args.top}")
        for _, s in [*runs, *([("guideline", guide)] if guide is not None else [])]:
            exposure[s.layout_id], traffic[s.layout_id] = SE.exposure_of(search, groups, s.index, rate_per_h=rate, band_m=args.band, speed_mps=args.speed)
    ranking = {layout_id: max(values.values()) for layout_id, values in exposure.items()} or None
    _print_runs(runs, args.speed, ranking, exposure)
    if guide is not None:
        _print_runs([("guideline", guide)], args.speed, ranking, exposure)
        _log(f"guideline {guide.layout_id}: saved as a copy and recorded in the manifest; the selection does not run it, `flowedit infra evaluation` does")
    unique = {s.layout_id for _, s in runs}
    doc = IN.layout_suite(sorted(unique), IN.fleet_config(rate), seeds, scenario=space.scenario_name, duration=args.duration, side=space.side)
    _log(f"selection suite: {len(unique)} layouts, {_summary(doc)}; with the site as is at seeds {', '.join(map(str, seeds))} from {IN.REFERENCE_SUITE}: {IN.episodes(doc) + len(seeds)} episodes")
    _plan_parts(args, sorted(unique))
    if args.dry_run:
        _log("(dry run - no world or suite written)")
        return 0
    header = [
        f"INFRA Stage 1 selection (§5, §10) on {space.site.key}: {len(unique)} layouts x seeds {', '.join(map(str, seeds))}, {args.duration:g} s each, warm-up {IN.WARMUP_S:g} s cut in the analysis",
        f"layouts: robot-only {robot.layout_id} (travel model with {search.mix.top_ups_per_job:g} top-up drives per job); joint top {len(joint)} distinct ({args.distinct:g} m) by predicted conflict exposure ({args.band:g} m band, worst group), among the layouts within {args.travel_slack:g} x the robot-only travel ({state['cap']:.1f} m per job); {len(random)} random valid",
        f"the layouts' runs without robots pair with suite {IN.REFERENCE_SUITE} (the site as is) at the same seed; the pick: `evaluation infra pick`",
    ]

    def predicted(layout_id: str) -> dict[str, Any]:
        return {"exposure": exposure[layout_id], "exposure_traffic": traffic[layout_id]} if layout_id in exposure else {}

    def record(scored: SE.Scored) -> dict[str, Any]:
        return {**IN.picks_record({"guideline": scored}, ranking)[0], **predicted(scored.layout_id)}

    extra = {
        "occupancy": None if args.occupancy is None else str(args.occupancy),
        "group_occupancy": [str(p) for p in args.group_occupancy or []],
        "joint_ranking": {"score": "predicted conflict exposure, worst group (s per h per member)", "groups": sorted(groups or {}), "band_m": args.band, "rate_per_h": rate, "speed_mps": args.speed, "distinct_m": args.distinct, "top_ups_per_job": search.mix.top_ups_per_job, "top_up_leg": "one way, bay to charger"},
        "travel_prefilter": {"slack": args.travel_slack, "max_travel_m": round(state["cap"], 3), "candidates_within": state["within"], "candidates": len(search.candidates)},
        "runs": [{**r, **predicted(r["layout_id"])} for r in IN.picks_record({"robot_only": robot, "joint": joint, "random": random}, ranking)],
        "joint_top": IN.picks_record({"joint": joint}, ranking),
    }
    kept = {"guideline": (guide, {k: v for k, v in record(guide).items() if k != "role"} | {"role": "guideline"})} if guide is not None else None
    return _write_layout_suite(args, space, IN.LAYOUT_SUITE, runs, seeds, rate, header, extra, kept)


def _pairs(text: str) -> dict[str, float]:
    """`adult=0.3,wheelchair=0.5`: a number per walker group."""
    out = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        if name.strip() not in SE.GROUPS:
            raise argparse.ArgumentTypeError(f"{name.strip()!r} is no walker group ({', '.join(SE.GROUPS)})")
        out[name.strip()] = float(value)
    return out


def _floats(text: str) -> list[float]:
    """`30,40`: numbers in order."""
    out = [float(v) for v in text.split(",") if v.strip()]
    if not out:
        raise argparse.ArgumentTypeError(f"no numbers in {text!r}")
    return out


def _costs(args: argparse.Namespace, space: Any, groups: dict[str, SE.Occupancy]) -> tuple[list[Any], str]:  # noqa: ANN401 - a demand.EntranceSearch
    """The ranking of the pedestrian-aware search by `--rank`, as costs in order (`demand.pedestrian_aware`), and what
    it is in words. ValueError for a pooled cost at a band without default weights and without `--delay-per-exposure`."""
    from . import demand as DE

    scored = [g for g in SE.GROUPS if g in groups]
    speeds = {**DE.WALK_SPEED_MPS, **(args.walk_speed or {})}
    exposure = DE.Cost(exposure=dict.fromkeys(scored, 1.0), band_m=args.band) if scored else None
    trip = DE.Cost(walk={g: 2.0 / speeds[g] for g in SE.GROUPS})
    if args.rank == "walk":
        return [trip, *([exposure] if exposure is not None else [])], "the worst-off group's walk to the locker and back (s), ties by its predicted exposure"
    if exposure is None:
        raise ValueError(f"--rank {args.rank} needs the walker groups' occupancy (--group-occupancy); --rank walk ranks without")
    if args.rank == "exposure":
        return [exposure], f"the worst-off group's predicted exposure (s per h per member, {args.band:g} m band)"
    given = args.delay_per_exposure or {}
    missing = [g for g in scored if g not in given]
    per = {**(DE.delay_per_exposure(args.band) if missing else {}), **given}
    element = {**DE.DELAY_PER_ELEMENT, **(args.delay_per_element or {})}
    present = sum(groups[g].present for g in scored)
    cost = DE.Cost.pooled({g: per[g] for g in scored}, space.demand.deliveries_per_h, present, speeds, args.band, {g: element[g] for g in scored})
    words = (
        f"the worst-off group's time lost per hour on the site (s per h per member): its time within {args.band:g} m of a driving robot times "
        f"{', '.join(f'{g} {per[g]:g}' for g in scored)}, its time on an element's floor times {', '.join(f'{g} {element[g]:g}' for g in scored)}, "
        f"plus its recipients' walk to the locker and back ({space.demand.deliveries_per_h:g} recipients per hour among {present:.0f} walkers present)"
    )
    if args.rank == "step-free":
        words = f"rule S: the layouts whose every recipient entrance is served from within the reach step-free first, where the travel cap holds one; among them (else among all) {words}"
    return [cost], words


def _cmd_search(args: argparse.Namespace) -> int:
    from . import demand as DE

    if args.out is not None and args.out.exists() and not args.force:
        raise ValueError(f"{args.out} exists: pick another name, or pass --force")
    space = _Space(args, elements={SE.SEARCHED[2]: DE.locker_element(outlet=args.outlet_rule)}, count=False)
    if space.config is None:
        raise ValueError(f"{space.site.key}: the recipient entrances come from the site config's entrances; there is none")
    search = space.search
    entrances = DE.recipient_entrances(space.config.facts, args.recipient_types, search.loading.level)
    groups: dict[str, SE.Occupancy] = dict(SE.Occupancy.from_group_cells(args.group_occupancy, space.site.key, args.duration)) if args.group_occupancy else {}
    if args.occupancy is not None:
        groups["all"] = SE.Occupancy.from_flow_cells(args.occupancy, space.site.key, args.duration)
    demand = DE.Demand(rate_per_h=space.mix.rate_per_h, pickup_share=args.pickup_share, bay_share=args.bay_share, charging_trips_per_job=args.charging_trips)
    found = DE.EntranceSearch(search, entrances, links=space.config.links, reach_max_m=max(args.reach), demand=demand, groups=groups, bands=(args.band,), speed_mps=args.speed, walk_speeds={**DE.WALK_SPEED_MPS, **(args.walk_speed or {})})
    _log(f"recipient entrances ({', '.join(args.recipient_types)}): {', '.join(e.name for e in entrances)}")
    _log(f"search space: {json.dumps(found.counts())}")
    costs, ranking = _costs(args, found, groups)
    step_free = args.rank == "step-free"
    _log(f"pedestrian-aware ranking: {ranking}")
    report: dict[str, Any] = {
        "site": space.site.key,
        "site_config": space.config.record(),
        "scenario": str(space.scenario),
        "loading": space.loading_record(),
        "demand": {"rate_per_h": demand.rate_per_h, "pickup_share": demand.pickup_share, "bay_share": demand.bay_share, "charging_trips_per_job": demand.charging_trips_per_job},
        "recipient_entrances": [e.name for e in entrances],
        "locker_rules": [str(rule) for rule in search.elements[SE.SEARCHED[2]].rules],
        "rank": args.rank,
        "ranking": ranking,
        "margins": list(args.margins),
        "band_m": args.band,
        "group_occupancy": [str(p) for p in args.group_occupancy or []],
        "search": found.counts(),
        "reaches": [],
    }

    def line(role: str, placed: Any, base: float) -> str:  # noqa: ANN401 - a demand.Placed
        record = found.record(placed, costs[0])
        walk = "; ".join(f"{g} {v['mean_m']:.0f} / {v['longest_m']:.0f} m" for g, v in record["walk"].items() if v["mean_m"] is not None)
        exposure = "" if "exposure" not in record else ", exposure s/h " + " ".join(f"{g} {v:.2f}" for g, v in record["exposure"][f"{args.band:g}"].items())
        worst = record["cost"][record["worst_group"]]
        set_ = "" if placed.step_free_set is None else (", step-free set" if placed.step_free_set else ", no step-free layout within the cap")
        return f"  {role:22s} {placed.layout_id}: {placed.travel_m:7.1f} m per job ({placed.travel_m / base:.3f} x), walk mean / longest {walk}{exposure}, worst-off {record['worst_group']} {'unreachable' if worst is None else f'{worst:.3f}'}{set_}, lockers {' / '.join(found.lockers[k].zone for k in placed.lockers)}"

    for reach in args.reach:
        entry: dict[str, Any] = {"reach_m": reach}
        report["reaches"].append(entry)
        try:
            cover = found.cover(reach)
            optimal = DE.travel_optimal(found, reach)
        except ValueError as e:
            _log(f"reach {reach:g} m: no valid layout: {e}")
            entry.update(valid=False, reason=str(e))
            continue
        layouts = [{"role": "travel_optimal", **found.record(optimal, costs[0])}]
        _log(f"reach {reach:g} m: {cover.count} lockers, {len(cover.lockers)} locker spots within the reach, {len(cover.partitions)} ways to split the entrances")
        _log(line("travel-optimal", optimal, optimal.travel_m))
        for margin in args.margins:
            cap = margin * optimal.travel_m
            try:
                pick = DE.pedestrian_aware(found, reach, costs, cap, start=[optimal], step_free=step_free)
            except ValueError as e:
                _log(f"  pedestrian-aware {margin:g}: {e}")
                layouts.append({"role": "pedestrian_aware", "margin": margin, "max_travel_m": round(cap, 3), "pick": None, "reason": str(e)})
                continue
            _log(line(f"pedestrian-aware {margin:g}", pick, optimal.travel_m))
            layouts.append({"role": "pedestrian_aware", "margin": margin, "max_travel_m": round(cap, 3), **found.record(pick, costs[0])})
        try:
            practice = DE.practice_based(found, reach)
        except ValueError as e:
            _log(f"  practice-based: {e}")
            layouts.append({"role": "practice_based", "pick": None, "reason": str(e)})
        else:
            _log(line("practice-based", practice, optimal.travel_m))
            layouts.append({"role": "practice_based", **found.record(practice, costs[0])})
        entry.update(valid=True, lockers=cover.count, locker_spots=len(cover.lockers), partitions=[[[entrances[e].name for e in group] for group in parts] for parts in cover.partitions], layouts=layouts)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
        print(f"wrote {args.out}")
    _log("(a search report: no world or suite written)")
    return 0


def _cmd_crowd(args: argparse.Namespace) -> int:
    import shapely

    from ...shared.navmesh import nav_walls

    site = open_site(_site_root(args.site))
    config = _checked(_config(args, site.key), site.root)
    if config is None:
        raise ValueError(f"{site.key}: the crowd comes from the site config's entrances; there is none")
    level = next(iter(site.world.levels.values()))
    floor = shapely.union_all([shapely.Polygon([(p.x, p.y) for p in z.polygon]) for z in level.zones if len(z.polygon) >= 3])
    # the walls HumanSim's walkers meet: an entrance area stays on the floor they reach
    walls = shapely.MultiLineString([[(w.start.x, w.start.y), (w.end.x, w.end.y)] for w in nav_walls(level)])
    robots = []
    template = IN.scenario_dir(site, config.scenario, args.scenarios_from) / "scenario.yaml"
    if template.is_file():
        import yaml

        robots = (yaml.safe_load(template.read_text()) or {}).get("robots") or []
    rate, how = _crowd_rate(args, config)
    scenario, visitors_doc, types = IN.entrance_crowd(config, floor, walls=walls, rate=rate, visitor_rate_per_h=args.visitor_rate, robots=robots)
    shares = IN.entrance_shares(config)
    lines = [f"{e.name} ({e.type}{', recipients' if e.recipients else ''}): {shares[e.name] * rate:.3f} ped/s" for e in config.facts.entrances]
    header = "\n".join(
        [
            f"INFRA study crowd from the entrances of {config.path.name if config.path else site.key} (study design §2, §9), written by `flowedit infra crowd`.",
            f"The inflow: {how}.",
            f"Every entrance is a PEDS source and sink. Walkers come in at {rate:.4g} ped/s in all, split by entrance type",
            "(flow_split, evenly within a group; groups without an entrance spread over the others), 70/15/15 adult /",
            "wheelchair / long cane per entrance, and leave by another entrance in proportion to its share. Each entrance's",
            f"area lies on the floor a walker reaches, {IN.ENTRANCE_WALL_M:g} m clear of the walls (HumanSim plans paths 0.38 m off them), without the pockets",
            f"they close off; an entrance with under {IN.ENTRANCE_MIN_M2:g} m2 of it in its square takes a larger square, up to {IN.ENTRANCE_GROW_M[-1]:g} m.",
            f"Parcel recipients ({args.visitor_rate:g}/h in all, locker_visitors.yaml) come in at a recipient entrance and leave by it.",
            "Robots as in the site config's scenario. Entrances and their inflow:",
            *[f"  {line}" for line in lines],
        ]
    )
    _log(header)
    if args.dry_run:
        _log("(dry run - nothing written)")
        return 0
    if args.out is None:
        raise ValueError("--out: the directory to write <name>/ into, e.g. the site's scenarios/")
    folder = IN.write_crowd(args.out / args.name, scenario, visitors_doc, types, header)
    _log(f"wrote {folder}")
    return 0


def _crowd_rate(args: argparse.Namespace, config: IN.SiteConfig) -> tuple[float, str]:
    """The crowd's inflow in all [ped/s] and how it was set: `--rate`, else the site config's crowd density (at
    `--density` if given) by Little's law, else `IN.CROWD_RATE`."""
    cd = config.crowd_density
    if args.rate is not None:
        if args.density is not None:
            raise ValueError("--rate or --density, not both: the density sets the rate")
        return args.rate, f"--rate {args.rate:g} ped/s"
    if cd is None:
        if args.density is not None:
            raise ValueError(f"--density: the site config {config.path} has no crowd_density (area and mean time) to turn it into a rate")
        return IN.CROWD_RATE, f"{IN.CROWD_RATE:g} ped/s (no crowd_density in the site config)"
    density = cd.density if args.density is None else args.density
    rate = cd.rate(density)
    return rate, (f"{density:g} ped/m2 on {cd.area_m2:g} m2 = {cd.present(density):.0f} walkers present; by Little's law over the mean time in the site, {cd.mean_time_s:g} s (site config crowd_density), {rate:.4g} ped/s in all")


def _named_layouts(text: str) -> list[tuple[str, str]]:
    """`travel_optimal=83298e5c,practice_based=d837888b`: layout ids, each under its role. Every layout needs a role and
    no two share one: the manifest keeps the layouts by role (`picks`)."""
    out = []
    for part in text.split(","):
        role, sep, layout_id = part.strip().rpartition("=")
        if not sep or not role.strip() or not layout_id.strip():
            raise argparse.ArgumentTypeError(f"a layout is <role>=<id>, not {part!r}")
        out.append((role.strip(), layout_id.strip()))
    if len({layout_id for _, layout_id in out}) != len(out) or len({role for role, _ in out}) != len(out):
        raise argparse.ArgumentTypeError(f"a layout id or a role twice in {text!r}")
    return out


def _seed_part(text: str) -> tuple[str, list[int]]:
    """`h1=71-74`: a part's tag and its seeds."""
    tag, sep, seeds = text.partition("=")
    if not sep or not re.fullmatch(r"[A-Za-z0-9]+", tag):
        raise argparse.ArgumentTypeError(f"a seed part is <tag>=<seeds>, the tag of letters and digits, not {text!r}")
    return tag, _seeds(seeds)


def _shares(text: str) -> dict[str, float]:
    out = _pairs(text)
    if min(out.values()) < 0 or sum(out.values()) <= 0:
        raise argparse.ArgumentTypeError(f"shares must not be negative and must add up to more than 0, not {text!r}")
    return out


def _cmd_entrance_runs(args: argparse.Namespace) -> int:
    """`flowedit infra evaluation --report`: the layouts of a search report as world copies with the seeds' parcel
    recipients (`recipients.py`), and their suites: the site as is, and each layout without robots as a crowd-only
    stage on the seeds that have such a run."""
    import yaml

    from . import demand as DE
    from . import entrance_fleet as EF
    from . import place
    from . import recipients as RC

    fleet_only = bool(args.fleet_only)
    arms = ["fleet"] if fleet_only else list(dict.fromkeys(args.arms))
    if not arms or sorted(set(arms) - {"site", "without", "fleet"}):
        raise ValueError(f"--arms names site, without and fleet, not {args.arms}")
    if args.charge_log and not fleet_only:
        raise ValueError("--charge-log belongs to a fleet-only run (--fleet-only)")
    if args.pick is not None or args.selection is not None or args.parts:
        raise ValueError("--report reads layouts from a search report: --pick, --selection and --parts belong to the first study's evaluation")
    if not args.layouts:
        raise ValueError("--report needs --layouts, the ids of the report's layouts to run")
    if args.rate is None:
        raise ValueError("--report needs --rate, the jobs per hour in all: the recipients follow the seeds' jobs")
    if not args.seeds:
        raise ValueError("--report needs --seeds")
    seeds = list(args.seeds)
    without = seeds if args.without_robots_seeds is None else list(args.without_robots_seeds)
    if sorted(set(without) - set(seeds)):
        raise ValueError(f"--without-robots-seeds {sorted(set(without) - set(seeds))} are not among --seeds")
    seed_parts = list(args.seed_part or [])
    if seed_parts:
        tags = [tag for tag, _ in seed_parts]
        spread = [seed for _, part in seed_parts for seed in part]
        if len(set(tags)) != len(tags) or sorted(spread) != sorted(seeds):
            raise ValueError(f"--seed-part: the parts need distinct tags and must hold each of --seeds once, not {seed_parts}")
    records = RC.read_report(args.report)
    unknown = [layout_id for _, layout_id in args.layouts if layout_id not in records]
    if unknown:
        raise ValueError(f"{args.report}: no layout {unknown} (it holds {len(records)} layouts)")
    space = _Space(args, elements={SE.SEARCHED[2]: DE.locker_element(outlet=not args.no_outlet_rule)})
    scenario = yaml.safe_load((space.scenario / "scenario.yaml").read_text()) or {}
    # the recipients' walker groups come from a crowd without tagged extras (the site's base scenario): extras added to
    # the stage scenario to oversample aid users would otherwise change the recipients' mix silently
    mix_from = args.recipient_mix_from or space.scenario_name
    mix = RC.crowd_mix(yaml.safe_load((IN.scenario_dir(space.site, mix_from, args.scenarios_from) / "scenario.yaml").read_text()) or {}) if mix_from != space.scenario_name else RC.crowd_mix(scenario)
    if args.recipient_mix:
        mix = {group: (agent_type, args.recipient_mix.get(group, 0.0)) for group, agent_type, _ in IN.GROUPS if args.recipient_mix.get(group, 0.0) > 0}
    total = sum(share for _, share in mix.values())
    _log("recipients' walker groups: " + ", ".join(f"{group} {share / total:.4f} ({agent_type})" for group, (agent_type, share) in mix.items()) + (" (--recipient-mix)" if args.recipient_mix else f" (the crowd's mix of {mix_from}; scenario {space.scenario_name})"))
    layouts, serves, roles = {}, {}, {}
    for role, layout_id in args.layouts:
        layout = RC.with_charge_power(RC.layout_of(space.search, records[layout_id]), args.charge_power)
        problems = place.check_layout(space.site, layout)
        if problems:
            raise ValueError(f"layout {layout_id}: " + "; ".join(problems))
        layouts[layout_id], serves[layout_id], roles[layout_id] = layout, RC.serves_of(records[layout_id], layout), role
    served = {layout_id: sorted(e for entrances in s.values() for e in entrances) for layout_id, s in serves.items()}
    entrances = next(iter(served.values()))
    if any(other != entrances for other in served.values()):
        raise ValueError(f"the layouts serve different recipient entrances: {served}")
    jobs = {seed: RC.jobs(seed, entrances, rate_per_h=args.rate, pickup_share=args.pickup_share, episode_s=args.duration) for seed in seeds}
    who = {layout_id: {seed: () if fleet_only else RC.recipients(seed, jobs[seed], serves[layout_id], mix, offset_s=args.recipient_offset, episode_s=args.duration, kinds=args.recipient_kinds) for seed in seeds} for layout_id in layouts}
    any_layout = next(iter(who.values()))
    _log(f"jobs at {args.rate:g} per hour over {len(entrances)} recipient entrances, {args.duration:g} s: " + ", ".join(f"seed {seed} {len(jobs[seed])}" for seed in seeds))
    _log(
        f"recipients ({'/'.join(args.recipient_kinds)}, {args.recipient_offset:g} s after the release): "
        + ", ".join(f"seed {seed} {len(any_layout[seed])}" for seed in seeds)
        + f"; {sum(len(v) for v in any_layout.values())} in all, by group "
        + json.dumps({group: sum(r.group == group for v in any_layout.values() for r in v) for group in mix})
    )
    for layout_id, layout in layouts.items():
        _log(f"{roles[layout_id]:22s} {layout_id}: " + "; ".join(f"{locker} ({next(s.zone for s in layout.spots for o in s.objects if o.name == locker)}) serves {', '.join(es)}" for locker, es in serves[layout_id].items()))
    names = {seed: f"{space.scenario_name}__{args.tag + '__' if args.tag else ''}fleet_only" if fleet_only else RC.scenario_name(space.scenario_name, seed, args.tag) for seed in seeds}
    out = _out(args, IN.ENTRANCE_SUITE)
    name = SU.name_of(out)
    outs = {"site": SU.sibling(out, f"{name}_site"), "without": SU.sibling(out, f"{name}_without"), "fleet": out}
    prefix = place.layout_prefix(space.site)
    states = None if args.charge_states is None else EF.load_charge_states(args.charge_states)
    states_out = IN.charge_states_path(out)
    contest = IN.contest_literal(EF.contest(soc_file=str(states_out) if states is not None else None, start_soc=args.start_charge))

    def fleet_block(layout_id: str, seed: int) -> dict[str, Any]:
        log = f"{args.charge_log}/{SU.stage_name(f'{prefix}{layout_id}', seed)}.csv" if args.charge_log else ""
        return EF.fleet_config(list(serves[layout_id]), EF.job_entries(jobs[seed], serves[layout_id]), rate_per_h=args.rate, pickup_share=args.pickup_share, charge_log=log, charge_log_s=args.charge_log_s)

    def docs(part: Sequence[int]) -> dict[str, dict[str, Any]]:
        """The suite of each arm on the seeds `part`; an arm without a stage on them is left out."""
        made = {
            "site": IN.reference_suite(space.site.key, part, scenario=space.scenario_name, duration=args.duration, side=space.side),
            "without": IN.without_suite([f"{prefix}{layout_id}" for layout_id in layouts], {seed: names[seed] for seed in part if seed in without}, duration=args.duration, side=space.side),
            "fleet": IN.stages_suite([IN.fleet_stage(f"{prefix}{layout_id}", seed, fleet_block(layout_id, seed), scenario=names[seed], duration=args.duration) for layout_id in layouts for seed in part], side=space.side),
        }
        return {arm: made[arm] for arm in arms if made[arm]["stages"]}

    full = docs(seeds)
    total_episodes = sum(IN.episodes(doc) for doc in full.values())
    said = {"site": "the site as is", "without": f"the layouts without robots (crowd only) on seeds {', '.join(map(str, without)) or 'none'}", "fleet": "the layouts with the fleet" + (", no crowd" if fleet_only else "")}
    _log("; ".join(f"{said[arm]}: {_summary(doc)}" for arm, doc in full.items()) + f"; {total_episodes} episodes")
    if "fleet" in full:
        _log(f"fleet: {IN.ROBOT}, charging by need, the nearest free charger, a jammed charging trip goes on; each robot's own battery; start charge " + (f"from {len(states['states'])} charge states of {args.charge_states}" if states is not None else f"{args.start_charge:g} % for every robot (no charge states given)"))
    else:
        _log("the runs with the fleet are not written (--arms fleet adds them)")
    # every file this writes, worked out before anything is written: a file that exists with other content stops the
    # command before it saves a copy, so a refused name leaves nothing half written
    common = f"seeds {', '.join(map(str, seeds))}, {args.duration:g} s each, warm-up {IN.WARMUP_S:g} s cut in the analysis"
    named = ", ".join(f"{roles[layout_id]} {layout_id}" for layout_id in layouts)
    files: dict[Path, str] = {}

    def plan(made: Mapping[str, Mapping[str, Any]], paths: Mapping[str, Path], what: str) -> None:
        heads = {
            "site": f"{space.site.key} as is without robots, scenario {space.scenario_name}, {common}",
            "without": f"{named} from {args.report}, each without robots as a crowd-only stage with the seed's parcel recipients (scenario {space.scenario_name}__s<seed>), seeds {', '.join(map(str, without))} of {common}",
            "fleet": f"{named} from {args.report}, each with the fleet" + (", no crowd" if fleet_only else " and the seed's parcel recipients") + f": the seed's jobs to the locker that serves each job's recipient entrance, {args.rate:g} jobs per hour; {common}",
        }
        for arm, doc in made.items():
            files[paths[arm]] = IN.suite_text(doc, [f"INFRA redesigned study{what}: {heads[arm]}", f"{_summary(doc)}", f"run: {SU.benchmark_command(SU.name_of(paths[arm]), args.lanes, contest if arm == 'fleet' else SU.CONTEST)}"])

    plan(full, outs, "")
    parts = []
    for tag, part in seed_parts:
        made = docs(part)
        paths = {"site": SU.sibling(out, f"{name}_site_{tag}"), "without": SU.sibling(out, f"{name}_without_{tag}"), "fleet": SU.sibling(out, f"{name}_{tag}")}
        plan(made, paths, f", part {tag} (seeds {', '.join(map(str, part))})")
        parts.append(
            {
                "tag": tag,
                "seeds": part,
                **({"suite": SU.name_of(paths["fleet"]) if "fleet" in made else None} if "fleet" in arms else {}),
                "site_suite": SU.name_of(paths["site"]) if "site" in made else None,
                "without_suite": SU.name_of(paths["without"]) if "without" in made else None,
                "episodes": sum(IN.episodes(doc) for doc in made.values()),
            }
        )
    worlds = {layout_id: RC.copy_path(space.site, layout) for layout_id, layout in layouts.items()}
    keep = ("layout_id", "reach_m", "lockers", "travel_m", "floor_loss_m2", "serves", "walk", "walk_by_entrance")
    block = {key: value for key, value in fleet_block(next(iter(layouts)), seeds[0]).items() if key not in ("jobs", "lockers", "soc_log")}
    crowd_only = without if "without" in full else []
    record = {
        "suite": name if "fleet" in full else None,
        "site_suite": SU.name_of(outs["site"]) if "site" in full else None,
        "without_suite": SU.name_of(outs["without"]) if "without" in full else None,
        "report": str(args.report),
        "writer_commit": args.writer_commit,
        "picks": {roles[layout_id]: layout_id for layout_id in layouts},
        "site": space.site.key,
        "site_root": str(space.site.root),
        "site_config": None if space.config is None else space.config.record(),
        "scenario": str(space.scenario),
        "scenarios": {str(seed): names[seed] for seed in seeds},
        "visitor_sources": sorted({r.source for per_seed in who.values() for group in per_seed.values() for r in group}),
        "episode_s": args.duration,
        "warmup_s": IN.WARMUP_S,
        "rate_per_h": args.rate,
        "pickup_share": args.pickup_share,
        "seeds": seeds,
        # the seeds with a crowd-only run of every layout. They are also `without_robots_seeds`, the key the analysis
        # reads, only when the runs with the fleet are written: with that key it asks every row for a run with the
        # fleet; without it a row of the site as is and the layout without robots gives the obstruction effect
        "crowd_only_seeds": crowd_only,
        **({IN.WITHOUT_ROBOTS_SEEDS: crowd_only} if "fleet" in full and not fleet_only else {}),
        "episodes": total_episodes,
        "sim_hours": round(sum(IN.sim_hours(doc) for doc in full.values()), 2),
        "loading": space.loading_record(),
        "charge_power_w": args.charge_power,
        "locker_rules": [str(rule) for rule in space.search.elements[SE.SEARCHED[2]].rules],
        "recipient_entrances": entrances,
        "recipients": {
            "kinds": list(args.recipient_kinds),
            "offset_s": args.recipient_offset,
            "stand_s": dict(RC.STAND_S),
            "patience_s": args.recipient_patience,
            "mix": {group: {"agent_type": agent_type, "share": share / total} for group, (agent_type, share) in mix.items()},
            "mix_from": "--recipient-mix" if args.recipient_mix else f"the sources of scenario {mix_from}",
            "by_layout": {layout_id: {str(seed): [r.record() for r in group] for seed, group in per_seed.items()} for layout_id, per_seed in who.items()},
        },
        "jobs": {str(seed): [job.record() for job in jobs[seed]] for seed in seeds},
        **({"fleet": block, "contest": contest, "fleet_only": fleet_only, "charge_states": str(states_out) if states is not None else None, "start_charge": None if states is not None else args.start_charge} if "fleet" in full else {}),
        "runs": [{"role": roles[layout_id], **{key: records[layout_id][key] for key in keep if key in records[layout_id]}} for layout_id in layouts],
        "worlds": {layout_id: str(path) for layout_id, path in worlds.items()},
        **({"seed_parts": parts} if parts else {}),
    }
    manifest_out = IN.manifest_path(out)
    files[manifest_out] = json.dumps(record, indent=1) + "\n"
    if states is not None and "fleet" in full:
        files[states_out] = json.dumps(states, indent=1) + "\n"
    taken = [str(path) for path, text in files.items() if path.is_file() and path.read_text() != text]
    if taken and not args.force:
        raise ValueError(f"{', '.join(taken)} exist with other content: written suites and manifests are not replaced, pick another name (--out) or pass --force")
    if args.dry_run:
        _log("(dry run - no world or suite written)")
        return 0

    def check(path: Path, name: str) -> str | None:
        """What is wrong with the copy (saved now or kept from before): a wall its site config removes or lacks one it
        adds (`check_site`), then, unless --no-check, what stops an episode at its start in its scenario `name` (the
        runtime's check, human.check); None when nothing is."""
        from arena_simulation_setup.metrics.use_points import blocking, describe

        if space.config is not None:
            try:
                space.config.check_site(path)
            except ValueError as e:
                return str(e)
        if args.no_check:
            return None
        findings = SU.check_stage(path, name, seeds[0])
        if stops := blocking(findings):
            return describe(stops)
        if findings:
            _log(f"{path.name}: {describe(findings)}")
        return None

    for layout_id, layout in layouts.items():
        if fleet_only:
            crowds = {names[seeds[0]]: (EF.without_crowd(scenario), {}, "a fleet-only run: the site's scenario without its sources, no walkers")}
        else:
            crowds = {names[seed]: RC.recipient_crowd(scenario, layout, serves[layout_id], who[layout_id][seed], mix, patience_s=args.recipient_patience) for seed in seeds}
        # the targets and agent types are the same at every seed: one scenario is checked; a copy that fails is taken away again
        saved = RC.save_layout(space.site, layout, space.scenario, crowds, check=check)
        assert saved == worlds[layout_id], (saved, worlds[layout_id])
        _log(f"saved {saved} with {len(crowds)} crowd scenarios ({names[seeds[0]]} ...)")
    for path, text in files.items():
        if not path.is_file() or path.read_text() != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    for part in parts:
        print(f"part {part['tag']}: seeds {', '.join(map(str, part['seeds']))}: " + ", ".join(str(part[key]) for key in ("suite", "site_suite", "without_suite") if part.get(key)) + f" ({part['episodes']} episodes)")
    print("wrote " + "; ".join(f"{outs[arm]}: {_summary(doc)}" for arm, doc in full.items()) + f"; {manifest_out}; worlds {', '.join(str(p) for p in worlds.values())}")
    for arm in full:
        print(SU.benchmark_command(SU.name_of(outs[arm]), args.lanes, contest if arm == "fleet" else SU.CONTEST))
    return 0


def _cmd_charge_states(args: argparse.Namespace) -> int:
    """`flowedit infra charge-states`: the charge states of fleet-only runs' charge logs, as the file the fleet's start
    charge is drawn from."""
    from . import entrance_fleet as EF

    if args.out.exists() and not args.force:
        raise ValueError(f"{args.out} exists: pick another name, or pass --force")
    doc = EF.charge_states(args.log, from_s=args.from_s, every_s=args.every_s)
    states = doc["states"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + "\n")
    flat = sorted(v for state in states for v in state)
    _log(
        f"{len(states)} charge states of {len(states[0])} robots from {doc['source']['rows']} rows ({doc['source']['skipped']} left out), from {args.from_s:g} s on, {args.every_s:g} s apart; charge min {flat[0]:.1f} %, median {flat[len(flat) // 2]:.1f} %, max {flat[-1]:.1f} %; by start spot "
        + ", ".join(f"{spot} {sum(state[i] for state in states) / len(states):.1f} %" for i, spot in enumerate(doc["spots"]))
    )
    print(f"wrote {args.out}")
    return 0


def _cmd_evaluation(args: argparse.Namespace) -> int:
    if args.report is not None:
        return _cmd_entrance_runs(args)
    if args.pick is None:
        raise ValueError("--pick (the first study's evaluation) or --report (layouts of `flowedit infra search`) is needed")
    pick = json.loads(Path(args.pick).read_text())
    selection = Path(args.selection) if args.selection else IN.find_manifest(IN.LAYOUT_SUITE)
    manifest = json.loads(selection.read_text())
    worlds = dict(manifest["worlds"])
    chosen = {"joint": pick.get("joint_pick"), "robot_only": pick.get("robot_only")}
    missing = [role for role, layout_id in chosen.items() if not layout_id]
    if missing:
        raise ValueError(f"{args.pick}: no {', '.join(missing)} layout (`evaluation infra pick` writes joint_pick and robot_only)")
    guide = manifest.get("guideline") if not args.no_guideline else None
    if guide is not None:
        chosen["guideline"] = guide["layout_id"]
        worlds.setdefault(guide["layout_id"], guide["world"])
    unknown = [layout_id for layout_id in chosen.values() if layout_id not in worlds]
    if unknown:
        raise ValueError(f"{args.pick}: layouts {unknown} are not in {selection}")
    seeds = args.seeds or list(IN.EVALUATION_SEEDS)
    overlap = sorted(set(seeds) & set(manifest["seeds"]))
    if overlap:
        raise ValueError(f"evaluation seeds {overlap} were selection seeds: the picks are scored on fresh seeds (§10)")
    scenario = Path(manifest["scenario"]).name
    duration = float(manifest["episode_s"])
    fleet = dict(manifest["fleet"])
    names = list(dict.fromkeys(Path(worlds[layout_id]).name for layout_id in chosen.values()))
    side = (manifest.get("site_config") or {}).get("crowd_side")
    doc = IN.layout_suite(names, fleet, seeds, scenario=scenario, duration=duration, side=side)
    site_doc = IN.reference_suite(manifest["site"], seeds, scenario=scenario, duration=duration, side=side)
    total = IN.episodes(doc) + IN.episodes(site_doc)
    _log(f"evaluation: joint pick {chosen['joint']}, robot-only pick {chosen['robot_only']}" + (f", guideline {chosen['guideline']}" if "guideline" in chosen else "") + f"; {_summary(doc)}, the site as is {_summary(site_doc)}: {total} episodes, {IN.sim_hours(doc) + IN.sim_hours(site_doc):g} sim-hours")
    if chosen["joint"] == chosen["robot_only"]:
        _log("the joint pick is the robot-only pick: the comparison's differences are 0 by construction")
    plan = _plan_parts(args, chosen.values())
    if args.dry_run:
        _log("(dry run - nothing written)")
        return 0
    contest = manifest.get("contest") or IN.contest_literal(IN.contest())
    out = _out(args, IN.EVALUATION_SUITE)
    name = SU.name_of(out)
    site_out = SU.sibling(out, IN.EVALUATION_SITE_SUITE if args.out is None else f"{name}_site")
    common = f"seeds {', '.join(map(str, seeds))}, {duration:g} s each, warm-up {float(manifest['warmup_s']):g} s cut in the analysis"
    arms = f"the joint pick {chosen['joint']}, the robot-only pick {chosen['robot_only']}" + (f" and the guideline layout {chosen['guideline']}" if "guideline" in chosen else "")
    IN.write(doc, out, [f"INFRA Stage 1 evaluation (§10, §12): {arms} from {selection}, with the fleet and without robots, {common}", f"{_summary(doc)}", f"run: {SU.benchmark_command(name, args.lanes, contest)}"], force=args.force)
    parts = _write_parts(args, out, plan, {layout_id: Path(worlds[layout_id]).name for layout_id in dict.fromkeys(chosen.values())}, lambda names: IN.layout_suite(names, fleet, seeds, scenario=scenario, duration=duration, side=side), contest)
    IN.write(site_doc, site_out, [f"INFRA Stage 1 evaluation reference: {manifest['site']} as is without robots, {common}", f"{_summary(site_doc)}", f"run: {SU.benchmark_command(SU.name_of(site_out), args.lanes)}"], force=args.force)
    record = {
        "suite": name,
        "site_suite": SU.name_of(site_out),
        "selection": str(selection),
        "pick": str(args.pick),
        "picks": chosen,
        **{key: manifest[key] for key in ("site", "site_root", "scenario", "visitor_sources", "episode_s", "warmup_s", "rate_per_h", "fleet", "contest") if key in manifest},
        "seeds": seeds,
        "episodes": total,
        "sim_hours": round(IN.sim_hours(doc) + IN.sim_hours(site_doc), 2),
        "runs": [{**(next(r for r in manifest["runs"] if r["layout_id"] == layout_id) if role != "guideline" else {k: v for k, v in (guide or {}).items() if k != "world"}), "role": role} for role, layout_id in chosen.items()],
        "worlds": {layout_id: worlds[layout_id] for layout_id in chosen.values()},
        **({"parts": parts} if parts else {}),
    }
    manifest_out = IN.manifest_path(out)
    manifest_out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}: {_summary(doc)}; {site_out}: {_summary(site_doc)}; {manifest_out}")
    print(SU.benchmark_command(SU.name_of(site_out), args.lanes))
    print(SU.benchmark_command(name, args.lanes, contest))
    return 0


def _report_layout(site: Any, pools: dict[str, Any], record: Mapping[str, Any]) -> Any:  # noqa: ANN401 - a Layout
    """The layout of a report record from `pools`, every candidate spot by element (valid or not), by spot id."""
    from . import demand as DE
    from . import recipients as RC

    chosen = []
    for entry in record["spots"]:
        name, spot_id = str(entry["element"]), str(entry["spot"])
        number = RC._locker_number(name)
        pool = pools["locker"] if number is not None else pools[name]
        found = next((s for s in pool if s.id == spot_id), None) if number in (None, 1) else next((DE.numbered(s, number) for s in pool if DE.numbered(s, number).id == spot_id), None)
        if found is None:
            raise ValueError(f"layout {record['layout_id']}: no {name} spot {spot_id} on {site.key}: the record comes from another site")
        chosen.append(found)
    layout = place.Layout.of(chosen)
    if layout.layout_id != record["layout_id"]:
        raise ValueError(f"the spots of record {record['layout_id']} give layout {layout.layout_id}")
    return layout


def _narrowest(stretches: Mapping[str, Sequence[Any]], *, two_way: bool) -> dict[str, Any] | None:
    """The narrowest two-way (or one-way) stretch of `stretches` (`walkway.Stretch` by element name) after the layout."""
    best = min(((st.after, name, st) for name, found in stretches.items() for st in found if st.after is not None and (st.required >= WK.TWO_WAY_M) == two_way), default=None, key=lambda t: t[0])
    if best is None:
        return None
    width, name, st = best
    return {"width_m": width, "before_m": st.before, "element": name, "at": list(st.at), "flow_ped_h": [st.flow_ab, st.flow_ba]}


def _cmd_walkways(args: argparse.Namespace) -> int:
    """`flowedit infra walkways`: the walkway rule on the layouts of a search report, against the site's crowd."""
    from . import recipients as RC

    site = open_site(_site_root(args.site))
    config = _checked(_config(args, site.key), site.root)
    scenario = IN.scenario_dir(site, _scenario(args, config), args.scenarios_from)
    walkers = WK.WalkerDemand.load(scenario)
    facts = attrs.evolve(config.facts if config is not None else place.SiteFacts(), walkers=walkers)
    records = RC.read_report(args.report)
    ids = args.layouts or sorted(records)
    unknown = [i for i in ids if i not in records]
    if unknown:
        raise ValueError(f"{args.report}: no layout {unknown}")
    _log(f"crowd {scenario}: {len(walkers.flows)} flows, {walkers.total * 3600:.0f} ped/h, {len(walkers.links)} links")
    # every candidate spot, valid or not: a layout's spots by id, its rules then run against the crowd
    pools = {name: place.candidate_spots(site, name, exclude=(), cut_off=False) for name in place.CATALOG}
    out = []
    for layout_id in ids:
        layout = _report_layout(site, pools, records[layout_id])
        spots = [s for spot in layout.spots for s in place._with_walkway(site, [spot], facts)]
        layout = place.Layout.of(spots)
        stretches: dict[str, list[Any]] = {}
        for s in layout.spots:
            found: list[Any] = []
            place.walkway_narrowings(site, s, (), walkers, stretches=found)
            stretches[s.element.name] = found
        problems = place.check_layout(site, layout)

        row = {
            "layout_id": layout_id,
            "valid": not problems,
            "problems": problems,
            "narrowings": [{"element": s.element.name, **attrs.asdict(n)} for s in layout.spots for n in s.narrowings],
            "narrowest_two_way": _narrowest(stretches, two_way=True),
            "narrowest_one_way": _narrowest(stretches, two_way=False),
        }
        out.append(row)
        two, one = row["narrowest_two_way"], row["narrowest_one_way"]
        _log(f"{layout_id}: {'valid' if row['valid'] else 'INVALID'}; narrowest two-way {two['width_m'] if two else '-'} m, one-way {one['width_m'] if one else '-'} m" + "".join(f"\n    {p}" for p in problems))
    if args.out is not None:
        args.out.write_text(json.dumps({"site": site.key, "scenario": str(scenario), "layouts": out}, indent=1) + "\n")
        _log(f"wrote {args.out}")
    return 0


def _cmd_walker_floor(args: argparse.Namespace) -> int:
    """`flowedit infra walker-floor`: the floor the crowd walks on, in HumanSim's terms (`walkway.WalkerFloor.export`)."""
    site = open_site(_site_root(args.site))
    levels = [args.level] if args.level is not None else list(site.world.levels)
    doc = {"site": site.key, "levels": [place.walker_floor(site, level).export() for level in levels]}
    args.out.write_text(json.dumps(doc) + "\n")
    _log(f"wrote {args.out}: " + ", ".join(f"level {lv['level']}: {len(lv['walls'])} walls, {len(lv['obstacles'])} obstacles, {lv['bare_edges_m']} m bare floor edge" for lv in doc["levels"]))
    return 0


def _cmd_site_walls(args: argparse.Namespace) -> int:
    """`flowedit infra site-walls` (also `remove-walls`): the walls of the site config's `removed_walls` out of, and those
    of its `added_walls` into, the site's world and each of its layout copies beside it (or the worlds `--world` names),
    in place."""
    from . import site_walls as SW

    root = _site_root(args.site)
    config = IN.site_config(root.name, args.site_config)
    if not config.removed_walls and not config.added_walls:
        _log(f"{config.path}: no removed_walls or added_walls, nothing to do")
        return 0
    worlds = [Path(w) for w in args.world] if args.world else SW.copies(root)
    names = [*(f"removing {w.name}" for w in config.removed_walls), *(f"adding {w.name}" for w in config.added_walls)]
    _log(f"{config.path}: {', '.join(names)} in {len(worlds)} worlds" + (" (dry run)" if args.dry_run else ""))

    def pixels(report: dict[str, Any]) -> str:
        return ", ".join(f"map {level}: {'no map' if n is None else f'{n} pixels'}" for level, n in report["map_pixels"].items())

    for world in worlds:
        if config.removed_walls:
            report = SW.remove(world, config.removed_walls, dry_run=args.dry_run)
            walls = ", ".join(f"{w['name']} (level {w['level']} wall {w['index']}, bbox {w['bbox']})" for w in report["walls"]) or "none left"
            print(f"{world}: removed {walls}" + (f"; {pixels(report)}" if report["map_pixels"] else ""))
        if config.added_walls:
            report = SW.add(world, config.added_walls, dry_run=args.dry_run)
            walls = ", ".join(f"{w['name']} (level {w['level']} wall {w['index']})" for w in report["added"]) or "none missing"
            print(f"{world}: added {walls}" + (f"; {pixels(report)}" if report["map_pixels"] else ""))
        if not args.dry_run and (SW.present(world, config.removed_walls) or SW.missing(world, config.added_walls)):
            raise ValueError(f"{world}: still holds {SW.present(world, config.removed_walls)}, lacks {SW.missing(world, config.added_walls)}")
    return 0


def add_parser(verbs: Any) -> None:  # noqa: ANN401
    """The `infra` verb and its subcommands."""
    p = verbs.add_parser("infra", help="INFRA Stage 1 suites: the site as is, the premise check, the selection and the evaluation", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    infra = p.add_subparsers(dest="infra", required=True, metavar="{reference,premise,layouts,search,evaluation,charge-states,site-walls}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--site", default=IN.SITE, help=f"the site: a world name or directory (default {IN.SITE}); layout copies hardlink its assets, so save on its mount")
    common.add_argument("--site-config", type=Path, default=None, help=f"the site config (default {IN.SITES_DIR}/<site>.yaml): where each element may go, the excluded zones, the crowd scenario")
    common.add_argument("--scenario", default=None, help=f"the crowd scenario of every stage (default the site config's, else {IN.SCENARIO})")
    common.add_argument("--scenarios-from", type=Path, default=None, help="world directory to copy the scenario from (default: the site's, else the world its name resolves to)")
    common.add_argument("--seeds", type=_seeds, default=None, help=f"episode seeds, e.g. 1-5 (default: reference and layouts {IN.SELECTION_SEEDS[0]}-{IN.SELECTION_SEEDS[-1]}, premise {IN.PREMISE_SEEDS[0]}-{IN.PREMISE_SEEDS[-1]}, evaluation {IN.EVALUATION_SEEDS[0]}-{IN.EVALUATION_SEEDS[-1]})")
    common.add_argument("--duration", type=float, default=IN.EPISODE_S, help=f"sim seconds per episode (default {IN.EPISODE_S:g})")
    common.add_argument("--out", type=Path, default=None, help="suite bundle (default: <imported assets root>/suites/<suite name>/suite.yaml, where `--suite <suite name>` finds it); a <name>.yaml writes the bundle <name>/suite.yaml beside it")
    common.add_argument("--lanes", type=int, default=2, help="lanes in the printed benchmark call (default 2)")
    common.add_argument("--force", action="store_true", help="replace a suite file flowedit infra did not write")
    common.add_argument("--dry-run", action="store_true", help="search and print without writing worlds or suites")
    searched = argparse.ArgumentParser(add_help=False)
    searched.add_argument("--loading", type=_point, default=None, help="x,y of the loading point, snapped to the nearest valid spot (default the site config's pin)")
    searched.add_argument("--pin", type=_pin, action="append", default=None, help="pin a searched element, e.g. locker=150.1,160.2 (over the site config); repeatable")
    searched.add_argument("--rate", type=float, default=None, help="jobs per hour in every layout (the premise check's measured rate, 75 %% of the fleet's capacity in the wide premise layout; default: the same share of that layout's capacity estimated at --speed without charging or crowd, an upper bound)")
    searched.add_argument("--speed", type=float, default=IN.SPEED_MPS, help=f"robot speed for the rate estimate, m/s (default {IN.SPEED_MPS:g}, the jackal's measured mean speed on open floor)")
    searched.add_argument("--pickup-share", type=float, default=IN.PICKUP_SHARE, help=f"share of pickups (default {IN.PICKUP_SHARE:g})")
    searched.add_argument("--charge-below", type=float, default=IN.CHARGE_BELOW, help=f"battery percent that sends a free robot to charge (default {IN.CHARGE_BELOW:g})")
    searched.add_argument("--charge-until", type=float, default=IN.CHARGE_UNTIL, help=f"battery percent that frees a charging robot (default {IN.CHARGE_UNTIL:g})")
    searched.add_argument("--capacity-wh", type=float, default=IN.CAPACITY_WH, help=f"battery capacity of every robot, Wh (default {IN.CAPACITY_WH:g})")
    searched.add_argument("--visitor-rate", type=float, default=None, help=f"locker visitors per hour in all (default the scenario's {IN.VISITORS})")
    searched.add_argument("--soc-min", type=float, default=IN.SOC_MIN, help=f"lowest start charge each robot draws per seed, %% (default {IN.SOC_MIN:g})")
    searched.add_argument("--soc-max", type=float, default=IN.SOC_MAX, help=f"highest start charge, %% (default {IN.SOC_MAX:g})")
    searched.add_argument("--top-ups", type=float, default=0.0, help="top-up drives bay - charger per job in the robot travel model, one way each (the task-rate run's top-ups dispatched after the warm-up per completed job released after it; default 0)")
    parts_help = "also split the suite by layout copy into one suite per weight, <suite>_p1, <suite>_p2, ..., in proportion to the weights (e.g. 4,6: the lanes of two machines); the manifest lists the parts; each part runs as its own run on its own machine, and `evaluation infra` reads their run dirs together (repeated --layouts)"
    searched.add_argument("--parts", type=_weights, default=None, metavar="W1,W2[,...]", help=parts_help)

    q = infra.add_parser("reference", parents=[common], help="the site as is without robots, one stage per seed", description=f"write suite {IN.REFERENCE_SUITE}: the site as is without robots at each seed, the robots parked off the map. Run it first: its flow cells are the joint search's occupancy.")
    q.set_defaults(fn=_cmd_reference)
    q = infra.add_parser(
        "premise",
        parents=[common, searched],
        help="the premise check: the two most different layouts with and without the fleet",
        description=f"write suite {IN.PREMISE_SUITE} (§10): the valid layout with every element at the narrowest section and the one with every element at the widest, saved as world copies with the crowd, each with the fleet and without robots on seeds {IN.PREMISE_SEEDS[0]}-{IN.PREMISE_SEEDS[-1]}; with the site as is from {IN.REFERENCE_SUITE}, §10's 15 episodes",
    )
    q.set_defaults(fn=_cmd_premise)
    q = infra.add_parser(
        "layouts",
        parents=[common, searched],
        help="the selection: robot-only pick, joint top 10, 5 random layouts",
        description=f"search the site's layouts (generative/flowedit/search.py): the robot-only pick, the joint top {SE.JOINT_TOP} distinct layouts by predicted conflict exposure (worst walker group) among the layouts within {SE.TRAVEL_SLACK:g} x the robot-only travel per job, and {SE.RANDOM_LAYOUTS} random valid layouts; save them as world copies with the crowd and write suite {IN.LAYOUT_SUITE}: each with the fleet and without robots on the selection seeds; writes the manifest layouts.json in the suite's bundle",
    )
    q.add_argument("--occupancy", type=Path, default=None, help="the reference runs' flow cells: flow_cells.parquet or the benchmark dir holding it (all walkers: the joint ranking's one group without --group-occupancy; the random control's occupancy column)")
    q.add_argument("--group-occupancy", type=Path, action="append", default=None, help="the reference runs' walker-group cells (`evaluation infra occupancy`: infra_occupancy.parquet or the benchmark dir holding it); repeatable, read as one")
    q.add_argument("--band", type=float, default=SE.CONFLICT_BAND_M, help=f"the conflict band: walkers within this distance of a driving robot count as exposed, m (default {SE.CONFLICT_BAND_M:g})")
    q.add_argument("--distinct", type=float, default=SE.DISTINCT_M, help=f"a joint layout enters the top only if, against every better-ranked one, at least one of its elements sits at least this far from that element's spot there, m (default {SE.DISTINCT_M:g})")
    q.add_argument("--guideline", action="store_true", help="also place the guideline layout (search.guideline, a practice-based baseline, not held to the travel pre-filter), save its copy and record it in the manifest for `flowedit infra evaluation` (the selection does not run it)")
    q.add_argument("--top", type=int, default=SE.JOINT_TOP, help=f"joint layouts that run in full (default {SE.JOINT_TOP})")
    q.add_argument("--random", type=int, default=SE.RANDOM_LAYOUTS, help=f"random valid layouts (default {SE.RANDOM_LAYOUTS})")
    q.add_argument("--travel-slack", type=float, default=SE.TRAVEL_SLACK, help=f"the travel pre-filter: robot travel per job at most this times the robot-only pick's (default {SE.TRAVEL_SLACK:g})")
    q.add_argument("--seed", type=int, default=0, help="seed of the random draw (default 0)")
    q.set_defaults(fn=_cmd_layouts)
    from . import demand as DE

    q = infra.add_parser(
        "search",
        help="demand at the recipient entrances: the travel-optimal, pedestrian-aware and practice-based layouts per reach",
        description="search the site's layouts for demand at the recipient entrances (generative/flowedit/demand.py; study design Part A): per reach, the fewest lockers that give every recipient entrance one within the reach as people walk, the travel-optimal layout, the pedestrian-aware pick at each travel margin and the practice-based layout, each with its robot travel per job, the predicted exposure per walker group and the walk to the locker per group. Prints them and writes a report with --out; writes no world or suite",
    )
    q.add_argument("--site", default=IN.SITE, help=f"the site: a world name or directory (default {IN.SITE})")
    q.add_argument("--site-config", type=Path, default=None, help=f"the site config (default {IN.SITES_DIR}/<site>.yaml): its entrances, links, courier access point and outlets")
    q.add_argument("--scenario", default=None, help="the crowd scenario whose sources and sinks no element takes (default the site config's)")
    q.add_argument("--scenarios-from", type=Path, default=None, help="world directory to read the scenario from (default: the site's)")
    q.add_argument("--duration", type=float, default=IN.EPISODE_S, help=f"sim seconds per episode of the occupancy's runs (default {IN.EPISODE_S:g})")
    q.add_argument("--loading", type=_point, default=None, help="x,y of the loading point, snapped to the nearest valid spot (default the site config's courier access point)")
    q.add_argument("--recipient-types", type=lambda text: [t.strip() for t in text.split(",") if t.strip()], default=list(DE.RECIPIENT_TYPES), help=f"entrance types that are recipient entrances (default {','.join(DE.RECIPIENT_TYPES)})")
    q.add_argument("--reach", type=_floats, default=list(DE.REACHES_M), help=f"the reaches, m: every recipient entrance has a locker within it as people walk (default {','.join(f'{r:g}' for r in DE.REACHES_M)})")
    q.add_argument("--margins", type=_floats, default=list(DE.MARGINS), help=f"the travel margins of the pedestrian-aware search (default {','.join(f'{m:g}' for m in DE.MARGINS)})")
    q.add_argument("--rate", type=float, default=None, help=f"jobs per hour in all, the same share to every recipient entrance (default {SE.TaskMix().rate_per_h:g})")
    q.add_argument("--pickup-share", type=float, default=IN.PICKUP_SHARE, help=f"share of pickups (default {IN.PICKUP_SHARE:g})")
    q.add_argument("--bay-share", type=float, default=DE.BAY_SHARE, help=f"share of the robots that wait at a bay slot, the others at a charger (default {DE.BAY_SHARE:g})")
    q.add_argument("--charging-trips", type=float, default=0.0, help="charging trips per job in the robot travel model, one drive between a bay slot and a charger each (the fleet-only run's rate; default 0)")
    q.add_argument("--speed", type=float, default=IN.SPEED_MPS, help=f"robot speed on its routes, m/s (default {IN.SPEED_MPS:g})")
    q.add_argument("--outlet-rule", action="store_true", help="apply the catalog's outlet cable rule to the lockers too (default: not applied to lockers; chargers keep it)")
    q.add_argument("--occupancy", type=Path, default=None, help="the reference runs' flow cells (all walkers): reported as group `all`, not ranked")
    q.add_argument("--group-occupancy", type=Path, action="append", default=None, help="the reference runs' walker-group cells (`evaluation infra occupancy`); repeatable, read as one. Without them no exposure is predicted and only --rank walk ranks")
    q.add_argument("--band", type=float, default=SE.CONFLICT_BAND_M, help=f"the conflict band of the predicted exposure, m (default {SE.CONFLICT_BAND_M:g})")
    q.add_argument(
        "--rank",
        choices=("step-free", "pooled", "exposure", "walk"),
        default="step-free",
        help="the pedestrian-aware ranking, by the worst-off group: step-free (rule S: the layouts that serve every recipient entrance from within the reach step-free first, where the travel cap holds one, then pooled; default), pooled (time lost per hour on the site: time near robots times --delay-per-exposure, time on element floor times --delay-per-element, plus the recipients' walk), exposure (the first study's term alone), walk (the walk to the locker, ties by exposure)",
    )
    q.add_argument(
        "--delay-per-exposure", type=_pairs, default=None, help=f"seconds of delay per second within --band of a driving robot per walker group, e.g. adult=0.3,wheelchair=0.5 (default at 2 m {','.join(f'{g}={v:g}' for g, v in DE.DELAY_PER_EXPOSURE.items())}, at 1 m the calibration; another band needs them given)"
    )
    q.add_argument("--delay-per-element", type=_pairs, default=None, help=f"seconds of delay per second on an element's floor per walker group (default {','.join(f'{g}={v:g}' for g, v in DE.DELAY_PER_ELEMENT.items())}, the calibration, the same in every band)")
    q.add_argument("--walk-speed", type=_pairs, default=None, help=f"walking speed per walker group, m/s (default {','.join(f'{g}={v:g}' for g, v in DE.WALK_SPEED_MPS.items())})")
    q.add_argument("--out", type=Path, default=None, help="the report file (JSON); an existing file is kept unless --force")
    q.add_argument("--force", action="store_true", help="replace an existing report file")
    q.set_defaults(fn=_cmd_search)
    q = infra.add_parser(
        "crowd",
        parents=[common],
        help="the study crowd from the site config's entrances",
        description="write a scenario directory <out>/<name>: every entrance of the site config a PEDS source and sink, the inflow split by entrance type (flow_split), through-walkers between entrances and locker visitors from the recipient entrances",
    )
    q.add_argument("--name", default="study_entrances", help="scenario name (default %(default)s)")
    q.add_argument("--rate", type=float, default=None, help=f"the inflow in all [ped/s] (default: the site config's crowd_density by Little's law, density x area / mean time in the site; without one {IN.CROWD_RATE:g}, study_mix_v2's)")
    q.add_argument("--density", type=float, default=None, help="the crowd density [ped/m2] the inflow keeps, in place of the site config's (crowd_density: its area and mean time set the rate)")
    q.add_argument("--visitor-rate", type=float, default=12.0, help="locker visitors per hour in all (default %(default)s)")
    q.set_defaults(fn=_cmd_crowd)
    q = infra.add_parser(
        "evaluation",
        parents=[common],
        help="the evaluation: the picks, the guideline layout and the site as is on fresh seeds",
        description=f"write suites {IN.EVALUATION_SUITE} (the joint pick, the robot-only pick and the guideline layout the selection recorded, each with the fleet and without robots) and {IN.EVALUATION_SITE_SUITE} (the site as is) on seeds {IN.EVALUATION_SEEDS[0]}-{IN.EVALUATION_SEEDS[-1]}, from the pick `evaluation infra pick` wrote and the selection's manifest",
    )
    q.add_argument("--pick", type=Path, default=None, help="the pick file `evaluation infra pick` writes (joint_pick, robot_only); needed without --report")
    q.add_argument("--no-guideline", action="store_true", help="leave out the guideline layout the selection's manifest records (`layouts --guideline`)")
    q.add_argument("--parts", type=_weights, default=None, metavar="W1,W2", help=parts_help)
    q.add_argument("--selection", type=Path, default=None, help=f"the selection's manifest (default <imported assets root>/suites/{IN.LAYOUT_SUITE}/layouts.json, else an older writer's {IN.LAYOUT_SUITE}.layouts.json beside it)")
    from . import recipients as RC

    g = q.add_argument_group(
        "layouts of `flowedit infra search` (study design Part A)",
        f"with --report: save the named layouts of a search report as world copies, each with one crowd scenario per seed that holds the seed's parcel recipients (generative/flowedit/recipients.py), and write suites {IN.ENTRANCE_SUITE}_site (the site as is) and {IN.ENTRANCE_SUITE}_without (each layout without robots as a crowd-only stage) with their manifest layouts.json in bundle {IN.ENTRANCE_SUITE}. The search space is built as the report's search built it: pass the same --site, --site-config, --scenario, --loading and --no-outlet-rule. Without --report none of these options is read",
    )
    g.add_argument("--report", type=Path, default=None, help="the report `flowedit infra search --out` wrote (any JSON that holds its layout records)")
    g.add_argument("--layouts", type=_named_layouts, default=None, metavar="ROLE=ID,...", help="the report's layouts to run, each as <role>=<id>, e.g. travel_optimal=83298e5c,practice_based=d837888b; the roles are the manifest's `picks`, no two layouts share one")
    g.add_argument("--without-robots-seeds", type=_seeds, default=None, help="the seeds at which each layout runs without robots as a crowd-only stage (default every seed); the manifest records them as crowd_only_seeds")
    g.add_argument("--writer-commit", default=None, help="the commit of the code that runs this command, for the manifest (a script that runs a copy of the code names the copy's commit)")
    g.add_argument(
        "--seed-part",
        type=_seed_part,
        action="append",
        default=None,
        metavar="TAG=SEEDS",
        help="also write the suites of a part of the seeds, <suite>_site_<tag> and <suite>_without_<tag>, e.g. h1=71-72: one lane of one machine, so that all runs of a seed are recorded on one machine; repeatable, the parts hold each seed once",
    )
    g.add_argument("--rate", type=float, default=None, help="jobs per hour in all, the same share to every recipient entrance; the recipients follow the seeds' jobs")
    g.add_argument("--pickup-share", type=float, default=IN.PICKUP_SHARE, help=f"share of pickups among the jobs (default {IN.PICKUP_SHARE:g})")
    g.add_argument("--recipient-offset", type=float, default=RC.OFFSET_S, help=f"seconds from a job's release to its recipient's arrival at the recipient entrance (default {RC.OFFSET_S:g}, the warm-up; a placeholder)")
    g.add_argument("--recipient-patience", type=float, default=RC.PATIENCE_S, help=f"seconds a recipient waits for the locker before it gives up and walks back (default {RC.PATIENCE_S:g}; the first study's visitors waited {RC.FIRST_STUDY_PATIENCE_S:g})")
    g.add_argument("--recipient-mix-from", default=None, help="the scenario whose sources give the recipients' walker groups (default the stage scenario); name the site's base crowd when the stage scenario holds tagged extras")
    g.add_argument("--recipient-kinds", type=lambda text: [k.strip() for k in text.split(",") if k.strip()], default=list(RC.RECIPIENT_KINDS), help=f"the job kinds that bring a recipient (default {','.join(RC.RECIPIENT_KINDS)}; delivery,pickup: every job)")
    g.add_argument("--recipient-mix", type=_shares, default=None, help="the recipients' walker groups as shares, e.g. adult=0.985,wheelchair=0.01,long_cane=0.005 (default the crowd's mix, from the scenario's sources)")
    g.add_argument(
        "--charge-power", type=float, default=RC.CHARGE_POWER_W, help=f"the power the copies' chargers charge with, W (default {RC.CHARGE_POWER_W:g}: the jackal's {RC.BATTERY_WH:g} Wh in {RC.CHARGE_TIME_H:g} h on top of its {RC.STANDBY_W:g} W standby draw; the first study's worlds have {IN.place.CHARGE_POWER_W:g})"
    )
    g.add_argument("--tag", default="", help="a tag in the crowd scenarios' names, <scenario>__<tag>__s<seed>: written scenarios are never replaced, other recipient settings at a seed need another name")
    g.add_argument("--loading", type=_point, default=None, help="x,y of the loading point, snapped to the nearest valid spot (default the site config's courier access point)")
    g.add_argument("--no-outlet-rule", action="store_true", help="the report's search ran without the locker's outlet cable rule (`flowedit infra search --no-outlet-rule`)")
    g.add_argument("--no-check", action="store_true", help="skip the check of each layout's crowd scenario that the runtime runs at an episode's start")
    from . import entrance_fleet as EF

    g.add_argument(
        "--arms",
        type=lambda text: [a.strip() for a in text.split(",") if a.strip()],
        default=["site", "without"],
        help="the suites to write: site (the site as is), without (each layout without robots, crowd only), fleet (each layout with the fleet: the seed's jobs to the locker that serves each job's recipient entrance, robots on their own battery, charging by need); default site,without. A capacity run of one layout is --arms fleet at each job rate, with a --tag and an --out per rate",
    )
    g.add_argument("--charge-states", type=Path, default=None, help="the fleet's start charge: a file of charge states (`flowedit infra charge-states`), copied beside the suite; one state is drawn per episode seed, each robot its place in it (default: every robot starts at --start-charge)")
    g.add_argument("--start-charge", type=float, default=EF.START_SOC, help=f"start charge of every robot without --charge-states, %% (default {EF.START_SOC:g}: full)")
    g.add_argument("--fleet-only", action="store_true", help="a fleet-only run: each layout with the fleet and no walkers (the site's scenario without its sources, <scenario>__fleet_only), only the fleet suite; with --charge-log the run that gives the charge states")
    g.add_argument("--charge-log", default="", help="with --fleet-only: the directory (as the recording machine sees it) the fleet logs the robots' charge to, one <stage>.csv per stage, for `flowedit infra charge-states`")
    g.add_argument("--charge-log-s", type=float, default=EF.CHARGE_LOG_S, help=f"seconds between two lines of the charge log (default {EF.CHARGE_LOG_S:g})")
    q.set_defaults(fn=_cmd_evaluation)
    q = infra.add_parser(
        "charge-states",
        help="the charge states of fleet-only runs, for the fleet's start charge",
        description="read the charge logs fleet-only runs wrote (`flowedit infra evaluation --report ... --fleet-only --charge-log DIR`) and write the charge states the fleet's start charge is drawn from (`--charge-states`): the robots' charges at moments of the runs, after a run-in and a fixed time apart, in the order of the robots' start spots",
    )
    q.add_argument("--log", type=Path, action="append", required=True, help="a charge log (<stage>.csv); repeatable")
    q.add_argument("--from-s", type=float, default=EF.RUN_IN_S, help=f"the run-in: rows before this many seconds of an episode are left out (default {EF.RUN_IN_S:g})")
    q.add_argument("--every-s", type=float, default=EF.STATE_EVERY_S, help=f"seconds between two states taken from an episode (default {EF.STATE_EVERY_S:g})")
    q.add_argument("--out", type=Path, required=True, help="the charge-state file (JSON)")
    q.add_argument("--force", action="store_true", help="replace an existing file")
    q.set_defaults(fn=_cmd_charge_states)
    q = infra.add_parser(
        "walkways",
        help="the walkway rule on a search report's layouts: their narrowest two-way and one-way walkway",
        description="check the layouts of a search report against the walkway rule (generative/flowedit/walkway.py): the width each element leaves where the site's crowd walks, 1.8 m where walkers pass both ways and 0.9 m one way, from the crowd scenario's routes; prints every layout's verdict, its narrowest two-way and one-way walkway and what fails, and writes them with --out",
    )
    q.add_argument("--site", default=IN.SITE, help=f"the site: a world name or directory (default {IN.SITE})")
    q.add_argument("--site-config", type=Path, default=None, help=f"the site config (default {IN.SITES_DIR}/<site>.yaml)")
    q.add_argument("--scenario", default=None, help="the crowd scenario (default the site config's)")
    q.add_argument("--scenarios-from", type=Path, default=None, help="world directory to read the scenario from (default: the site's)")
    q.add_argument("--report", type=Path, required=True, help="a search report (any JSON that holds its layout records)")
    q.add_argument("--layouts", type=lambda text: [t.strip() for t in text.split(",") if t.strip()], default=None, help="the layout ids to check (default all)")
    q.add_argument("--out", type=Path, default=None, help="the result file (JSON)")
    q.set_defaults(fn=_cmd_walkways)
    q = infra.add_parser(
        "walker-floor",
        help="export the floor the crowd walks on, in HumanSim's terms",
        description="write the walker floor of the site (generative/flowedit/walkway.py): the walls the simulator builds and the crowd's walls across exterior doors, every object with a bounding box and the docks, the zones' floor and its edges without a wall; what the walkway rule measures, from the same world model the runtime hands HumanSim",
    )
    q.add_argument("--site", default=IN.SITE, help=f"the site: a world name or directory (default {IN.SITE})")
    q.add_argument("--level", default=None, help="one level (default every level)")
    q.add_argument("--out", type=Path, required=True, help="the floor file (JSON)")
    q.set_defaults(fn=_cmd_walker_floor)
    q = infra.add_parser(
        "site-walls",
        aliases=["remove-walls"],
        help="take the walls the site config removes out of the site's world and its layout copies, and add the walls it adds",
        description="take the walls of the site config's removed_walls (objects the scan caught that are not part of the site, matched by level, kind and bounding box) out of the site's world and of every layout copy beside it (<site>__*), and add the walls of its added_walls (short walls closing gaps the site does not have, each end on another wall), in place: the wall entries in world.yaml and their pixels in map/<level>.png; everything else stays as it is. The other infra commands refuse a site world that still holds a removed wall or lacks an added one.",
    )
    q.add_argument("--site", default=IN.SITE, help=f"the site: a world name or directory (default {IN.SITE})")
    q.add_argument("--site-config", type=Path, default=None, help=f"the site config (default {IN.SITES_DIR}/<site>.yaml)")
    q.add_argument("--world", type=Path, action="append", default=None, help="a world directory to clean instead of the site and its copies; repeatable")
    q.add_argument("--dry-run", action="store_true", help="report what would be removed, change nothing")
    q.set_defaults(fn=_cmd_site_walls)
