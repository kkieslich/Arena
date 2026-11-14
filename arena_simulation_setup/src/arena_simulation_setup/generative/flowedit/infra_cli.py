"""`flowedit infra`: the INFRA Stage 1 suites (`infra.py`), in run order.

    flowedit infra reference                    the site as is without robots, selection seeds   -> infra_h1_site
    flowedit infra premise                      the 2 most different layouts, premise seeds      -> infra_h1_premise
    flowedit infra layouts --occupancy <run>    robot-only, joint top 10, 5 random, selection    -> infra_h1
    flowedit infra evaluation --pick <json>     the two picks and the site, evaluation seeds     -> infra_h1_eval(_site)

Each writes its suite under <imported assets root>/suites (where `arena evaluation benchmark --suite` finds it), the
layout suites a `<suite>.layouts.json` manifest beside it, and prints the benchmark call that runs it."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Collection, Iterable
from pathlib import Path
from typing import Any

from ..scenario.cli import world_root
from . import infra as IN
from . import search as SE
from . import suite as SU
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
    return args.out or IN.default_out(name)


def _summary(doc: dict[str, Any]) -> str:
    return f"{len(doc['stages'])} stages, {IN.episodes(doc)} episodes, {IN.sim_hours(doc):g} sim-hours"


def _cmd_reference(args: argparse.Namespace) -> int:
    site = open_site(_site_root(args.site))
    config = _config(args, site.key)
    scenario = _scenario(args, config)
    IN.scenario_dir(site, scenario, args.scenarios_from)
    seeds = args.seeds or list(IN.SELECTION_SEEDS)
    doc = IN.reference_suite(site.key, seeds, scenario=scenario, duration=args.duration, side=config.crowd_side if config is not None else None)
    name = args.out.stem if args.out else IN.REFERENCE_SUITE
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

    def __init__(self, args: argparse.Namespace) -> None:
        self.site = open_site(_site_root(args.site))
        self.config = _config(args, self.site.key)
        self.scenario_name = _scenario(args, self.config)
        self.side = self.config.crowd_side if self.config is not None else None
        self.scenario = IN.scenario_dir(self.site, self.scenario_name, args.scenarios_from)
        placements = dict(self.config.placements) if self.config is not None else {}
        exclude = self.config.exclude if self.config is not None else SE.EXCLUDED_ZONES
        loading = args.loading if args.loading is not None else (None if SE.LOADING in placements else SE.TWIN_LOADING_XY)
        self.mix = SE.TaskMix(rate_per_h=args.rate or SE.TaskMix().rate_per_h, pickup_share=args.pickup_share, episode_s=args.duration, top_ups_per_job=args.top_ups)
        self.rate_from: dict[str, Any] | None = None
        facts = self.config.facts if self.config is not None else None
        self.search = SE.Search(self.site, loading=loading, pins=dict(args.pin or []), exclude=exclude, placements=placements, keep_out=IN.keep_out(self.scenario / "scenario.yaml", self.config.links if self.config is not None else ()), mix=self.mix, facts=facts)
        _log(f"site config: {self.config.path if self.config is not None else '(none, the search defaults)'}")
        _log(f"search space: {json.dumps(self.search.counts())}")
        loading_spot = self.search.loading
        _log(f"loading point: {loading_spot.zone} piece {loading_spot.piece} step {loading_spot.step}, goto {loading_spot.objects[0].goto}")

    def loading_record(self) -> dict[str, Any]:
        s = self.search.loading
        return {"spot": s.id, "zone": s.zone, "piece": s.piece, "step": s.step, "goto": list(s.objects[0].goto)}

    def settle_rate(self, args: argparse.Namespace, pick: Any) -> tuple[float, Any]:
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
        for _ in range(8):
            _, wide = SE.premise_pair(self.search)
            estimate = IN.fleet_rate(wide.travel_m, speed_mps=args.speed, load_s=IN.LOAD_S, unload_s=IN.UNLOAD_S)
            if estimate == rate:
                break
            rate = estimate
            self.search.mix = attrs.evolve(self.mix, rate_per_h=rate)
        else:
            _log(f"the rate estimate did not settle, last {rate:g} jobs/h")
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
    """Remove the part suites an earlier run left beside `out` (`<suite>_p<N>.yaml`, written by flowedit infra) that
    this run did not write, so a suite's parts are always the ones its manifest lists."""
    pattern = re.compile(rf"^{re.escape(out.stem)}_p\d+\.yaml$")
    for path in sorted(out.parent.glob(f"{out.stem}_p*.yaml")):
        if pattern.match(path.name) and path.stem not in keep and path.read_text().startswith(IN.MARKER):
            path.unlink()
            _log(f"removed {path}: a part of an earlier run of {out.stem}")


def _write_parts(args: argparse.Namespace, out: Path, plan: list[list[str]] | None, worlds: dict[str, str], doc_for: Callable[[list[str]], dict[str, Any]], contest: str) -> list[dict[str, Any]]:
    """The suite of each part of `plan` (`_plan_parts`; `worlds`: layout id -> world name) beside `out`, each runnable on
    its own machine as its own run, and no part suite of an earlier run left; the parts as the manifest records them,
    [] without."""
    records: list[dict[str, Any]] = []
    plan = plan or []
    for index, (ids, weight) in enumerate(zip(plan, getattr(args, "parts", None) or [], strict=True), start=1):
        name = IN.part_name(out.stem, index)
        doc = doc_for([worlds[layout_id] for layout_id in ids])
        header = [
            f"part {index} of {len(plan)} of suite {out.stem} (weight {weight}): layouts {', '.join(ids)}",
            f"{_summary(doc)}; every part runs as its own run on its own machine, `evaluation infra` reads their run dirs together with {IN.manifest_path(out).name}",
            f"run: {SU.benchmark_command(name, weight, contest)}",
        ]
        path = IN.write(doc, out.with_name(f"{name}.yaml"), header, force=args.force)
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
    fleet = IN.fleet_config(rate, pickup_share=args.pickup_share, charge_below=args.charge_below, charge_until=args.charge_until)
    doc = IN.layout_suite([worlds[s.layout_id].name for s in unique], fleet, seeds, scenario=space.scenario_name, duration=args.duration, side=space.side)
    contest = IN.contest_literal(IN.contest(args.capacity_wh, soc_min=args.soc_min, soc_max=args.soc_max))
    out = _out(args, name)
    header = [
        *header,
        f"fleet: {IN.ROBOT}, rate {rate:g} jobs/h, charge below {args.charge_below:g} % until {args.charge_until:g} %; battery {args.capacity_wh:g} Wh and start charge {args.soc_min:g}-{args.soc_max:g} % through the contest, chargers {place.CHARGE_POWER_W:g} W in the worlds",
        f"references: {IN.REFERENCES} adds per stage the layout without robots, the traffic part's reference (§6)",
        f"{_summary(doc)}",
        f"run: {SU.benchmark_command(out.stem, args.lanes, contest)}",
    ]
    out = IN.write(doc, out, header, force=args.force)
    parts = _write_parts(args, out, plan, {s.layout_id: worlds[s.layout_id].name for s in unique}, lambda names: IN.layout_suite(names, fleet, seeds, scenario=space.scenario_name, duration=args.duration, side=space.side), contest)
    record = {
        "suite": out.stem,
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
    print(SU.benchmark_command(out.stem, args.lanes, contest))
    return 0


def _cmd_premise(args: argparse.Namespace) -> int:
    space = _Space(args)
    seeds = args.seeds or list(IN.PREMISE_SEEDS)
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
    seeds = args.seeds or list(IN.SELECTION_SEEDS)
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
    _log(f"travel model: {search.mix.top_ups_per_job:g} top-up round trips bay - charger per job; travel pre-filter: at most {args.travel_slack:g} x the robot-only pick's {robot.travel_m:.1f} m = {state['cap']:.1f} m per job: {state['within']} of {len(search.candidates)} candidate layouts (valid or not); the joint search ranks every one of them")
    guide = SE.guideline(search) if args.guideline else None
    runs = [("robot_only", robot), *(("joint", s) for s in joint), *(("random", s) for s in random)]
    exposure: dict[str, dict[str, float]] = {}
    if groups is None:
        _log("no group or flow-cell occupancy: the joint top 10 is left out of this dry run (the occupancy comes from the reference runs)")
    else:
        if len(joint) < args.top:
            _log(f"the travel pre-filter keeps only {len(joint)} distinct valid layouts, fewer than the top {args.top}")
        for _, s in [*runs, *([("guideline", guide)] if guide is not None else [])]:
            exposure[s.layout_id] = s.exposure if s.exposure is not None else SE.exposure_of(search, groups, s.index, rate_per_h=rate, band_m=args.band, speed_mps=args.speed)
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
        f"layouts: robot-only {robot.layout_id} (travel model with {search.mix.top_ups_per_job:g} top-ups per job); joint top {len(joint)} distinct ({args.distinct:g} m) by predicted conflict exposure ({args.band:g} m band, worst group), among the layouts within {args.travel_slack:g} x the robot-only travel ({state['cap']:.1f} m per job); {len(random)} random valid",
        f"the layouts' runs without robots pair with suite {IN.REFERENCE_SUITE} (the site as is) at the same seed; the pick: `evaluation infra pick`",
    ]

    def record(scored: SE.Scored) -> dict[str, Any]:
        return {**IN.picks_record({"guideline": scored}, ranking)[0], **({"exposure": exposure[scored.layout_id]} if scored.layout_id in exposure else {})}

    extra = {
        "occupancy": None if args.occupancy is None else str(args.occupancy),
        "group_occupancy": [str(p) for p in args.group_occupancy or []],
        "joint_ranking": {"score": "predicted conflict exposure, worst group (s per h per member)", "groups": sorted(groups or {}), "band_m": args.band, "rate_per_h": rate, "speed_mps": args.speed, "hold_s": list(SE.HOLD_S), "distinct_m": args.distinct, "top_ups_per_job": search.mix.top_ups_per_job},
        "travel_prefilter": {"slack": args.travel_slack, "max_travel_m": round(state["cap"], 3), "candidates_within": state["within"], "candidates": len(search.candidates)},
        "runs": [{**r, **({"exposure": exposure[r["layout_id"]]} if r["layout_id"] in exposure else {})} for r in IN.picks_record({"robot_only": robot, "joint": joint, "random": random}, ranking)],
        "joint_top": IN.picks_record({"joint": joint}, ranking),
    }
    kept = {"guideline": (guide, {k: v for k, v in record(guide).items() if k != "role"} | {"role": "guideline"})} if guide is not None else None
    return _write_layout_suite(args, space, IN.LAYOUT_SUITE, runs, seeds, rate, header, extra, kept)


def _cmd_crowd(args: argparse.Namespace) -> int:
    import shapely

    from ...shared.navmesh import nav_walls

    site = open_site(_site_root(args.site))
    config = _config(args, site.key)
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
    header = "\n".join([
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
    ])
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
    return rate, (
        f"{density:g} ped/m2 on {cd.area_m2:g} m2 = {cd.present(density):.0f} walkers present; by Little's law over the"
        f" mean time in the site, {cd.mean_time_s:g} s (site config crowd_density), {rate:.4g} ped/s in all"
    )


def _cmd_evaluation(args: argparse.Namespace) -> int:
    pick = json.loads(Path(args.pick).read_text())
    selection = Path(args.selection) if args.selection else IN.manifest_path(IN.default_out(IN.LAYOUT_SUITE))
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
    out = args.out or IN.default_out(IN.EVALUATION_SUITE)
    site_out = out.with_name(f"{IN.EVALUATION_SITE_SUITE}.yaml") if args.out is None else out.with_name(f"{out.stem}_site.yaml")
    common = f"seeds {', '.join(map(str, seeds))}, {duration:g} s each, warm-up {float(manifest['warmup_s']):g} s cut in the analysis"
    arms = f"the joint pick {chosen['joint']}, the robot-only pick {chosen['robot_only']}" + (f" and the guideline layout {chosen['guideline']}" if "guideline" in chosen else "")
    IN.write(doc, out, [f"INFRA Stage 1 evaluation (§10, §12): {arms} from {selection}, with the fleet and without robots, {common}", f"{_summary(doc)}", f"run: {SU.benchmark_command(out.stem, args.lanes, contest)}"], force=args.force)
    parts = _write_parts(args, out, plan, {layout_id: Path(worlds[layout_id]).name for layout_id in dict.fromkeys(chosen.values())}, lambda names: IN.layout_suite(names, fleet, seeds, scenario=scenario, duration=duration, side=side), contest)
    IN.write(site_doc, site_out, [f"INFRA Stage 1 evaluation reference: {manifest['site']} as is without robots, {common}", f"{_summary(site_doc)}", f"run: {SU.benchmark_command(site_out.stem, args.lanes)}"], force=args.force)
    record = {
        "suite": out.stem,
        "site_suite": site_out.stem,
        "selection": str(selection),
        "pick": str(args.pick),
        "picks": chosen,
        **{key: manifest[key] for key in ("site", "site_root", "scenario", "visitor_sources", "episode_s", "warmup_s", "rate_per_h", "fleet", "contest") if key in manifest},
        "seeds": seeds,
        "episodes": total,
        "sim_hours": round(IN.sim_hours(doc) + IN.sim_hours(site_doc), 2),
        "runs": [{**(next(r for r in manifest["runs"] if r["layout_id"] == layout_id) if role != "guideline" else {k: v for k, v in guide.items() if k != "world"}), "role": role} for role, layout_id in chosen.items()],
        "worlds": {layout_id: worlds[layout_id] for layout_id in chosen.values()},
        **({"parts": parts} if parts else {}),
    }
    manifest_out = IN.manifest_path(out)
    manifest_out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {out}: {_summary(doc)}; {site_out}: {_summary(site_doc)}; {manifest_out}")
    print(SU.benchmark_command(site_out.stem, args.lanes))
    print(SU.benchmark_command(out.stem, args.lanes, contest))
    return 0


def add_parser(verbs: Any) -> None:
    """The `infra` verb and its subcommands."""
    p = verbs.add_parser("infra", help="INFRA Stage 1 suites: the site as is, the premise check, the selection and the evaluation", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    infra = p.add_subparsers(dest="infra", required=True, metavar="{reference,premise,layouts,evaluation}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--site", default=IN.SITE, help=f"the site: a world name or directory (default {IN.SITE}); layout copies hardlink its assets, so save on its mount")
    common.add_argument("--site-config", type=Path, default=None, help=f"the site config (default {IN.SITES_DIR}/<site>.yaml): where each element may go, the excluded zones, the crowd scenario")
    common.add_argument("--scenario", default=None, help=f"the crowd scenario of every stage (default the site config's, else {IN.SCENARIO})")
    common.add_argument("--scenarios-from", type=Path, default=None, help="world directory to copy the scenario from (default: the site's, else the world its name resolves to)")
    common.add_argument("--seeds", type=_seeds, default=None, help=f"episode seeds, e.g. 1-5 (default: reference and layouts {IN.SELECTION_SEEDS[0]}-{IN.SELECTION_SEEDS[-1]}, premise {IN.PREMISE_SEEDS[0]}-{IN.PREMISE_SEEDS[-1]}, evaluation {IN.EVALUATION_SEEDS[0]}-{IN.EVALUATION_SEEDS[-1]})")
    common.add_argument("--duration", type=float, default=IN.EPISODE_S, help=f"sim seconds per episode (default {IN.EPISODE_S:g})")
    common.add_argument("--out", type=Path, default=None, help="suite file (default: <imported assets root>/suites/<suite name>.yaml)")
    common.add_argument("--lanes", type=int, default=2, help="lanes in the printed benchmark call (default 2)")
    common.add_argument("--force", action="store_true", help="replace a suite file flowedit infra did not write")
    common.add_argument("--dry-run", action="store_true", help="search and print without writing worlds or suites")
    searched = argparse.ArgumentParser(add_help=False)
    searched.add_argument("--loading", type=_point, default=None, help="x,y of the loading point, snapped to the nearest valid spot (default the site config's pin)")
    searched.add_argument("--pin", type=_pin, action="append", default=None, help="pin a searched element, e.g. locker=150.1,160.2 (over the site config); repeatable")
    searched.add_argument("--rate", type=float, default=None, help="jobs per hour in every layout (the premise check's measured rate, 75 %% of the fleet's capacity in the wide premise layout; default: the same share of that layout's capacity estimated at --speed without charging or crowd, an upper bound)")
    searched.add_argument("--speed", type=float, default=IN.SPEED_MPS, help=f"robot speed for the rate estimate, m/s (default {IN.SPEED_MPS:g}, the jackal's planner limit)")
    searched.add_argument("--pickup-share", type=float, default=IN.PICKUP_SHARE, help=f"share of pickups (default {IN.PICKUP_SHARE:g})")
    searched.add_argument("--charge-below", type=float, default=IN.CHARGE_BELOW, help=f"battery percent that sends a free robot to charge (default {IN.CHARGE_BELOW:g})")
    searched.add_argument("--charge-until", type=float, default=IN.CHARGE_UNTIL, help=f"battery percent that frees a charging robot (default {IN.CHARGE_UNTIL:g})")
    searched.add_argument("--capacity-wh", type=float, default=IN.CAPACITY_WH, help=f"battery capacity of every robot, Wh (default {IN.CAPACITY_WH:g})")
    searched.add_argument("--visitor-rate", type=float, default=None, help=f"locker visitors per hour in all (default the scenario's {IN.VISITORS})")
    searched.add_argument("--soc-min", type=float, default=IN.SOC_MIN, help=f"lowest start charge each robot draws per seed, %% (default {IN.SOC_MIN:g})")
    searched.add_argument("--soc-max", type=float, default=IN.SOC_MAX, help=f"highest start charge, %% (default {IN.SOC_MAX:g})")
    searched.add_argument("--top-ups", type=float, default=0.0, help="top-up round trips bay - charger per job in the robot travel model (the task-rate run's top-ups dispatched after the warm-up per completed job released after it; default 0)")
    parts_help = "also split the suite by layout copy into one suite per weight, <suite>_p1, <suite>_p2, ..., in proportion to the weights (e.g. 4,6: the lanes of two machines); the manifest lists the parts; each part runs as its own run on its own machine, and `evaluation infra` reads their run dirs together (repeated --layouts)"
    searched.add_argument("--parts", type=_weights, default=None, metavar="W1,W2[,...]", help=parts_help)

    q = infra.add_parser("reference", parents=[common], help="the site as is without robots, one stage per seed", description=f"write suite {IN.REFERENCE_SUITE}: the site as is without robots at each seed, the robots parked off the map. Run it first: its flow cells are the joint search's occupancy.")
    q.set_defaults(fn=_cmd_reference)
    q = infra.add_parser("premise", parents=[common, searched], help="the premise check: the two most different layouts with and without the fleet", description=f"write suite {IN.PREMISE_SUITE} (§10): the valid layout with every element at the narrowest section and the one with every element at the widest, saved as world copies with the crowd, each with the fleet and without robots on seeds {IN.PREMISE_SEEDS[0]}-{IN.PREMISE_SEEDS[-1]}; with the site as is from {IN.REFERENCE_SUITE}, §10's 15 episodes")
    q.set_defaults(fn=_cmd_premise)
    q = infra.add_parser("layouts", parents=[common, searched], help="the selection: robot-only pick, joint top 10, 5 random layouts", description=f"search the site's layouts (generative/flowedit/search.py): the robot-only pick, the joint top {SE.JOINT_TOP} distinct layouts by predicted conflict exposure (worst walker group) among the layouts within {SE.TRAVEL_SLACK:g} x the robot-only travel per job, and {SE.RANDOM_LAYOUTS} random valid layouts; save them as world copies with the crowd and write suite {IN.LAYOUT_SUITE}: each with the fleet and without robots on the selection seeds; writes <suite>.layouts.json beside it")
    q.add_argument("--occupancy", type=Path, default=None, help="the reference runs' flow cells: flow_cells.parquet or the benchmark dir holding it (all walkers: the joint ranking's one group without --group-occupancy; the random control's occupancy column)")
    q.add_argument("--group-occupancy", type=Path, action="append", default=None, help="the reference runs' walker-group cells (`evaluation infra occupancy`: infra_occupancy.parquet or the benchmark dir holding it); repeatable, read as one")
    q.add_argument("--band", type=float, default=SE.CONFLICT_BAND_M, help=f"the conflict band: walkers within this distance of a driving robot count as exposed, m (default {SE.CONFLICT_BAND_M:g})")
    q.add_argument("--distinct", type=float, default=SE.DISTINCT_M, help=f"a joint layout enters the top only if one of its elements sits at least this far from that element's spot in every better-ranked one, m (default {SE.DISTINCT_M:g})")
    q.add_argument("--guideline", action="store_true", help="also place the guideline layout (search.guideline), save its copy and record it in the manifest for `flowedit infra evaluation` (the selection does not run it)")
    q.add_argument("--top", type=int, default=SE.JOINT_TOP, help=f"joint layouts that run in full (default {SE.JOINT_TOP})")
    q.add_argument("--random", type=int, default=SE.RANDOM_LAYOUTS, help=f"random valid layouts (default {SE.RANDOM_LAYOUTS})")
    q.add_argument("--travel-slack", type=float, default=SE.TRAVEL_SLACK, help=f"the travel pre-filter: robot travel per job at most this times the robot-only pick's (default {SE.TRAVEL_SLACK:g})")
    q.add_argument("--seed", type=int, default=0, help="seed of the random draw (default 0)")
    q.set_defaults(fn=_cmd_layouts)
    q = infra.add_parser("crowd", parents=[common], help="the study crowd from the site config's entrances", description="write a scenario directory <out>/<name>: every entrance of the site config a PEDS source and sink, the inflow split by entrance type (flow_split), through-walkers between entrances and locker visitors from the recipient entrances")
    q.add_argument("--name", default="study_entrances", help="scenario name (default %(default)s)")
    q.add_argument("--rate", type=float, default=None, help=f"the inflow in all [ped/s] (default: the site config's crowd_density by Little's law, density x area / mean time in the site; without one {IN.CROWD_RATE:g}, study_mix_v2's)")
    q.add_argument("--density", type=float, default=None, help="the crowd density [ped/m2] the inflow keeps, in place of the site config's (crowd_density: its area and mean time set the rate)")
    q.add_argument("--visitor-rate", type=float, default=12.0, help="locker visitors per hour in all (default %(default)s)")
    q.set_defaults(fn=_cmd_crowd)
    q = infra.add_parser("evaluation", parents=[common], help="the evaluation: the picks, the guideline layout and the site as is on fresh seeds", description=f"write suites {IN.EVALUATION_SUITE} (the joint pick, the robot-only pick and the guideline layout the selection recorded, each with the fleet and without robots) and {IN.EVALUATION_SITE_SUITE} (the site as is) on seeds {IN.EVALUATION_SEEDS[0]}-{IN.EVALUATION_SEEDS[-1]}, from the pick `evaluation infra pick` wrote and the selection's manifest")
    q.add_argument("--pick", type=Path, required=True, help="the pick file `evaluation infra pick` writes (joint_pick, robot_only)")
    q.add_argument("--no-guideline", action="store_true", help="leave out the guideline layout the selection's manifest records (`layouts --guideline`)")
    q.add_argument("--parts", type=_weights, default=None, metavar="W1,W2", help=parts_help)
    q.add_argument("--selection", type=Path, default=None, help=f"the selection's manifest (default <imported assets root>/suites/{IN.LAYOUT_SUITE}.layouts.json)")
    q.set_defaults(fn=_cmd_evaluation)
