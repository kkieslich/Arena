"""`flowedit`: build the FLOWEDIT graph of a world and scenario, edit it, generate edits from a seed, read a site back into a scenario, record a PEDS run of it, write a split as a seed-paired benchmark suite, forecast its flow, and write the INFRA Stage 1 suites."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from arena_simulation_setup.kg import dumps, write
from arena_simulation_setup.tree.World.write import to_yaml

from ..scenario.cli import world_root
from ..scenario.materialize import WouldOverwrite
from . import settings as S
from . import split as SP
from . import suite as SU
from .build import build
from .edit import GRAPH, EditType, apply, dump_edits, open_site, save
from .generate import controls, dump_manifest, generate
from .read import read, render
from .read import write as write_scenario
from .run import DURATION_S, RUNS, Edit, base_site, check_writable, default_out, done, prepare, record

MANIFESTS = "flowedit"


def _cmd_build(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world)
    scenario = root / "scenarios" / args.scenario / "scenario.yaml" if args.scenario else None
    if scenario is not None and not scenario.is_file():
        raise FileNotFoundError(f"no scenario {args.scenario!r} at {scenario}")
    notes: list[str] = []
    scene = build(root, scenario, args.seed, notes)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    if not args.write:
        print(dumps(scene), end="")
        return 0
    out = args.out or (scenario.parent if scenario is not None else root) / GRAPH
    write(scene, out)
    print(f"wrote {out}")
    return 0


def _cmd_read(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world)
    notes: list[str] = []
    site = open_site(root, args.scenario, None, notes)
    name = args.name or f"flowedit_{site.scenario or 'world'}"
    settings = S.load(args.settings, count=args.count, scale=args.scale, local_planner=args.local_planner)
    base = base_site(args.worlds, site) if site.edits else None
    result = read(site, settings, args.seed, notes, base=base)
    for note in result.notes:
        print(f"note: {note}", file=sys.stderr)
    people = len(result.doc["humans"])
    print(f"{site.name}/{name}: {people} people, needs {', '.join(f'{n}->{p.target} ({p.kind}{', meet' if p.meet else ''})' for n, p in result.needs.items()) or 'none'}, entrances {result.provenance['entrances'] or 'none'}", file=sys.stderr)
    if not args.write:
        print(render(result, name)["scenario.yaml"], end="")
        print("(dry run - pass --write to write it)", file=sys.stderr)
        return 0
    check_writable(args.worlds, args.world, root)
    try:
        out = write_scenario(result, name, force=args.force)
    except WouldOverwrite as exc:
        print(f"flowedit: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


def _param(text: str) -> tuple[list[str], Any]:
    """`<key>[.<key>...]=<YAML value>`, the dotted key split."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    return key.split("."), yaml.safe_load(value)


def _params(pairs: list[tuple[list[str], Any]]) -> dict[str, Any]:
    """The `-p` pairs as one params mapping, dotted keys nested."""
    params: dict[str, Any] = {}
    for keys, value in pairs:
        node = params
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"param {'.'.join(keys)}: {key!r} is already a value")
            node = child
        node[keys[-1]] = value
    return params


def _cmd_edit(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world)
    notes: list[str] = []
    site = apply(open_site(root, args.scenario, args.seed, notes), args.type, args.target, _params(args.param))
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    if not args.write:
        print(to_yaml({"world": site.name, **dump_edits(site)}), end="")
        return 0
    print(f"wrote {save(site)}")
    return 0


def _types(text: str) -> list[EditType]:
    """Comma-separated edit types."""
    try:
        return [EditType(t.strip()) for t in text.split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"edit types are {', '.join(EditType)}, got {text!r}") from exc


def _cmd_generate(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world)
    notes: list[str] = []
    site = open_site(root, args.scenario, None, notes)
    chains = generate(site, args.seed, args.per_type, args.steps, args.types, notes)
    made = {chain.site.name: controls(site, chain) for chain in chains} if args.controls else {}
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    if not args.write:
        print(to_yaml(dump_manifest(site, chains, args.seed, args.per_type, args.steps, chain_controls=made)), end="")
        return 0
    worlds = {chain.site.name: save(chain.site) for chain in chains}
    worlds.update({control.name: save(control) for pairs in made.values() for _, control in pairs})
    for path in worlds.values():
        print(f"wrote {path}")
    if not worlds:
        print("flowedit: nothing generated, no manifest written", file=sys.stderr)
        return 0
    out = args.manifest or next(iter(worlds.values())).parent.parent / MANIFESTS / f"{site.name}__seed{args.seed}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_yaml(dump_manifest(site, chains, args.seed, args.per_type, args.steps, worlds, made)))
    print(f"wrote {out}")
    return 0


def _edit_type(text: str) -> EditType:
    try:
        return EditType(text)
    except ValueError as exc:
        raise ValueError(f"edit types are {', '.join(EditType)}, got {text!r}") from exc


def _cmd_run(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world)
    notes: list[str] = []
    site = open_site(root, args.scenario, args.edit_seed, notes)
    base = base_site(args.worlds, site)
    edits = [Edit(_edit_type(args.edit[0]), args.edit[1], _params(args.param))] if args.edit else []
    if args.param and not edits:
        raise ValueError("-p sets a param of --edit, pass --edit TYPE TARGET with it")
    if not edits and not args.dry_run:
        # the scenario goes into this world itself, the base world or one an earlier edit saved
        check_writable(args.worlds, args.world, root)
    settings = S.load(args.settings, count=args.count, scale=args.scale, local_planner=args.local_planner)
    out = args.out or default_out()
    prepared = prepare(site, base, edits, settings, args.seed, notes, dry_run=args.dry_run)
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    run_dir = out / prepared.run_id
    if args.dry_run:
        from .run import launch_command

        print(to_yaml({"run": str(run_dir), "world": prepared.site.name, "base": prepared.site.base, "edits": dump_edits(prepared.site)["edits"], "scenario": prepared.scenario, "launch": launch_command(prepared.site.name, prepared.scenario, run_dir / "humansim", args.launch_arg)}), end="")
        print("(dry run - nothing saved or launched)", file=sys.stderr)
        return 0
    if done(run_dir) and not args.again:
        print(f"{run_dir}: already recorded, pass --again to run it again")
        return 0
    try:
        recorded = record(prepared, out, args.duration, launch_args=args.launch_arg, startup_timeout=args.startup_timeout, log=lambda m: print(m, file=sys.stderr))
    except RuntimeError as exc:
        print(f"flowedit: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {recorded}")
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    doc = SP.load(args.split)
    out = args.out or default_out() / args.split.stem
    parts = args.partition or list(SP.PARTITIONS)
    crowds = _crowds(doc, args.crowd)
    if args.dry_run:
        for job in (j for name in crowds for j in SP.jobs(doc, parts, name)):
            print(f"{job.crowd}\t{job.partition}\t{job.world}\tseed {job.seed}" + ("\treference" if job.reference else "") + ("\tbaseline only" if job.baseline else ""))
        return 0
    counts = SP.sweep(doc, out, partitions=parts, crowds=crowds, settings=args.settings, jobs_at_once=args.jobs, domain_base=args.domain_base, regenerate=not args.no_generate, stagger=args.stagger, log=lambda m: print(m, file=sys.stderr))
    print(f"{out}: {counts['done']} recorded, {counts['skipped']} already there, {counts['failed']} failed")
    return 1 if counts["failed"] else 0


def _crowds(doc: dict[str, Any], chosen: list[str] | None) -> list[str]:
    """The crowds `--crowd` picks, `all` for every one; the base crowd without it."""
    if not chosen:
        return [SP.BASE_CROWD]
    names = SP.crowd_names(doc) if SP.ALL_CROWDS in chosen else list(dict.fromkeys(chosen))
    for name in names:
        SP.partitions_of(doc, name)
    return names


def _cmd_suite(args: argparse.Namespace) -> int:
    doc = SP.load(args.split)
    parts = args.partition or list(SP.PARTITIONS)
    SP.partitions_of(doc, args.crowd)
    crowd = SP.crowd(doc, args.settings, args.crowd)
    # before any world is generated or read: an empty suite would only show after every world was made
    if not SP.jobs(doc, parts, args.crowd):
        raise ValueError(f"crowd {args.crowd} runs on {SP.partitions_of(doc, args.crowd)}, none of {parts} lists a world for it: the suite would be empty")
    suite = SU.build(doc, crowd, parts, worlds=args.worlds, robot=args.robot, crowd_name=args.crowd, regenerate=not args.no_generate, dry_run=args.dry_run, log=lambda m: print(m, file=sys.stderr))
    stages = len(suite["stages"])
    contestants = SP.contest_of(doc)
    if args.dry_run:
        print(to_yaml(suite), end="")
        if contestants:
            print(f"(contest: {', '.join(c['name'] for c in contestants)})", file=sys.stderr)
        print(f"(dry run - no scenario or suite written, {stages} stages)", file=sys.stderr)
        return 0
    header = [f"split {args.split.name}, partitions {', '.join(parts)}, crowd {args.crowd}, settings {S.digest(crowd)}", f"{stages} stages, each one episode at the seed its name ends in; an edit stage pairs with its base world's stage at that seed"]
    out = Path(args.out or SU.default_out(args.split, args.crowd))
    try:
        if contestants:
            contest = SU.write(SU.contest_doc(out.stem, contestants), SU.contest_out(out), [f"split {args.split.name}: the planners of suite {out.stem}"], force=args.force)
        written = SU.write(suite, out, header, force=args.force)
    except ValueError as exc:
        print(f"flowedit: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {written}: {stages} stages")
    if contestants:
        print(f"wrote {contest}: {len(contestants)} contestants")
    # the contest has the suite's name, which `--contest` resolves as `--suite` does
    command = SU.benchmark_command(written.stem, args.lanes, *([written.stem] if contestants else []))
    root = SU.imported_for(written)
    if root is None:
        print(f"note: `--suite {written.stem}` resolves under {SU.default_out(args.split, args.crowd).parent}, not {written.parent}; write the suite with the default --out, or into a directory named {SU.SUITES}/", file=sys.stderr)
    elif root != SU.default_out(args.split, args.crowd).parent.parent.resolve():
        # the runner finds the suite under ARENA_ASSETS_DIR_IMPORTED, so the command names the root the suite was written to
        command = f"ARENA_ASSETS_DIR_IMPORTED={shlex.quote(str(root))} {command}"
    print(command)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowedit", description="FLOWEDIT graphs: each object's demand on its KG node. `flowedit <verb> --help` lists a verb's options.")
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{build,edit,generate,read,run,sweep,suite,forecast,infra}")
    p = verbs.add_parser("build", help="build the graph of a world and scenario", description="build the graph of a world and scenario, a dry run that prints it without --write")
    p.add_argument("world", help="the world to build from")
    p.add_argument("scenario", nargs="?", help="the scenario under the world's scenarios/ whose objects join the graph (default: world objects only)")
    p.add_argument("--seed", type=int, default=0, help="seed of the point drawn for an object posed at a zone name (default 0)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds)")
    p.add_argument("--out", type=Path, default=None, help=f"file to write (default: {GRAPH} next to the scenario.yaml, or in the world without a scenario)")
    p.add_argument("--write", action="store_true", help="write the graph (default: print it)")
    p.set_defaults(fn=_cmd_build)
    p = verbs.add_parser("edit", help="apply one edit and save the edited world", description="apply one edit to a world's graph and geometry, a dry run that prints the edit record without --write, which saves the world as <base>__<edit id> under $ARENA_ASSETS_DIR_LOCAL/worlds")
    p.add_argument("world", help="the world to edit, shipped or saved by an earlier edit")
    p.add_argument("type", type=EditType, choices=list(EditType), help="the edit type")
    p.add_argument("target", help="what it changes: obj:<name>, door:<name>, schedule:<name>, wall:<level>[/<index>]")
    p.add_argument("-p", "--param", type=_param, action="append", default=[], metavar="KEY=VALUE", help="an edit param, VALUE read as YAML, KEY dotted for nesting (satisfies.thirst=80); repeatable")
    p.add_argument("--scenario", default=None, help="the scenario whose objects join the graph of an unedited world (default: world objects only)")
    p.add_argument("--seed", type=int, default=None, help="seed of the point drawn for an object posed at a zone name (default 0, or the seed an edited world was built with)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
    p.add_argument("--write", action="store_true", help="save the edited world (default: print the edit)")
    p.set_defaults(fn=_cmd_edit)
    p = verbs.add_parser(
        "generate",
        help="sample edits of every type from a seed and save the edited worlds",
        description="sample per edit type --per-type chains of --steps edits on a world (and scenario) from --seed, a dry run that prints the manifest without --write, which saves every edited world under $ARENA_ASSETS_DIR_LOCAL/worlds and the manifest beside them",
    )
    p.add_argument("world", help="the world to edit, shipped or saved by an earlier edit")
    p.add_argument("--scenario", default=None, help="the scenario whose objects join the graph and travel with each edited world (default: world objects only)")
    p.add_argument("--seed", type=int, default=0, help="seed of the edits (default 0)")
    p.add_argument("--per-type", type=int, default=1, help="chains per edit type (default 1)")
    p.add_argument("--steps", type=int, default=1, help="edits per chain, the first of the chain's type (default 1)")
    p.add_argument("--types", type=_types, default=list(EditType), metavar="TYPE[,TYPE...]", help=f"edit types to start chains with and draw later steps from (default: {','.join(EditType)})")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
    p.add_argument("--controls", action="store_true", help="also make the geometry-only controls of every function-carrying move or insert: the model without demand at the new spot, and for a move the model moved with its demand left at the old spot")
    p.add_argument("--manifest", type=Path, default=None, help=f"manifest file to write (default: $ARENA_ASSETS_DIR_LOCAL/{MANIFESTS}/<world>__seed<seed>.yaml)")
    p.add_argument("--write", action="store_true", help="save the edited worlds and the manifest (default: print the manifest)")
    p.set_defaults(fn=_cmd_generate)

    p = verbs.add_parser(
        "read",
        help="read a site and crowd settings into a scenario",
        description="read the FLOWEDIT site of a world (shipped, or saved by `flowedit edit`) and crowd settings into <world>/scenarios/<name>/, a dry run that prints the scenario without --write. The people come from the unedited base world, so every edit of a world gets the same people for the same settings and seed",
    )
    p.add_argument("world", help="the world to read, shipped or saved by an edit")
    p.add_argument("scenario", nargs="?", help="the scenario whose objects join the graph of a shipped world (an edited world names its own)")
    p.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over the defaults, a distribution replacing the default one whole (see settings.py)")
    p.add_argument("--count", type=int, default=None, help="people at the start, overriding the settings")
    p.add_argument("--scale", type=float, default=None, help="scale of populate's per-room count, overriding the settings")
    p.add_argument("--local-planner", default=None, help="HumanSim local planner of every generated type, overriding the settings")
    p.add_argument("--seed", type=int, default=0, help="seed of every crowd draw (default 0)")
    p.add_argument("--name", default=None, help="scenario directory to write (default: flowedit_<scenario>)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
    p.add_argument("--write", action="store_true", help="write the scenario (default: print it)")
    p.add_argument("--force", action="store_true", help="replace a scenario of that name that was not generated")
    p.set_defaults(fn=_cmd_read)

    p = verbs.add_parser(
        "run",
        help="edit, read and record one PEDS run",
        description="apply one edit (or none) to a world and save it, read the crowd into it under the settings and seed, launch it headless with HumanSim recording, and write the run under --out: run.yaml, launch.log, humansim/ (the bag) and tracks.parquet",
    )
    p.add_argument("world", help="the world to run, shipped or saved by an earlier edit (a chain of edits runs on the world its last step saved)")
    p.add_argument("scenario", nargs="?", help="the scenario whose objects join the graph of a shipped world (an edited world names its own)")
    p.add_argument("--edit", nargs=2, metavar=("TYPE", "TARGET"), default=None, help=f"the edit to apply first, TYPE one of {', '.join(EditType)}, TARGET as for `flowedit edit` (default: run the world as it is)")
    p.add_argument("-p", "--param", type=_param, action="append", default=[], metavar="KEY=VALUE", help="a param of --edit, as for `flowedit edit`; repeatable")
    p.add_argument("--edit-seed", type=int, default=None, help="seed of the point drawn for an object posed at a zone name (default 0, or the seed an edited world was built with)")
    p.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over the defaults (see settings.py)")
    p.add_argument("--count", type=int, default=None, help="people at the start, overriding the settings")
    p.add_argument("--scale", type=float, default=None, help="scale of populate's per-room count, overriding the settings")
    p.add_argument("--local-planner", default=None, help="HumanSim local planner of every generated type, overriding the settings")
    p.add_argument("--seed", type=int, default=0, help="seed of every crowd draw (default 0)")
    p.add_argument("--duration", type=float, default=DURATION_S, help=f"seconds of the run recorded in the tracks (default {DURATION_S:g})")
    p.add_argument("--out", type=Path, default=None, help=f"runs root (default: $ARENA_DATA_DIR/{RUNS})")
    p.add_argument("--launch-arg", action="append", default=[], metavar="KEY:=VALUE", help="an extra `arena launch` argument; repeatable")
    p.add_argument("--startup-timeout", type=float, default=300.0, help="seconds to wait for HumanSim to start recording (default 300)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
    p.add_argument("--again", action="store_true", help="run again when the run directory already holds a finished run")
    p.add_argument("--dry-run", action="store_true", help="print the edited world, scenario and launch command without saving or launching")
    p.set_defaults(fn=_cmd_run)

    p = verbs.add_parser(
        "sweep",
        help="record every run of a split",
        description="generate the split's edited worlds again, then `flowedit run` every world of the chosen partitions once per seed and chosen crowd, --jobs launches at a time on their own ROS domains; finished runs are skipped",
    )
    p.add_argument("split", type=Path, help="the split file (see split.py)")
    p.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition to run; repeatable (default: all)")
    p.add_argument("--crowd", action="append", default=None, help=f"a crowd of the split to record, `{SP.BASE_CROWD}` or one of its `crowds`, `all` for every one; repeatable (default: {SP.BASE_CROWD}, the split's own settings)")
    p.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over every chosen crowd's settings")
    p.add_argument("--out", type=Path, default=None, help=f"runs root (default: $ARENA_DATA_DIR/{RUNS}/<split name>)")
    p.add_argument("--jobs", type=int, default=1, help="launches at a time (default 1)")
    p.add_argument("--domain-base", type=int, default=150, help="ROS domain of the first launch slot, the next slots counting up (default 150)")
    p.add_argument("--stagger", type=float, default=30.0, help="seconds at least between two launches starting (default 30)")
    p.add_argument("--no-generate", action="store_true", help="skip generating the edited worlds again")
    p.add_argument("--dry-run", action="store_true", help="list the runs without launching")
    p.set_defaults(fn=_cmd_sweep)

    p = verbs.add_parser(
        "suite",
        help="write a split as a seed-paired benchmark suite",
        description="generate the split's edited worlds again, read the crowd into every world at every seed of its partition, and write an arena_evaluation benchmark suite of one stage per world and seed, the stage's episode seed the reader's seed, so an edit stage pairs with its base world's stage at that seed; with the split's robot_task the robot drives it in every stage, seen by the people, and the split's contest is written next to the suite; prints the `arena evaluation benchmark` call",
    )
    p.add_argument("split", type=Path, help="the split file (see split.py)")
    p.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition to write; repeatable (default: all)")
    p.add_argument("--crowd", default=SP.BASE_CROWD, help=f"the crowd to write, `{SP.BASE_CROWD}` (the split's own settings, default) or one of its `crowds`; each crowd is its own suite")
    p.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over the crowd's settings")
    p.add_argument("--out", type=Path, default=None, help="suite file (default: <imported assets root>/suites/flowedit_<split name>[__<crowd>].yaml, where `--suite` finds it by that name)")
    p.add_argument("--robot", default=SU.ROBOT, help=f"robot of every stage (default {SU.ROBOT}, the robot of the shipped suites), parked off the map and unseen unless the split has a robot_task")
    p.add_argument("--lanes", type=int, default=2, help="lanes in the printed benchmark call (default 2)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
    p.add_argument("--no-generate", action="store_true", help="skip generating the edited worlds again")
    p.add_argument("--force", action="store_true", help="replace a suite file flowedit did not write")
    p.add_argument("--dry-run", action="store_true", help="print the suite without writing scenarios or the suite")
    p.set_defaults(fn=_cmd_suite)

    from .forecast.cli import register

    register(verbs)

    from . import infra_cli

    infra_cli.add_parser(verbs)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point `flowedit`."""
    args = _parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, ValueError, yaml.YAMLError, S.SettingsError) as exc:
        print(f"flowedit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
