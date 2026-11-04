"""`flowedit`: build the FLOWEDIT graph of a world and scenario, edit it, and generate edits from a seed."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from arena_simulation_setup.kg import dumps, write

from ..scenario.cli import world_root
from .build import build
from .edit import GRAPH, EditType, apply, dump_edits, open_site, save
from .generate import controls, dump_manifest, generate

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
        print(yaml.safe_dump({"world": site.name, **dump_edits(site)}, sort_keys=False, default_flow_style=None, width=1 << 16), end="")
        return 0
    print(f"wrote {save(site)}")
    return 0


def _types(text: str) -> list[EditType]:
    """Comma-separated edit types."""
    try:
        return [EditType(t.strip()) for t in text.split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"edit types are {', '.join(EditType)}, got {text!r}") from exc


def _dump(doc: object) -> str:
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=1 << 16)


def _cmd_generate(args: argparse.Namespace) -> int:
    root = world_root(args.worlds, args.world)
    notes: list[str] = []
    site = open_site(root, args.scenario, None, notes)
    chains = generate(site, args.seed, args.per_type, args.steps, args.types, notes)
    made = {chain.site.name: controls(site, chain) for chain in chains} if args.controls else {}
    for note in notes:
        print(f"note: {note}", file=sys.stderr)
    if not args.write:
        print(_dump(dump_manifest(site, chains, args.seed, args.per_type, args.steps, chain_controls=made)), end="")
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
    out.write_text(_dump(dump_manifest(site, chains, args.seed, args.per_type, args.steps, worlds, made)))
    print(f"wrote {out}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowedit", description="FLOWEDIT graphs: each object's demand on its KG node. `flowedit <verb> --help` lists a verb's options.")
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{build,edit,generate}")
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
    p = verbs.add_parser("generate", help="sample edits of every type from a seed and save the edited worlds", description="sample per edit type --per-type chains of --steps edits on a world (and scenario) from --seed, a dry run that prints the manifest without --write, which saves every edited world under $ARENA_ASSETS_DIR_LOCAL/worlds and the manifest beside them")
    p.add_argument("world", help="the world to edit, shipped or saved by an earlier edit")
    p.add_argument("--scenario", default=None, help="the scenario whose objects join the graph and travel with each edited world (default: world objects only)")
    p.add_argument("--seed", type=int, default=0, help="seed of the edits (default 0)")
    p.add_argument("--per-type", type=int, default=1, help="chains per edit type (default 1)")
    p.add_argument("--steps", type=int, default=1, help="edits per chain, the first of the chain's type (default 1)")
    p.add_argument("--types", type=_types, default=list(EditType), metavar="TYPE[,TYPE...]", help=f"edit types to start chains with and draw later steps from (default: {','.join(EditType)})")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
    p.add_argument("--controls", action="store_true", help="also make the geometry-only controls of every function-carrying move or insert: the model without demand at the new spot, and for a move the object moved without demand")
    p.add_argument("--manifest", type=Path, default=None, help=f"manifest file to write (default: $ARENA_ASSETS_DIR_LOCAL/{MANIFESTS}/<world>__seed<seed>.yaml)")
    p.add_argument("--write", action="store_true", help="save the edited worlds and the manifest (default: print the manifest)")
    p.set_defaults(fn=_cmd_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point `flowedit`."""
    args = _parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"flowedit: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
