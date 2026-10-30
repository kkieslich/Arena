"""`flowedit`: build the FLOWEDIT graph of a world and scenario, and edit it."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from arena_simulation_setup.kg import dumps, write

from ..scenario.cli import world_root
from .build import build
from .edit import GRAPH, OPERATIONS, apply, dump_edits, open_site, save

FILE = GRAPH


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
    out = args.out or (scenario.parent if scenario is not None else root) / FILE
    write(scene, out)
    print(f"wrote {out}")
    return 0


def _param(text: str) -> tuple[list[str], object]:
    """`<key>[.<key>...]=<YAML value>`, the dotted key split."""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected key=value, got {text!r}")
    return key.split("."), yaml.safe_load(value)


def _params(pairs: list[tuple[list[str], object]]) -> dict[str, object]:
    params: dict[str, object] = {}
    for keys, value in pairs:
        node = params
        for key in keys[:-1]:
            node = node.setdefault(key, {})  # type: ignore[assignment]
            if not isinstance(node, dict):
                raise ValueError(f"param {'.'.join(keys)}: {key!r} is already a value")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowedit", description="FLOWEDIT graphs: each object's demand on its KG node. `flowedit <verb> --help` lists a verb's options.")
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{build,edit}")
    p = verbs.add_parser("build", help="build the graph of a world and scenario", description="build the graph of a world and scenario, a dry run that prints it without --write")
    p.add_argument("world", help="the world to build from")
    p.add_argument("scenario", nargs="?", help="the scenario under the world's scenarios/ whose objects join the graph (default: world objects only)")
    p.add_argument("--seed", type=int, default=0, help="seed of the point drawn for an object posed at a zone name (default 0)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds)")
    p.add_argument("--out", type=Path, default=None, help=f"file to write (default: {FILE} next to the scenario.yaml, or in the world without a scenario)")
    p.add_argument("--write", action="store_true", help="write the graph (default: print it)")
    p.set_defaults(fn=_cmd_build)
    p = verbs.add_parser("edit", help="apply one edit and save the edited world", description="apply one edit to a world's graph and geometry, a dry run that prints the edit record without --write, which saves the world as <base>__<edit id> under $ARENA_ASSETS_DIR_LOCAL/worlds")
    p.add_argument("world", help="the world to edit, shipped or saved by an earlier edit")
    p.add_argument("type", choices=list(OPERATIONS), help="the edit type")
    p.add_argument("target", help="what it changes: obj:<name>, door:<name>, schedule:<name>, wall:<level>[/<index>]")
    p.add_argument("-p", "--param", type=_param, action="append", default=[], metavar="KEY=VALUE", help="an edit param, VALUE read as YAML, KEY dotted for nesting (satisfies.thirst=80); repeatable")
    p.add_argument("--scenario", default=None, help="the scenario whose objects join the graph of an unedited world (default: world objects only)")
    p.add_argument("--seed", type=int, default=0, help="seed of the point drawn for an object posed at a zone name (default 0)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
    p.add_argument("--write", action="store_true", help="save the edited world (default: print the edit)")
    p.set_defaults(fn=_cmd_edit)
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
