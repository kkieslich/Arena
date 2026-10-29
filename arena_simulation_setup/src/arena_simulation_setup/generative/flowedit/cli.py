"""`flowedit`: build the FLOWEDIT graph of a world and scenario."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from arena_simulation_setup.kg import dumps, write

from ..scenario.cli import world_root
from .build import build

FILE = "flowgraph.yaml"


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flowedit", description="FLOWEDIT graphs: each object's demand on its KG node. `flowedit <verb> --help` lists a verb's options.")
    verbs = parser.add_subparsers(dest="verb", required=True, metavar="{build}")
    p = verbs.add_parser("build", help="build the graph of a world and scenario", description="build the graph of a world and scenario, a dry run that prints it without --write")
    p.add_argument("world", help="the world to build from")
    p.add_argument("scenario", nargs="?", help="the scenario under the world's scenarios/ whose objects join the graph (default: world objects only)")
    p.add_argument("--seed", type=int, default=0, help="seed of the point drawn for an object posed at a zone name (default 0)")
    p.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds)")
    p.add_argument("--out", type=Path, default=None, help=f"file to write (default: {FILE} next to the scenario.yaml, or in the world without a scenario)")
    p.add_argument("--write", action="store_true", help="write the graph (default: print it)")
    p.set_defaults(fn=_cmd_build)
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
