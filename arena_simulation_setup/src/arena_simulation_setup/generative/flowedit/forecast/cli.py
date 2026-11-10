"""`flowedit forecast fit|predict|score`: fit the flow forecaster on a split's recorded maps, forecast a world, score it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .. import settings as S
from .. import split as SP
from .evaluate import Forecaster, Runs, fit, score, seed_floor, table, targets
from .maps import jsd
from .model import Params, predict
from .scene import CROWD_SEEDS

FORMAT = "flowedit-forecast/1"
CROWDS = ("expected", "runs")


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def load_params(path: Path | None) -> Params:
    """The params of a `fit` output file, the defaults without one."""
    if path is None:
        return Params()
    doc = yaml.safe_load(path.read_text()) or {}
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: format: expected {FORMAT!r}, got {doc.get('format')!r}")
    return Params.load(doc.get("params") or {})


def _forecaster(doc: dict[str, Any], args: argparse.Namespace) -> Forecaster:
    return Forecaster(SP.crowd(doc, args.settings), SP.duration_of(doc), worlds=args.worlds)


def _runs(args: argparse.Namespace, forecaster: Forecaster) -> Runs:
    return Runs.load(args.cells, forecaster.grid, planner=args.planner)


def _cmd_fit(args: argparse.Namespace) -> int:
    doc = SP.load(args.split)
    forecaster = _forecaster(doc, args)
    runs = _runs(args, forecaster)
    found = targets(forecaster, runs, doc, args.partition, args.crowd, _log)
    init = load_params(args.init)
    params, value = fit(found, init, max_evals=args.max_evals, log=_log)
    per_world = {t.world: round(float(jsd(predict(t.layers, params), t.truth)), 5) for t in found}
    out = {
        "format": FORMAT,
        "params": params.dump(),
        "fit": {
            "split": str(args.split),
            "partition": args.partition,
            "seeds": SP.seeds_of(doc, args.partition),
            "crowd": args.crowd if args.crowd == "runs" else {"expected": list(CROWD_SEEDS)},
            "settings": S.digest(forecaster.settings),
            "duration": forecaster.duration,
            "cells": str(args.cells),
            "worlds": len(found),
            "mean_jsd": round(value, 5),
            "init_mean_jsd": round(float(sum(jsd(predict(t.layers, init), t.truth) for t in found) / len(found)), 5),
            "per_world": per_world,
        },
    }
    text = yaml.safe_dump(out, sort_keys=False)
    if args.out:
        args.out.write_text(text)
        _log(f"wrote {args.out}")
    print(text, end="")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    doc = SP.load(args.split)
    forecaster = _forecaster(doc, args)
    runs = _runs(args, forecaster)
    params = load_params(args.params)
    report: dict[str, Any] = {"format": FORMAT, "params": params.dump(), "crowd": args.crowd, "cells": str(args.cells), "partitions": {}}
    for partition in args.partition or [p for p in SP.PARTITIONS if p != "train"]:
        rows = score(forecaster, runs, doc, partition, params, args.crowd, args.plots, _log)
        seeds = SP.seeds_of(doc, partition)
        floors = {}
        for base in sorted({SP.base_of(r["world"]) for r in rows}):
            other = [s for p in SP.PARTITIONS for s in SP.seeds_of(doc, p) if s not in seeds]
            if (value := seed_floor(runs, base, seeds, sorted(set(other)))) is not None:
                floors[base] = round(value, 5)
        report["partitions"][partition] = {"seeds": seeds, "rows": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows], "seed_floor": floors}
        print(f"## {partition} (seeds {seeds})\n\n{table(rows)}\n")
        if floors:
            print("seed floor (base world, these seeds against the split's other seeds): " + ", ".join(f"{b} {v:.3f}" for b, v in floors.items()) + "\n")
    if args.out:
        args.out.write_text(yaml.safe_dump(report, sort_keys=False))
        _log(f"wrote {args.out}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    settings: dict[str, Any] = SP.crowd(SP.load(args.split), args.settings) if args.split else S.load(args.settings)
    duration = args.duration if args.duration is not None else (SP.duration_of(SP.load(args.split)) if args.split else 300.0)
    forecaster = Forecaster(settings, duration, worlds=args.worlds)
    params = load_params(args.params)
    layers = forecaster.layers(args.world, args.scenario)
    forecast = predict(layers, params)
    scene = layers.scene
    grid = forecaster.grid(SP.base_of(args.world) if not SP.is_base(args.world) else args.world)
    print(f"{args.world}: {sum(p.weight for p in scene.people):.1f} people (expected over {len(CROWD_SEEDS)} crowd draws), {duration:g} s, {forecast.sum():.0f} pedestrian-seconds on {grid.nx} x {grid.ny} cells")
    for d in scene.demands:
        print(f"  {d.need}: obj:{d.name} at ({d.at[0]:.2f}, {d.at[1]:.2f}), used from ({d.spot(0)[0]:.2f}, {d.spot(0)[1]:.2f}), dwell {d.dwell:g} s, capacity {d.capacity}")
    for note in scene.notes:
        print(f"  note: {note}")
    if args.out:
        import polars as pl

        rows = grid.rows(forecast)
        pl.DataFrame({"flow_world": [args.world] * len(rows["x"]), **rows}).write_parquet(args.out)
        print(f"wrote {args.out}")
    if args.png:
        from .plot import plot_world

        plot_world(args.png, grid, None, forecast, None, scene, title=f"{args.world} forecast")
        print(f"wrote {args.png}")
    return 0


def register(verbs: argparse._SubParsersAction) -> None:
    """Add `forecast` and its verbs to the `flowedit` parser."""
    p = verbs.add_parser("forecast", help="fit, run and score the flow forecaster", description="the demand-driven flow forecaster: a world's flow map from its graph and geometry, no PEDS run (see forecast/README in the flowedit README)")
    sub = p.add_subparsers(dest="forecast_verb", required=True, metavar="{fit,predict,score,baseline}")

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over the split's (see settings.py)")
        q.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")

    def recorded(q: argparse.ArgumentParser) -> None:
        q.add_argument("split", type=Path, help="the split file (see split.py)")
        q.add_argument("--cells", type=Path, required=True, help="arena_evaluation's flow_cells.parquet of the split's benchmark")
        q.add_argument("--planner", default=None, help="the contestant whose maps to read, when the file holds several")
        q.add_argument("--crowd", choices=CROWDS, default="expected", help="people of the forecast: `expected` (default) draws them at crowd seeds no run uses, `runs` at the seeds of the runs scored")
        common(q)

    q = sub.add_parser("fit", help="fit the params on a partition's recorded maps", description="fit the forecaster's params by minimising the mean seed-pooled JSD over the partition's worlds, the base world included")
    recorded(q)
    q.add_argument("--partition", choices=SP.PARTITIONS, default="train", help="the partition to fit on (default train)")
    q.add_argument("--init", type=Path, default=None, help="a params file to start from (default: the built-in defaults)")
    q.add_argument("--max-evals", type=int, default=400, help="forecasts per world the fit may run (default 400)")
    q.add_argument("--out", type=Path, default=None, help="params file to write (also printed)")
    q.set_defaults(fn=_cmd_fit)

    q = sub.add_parser("score", help="score the forecast and the baselines on partitions", description="per world: the pooled JSD of the forecast, the anchored forecast, the stale and the transplanted base map against the recorded pooled map; prints a table per partition")
    recorded(q)
    q.add_argument("--params", type=Path, default=None, help="params file from `fit` (default: the built-in defaults)")
    q.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition to score; repeatable (default: held_out_edits and held_out_worlds)")
    q.add_argument("--out", type=Path, default=None, help="YAML file of every row")
    q.add_argument("--plots", type=Path, default=None, help="directory for a PNG per world: recorded, forecast, stale and their difference")
    q.set_defaults(fn=_cmd_score)

    q = sub.add_parser("predict", help="forecast one world", description="forecast a world's flow map from its graph and geometry, with the expected crowd of its base world")
    q.add_argument("world", help="the world, shipped or saved by an edit")
    q.add_argument("--scenario", default=None, help="the scenario of a base world whose objects join its graph (an edited world names its own)")
    q.add_argument("--params", type=Path, default=None, help="params file from `fit` (default: the built-in defaults)")
    q.add_argument("--split", type=Path, default=None, help="a split whose crowd settings and duration to use")
    q.add_argument("--duration", type=float, default=None, help="seconds of the run forecast (default: the split's, else 300)")
    q.add_argument("--out", type=Path, default=None, help="parquet file of the forecast's cells (arena_evaluation's CELL_SCHEMA columns and flow_world)")
    q.add_argument("--png", type=Path, default=None, help="PNG of the forecast")
    common(q)
    q.set_defaults(fn=_cmd_predict)

    from .baseline_cli import register as register_baselines

    register_baselines(sub)


__all__ = ["FORMAT", "load_params", "register"]
