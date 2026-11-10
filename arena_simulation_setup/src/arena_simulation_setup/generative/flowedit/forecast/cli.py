"""`flowedit forecast fit|predict|score`: fit the flow forecaster on a split's recorded maps, forecast a world, score it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from .. import settings as S
from .. import split as SP
from .evaluate import Forecaster, Runs, fit, targets
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


def _named(values: list[str] | None, what: str) -> list[tuple[str | None, Path]]:
    """`NAME=PATH` or `PATH` arguments as (name or None, path)."""
    out = []
    for v in values or []:
        name, sep, path = v.partition("=")
        out.append((name, Path(path)) if sep and name and "/" not in name else (None, Path(v)))
        if not out[-1][1].exists():
            raise SystemExit(f"{what} {out[-1][1]}: no such file")
    return out


def _cmd_train(args: argparse.Namespace) -> int:
    from .learn import Rasters, pick, samples, save, train

    doc = SP.load(args.split)
    settings = SP.crowd(doc, args.settings)
    rasters = Rasters.of(doc, settings, args.worlds)
    runs = Runs.load(args.cells, rasters.sites.grid, planner=args.planner)
    worlds = pick(doc, args.partition, args.fraction, args.pick_seed)
    found = samples(runs, worlds, doc, args.partition)
    seeds = sorted({s.seed for s in found})
    _log(f"train: {len(found)} runs of {len(worlds)} worlds at seeds {seeds} ({args.partition}, {'all' if args.fraction >= 1 else format(args.fraction, '.0%')} of its edit chains), demand {'on' if args.demand else 'off (demand ablation)'}")
    demand_model = None if args.demand_model_input is None else load_params(args.demand_model_input)
    if demand_model is not None:
        _log(f"train: the demand model's forecast ({args.demand_model_input}) is an input layer of every world")
    model, summary = train(rasters, found, demand=args.demand, steps=args.steps, batch=args.batch, seed=args.seed, device=args.device, demand_model=demand_model, log=_log)
    meta = {
        "demand_model_input": None if args.demand_model_input is None else str(args.demand_model_input),
        "split": str(args.split),
        "cells": str(args.cells),
        "partition": args.partition,
        "seeds": seeds,
        "fraction": args.fraction,
        "pick_seed": args.pick_seed,
        "worlds": worlds,
        "runs": len(found),
        "demand": args.demand,
        "settings": S.digest(settings),
        "crowd_seeds": list(CROWD_SEEDS),
        **summary,
    }
    save(args.out, model, meta)
    print(yaml.safe_dump({"net": str(args.out), **meta}, sort_keys=False), end="")
    return 0


def _net_methods(nets: list[tuple[str | None, Path]], rasters: Any, device: str | None) -> dict[str, Any]:  # noqa: ANN401
    from .learn import forecast, load

    methods: dict[str, Any] = {}
    for name, path in nets:
        model, meta = load(path, device)
        label = name or ("demand ablation" if not meta.get("demand", True) else "forecaster + demand model" if model.demand_model is not None else "forecaster")
        cache: dict[str, Any] = {}

        def method(world: str, model: Any = model, cache: dict[str, Any] = cache) -> Any:  # noqa: ANN401
            if world not in cache:
                cache[world] = forecast(model, rasters, world)
            return cache[world]

        methods[label] = method
    return methods


def _cmd_score(args: argparse.Namespace) -> int:
    from . import validity as V
    from .learn import Rasters

    doc = SP.load(args.split)
    settings = SP.crowd(doc, args.settings)
    rasters = Rasters.of(doc, settings, args.worlds)
    sites = rasters.sites
    runs = Runs.load(args.cells, sites.grid, planner=args.planner)
    forecasters = _net_methods(_named(args.net, "--net"), rasters, args.device)
    if args.params is not None:
        params = load_params(args.params)
        forecasters["demand model"] = lambda world: sites.predict(world, params, rasters.scenario_of.get(SP.base_of(world)))
    for name, path in _named(args.predicted, "--predicted"):
        forecasters[name or path.stem] = V.Predicted.load(path, sites.grid)
    report: dict[str, Any] = {"format": FORMAT, "cells": str(args.cells), "nets": [str(p) for _, p in _named(args.net, "--net")], "params": str(args.params) if args.params else None, "partitions": {}}
    for partition in args.partition or [p for p in SP.PARTITIONS if p != "train"]:
        seeds = SP.seeds_of(doc, partition)
        found = V.truths(runs, doc, partition, _log)
        methods = {**forecasters, "stale": V.stale(runs, seeds), "transplanted": V.transplant(runs, seeds, sites, rasters.scenario_of)}
        rows = V.score(found, methods, doc, coarse=[n for n in forecasters if n in ("forecaster",)])
        cols = [c for c in rows[0] if c not in ("world", "what", "seeds")] if rows else []
        floors = {base: V.floor(runs, base, V.world_seeds(doc, partition, base)) for base in sorted({SP.base_of(t.world) for t in found})}
        change = V.changes(found, runs, forecasters, doc)
        others = {base: V.other_seeds(runs, base, seeds) for base in floors}
        print(f"## {partition} (scored seeds {seeds}; stale and transplanted from base runs at seeds {others})\n\n{V.table(rows, cols)}\n")
        print("across-seed floor (base world at the scored seeds against its other seeds): " + ", ".join(f"{b} {'n/a (no base run at another seed)' if v is None else f'{v:.3f}'}" for b, v in floors.items()) + "\n")
        if change:
            print(f"### {partition}: change against the base world, recorded (PEDS, same seeds) and predicted\n\n{V.table(change, ['recorded', *forecasters])}\n")
        if args.plots is not None:
            from .plot import plot_world

            main = next(iter(forecasters), None)
            for t in found:
                base = SP.base_of(t.world)
                fc = forecasters[main](t.world) if main else None
                if fc is not None:
                    plot_world(args.plots / f"{t.world}.png", sites.grid(base), t.map, fc, methods["stale"](t.world), _scene(rasters, t.world), title=f"{t.world} ({partition}, seeds {list(t.seeds)}): {main}")
        report["partitions"][partition] = {
            "seeds": seeds,
            "other_seeds": others,
            "rows": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows],
            "floor": {b: None if v is None else round(v, 5) for b, v in floors.items()},
            "change": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in change],
        }
    if args.out:
        args.out.write_text(yaml.safe_dump(report, sort_keys=False))
        _log(f"wrote {args.out}")
    return 0


def _scene(rasters: Any, world: str) -> Any:  # noqa: ANN401
    """The scene of a world, for the plot's demand markers."""
    from .scene import scene_of

    s = rasters.sites
    site = s.site(world, rasters.scenario_of.get(SP.base_of(world)))
    people = next((v for (b, _), v in s._people.items() if b == site.base), [])
    return scene_of(site, s.settings, s.duration, grid=s.grid(site.base), people=people)


def _tracks_of(benchmark: Path, planner: str | None) -> Any:  # noqa: ANN401
    """(world, seed) -> the track table of that stage's first episode in a processed benchmark run dir, None without."""
    import polars as pl

    from ..suite import parse_stage_name

    metrics = pl.read_parquet(benchmark / "combined_metrics.parquet", columns=["episode", "stage", "planner", "robot"])
    if planner is not None:
        metrics = metrics.filter(pl.col("planner") == planner)
    where: dict[tuple[str, int], Path] = {}
    for episode, stage, _, robot in metrics.iter_rows():
        try:
            key = parse_stage_name(str(stage))
        except ValueError:
            continue
        env = str(robot or "env_0").split("_")
        env_name = "_".join(env[:2]) if len(env) >= 2 and env[0] == "env" else "env_0"
        where.setdefault(key, benchmark / "episodes" / f"episode_{int(episode):03d}" / "topics" / ".derived" / "ped_tracks.v1" / f"{env_name}.parquet")

    def tracks(world: str, seed: int) -> Any:  # noqa: ANN401
        path = where.get((world, seed))
        return pl.read_parquet(path) if path is not None and path.exists() else None

    return tracks


def _cmd_curve(args: argparse.Namespace) -> int:
    import numpy as np

    from . import validity as V
    from .learn import Rasters

    doc = SP.load(args.split)
    settings = SP.crowd(doc, args.settings)
    rasters = Rasters.of(doc, settings, args.worlds)
    sites = rasters.sites
    runs = Runs.load(args.benchmark / "flow_cells.parquet", sites.grid, planner=args.planner)
    tracks = _tracks_of(args.benchmark, args.planner)
    nets = _net_methods(_named(args.net, "--net"), rasters, args.device)
    train_ws = [(w, V.world_seeds(doc, "train", w)) for w in doc["partitions"].get("train") or []]
    priors_of: dict[str, Any] = dict(nets)
    priors_of["uniform (online only)"] = lambda world: V.uniform(sites.grid(SP.base_of(world)).room_cells())
    strengths = {}
    for name, prior in priors_of.items():
        strengths[name] = args.strength if args.strength is not None else V.fit_strength(train_ws, tracks, runs, sites.grid, prior)
        _log(f"curve: {name} prior strength {strengths[name]:g} pedestrian-seconds" + (" (fitted on train)" if args.strength is None else ""))
    out: dict[str, Any] = {"format": FORMAT, "benchmark": str(args.benchmark), "counts": list(V.TRACK_COUNTS), "strength": strengths, "partitions": {}}
    for partition in args.partition or ["held_out_edits"]:
        seeds = SP.seeds_of(doc, partition)
        curves: dict[str, list[list[float | None]]] = {n: [] for n in [*priors_of, V.ONE_SEED_FLOOR]}
        for world in doc["partitions"].get(partition) or []:
            if SP.is_base(world):
                continue
            grid = sites.grid(SP.base_of(world))
            res = V.curve(world, V.world_seeds(doc, partition, world), tracks, runs, grid, {n: (p(world), strengths[n]) for n, p in priors_of.items()})
            for n, v in res.items():
                curves[n].append(v)
        mean = {n: [float(np.mean([c[i] for c in v if c[i] is not None])) if any(c[i] is not None for c in v) else None for i in range(len(V.TRACK_COUNTS))] for n, v in curves.items()}
        out["partitions"][partition] = {"seeds": seeds, "mean": mean}
        head = "| tracks seen | " + " | ".join(str(k) for k in V.TRACK_COUNTS) + " |"
        lines = [head, "|---|" + "---|" * len(V.TRACK_COUNTS)] + [f"| {n} | " + " | ".join("" if x is None else f"{x:.3f}" for x in m) + " |" for n, m in mean.items()]
        print(f"## {partition}: JSD against the other seeds' pooled map after k tracks of one seed (mean over edits)\n\n" + "\n".join(lines) + "\n")
        if args.png is not None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 4))
            shown = [k for i, k in enumerate(V.TRACK_COUNTS) if any(m[i] is not None for n, m in mean.items() if n != V.ONE_SEED_FLOOR)]
            for n, m in mean.items():
                pts = [(max(k, 0.5), v) for k, v in zip(V.TRACK_COUNTS, m, strict=True) if v is not None and k in shown]
                style = {"linestyle": "--", "color": "grey"} if n == V.ONE_SEED_FLOOR else {"marker": "o"}
                ax.plot([x for x, _ in pts], [v for _, v in pts], label=n, **style)
            ax.set_xscale("log")
            ax.set_xticks([max(k, 0.5) for k in shown], [str(k) for k in shown])
            ax.minorticks_off()
            ax.set_xlabel("post-edit tracks seen, all from one held-out seed's run")
            ax.set_ylabel("pooled JSD (bits) against the other seeds")
            ax.set_title(f"{partition}: forecast error vs. tracks seen")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            png = args.png if len(args.partition or ["x"]) == 1 else args.png.with_name(f"{args.png.stem}_{partition}{args.png.suffix}")
            fig.savefig(png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            _log(f"wrote {png}")
    if args.out:
        args.out.write_text(yaml.safe_dump(out, sort_keys=False))
        _log(f"wrote {args.out}")
    return 0


def _cmd_scaling(args: argparse.Namespace) -> int:
    import numpy as np

    from . import validity as V
    from .learn import Rasters, forecast, pick, samples, save, train

    doc = SP.load(args.split)
    settings = SP.crowd(doc, args.settings)
    rasters = Rasters.of(doc, settings, args.worlds)
    runs = Runs.load(args.cells, rasters.sites.grid, planner=args.planner)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = args.partition or [p for p in SP.PARTITIONS if p != "train"]
    found = {p: V.truths(runs, doc, p, _log) for p in parts}
    result: dict[str, Any] = {"format": FORMAT, "fractions": list(args.fractions), "repeats": args.repeats, "runs": [], "mean": {}}
    for fraction in args.fractions:
        for r in range(args.repeats):
            worlds = pick(doc, "train", fraction, r)
            model, summary = train(rasters, samples(runs, worlds, doc), steps=args.steps, seed=r, device=args.device, log=_log)
            save(args.out_dir / f"net_f{fraction:g}_r{r}.pt", model, {"fraction": fraction, "pick_seed": r, "worlds": worlds, **summary})
            entry: dict[str, Any] = {"fraction": fraction, "repeat": r, "edits": sum(not SP.is_base(w) for w in worlds)}
            for p in parts:
                vals = [V.jsd(forecast(model, rasters, t.world), t.map) for t in found[p] if not SP.is_base(t.world)]
                entry[p] = float(np.mean(vals)) if vals else None
            result["runs"].append(entry)
            _log(f"scaling: fraction {fraction:g} repeat {r}: " + ", ".join(f"{p} {entry[p]:.3f}" for p in parts if entry[p] is not None))
    for p in parts:
        result["mean"][p] = {f"{f:g}": float(np.mean([e[p] for e in result["runs"] if e["fraction"] == f and e[p] is not None])) for f in args.fractions}
    lines = ["| training edits | " + " | ".join(parts) + " |", "|---|" + "---|" * len(parts)]
    for f in args.fractions:
        lines.append(f"| {f:g} | " + " | ".join(f"{result['mean'][p][f'{f:g}']:.3f}" for p in parts) + " |")
    print("## data scaling: mean pooled JSD over the held-out edits per share of the training edit chains (mean of repeats)\n\n" + "\n".join(lines) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for p in parts:
        xs = [e["fraction"] for e in result["runs"] if e[p] is not None]
        ax.scatter(xs, [e[p] for e in result["runs"] if e[p] is not None], alpha=0.4)
        ax.plot(list(args.fractions), [result["mean"][p][f"{f:g}"] for f in args.fractions], marker="o", label=p)
    ax.set_xlabel("share of the training edit chains")
    ax.set_ylabel("mean pooled JSD (bits)")
    ax.set_title("learned forecaster: data scaling")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(args.out_dir / "scaling.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    (args.out_dir / "scaling.yaml").write_text(yaml.safe_dump(result, sort_keys=False))
    _log(f"wrote {args.out_dir / 'scaling.yaml'} and scaling.png")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    settings: dict[str, Any] = SP.crowd(SP.load(args.split), args.settings) if args.split else S.load(args.settings)
    duration = args.duration if args.duration is not None else (SP.duration_of(SP.load(args.split)) if args.split else 300.0)
    forecaster = Forecaster(settings, duration, worlds=args.worlds)
    if args.net is not None:
        from .learn import Rasters, load
        from .learn import forecast as forecast_net

        rasters = Rasters(forecaster, {args.world: args.scenario})
        model, _ = load(args.net, args.device)
        forecast = forecast_net(model, rasters, args.world)
        scene = _scene(rasters, args.world)
        what = f"learned forecaster {args.net.name}"
    else:
        layers = forecaster.layers(args.world, args.scenario)
        forecast = predict(layers, load_params(args.params))
        scene = layers.scene
        what = "demand model"
    grid = forecaster.grid(SP.base_of(args.world) if not SP.is_base(args.world) else args.world)
    print(f"{args.world} ({what}): {sum(p.weight for p in scene.people):.1f} people (expected over {len(CROWD_SEEDS)} crowd draws), {duration:g} s, {forecast.sum():.0f} pedestrian-seconds on {grid.nx} x {grid.ny} cells")
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

        plot_world(args.png, grid, None, forecast, None, scene, title=f"{args.world} forecast ({what})")
        print(f"wrote {args.png}")
    return 0


def register(verbs: argparse._SubParsersAction) -> None:
    """Add `forecast` and its verbs to the `flowedit` parser."""
    p = verbs.add_parser("forecast", help="train, run and score the flow forecasters", description="flow forecasters: a world's flow map from its graph and geometry, no PEDS run (see Forecasting flow in the flowedit README)")
    sub = p.add_subparsers(dest="forecast_verb", required=True, metavar="{train,score,curve,scaling,fit,predict}")

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over the split's (see settings.py)")
        q.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots, then $ARENA_DIR/arena_simulation_setup/worlds, then the resolvers)")
        q.add_argument("--planner", default=None, help="the contestant whose maps to read, when the benchmark holds several")

    def recorded(q: argparse.ArgumentParser) -> None:
        q.add_argument("split", type=Path, help="the split file (see split.py)")
        q.add_argument("--cells", type=Path, required=True, help="arena_evaluation's flow_cells.parquet of the split's benchmark")
        common(q)

    def device(q: argparse.ArgumentParser) -> None:
        q.add_argument("--device", default=None, help="torch device (default: cuda when available, else cpu)")

    q = sub.add_parser("train", help="train the learned forecaster on a partition's single runs", description="train the learned forecaster (GNN over the graph, U-Net on the flow grid) on every single run of the partition, Poisson NLL per cell and bin, D4 augmentation")
    recorded(q)
    device(q)
    q.add_argument("--out", type=Path, required=True, help="the network file to write (.pt)")
    q.add_argument("--partition", choices=SP.PARTITIONS, default="train", help="the partition to train on (default train)")
    q.add_argument("--no-demand", dest="demand", action="store_false", help="leave out the need channels and the demand and need nodes: the demand ablation")
    q.add_argument("--demand-model-input", type=Path, default=None, metavar="PARAMS", help="a params file from `fit`: give the network the demand model's forecast of each world as extra input layers (a variant to compare against the plain forecaster)")
    q.add_argument("--fraction", type=float, default=1.0, help="share of the partition's edit chains to train on, the base world always kept (default 1)")
    q.add_argument("--pick-seed", type=int, default=0, help="seed of the chains `--fraction` keeps (default 0)")
    q.add_argument("--seed", type=int, default=0, help="training seed (default 0)")
    q.add_argument("--steps", type=int, default=3000, help="optimizer steps (default 3000)")
    q.add_argument("--batch", type=int, default=8, help="runs per step (default 8)")
    q.set_defaults(fn=_cmd_train)

    q = sub.add_parser("score", help="score forecasters and the fair baselines on partitions", description="per world: the pooled JSD of each forecaster and baseline against the recorded pooled map, the across-seed floor, and each edit's recorded and predicted change against its base world")
    recorded(q)
    device(q)
    q.add_argument("--net", action="append", default=None, metavar="[NAME=]NET", help="a network from `train`; repeatable (default name: forecaster, or demand ablation for a --no-demand net)")
    q.add_argument("--params", type=Path, default=None, help="a params file from `fit`: adds the demand model")
    q.add_argument("--predicted", action="append", default=None, metavar="[NAME=]PARQUET", help="any predicted map on the flow grid (columns flow_world, x, y, heading_rad, ped_seconds; see the README); repeatable")
    q.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition to score; repeatable (default: held_out_edits and held_out_worlds)")
    q.add_argument("--out", type=Path, default=None, help="YAML file of every row")
    q.add_argument("--plots", type=Path, default=None, help="directory for a PNG per world: recorded, the first forecaster, stale and their difference")
    q.set_defaults(fn=_cmd_score)

    q = sub.add_parser("curve", help="forecast error against post-edit tracks seen", description="the forecast as a Dirichlet prior updated with the tracks of one held-out seed's run, scored against the pooled map of the other seeds, beside the same update from a uniform prior")
    q.add_argument("split", type=Path, help="the split file (see split.py)")
    q.add_argument("--benchmark", type=Path, required=True, help="the split's processed benchmark run dir (flow_cells.parquet, combined_metrics.parquet and the track store)")
    common(q)
    device(q)
    q.add_argument("--net", action="append", required=True, metavar="[NAME=]NET", help="a network from `train`; repeatable")
    q.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition; repeatable (default held_out_edits)")
    q.add_argument("--strength", type=float, default=None, help="the prior's weight in pedestrian-seconds, the same for every prior (default: fitted per prior on the train partition, where a learned forecaster is in-sample, so its fitted weight leans high)")
    q.add_argument("--out", type=Path, default=None, help="YAML file of the curves")
    q.add_argument("--png", type=Path, default=None, help="PNG of the curves")
    q.set_defaults(fn=_cmd_curve)

    q = sub.add_parser("scaling", help="the learned forecaster trained on shares of the training edits", description="train on each share of the train partition's edit chains (repeats draw other chains and seeds) and score the held-out partitions")
    recorded(q)
    device(q)
    q.add_argument("--out-dir", type=Path, required=True, help="directory for the networks, scaling.yaml and scaling.png")
    q.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0], help="shares of the edit chains (default 0.25 0.5 1)")
    q.add_argument("--repeats", type=int, default=3, help="draws per share (default 3)")
    q.add_argument("--steps", type=int, default=3000, help="optimizer steps per network (default 3000)")
    q.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition to score; repeatable (default: held_out_edits and held_out_worlds)")
    q.set_defaults(fn=_cmd_scaling)

    q = sub.add_parser("fit", help="fit the demand model's params on a partition's recorded maps", description="fit the demand model's params by minimising the mean seed-pooled JSD over the partition's worlds, the base world included")
    recorded(q)
    q.add_argument("--crowd", choices=CROWDS, default="expected", help="people of the forecast: `expected` (default) draws them at crowd seeds no run uses; `runs` at the seeds of the runs fitted, a diagnostic only")
    q.add_argument("--partition", choices=SP.PARTITIONS, default="train", help="the partition to fit on (default train)")
    q.add_argument("--init", type=Path, default=None, help="a params file to start from (default: the built-in defaults)")
    q.add_argument("--max-evals", type=int, default=400, help="forecasts per world the fit may run (default 400)")
    q.add_argument("--out", type=Path, default=None, help="params file to write (also printed)")
    q.set_defaults(fn=_cmd_fit)

    q = sub.add_parser("predict", help="forecast one world", description="forecast a world's flow map from its graph and geometry: the learned forecaster with --net, else the demand model")
    q.add_argument("world", help="the world, shipped or saved by an edit")
    q.add_argument("--scenario", default=None, help="the scenario of a base world whose objects join its graph (an edited world names its own)")
    q.add_argument("--net", type=Path, default=None, help="a network from `train` (default: the demand model)")
    q.add_argument("--params", type=Path, default=None, help="the demand model's params file from `fit` (default: the built-in defaults)")
    q.add_argument("--split", type=Path, default=None, help="a split whose crowd settings and duration to use")
    q.add_argument("--duration", type=float, default=None, help="seconds of the run forecast (default: the split's, else 300)")
    q.add_argument("--out", type=Path, default=None, help="parquet file of the forecast's cells (the format `score --predicted` reads)")
    q.add_argument("--png", type=Path, default=None, help="PNG of the forecast")
    common(q)
    device(q)
    q.set_defaults(fn=_cmd_predict)


__all__ = ["FORMAT", "load_params", "register"]
