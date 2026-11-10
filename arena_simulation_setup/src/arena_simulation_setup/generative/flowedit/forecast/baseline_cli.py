"""`flowedit forecast baseline cliff|controls|geometry|deepflow`: the forecaster's map baselines on a split's recorded runs."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .. import split as SP
from . import baselines as B
from .evaluate import Forecaster, Runs
from .validity import floor as across_seed_floor

FORMAT = "flowedit-forecast-baseline/1"
PRIORS = ("empty", "pre-edit")
#: categorical series colours, in fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _fmt(v: Any) -> str:  # noqa: ANN401
    return "" if v is None else f"{v:.3f}"


def _round(v: Any) -> Any:  # noqa: ANN401
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, list):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v


def _forecaster(doc: Mapping[str, Any], args: argparse.Namespace) -> Forecaster:
    return Forecaster(SP.crowd(doc, args.settings), SP.duration_of(doc), worlds=args.worlds)


def _with_bases(doc: Mapping[str, Any], partitions: Sequence[str]) -> list[str]:
    """The worlds of `partitions` with each one's base world before it, so a map file also holds the bases (for the predicted change)."""
    return list(dict.fromkeys(w for p in partitions for v in doc["partitions"].get(p) or [] for w in (SP.base_of(v), v)))


def _edits(doc: Mapping[str, Any], partition: str) -> list[str]:
    return [w for w in doc["partitions"].get(partition) or [] if not SP.is_base(w)]


def _write(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_round(dict(report)), sort_keys=False))
        _log(f"wrote {path}")


# --- cliff -----------------------------------------------------------------------------------------------------------


def _cmd_cliff(args: argparse.Namespace) -> int:
    from .cliff import CliffParams

    doc = SP.load(args.split)
    sites = _forecaster(doc, args)
    runs = Runs.load(args.benchmark / "flow_cells.parquet", sites.grid, planner=args.planner)
    tracks = B.Tracks.load(args.benchmark, args.planner)
    params = CliffParams.load(yaml.safe_load(args.params.read_text()) or {}) if args.params else CliffParams()
    priors = args.prior or list(PRIORS)
    counts = tuple(args.counts or B.TRACK_COUNTS)

    def make(prior: str, decay: float) -> B.OnlineCliff:
        import attrs

        return B.OnlineCliff(sites.grid, tracks, runs, attrs.evolve(params, decay=decay), prior)

    given: dict[str, float] = {}
    for item in args.decay or []:
        name, _, value = item.rpartition("=")
        for prior in [name] if name else priors:
            if prior not in PRIORS:
                raise SystemExit(f"--decay {item}: the prior is one of {', '.join(PRIORS)}")
            given[prior] = float(value)
    decays: dict[str, float] = {}
    fits: dict[str, Any] = {}
    for prior in priors:
        if prior in given:
            decays[prior] = given[prior]
            continue
        train = [(w, SP.seeds_of(doc, "train")) for w in _edits(doc, "train")]
        if args.fit_worlds:
            train = train[: args.fit_worlds]
        best, scores = B.fit_decay(train, tracks, runs, lambda lam, prior=prior: make(prior, lam), log=_log)
        decays[prior], fits[prior] = best, scores
        _log(f"cliff ({prior}): decay {best:g}, fitted on {len(train)} training edits")
    methods = {f"cliff ({p})": make(p, decays[p]) for p in priors}
    report: dict[str, Any] = {"format": FORMAT, "baseline": "cliff", "benchmark": str(args.benchmark), "params": params.dump(), "decay": decays, "decay_fit": fits, "counts": list(counts), "orders": args.orders, "partitions": {}}
    for partition in args.partition or ["held_out_edits"]:
        seeds = SP.seeds_of(doc, partition)
        rows = []
        for world in _edits(doc, partition):
            if not B.folds(runs, world, seeds):
                _log(f"{world}: fewer than two recorded seeds of {seeds}, left out")
                continue
            res = B.curve(world, seeds, tracks, runs, methods, counts, args.orders)
            seen = [t for f in B.folds(runs, world, seeds) if (t := tracks(world, f.seen)) is not None]
            row = {"world": world, "seeds": runs.seeds(world, seeds), "tracks_per_run": min((t["key"].n_unique() for t in seen), default=0), **res}
            row["one_seed_floor"] = B.one_seed_floor(runs, world, seeds)
            row["stale_at_seen"] = B.stale_at_seen(runs, world, seeds)
            row["stale_outside"] = B.stale_outside(runs, world, seeds)
            rows.append(row)
            _log(f"{world}: " + ", ".join(f"{n} {_fmt(v[-1])} at {counts[-1]}" for n, v in res.items()) + f"; one-seed floor {_fmt(row['one_seed_floor'])}")
        means = {n: [B.mean(r[n][i] for r in rows) for i in range(len(counts))] for n in methods}
        floors = {k: B.mean(r[k] for r in rows) for k in ("one_seed_floor", "stale_at_seen", "stale_outside")}
        report["partitions"][partition] = {"seeds": seeds, "rows": rows, "mean": means, **floors}
        head = "| tracks seen | " + " | ".join(str(k) for k in counts) + " |"
        lines = [head, "|---|" + "---|" * len(counts)] + [f"| {n} | " + " | ".join(_fmt(x) for x in m) + " |" for n, m in means.items()]
        print(f"## {partition}: JSD against the other seeds' pooled map after k tracks of one seed (mean over {len(rows)} edits)\n\n" + "\n".join(lines))
        print(f"\none-seed floor {_fmt(floors['one_seed_floor'])}, stale@s {_fmt(floors['stale_at_seen'])}, stale (outside) {_fmt(floors['stale_outside'])}; tracks per run: {sorted({r['tracks_per_run'] for r in rows})}\n")
        if args.png is not None:
            png = args.png if len(args.partition or ["x"]) == 1 else args.png.with_name(f"{args.png.stem}_{partition}{args.png.suffix}")
            _plot_curve(png, partition, counts, means, floors, max((r["tracks_per_run"] for r in rows), default=0))
    _write(args.out, report)
    return 0


def _plot_curve(png: Path, partition: str, counts: Sequence[int], means: Mapping[str, Sequence[float | None]], floors: Mapping[str, float | None], max_tracks: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # counts beyond a run's tracks repeat its last point: plot each at the tracks actually seen
    shown = [i for i, k in enumerate(counts) if k <= max(max_tracks, 1) or (i > 0 and counts[i - 1] < max_tracks)]
    for k, (n, m) in enumerate(means.items()):
        pts = [(max(min(counts[i], max_tracks or counts[i]), 0.5), m[i]) for i in shown if m[i] is not None]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", markersize=6, linewidth=1.6, color=SERIES[k % len(SERIES)], label=n)
    if floors.get("stale_outside") is not None:
        ax.axhline(floors["stale_outside"], color="#eb6834", linewidth=1.2, linestyle=":", label="stale (base runs outside the truth: the pre-edit map's runs)")
    if floors.get("stale_at_seen") is not None:
        ax.axhline(floors["stale_at_seen"], color="#6b6a64", linewidth=1.2, linestyle=":", label="stale@s (base run at the seen seed)")
    if floors.get("one_seed_floor") is not None:
        ax.axhline(floors["one_seed_floor"], color="#2b2a27", linewidth=1.2, linestyle="--", label="one-seed floor (the edit's own run)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlabel(f"post-edit tracks seen (0 plotted at 0.5; {max_tracks} tracks per run)")
    ax.set_ylabel("pooled JSD (bits) against the other seeds")
    ax.set_title(f"{partition}: online flow maps vs. tracks seen")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {png}")


# --- controls --------------------------------------------------------------------------------------------------------


def _cmd_controls(args: argparse.Namespace) -> int:
    doc = SP.load(args.split)
    sites = _forecaster(doc, args)
    runs = Runs.load(args.cells, sites.grid, planner=args.planner)
    report: dict[str, Any] = {"format": FORMAT, "baseline": "controls", "cells": str(args.cells), "partitions": {}}
    cols = ("geometry_simulation", "stale_at_seen", "one_seed_floor")
    for partition in args.partition or [p for p in SP.PARTITIONS if p != "train"]:
        rows = B.controls(runs, doc, partition)
        report["partitions"][partition] = {"seeds": SP.seeds_of(doc, partition), "rows": rows}
        lines = ["| Edit | Control | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        lines += [f"| {r['world']} | {r['control']} | " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
        if rows:
            lines.append("| mean | | " + " | ".join(_fmt(B.mean(r[c] for r in rows)) for c in cols) + " |")
        print(f"## {partition}: each control's run at seed s as the edit's map, against the edit's other seeds (leave one seed out)\n\n" + "\n".join(lines) + "\n")
    _write(args.out, report)
    return 0


# --- geometry --------------------------------------------------------------------------------------------------------


def _cmd_geometry(args: argparse.Namespace) -> int:
    doc = SP.load(args.split)
    sites = _forecaster(doc, args)
    scenario_of = {b: src.get("scenario") for b, src in (doc.get("sources") or {}).items()}
    maps: dict[str, Any] = {}
    info: dict[str, Any] = {}
    parts = args.partition or [p for p in SP.PARTITIONS if p != "train"]
    for world in _with_bases(doc, parts):
        base = SP.base_of(world)
        site = sites.site(world, scenario_of.get(base))
        m, what = B.geometry_prior(site, sites.grid(base), sites.settings, sites.duration)
        maps[world], info[world] = m, what
        _log(f"{world}: {what['walks']} walks between {len(what['entrances'])} entrances ({what['kind']})" + (f", {len(what['unreachable'])} pairs unreachable" if what["unreachable"] else ""))
        if args.plots is not None:
            from .plot import plot_world
            from .scene import scene_of

            scene = scene_of(site, sites.settings, sites.duration, grid=sites.grid(base), people=[])
            plot_world(args.plots / f"{world}.png", sites.grid(base), None, m, None, scene, title=f"{world}: geometry-only prior ({what['kind']})")
    B.write_maps(args.out, maps, sites.grid)
    _log(f"wrote {args.out}")
    report: dict[str, Any] = {"format": FORMAT, "baseline": "geometry", "maps": str(args.out), "worlds": info}
    if args.cells is not None:
        report["partitions"] = _score_maps(doc, Runs.load(args.cells, sites.grid, planner=args.planner), maps, parts, "geometry_prior")
    _write(args.report, report)
    return 0


def _score_maps(doc: Mapping[str, Any], runs: Runs, maps: Mapping[str, Any], partitions: Sequence[str], name: str) -> dict[str, Any]:
    """Each map's pooled JSD against the partition's recorded worlds, with the across-seed floor of each base world; and,
    leave one seed out, the same map against the other seeds beside the one-seed floor."""
    from .maps import jsd, occupancy

    out = {}
    for partition in partitions:
        seeds = SP.seeds_of(doc, partition)
        rows = []
        for world in doc["partitions"].get(partition) or []:
            have = runs.seeds(world, seeds)
            if not have or world not in maps:
                continue
            truth = runs.pooled(world, have)
            m = maps[world]
            rows.append(
                {
                    "world": world,
                    "seeds": have,
                    name: jsd(m, truth),
                    f"{name} occupancy": jsd(occupancy(m), occupancy(truth)),
                    f"{name} loso": B.loso(runs, world, seeds, lambda f, m=m: m),
                    "one_seed_floor": B.one_seed_floor(runs, world, seeds),
                }
            )
        floors = {}
        for base in sorted({SP.base_of(r["world"]) for r in rows}):
            other = B.outside(runs, base, seeds)
            # the floor of `forecast score`: equal-size pools of the base world, scored seeds against other seeds
            if other and (v := across_seed_floor(runs, base, seeds)) is not None:
                floors[base] = v.mean
        cols = (name, f"{name} occupancy", f"{name} loso", "one_seed_floor")
        lines = ["| World | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)] + [f"| {r['world']} | " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in rows]
        edits = [r for r in rows if not SP.is_base(r["world"])]
        if edits:
            lines.append("| mean over edits | " + " | ".join(_fmt(B.mean(r[c] for r in edits)) for c in cols) + " |")
        print(f"## {partition} (seeds {seeds})\n\n" + "\n".join(lines))
        print("\nacross-seed floor (base world at these seeds against as many of its other seeds, each side pooled; a scale, not a bound): " + (", ".join(f"{b} {v:.3f}" for b, v in floors.items()) or "none (no run of the base world at another seed)") + "\n")
        out[partition] = {"seeds": seeds, "rows": rows, "seed_floor": floors}
    return out


# --- figure ----------------------------------------------------------------------------------------------------------

#: reference lines a curve file may carry per partition, as `baseline cliff` writes them, and their labels
REFERENCES = {"one_seed_floor": "one-seed floor (the edit's own run)", "stale_at_seen": "stale@s (base run at the seen seed)", "stale_outside": "stale (base runs outside the truth)"}


def _curves(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text()) or {}
    if "counts" not in doc or not isinstance(doc.get("partitions"), dict):
        raise SystemExit(f"{path}: not a tracks-seen curve file (needs counts and partitions.<partition>.mean)")
    return doc


def _cmd_figure(args: argparse.Namespace) -> int:
    docs = [(path, _curves(path)) for path in args.curves]
    counts = docs[0][1]["counts"]
    for path, doc in docs[1:]:
        if doc["counts"] != counts:
            raise SystemExit(f"{path}: counts {doc['counts']} differ from {docs[0][0]}'s {counts}: not the same protocol")
    parts = args.partition or list(dict.fromkeys(p for _, d in docs for p in d["partitions"]))
    for partition in parts:
        found = [(path, d["partitions"][partition]) for path, d in docs if partition in d["partitions"]]
        if not found:
            raise SystemExit(f"no curve file has partition {partition}")
        seeds = {tuple(part.get("seeds") or []) for _, part in found}
        if len(seeds) > 1:
            raise SystemExit(f"{partition}: the curve files score different seeds {sorted(seeds)}: not the same folds")
        curves: dict[str, list[float | None]] = {}
        refs: dict[str, float] = {}
        tracks = 0
        for path, part in found:
            tracks = max([tracks, *(r.get("tracks_per_run") or 0 for r in part.get("rows") or [])])
            for name, values in (part.get("mean") or {}).items():
                vals = [v for v in values if v is not None]
                if "floor" in name.lower() and vals and max(vals) - min(vals) < 1e-9:
                    refs.setdefault(REFERENCES["one_seed_floor"], vals[0])
                elif name in curves:
                    raise SystemExit(f"{partition}: curve {name!r} is in more than one file")
                else:
                    curves[name] = values
            for key, label in REFERENCES.items():
                if part.get(key) is not None:
                    if label in refs and abs(refs[label] - part[key]) > 5e-4:
                        raise SystemExit(f"{partition}: {key} {part[key]:.4f} in {path} differs from {refs[label]:.4f}: not the same folds")
                    refs.setdefault(label, part[key])
        lines = ["| tracks seen | " + " | ".join(str(k) for k in counts) + " |", "|---|" + "---|" * len(counts)]
        lines += [f"| {n} | " + " | ".join(_fmt(x) for x in m) + " |" for n, m in curves.items()]
        print(f"## {partition}\n\n" + "\n".join(lines) + "\n\n" + ", ".join(f"{k} {v:.3f}" for k, v in refs.items()) + "\n")
        png = args.png if len(parts) == 1 else args.png.with_name(f"{args.png.stem}_{partition}{args.png.suffix}")
        _plot_figure(png, partition, counts, curves, refs, tracks)
    return 0


def _plot_figure(png: Path, partition: str, counts: Sequence[int], curves: Mapping[str, Sequence[float | None]], refs: Mapping[str, float], max_tracks: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, (name, values) in enumerate(curves.items()):
        # counts beyond a run's tracks repeat its last point: each is plotted at the tracks actually seen
        pts = {}
        for c, v in zip(counts, values, strict=True):
            if v is not None:
                pts[max(min(c, max_tracks) if max_tracks else c, 0.5)] = v
        ax.plot(list(pts), list(pts.values()), marker="o", markersize=6, linewidth=1.6, color=SERIES[k % len(SERIES)], label=name)
    styles = {REFERENCES["one_seed_floor"]: ("#2b2a27", "--"), REFERENCES["stale_at_seen"]: ("#6b6a64", ":"), REFERENCES["stale_outside"]: ("#2b2a27", ":")}
    for label, value in refs.items():
        color, style = styles.get(label, ("#6b6a64", "-."))
        ax.axhline(value, color=color, linewidth=1.2, linestyle=style, label=label)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xscale("log")
    ax.set_xlabel("post-edit tracks seen (0 plotted at 0.5" + (f"; {max_tracks} tracks per run)" if max_tracks else ")"))
    ax.set_ylabel("pooled JSD (bits) against the other seeds")
    ax.set_title(f"{partition}: forecast error vs. tracks seen (leave one seed out)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.6)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    _log(f"wrote {png}")


# --- register --------------------------------------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Add `baseline` and its verbs to `flowedit forecast`."""
    p = sub.add_parser("baseline", help="the map baselines: online CLiFF-maps, geometry-only simulation and prior, Deep Crowd-Flow", description="the forecaster's map baselines on a split's recorded runs (see the flowedit README, Map baselines)")
    verbs = p.add_subparsers(dest="baseline_verb", required=True, metavar="{cliff,controls,geometry,deepflow,figure}")

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("split", type=Path, help="the split file (see split.py)")
        q.add_argument("--planner", default=None, help="the contestant whose runs to read, when the benchmark holds several")
        q.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition to score; repeatable")
        q.add_argument("--settings", type=Path, default=None, help="crowd settings YAML merged over the split's")
        q.add_argument("--worlds", type=Path, default=None, help="worlds root to read (default: the ARENA_WORLD_PATH roots and the resolvers)")

    q = verbs.add_parser("cliff", help="online CLiFF-maps: error against post-edit tracks seen", description="online CLiFF-maps fed the tracks of one seed's run, one track per iteration, scored against the pooled map of the other seeds (leave one seed out), from an empty map and from the pre-edit map")
    common(q)
    q.add_argument("--benchmark", type=Path, required=True, help="the split's processed benchmark run dir (flow_cells.parquet, combined_metrics.parquet, the track store)")
    q.add_argument("--prior", action="append", choices=PRIORS, default=None, help="start from an empty map or the pre-edit map; repeatable (default both)")
    q.add_argument("--decay", action="append", default=None, metavar="[PRIOR=]LAMBDA", help="the decay rate lambda, for every prior or one (e.g. empty=0.99); repeatable (default: fitted on the train partition's curve)")
    q.add_argument("--fit-worlds", type=int, default=None, help="fit lambda on the first N training edits only")
    q.add_argument("--params", type=Path, default=None, help="YAML of CliffParams fields to override")
    q.add_argument("--orders", type=int, default=B.ORDERS, help=f"random track orders per fold (default {B.ORDERS})")
    q.add_argument("--counts", type=int, nargs="+", default=None, help=f"track counts to score at (default {' '.join(map(str, B.TRACK_COUNTS))})")
    q.add_argument("--out", type=Path, default=None, help="YAML of every curve")
    q.add_argument("--png", type=Path, default=None, help="PNG of the mean curve per partition")
    q.set_defaults(fn=_cmd_cliff)

    q = verbs.add_parser("controls", help="geometry-only simulation: the controls' runs as their edit's map", description="each geometry-only control's recorded run at one seed as the map of its function-carrying edit, against the edit's other seeds")
    common(q)
    q.add_argument("--cells", type=Path, required=True, help="arena_evaluation's flow_cells.parquet of the split's benchmark")
    q.add_argument("--out", type=Path, default=None, help="YAML of every row")
    q.set_defaults(fn=_cmd_controls)

    q = verbs.add_parser("geometry", help="geometry-only prior: shortest paths between the entrances", description="flow along the shortest paths between every pair of entrances of the edited floor, no demand; writes a map file")
    common(q)
    q.add_argument("--out", type=Path, required=True, help="map file to write (parquet; `forecast score --predicted` reads it)")
    q.add_argument("--cells", type=Path, default=None, help="flow_cells.parquet to score the maps against, beside the floors")
    q.add_argument("--report", type=Path, default=None, help="YAML of the entrances walked and the scores")
    q.add_argument("--plots", type=Path, default=None, help="directory for a PNG per world")
    q.set_defaults(fn=_cmd_geometry)

    q = verbs.add_parser("figure", help="one tracks-seen figure from several curve files", description="the curves of `baseline cliff` and `forecast curve` files in one figure per partition; the files must share the track counts and the scored seeds (the same leave-one-seed-out folds)")
    q.add_argument("curves", type=Path, nargs="+", help="curve YAML files (`baseline cliff --out`, `forecast curve --out`)")
    q.add_argument("--partition", action="append", choices=SP.PARTITIONS, default=None, help="a partition to draw; repeatable (default: every partition of the files)")
    q.add_argument("--png", type=Path, required=True, help="PNG to write (one per partition, suffixed, when there are several)")
    q.set_defaults(fn=_cmd_figure)

    from .deepflow import register as register_deepflow

    register_deepflow(verbs, common, _score_maps)


__all__ = ["FORMAT", "register"]
