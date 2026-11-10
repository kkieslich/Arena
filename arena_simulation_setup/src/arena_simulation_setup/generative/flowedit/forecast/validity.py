"""The forecaster's validity checks on a split's recorded maps: scores against fair baselines and the across-seed floor,
the predicted change of each edit and its geometry-only controls against PEDS, and the tracks-seen curve.

**Score.** A world's truth is its pooled map over the partition's seeds (the mean of its normalized maps, as
`process_benchmark` pools them); a method's score is the JSD in bits of its map against it. Every method is a map per
world, so any predicted map on the grid takes part (`Predicted`, the file format is in the flowedit README).
- The forecasters see no recorded flow at all: the learned forecaster and its demand ablation (`learn.py`), the demand
  model (`model.py`, fitted by `forecast fit`).
- **stale** is the base world's pooled map, **transplanted** that map with the edited object's surroundings moved with it
  (`evaluate.transplanted`), both from the base world's runs at the split's *other* seeds, never the scored crowd draws.
  A partition whose base world has no run at another seed (the pilot's office_1) has neither.
- **floor**: the base world's pooled map at the scored seeds against the one at the other seeds, what a new crowd draw of
  an unchanged world costs.

**Change.** Per edit and control: the recorded change (pooled JSD of the world against its base at the same seeds, as
`process_benchmark` reports it) beside each forecaster's predicted change (its map of the world against its map of the
base). A forecaster that has learnt that demand moves flow predicts a large change for a function-carrying edit and a
small one for its inert copy and for the object moved without its demand.

**Tracks seen** (`curve`): the forecast as a Dirichlet prior over cells and bins, updated with the post-edit tracks seen so
far (conjugate: the posterior mean is (c p + n) / (c + N), as the Bayesian Floor Field updates a cell's direction prior),
scored against the pooled map of the partition's other seeds. Zero tracks is the forecast itself; the same update from a
uniform prior over the rooms is the online-only reference.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from .. import split as SP
from .maps import BINS, FlowGrid, jsd, normalized

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

    from .evaluate import Runs

#: a method: the map it forecasts for a world, None when it has none
Method = Callable[[str], "np.ndarray | None"]
PREDICTED_COLUMNS = ("flow_world", "x", "y", "heading_rad", "ped_seconds")


@attrs.frozen
class Truth:
    """A world of a partition with recorded maps at its seeds: the pooled map and the seeds it pools."""

    world: str
    seeds: tuple[int, ...]
    map: Any


def world_seeds(doc: Mapping[str, Any], partition: str, world: str) -> list[int]:
    """The seeds `partition` scores `world` at: the partition's list, or its `bases` / `edits` list when the split gives
    them apart. A partition's `baseline_seeds` are never scored."""
    entry = (doc.get("seeds") or {}).get(partition) if isinstance(doc.get("seeds"), Mapping) else None
    if isinstance(entry, Mapping):
        return [int(s) for s in entry["bases" if SP.is_base(world) else "edits"]]
    return SP.seeds_of(doc, partition)


def truths(runs: Runs, doc: Mapping[str, Any], partition: str, log: Callable[[str], None] = print) -> list[Truth]:
    out = []
    for world in doc["partitions"].get(partition) or []:
        seeds = world_seeds(doc, partition, world)
        have = runs.seeds(world, seeds)
        if not have:
            log(f"{world}: no recorded map at seeds {seeds}, left out")
            continue
        out.append(Truth(world, tuple(have), runs.pooled(world, have)))
    return out


def other_seeds(runs: Runs, world: str, seeds: Iterable[int]) -> list[int]:
    """The seeds `world` has recorded maps at, but not among `seeds`."""
    scored = set(seeds)
    return sorted(s for (w, s) in runs.maps if w == world and s not in scored)


def stale(runs: Runs, seeds: Sequence[int]) -> Method:
    """The base world's pooled map at the seeds other than `seeds`."""

    def method(world: str) -> np.ndarray | None:
        base = SP.base_of(world)
        other = other_seeds(runs, base, seeds)
        return runs.pooled(base, other) if other else None

    return method


def transplant(runs: Runs, seeds: Sequence[int], sites: Any, scenario_of: Mapping[str, str | None]) -> Method:  # noqa: ANN401 - evaluate.Forecaster
    """`stale` with the edited object's surroundings moved, copied or dropped with it (`evaluate.transplanted`)."""
    from .evaluate import transplant_moves, transplanted

    base_map = stale(runs, seeds)

    def method(world: str) -> np.ndarray | None:
        m = base_map(world)
        if m is None:
            return None
        base = SP.base_of(world)
        if world == base:
            return m
        return transplanted(m, sites.grid(base), transplant_moves(sites.site(base, scenario_of.get(base)), sites.site(world)))

    return method


@attrs.frozen
class Floor:
    """The JSD between two equal-size seed pools of an unchanged world, over the pairs `floor` draws: mean, spread, pool size."""

    mean: float
    sd: float
    seeds: int
    pairs: int


#: pool pairs `floor` averages over at most
FLOOR_PAIRS = 50


def floor(runs: Runs, base: str, seeds: Sequence[int], max_pairs: int = FLOOR_PAIRS) -> Floor | None:
    """Another crowd draw of the unchanged base world: n of its scored `seeds` pooled against n of its other seeds pooled,
    n the smaller count, over every such pair (at most `max_pairs`, drawn by a fixed seed); None without either side.

    It is the noise between two finite n-seed pools. A smooth expected-flow forecast is scored against one finite pool
    only, so it can lie below this floor; the floor is a scale, not a bound, and stale at the other seeds stays the main
    comparison.
    """
    import itertools

    import numpy as np

    scored = runs.seeds(base, seeds)
    other = other_seeds(runs, base, seeds)
    n = min(len(scored), len(other))
    if n == 0:
        return None
    pairs = [(a, b) for a in itertools.combinations(scored, n) for b in itertools.combinations(other, n)]
    if len(pairs) > max_pairs:
        pairs = random.Random(0).sample(pairs, max_pairs)
    values = [jsd(runs.pooled(base, a), runs.pooled(base, b)) for a, b in pairs]
    values = [float(v) for v in values if v is not None]
    if not values:
        return None
    return Floor(float(np.mean(values)), float(np.std(values)), n, len(values))


@attrs.define
class Predicted:
    """Maps from a parquet file of cell rows (PREDICTED_COLUMNS), one map per flow_world on its base world's grid."""

    maps: dict[str, Any]

    @classmethod
    def load(cls, path: Path, grid_of: Callable[[str], FlowGrid]) -> Predicted:
        import polars as pl

        cells = pl.read_parquet(path)
        missing = set(PREDICTED_COLUMNS) - set(cells.columns)
        if missing:
            raise ValueError(f"{path}: a predicted map needs columns {list(PREDICTED_COLUMNS)}, missing {sorted(missing)}")
        maps = {}
        for (world,), part in cells.group_by(["flow_world"], maintain_order=True):
            grid = grid_of(SP.base_of(str(world)))
            maps[str(world)] = grid.dense(part["x"].to_numpy(), part["y"].to_numpy(), part["heading_rad"].to_numpy(), part["ped_seconds"].to_numpy())
        return cls(maps)

    def __call__(self, world: str) -> np.ndarray | None:
        return self.maps.get(world)


def write_predicted(path: Path, maps: Mapping[str, np.ndarray], grid_of: Callable[[str], FlowGrid]) -> Path:
    """Write maps by world in the format `Predicted.load` reads (and `forecast score --predicted` takes)."""
    import polars as pl

    frames = []
    for world, m in maps.items():
        rows = grid_of(SP.base_of(world)).rows(m)
        frames.append(pl.DataFrame({"flow_world": [world] * len(rows["x"]), **rows}, schema_overrides={"heading_rad": pl.Float64}))
    pl.concat(frames).write_parquet(path)
    return path


def label(doc: Mapping[str, Any], world: str) -> str:
    """What a world of a split is: base, an edit (`<type>_<target>`), or a control of one."""
    from .evaluate import label as _label

    return _label(doc, world)


def score(found: Sequence[Truth], methods: Mapping[str, Method], doc: Mapping[str, Any], coarse: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One row per world: each method's JSD against the pooled truth (None where it has no map), and for the methods in
    `coarse` also the JSD of the occupancy on 1 m cells (`<method> 1m`)."""
    from .learn import jsd_coarse

    rows = []
    for t in found:
        row: dict[str, Any] = {"world": t.world, "what": label(doc, t.world), "seeds": list(t.seeds)}
        for name, method in methods.items():
            m = method(t.world)
            row[name] = None if m is None else jsd(m, t.map)
            if name in coarse:
                row[f"{name} 1m"] = None if m is None else jsd_coarse(m, t.map)
        rows.append(row)
    return rows


def changes(found: Sequence[Truth], runs: Runs, methods: Mapping[str, Method], doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per edit or control: its recorded change against the base world at the same seeds, and each method's
    predicted change (its map of the world against its map of the base)."""
    rows = []
    for t in found:
        base = SP.base_of(t.world)
        if base == t.world:
            continue
        ref = runs.pooled(base, t.seeds) if runs.seeds(base, t.seeds) == list(t.seeds) else None
        row: dict[str, Any] = {"world": t.world, "what": label(doc, t.world), "recorded": None if ref is None else jsd(t.map, ref)}
        for name, method in methods.items():
            a, b = method(t.world), method(base)
            row[name] = None if a is None or b is None else jsd(a, b)
        rows.append(row)
    return rows


def table(rows: Sequence[Mapping[str, Any]], cols: Sequence[str], mean_label: str = "mean over edits") -> str:
    """Rows as a markdown table, the mean over the edit rows (not the base world) last."""

    def fmt(v: Any) -> str:  # noqa: ANN401
        return "" if v is None else f"{v:.3f}"

    out = ["| World | What | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        out.append(f"| {r['world']} | {r['what']} | " + " | ".join(fmt(r.get(c)) for c in cols) + " |")
    edits = [r for r in rows if r["what"] != "base"]
    if edits:
        means = []
        for c in cols:
            vals = [r[c] for r in edits if r.get(c) is not None]
            means.append(fmt(sum(vals) / len(vals)) if len(vals) == len(edits) else ("" if not vals else f"({sum(vals) / len(vals):.3f}, {len(vals)} of {len(edits)})"))
        out.append(f"| {mean_label} | | " + " | ".join(means) + " |")
    return "\n".join(out)


# --- tracks seen -----------------------------------------------------------------------------------------------------


def track_maps(tracks: pl.DataFrame, grid: FlowGrid) -> dict[str, np.ndarray]:
    """Each pedestrian's map (pedestrian-seconds per cell and bin, `maps.counts`) by track key."""
    from .maps import counts

    return {str(key): counts(part, grid) for (key,), part in tracks.group_by(["key"], maintain_order=True)}


def dirichlet(prior: np.ndarray, strength: float, seen: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The posterior mean (strength * prior + seen) / (strength + N) over the rooms' cells and bins."""
    import numpy as np

    p = normalized(prior * mask[:, :, None])
    p = np.zeros_like(seen) if p is None else p
    return (strength * p + seen * mask[:, :, None]) / max(strength + float((seen * mask[:, :, None]).sum()), 1e-12)


def uniform(mask: np.ndarray) -> np.ndarray:
    import numpy as np

    return np.repeat(mask[:, :, None].astype(float), BINS, axis=2)


#: tracks seen at which the curve is scored
TRACK_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64)
#: the curve's row of the seen run's own map against the other seeds' pool
ONE_SEED_FLOOR = "one-seed floor"


def curve(
    world: str,
    seeds: Sequence[int],
    tracks_of: Callable[[str, int], pl.DataFrame | None],
    runs: Runs,
    grid: FlowGrid,
    priors: Mapping[str, tuple[np.ndarray, float]],
    counts: Sequence[int] = TRACK_COUNTS,
    orders: int = 20,
    seed: int = 0,
    cache: dict[tuple[str, int], dict[str, np.ndarray]] | None = None,
) -> dict[str, list[float | None]]:
    """Per prior (map, strength): the mean JSD against the pooled map of the other seeds after seeing k tracks of one seed's
    run, for each k of `counts`, averaged over which seed is seen and `orders` random track orders. A k beyond a run's
    tracks is not scored for that run (None when no run has that many). Row ONE_SEED_FLOOR: the seen run's own map
    against the same pool, what all of its tracks reach."""
    import numpy as np

    mask = grid.room_cells()
    rng = random.Random(seed)
    out: dict[str, list[list[float]]] = {name: [[] for _ in counts] for name in priors}
    floors: list[float] = []
    for s in seeds:
        rest = [x for x in seeds if x != s]
        truth = runs.pooled(world, rest) if rest else None
        if truth is None:
            continue
        if (world, s) in runs.maps:
            if (value := jsd(runs.maps[(world, s)], truth)) is not None:
                floors.append(float(value))
        per = None if cache is None else cache.get((world, s))
        if per is None:
            tracks = tracks_of(world, s)
            per = {} if tracks is None or len(tracks) == 0 else track_maps(tracks, grid)
            if cache is not None:
                cache[(world, s)] = per
        if not per:
            continue
        keys = sorted(per)
        for _ in range(orders):
            order = keys[:]
            rng.shuffle(order)
            cum = np.zeros(grid.shape)
            taken = 0
            for ci, k in enumerate(counts):
                if k > len(order):
                    break  # the run has fewer tracks: no score at k, rather than repeating the last one
                while taken < k:
                    cum += per[order[taken]]
                    taken += 1
                for name, (prior, strength) in priors.items():
                    # an empty posterior (a prior with no mass on the rooms and no track yet) has no score, not a perfect one
                    if (value := jsd(dirichlet(prior, strength, cum, mask), truth)) is not None:
                        out[name][ci].append(float(value))
    result = {name: [float(np.mean(v)) if v else None for v in vals] for name, vals in out.items()}
    # what every track of the seen run reaches: its own recorded map against the other seeds' pool
    result[ONE_SEED_FLOOR] = [float(np.mean(floors)) if floors else None] * len(counts)
    return result


#: prior strengths (pedestrian-seconds) the cross-validation tries
STRENGTHS = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)
#: tracks seen the cross-validation scores a strength at (k = 0 is the prior itself, the same for every strength)
STRENGTH_COUNTS = (1, 2, 4, 8)


def seed_folds(seeds: Sequence[int]) -> list[tuple[int, int]]:
    """Leave-two-seeds-out folds over `seeds`: each seed with the next (cyclic), so every seed is held out twice. Two held-out
    seeds per fold, because the curve needs one run to take tracks from and another to score against, neither seen in
    training."""
    s = sorted(seeds)
    if len(s) < 3:
        raise ValueError(f"leave-two-seeds-out needs at least 3 training seeds, got {s}")
    return [(s[i], s[(i + 1) % len(s)]) for i in range(len(s))]


def fold_strengths(worlds: Sequence[str], fold: Sequence[int], tracks_of: Callable[[str, int], pl.DataFrame | None], runs: Runs, grid_of: Callable[[str], FlowGrid], prior_of: Callable[[str], np.ndarray], strengths: Sequence[float] = STRENGTHS) -> dict[float, float]:
    """Per strength, the mean curve (STRENGTH_COUNTS) over `worlds` with tracks from one seed of `fold` and the truth at the
    other: the fold's score of each strength for a prior that saw neither seed."""
    import numpy as np

    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    scores: dict[float, list[float]] = {c: [] for c in strengths}
    for world in worlds:
        prior = prior_of(world)
        res = curve(world, list(fold), tracks_of, runs, grid_of(SP.base_of(world)), {str(c): (prior, c) for c in strengths}, counts=STRENGTH_COUNTS, orders=10, cache=cache)
        for c in strengths:
            scores[c].extend(v for v in res[str(c)] if v is not None)
    return {c: float(np.mean(v)) for c, v in scores.items() if v}


def choose_strength(folds: Sequence[Mapping[float, float]]) -> tuple[float, dict[str, Any]]:
    """The strength with the lowest mean over folds, and the spread: each fold's own best, and the folds' scores at the choice."""
    import numpy as np

    common = sorted(set.intersection(*(set(f) for f in folds)))
    mean = {c: float(np.mean([f[c] for f in folds])) for c in common}
    chosen = min(mean, key=mean.__getitem__)
    best = [min(f, key=f.__getitem__) for f in folds]
    at = [f[chosen] for f in folds]
    return chosen, {"fold_best": best, "fold_best_min": min(best), "fold_best_max": max(best), "score_at_chosen": {"mean": mean[chosen], "sd": float(np.std(at)), "min": min(at), "max": max(at)}, "mean_by_strength": mean}


def fit_strength(world_seeds: Sequence[tuple[str, Sequence[int]]], tracks_of: Callable[[str, int], pl.DataFrame | None], runs: Runs, grid_of: Callable[[str], FlowGrid], prior_of: Callable[[str], np.ndarray], grid: Sequence[float] = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)) -> float:
    """The prior strength (pedestrian-seconds) with the lowest mean curve over `world_seeds` (training worlds and seeds).
    Valid only for a prior that did not learn from those runs (the uniform prior); a learned forecaster needs the
    cross-validated `forecast strength` (`seed_folds`, `fold_strengths`, `choose_strength`)."""
    import numpy as np

    best, value = grid[0], np.inf
    cache: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    priors = {world: prior_of(world) for world, _ in world_seeds}
    for c in grid:
        scores = []
        for world, seeds in world_seeds:
            g = grid_of(SP.base_of(world))
            res = curve(world, seeds, tracks_of, runs, g, {"p": (priors[world], c)}, counts=(1, 4, 16), orders=4, cache=cache)
            scores.extend(v for v in res["p"] if v is not None)
        if scores and float(np.mean(scores)) < value:
            best, value = c, float(np.mean(scores))
    return best


__all__ = [
    "FLOOR_PAIRS",
    "Floor",
    "Method",
    "ONE_SEED_FLOOR",
    "PREDICTED_COLUMNS",
    "Predicted",
    "STRENGTHS",
    "STRENGTH_COUNTS",
    "TRACK_COUNTS",
    "Truth",
    "changes",
    "choose_strength",
    "curve",
    "dirichlet",
    "fit_strength",
    "floor",
    "fold_strengths",
    "other_seeds",
    "score",
    "seed_folds",
    "stale",
    "table",
    "track_maps",
    "transplant",
    "truths",
    "uniform",
    "world_seeds",
    "write_predicted",
]
