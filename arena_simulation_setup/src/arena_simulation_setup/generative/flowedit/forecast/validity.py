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


def truths(runs: Runs, doc: Mapping[str, Any], partition: str, log: Callable[[str], None] = print) -> list[Truth]:
    seeds = SP.seeds_of(doc, partition)
    out = []
    for world in doc["partitions"].get(partition) or []:
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


def floor(runs: Runs, base: str, seeds: Sequence[int]) -> float | None:
    """The base world's pooled map at `seeds` against the one at its other seeds, None without either."""
    other = other_seeds(runs, base, seeds)
    a, b = runs.pooled(base, seeds), runs.pooled(base, other) if other else None
    return None if a is None or b is None else jsd(a, b)


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
    run, for each k of `counts`, averaged over which seed is seen and `orders` random track orders."""
    import numpy as np

    mask = grid.room_cells()
    rng = random.Random(seed)
    out: dict[str, list[list[float]]] = {name: [[] for _ in counts] for name in priors}
    for s in seeds:
        rest = [x for x in seeds if x != s]
        truth = runs.pooled(world, rest) if rest else None
        if truth is None:
            continue
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
                while taken < min(k, len(order)):
                    cum += per[order[taken]]
                    taken += 1
                for name, (prior, strength) in priors.items():
                    out[name][ci].append(float(jsd(dirichlet(prior, strength, cum, mask), truth) or 0.0))
    return {name: [float(np.mean(v)) if v else None for v in vals] for name, vals in out.items()}


def fit_strength(world_seeds: Sequence[tuple[str, Sequence[int]]], tracks_of: Callable[[str, int], pl.DataFrame | None], runs: Runs, grid_of: Callable[[str], FlowGrid], prior_of: Callable[[str], np.ndarray], grid: Sequence[float] = (30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)) -> float:
    """The prior strength (pedestrian-seconds) with the lowest mean curve over `world_seeds` (training worlds and seeds)."""
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


__all__ = ["PREDICTED_COLUMNS", "TRACK_COUNTS", "Method", "Predicted", "Truth", "changes", "curve", "dirichlet", "fit_strength", "floor", "other_seeds", "score", "stale", "table", "track_maps", "transplant", "truths", "uniform", "write_predicted"]
