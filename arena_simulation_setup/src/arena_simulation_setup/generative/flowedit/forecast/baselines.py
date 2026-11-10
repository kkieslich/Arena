"""Map baselines of the forecaster: online CLiFF-maps fed post-edit tracks, the geometry-only prior and the geometry-only
simulation. Deep Crowd-Flow is in `deepflow.py`; stale and transplanted in `evaluate.py`.

**Leave one seed out.** A baseline built from recorded runs never sees the crowd draws it is scored on. Held-out edits run
only at their partition's seeds (the pilot's 6 to 8), so a baseline that reads runs of the edited world itself, or of a
control paired with it, takes them from one seed s, and the truth is the pooled map of the partition's other seeds. A
score is the mean over the folds s. Its floor is the edited world's own run at s against the same truth (the
**one-seed floor**: what a perfect estimate of the flow behind one run still misses), reported beside it, with
**stale@s**, the base world's run at s, which shares the crowd draw of the baseline's run, and **stale (outside)**, the
base world's pooled map at every seed outside the truth, the runs the pre-edit CLiFF-map is built from.

**Tracks seen** (`curve`): an online map starts empty, or from the pre-edit map (the base world's runs at the seeds
outside the truth), and is fed one post-edit track (one pedestrian of the run at seed s) per iteration. The curve is its
JSD against the other seeds' pooled map after k tracks, averaged over the folds and over random track orders. The folds,
the orders (`random.Random(0)` per world, the tracks' keys sorted, shuffled per order) and the track counts are those of
the learned forecaster's Dirichlet curve, so both curves see the same tracks in the same order.

**Geometry-only prior** (`geometry_prior`): walks along the shortest paths between every ordered pair of entrances of the
edited floor, no demand and no people. The entrances are its exterior doors (doors only one room contains, the reader's
entrances); a floor with fewer than two takes every open door as an entrance, each room's way in. Paths run on the
forecaster's free space of the edited world (`scene.py`: HumanSim's inflation and comfort cost, locked doors closed),
walked at `speed`, headings spread and the time blurred as the demand model does (`model.predict`) with its default
spreads, kept to the reachable cells of the rooms.

**Geometry-only simulation** (`controls`): the recorded runs of a function-carrying edit's geometry-only controls, each
scored as a prediction of its edit, leave one seed out: the control's run at s against the edit's other seeds.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import attrs

from .. import split as SP
from .maps import FlowGrid, jsd

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

    from ..edit import Site
    from .evaluate import Runs

#: tracks seen at which a curve is scored: the learned forecaster's counts
TRACK_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64)
#: random track orders per fold
ORDERS = 20
#: the demand model's default spreads (`model.Params`): kernel width (m) and heading spread (rad)
PRIOR_SIGMA_M = 0.5
PRIOR_HEADING_RAD = 0.3
#: walking speed of the geometry-only prior's walks (m/s); only the share of time per cell matters
PRIOR_SPEED = 1.0
#: columns of a map file (`forecast score --predicted`)
MAP_COLUMNS = ("flow_world", "x", "y", "heading_rad", "ped_seconds")


# --- recorded tracks -------------------------------------------------------------------------------------------------


@attrs.define
class Tracks:
    """The per-pedestrian track store of a processed benchmark run dir, by (world, seed): each stage's first episode."""

    where: dict[tuple[str, int], Path]

    @classmethod
    def load(cls, benchmark: Path, planner: str | None = None) -> Tracks:
        import polars as pl

        from ..suite import parse_stage_name

        metrics = pl.read_parquet(Path(benchmark) / "combined_metrics.parquet", columns=["episode", "stage", "planner", "robot"])
        if planner is not None:
            metrics = metrics.filter(pl.col("planner") == planner)
        where: dict[tuple[str, int], Path] = {}
        for episode, stage, _, robot in metrics.sort("episode").iter_rows():
            try:
                key = parse_stage_name(str(stage))
            except ValueError:
                continue
            env = str(robot or "env_0").split("_")
            env_name = "_".join(env[:2]) if len(env) >= 2 and env[0] == "env" else "env_0"
            where.setdefault(key, Path(benchmark) / "episodes" / f"episode_{int(episode):03d}" / "topics" / ".derived" / "ped_tracks.v1" / f"{env_name}.parquet")
        return cls(where)

    def __call__(self, world: str, seed: int) -> pl.DataFrame | None:
        import polars as pl

        path = self.where.get((world, seed))
        return pl.read_parquet(path) if path is not None and path.exists() else None


# --- leave one seed out ----------------------------------------------------------------------------------------------


@attrs.frozen
class Fold:
    """One fold: the seed whose runs a baseline reads, and the pooled map of the other seeds it is scored against."""

    seen: int
    rest: tuple[int, ...]
    truth: Any


def folds(runs: Runs, world: str, seeds: Sequence[int]) -> list[Fold]:
    """The folds of `world` over those of `seeds` it has recorded maps at (at least two)."""
    have = runs.seeds(world, seeds)
    out = []
    for s in have:
        rest = tuple(x for x in have if x != s)
        if rest:
            out.append(Fold(s, rest, runs.pooled(world, rest)))
    return out


def outside(runs: Runs, world: str, seeds: Iterable[int]) -> list[int]:
    """The seeds `world` has recorded maps at, outside `seeds`: the runs a baseline may read when `seeds` are scored."""
    scored = set(seeds)
    return sorted(s for (w, s) in runs.maps if w == world and s not in scored)


def mean(values: Iterable[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def loso(runs: Runs, world: str, seeds: Sequence[int], predict_at: Callable[[Fold], np.ndarray | None]) -> float | None:
    """A baseline's mean JSD over the folds of `world`, None where it has no map in any."""
    return mean(None if (m := predict_at(f)) is None else jsd(m, f.truth) for f in folds(runs, world, seeds))


def one_seed_floor(runs: Runs, world: str, seeds: Sequence[int]) -> float | None:
    """The world's own run at the seen seed against the other seeds' pooled map, averaged over the folds."""
    return loso(runs, world, seeds, lambda f: runs.maps.get((world, f.seen)))


def stale_at_seen(runs: Runs, world: str, seeds: Sequence[int]) -> float | None:
    """The base world's run at the seen seed against the edit's other seeds, averaged over the folds."""
    base = SP.base_of(world)
    return loso(runs, world, seeds, lambda f: runs.maps.get((base, f.seen)))


def stale_outside(runs: Runs, world: str, seeds: Sequence[int]) -> float | None:
    """The base world's pooled map at every seed outside the fold's truth (the pre-edit map's runs) against the edit's
    other seeds, averaged over the folds."""
    base = SP.base_of(world)
    return loso(runs, world, seeds, lambda f: runs.pooled(base, o) if (o := outside(runs, base, f.rest)) else None)


# --- tracks seen -----------------------------------------------------------------------------------------------------


class Online(Protocol):
    """An online map: started per fold, fed one track per iteration, read as a flow map."""

    def start(self, world: str, fold: Fold) -> Any: ...  # noqa: ANN401
    def feed(self, state: Any, track: pl.DataFrame) -> Any: ...  # noqa: ANN401
    def map(self, state: Any) -> np.ndarray | None: ...  # noqa: ANN401


@attrs.define
class OnlineCliff:
    """The online CLiFF-map: empty at the start of a fold, or the pre-edit map built from the base world's runs at the seeds
    outside the fold's truth (`prior="pre-edit"`)."""

    grid_of: Callable[[str], FlowGrid]
    tracks_of: Callable[[str, int], pl.DataFrame | None]
    runs: Runs
    params: Any = None
    prior: str = "empty"
    #: pre-edit maps by (base, seeds, params but the decay), shareable between instances that differ only in the decay
    built: dict[tuple[Any, ...], Any] = attrs.Factory(dict)

    def __attrs_post_init__(self) -> None:
        from .cliff import CliffParams

        if self.params is None:
            self.params = CliffParams()
        if self.prior not in ("empty", "pre-edit"):
            raise ValueError(f"prior: 'empty' or 'pre-edit', got {self.prior!r}")

    def pre_edit(self, world: str, fold: Fold) -> Any:  # noqa: ANN401
        """The base world's CLiFF-map from its runs at every seed outside the fold's truth, built once per seed set."""
        from .cliff import from_tracks

        import attrs

        base = SP.base_of(world)
        seeds = tuple(outside(self.runs, base, fold.rest))
        # one iteration from empty: the decay rate changes nothing in a pre-edit map
        key = (base, seeds, attrs.evolve(self.params, decay=1.0))
        if key not in self.built:
            found = [t for s in seeds if (t := self.tracks_of(base, s)) is not None]
            if not found:
                raise ValueError(f"{world}: no track store of {base} at seeds {list(seeds)} for a pre-edit map")
            self.built[key] = from_tracks(self.grid_of(base), found, self.params)
        return self.built[key]

    def start(self, world: str, fold: Fold) -> Any:  # noqa: ANN401
        from .cliff import CliffMap

        if self.prior == "pre-edit":
            # the cached map may come from an instance with another decay rate: this one's params update it
            state = self.pre_edit(world, fold).copy()
            state.params = self.params
            return state
        return CliffMap.empty(self.grid_of(SP.base_of(world)), self.params)

    def feed(self, state: Any, track: pl.DataFrame) -> Any:  # noqa: ANN401
        from .cliff import observations

        return state.update(observations(track, self.params.rate_hz))

    def map(self, state: Any) -> np.ndarray | None:  # noqa: ANN401
        m = state.map()
        return m if float(m.sum()) > 0 else None


def curve(world: str, seeds: Sequence[int], tracks_of: Callable[[str, int], pl.DataFrame | None], runs: Runs, methods: Mapping[str, Online], counts: Sequence[int] = TRACK_COUNTS, orders: int = ORDERS, seed: int = 0) -> dict[str, list[float | None]]:
    """Per method: the mean JSD against the pooled map of the other seeds after k tracks of the seen seed's run, for each k
    of `counts`, over the folds and `orders` random track orders; None where it has no map yet (an empty map at 0)."""
    rng = random.Random(seed)
    out: dict[str, list[list[float]]] = {name: [[] for _ in counts] for name in methods}
    for f in folds(runs, world, seeds):
        tracks = tracks_of(world, f.seen)
        if tracks is None or len(tracks) == 0:
            continue
        per = {str(key): part for (key,), part in tracks.group_by(["key"], maintain_order=True)}
        keys = sorted(per)
        for _ in range(orders):
            order = keys[:]
            rng.shuffle(order)
            for name, method in methods.items():
                state = method.start(world, f)
                taken = 0
                for ci, k in enumerate(counts):
                    while taken < min(k, len(order)):
                        state = method.feed(state, per[order[taken]])
                        taken += 1
                    m = method.map(state)
                    if m is not None:
                        out[name][ci].append(float(jsd(m, f.truth) or 0.0))
    return {name: [mean(v) for v in vals] for name, vals in out.items()}


# --- tuning ----------------------------------------------------------------------------------------------------------

#: the share of the train partition's edit chains (of the function-carrying ones and of the others) that validate a
#: baseline's hyperparameters, and the draw's seed
VALIDATION_SHARE = 0.2
VALIDATION_SEED = 0
#: track counts and orders of the tracks-seen curve when it scores a setting on the validation worlds
TUNE_COUNTS = (1, 4, 16)
TUNE_ORDERS = 4
#: the grids, searched one after the other at the values chosen so far (one coordinate pass)
CLIFF_GRID: Mapping[str, Sequence[Any]] = {
    "decay": (0.5, 0.8, 0.9, 0.95, 0.99),
    "radius_m": (0.354, 0.5, 0.75),
    "bandwidth": ((0.25, 0.15), (0.5, 0.3), (1.0, 0.6)),
}
GEOMETRY_GRID: Mapping[str, Sequence[float]] = {"sigma": (0.25, 0.5, 1.0), "heading": (0.15, 0.3, 0.6)}


def chains(doc: Mapping[str, Any], partition: str = "train") -> list[list[str]]:
    """The edit chains of `partition`: each edited world with its geometry-only controls (`control_worlds`), sorted; base worlds left out."""
    worlds = [w for w in doc["partitions"].get(partition) or [] if not SP.is_base(w)]
    links = {e: [c for c in cs if c in worlds] for e, cs in (doc.get("control_worlds") or {}).items() if e in worlds}
    controls = {c for cs in links.values() for c in cs}
    return sorted([w, *links.get(w, [])] for w in worlds if w not in controls)


def validation_split(doc: Mapping[str, Any], share: float = VALIDATION_SHARE, seed: int = VALIDATION_SEED) -> tuple[list[str], list[str]]:
    """(training worlds, validation worlds) of the train partition. Its edit chains fall in two groups, the
    function-carrying ones (an edit with geometry-only controls) and the others; of each group a fixed `share` (rounded
    up, but one chain of the group always stays in training) validates, drawn with `random.Random(seed)`. The base worlds
    and the other chains train. Nothing of another partition is read."""
    import math

    found = chains(doc)
    rng = random.Random(seed)
    picked: list[list[str]] = []
    for group in ([c for c in found if len(c) > 1], [c for c in found if len(c) == 1]):
        k = min(len(group) - 1, math.ceil(share * len(group))) if group else 0
        picked += [group[i] for i in sorted(rng.sample(range(len(group)), k))]
    if not picked:
        raise ValueError(f"the train partition has {len(found)} edit chain(s): too few to hold any out for validation")
    validation = [w for c in picked for w in c]
    training = [w for w in doc["partitions"].get("train") or [] if w not in set(validation)]
    return training, validation


def map_score(runs: Runs, worlds: Sequence[str], seeds: Sequence[int], predict: Callable[[str], np.ndarray | None]) -> float | None:
    """A map baseline's tuning score: the mean pooled JSD of its maps against the worlds' pooled maps at `seeds`."""
    vals = []
    for w in worlds:
        have = runs.seeds(w, seeds)
        if have and (m := predict(w)) is not None:
            vals.append(jsd(m, runs.pooled(w, have)))
    return mean(vals)


def tune_cliff(worlds: Sequence[tuple[str, Sequence[int]]], tracks_of: Callable[[str, int], pl.DataFrame | None], runs: Runs, make: Callable[[Any], Online], start: Any, grid: Mapping[str, Sequence[Any]] = CLIFF_GRID, log: Callable[[str], None] = print) -> tuple[Any, list[dict[str, Any]]]:
    """CliffParams by one coordinate pass over `grid` from `start`: per field in order, the value with the lowest mean curve
    over `worlds` (validation worlds and their seeds) at TUNE_COUNTS, the others at their values so far; ties keep the first."""
    import attrs

    params = start
    table = []
    for field, values in grid.items():
        scores: dict[Any, float] = {}
        for v in values:
            trial = attrs.evolve(params, **{field: v})
            vals = [x for w, s in worlds for x in curve(w, s, tracks_of, runs, {"m": make(trial)}, TUNE_COUNTS, TUNE_ORDERS)["m"] if x is not None]
            if vals:
                scores[v] = sum(vals) / len(vals)
                table.append({"field": field, "value": list(v) if isinstance(v, tuple) else v, "score": scores[v]})
                log(f"cliff tuning: {field} {v}: mean JSD {scores[v]:.4f} on {len(worlds)} validation worlds")
        if not scores:
            raise ValueError("no validation world with tracks to tune on")
        params = attrs.evolve(params, **{field: min(scores, key=scores.get)})
    return params, table


def tune_geometry(walks_of: Callable[[str], Walks], runs: Runs, worlds: Sequence[str], seeds: Sequence[int], grid: Mapping[str, Sequence[float]] = GEOMETRY_GRID) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """The geometry-only prior's spreads with the lowest map_score on the validation `worlds` over the full `grid`; ties keep the first."""
    import itertools

    table = []
    best: tuple[float, dict[str, float]] | None = None
    for combo in itertools.product(*grid.values()):
        values = dict(zip(grid, combo, strict=True))
        score = map_score(runs, worlds, seeds, lambda w, values=values: geometry_map(walks_of(w), **values))
        table.append({**values, "score": score})
        if score is not None and (best is None or score < best[0]):
            best = (score, values)
    if best is None:
        raise ValueError("no validation world with recorded runs to tune on")
    return best[1], table


# --- geometry-only simulation ----------------------------------------------------------------------------------------


def controls(runs: Runs, doc: Mapping[str, Any], partition: str) -> list[dict[str, Any]]:
    """One row per function-carrying edit of `partition` and geometry-only control: the control's runs as predictions of
    the edit, leave one seed out, beside stale@s and the one-seed floor."""
    seeds = SP.seeds_of(doc, partition)
    members = set(doc["partitions"].get(partition) or [])
    rows = []
    for edit, ctrls in (doc.get("control_worlds") or {}).items():
        if edit not in members or not folds(runs, edit, seeds):
            continue
        for c in ctrls:
            rows.append(
                {
                    "world": edit,
                    "control": c,
                    "seeds": runs.seeds(edit, seeds),
                    "geometry_simulation": loso(runs, edit, seeds, lambda f, c=c: runs.maps.get((c, f.seen))),
                    "stale_at_seen": stale_at_seen(runs, edit, seeds),
                    "one_seed_floor": one_seed_floor(runs, edit, seeds),
                }
            )
    return rows


# --- geometry-only prior ---------------------------------------------------------------------------------------------


def entrances(site: Site) -> tuple[list[tuple[str, tuple[float, float]]], str]:
    """The entrances of the edited site's people floor as (door, midpoint in the world frame), and which kind they are:
    its exterior doors, or every open door when it has fewer than two."""
    from .. import read as R

    graph = R.read_graph(site.graph)
    floor = R._floor(site.root, site.world, graph)
    level = site.world.levels.get(floor.level_id)
    doors = {d.name: d for d in (level.doors if level is not None else [])}
    locked = {str(f.subject).partition(":")[2] for f in site.graph.facts if f.verb == "gated" and f.object is True}
    ox, oy = floor.origin

    def mid(name: str) -> tuple[float, float]:
        d = doors[name]
        return ((d.start.x + d.end.x) / 2 + ox, (d.start.y + d.end.y) / 2 + oy)

    exterior = [d for d in graph.exterior_doors() if d in doors and d not in locked]
    if len(exterior) >= 2:
        return [(d, mid(d)) for d in exterior], "exterior doors"
    return [(d, mid(d)) for d in sorted(doors) if d not in locked], "every open door (fewer than two exterior doors)"


@attrs.frozen
class Walks:
    """The geometry-only prior's walks on a grid before any spread: walking seconds per cell and path heading, and the
    reachable room cells."""

    grid: FlowGrid
    flat: Any
    heading: Any
    seconds: Any
    reach: Any
    info: dict[str, Any]


def geometry_walks(site: Site, grid: FlowGrid, settings: Mapping[str, Any], duration: float) -> Walks:
    """The walks along the shortest paths between every ordered pair of entrances of `site`, on its base world's `grid`."""
    import numpy as np

    from .model import _Paths
    from .scene import scene_of

    scene = scene_of(site, settings, duration, grid=grid, people=[])
    paths = _Paths(scene)
    found, kind = entrances(site)
    nodes = [(name, paths.node(p)) for name, p in found]
    trees = paths.trees([n for _, n in nodes])
    flats, heads, secs = [], [], []
    walked, unreachable = 0, []
    for a, na in nodes:
        dist, pred = trees[na]
        for b, nb in nodes:
            if a == b or na == nb:
                continue
            route = paths.trace(pred, na, nb) if np.isfinite(dist[nb]) else None
            if route is None:
                unreachable.append(f"{a}->{b}")
                continue
            flat, head, sec, _, _ = paths.deposit(route, PRIOR_SPEED)
            flats.append(flat)
            heads.append(head)
            secs.append(sec)
            walked += 1
    reach = np.zeros(grid.ny * grid.nx, bool)
    seen = np.unique(paths.cell[np.concatenate([np.flatnonzero(np.isfinite(trees[n][0])) for _, n in nodes])]) if nodes else np.zeros(0, np.int64)
    reach[seen[seen >= 0]] = True
    cat = (lambda xs, dtype: np.concatenate(xs) if xs else np.zeros(0, dtype))
    info = {"entrances": [n for n, _ in found], "kind": kind, "walks": walked, "unreachable": unreachable}
    return Walks(grid, cat(flats, np.int64), cat(heads, np.int64), cat(secs, float), reach.reshape(grid.ny, grid.nx) & grid.room_cells(), info)


def geometry_map(walks: Walks, sigma: float = PRIOR_SIGMA_M, heading: float = PRIOR_HEADING_RAD) -> np.ndarray:
    """The prior's map: walking time spread by `heading` rad over the bins and blurred by `sigma` m, kept to the reachable rooms."""
    import numpy as np
    import scipy.ndimage

    from .maps import BINS, HEADINGS
    from .model import heading_table

    g = walks.grid
    size = g.ny * g.nx * BINS
    out = np.bincount((walks.flat[:, None] + np.arange(HEADINGS)).ravel(), weights=(walks.seconds[:, None] * heading_table(heading)[walks.heading]).ravel(), minlength=size)[:size]
    m = out.reshape(g.ny, g.nx, BINS)
    if sigma >= 0.05:
        s = sigma / g.cell
        m = scipy.ndimage.gaussian_filter(m, sigma=(s, s, 0), mode="constant", truncate=3.0)
    return m * walks.reach[:, :, None]


def geometry_prior(site: Site, grid: FlowGrid, settings: Mapping[str, Any], duration: float, sigma: float = PRIOR_SIGMA_M, heading: float = PRIOR_HEADING_RAD) -> tuple[np.ndarray, dict[str, Any]]:
    """The geometry-only prior of `site` on its base world's `grid`, and what it walked (entrances, pairs, unreachable)."""
    walks = geometry_walks(site, grid, settings, duration)
    return geometry_map(walks, sigma, heading), walks.info


# --- map files -------------------------------------------------------------------------------------------------------


def write_maps(path: Path, maps: Mapping[str, np.ndarray], grid_of: Callable[[str], FlowGrid]) -> Path:
    """Maps by world as a map file (MAP_COLUMNS, one map per flow_world on its base world's grid), as `forecast score
    --predicted` reads one."""
    import polars as pl

    frames = []
    for world, m in maps.items():
        rows = grid_of(SP.base_of(world)).rows(m)
        frames.append(pl.DataFrame({"flow_world": [world] * len(rows["x"]), **{k: rows[k] for k in MAP_COLUMNS[1:]}}, schema_overrides={"heading_rad": pl.Float64}))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pl.concat(frames).write_parquet(path)
    return Path(path)


__all__ = [
    "MAP_COLUMNS",
    "ORDERS",
    "TRACK_COUNTS",
    "Fold",
    "Online",
    "OnlineCliff",
    "Tracks",
    "controls",
    "curve",
    "entrances",
    "chains",
    "map_score",
    "tune_cliff",
    "tune_geometry",
    "validation_split",
    "Walks",
    "geometry_map",
    "geometry_walks",
    "folds",
    "geometry_prior",
    "loso",
    "one_seed_floor",
    "outside",
    "stale_at_seen",
    "stale_outside",
    "write_maps",
]
