"""The demand model's fit on a split's recorded flow maps, the recorded maps themselves, and the transplanted baseline.

Recorded maps come from arena_evaluation's `flow_cells.parquet` (one map per stage `<world>__s<seed>`, on the base world's
grid). The target of a world is its seed-pooled map (the mean of its normalized maps over the partition's seeds), the score
the JSD in bits against it, as `process_benchmark` pools them. Scoring against the baselines is `validity.py`.

- **transplanted**: a base map with the mass within `TRANSPLANT_M` of the edited object moved with it (`move`, headings
  turned by its yaw change), copied to the new object (`insert`, from the object it copies) or dropped (`remove`); other
  edits leave it stale.
- **anchored** forecast: a recorded base map plus the demand model's change (forecast of the edit minus forecast of the
  base). It reads recorded flow, so it is not an H1 forecaster.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from .. import split as SP
from .maps import HEADINGS, STANDING, FlowGrid, jsd, normalized, pooled
from .model import Layers, Params, layers, predict
from .scene import CROWD_SEEDS, base_people, scene_of

if TYPE_CHECKING:
    import numpy as np

    from ..edit import Site

#: radius of the mass a transplant moves with its object (m)
TRANSPLANT_M = 2.0
#: CELL_SCHEMA columns of a flow_cells.parquet
CELL_COLUMNS = ("planner", "stage", "flow_world", "flow_base", "flow_seed", "x", "y", "cell_m", "heading_rad", "ped_seconds")


@attrs.define
class Runs:
    """Recorded flow maps by (world, seed), each on its base world's grid."""

    maps: dict[tuple[str, int], Any]

    @classmethod
    def load(cls, path: Path, grid_of: Callable[[str], FlowGrid], planner: str | None = None) -> Runs:
        """The maps of a `flow_cells.parquet`; `planner` picks one contestant (needed when it holds several)."""
        import polars as pl

        cells = pl.read_parquet(path)
        missing = set(CELL_COLUMNS) - set(cells.columns)
        if missing:
            raise ValueError(f"{path}: not a flow_cells.parquet, no columns {sorted(missing)}")
        planners = sorted(cells["planner"].unique().to_list())
        if planner is None and len(planners) > 1:
            raise ValueError(f"{path}: contestants {planners}, pick one with --planner")
        if planner is not None:
            cells = cells.filter(pl.col("planner") == planner)
        maps = {}
        for (world, base, seed), part in cells.group_by(["flow_world", "flow_base", "flow_seed"], maintain_order=True):
            grid = grid_of(str(base))
            maps[(str(world), int(seed))] = grid.dense(part["x"].to_numpy(), part["y"].to_numpy(), part["heading_rad"].to_numpy(), part["ped_seconds"].to_numpy())
        return cls(maps)

    def pooled(self, world: str, seeds: Iterable[int]) -> np.ndarray | None:
        """The world's pooled map over those of `seeds` it has, None without any."""
        found = [self.maps[(world, s)] for s in seeds if (world, s) in self.maps]
        return pooled(found) if found else None

    def seeds(self, world: str, seeds: Iterable[int]) -> list[int]:
        return [s for s in seeds if (world, s) in self.maps]


@attrs.define
class Forecaster:
    """Forecasts of a split's worlds under one crowd setting: base sites, grids and people cached per base world, layers per world."""

    settings: Mapping[str, Any]
    duration: float
    worlds: Path | None = None
    crowd_seeds: tuple[int, ...] = CROWD_SEEDS
    _sites: dict[str, Site] = attrs.Factory(dict)
    _grids: dict[str, FlowGrid] = attrs.Factory(dict)
    _people: dict[tuple[str, tuple[int, ...]], list] = attrs.Factory(dict)
    _layers: dict[tuple[str, tuple[int, ...]], Layers] = attrs.Factory(dict)

    def site(self, world: str, scenario: str | None = None) -> Site:
        """The site of `world`, a base world with `scenario` (the split's source scenario)."""
        from ...scenario.cli import world_root
        from ..edit import open_site

        if world not in self._sites:
            self._sites[world] = open_site(world_root(self.worlds, world), scenario if SP.is_base(world) else None)
        return self._sites[world]

    def grid(self, base: str) -> FlowGrid:
        if base not in self._grids:
            from ...scenario.cli import world_root

            self._grids[base] = FlowGrid.from_world(world_root(self.worlds, base))
        return self._grids[base]

    def layers(self, world: str, scenario: str | None = None, crowd_seeds: Sequence[int] | None = None) -> Layers:
        """The layers of `world`, with the people of its base world drawn at `crowd_seeds` (default: this forecaster's)."""
        seeds = tuple(crowd_seeds or self.crowd_seeds)
        key = (world, seeds)
        if key not in self._layers:
            site = self.site(world, scenario)
            base = self.site(site.base, site.scenario)
            if (site.base, seeds) not in self._people:
                self._people[(site.base, seeds)] = base_people(base, self.settings, seeds)
            scene = scene_of(site, self.settings, self.duration, grid=self.grid(site.base), people=self._people[(site.base, seeds)])
            self._layers[key] = layers(scene)
        return self._layers[key]

    def predict(self, world: str, params: Params, scenario: str | None = None, crowd_seeds: Sequence[int] | None = None) -> np.ndarray:
        return predict(self.layers(world, scenario, crowd_seeds), params)


def anchored(base_map: np.ndarray, forecast_edit: np.ndarray, forecast_base: np.ndarray) -> np.ndarray:
    """The recorded base map plus the forecast change, as distributions, negative cells cut to zero."""
    import numpy as np

    a, e, b = normalized(base_map), normalized(forecast_edit), normalized(forecast_base)
    if a is None or e is None or b is None:
        return base_map
    return np.maximum(0.0, a + e - b) * float(base_map.sum())


# --- transplanted baseline -------------------------------------------------------------------------------------------


def _pose(site: Site, obj: str) -> tuple[float, float, float] | None:
    """Map-frame (x, y, yaw) of `obj` in the site's graph."""
    from .. import read as R

    fact = next((f for f in site.graph.facts if f.subject == obj and f.verb == "at"), None)
    if fact is None:
        return None
    p = dict(fact.params)
    ox, oy = R._origins(site.root, site.world).get(str(fact.object).partition(":")[2], (0.0, 0.0))
    return (float(p["x"]) + ox, float(p["y"]) + oy, float(p.get("yaw", 0.0)))


def _copied(base: Site, site: Site, obj: str) -> str | None:
    """The base object an inserted `obj` copies: `<name>` of `<name>_inert`, else a base object of its model, one with a need first."""
    if obj.endswith("_inert") and obj[: -len("_inert")] in base.graph.entities:
        return obj[: -len("_inert")]
    model = (site.graph.entities.get(obj).props if obj in site.graph.entities else {}).get("model")
    same = sorted(e for e, ent in base.graph.entities.items() if e.startswith("obj:") and ent.props.get("model") == model)
    needy = {f.subject for f in base.graph.facts if f.verb == "satisfies"}
    return next((e for e in same if e in needy), next(iter(same), None))


def transplant_moves(base: Site, site: Site) -> list[tuple[tuple[float, float, float], tuple[float, float, float] | None, str]]:
    """What the transplant does for `site`'s last edit: (old pose, new pose or None, 'move' / 'copy' / 'drop'), none for other edits."""
    if not site.edits:
        return []
    edit = site.edits[-1]
    obj = edit.target
    kind = edit.type.value
    if kind == "move":
        old, new = _pose(base, obj), _pose(site, obj)
        return [(old, new, "move")] if old and new else []
    if kind == "insert":
        source = _copied(base, site, obj)
        old, new = (_pose(base, source) if source else None), _pose(site, obj)
        return [(old, new, "copy")] if old and new else []
    if kind == "remove":
        old = _pose(base, obj)
        return [(old, None, "drop")] if old else []
    return []


def transplanted(base_map: np.ndarray, grid: FlowGrid, moves: Sequence[tuple[tuple[float, float, float], tuple[float, float, float] | None, str]]) -> np.ndarray:
    """`base_map` with the mass within TRANSPLANT_M of each old pose moved, copied or dropped, kept to the room cells."""
    import numpy as np

    out = base_map.copy()
    xs, ys = grid.centres()
    rooms = grid.room_cells()
    for old, new, mode in moves:
        near = np.hypot(xs - old[0], ys - old[1]) <= TRANSPLANT_M
        moved = base_map[near]
        if mode in ("move", "drop"):
            out[near] = 0.0
        if new is None:
            continue
        turn = math.remainder(new[2] - old[2], math.tau)
        c, s = math.cos(turn), math.sin(turn)
        rx, ry = xs[near] - old[0], ys[near] - old[1]
        i, j, ok = grid.index(new[0] + c * rx - s * ry, new[1] + s * rx + c * ry)
        shift = round(turn / (2 * math.pi / HEADINGS))
        order = [(b - shift) % HEADINGS for b in range(HEADINGS)] + [STANDING]
        np.add.at(out, (i[ok], j[ok]), moved[ok][:, order])
    out *= rooms[:, :, None]
    total = float(out.sum())
    return out * (float(base_map.sum()) / total) if total > 0 else out


# --- fit and score ---------------------------------------------------------------------------------------------------


@attrs.frozen
class Target:
    """A world to fit or score: its layers (and its base world's, for the anchored forecast), its pooled recorded map and seeds."""

    world: str
    layers: Layers
    truth: Any
    seeds: tuple[int, ...]


def targets(forecaster: Forecaster, runs: Runs, doc: Mapping[str, Any], partition: str, crowd: str = "expected", log: Callable[[str], None] = print) -> list[Target]:
    """Every world of `partition` with recorded maps at its seeds, base worlds included; `crowd` 'runs' draws the people at those seeds."""
    seeds = SP.seeds_of(doc, partition)
    scenario_of = {base: src.get("scenario") for base, src in (doc.get("sources") or {}).items()}
    out = []
    for world in doc["partitions"].get(partition) or []:
        have = runs.seeds(world, seeds)
        if not have:
            log(f"{world}: no recorded map at seeds {seeds}, left out")
            continue
        crowd_seeds = tuple(have) if crowd == "runs" else None
        out.append(Target(world, forecaster.layers(world, scenario_of.get(SP.base_of(world)), crowd_seeds), runs.pooled(world, have), tuple(have)))
    return out


def objective(params: Params, found: Sequence[Target]) -> float:
    """Mean JSD of the forecasts against the pooled maps."""
    import numpy as np

    return float(np.mean([jsd(predict(t.layers, params), t.truth) for t in found]))


def fit(found: Sequence[Target], init: Params | None = None, max_evals: int = 400, log: Callable[[str], None] = print) -> tuple[Params, float]:
    """The params minimising the mean pooled JSD over `found`: Nelder-Mead in log space from `init` (default the built-in params), within Params.BOUNDS."""
    init = init or Params()
    import numpy as np
    import scipy.optimize

    if not found:
        raise ValueError("nothing to fit: no world with recorded maps")
    x0 = np.log(init.vector())
    simplex = np.vstack([x0, *(x0 + 0.7 * np.eye(len(x0)))])
    evals = [0]

    def f(z: np.ndarray) -> float:
        evals[0] += 1
        value = objective(Params.from_vector(np.exp(z)), found)
        if evals[0] % 50 == 0:
            log(f"fit: {evals[0]} evaluations, mean JSD {value:.4f}")
        return value

    res = scipy.optimize.minimize(f, x0, method="Nelder-Mead", options={"initial_simplex": simplex, "maxfev": max_evals, "xatol": 1e-3, "fatol": 1e-5})
    best = Params.from_vector(np.exp(res.x))
    return best, objective(best, found)


def label(doc: Mapping[str, Any], world: str) -> str:
    """What a world of a split is: base, an edit (`<type>_<target>`), or a control of one."""
    if SP.is_base(world):
        return "base"
    controls = {c for cs in (doc.get("control_worlds") or {}).values() for c in cs}
    parts = world.split("__")
    what = parts[1] if len(parts) >= 3 else world
    return f"{what} (control)" if world in controls else what


__all__ = ["Forecaster", "Runs", "Target", "anchored", "fit", "label", "objective", "targets", "transplant_moves", "transplanted"]
