"""Deep Crowd-Flow (Sohn et al., "Laying the Foundations of Deep Long-Term Crowd Flow Prediction", ECCV 2020; arXiv
1910.05810), retrained on a split's PEDS training runs with the goals at the demand objects.

**CAGE input** on the base world's flow grid. The paper's cell is one agent's diameter (0.6 m); the flow grid's 0.5 m is
the diameter of HumanSim's pedestrians (radius 0.25 m). Five channels, X = (Cx, Cy, A, G, E):
- **E**, the environment: 1 where no one can stand (outside the rooms, walls, furniture: no free 0.25 m sub-cell of the
  forecaster's free space, locked doors closed), 0 navigable;
- **A**, the agents: the people the reader places in the unedited base world, at the forecaster's 16 crowd seeds that no
  split uses, as a density per cell (at most 1). The paper's A holds the run's own initial positions; here it is the
  expected crowd, the information the learned forecaster gets too;
- **G**, the goals: the shortest-path distance from each cell to the nearest front of a demand object of the edited
  graph (where people use it), normalized to [0, 1], 1 where unreachable or without any goal;
- **Cx, Cy**, the capacities: 1 / the number of columns (rows) merged into a cell by the compression, 1 without it.

**Compression.** The paper's network takes a fixed n' = 112 cells per side. A smaller grid is padded (E = 1, G = 1). A
larger one is compressed by merging whole adjacent columns (rows) whose E is identical, which keeps navigability exactly
(the paper's consistency), the most alike pair first when none is identical, a merged line taking E's walls from either.
This is CAGE's merge restricted to full lines; the paper merges any rectangular region of equal visibility. A merged
cell's A and the target take the sum over its cells divided by its capacity, G the minimum; decoding copies a merged
cell's value back to each of its cells, as the paper decodes.

**Output.** The paper predicts one channel, the normalized log of how often agents appear in a cell over the whole run.
The flow grid also needs the heading, so the network predicts the same quantity per bin, 9 channels (8 headings and
standing): Y = log(1 + n) / log(1 + max n) of the run's appearances per cell and bin, n = pedestrian-seconds / 0.02 s
(the paper's time step). A forecast inverts it with the training runs' mean log(1 + max n).

**Network and training,** as the paper: SegNet without its 3 outermost encoder and decoder layers (3 x 3 convolutions,
encoder stages of 2, 3, 3, 3 convolutions with 128, 256, 512, 512 channels, batch norm, max pooling whose indices the
decoder unpools), a sigmoid instead of the softmax, He initialization, batch size 64, mean absolute error, 10 % of the
samples held out for validation. Two changes, both for a split's few training worlds:
- **updates, not epochs:** the paper trains 200 epochs over 12k samples, 37,500 updates; a split has tens of training
  worlds, so training keeps the paper's number of updates, and the weights of the epoch with the lowest validation error;
- **Adam (learning rate 1e-3) instead of SGD:** with the paper's SGD (fixed learning rate 0.01, momentum 0.9) the
  sigmoid saturates near 0 within a few hundred updates on the mostly empty targets and stays there (on the pilot,
  3,000 updates: outputs about 0.001 everywhere, error no better than predicting 0); the paper reports trying adaptive
  learning rates. `--optimizer sgd` trains the paper's way.

**Samples.** The paper's target is a scenario's crowd flow aggregated over its whole simulation. A sample here is one
training world: its CAGE input and its runs pooled over the partition's seeds (scaled to their mean pedestrian-seconds),
the aggregate of a longer simulation. Single runs do not work as samples: every seed of a world has the same input, so
the mean absolute error asks for the median over seeds per cell and bin, which is 0 wherever a 300 s run of a sparse
crowd leaves the cell empty more often than not, and the network learns to predict nothing (seen on the pilot).

The authors' code (github.com/SSSohn/LTCF) has no license, so this is a reimplementation from the paper.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from .. import split as SP
from .maps import BINS, FlowGrid

if TYPE_CHECKING:
    import numpy as np

    from .evaluate import Forecaster, Runs

FORMAT = "flowedit-deepflow/1"
#: the network's side (cells)
SIDE = 112
#: the paper's simulation step (s): appearances per pedestrian-second
STEP_S = 0.02
CHANNELS = ("Cx", "Cy", "A", "G", "E")
EPOCHS = 200
#: the paper's updates: 200 epochs over 12k training samples at batch 64; a split's ~100 runs keep the updates, not the epochs
STEPS = 200 * 12_000 // 64
BATCH = 64
LR = 0.01
MOMENTUM = 0.9
#: the learning rate with Adam
ADAM_LR = 1e-3
OPTIMIZERS = ("sgd", "adam")
VALIDATION = 0.1


# --- CAGE ------------------------------------------------------------------------------------------------------------


@attrs.frozen
class Lines:
    """How the grid's lines along one axis map onto the network's: `group[k]` the original lines merged into line k."""

    group: tuple[tuple[int, ...], ...]

    @property
    def size(self) -> int:
        return len(self.group)

    def index(self) -> np.ndarray:
        """For every original line, the network line it went to."""
        import numpy as np

        out = np.zeros(sum(len(g) for g in self.group), np.int64)
        for k, g in enumerate(self.group):
            out[list(g)] = k
        return out


def merge_lines(E: np.ndarray, side: int = SIDE) -> Lines:
    """Merge adjacent lines (rows of `E`) until at most `side` remain: identical ones first, then the most alike."""
    import numpy as np

    groups = [[i] for i in range(E.shape[0])]
    rows = [E[i].astype(bool) for i in range(E.shape[0])]
    while len(groups) > side:
        diff = np.array([int(np.count_nonzero(rows[k] != rows[k + 1])) for k in range(len(rows) - 1)])
        size = np.array([len(groups[k]) + len(groups[k + 1]) for k in range(len(rows) - 1)])
        k = int(np.lexsort((size, diff))[0])
        groups[k] = groups[k] + groups.pop(k + 1)
        rows[k] = rows[k] | rows.pop(k + 1)
    return Lines(tuple(tuple(g) for g in groups))


@attrs.frozen
class Cage:
    """A world's CAGE input (5, SIDE, SIDE) and how its grid maps onto it: rows (y) and columns (x) merged."""

    x: Any
    rows: Lines
    cols: Lines
    grid_shape: tuple[int, int]

    def encode(self, values: np.ndarray, how: str = "mean") -> np.ndarray:
        """(ny, nx, ...) on the grid to (SIDE, SIDE, ...): the sum over a merged cell divided by its capacity ('mean') or the minimum ('min'); padding 0."""
        import numpy as np

        ri, ci = self.rows.index(), self.cols.index()
        out_shape = (SIDE, SIDE, *values.shape[2:])
        if how == "min":
            out = np.full(out_shape, np.inf)
            np.minimum.at(out, (ri[:, None], ci[None, :]), values)
            return np.where(np.isfinite(out), out, 0.0)
        out = np.zeros(out_shape)
        np.add.at(out, (ri[:, None], ci[None, :]), values)
        cap = np.zeros((SIDE, SIDE))
        np.add.at(cap, (ri[:, None], ci[None, :]), 1.0)
        cap = cap.reshape(cap.shape + (1,) * (values.ndim - 2))
        return np.divide(out, cap, out=np.zeros_like(out), where=cap > 0)

    def decode(self, values: np.ndarray) -> np.ndarray:
        """(SIDE, SIDE, ...) back to the grid (ny, nx, ...): each cell takes its merged cell's value."""
        ri, ci = self.rows.index(), self.cols.index()
        return values[ri[:, None], ci[None, :]]


def cage(site: Any, grid: FlowGrid, settings: Mapping[str, Any], duration: float, people: Sequence[Any]) -> Cage:  # noqa: ANN401 - edit.Site, model.Person
    """The CAGE input of `site` on its base world's `grid`, with `people` (the base world's expected crowd)."""
    import numpy as np
    import scipy.sparse.csgraph

    from .model import _Paths
    from .scene import NAV_SUB, scene_of

    scene = scene_of(site, settings, duration, grid=grid, people=list(people))
    ny, nx = grid.ny, grid.nx
    free = np.asarray(scene.free, bool).reshape(ny, NAV_SUB, nx, NAV_SUB).any(axis=(1, 3)) & grid.room_cells()
    E = (~free).astype(float)
    A = np.zeros((ny, nx))
    for p in people:
        i, j, ok = grid.index(p.start[0], p.start[1])
        if bool(ok):
            A[int(i), int(j)] += p.weight
    paths = _Paths(scene)
    goals = sorted({paths.node(d.spot(0)) for d in scene.demands})
    G = np.ones((ny, nx))
    if goals:
        dist = scipy.sparse.csgraph.dijkstra(paths.graph, directed=False, indices=goals, min_only=True).reshape(ny, NAV_SUB, nx, NAV_SUB).min(axis=(1, 3))
        reach = np.isfinite(dist) & free
        if reach.any():
            G = np.where(reach, dist / max(float(dist[reach].max()), 1e-9), 1.0)
    rows, cols = merge_lines(E), merge_lines(E.T)
    c = Cage(None, rows, cols, (ny, nx))
    ri, ci = rows.index(), cols.index()
    Cy = np.ones((SIDE, SIDE))
    Cx = np.ones((SIDE, SIDE))
    for k, g in enumerate(rows.group):
        Cy[k, :] = 1.0 / len(g)
    for k, g in enumerate(cols.group):
        Cx[:, k] = 1.0 / len(g)
    Ec = np.ones((SIDE, SIDE))
    Ec[: rows.size, : cols.size] = 0.0
    np.maximum.at(Ec, (ri[:, None], ci[None, :]), E)
    Gc = np.ones((SIDE, SIDE))
    Gc[: rows.size, : cols.size] = c.encode(G, "min")[: rows.size, : cols.size]
    Ac = np.minimum(c.encode(A), 1.0)
    x = np.stack([Cx, Cy, Ac, Gc, Ec]).astype(np.float32)
    return Cage(x, rows, cols, (ny, nx))


def target(c: Cage, counts: np.ndarray) -> tuple[np.ndarray, float]:
    """A run's map as the network's target (BINS, SIDE, SIDE): normalized log appearances per merged cell and bin, and its log(1 + max n)."""
    import numpy as np

    n = c.encode(counts / STEP_S)
    top = float(np.log1p(n.max()))
    y = np.log1p(n) / top if top > 0 else np.zeros_like(n)
    return np.moveaxis(y, -1, 0).astype(np.float32), top


def decoded(c: Cage, y: np.ndarray, log_max: float, grid: FlowGrid) -> np.ndarray:
    """The network's output (BINS, SIDE, SIDE) as a flow map (ny, nx, BINS) in pedestrian-seconds, kept to the rooms."""
    import numpy as np

    n = np.expm1(np.clip(np.moveaxis(y, 0, -1), 0.0, 1.0) * log_max) * STEP_S
    return c.decode(n) * grid.room_cells()[:, :, None]


# --- network ---------------------------------------------------------------------------------------------------------


def segnet(inputs: int = len(CHANNELS), outputs: int = BINS) -> Any:  # noqa: ANN401 - torch.nn.Module
    """The paper's modified SegNet: encoder stages of 2, 3, 3, 3 convolutions (128, 256, 512, 512 channels) with max
    pooling, a mirrored decoder that unpools with the pooling indices, a sigmoid output; He-initialized."""
    import torch
    from torch import nn

    def block(c_in: int, c_out: int) -> list[nn.Module]:
        return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out, momentum=0.1), nn.ReLU(inplace=True)]

    class SegNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            enc = [(inputs, 128, 128), (128, 256, 256, 256), (256, 512, 512, 512), (512, 512, 512, 512)]
            dec = [(512, 512, 512, 512), (512, 512, 512, 256), (256, 256, 256, 128)]
            self.enc = nn.ModuleList(nn.Sequential(*[layer for a, b in zip(s[:-1], s[1:], strict=True) for layer in block(a, b)]) for s in enc)
            self.dec = nn.ModuleList(nn.Sequential(*[layer for a, b in zip(s[:-1], s[1:], strict=True) for layer in block(a, b)]) for s in dec)
            self.last = nn.Sequential(*block(128, 128), nn.Conv2d(128, outputs, 3, padding=1))
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            idx, sizes = [], []
            for stage in self.enc:
                x = stage(x)
                sizes.append(x.shape)
                x, i = nn.functional.max_pool2d(x, 2, 2, return_indices=True)
                idx.append(i)
            for stage, i, size in zip([*self.dec, self.last], reversed(idx), reversed(sizes), strict=True):
                x = nn.functional.max_unpool2d(x, i, 2, 2, output_size=size[-2:])
                x = stage(x)
            return torch.sigmoid(x)

    return SegNet()


# --- data, training, forecasts ---------------------------------------------------------------------------------------


@attrs.define
class Inputs:
    """CAGE inputs of a split's worlds, the base worlds' expected crowds cached."""

    sites: Forecaster
    scenario_of: dict[str, str | None]
    _cages: dict[str, Cage] = attrs.Factory(dict)

    @classmethod
    def of(cls, doc: Mapping[str, Any], sites: Forecaster) -> Inputs:
        return cls(sites, {b: src.get("scenario") for b, src in (doc.get("sources") or {}).items()})

    def __call__(self, world: str) -> Cage:
        if world not in self._cages:
            from .scene import base_people

            s = self.sites
            base = SP.base_of(world)
            site = s.site(world, self.scenario_of.get(base))
            key = (base, tuple(s.crowd_seeds))
            if key not in s._people:
                s._people[key] = base_people(s.site(base, self.scenario_of.get(base)), s.settings, s.crowd_seeds)
            self._cages[world] = cage(site, s.grid(base), s.settings, s.duration, s._people[key])
        return self._cages[world]


def samples(inputs: Inputs, runs: Runs, worlds: Sequence[str], seeds: Sequence[int]) -> list[tuple[str, tuple[int, ...], np.ndarray, np.ndarray, float]]:
    """(world, seeds, x, y, log max) per world of `worlds` with recorded runs at `seeds`: its runs pooled, scaled to their mean pedestrian-seconds."""
    import numpy as np

    out = []
    for w in worlds:
        have = tuple(runs.seeds(w, seeds))
        if not have:
            continue
        c = inputs(w)
        y, top = target(c, runs.pooled(w, have) * float(np.mean([runs.maps[(w, s)].sum() for s in have])))
        out.append((w, have, c.x, y, top))
    return out


def train(found: Sequence[tuple[str, tuple[int, ...], np.ndarray, np.ndarray, float]], steps: int | None = STEPS, epochs: int = EPOCHS, batch: int = BATCH, optimizer: str = "adam", seed: int = 0, device: str | None = None, log: Callable[[str], None] = print) -> tuple[Any, dict[str, Any]]:  # noqa: ANN401
    """The SegNet trained on `found` by SGD on the mean absolute error, 10 % of the samples held out for validation: for
    `steps` updates (whole epochs, at least the one that reaches it), or `epochs` epochs when `steps` is None."""
    import numpy as np
    import torch

    if not found:
        raise ValueError("nothing to train on: no recorded run of the partition")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(found))
    n_val = max(1, round(VALIDATION * len(found))) if len(found) >= 10 else 0
    val, tr = order[:n_val], order[n_val:]
    X = torch.from_numpy(np.stack([f[2] for f in found]))
    Y = torch.from_numpy(np.stack([f[3] for f in found]))
    model = segnet().to(dev)
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer: one of {OPTIMIZERS}, got {optimizer!r}")
    lr = LR if optimizer == "sgd" else ADAM_LR
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM) if optimizer == "sgd" else torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    history = []
    per_epoch = max(1, sum(1 for k in range(0, len(tr), batch) if len(tr[k : k + batch]) >= 2))
    if steps is not None:
        epochs = -(-steps // per_epoch)
    report = max(1, epochs // 10)
    best: tuple[float, int, dict[str, Any]] | None = None
    for epoch in range(epochs):
        model.train()
        perm = tr[rng.permutation(len(tr))]
        losses = []
        for k in range(0, len(perm), batch):
            idx = torch.from_numpy(perm[k : k + batch])
            if len(idx) < 2:
                continue
            out = model(X[idx].to(dev))
            loss = loss_fn(out, Y[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        vloss = None
        if len(val):
            model.eval()
            with torch.no_grad():
                vloss = loss_fn(model(X[torch.from_numpy(val)].to(dev)), Y[torch.from_numpy(val)].to(dev)).item()
        history.append((float(np.mean(losses)) if losses else None, vloss))
        if vloss is not None and (best is None or vloss < best[0]):
            best = (vloss, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if epoch % report == 0 or epoch == epochs - 1:
            log(f"deepflow: epoch {epoch + 1}/{epochs}, train MAE {history[-1][0]:.5f}" + (f", validation MAE {vloss:.5f}" if vloss is not None else ""))
    if best is not None:
        model.load_state_dict(best[2])
        log(f"deepflow: kept epoch {best[1] + 1}, the lowest validation MAE {best[0]:.5f}")
    meta = {
        "samples": len(found),
        "kept_epoch": None if best is None else best[1] + 1,
        "validation": [found[i][0] for i in val],
        "epochs": epochs,
        "steps": epochs * per_epoch,
        "batch": batch,
        "optimizer": optimizer,
        "lr": lr,
        "momentum": MOMENTUM if optimizer == "sgd" else None,
        "seed": seed,
        "device": dev,
        "log_max": float(np.mean([f[4] for f in found])),
        "train_mae": [h[0] for h in history],
        "validation_mae": [h[1] for h in history],
    }
    return model, meta


def save(path: Path, model: Any, meta: Mapping[str, Any]) -> None:  # noqa: ANN401
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "state": model.state_dict(), "meta": dict(meta)}, path)


def load(path: Path, device: str | None = None) -> tuple[Any, dict[str, Any]]:  # noqa: ANN401
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    doc = torch.load(path, map_location=dev, weights_only=False)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: format: expected {FORMAT!r}, got {doc.get('format')!r}")
    model = segnet().to(dev)
    model.load_state_dict(doc["state"])
    model.eval()
    return model, doc["meta"]


def forecast(model: Any, meta: Mapping[str, Any], c: Cage, grid: FlowGrid) -> np.ndarray:  # noqa: ANN401
    """A world's forecast map (ny, nx, BINS) from its CAGE input."""
    import torch

    dev = next(model.parameters()).device
    with torch.no_grad():
        y = model(torch.from_numpy(c.x[None]).to(dev))[0].cpu().numpy()
    return decoded(c, y, float(meta["log_max"]), grid)


# --- CLI -------------------------------------------------------------------------------------------------------------


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _cmd_train(args: argparse.Namespace) -> int:
    from .evaluate import Forecaster, Runs

    doc = SP.load(args.split)
    sites = Forecaster(SP.crowd(doc, args.settings), SP.duration_of(doc), worlds=args.worlds)
    runs = Runs.load(args.cells, sites.grid, planner=args.planner)
    inputs = Inputs.of(doc, sites)
    part = (args.partition or ["train"])[0]
    found = samples(inputs, runs, doc["partitions"].get(part) or [], SP.seeds_of(doc, part))
    _log(f"deepflow: {len(found)} worlds of {part}, each its runs pooled over seeds {SP.seeds_of(doc, part)}; CAGE sides " + ", ".join(sorted({f"{SP.base_of(w)} {inputs(w).rows.size}x{inputs(w).cols.size} of {inputs(w).grid_shape[0]}x{inputs(w).grid_shape[1]}" for w, *_ in found})))
    model, meta = train(found, steps=None if args.steps == 0 else args.steps, epochs=args.epochs, batch=args.batch, optimizer=args.optimizer, seed=args.seed, device=args.device, log=_log)
    meta.update({"split": str(args.split), "partition": part, "cells": str(args.cells), "crowd_seeds": list(sites.crowd_seeds)})
    save(args.out, model, meta)
    _log(f"wrote {args.out}")
    return 0


def _cmd_predict(args: argparse.Namespace, score_maps: Callable[..., Any]) -> int:  # noqa: ANN401
    import yaml

    from .baselines import write_maps
    from .evaluate import Forecaster, Runs

    doc = SP.load(args.split)
    sites = Forecaster(SP.crowd(doc, args.settings), SP.duration_of(doc), worlds=args.worlds)
    inputs = Inputs.of(doc, sites)
    model, meta = load(args.model, args.device)
    parts = args.partition or [p for p in SP.PARTITIONS if p != "train"]
    maps = {}
    for world in dict.fromkeys(w for p in parts for v in doc["partitions"].get(p) or [] for w in (SP.base_of(v), v)):
        c = inputs(world)
        maps[world] = forecast(model, meta, c, sites.grid(SP.base_of(world)))
        if args.plots is not None:
            from .plot import plot_world
            from .scene import scene_of

            base = SP.base_of(world)
            scene = scene_of(sites.site(world, inputs.scenario_of.get(base)), sites.settings, sites.duration, grid=sites.grid(base), people=[])
            plot_world(args.plots / f"{world}.png", sites.grid(base), None, maps[world], None, scene, title=f"{world}: Deep Crowd-Flow")
    write_maps(args.out, maps, sites.grid)
    _log(f"wrote {args.out}")
    if args.cells is not None:
        report = {"format": FORMAT, "model": str(args.model), "maps": str(args.out), "partitions": score_maps(doc, Runs.load(args.cells, sites.grid, planner=args.planner), maps, parts, "deep_crowd_flow")}
        if args.report is not None:
            args.report.write_text(yaml.safe_dump(report, sort_keys=False))
            _log(f"wrote {args.report}")
    return 0


def register(verbs: argparse._SubParsersAction, common: Callable[[argparse.ArgumentParser], None], score_maps: Callable[..., Any]) -> None:
    """Add `deepflow train|predict` to `flowedit forecast baseline`."""
    p = verbs.add_parser("deepflow", help="Deep Crowd-Flow (CAGE + SegNet): train on the split's runs, forecast its worlds", description="Deep Crowd-Flow retrained on the split's PEDS training runs, goals at the demand objects")
    sub = p.add_subparsers(dest="deepflow_verb", required=True, metavar="{train,predict}")
    q = sub.add_parser("train", help="train the network on a partition's runs (default train)")
    common(q)
    q.add_argument("--cells", type=Path, required=True, help="arena_evaluation's flow_cells.parquet of the split's benchmark")
    q.add_argument("--steps", type=int, default=STEPS, help=f"SGD updates (default {STEPS}: the paper's 200 epochs over 12k samples at batch 64); 0 trains --epochs epochs instead")
    q.add_argument("--epochs", type=int, default=EPOCHS, help=f"epochs when --steps is 0 (default {EPOCHS}, the paper's)")
    q.add_argument("--batch", type=int, default=BATCH, help=f"batch size (default {BATCH}, the paper's)")
    q.add_argument("--optimizer", choices=OPTIMIZERS, default="adam", help=f"adam (default): learning rate {ADAM_LR}; sgd: the paper's (learning rate {LR}, momentum {MOMENTUM}), which stays at the all-zero output on a split's few worlds")
    q.add_argument("--seed", type=int, default=0, help="seed of the initialization, the validation split and the batches")
    q.add_argument("--device", default=None, help="torch device (default cuda when available)")
    q.add_argument("--out", type=Path, required=True, help="model file to write")
    q.set_defaults(fn=_cmd_train)
    q = sub.add_parser("predict", help="forecast a split's worlds as a map file")
    common(q)
    q.add_argument("--model", type=Path, required=True, help="model file from `deepflow train`")
    q.add_argument("--device", default=None, help="torch device (default cuda when available)")
    q.add_argument("--out", type=Path, required=True, help="map file to write (parquet; `forecast score --predicted` reads it)")
    q.add_argument("--cells", type=Path, default=None, help="flow_cells.parquet to score the maps against, beside the floors")
    q.add_argument("--report", type=Path, default=None, help="YAML of the scores")
    q.add_argument("--plots", type=Path, default=None, help="directory for a PNG per world")
    q.set_defaults(fn=lambda a: _cmd_predict(a, score_maps))


__all__ = ["CHANNELS", "FORMAT", "SIDE", "Cage", "Inputs", "Lines", "cage", "decoded", "forecast", "load", "merge_lines", "register", "samples", "save", "segnet", "target", "train"]
