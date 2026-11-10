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
decoder unpools), a sigmoid instead of the softmax, He initialization, batch size 64, mean absolute error. Changes, all
for a split's few training worlds where the paper had 12k scenarios:
- **Adam instead of SGD:** with the paper's SGD (fixed learning rate 0.01, momentum 0.9) the sigmoid saturates near 0
  within a few hundred updates on the mostly empty targets and stays there (on the pilot: outputs about 0.001
  everywhere, error no better than predicting 0); the paper reports trying adaptive learning rates. `--optimizer sgd`
  trains the paper's way;
- **the tuning rule picks the learning rate and the training length** (the paper picks its settings on a 10 %
  validation split): `deepflow tune` trains on the train partition without its validation worlds
  (`baselines.validation_split`, a fixed 20 % of its edit chains) at each learning rate of TUNE_LRS and scores the
  validation worlds' forecasts (mean pooled JSD against their pooled runs) at each length of TUNE_STEPS, up to the
  paper's 37,500 updates (200 epochs over 12k samples); `deepflow train --tuned` takes the lowest score (ties: fewer
  updates, then the lower rate) and trains on the whole train partition.

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
#: the paper's updates: 200 epochs over 12k training samples at batch 64
STEPS = 200 * 12_000 // 64
BATCH = 64
LR = 0.01
MOMENTUM = 0.9
#: the learning rate with Adam
ADAM_LR = 1e-3
OPTIMIZERS = ("sgd", "adam")
TUNE_FORMAT = "flowedit-deepflow-tune/1"
#: the tuning rule's grid: learning rates (Adam) and training lengths (updates; the last is the paper's)
TUNE_LRS = (1e-4, 3e-4, 1e-3)
TUNE_STEPS = (1000, 2000, 4000, 8000, 16000, STEPS)


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


def train(
    found: Sequence[tuple[str, tuple[int, ...], np.ndarray, np.ndarray, float]],
    steps: int = STEPS,
    lr: float | None = None,
    optimizer: str = "adam",
    batch: int = BATCH,
    seed: int = 0,
    device: str | None = None,
    checkpoints: Sequence[int] = (),
    on_checkpoint: Callable[[int, Any], None] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Any, dict[str, Any]]:  # noqa: ANN401
    """The SegNet trained on every sample of `found` for `steps` updates on the mean absolute error, at learning rate `lr`
    (default: ADAM_LR with Adam, LR with SGD); `on_checkpoint(step, model)` runs after each update in `checkpoints`."""
    import numpy as np
    import torch

    if not found:
        raise ValueError("nothing to train on: no recorded run of the partition")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer: one of {OPTIMIZERS}, got {optimizer!r}")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(np.stack([f[2] for f in found])).to(dev)
    Y = torch.from_numpy(np.stack([f[3] for f in found])).to(dev)
    model = segnet().to(dev)
    lr = lr if lr is not None else (LR if optimizer == "sgd" else ADAM_LR)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM) if optimizer == "sgd" else torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.L1Loss()
    marks = set(checkpoints)
    step, losses, history = 0, [], []
    report = max(1, steps // 10)
    model.train()
    while step < steps:
        perm = rng.permutation(len(found))
        for k in range(0, len(perm), batch):
            idx = torch.from_numpy(perm[k : k + batch]).to(dev)
            loss = loss_fn(model(X[idx]), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(loss.item())
            if step % report == 0 or step == steps:
                history.append((step, float(np.mean(losses))))
                log(f"deepflow: update {step}/{steps}, train MAE {history[-1][1]:.5f}")
                losses = []
            if step in marks and on_checkpoint is not None:
                model.eval()
                on_checkpoint(step, model)
                model.train()
            if step >= steps:
                break
    model.eval()
    meta = {
        "samples": len(found),
        "worlds": [f[0] for f in found],
        "steps": steps,
        "batch": batch,
        "optimizer": optimizer,
        "lr": lr,
        "momentum": MOMENTUM if optimizer == "sgd" else None,
        "seed": seed,
        "device": dev,
        "log_max": float(np.mean([f[4] for f in found])),
        "train_mae": [list(h) for h in history],
    }
    return model, meta


def tune(inputs: Inputs, runs: Runs, doc: Mapping[str, Any], lrs: Sequence[float] = TUNE_LRS, checkpoints: Sequence[int] = TUNE_STEPS, optimizer: str = "adam", seed: int = 0, device: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The tuning rule's runs: per learning rate, the network trained on the train partition's training worlds
    (`baselines.validation_split`), and at each checkpoint the mean pooled JSD of its forecasts of the validation worlds."""
    from .baselines import map_score, validation_split

    training, validation = validation_split(doc)
    seeds = SP.seeds_of(doc, "train")
    found = samples(inputs, runs, training, seeds)
    scores: dict[float, dict[int, float | None]] = {}
    for lr in lrs:
        scores[lr] = {}
        log_max = float(sum(f[4] for f in found) / len(found))

        def check(step: int, model: Any, lr: float = lr) -> None:  # noqa: ANN401
            meta = {"log_max": log_max}
            value = map_score(runs, validation, seeds, lambda w: forecast(model, meta, inputs(w), inputs.sites.grid(SP.base_of(w))))
            scores[lr][step] = value
            log(f"deepflow tuning: lr {lr:g}, {step} updates: validation JSD {value:.4f}")

        train(found, steps=max(checkpoints), lr=lr, optimizer=optimizer, seed=seed, device=device, checkpoints=checkpoints, on_checkpoint=check, log=log)
    return {"format": TUNE_FORMAT, "optimizer": optimizer, "seed": seed, "validation": validation, "training": training, "scores": {float(lr): {int(k): v for k, v in by.items()} for lr, by in scores.items()}}


def chosen(tuned: Sequence[Mapping[str, Any]]) -> tuple[float, int, float, str]:
    """(learning rate, updates, validation JSD, optimizer) with the lowest validation JSD over tuning files; ties take fewer updates, then the lower rate."""
    best: tuple[float, int, float, str] | None = None
    validation = None
    for doc in tuned:
        if doc.get("format") != TUNE_FORMAT:
            raise ValueError(f"not a tuning file: format {doc.get('format')!r}, expected {TUNE_FORMAT!r}")
        if validation is not None and doc["validation"] != validation:
            raise ValueError("the tuning files validate on different worlds")
        validation = doc["validation"]
        for lr, by in doc["scores"].items():
            for step, value in by.items():
                if value is None:
                    continue
                key = (float(value), int(step), float(lr))
                if best is None or key < (best[2], best[1], best[0]):
                    best = (float(lr), int(step), float(value), str(doc["optimizer"]))
    if best is None:
        raise ValueError("no validation score in the tuning files")
    return best


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


def _setup(args: argparse.Namespace) -> tuple[dict[str, Any], Any, Inputs]:  # noqa: ANN401
    from .evaluate import Forecaster, Runs

    doc = SP.load(args.split)
    sites = Forecaster(SP.crowd(doc, args.settings), SP.duration_of(doc), worlds=args.worlds)
    return doc, Runs.load(args.cells, sites.grid, planner=args.planner), Inputs.of(doc, sites)


def _cmd_tune(args: argparse.Namespace) -> int:
    import yaml

    doc, runs, inputs = _setup(args)
    out = tune(inputs, runs, doc, lrs=args.lr or TUNE_LRS, optimizer=args.optimizer, seed=args.seed, device=args.device, log=_log)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(out, sort_keys=False))
    _log(f"wrote {args.out}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    import yaml

    from .evaluate import listed

    doc, runs, inputs = _setup(args)
    lr, steps, optimizer, picked = args.lr, args.steps, args.optimizer, None
    if args.tuned:
        lr, steps, value, optimizer = chosen([yaml.safe_load(t.read_text()) for t in args.tuned])
        picked = {"files": [str(t) for t in args.tuned], "lr": lr, "steps": steps, "validation_jsd": value}
        _log(f"deepflow: the tuning rule chose {optimizer} at learning rate {lr:g} for {steps} updates (validation JSD {value:.4f})")
    found = samples(inputs, runs, doc["partitions"].get("train") or [], SP.seeds_of(doc, "train"))
    _log(f"deepflow: {len(found)} worlds of train, each its runs pooled; CAGE sides " + ", ".join(sorted({f"{SP.base_of(w)} {inputs(w).rows.size}x{inputs(w).cols.size} of {inputs(w).grid_shape[0]}x{inputs(w).grid_shape[1]}" for w, *_ in found})))
    model, meta = train(found, steps=steps, lr=lr, optimizer=optimizer, batch=args.batch, seed=args.seed, device=args.device, log=_log)
    meta.update({"split": str(args.split), "cells": listed(args.cells), "crowd_seeds": list(inputs.sites.crowd_seeds), "tuned": picked})
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
    from .evaluate import CELLS_HELP, SEVERAL

    p = verbs.add_parser("deepflow", help="Deep Crowd-Flow (CAGE + SegNet): train on the split's runs, forecast its worlds", description="Deep Crowd-Flow retrained on the split's PEDS training runs, goals at the demand objects")
    sub = p.add_subparsers(dest="deepflow_verb", required=True, metavar="{tune,train,predict}")

    def training(q: argparse.ArgumentParser) -> None:
        common(q)
        q.add_argument("--cells", type=Path, action="append", required=True, help=CELLS_HELP)
        q.add_argument("--optimizer", choices=OPTIMIZERS, default="adam", help=f"adam (default, learning rate {ADAM_LR}); sgd: the paper's (learning rate {LR}, momentum {MOMENTUM}), which stays at the all-zero output on a split's few worlds")
        q.add_argument("--seed", type=int, default=0, help="seed of the initialization and the batches")
        q.add_argument("--device", default=None, help="torch device (default cuda when available)")

    q = sub.add_parser("tune", help="the tuning rule's runs: validation JSD per learning rate and training length", description=f"train on the train partition's training worlds and score the validation worlds (a fixed 20 %% of its edit chains) at {', '.join(map(str, TUNE_STEPS))} updates")
    training(q)
    q.add_argument("--lr", type=float, action="append", default=None, help=f"a learning rate to try; repeatable (default {', '.join(f'{v:g}' for v in TUNE_LRS)}); run one per process to use several GPU claims")
    q.add_argument("--out", type=Path, required=True, help="tuning file to write (YAML)")
    q.set_defaults(fn=_cmd_tune)
    q = sub.add_parser("train", help="train the network on the train partition, at the tuned or the given settings")
    training(q)
    q.add_argument("--tuned", type=Path, action="append", default=None, help="tuning files from `deepflow tune`: train at the learning rate and length with the lowest validation JSD; repeatable")
    q.add_argument("--steps", type=int, default=STEPS, help=f"updates without --tuned (default {STEPS}, the paper's)")
    q.add_argument("--lr", type=float, default=None, help=f"learning rate without --tuned (default {ADAM_LR} with adam, {LR} with sgd)")
    q.add_argument("--batch", type=int, default=BATCH, help=f"batch size (default {BATCH}, the paper's)")
    q.add_argument("--out", type=Path, required=True, help="model file to write")
    q.set_defaults(fn=_cmd_train)
    q = sub.add_parser("predict", help="forecast a split's worlds as a map file")
    common(q)
    q.add_argument("--model", type=Path, required=True, help="model file from `deepflow train`")
    q.add_argument("--device", default=None, help="torch device (default cuda when available)")
    q.add_argument("--out", type=Path, required=True, help="map file to write (parquet; `forecast score --predicted` reads it)")
    q.add_argument("--cells", type=Path, action="append", default=None, help=f"flow_cells.parquet to score the maps against, beside the floors; {SEVERAL}")
    q.add_argument("--report", type=Path, default=None, help="YAML of the scores")
    q.add_argument("--plots", type=Path, default=None, help="directory for a PNG per world")
    q.set_defaults(fn=lambda a: _cmd_predict(a, score_maps))


__all__ = ["CHANNELS", "FORMAT", "SIDE", "TUNE_LRS", "TUNE_STEPS", "Cage", "Inputs", "Lines", "cage", "chosen", "decoded", "forecast", "load", "merge_lines", "register", "samples", "save", "segnet", "target", "train", "tune"]
