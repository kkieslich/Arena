"""Train the learned forecaster on a split's recorded single runs, and forecast with it.

Training reads arena_evaluation's `flow_cells.parquet`: one map per stage `<world>__s<seed>`. Every single run of the
partition (default train) is a sample: the world's raster (`raster.py`, graph and geometry only) in, the run's
pedestrian-seconds per cell and bin as Poisson counts out, so the network learns the expected flow from noisy single
seeds. Each step draws `BATCH` runs, each under a random D4 symmetry (`net.transform`). `fraction` keeps that share of
the partition's edits, picked by `seed`, whole chains at a time: an edit with its geometry-only controls; the base world
always stays.

A forecast is the mean over the eight symmetries of the inverse-transformed rates.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs

from .. import split as SP
from .evaluate import Forecaster as Sites
from .evaluate import Runs

if TYPE_CHECKING:
    import numpy as np

    from .raster import Raster

NET_FORMAT = "flowedit-forecast-net/1"
STEPS = 3000
BATCH = 8
LR = 2e-3
WEIGHT_DECAY = 1e-4
COARSE = 0.5


@attrs.define
class Rasters:
    """Rasters of a split's worlds under one crowd setting, cached per (world, demand)."""

    sites: Sites
    scenario_of: Mapping[str, str | None]
    _cache: dict[tuple[str, bool], Raster] = attrs.Factory(dict)

    @classmethod
    def of(cls, doc: Mapping[str, Any], settings: Mapping[str, Any], worlds: Path | None = None) -> Rasters:
        return cls(Sites(settings, SP.duration_of(doc), worlds=worlds), {base: src.get("scenario") for base, src in (doc.get("sources") or {}).items()})

    def __call__(self, world: str, demand: bool = True) -> Raster:
        from .raster import raster
        from .scene import base_people, scene_of

        key = (world, demand)
        if key not in self._cache:
            s = self.sites
            site = s.site(world, self.scenario_of.get(SP.base_of(world)))
            base = s.site(site.base, site.scenario)
            people_key = (site.base, tuple(s.crowd_seeds))
            if people_key not in s._people:
                s._people[people_key] = base_people(base, s.settings, s.crowd_seeds)
            scene = scene_of(site, s.settings, s.duration, grid=s.grid(site.base), people=s._people[people_key])
            self._cache[key] = raster(site, scene, s.settings, demand=demand)
        return self._cache[key]


def chains(doc: Mapping[str, Any], partition: str) -> list[list[str]]:
    """The partition's edits grouped into chains: each function-carrying edit with its geometry-only controls, every other
    edit alone; the base worlds left out."""
    worlds = [w for w in doc["partitions"].get(partition) or [] if not SP.is_base(w)]
    controls = doc.get("control_worlds") or {}
    of_control = {c: e for e, cs in controls.items() for c in cs}
    out: list[list[str]] = []
    for w in worlds:
        if w in of_control and of_control[w] in worlds:
            continue
        out.append([w, *(c for c in controls.get(w, []) if c in worlds)])
    return out


def pick(doc: Mapping[str, Any], partition: str, fraction: float, seed: int) -> list[str]:
    """The partition's base worlds and `fraction` of its edit chains (rounded, at least one), drawn by `seed`."""
    groups = chains(doc, partition)
    k = len(groups) if fraction >= 1 else max(1, round(fraction * len(groups)))
    chosen = sorted(random.Random(seed).sample(range(len(groups)), k))
    bases = [w for w in doc["partitions"].get(partition) or [] if SP.is_base(w)]
    return bases + [w for i in chosen for w in groups[i]]


@attrs.frozen
class Sample:
    world: str
    seed: int
    counts: Any  # (BINS, ny, nx)


def held_out_seeds(doc: Mapping[str, Any], base: str, partition: str = "train") -> set[int]:
    """The seeds the other partitions score worlds of `base` at: a training run of `base`'s worlds at one of them would be
    a crowd draw the scores see (the pilot's base world runs at the held-out edits' seeds as their reference)."""
    out: set[int] = set()
    for p in SP.PARTITIONS:
        if p != partition and any(SP.base_of(w) == base for w in doc["partitions"].get(p) or []):
            out |= set(SP.seeds_of(doc, p))
    return out


def samples(runs: Runs, worlds: Sequence[str], doc: Mapping[str, Any], partition: str = "train") -> list[Sample]:
    """Every recorded run of `worlds` at a seed no other partition scores their base world at: the partition's seeds and
    any extra base-world seeds the split recorded (the design's bases at more seeds than the edits)."""
    import numpy as np

    out = []
    for w in worlds:
        skip = held_out_seeds(doc, SP.base_of(w), partition)
        out.extend(Sample(w, s, np.moveaxis(m, -1, 0).astype(np.float32)) for (world, s), m in sorted(runs.maps.items()) if world == w and s not in skip)
    return out


def train(
    rasters: Rasters,
    found: Sequence[Sample],
    *,
    demand: bool = True,
    steps: int = STEPS,
    batch: int = BATCH,
    seed: int = 0,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Any, dict[str, Any]]:
    """The trained network and a summary (losses, time) of `steps` Adam steps of Poisson NLL on `found`."""
    import numpy as np
    import torch

    from .net import Forecaster, Tensors, poisson_nll, transform

    if not found:
        raise ValueError("nothing to train on: no recorded run of the chosen worlds")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tensors = {w: Tensors(rasters(w, demand), dev, demand) for w in sorted({s.world for s in found})}
    targets = [torch.as_tensor(s.counts, device=dev) for s in found]
    model = Forecaster(demand).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
    losses = []
    t0 = time.perf_counter()
    for step in range(steps):
        picks = [rng.randrange(len(found)) for _ in range(batch)]
        sym = [(bool(rng.getrandbits(1)), rng.randrange(4)) for _ in picks]
        rates = model([(tensors[found[i].world], f, k) for i, (f, k) in zip(picks, sym, strict=True)])
        loss = torch.stack([poisson_nll(r, transform(targets[i], f, k, bins=True), transform(tensors[found[i].world].mask, f, k), COARSE) for r, i, (f, k) in zip(rates, picks, sym, strict=True)]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if (step + 1) % max(1, steps // 10) == 0:
            log(f"train: step {step + 1}/{steps}, loss {np.mean(losses[-50:]):.4f}, {time.perf_counter() - t0:.0f} s")
    summary = {"steps": steps, "batch": batch, "seed": seed, "device": str(dev), "seconds": round(time.perf_counter() - t0, 1), "loss_first": round(float(np.mean(losses[:50])), 5), "loss_last": round(float(np.mean(losses[-50:])), 5)}
    return model, summary


def predict(model: Any, raster: Raster, device: str | None = None) -> np.ndarray:  # noqa: ANN401 - net.Forecaster
    """(ny, nx, BINS) expected pedestrian-seconds of `raster`: the mean over the eight D4 symmetries."""
    import numpy as np
    import torch

    from .net import Tensors, untransform

    dev = next(model.parameters()).device if device is None else torch.device(device)
    t = Tensors(raster, dev, model.demand)
    model.eval()
    with torch.no_grad():
        syms = [(f, k) for f in (False, True) for k in range(4)]
        rates = model([(t, f, k) for f, k in syms])
        mean = torch.stack([untransform(r, f, k, bins=True) for r, (f, k) in zip(rates, syms, strict=True)]).mean(0)
    return np.moveaxis(mean.cpu().numpy().astype(np.float64), 0, -1)


def save(path: Path, model: Any, meta: Mapping[str, Any]) -> None:  # noqa: ANN401
    import torch

    torch.save({"format": NET_FORMAT, "demand": model.demand, "state": model.state_dict(), "meta": dict(meta)}, path)


def load(path: Path, device: str | None = None) -> tuple[Any, dict[str, Any]]:
    import torch

    from .net import Forecaster

    doc = torch.load(path, map_location=device or "cpu", weights_only=False)
    if doc.get("format") != NET_FORMAT:
        raise ValueError(f"{path}: format: expected {NET_FORMAT!r}, got {doc.get('format')!r}")
    model = Forecaster(bool(doc["demand"]))
    model.load_state_dict(doc["state"])
    if device:
        model = model.to(device)
    return model, dict(doc.get("meta") or {})


def jsd_coarse(p: np.ndarray, q: np.ndarray, factor: int = 2) -> float | None:
    """The JSD of two maps summed over factor x factor blocks of cells and over their headings: occupancy at 1 m."""
    import numpy as np

    from .maps import jsd

    def coarse(m: np.ndarray) -> np.ndarray:
        occ = m.sum(-1)
        ny, nx = occ.shape
        occ = np.pad(occ, ((0, (-ny) % factor), (0, (-nx) % factor)))
        return occ.reshape(occ.shape[0] // factor, factor, occ.shape[1] // factor, factor).sum(axis=(1, 3))[..., None]

    return jsd(coarse(p), coarse(q))


__all__ = ["NET_FORMAT", "Rasters", "Sample", "chains", "held_out_seeds", "jsd_coarse", "load", "pick", "predict", "samples", "save", "train"]
