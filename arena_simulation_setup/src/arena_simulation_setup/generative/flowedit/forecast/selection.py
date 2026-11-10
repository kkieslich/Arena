"""Which network is the H1 forecaster: the plain one or the one with the demand model's forecast as input
(`train --demand-model-input`), chosen by leave-one-world-out validation on the training worlds alone.

A training world is a base world of the train partition with its edits there. For each of them (a fold), both arms train
on the other training worlds' runs and are scored on the held-out world's base and edits, the pooled JSD at their train
seeds. The arm with the lower mean over folds is the forecaster; `score` and `curve` then call it "forecaster" and the
other arm by its own name (`--select`).

The demand model, whose forecast the second arm reads, is refitted on each fold's training worlds, so no fold sees the
held-out world through its six parameters; `params` instead reuses one params file for every fold and says so.

Only the train partition's worlds at their train seeds are read: the recorded maps are cut to them before anything else,
so neither the test partitions nor the base worlds' reference runs at the test seeds can reach a fold.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .. import split as SP
from .evaluate import Runs

if TYPE_CHECKING:
    from .model import Params

SELECT_FORMAT = "flowedit-forecast-select/1"
PLAIN = "plain"
DEMAND_MODEL_INPUT = "demand_model_input"
ARMS = (PLAIN, DEMAND_MODEL_INPUT)
#: what `score` and `curve` call the arm that was not chosen
ARM_LABELS = {PLAIN: "plain network", DEMAND_MODEL_INPUT: "network + demand model input"}


def arm_of(model: Any) -> str:  # noqa: ANN401 - net.Forecaster
    """The arm a trained network belongs to."""
    return PLAIN if getattr(model, "demand_model", None) is None else DEMAND_MODEL_INPUT


def arm_label(model: Any, chosen: str | None) -> str:  # noqa: ANN401 - net.Forecaster
    """What a network is called: the demand ablation; "forecaster" for the chosen arm (for the plain arm without a choice);
    the other arm by its own name."""
    if not getattr(model, "demand", True):
        return "demand ablation"
    arm = arm_of(model)
    if chosen is None:
        return "forecaster" if arm == PLAIN else "forecaster + demand model"
    return "forecaster" if arm == chosen else ARM_LABELS[arm]


def training_worlds(doc: Mapping[str, Any]) -> dict[str, list[str]]:
    """The train partition's worlds grouped by base world: one group per training world, in the split's order."""
    groups: dict[str, list[str]] = {}
    for world in doc["partitions"].get("train") or []:
        groups.setdefault(SP.base_of(world), []).append(world)
    return groups


def train_runs(runs: Runs, doc: Mapping[str, Any]) -> Runs:
    """`runs` cut to the train partition's worlds at their own train seeds: all a selection may read."""
    from .validity import world_seeds

    keep = {w: set(world_seeds(doc, "train", w)) for w in doc["partitions"].get("train") or []}
    return Runs({(w, s): m for (w, s), m in runs.maps.items() if w in keep and s in keep[w]})


def select(
    rasters: Any,  # noqa: ANN401 - learn.Rasters
    runs: Runs,
    doc: Mapping[str, Any],
    *,
    params: Params | None = None,
    fit_params: Callable[[Sequence[str], Runs], Params] | None = None,
    seeds: Sequence[int] = (0,),
    steps: int = 3000,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Leave-one-world-out over the training worlds for both arms: per fold and arm the mean pooled JSD over the held-out
    world's base and edits (and over `seeds`), the mean over folds, and the chosen arm. The demand model's params come
    from `params` for every fold, else from `fit_params(fold training worlds, runs)`."""
    import numpy as np

    from .learn import forecast, samples, train
    from .maps import jsd
    from .validity import world_seeds

    if params is None and fit_params is None:
        raise ValueError("select needs the demand model's params or a way to fit them per fold")
    groups = training_worlds(doc)
    if len(groups) < 2:
        raise ValueError(f"leave-one-world-out needs at least 2 training worlds, the train partition has {sorted(groups)}")
    runs = train_runs(runs, doc)
    folds = []
    for held, worlds in groups.items():
        others = [w for base, ws in groups.items() if base != held for w in ws]
        found = samples(runs, others, doc)
        truths = {w: m for w in worlds if (m := runs.pooled(w, world_seeds(doc, "train", w))) is not None}
        fold_params = params if params is not None else fit_params(others, runs)  # type: ignore[misc]
        log(f"select: fold {held}: {len(found)} training runs of {len(others)} worlds, {len(truths)} held-out worlds")
        arms: dict[str, Any] = {}
        for arm in ARMS:
            per_world: dict[str, list[float]] = {w: [] for w in truths}
            for seed in seeds:
                model, _ = train(rasters, found, seed=seed, steps=steps, device=device, demand_model=fold_params if arm == DEMAND_MODEL_INPUT else None, log=log)
                for w, truth in truths.items():
                    value = jsd(forecast(model, rasters, w), truth)
                    if value is not None:
                        per_world[w].append(float(value))
            means = {w: float(np.mean(v)) for w, v in per_world.items() if v}
            arms[arm] = {"mean": float(np.mean(list(means.values()))) if means else None, "per_world": means}
            log(f"select: fold {held}, {ARM_LABELS[arm]}: {arms[arm]['mean']}")
        folds.append({"held_out": held, "worlds": len(truths), "training_runs": len(found), "params": "given" if params is not None else "refitted", "demand_model": fold_params.dump(), "arms": arms})
    mean = {arm: float(np.mean([f["arms"][arm]["mean"] for f in folds if f["arms"][arm]["mean"] is not None])) for arm in ARMS}
    chosen = min(ARMS, key=lambda arm: (mean[arm], ARMS.index(arm)))
    out: dict[str, Any] = {"format": SELECT_FORMAT, "chosen": chosen, "chosen_label": ARM_LABELS[chosen], "mean": mean, "seeds": list(seeds), "steps": steps, "folds": folds}
    if params is not None:
        out["note"] = "one params file for every fold: the held-out world took part in fitting the demand model's six parameters"
    return out


__all__ = ["ARMS", "ARM_LABELS", "DEMAND_MODEL_INPUT", "PLAIN", "SELECT_FORMAT", "arm_label", "arm_of", "select", "train_runs", "training_worlds"]
