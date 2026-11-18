"""Start charges from a file of charge states (`task.episode.battery.soc.file`).

A charge state is the battery charge of every robot of a fleet at one moment of a long run, in percent, in robot
order: the robot that held the first robot's start spot first. An episode draws one state from the file with its
seed, and each robot starts with the charge at its place in that state, so the robots' charges at the start are those
of a fleet that has been running, with the spread and the dependence between the robots that such a run has.

The file is JSON: `{"format": "infra-charge-states/1", "states": [[soc, soc, ...], ...]}`, every state with the same
number of charges; other keys (where the states came from) are not read here."""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

FORMAT = "infra-charge-states/1"
#: The key of the episode RNG stream a state is drawn from
STREAM = ("robots", "soc", "state")


def parse(doc: object, where: str = "charge states") -> tuple[tuple[float, ...], ...]:
    """The states of a charge-state document. ValueError for another format, no state, states of unequal length or a charge outside 0 to 100."""
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError(f"{where}: format: expected {FORMAT!r}")
    states = doc.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"{where}: no states")
    out = []
    for index, state in enumerate(states):
        if not isinstance(state, list) or not state or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 100.0 for v in state):
            raise ValueError(f"{where}: state {index} is no list of charges between 0 and 100: {state!r}")
        out.append(tuple(float(v) for v in state))
    if len({len(state) for state in out}) != 1:
        raise ValueError(f"{where}: the states hold different numbers of robots")
    return tuple(out)


@functools.lru_cache(maxsize=8)
def _load(path: str, stamp: int) -> tuple[tuple[float, ...], ...]:
    del stamp
    return parse(json.loads(Path(path).read_text()), path)


def load(path: str | Path) -> tuple[tuple[float, ...], ...]:
    """The states of the file at `path`, read again when the file changes. ValueError when there is no such file."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"task.episode.battery.soc.file: no file {path}")
    return _load(str(path), path.stat().st_mtime_ns)


def draw(states: tuple[tuple[float, ...], ...], rng: np.random.Generator) -> tuple[float, ...]:
    """One state, drawn with equal weights from `rng` (one draw)."""
    return states[int(rng.integers(len(states)))]


def start_charge(state: tuple[float, ...], index: int) -> float:
    """The start charge of robot `index` (robot order) in a drawn state; a fleet larger than the state's goes round it."""
    return state[index % len(state)]


__all__ = ["FORMAT", "STREAM", "draw", "load", "parse", "start_charge"]
