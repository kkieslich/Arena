"""The fleet of the entrance layouts' runs (study design Part A, A3 and A4): what `flowedit infra evaluation --report`
writes into the stages with the fleet, and the charge states a long fleet-only run gives.

- Jobs (`job_entries`): the seed's job list (`recipients.jobs`) as the fleet mode's `task.fleet.jobs`, each job with the
  locker that serves its recipient entrance in the layout. A delivery runs from the loading point to that locker, a
  pickup back the same way; every layout gets the same jobs at the same release times.
- Rules (`fleet_config`): robots charge by need (`charge_rule: need`: a robot goes to charge only when its next job
  would leave it under the reserve by the job's energy estimate; no threshold, no top-ups of idle robots), take the
  nearest free charger, and go on to it after a jam on the way (`charge_jam: retry`) instead of being reset to a far
  waiting spot. The first study's suites (`infra.fleet_config`) set none of these and keep its behaviour.
- Battery (`contest`): no capacity override, so every robot runs on its model's own battery (the jackal's 270 Wh). The
  start charge comes from a file of charge states (`task.episode.battery.soc.file`: one state per episode seed, each
  robot its place in it), else every robot starts at `start_soc` (full by default).
- Charge states (`read_charge_log`, `charge_states`): the fleet mode's charge log of a long fleet-only run
  (`task.fleet.soc_log`), cut after a run-in and thinned, each state the charges in the order of the robots' start
  spots (the charge of whichever robot holds the first robot's start spot comes first, and so on): a robot that starts
  an episode on a charger gets the charge of a robot that stood on that charger."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import infra as IN
from . import place
from .recipients import Job
from .search import HOMES

#: The format of a charge-state file, as arena_runtime reads it (`env/charge_states.py`)
CHARGE_STATES_FORMAT = "infra-charge-states/1"
#: The start charge of every robot without a file of charge states [%]: full
START_SOC = 100.0
#: The run-in a fleet-only run's charge log is cut by, and the time between two states taken from it [s]: placeholders
#: until such a run shows how long the fleet takes to settle
RUN_IN_S = 3600.0
STATE_EVERY_S = 600.0
#: How often a fleet-only run logs the robots' charge [s]
CHARGE_LOG_S = 60.0


def job_entries(stream: Iterable[Job], serves: Mapping[str, Sequence[str]]) -> list[str]:
    """A seed's jobs as the fleet mode's job list, `<release s> <kind> <locker> <recipient entrance>` each, the locker
    the one that serves the job's entrance in the layout (`serves`: locker object -> entrances)."""
    locker_of = {entrance: locker for locker, entrances in serves.items() for entrance in entrances}
    out = []
    for job in stream:
        if job.entrance not in locker_of:
            raise ValueError(f"job {job.index} is addressed to {job.entrance!r}, which no locker serves ({sorted(locker_of)})")
        out.append(f"{job.release_s:.3f} {job.kind} {locker_of[job.entrance]} {job.entrance}")
    return out


def fleet_config(
    lockers: Sequence[str],
    entries: Sequence[str],
    *,
    rate_per_h: float,
    pickup_share: float = IN.PICKUP_SHARE,
    load_s: float = IN.LOAD_S,
    unload_s: float = IN.UNLOAD_S,
    charge_until: float = IN.CHARGE_UNTIL,
    charge_log: str = "",
    charge_log_s: float = CHARGE_LOG_S,
) -> dict[str, Any]:
    """The `fleet` mode block of a stage of an entrance layout: homes on the bay slots and chargers, the loading point,
    the layout's lockers, the seed's job list, and the rules of A4 (see the module doc). `rate_per_h` and `pickup_share`
    are what the job list was drawn with, kept for the record: the mode runs the list. `charge_log` is the file the mode
    logs the robots' charge to (a fleet-only run)."""
    if not lockers:
        raise ValueError("a layout has at least one locker")
    (loading,) = place.CATALOG["loading_point"].members
    block: dict[str, Any] = {
        "homes": list(HOMES),
        "loading": loading,
        "lockers": list(lockers),
        "jobs": list(entries),
        "rate": float(rate_per_h),
        "pickup_share": float(pickup_share),
        "load_s": float(load_s),
        "unload_s": float(unload_s),
        "charge_until": float(charge_until),
        "charge_rule": "need",
        "nearest_charger": True,
        "charge_jam": "retry",
        "top_up": False,
    }
    if charge_log:
        block |= {"soc_log": str(charge_log), "soc_log_s": float(charge_log_s)}
    return block


def contest(planner: Mapping[str, str] | None = None, *, soc_file: str | None = None, start_soc: float = START_SOC) -> list[dict[str, Any]]:
    """The contest of the entrance layouts' fleet stages: the fleet's planner, no battery capacity (each robot its
    model's own), and the start charge: from the charge states in `soc_file` where given, else `start_soc` % for every
    robot."""
    if not 0.0 <= start_soc <= 100.0:
        raise ValueError(f"start_soc {start_soc} must lie in 0 to 100")
    entry: dict[str, Any] = {"name": IN.FLEET_CONTESTANT, "mobile": dict(IN.PLANNER if planner is None else planner)}
    if soc_file:
        entry["task.episode.battery.soc.file"] = str(soc_file)
    else:
        entry |= {"task.episode.battery.soc.min": float(start_soc), "task.episode.battery.soc.max": float(start_soc)}
    return [entry]


def without_crowd(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """A crowd scenario without its walkers: its sources left out, everything else as it is (a fleet-only run)."""
    doc = copy.deepcopy(dict(scenario))
    doc["regions"] = {name: region for name, region in (doc.get("regions") or {}).items() if region.get("type") != "source"}
    return doc


# --- charge states -----------------------------------------------------------------------------------------------------


def read_charge_log(path: Path) -> list[dict[str, Any]]:
    """The episodes of a charge log the fleet mode wrote (`task.fleet.soc_log`): per episode its robots and its rows
    (seconds since the reset, each robot's charge or None, each robot's home), a header line starting each episode.
    ValueError for a line that fits no header."""
    episodes: list[dict[str, Any]] = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        cells = line.strip().split(",")
        if not line.strip():
            continue
        if cells[0] == "t_s":
            robots = [c.partition(":")[2] for c in cells[1:] if c.startswith("soc:")]
            if not robots or cells[1:] != [*(f"soc:{r}" for r in robots), *(f"home:{r}" for r in robots)]:
                raise ValueError(f"{path}:{number}: no charge log header: {line!r}")
            episodes.append({"robots": robots, "rows": []})
            continue
        if not episodes or len(cells) != 1 + 2 * len(episodes[-1]["robots"]):
            raise ValueError(f"{path}:{number}: a row that fits no header: {line!r}")
        n = len(episodes[-1]["robots"])
        try:
            row = (float(cells[0]), [float(c) if c else None for c in cells[1 : 1 + n]], cells[1 + n :])
        except ValueError:
            raise ValueError(f"{path}:{number}: no numbers in {line!r}") from None
        episodes[-1]["rows"].append(row)
    return episodes


def charge_states(logs: Sequence[Path], *, from_s: float = RUN_IN_S, every_s: float = STATE_EVERY_S) -> dict[str, Any]:
    """The charge-state document of the charge logs of fleet-only runs: from each episode the rows from `from_s` on, at
    least `every_s` apart, each as the charges in the order of the start spots (the homes at the episode's first row).
    A row with an unknown charge, or whose homes are not the start spots, is left out and counted. ValueError when no
    state is left, or the episodes have different start spots."""
    states: list[list[float]] = []
    spots: list[str] | None = None
    skipped = rows = 0
    for path in logs:
        for episode in read_charge_log(path):
            if not episode["rows"]:
                continue
            start = list(episode["rows"][0][2])
            if spots is None:
                spots = start
            elif start != spots:
                raise ValueError(f"{path}: an episode starts on {start}, another on {spots}")
            last = None
            for t_s, socs, homes in episode["rows"]:
                rows += 1
                if t_s < from_s or (last is not None and t_s - last < every_s):
                    continue
                if any(soc is None for soc in socs) or sorted(homes) != sorted(start):
                    skipped += 1
                    continue
                at = dict(zip(homes, socs, strict=True))
                states.append([round(min(100.0, max(0.0, float(at[spot]))), 3) for spot in start])
                last = t_s
    if not states:
        raise ValueError(f"no charge state from {from_s:g} s on in {', '.join(str(p) for p in logs)} ({rows} rows)")
    return {"format": CHARGE_STATES_FORMAT, "states": states, "spots": spots, "source": {"logs": [str(p) for p in logs], "from_s": float(from_s), "every_s": float(every_s), "rows": rows, "skipped": skipped}}


def load_charge_states(path: Path) -> dict[str, Any]:
    """The charge-state document at `path`, checked as the runtime checks it: its format, and states of one length with
    charges from 0 to 100."""
    doc = json.loads(Path(path).read_text())
    states = doc.get("states") if isinstance(doc, dict) else None
    if not isinstance(doc, dict) or doc.get("format") != CHARGE_STATES_FORMAT or not isinstance(states, list) or not states:
        raise ValueError(f"{path}: no charge states (format {CHARGE_STATES_FORMAT})")
    if len({len(s) for s in states}) != 1 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 100.0 for s in states for v in s) or not states[0]:
        raise ValueError(f"{path}: every state holds the same number of charges, each from 0 to 100")
    return doc


__all__ = ["CHARGE_STATES_FORMAT", "START_SOC", "charge_states", "contest", "fleet_config", "job_entries", "load_charge_states", "read_charge_log", "without_crowd"]
