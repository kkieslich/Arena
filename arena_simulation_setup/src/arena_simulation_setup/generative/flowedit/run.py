"""`flowedit run`: (world, edits, crowd settings, seed) -> edited world -> the reader's scenario -> a recorded PEDS run -> tracks.

Every run is one directory under the runs root (`$ARENA_DATA_DIR/flowedit_runs` by default):

    <run id>/run.yaml        what was run: world, base, edits, settings, seed, launch command, track summary
    <run id>/launch.log      the `arena launch` output
    <run id>/humansim/       HumanSim's own recording (`humansim.record:=true`): the bag and recording.yaml
    <run id>/tracks.parquet  one row per agent per frame, with `key`, the id that pairs an agent across runs
    <run id>/interactions.parquet  HumanSim's interactions (an object's use and its queue, a talk), members keyed like the tracks

The run id is `<world>__<scenario>`, the scenario named after the settings digest and seed, so the same inputs land in
the same directory and a finished run is skipped unless asked to run again.

A run is a look at one world, not one side of a pair: `--seed` keys the reader's draws, but the runtime seeds every
episode from a `run_seed` drawn at random per launch (and the world's name), so a base and its edit run here never get
the same crowd. `run.yaml` says so (`paired: false`); paired runs go through arena_evaluation's benchmark runner, which
gives each stage its seed.
"""

from __future__ import annotations

import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from arena_simulation_setup.tree.World.write import to_yaml

from ..scenario.cli import _writable, world_root
from . import settings as S
from .edit import EDITS, EditType, Site, apply, dump_edits, open_site, save
from .read import read
from .read import write as write_scenario

RUNS = "flowedit_runs"
DURATION_S = 300.0  # sim seconds a run records unless told otherwise
RUN_FORMAT = "flowedit-run/1"
RUN_FILE = "run.yaml"
TRACKS = "tracks.parquet"
INTERACTIONS = "interactions.parquet"
INTERACTION_TOPIC_TYPE = "arena_humansim_msgs/msg/InteractionFrame"
AGENT_META_TYPE = "arena_humansim_msgs/msg/AgentMeta"
#: InteractionEvent's edges by value
EVENTS = ("activated", "hold_onset", "released", "interrupted", "canceled")
INTERACTION_COLUMNS = ("time", "record", "interaction_id", "type", "outcome", "event", "object_id", "arrived", "holding", "duration", "agent_id", "key", "role", "position", "seat")
_COLUMN_TYPES = {"time": float, "duration": float, "arrived": bool, "holding": bool, "interaction_id": "int64", "agent_id": "int64", "position": "int64", "seat": "int64"}


@dataclass(frozen=True)
class Edit:
    type: EditType
    target: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Prepared:
    #: the world the run launches: saved by the edit, or the base itself without one
    site: Site
    #: the generated scenario's name inside `site`'s world
    scenario: str
    settings: dict[str, Any]
    seed: int
    reader: dict[str, Any]
    notes: list[str]

    @property
    def run_id(self) -> str:
        return f"{self.site.name}__{self.scenario}"


def scenario_name(scenario: str | None, settings: Mapping[str, Any], seed: int) -> str:
    """The generated scenario's name: one per settings and seed, so runs under other settings never overwrite it."""
    return f"flowedit_{scenario or 'world'}_{S.digest(settings)[:8]}_s{seed}"


def base_site(worlds: Path | None, site: Site) -> Site:
    """The unedited world `site` was edited from, the people's source: under `worlds` when it is there, else wherever the name resolves."""
    if not site.edits:
        return site
    try:
        base_root = world_root(worlds, site.base)
    except FileNotFoundError:
        base_root = world_root(None, site.base)
    return open_site(base_root, site.scenario, site.seed, [])


def check_writable(worlds: Path | None, world: str, root: Path) -> None:
    """Refuse a world a scenario must not be written into."""
    if not (root / EDITS).is_file():
        # a shipped world is written in the source tree only
        _writable(world_root(worlds, world, write=True))
    elif worlds is None:
        from arena_simulation_setup.tree.World import WorldIdentifier

        # an edited world is written where `flowedit edit` saved it, the local worlds root, not a pulled copy
        if WorldIdentifier(root.name).resolve_write_sync().path.resolve() != root.resolve():
            raise ValueError(f"{root} is not where `flowedit edit` saves {root.name}, so it is not written")


def prepare(site: Site, base: Site, edits: Sequence[Edit], settings: Mapping[str, Any], seed: int, notes: list[str] | None = None, *, dry_run: bool = False) -> Prepared:
    """Apply `edits` to `site` and save the edited world, then read the crowd into it. `base` is the unedited world the people come from.

    A dry run applies the edits and reads the scenario without writing either."""
    notes = notes if notes is not None else []
    for edit in edits:
        site = apply(site, edit.type, edit.target, dict(edit.params))
    if edits and not dry_run:
        # reopened from where it was saved, so the scenario is written into the saved world
        site = open_site(save(site), None, None, notes)
    result = read(site, settings, seed, notes, base=base)
    name = scenario_name(site.scenario, settings, seed)
    if not dry_run:
        write_scenario(result, name)
    return Prepared(site, name, dict(settings), seed, result.provenance, notes)


def launch_command(world: str, scenario: str, record_dir: Path, extra: Sequence[str] = ()) -> list[str]:
    """`arena launch` of a headless PEDS run of `scenario` in `world`, HumanSim recording into `record_dir`."""
    return [
        "arena",
        "launch",
        "sim:=dummy",
        "human:=arena",
        "headless:=true",
        f"world:={world}",
        "task.robots:=scenario",
        "task.obstacles:=scenario",
        f"task.scenario.file:={scenario}",
        "humansim.record:=true",
        f"humansim.record_dir:={record_dir}",
        *extra,
    ]


def _arena(env: dict[str, str]) -> tuple[list[str], str | None]:
    """The `arena` CLI as a command, and the directory to run it in.

    In a sourced workspace `arena` is a shell function, which a subprocess cannot call: it runs `python3 -m arena_cli`
    from $ARENA_WS_DIR with $ARENA_DIR/_meta on the path, and so does this."""
    found = shutil.which("arena", path=env.get("PATH"))
    if found:
        return [found], None
    arena_dir = env.get("ARENA_DIR")
    if not arena_dir:
        raise RuntimeError("no `arena` command and no ARENA_DIR: source the Arena workspace first")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(Path(arena_dir) / "_meta"), env.get("PYTHONPATH", "")) if p)
    return [sys.executable, "-m", "arena_cli"], env.get("ARENA_WS_DIR")


#: Nodes a run cannot go on without: the launch itself keeps running when one of them dies
VITAL = ("arena_env_node", "arena_humansim_node")


def _died(log: Path) -> str | None:
    """The first vital node `log` reports dead, else None."""
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if "process has died" in line and (node := next((n for n in VITAL if f"[{n}-" in line), None)):
            return node
    return None


def _stop(proc: subprocess.Popen, grace: float) -> None:
    """Interrupt the launch's whole process group, as Ctrl-C would, and kill what is left after `grace` seconds."""
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait)
        except subprocess.TimeoutExpired:
            continue


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _commit(path: Path | None) -> tuple[str | None, bool | None]:
    """The checkout of `path`: its HEAD, and whether tracked files differ from it (untracked files, generated
    scenarios among them, do not count)."""
    if path is None or (head := _git(path, "rev-parse", "HEAD")) is None:
        return None, None
    status = _git(path, "status", "--porcelain", "--untracked-files=no")
    return head, None if status is None else bool(status)


def provenance() -> dict[str, Any]:
    """What code a run ran: `code` (this flowedit's checkout), `humansim` (the arena_humansim the launch imports), each with
    a dirty flag, and `runtime_code` when $ARENA_DIR, the Arena the launch runs, is another checkout than `code`."""
    here = Path(__file__).resolve().parent
    code, code_dirty = _commit(here)
    try:
        import arena_humansim

        # the build tree links back to the source checkout
        humansim_dir: Path | None = Path(arena_humansim.__file__).resolve().parent
    except ImportError:
        humansim_dir = None
    humansim, humansim_dirty = _commit(humansim_dir)
    out: dict[str, Any] = {"code": code, "code_dirty": code_dirty, "humansim": humansim, "humansim_dirty": humansim_dirty}
    arena_dir = os.environ.get("ARENA_DIR")
    if arena_dir and (top := _git(Path(arena_dir), "rev-parse", "--show-toplevel")) and top != _git(here, "rev-parse", "--show-toplevel"):
        out["runtime_code"], out["runtime_dirty"] = _commit(Path(arena_dir))
    return out


#: Agent names as AgentMeta gave them: one {agent id: name} snapshot per message, by its time (s). A plain mapping is one
#: snapshot that holds for the whole run.
Names = Mapping[int, str] | Sequence[tuple[float, Mapping[int, str]]]


@dataclass
class Extracted:
    #: one row per agent per frame (`keyed`)
    tracks: Any
    #: one row per interaction member per frame, and per participant per edge (`interaction_rows`)
    interactions: Any


def extract(bag: Path, duration: float | None = None) -> Extracted:
    """The tracks and interactions of HumanSim's `bag`, both on the tracks' time base and cut at `duration`."""
    from arena_humansim.utils.bag_io import extract_agent_states
    from rosbags.highlevel import AnyReader

    names: list[tuple[float, dict[int, str]]] = []
    frames: list[tuple[float, Any]] = []
    with AnyReader([bag]) as reader:
        wanted = [c for c in reader.connections if c.msgtype in (AGENT_META_TYPE, INTERACTION_TOPIC_TYPE)]
        for connection, stamp, raw in reader.messages(connections=wanted):
            msg = reader.deserialize(raw, connection.msgtype)
            if connection.msgtype == AGENT_META_TYPE:
                names.append((stamp * 1e-9, {int(i): str(n) for i, n in zip(msg.agent_id, msg.name, strict=False)}))
            else:
                frames.append((stamp * 1e-9, msg))
    states = extract_agent_states(bag)
    t0 = float(states["time"].min()) if not states.empty else 0.0
    df = keyed(states, names, duration)
    return Extracted(df, interaction_rows(frames, t0, df, duration))


def tracks(bag: Path, duration: float | None = None) -> Any:  # noqa: ANN401 - a pandas DataFrame, pandas imported lazily
    """The agent states of `bag` keyed by `keyed`, trimmed to the first `duration` seconds of the run."""
    return extract(bag, duration).tracks


def names_at(time: Any, agent_id: Any, names: Names) -> Any:  # noqa: ANN401 - numpy arrays
    """Each row's agent name from the latest AgentMeta at or before its time, else the first later one naming the agent,
    empty when none does: HumanSim reuses an agent id once its agent left, and the next agent under it has its own name."""
    import numpy as np
    import pandas as pd

    snapshots = [(-math.inf, names)] if isinstance(names, Mapping) else sorted(names, key=lambda s: s[0])
    t = np.asarray(time, dtype=float)
    ids = np.asarray(agent_id, dtype=np.int64)
    if not snapshots:
        return np.full(len(ids), "", dtype=object)
    snap_t = np.array([s for s, _ in snapshots], dtype=float)
    table = pd.DataFrame([(k, s, int(i), str(n)) for k, (s, snap) in enumerate(snapshots) for i, n in snap.items()], columns=["k", "t", "agent_id", "name"]).drop_duplicates(["k", "agent_id"], keep="last")
    k = np.searchsorted(snap_t, t, side="right") - 1
    rows = pd.DataFrame({"k": k, "agent_id": ids}).merge(table.drop(columns="t"), on=["k", "agent_id"], how="left")
    unnamed = np.flatnonzero(rows["name"].isna().to_numpy())
    if len(unnamed) and not table.empty:
        probe = pd.DataFrame({"t": t[unnamed], "agent_id": ids[unnamed], "row": unnamed}).sort_values("t", kind="stable")
        later = pd.merge_asof(probe, table.drop(columns="k").sort_values("t", kind="stable"), on="t", by="agent_id", direction="forward").dropna(subset=["name"])
        rows.loc[later["row"].to_numpy(), "name"] = later["name"].to_numpy()
    return rows["name"].fillna("").to_numpy(dtype=object)


def keyed(df: Any, names: Names, duration: float | None = None) -> Any:  # noqa: ANN401 - a pandas DataFrame
    """`df` (HumanSim's agent states) with each agent's `name` and `key`, its time from the first frame, cut at `duration`.

    `name` is the agent's name at that frame (`names_at`). `key` pairs an agent across runs: `<source_name>#<source_seq>`
    for a person a source released (HumanSim pairs those by their draws, never by agent id), the scenario name of a person
    the scenario places without its env prefix (the reader gives every edit of a world the same people), and `id<agent id>`
    for anything else."""
    if df.empty:
        return df
    df = df.copy()
    # HumanSim names a scenario's people inside their env (`env_1/flow_1`), and the env index is not the same every launch
    df["name"] = [n.rpartition("/")[2] for n in names_at(df["time"], df["agent_id"], names)]
    released = df["source_name"] != ""
    df["key"] = ("id" + df["agent_id"].astype(str)).where(df["name"] == "", df["name"])
    df.loc[released, "key"] = df.loc[released, "source_name"] + "#" + df.loc[released, "source_seq"].astype(str)
    df["time"] = df["time"] - df["time"].min()
    if duration is not None:
        df = df[df["time"] <= duration]
    return df.reset_index(drop=True)


def _label(enum_name: str, value: int) -> str:
    """The lower-case name of HumanSim's `enum_name` member `value`, else the value itself."""
    try:
        from arena_humansim.core import interaction_kinds
        from arena_humansim.utils import types

        enum = getattr(interaction_kinds, enum_name, None) or getattr(types, enum_name)
        return enum(int(value)).name.lower()
    except (ImportError, AttributeError, ValueError):
        return str(int(value))


def _keys_at(time: Any, agent_id: Any, tracks: Any) -> list[str]:  # noqa: ANN401 - numpy arrays and a DataFrame
    """The track `key` of each (time, agent id): the agent's latest frame at or before it, else its first later one, else `id<agent id>`."""
    import numpy as np
    import pandas as pd

    ids = np.asarray(agent_id, dtype=np.int64)
    fallback = [f"id{i}" for i in ids]
    if not len(ids) or tracks is None or tracks.empty:
        return fallback
    probe = pd.DataFrame({"time": np.asarray(time, dtype=float), "agent_id": ids, "row": np.arange(len(ids))}).sort_values("time", kind="stable")
    table = tracks[["time", "agent_id", "key"]].astype({"time": float, "agent_id": np.int64}).sort_values("time", kind="stable")
    keys = pd.merge_asof(probe, table, on="time", by="agent_id", direction="backward")
    missing = keys["key"].isna()
    if missing.any():
        later = pd.merge_asof(probe[missing.to_numpy()], table, on="time", by="agent_id", direction="forward")
        keys.loc[missing.to_numpy(), "key"] = later["key"].to_numpy()
    found = dict(zip(keys["row"], keys["key"], strict=True))
    return [k if isinstance(k := found.get(i), str) else fallback[i] for i in range(len(ids))]


def interaction_rows(frames: Sequence[tuple[float, Any]], t0: float, tracks: Any = None, duration: float | None = None) -> Any:  # noqa: ANN401 - a pandas DataFrame
    """HumanSim's InteractionFrames (`(time s, message)`, the bag's time base) as rows on the tracks' time base: seconds
    from `t0`, the first agent frame, cut at `duration`.

    `record` tells two kinds of row apart. A `member` row is one member of a live interaction in one frame: its `type`
    (`queue_use`, `sit_on`, `talk_to`, ...), `outcome` (`forming`, `active`), anchor `object_id` (empty for none),
    `arrived`, `holding` (the hold has started), the hold `duration` (s, -1 unbounded), and the member's `role`
    (`participant`, `queued`), `position` in the member list (participants first, then the queue in FIFO order) and
    `seat` (-1 none). An `event` row is one participant of an edge fired that tick: `event` is `activated`, `hold_onset`,
    `released`, `interrupted` or `canceled`, and `object_id` the interaction's last known anchor.

    The topic is latched and sent on change, so the last frame before `t0` holds what is live at the start: it is kept at
    time 0, its edges dropped. `key` pairs a member with its track (`_keys_at`)."""
    import pandas as pd

    frames = sorted(frames, key=lambda f: f[0])
    before = [f for f in frames if f[0] < t0]
    kept = ([(t0, before[-1][1], False)] if before else []) + [(t, m, True) for t, m in frames if t >= t0 and (duration is None or t - t0 <= duration)]
    rows: list[dict[str, Any]] = []
    anchors: dict[int, str] = {}
    for t, msg, with_events in kept:
        time = round(t - t0, 6)
        offsets = list(msg.member_offset)
        for i, iid in enumerate(msg.interaction_id):
            anchors[int(iid)] = str(msg.object_id[i])
            span = range(offsets[i], offsets[i + 1]) if i + 1 < len(offsets) else range(0)
            for position, m in enumerate(span):
                rows.append(
                    {
                        "time": time,
                        "record": "member",
                        "interaction_id": int(iid),
                        "type": _label("InteractionType", msg.interaction_type[i]),
                        "outcome": _label("InteractionOutcome", msg.outcome[i]),
                        "event": "",
                        "object_id": str(msg.object_id[i]),
                        "arrived": bool(msg.arrived[i]),
                        "holding": bool(msg.holding[i]),
                        "duration": float(msg.duration[i]),
                        "agent_id": int(msg.member_id[m]),
                        "role": _label("MembershipRole", msg.member_role[m]),
                        "position": position,
                        "seat": int(msg.member_seat[m]),
                    }
                )
        if not with_events:
            continue
        for ev in msg.events:
            for agent in ev.participants:
                rows.append(
                    {
                        "time": time,
                        "record": "event",
                        "interaction_id": int(ev.interaction_id),
                        "type": _label("InteractionType", ev.interaction_type),
                        "outcome": "",
                        "event": EVENTS[ev.event] if ev.event < len(EVENTS) else str(int(ev.event)),
                        "object_id": anchors.get(int(ev.interaction_id), ""),
                        "arrived": False,
                        "holding": False,
                        "duration": -1.0,
                        "agent_id": int(agent),
                        "role": "participant",
                        "position": -1,
                        "seat": -1,
                    }
                )
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=_COLUMN_TYPES.get(c, object)) for c in INTERACTION_COLUMNS})
    df = pd.DataFrame(rows)
    df["key"] = _keys_at(df["time"], df["agent_id"], tracks)
    return df[list(INTERACTION_COLUMNS)]


def uses(interactions: Any) -> dict[str, dict[str, int]]:  # noqa: ANN401 - a pandas DataFrame
    """Per anchor object: `uses` (interactions at it that reached their hold), `users` (participants of those, by key) and
    `queue_max` (the longest queue seen at it)."""
    if interactions is None or interactions.empty:
        return {}
    members = interactions[(interactions["record"] == "member") & (interactions["object_id"] != "")]
    held = members[members["holding"]]
    out: dict[str, dict[str, int]] = {}
    for obj, group in members.groupby("object_id"):
        used = held[held["object_id"] == obj]
        queued = group[group["role"] == "queued"].groupby("time").size()
        out[str(obj)] = {
            "uses": int(used["interaction_id"].nunique()),
            "users": int(used.loc[used["role"] == "participant", "key"].nunique()),
            "queue_max": int(queued.max()) if len(queued) else 0,
        }
    return dict(sorted(out.items()))


def summary(df: Any) -> dict[str, Any]:  # noqa: ANN401 - a pandas DataFrame
    if df.empty:
        return {"frames": 0, "agents": 0, "seconds": 0.0}
    humans = df[df["agent_id"] > 0]
    return {
        "frames": int(df["time"].nunique()),
        "agents": int(humans["key"].nunique()),
        "released": int(humans.loc[humans["source_name"] != "", "key"].nunique()),
        "seconds": round(float(df["time"].max()), 2),
    }


def record(
    prepared: Prepared,
    out: Path,
    duration: float,
    *,
    launch_args: Sequence[str] = (),
    startup_timeout: float = 300.0,
    grace: float = 60.0,
    margin: float = 30.0,
    log: Callable[[str], None] = print,
) -> Path:
    """Launch the prepared scenario headless with HumanSim recording, stop it after `duration` seconds of frames, and write the run directory.

    The run's tracks are trimmed to the first `duration` seconds after the first frame. Raises RuntimeError when the launch
    dies or records no frames, the launch log kept for the reason."""
    run_dir = out / prepared.run_id
    humansim = run_dir / "humansim"
    if humansim.exists():
        shutil.rmtree(humansim)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / RUN_FILE).unlink(missing_ok=True)
    command = launch_command(prepared.site.name, prepared.scenario, humansim, launch_args)
    env = dict(os.environ)
    # the edited world is saved outside the source tree, the base world's generated scenario inside it: the launch sees both
    env["ARENA_WORLD_PATH"] = os.pathsep.join(p for p in (str(prepared.site.root.parent), env.get("ARENA_WORLD_PATH", "")) if p)
    arena, cwd = _arena(env)
    started = datetime.now(timezone.utc)
    log(f"{prepared.run_id}: {shlex.join(command)}")
    with (run_dir / "launch.log").open("w") as out_log:
        proc = subprocess.Popen([*arena, *command[1:]], stdout=out_log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
        try:
            bag = humansim / "bag"
            deadline = time.monotonic() + startup_timeout
            while not bag.exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"{prepared.run_id}: the launch exited ({proc.returncode}) before HumanSim recorded, see {run_dir / 'launch.log'}")
                if died := _died(run_dir / "launch.log"):
                    raise RuntimeError(f"{prepared.run_id}: {died} died before HumanSim recorded, see {run_dir / 'launch.log'}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"{prepared.run_id}: HumanSim did not start recording within {startup_timeout:.0f} s, see {run_dir / 'launch.log'}")
                time.sleep(2.0)
            log(f"{prepared.run_id}: recording, stopping in {duration:.0f} s")
            # the bag opens before the scenario loads: a margin so the trimmed tracks still span `duration`
            end = time.monotonic() + duration + margin
            while time.monotonic() < end:
                if proc.poll() is not None:
                    raise RuntimeError(f"{prepared.run_id}: the launch exited ({proc.returncode}) while recording, see {run_dir / 'launch.log'}")
                if died := _died(run_dir / "launch.log"):
                    raise RuntimeError(f"{prepared.run_id}: {died} died while recording, see {run_dir / 'launch.log'}")
                time.sleep(2.0)
        finally:
            _stop(proc, grace)
    got = extract(humansim / "bag", duration)
    df = got.tracks
    if df.empty:
        raise RuntimeError(f"{prepared.run_id}: no agent frames recorded, see {run_dir / 'launch.log'}")
    df.to_parquet(run_dir / TRACKS, index=False)
    got.interactions.to_parquet(run_dir / INTERACTIONS, index=False)
    stats = summary(df)
    doc = {
        "format": RUN_FORMAT,
        "world": prepared.site.name,
        "base": prepared.site.base,
        "scenario": prepared.site.scenario,
        "generated_scenario": prepared.scenario,
        "seed": prepared.seed,
        # the runtime's own seed is random per launch: this run pairs with no other, see the module docstring
        "paired": False,
        "settings_digest": S.digest(prepared.settings),
        "settings": prepared.settings,
        "duration": duration,
        "launch": command,
        "started": started.isoformat(timespec="seconds"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **provenance(),
        "edits": dump_edits(prepared.site)["edits"],
        "edit_seed": prepared.site.seed,
        "reader": prepared.reader,
        "tracks": stats,
        "uses": uses(got.interactions),
    }
    if stats["seconds"] < duration - 1.0:
        doc["warning"] = f"tracks span {stats['seconds']} s of the {duration} s asked: the sim ran slower than real time"
        log(f"{prepared.run_id}: {doc['warning']}")
    (run_dir / RUN_FILE).write_text(to_yaml(doc))
    return run_dir


def done(run_dir: Path) -> bool:
    """Whether `run_dir` holds a finished run: run.yaml is written last."""
    return (run_dir / RUN_FILE).is_file() and (run_dir / TRACKS).is_file()


def default_out() -> Path:
    data = os.environ.get("ARENA_DATA_DIR")
    if not data:
        raise ValueError("pass --out, or set ARENA_DATA_DIR for the default runs root")
    return Path(data) / RUNS


__all__ = ["DURATION_S", "EDITS", "INTERACTIONS", "RUNS", "Edit", "Extracted", "Prepared", "base_site", "check_writable", "default_out", "done", "extract", "interaction_rows", "keyed", "launch_command", "names_at", "prepare", "provenance", "record", "scenario_name", "summary", "tracks", "uses"]
