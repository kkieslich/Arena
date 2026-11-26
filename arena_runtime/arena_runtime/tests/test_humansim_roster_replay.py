"""The adapter's per-tick output for a fixed crowd: the same content as recorded, and cheap enough for a crowd of 350.

A synthetic engine stream (350 walkers with churn, gestures, interactions, teleports, 4 robots) drives the adapter's
publish paths at their rates: an engine frame every 50 ms, a roster render on receipt and every 20 ms, the update pass
to the simulator every 20 ms and the world state every 50 ms. Every message they hand on (arena_peds, the simulator's
update batch, world_state) is encoded field by field and must match the content recorded before the adapter's hot
paths were reworked (`golden/humansim_roster_replay.json`): a sha256 of everything but the floats (ints, bools,
strings, sequence lengths, field order and where each float sits) compared exactly, and per message the floats' count
and three sums (plain, of squares, index-weighted) compared to 1e-9 of the message's own scale (its sum of absolute
values and so on), which catches one walker moved by a millimetre in one frame. The floats are not hashed bit for bit:
libm and numpy pick their sin/cos/atan2/exp code by CPU (FMA, AVX2, AVX-512), and the last bits differ between CI
runners (the byte digests came out one of two ways from run to run on the same commit). The encoding skips the CDR
alignment padding, which rclpy's serializer leaves uninitialized.

`python tests/test_humansim_roster_replay.py --record` rewrites the golden file from the current adapter.

`python tests/test_humansim_roster_replay.py [walkers] [seconds]` prints the timings and the fingerprints.
"""

from __future__ import annotations

import array
import asyncio
import hashlib
import json
import math
import struct
import sys
import time
import types
import typing
import uuid
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim_msgs.msg")

from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim_msgs.msg import AgentFrame, AgentGestures, AgentMeta, Gesture
from arena_people_msgs.msg import Gesture as PedestrianGesture
from arena_people_msgs.msg import Pedestrian, Pedestrians
from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.realizer import Realizer
from arena_runtime.humans import TrackedRobot
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator, Seat, SeatHold
from arena_runtime.humans.arena_humansim.blank import blank_factory
from arena_runtime.shared import Orientation, Pose, Position
from arena_runtime.sim.dummy_simulator import DummySimulator

ENGINE_DT_NS = 50_000_000
TICK_NS = 20_000_000
ORIGIN = (12.5, -3.25)

COUNTS = {"arena_peds": 365, "pedestrian_update": 360, "world_state": 120}
#: per stream of the default replay (350 walkers, 4 robots, 6 s, skeleton off): the sha256 of all but the floats, and per
#: message the floats' count, sum, sum of squares and index-weighted sum. Recorded on a machine where the adapter still
#: reproduced the byte digests recorded before its rework (sha256 arena_peds cfa009f6.., pedestrian_update b22ab93e..,
#: world_state 48bb6b21..), so it is the content of the adapter before the rework.
GOLDEN = Path(__file__).resolve().parent / "golden" / "humansim_roster_replay.json"
#: a message's float sums agree to this share of their scale (the CPU-dependent last bits move them by ~1e-16 of it)
FLOAT_RTOL = 1e-9


class Fingerprint:
    """One stream's content: the non-float bytes hashed exactly, per message the floats' count and sums with their scales."""

    def __init__(self) -> None:
        self.exact = hashlib.sha256()
        self.chunks: list[tuple[int, float, float, float]] = []
        self.scales: list[tuple[float, float, float]] = []

    def message(self, out: bytes, floats: list[float]) -> None:
        self.exact.update(out)
        n, total, squares, weighted, absolute, weighted_absolute = 0, 0.0, 0.0, 0.0, 0.0, 0.0
        for value in floats:
            if not math.isfinite(value):
                self.exact.update(repr(value).encode())
                continue
            weight = n % 101
            total += value
            squares += value * value
            weighted += weight * value
            absolute += abs(value)
            weighted_absolute += weight * abs(value)
            n += 1
        self.chunks.append((n, total, squares, weighted))
        self.scales.append((absolute, squares, weighted_absolute))

    def value(self) -> dict[str, typing.Any]:
        return {"exact": self.exact.hexdigest(), "chunks": [list(chunk) for chunk in self.chunks]}


def mismatches(got: dict[str, Fingerprint], golden: dict[str, dict[str, typing.Any]]) -> list[str]:
    """Where `got` differs from `golden`: a stream's non-float content, its message count, or the floats of a message."""
    found: list[str] = []
    for stream, ref in golden.items():
        fingerprint = got[stream]
        if fingerprint.exact.hexdigest() != ref["exact"]:
            found.append(f"{stream}: non-float content")
        if len(fingerprint.chunks) != len(ref["chunks"]):
            found.append(f"{stream}: {len(fingerprint.chunks)} messages, recorded {len(ref['chunks'])}")
            continue
        for i, (chunk, scale, (n, *sums)) in enumerate(zip(fingerprint.chunks, fingerprint.scales, ref["chunks"], strict=True)):
            if chunk[0] != n or any(abs(a - b) > FLOAT_RTOL * s + 1e-12 for a, b, s in zip(chunk[1:], sums, scale, strict=True)):
                found.append(f"{stream}: floats of message {i}")
    return found


_FIELDS: dict[type, tuple[str, ...]] = {}


def encode(msg: object, out: bytearray, floats: list[float]) -> None:
    """Append every field value of `msg`, depth first in definition order, sequences with their length; a float goes to
    `floats` and leaves a marker in `out`."""
    cls = type(msg)
    fields = _FIELDS.get(cls)
    if fields is None:
        fields = _FIELDS[cls] = tuple(cls.get_fields_and_field_types())  # pyright: ignore[reportAttributeAccessIssue]
    for field in fields:
        value = getattr(msg, field)
        if isinstance(value, float):
            out += b"f"
            floats.append(value)
        elif isinstance(value, bool):
            out += b"\x01" if value else b"\x00"
        elif isinstance(value, int):
            out += struct.pack("<q", value)
        elif isinstance(value, str):
            raw = value.encode()
            out += struct.pack("<I", len(raw)) + raw
        elif isinstance(value, (array.array, np.ndarray)) and (value.typecode if isinstance(value, array.array) else value.dtype.kind) in ("f", "d"):
            out += struct.pack("<I", len(value)) + b"F"
            floats.extend(float(v) for v in value)
        elif isinstance(value, (bytes, array.array, np.ndarray)):
            raw = value if isinstance(value, bytes) else value.tobytes()
            out += struct.pack("<I", len(value)) + (value.typecode.encode() if isinstance(value, array.array) else b"") + raw
        elif isinstance(value, list):
            out += struct.pack("<I", len(value))
            for item in value:
                if isinstance(item, str):
                    raw = item.encode()
                    out += struct.pack("<I", len(raw)) + raw
                elif isinstance(item, float):
                    out += b"f"
                    floats.append(item)
                elif isinstance(item, int):
                    out += struct.pack("<q", item)
                else:
                    encode(item, out, floats)
        else:
            encode(value, out, floats)


class _Simulator(DummySimulator):
    def __init__(self, sink: typing.Callable[[str, object], None], **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self._sink = sink

    async def pedestrian_update(self, pedestrians: Pedestrians) -> typing.Sequence[bool]:
        self._sink("pedestrian_update", pedestrians)
        return [True] * len(pedestrians.pedestrians)


def _crowd(walkers: int, seconds: float, seed: int = 7) -> list[tuple[AgentFrame, AgentGestures | None, dict[int, int]]]:
    """Engine frames at 20 Hz: walkers drift and turn, a few leave and new ids enter, some jump (teleport), some gesture or interact."""
    rng = np.random.default_rng(seed)
    ids = list(range(100, 100 + walkers))
    next_id = 100 + walkers
    x = rng.uniform(-40.0, 40.0, walkers)
    y = rng.uniform(-25.0, 25.0, walkers)
    theta = rng.uniform(-math.pi, math.pi, walkers)
    speed = rng.uniform(0.0, 1.6, walkers)
    speed[rng.random(walkers) < 0.15] = 0.0
    phase = rng.uniform(0.0, 2.0 * math.pi, walkers)
    radius = rng.uniform(0.25, 0.4, walkers)
    frames: list[tuple[AgentFrame, AgentGestures | None, dict[int, int]]] = []
    for k in range(int(seconds * 1e9 / ENGINE_DT_NS)):
        # churn: a sink takes one walker, a source releases one
        if k % 7 == 3:
            drop = int(rng.integers(len(ids)))
            ids.pop(drop)
            x, y, theta, speed, phase, radius = (np.delete(a, drop) for a in (x, y, theta, speed, phase, radius))
            ids.append(next_id)
            next_id += 1
            x, y = np.append(x, rng.uniform(-40.0, 40.0)), np.append(y, rng.uniform(-25.0, 25.0))
            theta, speed = np.append(theta, rng.uniform(-math.pi, math.pi)), np.append(speed, rng.uniform(0.5, 1.5))
            phase, radius = np.append(phase, 0.0), np.append(radius, 0.3)
        theta = np.arctan2(np.sin(theta + rng.normal(0.0, 0.15, len(ids))), np.cos(theta + rng.normal(0.0, 0.15, len(ids))))
        vx, vy = speed * np.cos(theta), speed * np.sin(theta)
        x, y = x + vx * ENGINE_DT_NS / 1e9, y + vy * ENGINE_DT_NS / 1e9
        teleported: dict[int, int] = {}
        if k % 23 == 11:
            j = int(rng.integers(len(ids)))
            x[j] += 30.0
            teleported[ids[j]] = (k + 1) * ENGINE_DT_NS - 1
        phase = np.mod(phase + speed * 2.0 * ENGINE_DT_NS / 1e9, 2.0 * math.pi)
        n = len(ids)
        interaction_id = np.full(n, -1, dtype=np.int32)
        interaction_id[: n // 20] = 4
        frame = AgentFrame(
            agent_id=ids,
            x=x.tolist(),
            y=y.tolist(),
            theta=theta.tolist(),
            vx=vx.tolist(),
            vy=vy.tolist(),
            desired_velocity=np.full(n, 1.3).tolist(),
            radius=radius.tolist(),
            kind=bytes(n),
            animation_state=bytes(int(s > 0.0) for s in speed),
            policy_idx=[0] * n,
            gait_phase=phase.tolist(),
            gait_cadence=(speed * 0.9).tolist(),
            interaction_id=interaction_id.tolist(),
            interaction_type=bytes([1] * (n // 20) + [0] * (n - n // 20)),
        )
        frame.header.frame_id = "map"
        frame.header.stamp.sec = (k + 1) * ENGINE_DT_NS // 1_000_000_000
        frame.header.stamp.nanosec = (k + 1) * ENGINE_DT_NS % 1_000_000_000
        gestures = None
        if k % 10 == 0:
            gestures = AgentGestures()
            for i in range(0, n, 17):
                gesture = Gesture(slot="head", clip="", hand="")
                gesture.at.x, gesture.at.y, gesture.at.z = float(x[i] + 1.0), float(y[i]), 1.6
                gestures.agent_id.append(ids[i])  # pyright: ignore[reportAttributeAccessIssue]
                gestures.gestures.append(gesture)  # pyright: ignore[reportAttributeAccessIssue]
            for i in range(5, n, 41):
                gesture = Gesture(slot="body", clip="", hand="", render_pose_override=True)
                gesture.at.x, gesture.at.y = float(x[i] + 0.2), float(y[i] - 0.1)
                gestures.agent_id.append(ids[i])  # pyright: ignore[reportAttributeAccessIssue]
                gestures.gestures.append(gesture)  # pyright: ignore[reportAttributeAccessIssue]
        frames.append((frame, gestures, teleported))
    return frames


class _Clock(typing.Protocol):
    now_ns: int


class Replay:
    """The adapter on a live node, its publish paths driven by the clock of the replay."""

    def __init__(self, adapter: ArenaHumanSimulator, node: _Clock, *, recording: bool) -> None:
        self.adapter = adapter
        self.node = node
        self.recording = recording
        self.counts = {"arena_peds": 0, "pedestrian_update": 0, "world_state": 0}
        self.fingerprint = {stream: Fingerprint() for stream in self.counts}
        self.timings: dict[str, list[float]] = {"render": [], "update": [], "world_state": []}

    def record(self, stream: str, msg: object) -> None:
        """Encode now: a receiver may mutate what it was handed later on."""
        self.counts[stream] += 1
        if self.recording:
            out = bytearray()
            floats: list[float] = []
            encode(msg, out, floats)
            self.fingerprint[stream].message(bytes(out), floats)

    def _meta(self, frame: AgentFrame) -> None:
        n = len(frame.agent_id)
        meta = AgentMeta(agent_id=list(frame.agent_id), agent_type=[""] * n, source_name=["lobby"] * n, source_seq=list(range(n)), name=[""] * n)
        self.adapter._agent_meta_callback(meta)
        for aid in frame.agent_id:
            if aid % 11:
                self.adapter._agent_names.setdefault(int(aid), self.adapter._realizer.prefix(f"flow_{aid}"))
                self.adapter._ped_model_uris.setdefault(self.adapter._realizer.prefix(f"flow_{aid}"), f"/models/human_{aid % 5}/model.sdf")
        self.adapter._flow_agent_ids = set(self.adapter._agent_names)

    async def run(self, frames: list[tuple[AgentFrame, AgentGestures | None, dict[int, int]]]) -> None:
        adapter = self.adapter
        now_ns = 0
        timer = time.perf_counter
        for k, (frame, gestures, teleported) in enumerate(frames):
            if gestures is not None:
                adapter._agent_gestures_callback(gestures)
            if k % 20 == 0:
                self._meta(frame)
            adapter._teleport_stamps.update(teleported)
            # one engine frame, then the 50 Hz ticks until the next one
            self.node.now_ns = now_ns
            t0 = timer()
            adapter._agent_states_callback(frame)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.timings["render"].append(timer() - t0)
            frame_end = now_ns + ENGINE_DT_NS
            while now_ns < frame_end:
                self.node.now_ns = now_ns
                t0 = timer()
                await adapter._publish_interpolated()
                self.timings["render"].append(timer() - t0)
                t0 = timer()
                await adapter.relay_pedestrian_update(adapter._named(adapter._arena_pedestrians))
                self.timings["update"].append(timer() - t0)
                now_ns += TICK_NS
            t0 = timer()
            adapter._publish_world_state()
            self.timings["world_state"].append(timer() - t0)


def replay(walkers: int = 350, seconds: float = 6.0, *, skeleton: bool = False, recording: bool = True, nudge: tuple[int, int, float | None] | None = None) -> Replay:
    """Drive a fresh adapter through the synthetic crowd and return what it published (fingerprints only while `recording`) and how long each path took."""
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace
    from arena_rclpy_mixins.Time import Time
    from arena_runtime.acoustics import AcousticsSimulatorRegistry
    from arena_runtime.env.constants import Constants

    class _Node(ArenaMixinNode):
        now_ns = 0

        def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
            super().__init__(*args, **kwargs)
            self.conf = types.SimpleNamespace(General=types.SimpleNamespace(RNG=EpisodeRng()))
            self.tf_buffer = None  # the robots are placed, not followed through TF

        @property
        def sim_time(self) -> Time:  # pyright: ignore[reportIncompatibleVariableOverride]
            return Time(sec=self.now_ns // 1_000_000_000, nanosec=self.now_ns % 1_000_000_000)

    frames = _crowd(walkers, seconds)
    if nudge is not None:
        # move one walker of one engine frame along x: by `dx` metres, or by one unit in the last place when it is None
        k, j, dx = nudge
        xs = frames[k][0].x
        xs[j] = math.nextafter(xs[j], math.inf) if dx is None else xs[j] + dx
    result: list[Replay] = []

    async def main() -> None:
        suffix = uuid.uuid4().hex[:8]
        namespace = Namespace(f"/humansim_roster_replay_{suffix}")
        node = _Node(f"humansim_roster_replay_{suffix}", namespace=str(namespace))
        realizer = Realizer(Realizer._Configuration(x=ORIGIN[0], y=ORIGIN[1], prefix="env_0"))
        holder: list[Replay] = []
        simulator = _Simulator(lambda stream, msg: holder[0].record(stream, msg), node=node, namespace=namespace, realizer=realizer)
        acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=namespace)
        adapter = ArenaHumanSimulator(node=node, namespace=namespace, simulator=simulator, realizer=realizer, acoustics=acoustics, skeleton=skeleton)
        run = Replay(adapter, node, recording=recording)
        holder.append(run)
        adapter.on_arena_peds(lambda msg: run.record("arena_peds", msg))
        adapter._world_state_pub = types.SimpleNamespace(publish=lambda msg: run.record("world_state", msg))  # pyright: ignore[reportAttributeAccessIssue]
        adapter._engine_dt_ns = ENGINE_DT_NS
        adapter._viz_config = realizer.get_config()  # no marker node to point at this env
        # the walkers an episode's sources released and its sinks took before: their model paths stay on file
        adapter._ped_model_uris.update({realizer.prefix(f"flow_{aid}"): f"/models/human_{aid % 5}/model.sdf" for aid in range(-4000, 0)})
        adapter._ped_model_uris.update({f"flow_{aid}": f"/models/human_{aid % 5}/model.sdf" for aid in range(-4000, 0)})
        adapter._seats = {realizer.prefix("bench_0"): [Seat(x=1.0, y=2.0, height=0.45, front=None, back=None)]}
        adapter._seat_holds = {100: SeatHold(kind=InteractionType.LIE_ON, object_id=realizer.prefix("bench_0"), seat=0)}
        for i in range(4):
            name = f"robot_{i}" if i else "jackal"
            pose = Pose(Position(ORIGIN[0] + 2.0 * i, ORIGIN[1] - 1.0), Orientation.from_yaw(0.3 * i))
            adapter._robots[name] = TrackedRobot(robot=types.SimpleNamespace(name=name), tf_frame=None, pose=pose, velocity=(0.4 * i, -0.1), radius=0.35)  # pyright: ignore[reportArgumentType]
        try:
            await run.run(frames)
        finally:
            await simulator.stop_mechanisms()
            node.destroy_node()
        result.append(run)

    if not rclpy.ok():
        rclpy.init()
    asyncio.run(main())
    return result[0]


def test_a_blank_pedestrian_is_a_default_one_sharing_nothing_mutable() -> None:
    blank = blank_factory(Pedestrian)
    a, b = blank(), blank()
    assert a == Pedestrian() and type(a) is Pedestrian
    assert all(type(getattr(a, slot)) is type(getattr(Pedestrian(), slot)) for slot in Pedestrian.__slots__)
    a.pose.position.x = 1.0
    a.gestures.append(PedestrianGesture())  # pyright: ignore[reportAttributeAccessIssue]
    a.joint_state.position.append(0.5)  # pyright: ignore[reportAttributeAccessIssue]
    a.body.joints.append("hip")  # pyright: ignore[reportAttributeAccessIssue]
    assert b == Pedestrian()


def _golden() -> dict[str, typing.Any]:
    return json.loads(GOLDEN.read_text())


def test_the_adapter_publishes_the_recorded_content() -> None:
    golden = _golden()
    run = replay()
    assert run.counts == COUNTS == golden["counts"]
    assert mismatches(run.fingerprint, golden["streams"]) == []


def test_one_walker_moved_by_a_millimetre_in_one_frame_is_caught() -> None:
    found = mismatches(replay(nudge=(40, 7, 1e-3)).fingerprint, _golden()["streams"])
    assert any(m.startswith("arena_peds: floats") for m in found), found
    assert any(m.startswith("pedestrian_update: floats") for m in found), found


def test_one_walker_moved_by_its_last_bit_is_not() -> None:
    assert mismatches(replay(nudge=(40, 7, None)).fingerprint, _golden()["streams"]) == []


def test_a_render_of_350_walkers_stays_cheap() -> None:
    run = replay(seconds=2.0, recording=False)
    render_ms = 1e3 * float(np.median(run.timings["render"]))
    # generous, for a shared or slow machine: it catches a render that grew by a factor, not by a few percent (350 walkers rendered
    # in 15 to 18 ms, the median, on a busy shared machine; the 50 Hz render and the 20 Hz receipt share one GIL with the env node)
    assert render_ms < 40.0, f"median render {render_ms:.2f} ms"


if __name__ == "__main__":
    if sys.argv[1:] == ["--record"]:
        recorded = replay()
        GOLDEN.write_text(json.dumps({"counts": recorded.counts, "streams": {stream: fp.value() for stream, fp in recorded.fingerprint.items()}}, separators=(",", ":")) + "\n")
        print(f"wrote {GOLDEN}")
        sys.exit(0)
    walkers = int(sys.argv[1]) if len(sys.argv) > 1 else 350
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    run = replay(walkers, seconds, recording=False)
    for path, samples in run.timings.items():
        ms = 1e3 * np.asarray(samples)
        print(f"{path:12s} n={len(ms):4d} median {np.median(ms):7.3f} ms  mean {ms.mean():7.3f} ms  p95 {np.percentile(ms, 95):7.3f} ms")
    recorded = replay(walkers, seconds)
    print(recorded.counts)
    for stream, fingerprint in recorded.fingerprint.items():
        print(f"{stream} {fingerprint.exact.hexdigest()} {len(fingerprint.chunks)} messages")
