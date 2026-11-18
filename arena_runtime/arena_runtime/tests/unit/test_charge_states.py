"""Start charges drawn from a file of charge states."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from arena_runtime.env import charge_states  # noqa: E402
from arena_runtime.env.constants.rng import EpisodeRng  # noqa: E402

STATES = [[90.0, 80.0, 70.0, 60.0], [50.0, 40.0, 30.0, 20.0], [15.0, 100.0, 65.0, 5.0]]


def _rng(seed: int) -> EpisodeRng:
    rng = EpisodeRng()
    rng.reseed(seed)
    return rng


def _file(tmp_path: Path, states: object = STATES, **extra: object) -> Path:
    path = tmp_path / "states.json"
    path.write_text(json.dumps({"format": charge_states.FORMAT, "states": states, **extra}))
    return path


def test_a_state_is_drawn_per_episode_seed_and_every_robot_takes_its_place_in_it(tmp_path: Path) -> None:
    states = charge_states.load(_file(tmp_path, source={"runs": ["x"]}))
    assert states == tuple(tuple(state) for state in STATES)
    drawn = {seed: charge_states.draw(states, _rng(seed).stream(*charge_states.STREAM)) for seed in range(200)}
    # the same seed draws the same state, and over the seeds every state comes up about equally often
    assert all(charge_states.draw(states, _rng(seed).stream(*charge_states.STREAM)) == drawn[seed] for seed in range(200))
    counts = [sum(state == tuple(want) for state in drawn.values()) for want in STATES]
    assert sum(counts) == 200 and min(counts) > 40
    # the robots of one episode start with the charges of one state, in robot order; a fifth robot is refused
    state = drawn[7]
    assert [charge_states.start_charge(state, index) for index in range(4)] == list(state)
    with pytest.raises(ValueError, match="robot 4 of the fleet has no place in a charge state of 4 robots"):
        charge_states.start_charge(state, 4)
    # the draw does not touch the stream each robot's own start charge comes from
    rng = _rng(7)
    before = float(_rng(7).stream("robots", "soc", "jackal").uniform(40.0, 100.0))
    charge_states.draw(states, rng.stream(*charge_states.STREAM))
    assert float(rng.stream("robots", "soc", "jackal").uniform(40.0, 100.0)) == before


def test_a_changed_file_is_read_again(tmp_path: Path) -> None:
    path = _file(tmp_path)
    assert len(charge_states.load(path)) == 3
    path.write_text(json.dumps({"format": charge_states.FORMAT, "states": [[1.0, 2.0]]}))
    import os

    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000))
    assert charge_states.load(path) == ((1.0, 2.0),)


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"format": "other", "states": STATES}, "format: expected"),
        ({"format": charge_states.FORMAT, "states": []}, "no states"),
        ({"format": charge_states.FORMAT, "states": [[90.0], [80.0, 70.0]]}, "different numbers of robots"),
        ({"format": charge_states.FORMAT, "states": [[120.0]]}, "state 0 is no list of charges"),
        ({"format": charge_states.FORMAT, "states": [[True]]}, "state 0 is no list of charges"),
        ({"format": charge_states.FORMAT, "states": [[]]}, "state 0 is no list of charges"),
        ([], "format: expected"),
    ],
)
def test_a_file_that_holds_no_charge_states_is_refused(doc: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        charge_states.parse(doc)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no file"):
        charge_states.load(tmp_path / "none.json")
