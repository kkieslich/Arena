"""The goal-reply-delay fault switch: off without its variable, once for the named robot from the given sim time on."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_runtime.env import faults

    monkeypatch.setattr(faults, "_spent", set())


def test_off_without_the_variable() -> None:
    from arena_runtime.env import faults

    assert faults._goal_reply_delay("jackal_1", 900.0, None) == 0.0


def test_once_for_the_named_robot_from_its_sim_time() -> None:
    from arena_runtime.env import faults

    spec = "jackal_1@600=25"
    assert faults._goal_reply_delay("jackal_1", 599.0, spec) == 0.0
    assert faults._goal_reply_delay("jackal_2", 700.0, spec) == 0.0
    assert faults._goal_reply_delay("jackal_1", 700.0, spec) == 25.0
    assert faults._goal_reply_delay("jackal_1", 800.0, spec) == 0.0


def test_a_star_takes_the_first_robot_to_dispatch() -> None:
    from arena_runtime.env import faults

    assert faults._goal_reply_delay("jackal_3", 650.0, "*@600=25") == 25.0
    assert faults._goal_reply_delay("jackal_0", 700.0, "*@600=25") == 0.0


def test_only_the_process_environment_can_turn_it_on() -> None:
    """No parameter, launch argument or suite config maps onto the switch: its name appears in the faults module only."""
    from pathlib import Path

    import arena_runtime
    from arena_runtime.env import faults

    root = Path(arena_runtime.__file__).resolve().parent
    readers = [p for p in root.rglob("*") if p.suffix in {".py", ".yaml", ".yml", ".json"} and faults.GOAL_REPLY_DELAY_ENV in p.read_text(errors="ignore")]
    assert readers == [Path(faults.__file__).resolve()]
    launch = root.parent / "launch"
    assert not any(faults.GOAL_REPLY_DELAY_ENV in p.read_text(errors="ignore") for p in launch.rglob("*.py")) if launch.is_dir() else True
