"""Every shipped world and scenario passes the check the arena human backend runs at episode start (`use_points.check` with the runtime's seat standing points): no use point, queue slot, seat, source, sink, start, waypoint or room its crowd cannot reach.

Models resolve through the asset buckets and their cache, as the runtime resolves them; whether every model resolves is
`arena_simulation_setup`'s `test_world_models.py`.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from arena_runtime.humans.support import seat_stand
from arena_simulation_setup.metrics.use_points import check_scenario, check_world
from arena_simulation_setup.shared import Obstacle
from arena_simulation_setup.tree import Identifier
from arena_simulation_setup.tree.World import WorldView

WORLDS = Path(__file__).resolve().parents[3] / "arena_simulation_setup" / "worlds"
#: a bucket model every FLOWEDIT world uses, resolved first to see whether the buckets or their cache are at hand
PROBE = "Hospital/SM_WaterDispenser_01a"


def _cases() -> list[tuple[str, str | None]]:
    cases: list[tuple[str, str | None]] = []
    for world in sorted(p for p in WORLDS.iterdir() if p.is_dir()):
        scenarios = sorted(p.name for p in (world / "scenarios").iterdir() if (p / "scenario.yaml").is_file() or (p / "scenario.json").is_file()) if (world / "scenarios").is_dir() else []
        cases += [(world.name, scenario) for scenario in scenarios] or [(world.name, None)]
    return cases


@pytest.fixture(scope="module")
def bucket_models() -> Iterator[None]:
    """The asset buckets and their cache for this module, not the offline stand-ins of the test session."""
    previous = os.environ.pop("ARENA_ASSETS_NET", None)
    try:
        try:
            Identifier._run_sync(Obstacle.parse({"name": "probe", "model": PROBE, "pose": [0.0, 0.0, 0.0]}).model.resolve())
        except FileNotFoundError:
            pytest.skip(f"bucket models do not resolve here ({PROBE})")
        yield
    finally:
        if previous is not None:
            os.environ["ARENA_ASSETS_NET"] = previous


@pytest.mark.usefixtures("bucket_models")
@pytest.mark.parametrize(("world", "scenario"), _cases())
def test_the_crowd_reaches_every_point_a_shipped_scenario_sends_it_to(world: str, scenario: str | None) -> None:
    view = WorldView(WORLDS / world)
    findings = check_world(view, stand=seat_stand) if scenario is None else check_scenario(view, scenario, stand=seat_stand)
    assert [str(f) for f in findings if f.point != "model" and not f.detour] == []
