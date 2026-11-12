"""The adapter's episode RNG streams are keyed on stable names: the same seed gives the same pedestrians in any env."""

from __future__ import annotations

import types

import pytest

pytest.importorskip("rclpy")

from arena_humansim_msgs.msg import AgentMeta

from arena_simulation_setup.utils.affordance import placed_slots

from arena_runtime.env.constants.rng import EpisodeRng
from arena_runtime.env.realizer import Realizer
from arena_runtime.humans.arena_humansim.agent import ArenaHumanDynamicObstacle
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.humans.arena_humansim.rng_keys import AgentSource, flow_model_key, seats_key, walker_key
from arena_runtime.shared import Orientation, Pose, Position

# a sofa whose third sit slot faces one of two yaws, drawn from the object's seats stream
SOFA = placed_slots(
    {
        "sit": [
            {"bbox": [[-0.46, 0.48], [0.03, 0.72], [0.0, 0.47]], "yaw": "-y"},
            {"bbox": [[-1.30, -0.56], [-1.13, -0.26], [0.0, 0.47]], "yaw": ["+x", "-y"]},
        ]
    }
)


def _rng(seed: int) -> EpisodeRng:
    rng = EpisodeRng()
    rng.reseed(seed)
    return rng


def _realizer(prefix: str) -> Realizer:
    return Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=prefix))


def _walker(seed: int, sim_path: str, prefix: str, before: tuple[str, ...] = ()) -> tuple[float, ...]:
    rng = _rng(seed)
    realizer = _realizer(prefix)
    for other in before:
        # another person's draw first: a spawn order that differs between runs
        ArenaHumanDynamicObstacle(agent_type="adult").sample_params(rng.stream(*walker_key(realizer.unprefix(other))))
    params = ArenaHumanDynamicObstacle(agent_type="adult").sample_params(rng.stream(*walker_key(realizer.unprefix(sim_path))))
    assert params is not None
    return (params.desired_velocity, params.agent_radius, params.perception.vision_range, params.perception.vision_fov, *sorted(params.local_planner_params.values()), params.handedness)


def _adapter(seed: int, prefix: str) -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._NodeInterface__node = types.SimpleNamespace(conf=types.SimpleNamespace(General=types.SimpleNamespace(RNG=_rng(seed))))  # NodeInterface keeps it private
    adapter._realizer = _realizer(prefix)
    adapter._pose_profiles = {}
    adapter._agent_type_paths = {}
    return adapter


def _seat_yaws(seed: int, prefix: str) -> list[float]:
    sofa = types.SimpleNamespace(humansim={}, sim_path=f"{prefix}/sofa", pose=Pose(Position(1.0, 2.0, 0.0), Orientation.from_yaw(0.3)))
    poses, _, _, _ = _adapter(seed, prefix)._seat_poses(sofa, SOFA, None)
    return [round(p.theta, 9) for p in poses]


def test_unprefix_drops_only_the_own_env_prefix() -> None:
    assert _realizer("env_1").unprefix("env_1/flow_3") == "flow_3"
    assert _realizer("env_12").unprefix("/env_12/sofa/L1") == "sofa/L1"
    assert _realizer("env_1").unprefix("env_12/sofa") == "env_12/sofa"
    assert _realizer("").unprefix("flow_3/") == "flow_3"
    assert _realizer("env_1").unprefix(_realizer("env_1").prefix("flow_3")) == "flow_3"


def test_walker_and_seat_keys_name_the_person_not_the_env() -> None:
    assert walker_key(_realizer("env_1").unprefix("env_1/flow_3")) == walker_key(_realizer("env_2").unprefix("env_2/flow_3")) != seats_key("flow_3")


def test_a_person_draws_the_same_walker_in_any_env_with_the_same_seed() -> None:
    assert _walker(7, "env_1/flow_3", "env_1") == _walker(7, "env_2/flow_3", "env_2") == _walker(7, "flow_3", "")
    assert _walker(7, "env_1/flow_3", "env_1") != _walker(8, "env_1/flow_3", "env_1")
    assert _walker(7, "env_1/flow_3", "env_1") != _walker(7, "env_1/flow_4", "env_1")


def test_a_walker_does_not_depend_on_who_else_was_drawn_before() -> None:
    alone = _walker(7, "env_1/flow_3", "env_1")
    assert _walker(7, "env_1/flow_3", "env_1", before=("env_1/flow_1", "env_1/flow_2")) == alone
    assert _walker(7, "env_2/flow_3", "env_2", before=("env_2/flow_9",)) == alone


def test_seats_follow_the_seed_not_the_env() -> None:
    for seed in range(10):
        assert _seat_yaws(seed, "env_1") == _seat_yaws(seed, "env_2")
    assert len({tuple(_seat_yaws(seed, "env_1")) for seed in range(10)}) > 1


def test_flow_models_are_keyed_on_source_and_release_index() -> None:
    assert flow_model_key(41, AgentSource("door_a", 4, "door_a_4")) == flow_model_key(57, AgentSource("door_a", 4, "x")) != flow_model_key(41, AgentSource("door_a", 5))
    assert flow_model_key(41, AgentSource(name="flow_1")) == flow_model_key(9, AgentSource(name="flow_1"))
    assert flow_model_key(41, AgentSource()) != flow_model_key(42, AgentSource())
    adapter = _adapter(0, "env_3")
    adapter._agent_meta_callback(AgentMeta(agent_id=[41, 42], name=["env_3/flow_1", ""], handedness=["", ""], agent_type=["", ""], source_name=["", "door_a"], source_seq=[0, 4]))
    assert adapter._agent_sources == {41: AgentSource("", 0, "flow_1"), 42: AgentSource("door_a", 4, "")}
    adapter._agent_meta_callback(AgentMeta(agent_id=[1], name=["a"], handedness=[""]))
    assert adapter._agent_sources == {1: AgentSource("", 0, "a")}
