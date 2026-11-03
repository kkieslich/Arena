from __future__ import annotations

import pytest

from arena_simulation_setup.shared.entities import (
    CustomDynamicObstacle,
    DynamicObstacle,
    Entity,
    Named,
    Obstacle,
)
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.tree.World.Scenario import parse_scenario, upgrade_scenario
from arena_simulation_setup.utils.geometry import Pose, Scale


def _model_id():
    return ObjectIdentifier("test_model")


def _make_entity(**kwargs) -> Entity:
    defaults = dict(
        name="test_entity",
        pose=Pose(),
        model=_model_id(),
    )
    defaults.update(kwargs)
    return Entity(**defaults)


def _make_obstacle(**kwargs) -> Obstacle:
    defaults = dict(
        name="test_obstacle",
        pose=Pose(),
        model=_model_id(),
        scale=None,
    )
    defaults.update(kwargs)
    return Obstacle(**defaults)


# ---------------------------------------------------------------------------
# Named
# ---------------------------------------------------------------------------


def test_named_sim_path_default_is_name():
    n = Named(name="foo")
    assert n.sim_path == "foo"


def test_named_sim_path_override():
    n = Named(name="foo", extra={"sim_path": "bar"})
    assert n.sim_path == "bar"


def test_named_sim_path_setter():
    n = Named(name="foo")
    n.sim_path = "custom"
    assert n.sim_path == "custom"
    assert n.extra["sim_path"] == "custom"


def test_named_extra_dict_preserved():
    n = Named(name="test", extra={"key": "value"})
    assert n.extra["key"] == "value"


def test_named_serialize_omits_empty_extra():
    n = Named(name="hello")
    d = n.serialize()
    assert "extra" not in d or not d["extra"]


def test_named_serialize_leaves_out_runtime_extra():
    n = Named(name="hello", extra={"custom_key": 42})
    d = n.serialize()
    assert d == {"name": "hello"}
    assert n.extra == {"custom_key": 42}


def test_named_parse_rejects_the_pos_alias_and_extra():
    with pytest.raises(ValueError, match=r"unknown keys \['pos'\]"):
        Named.parse({"name": "x", "pos": [1.0, 2.0]})
    with pytest.raises(ValueError, match=r"unknown keys \['extra'\]"):
        Named.parse({"name": "x", "extra": {"sim_path": "y"}})


def test_legacy_scenario_maps_pos_to_pose():
    upgraded, _ = upgrade_scenario({"static": [{"name": "x", "model": "box", "pos": [1.0, 2.0]}]})
    (obj,) = parse_scenario(upgraded).objects
    assert (obj.pose.position.x, obj.pose.position.y) == pytest.approx((1.0, 2.0))


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


def test_entity_pose_converter():
    e = _make_entity(pose=Pose.parse([1.0, 2.0, 0.5]))
    assert e.pose.position.x == pytest.approx(1.0)


def test_entity_model_is_object_identifier():
    e = _make_entity()
    assert isinstance(e.model, ObjectIdentifier)


def test_entity_extra_is_read_directly():
    e = _make_entity()
    e.extra["my_key"] = "my_val"
    assert e.extra.get("my_key") == "my_val"
    assert e.extra.get("bbox") is None


def test_entity_serialize_leaves_out_extra():
    e = _make_entity()
    e.extra["my_key"] = "my_val"
    d = e.serialize()
    assert "my_key" not in d
    assert "extra" not in d


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------


def test_obstacle_scale_set():
    obs = _make_obstacle()
    obs.scale = Scale(2.0, 2.0, 2.0)
    assert obs.scale.x == pytest.approx(2.0)


def test_obstacle_serialize_returns_dict():
    obs = _make_obstacle()
    d = obs.serialize()
    assert isinstance(d, dict)
    assert d["name"] == obs.name


# ---------------------------------------------------------------------------
# DynamicObstacle
# ---------------------------------------------------------------------------


def test_dynamic_obstacle_empty_waypoints():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    do = DynamicObstacle(
        name="dyn",
        pose=Pose(),
        model=HumanIdentifier.converter("some_pedestrian"),
        waypoints=[],
        velocity=1.5,
    )
    assert do.waypoints == []
    assert do.velocity == pytest.approx(1.5)


def test_dynamic_obstacle_velocity_converter_string():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    do = DynamicObstacle(
        name="dyn",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
        velocity="2.5",
    )
    assert do.velocity == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# CustomDynamicObstacle
# ---------------------------------------------------------------------------


def test_custom_dynamic_obstacle_getattr_hit():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    cdo = CustomDynamicObstacle(
        name="cdo",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
    )
    cdo.extra["custom_prop"] = 99
    assert cdo.custom_prop == 99


def test_custom_dynamic_obstacle_getattr_miss():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    cdo = CustomDynamicObstacle(
        name="cdo",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
    )
    with pytest.raises(AttributeError):
        _ = cdo.nonexistent_field


def test_custom_dynamic_obstacle_parse_rejects_unknown_fields():
    data = {
        "name": "cdo",
        "pose": [0.0, 0.0],
        "model": "p",
        "custom_field": "hello",
    }
    with pytest.raises(ValueError, match=r"unknown keys \['custom_field'\]"):
        CustomDynamicObstacle.parse(data)
    cdo = CustomDynamicObstacle(name="cdo", pose=Pose(), model="p", extra={"custom_field": "hello"})
    assert cdo.custom_field == "hello"


@pytest.fixture
def bench_assets(tmp_path):
    from arena_simulation_setup.tree import DynamicPaths

    bench = tmp_path / "Hospital" / "Object" / "TestBench"
    bench.mkdir(parents=True)
    (bench / "annotation.yaml").write_text("bounding_box: [[-0.5, 0.5], [0.0, 0.4], [0.0, 0.8]]\naffordances:\n  sit:\n    - {bbox: [[-0.5, 0.5], [0.0, 0.4], [0.0, 0.45]], yaw: -y}\n")
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield tmp_path
    DynamicPaths.ARENA.path = previous


def test_obstacle_affordance_slots_read_the_asset_annotation(bench_assets):
    import asyncio

    slots = asyncio.run(Obstacle.parse({"name": "bench", "model": "Hospital/TestBench", "pose": [1.0, 2.0, 0.0]}).affordance_slots())
    assert [(slot.kind, slot.index, slot.bbox[2]) for slot in slots] == [("sit", 0, (0.0, 0.45))]


def test_obstacle_affordance_slots_are_empty_for_an_unresolvable_model(bench_assets):
    import asyncio

    assert asyncio.run(Obstacle.parse({"name": "ghost", "model": "Hospital/TestMissingSeat", "pose": [0.0, 0.0, 0.0]}).affordance_slots()) == []


@pytest.mark.parametrize(
    "joints",
    [
        "open",
        0.5,
        {"door": 1.0, "drawer": "open"},
        {"openness": {"door": 0.5}, "setting": {"dial": "high"}, "pos": {"knob": 0.25}, "q": {"wheel": -1.0}},
    ],
)
def test_obstacle_joints_keep_each_form_and_serialize_it_back(joints):
    obstacle = Obstacle.parse({"name": "cabinet", "model": "cabinet", "pose": [0.0, 0.0, 0.0], "joints": joints})
    assert obstacle.joints == joints
    assert obstacle.serialize()["joints"] == joints
    assert Obstacle.parse(obstacle.serialize()).joints == joints


def test_obstacle_joints_read_integers_as_floats_and_default_to_none():
    base = {"name": "cabinet", "model": "cabinet", "pose": [0.0, 0.0, 0.0]}
    assert Obstacle.parse({**base, "joints": 1}).joints == 1.0
    assert Obstacle.parse({**base, "joints": {"door": 1}}).joints == {"door": 1.0}
    assert Obstacle.parse({**base, "joints": {"q": {"door": 1}}}).joints == {"q": {"door": 1.0}}
    assert Obstacle.parse(base).joints is None
    assert "joints" not in Obstacle.parse(base).serialize()


@pytest.mark.parametrize(
    ("joints", "message"),
    [
        ([1.0], "joints must be a state name, an openness in \\[0, 1\\], a mapping of joint names to values or state names, or a mapping of families \\(openness, pos, setting, q\\)"),
        (True, "joints must be a state name"),
        (1.5, "joints as a number is an openness in \\[0, 1\\], got 1.5"),
        ({"door": True}, "joint 'door' takes a number or a state name"),
        ({"openness": {"door": 0.5}, "lid": 1.0}, "joints mixes family keys \\['openness'\\] with joint names"),
        ({"openness": 0.5}, "joints family openness maps its joints to fractions, detent names or values, got 0.5"),
        ({"setting": {"dial": None}}, "joint 'dial' takes a number or a state name"),
    ],
)
def test_obstacle_joints_refuse_other_values(joints, message):
    with pytest.raises(ValueError, match=message):
        Obstacle.parse({"name": "cabinet", "model": "cabinet", "pose": [0.0, 0.0, 0.0], "joints": joints})


def test_obstacle_still_refuses_unknown_keys_next_to_joints():
    with pytest.raises(ValueError, match=r"unknown keys \['joint'\]"):
        Obstacle.parse({"name": "cabinet", "model": "cabinet", "pose": [0.0, 0.0, 0.0], "joints": "open", "joint": "open"})


@pytest.fixture
def cabinet_assets(tmp_path):
    from arena_simulation_setup.tree import DynamicPaths

    cabinet = tmp_path / "Kitchen" / "Object" / "TestSweptCabinet"
    cabinet.mkdir(parents=True)
    articulation = "{links: {body: [], door: []}, joints: [{name: door, type: revolute, parent: body, child: door, origin: [0.5, -0.3, 0.0], axis: [0, 0, 1], limit: [0.0, 1.57]}], swept: [[-0.5, 0.5], [-1.3, 0.3], [0.0, 1.0]]}"
    (cabinet / "annotation.yaml").write_text(f"bounding_box: [[-0.5, 0.5], [-0.3, 0.3], [0.0, 1.0]]\narticulation: {articulation}\n")
    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield tmp_path
    DynamicPaths.ARENA.path = previous


def test_obstacle_footprint_and_span_cover_the_swept_box_through_the_full_pose(cabinet_assets):
    import asyncio
    import math

    from arena_simulation_setup.utils.geometry import Orientation, Position

    upright = Obstacle(name="cabinet", model="Kitchen/TestSweptCabinet", pose=Pose(position=Position(2.0, 1.0, 0.0), orientation=Orientation.from_yaw(math.pi / 2)))
    footprint = asyncio.run(upright.footprint())
    assert footprint is not None
    assert footprint.bounds == pytest.approx((1.7, 0.5, 3.3, 1.5))
    assert footprint.area == pytest.approx(1.6)
    assert asyncio.run(upright.span()) == pytest.approx((0.0, 1.0))
    lying = Obstacle(name="cabinet", model="Kitchen/TestSweptCabinet", pose=Pose(position=Position(0.0, 0.0, 0.5), orientation=Orientation.from_euler((0.0, math.pi / 2, 0.0))))
    footprint = asyncio.run(lying.footprint())
    assert footprint is not None
    assert footprint.bounds == pytest.approx((0.0, -1.3, 1.0, 0.3))
    assert asyncio.run(lying.span()) == pytest.approx((0.0, 1.0))


def test_obstacle_footprint_and_span_are_none_for_an_unresolvable_model(cabinet_assets):
    import asyncio

    ghost = Obstacle(name="ghost", model="Kitchen/TestMissingCabinet", pose=Pose())
    assert asyncio.run(ghost.footprint()) is None
    assert asyncio.run(ghost.span()) is None
