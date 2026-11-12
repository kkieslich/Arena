"""The use-point check: what HumanSim's crowd cannot reach in a world and its scenario, on worlds with annotation-only local assets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml
from arena_simulation_setup.metrics.use_points import Finding, check_scenario, describe, line_slots, object_points
from arena_simulation_setup.shared import Obstacle
from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.utils.face import front_angle, queue_theta

pytest.importorskip("scipy")

#: a dispenser 0.4 x 0.34 m with its front at +x, as Hospital/SM_WaterDispenser_01a
FOUNTAIN = {"face": "+x", "bounding_box": [[-0.2, 0.2], [-0.17, 0.17], [0.0, 1.5]]}
#: a desk 1.0 x 0.8 m without a face
DESK = {"bounding_box": [[-0.5, 0.5], [-0.4, 0.4], [0.0, 0.8]]}
LINE = {"type": "line", "params": {"base_step": 1.0, "front_offset": 0.8}}


def _fountain(x: float, y: float, yaw: float, name: str = "fountain") -> dict[str, Any]:
    return {"name": name, "model": "Common/fountain", "pose": [x, y, yaw], "humansim": {"type": "fountain", "capacity": 1, "satisfies": {"thirst": 100.0}, "formation": LINE}}


def _world(tmp_path: Path, scenario: dict[str, Any], *, walls: list[list[list[float]]] = (), door: dict[str, Any] | None = None, objects: list[dict[str, Any]] = ()) -> WorldView:
    """A 6 x 4 m room west of a 8 x 4 m hall, a 1.5 m door between them, the models as world-local assets (a marker without annotation.yaml)."""
    root = tmp_path / "w"
    for name, annotation in (("fountain", FOUNTAIN), ("desk", DESK), ("marker", None)):
        directory = root / "assets" / "Common" / "Object" / name
        directory.mkdir(parents=True)
        if annotation is None:
            (directory / f"{name}.usdz").write_bytes(b"")
        else:
            (directory / "annotation.yaml").write_text(yaml.safe_dump(annotation))
    level = {
        "id": "0",
        "zones": [{"name": "room", "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]]}, {"name": "hall", "polygon": [[6, 0], [14, 0], [14, 4], [6, 4]]}],
        "walls": [{"path": [[0, 0], [14, 0], [14, 4], [0, 4]], "closed": True}, {"path": [[6, 0], [6, 4]]}, *({"path": path} for path in walls)],
        "doors": [door or {"name": "door", "span": [[6, 1.0], [6, 2.5]]}],
        "objects": list(objects),
    }
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [level]}))
    (root / "scenarios" / "s").mkdir(parents=True)
    (root / "scenarios" / "s" / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", **scenario}))
    return WorldView(root)


def _points(findings: list[Finding]) -> list[tuple[str, str]]:
    return [(f.object, f.point) for f in findings]


def test_a_fountain_facing_into_the_room_passes(tmp_path: Path) -> None:
    assert check_scenario(_world(tmp_path, {"objects": [_fountain(0.25, 2.0, 0.0)]}), "s") == []


def test_a_fountain_facing_the_wall_fails_at_its_use_point_and_first_three_slots(tmp_path: Path) -> None:
    findings = check_scenario(_world(tmp_path, {"objects": [_fountain(0.25, 2.0, math.pi)]}), "s")
    assert _points(findings) == [("fountain", "use point"), *(("fountain", f"queue slot {k}") for k in (1, 2, 3))]
    use = findings[0]
    assert (use.x, use.y) == pytest.approx((-0.55, 2.0)) and use.why == "lies off the floor or on a wall" and use.distance >= use.reach == 0.5
    assert "fountain: use point at (-0.55, 2.00) on level 0 lies off the floor or on a wall" in str(use)


def test_a_desk_in_front_of_the_use_point_is_named(tmp_path: Path) -> None:
    desk = {"name": "desk", "model": "Common/desk", "pose": [1.15, 2.0, 0.0]}
    (finding,) = [f for f in check_scenario(_world(tmp_path, {"objects": [_fountain(0.25, 2.0, 0.0), desk]}), "s") if f.point == "use point"]
    assert finding.why == "lies inside the box of desk"


def test_a_queue_slot_behind_a_wall_from_the_one_before_fails(tmp_path: Path) -> None:
    findings = check_scenario(_world(tmp_path, {"objects": [_fountain(0.25, 2.0, 0.0)]}, walls=[[[2.5, 1.7], [2.5, 2.3]]]), "s")
    assert _points(findings) == [("fountain", "queue slot 2")]
    assert findings[0].why == "lies behind a wall from the object or the queue slot before it"


def test_a_room_shut_by_a_locked_door_cuts_off_its_fountain_and_its_floor(tmp_path: Path) -> None:
    door = {"name": "door", "span": [[6, 1.0], [6, 2.5]], "semantics": [{"preset": "gate", "params": {"locked": True, "authorized": []}}]}
    open_door = _world(tmp_path / "open", {"objects": [_fountain(0.25, 2.0, 0.0)]})
    shut = _world(tmp_path / "shut", {"objects": [_fountain(0.25, 2.0, 0.0)]}, door=door)
    assert check_scenario(open_door, "s") == []
    findings = check_scenario(shut, "s")
    assert ("fountain", "use point") in _points(findings) and ("room", "floor") in _points(findings)
    assert next(f for f in findings if f.point == "use point").why == "lies on floor its walkers cannot reach"


def test_a_counter_across_the_doorway_cuts_off_the_room_behind_it(tmp_path: Path) -> None:
    counter = {"name": "counter", "model": "Common/desk", "pose": [6.6, 1.75, math.pi / 2]}
    findings = check_scenario(_world(tmp_path, {"objects": [_fountain(0.25, 2.0, 0.0)]}, objects=[counter]), "s")
    assert ("room", "floor") in _points(findings)


def test_a_model_that_does_not_resolve_is_named_once_with_what_the_buckets_answered(tmp_path: Path) -> None:
    lost = [{"name": f"cooler_{i}", "model": "Common/no_such_model", "pose": [3.0, 1.0 + i, 0.0]} for i in range(3)]
    findings = check_scenario(_world(tmp_path, {"objects": lost}), "s")
    assert [(f.object, f.point) for f in findings] == [(f"cooler_{i}", "model") for i in range(3)] and all(math.isnan(f.distance) for f in findings)
    # the test session's asset network is off
    assert findings[0].why == "'no_such_model' does not resolve (the asset network is off and no asset root holds it)"
    assert describe(findings, limit=2) == "model 'no_such_model' does not resolve (the asset network is off and no asset root holds it), HumanSim never gets its 3 object(s) and walkers pass through them: cooler_0, cooler_1, ..."


def test_a_marker_without_annotation_is_a_target_without_an_obstacle(tmp_path: Path) -> None:
    exit_sign = {"name": "exit", "model": "Common/marker", "pose": [13.8, 2.0, 0.0], "humansim": {"type": "exit"}}
    human = {"name": "visitor", "model": "worker_male_asian_middleage", "pose": [10.0, 2.0, 0.0], "agent": {"agent_type": "./leaver.yaml"}}
    view = _world(tmp_path, {"objects": [exit_sign], "humans": [human]})
    leaver = {"name": "leaver", "mode": "behavior_tree", "initial_sequence": "leave", "sequences": {"leave": {"steps": {"out": {"kind": "go_to", "target": "exit"}}}}}
    (view.path / "scenarios" / "s" / "leaver.yaml").write_text(yaml.safe_dump(leaver))
    assert check_scenario(view, "s") == []
    exit_sign["pose"] = [16.0, 2.0, 0.0]
    (view.path / "scenarios" / "s" / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", "objects": [exit_sign], "humans": [human]}))
    (finding,) = check_scenario(view, "s")
    assert (finding.object, finding.point) == ("exit", "target")


def test_a_pocket_the_walls_close_off_by_themselves_is_no_room_cut_off(tmp_path: Path) -> None:
    # a 2.5 x 3.5 m closet in the hall's corner without a door (4.8 m2 of walkable floor) and a sink in the hall: nobody is sent into the closet
    sink = {"out": {"type": "sink", "polygon": [{"x": 8.0, "y": 2.5}, {"x": 9.0, "y": 2.5}, {"x": 9.0, "y": 3.5}, {"x": 8.0, "y": 3.5}]}}
    assert check_scenario(_world(tmp_path / "walls", {"regions": sink}, walls=[[[11.5, 0.0], [11.5, 3.5], [14.0, 3.5]]]), "s") == []
    # the same closet shut by desks across its opening instead of a wall counts
    desks = [{"name": f"desk_{i}", "model": "Common/desk", "pose": [11.5, y, math.pi / 2]} for i, y in enumerate((0.5, 1.5, 2.5, 3.0))]
    findings = check_scenario(_world(tmp_path / "desks", {"regions": sink, "objects": desks}, walls=[[[11.5, 3.5], [14.0, 3.5]]]), "s")
    assert [(f.object, f.point) for f in findings] == [("hall", "floor")]


def test_a_walker_reaches_only_the_floor_it_starts_on(tmp_path: Path) -> None:
    door = {"name": "door", "span": [[6, 1.0], [6, 2.5]], "semantics": [{"preset": "gate", "params": {"locked": True}}]}
    human = {"name": "visitor", "model": "worker_male_asian_middleage", "pose": [10.0, 2.0, 0.0], "waypoints": [[10.0, 3.0], [3.0, 2.0]]}
    findings = check_scenario(_world(tmp_path, {"humans": [human]}, door=door), "s")
    assert _points(findings) == [("visitor", "waypoint 1"), ("room", "floor")]


def test_a_sink_off_the_floor_and_a_source_on_a_wall_fail(tmp_path: Path) -> None:
    regions = {
        "in": {"type": "source", "polygon": [{"x": 13.9, "y": 1.0}, {"x": 14.6, "y": 1.0}, {"x": 14.6, "y": 1.5}, {"x": 13.9, "y": 1.5}]},
        "out": {"type": "sink", "polygon": [{"x": 15.0, "y": 1.0}, {"x": 16.0, "y": 1.0}, {"x": 16.0, "y": 2.0}, {"x": 15.0, "y": 2.0}]},
        "back": {"type": "sink", "polygon": [{"x": 12.0, "y": 1.0}, {"x": 13.0, "y": 1.0}, {"x": 13.0, "y": 2.0}, {"x": 12.0, "y": 2.0}]},
    }
    assert _points(check_scenario(_world(tmp_path, {"regions": regions}), "s")) == [("out", "sink"), ("in", "source")]


def _flat(points: list[tuple[float, float]]) -> list[float]:
    return [c for point in points for c in point]


def test_the_queue_follows_humansims_line_from_the_turned_pose() -> None:
    obstacle = Obstacle.parse(_fountain(1.0, 2.0, math.pi / 2))
    theta = queue_theta(math.pi / 2, front_angle("+x"))
    ahead = _flat([(1.0, 2.8 + k) for k in range(4)])
    assert _flat([(p.x, p.y) for p in object_points(obstacle, FOUNTAIN)]) == pytest.approx(ahead)
    assert _flat(line_slots(1.0, 2.0, theta, LINE["params"])) == pytest.approx(ahead)
    # without a face the line forms behind the yaw
    assert _flat([(p.x, p.y) for p in object_points(obstacle, {})]) == pytest.approx(_flat([(1.0, 1.2 - k) for k in range(4)]))
