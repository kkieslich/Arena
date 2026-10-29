from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.flowedit.build import build
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.kg import Scene, Value, dump, load, write

pytest.importorskip("numpy")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
QUEUE = HOSPITAL / "scenarios" / "queue" / "scenario.yaml"


def _facts(scene: Scene, subject: str, verb: str) -> list[tuple[Value, dict[str, Value]]]:
    return [(f.object, dict(f.params)) for f in scene.facts if f.subject == subject and f.verb == verb]


@pytest.fixture(scope="module")
def queue() -> Scene:
    return build(HOSPITAL, QUEUE)


def test_queue_fountain_carries_pose_demand_and_dwell(queue: Scene) -> None:
    assert queue.uses == ["core@2", "space@1", "flow@1"]
    assert queue.entities["obj:fountain"].props == {"model": "Hospital/SM_WaterDispenser_01a"}
    assert _facts(queue, "obj:fountain", "at") == [("floor:0", {"x": 11.0, "y": 10.0, "yaw": 1.57})]
    assert _facts(queue, "obj:fountain", "serves") == [("service:fountain", {"capacity": 1})]
    assert _facts(queue, "obj:fountain", "satisfies") == [("need:thirst", {"amount": 100.0})]
    assert _facts(queue, "obj:fountain", "has_formation") == [("formation:line", {"base_step": 1.0, "front_offset": 0.8})]
    assert _facts(queue, "obj:fountain", "dwells") == [(5.0, {"std": 1.5, "clip_low": 4.0, "clip_high": 6.0, "from": "colleague.yaml#drink.queue_and_drink"})]
    assert ("obj:fountain", "contains") in {(str(f.object), f.verb) for f in queue.facts if f.subject == "room:central_hallway"}


def test_queue_rooms_all_have_a_purpose(queue: Scene) -> None:
    rooms = [id for id in queue.entities if id.startswith("room:")]
    purposes = {f.subject: str(f.object) for f in queue.facts if f.verb == "has_purpose"}
    assert rooms and set(rooms) == set(purposes)
    assert purposes["room:central_hallway"] == "purpose:transit"
    assert purposes["room:waiting_area"] == "purpose:break"


def test_zone_posed_object_keeps_its_zone_name(queue: Scene) -> None:
    ((floor, pose),) = _facts(queue, "obj:reception_desk", "at")
    assert floor == "floor:0" and pose["pose_ref"] == "reception"
    assert 0.0 <= pose["x"] <= 8.0 and 0.0 <= pose["y"] <= 5.0


def test_saved_graph_loads_back_the_same(queue: Scene, tmp_path: Path) -> None:
    path = tmp_path / "flowgraph.yaml"
    write(queue, path)
    assert dump(load(path)) == dump(queue)


def test_same_seed_builds_the_same_graph(queue: Scene) -> None:
    assert dump(build(HOSPITAL, QUEUE, seed=0)) == dump(queue)


def test_cli_writes_the_graph_where_asked(tmp_path: Path) -> None:
    out = tmp_path / "graph.yaml"
    assert main(["build", "hospital_1", "queue", "--worlds", str(WORLDS), "--out", str(out), "--write"]) == 0
    assert "obj:fountain" in load(out).entities


def test_cli_names_a_missing_scenario(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["build", "hospital_1", "no_such_scenario", "--worlds", str(WORLDS)]) == 1
    assert "no scenario 'no_such_scenario'" in capsys.readouterr().err


def _small_world(root: Path, kg_facts: list[dict[str, str]], extra: dict[str, object] | None = None) -> Path:
    root.mkdir()
    zones = [{"name": "kitchen", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor": False}, {"name": "office_a", "polygon": [[4, 0], [8, 0], [8, 4], [4, 4]], "floor": False}]
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": zones}]}))
    (root / "kg.yaml").write_text(yaml.safe_dump({"format": "arena-kg/1", "uses": ["core@2", "space@1"], "entities": [{"id": "room:kitchen"}], "facts": kg_facts}))
    scenario = root / "scenarios" / "s"
    scenario.mkdir(parents=True)
    agent = {"name": "drinker", "mode": "behavior_tree", "extends": "adult", "sequences": {"drink": {"steps": {"sip": {"target": "cooler", "interaction": "USE", "duration": 7.5}}}}}
    (scenario / "drinker.yaml").write_text(yaml.safe_dump(agent))
    objects = [{"name": "cooler_1", "model": "Hospital/SM_WaterDispenser_01a", "pose": "kitchen", "humansim": {"type": "cooler", "capacity": 2, **(extra or {})}}]
    humans = [{"name": "h", "model": "worker_male_asian_middleage", "pose": [6.0, 2.0], "agent": {"agent_type": "./drinker.yaml"}}]
    (scenario / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", "objects": objects, "humans": humans}))
    return scenario / "scenario.yaml"


def test_stated_purpose_wins_and_dwell_matches_by_type(tmp_path: Path) -> None:
    root = tmp_path / "w"
    scenario = _small_world(root, [{"subject": "room:kitchen", "verb": "has_purpose", "object": "purpose:work"}])
    scene = build(root, scenario, seed=3)
    assert _facts(scene, "room:kitchen", "has_purpose") == [("purpose:work", {})]
    assert _facts(scene, "room:office_a", "has_purpose") == [("purpose:work", {"by": "rule"})]
    assert _facts(scene, "obj:cooler_1", "dwells") == [(7.5, {"from": "drinker.yaml#drink.sip"})]
    ((_, pose),) = _facts(scene, "obj:cooler_1", "at")
    assert 0.0 <= pose["x"] <= 4.0 and pose["pose_ref"] == "kitchen"


def test_seats_are_noted_and_left_out(tmp_path: Path) -> None:
    root = tmp_path / "w"
    notes: list[str] = []
    scene = build(root, _small_world(root, [], {"seats": [{"x": 0.5, "y": 0.0}]}), notes=notes)
    assert "obj:cooler_1: seats are not in flow@1, left out" in notes
    assert _facts(scene, "obj:cooler_1", "serves") == [("service:cooler", {"capacity": 2})]


def test_formation_anchored_at_a_pose_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "w"
    scenario = _small_world(root, [], {"formation": {"type": "line", "anchor": {"pose": [1.0, 1.0]}}})
    with pytest.raises(ValueError, match="anchored at a pose is not in flow@1"):
        build(root, scenario)
