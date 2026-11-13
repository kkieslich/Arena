from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import attrs
import pytest
import yaml
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit.build import build
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.generative.flowedit.edit import Site, apply, open_site, save
from arena_simulation_setup.generative.flowedit.interactions import by_rule
from arena_simulation_setup.generative.flowedit.read import Result, _floor, read, read_graph, render, write
from arena_simulation_setup.generative.scenario.materialize import WouldOverwrite
from arena_simulation_setup.kg import dump, load

pytest.importorskip("numpy")
pytest.importorskip("arena_humansim")
pytestmark = pytest.mark.usefixtures("local_root")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
DWELL = {"mean": 5.0, "std": 1.5, "clip_low": 4.0, "clip_high": 6.0}
FOUNTAIN_2 = {"model": "Hospital/SM_WaterDispenser_01a", "x": 20.0, "y": 5.0, "type": "fountain", "capacity": 1, "satisfies": {"thirst": 100.0}}


def _types(result: Result) -> dict[str, dict[str, Any]]:
    return {ref: doc for ref, doc in result.types.items() if "/flow__" in ref}


def _object(result: Result, name: str) -> dict[str, Any]:
    return next(o for o in result.doc["objects"] if o["name"] == name)


def _without_seek(types: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every type with its `seek_*` steps left out: what an edit at an object must not change."""
    return {ref: {**doc, "sequences": {k: v for k, v in doc["sequences"].items() if not k.startswith("seek_")}} for ref, doc in types.items()}


@pytest.fixture(scope="module")
def queue() -> Site:
    return open_site(HOSPITAL, "queue")


@pytest.fixture(scope="module")
def hospital(queue: Site) -> Result:
    return read(queue)


@pytest.fixture(scope="module")
def every_object(queue: Site) -> Result:
    return read(queue, S.load({"synthesize_needs": True}))


def test_hospital_graph_queues_at_the_fountain(hospital: Result) -> None:
    fountain = _object(hospital, "fountain")
    assert fountain["pose"] == [11.0, 10.0, 1.57]
    assert fountain["humansim"]["capacity"] == 1 and fountain["humansim"]["formation"]["type"] == "line"
    thirst = hospital.needs["thirst"]
    assert (thirst.target, thirst.kind, thirst.meet, thirst.dwell) == ("fountain", "USE", False, DWELL)
    people = _types(hospital)
    assert people and len(people) == len(hospital.doc["humans"])
    for doc in people.values():
        assert doc["mode"] == "behavior_tree" and doc["needs"]["thirst"]
        assert {"when": {"thirst": {"below": 30.0}}, "goto": "seek_thirst"} in doc["sequences"]["idle"]["transitions"]
        assert doc["sequences"]["seek_thirst"]["steps"]["use"]["target"] == "fountain"
    assert all(h["name"].startswith("flow_") and "colleague" not in h["agent"]["agent_type"] for h in hospital.doc["humans"])


def test_people_start_at_points_in_their_home_room(queue: Site, hospital: Result) -> None:
    # a zone name would leave the start point to the runtime, which draws it per world: an edit would move the people
    import shapely
    from arena_simulation_setup.generative.flowedit.read import _base

    base = _base(queue, None, S.load(), None, 0, [])
    for person, human in zip(base.crowd.people, hospital.doc["humans"], strict=True):
        x, y, yaw = human["pose"]
        shape = base.floor.shape(person.home)
        assert shape is not None and shape.buffer(1e-6).contains(shapely.Point(x - base.floor.origin[0], y - base.floor.origin[1]))
        assert -3.15 <= yaw <= 3.15
    assert read(queue, seed=1).doc["humans"][0]["pose"] != hospital.doc["humans"][0]["pose"]


def test_a_start_an_edit_covers_moves_to_the_nearest_free_point(queue: Site, hospital: Result) -> None:
    # the base draws every start: an object an edit puts over one moves that person only, everyone else keeps their spot
    from arena_simulation_setup.generative.flowedit.read import PERSON_CLEAR

    first = hospital.doc["humans"][0]
    origin = _floor(queue.root, queue.world, read_graph(queue.graph)).origin
    covered = apply(queue, "insert", "obj:crate_1", {"model": "Hospital/SM_WaterDispenser_01a", "x": first["pose"][0] - origin[0], "y": first["pose"][1] - origin[1]})
    notes: list[str] = []
    edited = read(covered, notes=notes, base=queue)
    floor = _floor(covered.root, covered.world, read_graph(covered.graph))

    def free(pose: list[float]) -> bool:
        return floor.grid.free((pose[0] - floor.origin[0], pose[1] - floor.origin[1]), PERSON_CLEAR)

    moved = {b["name"] for b, e in zip(hospital.doc["humans"], edited.doc["humans"], strict=True) if b["pose"] != e["pose"]}
    assert first["name"] in moved and len(moved) < len(hospital.doc["humans"])
    for before, after in zip(hospital.doc["humans"], edited.doc["humans"], strict=True):
        # covered starts move to a free point with their heading, free ones stay to the centimetre
        assert (before["name"] in moved) == (not free(before["pose"]))
        assert free(after["pose"]) and after["pose"][2] == before["pose"][2]
    assert {n.split()[1] for n in notes if n.startswith("people: ") and "starts at" in n and "moved to" in n} == moved
    assert not any(n.startswith("people: ") and "starts at" in n for n in hospital.notes)


def _stops(result: Result) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Every go_to stop of the people's and visitors' types, by type, sequence and step."""
    return {(ref, seq, name): (step["target_pose"]["x"], step["target_pose"]["y"]) for ref, doc in result.types.items() for seq, spec in doc["sequences"].items() for name, step in (spec.get("steps") or {}).items() if "target_pose" in step}


def test_a_stop_an_edit_covers_moves_to_the_nearest_free_point(queue: Site, hospital: Result) -> None:
    # stops are drawn in the base world like the starts: an object an edit puts over one moves that stop only
    from arena_simulation_setup.generative.flowedit.read import PERSON_CLEAR

    stops = _stops(hospital)
    key, (x, y) = next((k, v) for k, v in stops.items() if "/flow__" in k[0])
    origin = _floor(queue.root, queue.world, read_graph(queue.graph)).origin
    covered = apply(queue, "insert", "obj:crate_1", {"model": "Hospital/SM_WaterDispenser_01a", "x": x - origin[0], "y": y - origin[1]})
    notes: list[str] = []
    edited = _stops(read(covered, notes=notes, base=queue))
    floor = _floor(covered.root, covered.world, read_graph(covered.graph))

    def free(pt: tuple[float, float]) -> bool:
        return floor.grid.free((pt[0] - floor.origin[0], pt[1] - floor.origin[1]), PERSON_CLEAR)

    moved = {k for k in stops if stops[k] != edited[k]}
    assert key in moved and len(moved) < len(stops)
    assert all((k in moved) == (not free(stops[k])) and free(edited[k]) for k in stops)
    assert any(n.startswith("people: ") and "stops at" in n and "moved to" in n for n in notes)


def test_people_start_and_stop_off_the_footprint_of_every_scenario_object(local_root: Path) -> None:
    # station_hall's reception desk is a scenario object with its origin at a corner (its asset's box below): stamped as a
    # 1 m box around its pose, starts and idle stops fell inside it, where nobody can stand and HumanSim never arrives
    import shapely
    from arena_simulation_setup.shared.entities import Obstacle

    desk_model = "Hospital/SM_ReceptionDesk_01a"
    directory = local_root / Obstacle.parse({"name": "probe", "model": desk_model, "pose": [0.0, 0.0, 0.0]}).model.relpath()
    directory.mkdir(parents=True)
    (directory / "annotation.yaml").write_text(yaml.safe_dump({"face": "+x", "bounding_box": [[0.0, 2.4197], [-2.7646, 0.1027], [0.0, 1.1]]}))
    site = open_site(WORLDS / "station_hall", "demand")
    floor = _floor(site.root, site.world, read_graph(site.graph))
    desk = next(s for s in floor.statics if s["name"] == "reception_desk")
    assert desk["model"] == desk_model and desk["resolved"]
    box = shapely.Polygon(desk["polygon"])
    assert box.area == pytest.approx(2.4197 * 2.8673, rel=1e-3) and not box.contains(shapely.Point(*desk["pose"]).buffer(0.6))
    for seed in range(1, 11):
        result = read(site, seed=seed)
        points = [tuple(h["pose"][:2]) for h in result.doc["humans"]] + list(_stops(result).values())
        inside = [(x, y) for x, y in points if box.contains(shapely.Point(x - floor.origin[0], y - floor.origin[1]))]
        assert not inside, (seed, inside)


def test_only_stated_needs_are_sought_by_default(hospital: Result) -> None:
    assert list(hospital.needs) == ["thirst"]
    assert {o["name"] for o in hospital.doc["objects"]} == {"exam_chair_1", "fountain", "gathering_area", "patient_bed_1", "reception_desk"}


def test_hospital_reads_every_interaction_kind(every_object: Result) -> None:
    kinds = {n: (p.target, p.kind, p.meet) for n, p in every_object.needs.items()}
    assert kinds == {
        "thirst": ("fountain", "USE", False),
        "visit_chair": ("exam_chair_1", "SIT_ON", False),
        "visit_desk": ("reception_desk", "QUEUE_USE", False),
        "visit_gathering_area": ("gathering_area", "GROUP_CONVERSATION", True),
        "visit_patient_bed": ("patient_bed_1", "LIE_ON", False),
    }
    seek = next(iter(_types(every_object).values()))["sequences"]["seek_visit_gathering_area"]["steps"]
    assert seek["go"] == {"kind": "go_to", "target": "gathering_area", "patience": S.DEFAULTS["patience"]}
    assert seek["use"]["interaction"] == "GROUP_CONVERSATION" and seek["use"]["duration"]["mean"] == 45.0


def test_patience_outlasts_the_longest_use(every_object: Result) -> None:
    for doc in _types(every_object).values():
        for name, sequence in doc["sequences"].items():
            for step in sequence["steps"].values():
                if "duration" in step and "patience" in step:
                    assert step["patience"]["clip_low"] > step["duration"].get("clip_high", step["duration"]["mean"]), (name, step)


def test_moving_the_fountain_moves_the_queue_and_keeps_the_people(queue: Site, hospital: Result) -> None:
    moved = read(apply(queue, "move", "obj:fountain", {"x": 11.2, "y": 24.0}), base=queue)
    assert _object(moved, "fountain")["pose"] == [11.2, 24.0, 1.57]
    assert moved.needs["thirst"].target == "fountain"
    assert [o for o in moved.doc["objects"] if o["name"] != "fountain"] == [o for o in hospital.doc["objects"] if o["name"] != "fountain"]
    assert moved.doc["humans"] == hospital.doc["humans"] and moved.types == hospital.types


def test_an_edit_changes_the_step_at_the_edited_object_only(queue: Site, hospital: Result) -> None:
    edited = apply(queue, "function", "obj:fountain", {"dwell": {"mean": 20.0, "std": 1.5, "clip_low": 15.0, "clip_high": 25.0}})
    assert edited.name != queue.name
    raised = read(edited, base=queue)
    assert raised.needs["thirst"].dwell == {"mean": 20.0, "std": 1.5, "clip_low": 15.0, "clip_high": 25.0}
    assert raised.doc == hospital.doc
    assert _without_seek(raised.types) == _without_seek(hospital.types)
    for ref, doc in raised.types.items():
        use, before = doc["sequences"]["seek_thirst"]["steps"]["use"], hospital.types[ref]["sequences"]["seek_thirst"]["steps"]["use"]
        assert use["duration"]["mean"] == 20.0 and before["duration"]["mean"] == 5.0
        assert {k: v for k, v in use.items() if k not in ("duration", "patience")} == {k: v for k, v in before.items() if k not in ("duration", "patience")}


def test_an_edited_world_is_read_with_the_people_of_its_base(queue: Site, hospital: Result, local_root: Path) -> None:
    path = save(apply(queue, "function", "obj:fountain", {"dwell": 12.0}))
    reopened = open_site(path)
    assert reopened.base == "hospital_1" and reopened.name == path.name
    result = read(reopened, base=queue)
    assert result.root == path and result.doc["humans"] == hospital.doc["humans"]
    assert _without_seek(result.types) == _without_seek(hospital.types)
    out = write(result, "flowedit_queue")
    assert out == path / "scenarios" / "flowedit_queue" and (out / "scenario.yaml").is_file()
    provenance = yaml.safe_load((out / "provenance.yaml").read_text())
    assert (provenance["world"], provenance["base"], provenance["edits"]) == (path.name, "hospital_1", [reopened.edits[-1].id])


def test_an_unsaved_edit_is_not_written(queue: Site) -> None:
    with pytest.raises(ValueError, match="not saved"):
        write(read(apply(queue, "function", "obj:fountain", {"dwell": 7.0}), base=queue), "flowedit_queue")


def test_the_base_must_be_the_unedited_world(queue: Site) -> None:
    edited = apply(queue, "function", "obj:fountain", {"dwell": 7.0})
    with pytest.raises(ValueError, match="not the unedited base world"):
        read(edited, base=edited)


def test_a_demand_left_behind_is_a_target_without_an_obstacle(queue: Site, hospital: Result) -> None:
    moved = apply(queue, "move", "obj:fountain", {"x": 11.2, "y": 24.0, "leave_demand": True})
    result = read(moved, base=queue)
    model = _object(result, "fountain")
    assert model["pose"] == [11.2, 24.0, 1.57] and "humansim" not in model
    demand = _object(result, "fountain_demand")
    assert demand["model"] == "" and demand["pose"] == [11.0, 10.0, 1.57]
    assert demand["humansim"] == _object(hospital, "fountain")["humansim"]
    assert (result.needs["thirst"].target, result.needs["thirst"].kind) == ("fountain_demand", "USE")
    assert result.doc["humans"] == hospital.doc["humans"]
    floor = _floor(moved.root, moved.world, read_graph(moved.graph))
    assert "fountain_demand" not in {s["name"] for s in floor.statics} and "fountain" in {s["name"] for s in floor.statics}
    assert any("fountain_demand: no model" in note for note in result.notes)


def test_objects_for_one_need_that_differ_are_an_error(queue: Site) -> None:
    other_dwell = apply(queue, "insert", "obj:fountain_2", {**FOUNTAIN_2, "dwell": 9.0})
    with pytest.raises(ValueError, match="need:thirst: the objects that satisfy it differ"):
        read(other_dwell, base=queue)
    no_type = apply(queue, "insert", "obj:fountain_2", {**{k: v for k, v in FOUNTAIN_2.items() if k != "type"}, "type": "kiosk", "dwell": DWELL})
    with pytest.raises(ValueError, match="differ"):
        read(no_type, base=queue)


def test_a_copy_of_an_object_shares_its_need_by_type(queue: Site, hospital: Result) -> None:
    copied = read(apply(queue, "insert", "obj:fountain_2", {**FOUNTAIN_2, "dwell": DWELL}), base=queue)
    thirst = copied.needs["thirst"]
    assert (thirst.target, thirst.kind, thirst.objects) == ("fountain", "USE", ("fountain", "fountain_2"))
    assert copied.doc["humans"] == hospital.doc["humans"]


def test_every_satisfies_goes_into_the_block() -> None:
    showcase = open_site(HOSPITAL, "showcase")
    fact = next(f for f in showcase.graph.facts if f.subject == "obj:reception_desk" and f.verb == "satisfies")
    assert fact.params.get("from")
    result = read(showcase, S.load({"count": 2}))
    assert _object(result, "reception_desk")["humansim"]["satisfies"] == {"checkin": float(fact.params["amount"])}


def test_no_robot_unless_the_settings_name_one(queue: Site, hospital: Result) -> None:
    assert "robots" not in hospital.doc
    assert all("robot" not in note for note in hospital.notes)
    driven = read(queue, S.load({"robot": {"start": "reception", "goal": "patient_ward"}}))
    (robot,) = driven.doc["robots"]
    assert robot["start"] == "reception"


def test_same_site_settings_and_seed_render_identically(queue: Site, hospital: Result) -> None:
    again = read(queue, S.load(), seed=0)
    assert render(again, "flowedit_queue") == render(hospital, "flowedit_queue")
    other = read(queue, S.load(), seed=1)
    assert render(other, "flowedit_queue")["scenario.yaml"] != render(hospital, "flowedit_queue")["scenario.yaml"]


def test_settings_set_the_crowd_not_the_site(queue: Site) -> None:
    settings = S.load({"count": 3, "local_planner": "orca", "synthesize_needs": True, "needs": {"thirst": {"decay_rate": {"mean": 4.0}}}, "entrances": {"rate": 0.0}})
    result = read(queue, settings)
    assert len(result.doc["humans"]) == 3 and not result.doc["regions"]
    doc = next(iter(_types(result).values()))
    assert doc["local_planner"] == "orca"
    assert doc["needs"]["thirst"]["decay_rate"] == {"mean": 4.0}
    assert doc["needs"]["thirst"]["initial"] == S.DEFAULTS["needs"]["*"]["initial"]
    assert doc["needs"]["visit_desk"]["decay_rate"]["mean"] == 0.5
    assert result.needs["thirst"].target == "fountain"


def test_a_distribution_in_a_settings_file_replaces_the_default_whole() -> None:
    settings = S.load({"needs": {"*": {"decay_rate": {"mean": 4.0}}}, "dwell": {"mean": 50.0, "std": 5.0}, "patience": {"mean": 400.0}, "idle": {"stand": {"mean": 30.0, "clip_high": 40.0}}})
    assert settings["needs"]["*"]["decay_rate"] == {"mean": 4.0}
    assert settings["needs"]["*"]["initial"] == S.DEFAULTS["needs"]["*"]["initial"]
    assert S.need(settings, "thirst")["decay_rate"] == {"mean": 4.0}
    assert settings["dwell"] == {"mean": 50.0, "std": 5.0}
    assert settings["patience"] == {"mean": 400.0}
    assert settings["idle"]["stand"] == {"mean": 30.0, "clip_high": 40.0}


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"dwell": {"mean": 50.0, "clip_high": 30.0}}, "mean 50.0 lies outside"),
        ({"needs": {"thirst": {"decay_rate": {"mean": 0.05, "clip_low": 0.1}}}}, "needs.thirst.decay_rate: mean 0.05 lies outside"),
        ({"patience": {"std": 3.0}}, "needs a mean"),
        ({"dwell": {"mean": 5.0, "max": 9.0}}, "unknown keys"),
        ({"dwell": {"mean": 5.0, "clip_low": 9.0, "clip_high": 6.0}}, "clip_low 9.0 is above"),
        ({"idle": {"stand": {"mean": 5.0, "std": -1.0}}}, "negative"),
    ],
)
def test_a_bad_distribution_is_an_error(raw: dict[str, Any], match: str) -> None:
    with pytest.raises(S.SettingsError, match=match):
        S.load(raw)


def test_settings_refuse_unknown_keys() -> None:
    with pytest.raises(S.SettingsError, match="unknown crowd setting 'crowd'"):
        S.load({"crowd": 3})
    with pytest.raises(S.SettingsError, match="needs.thirst: unknown keys"):
        S.need(S.load({"needs": {"thirst": {"rate": 1.0}}}), "thirst")


def test_the_exterior_door_is_a_source_and_a_sink_inside_it(hospital: Result) -> None:
    regions = hospital.doc["regions"]
    assert sorted(regions) == ["main_hallway_entrance_sink", "main_hallway_entrance_source"]
    source, sink = regions["main_hallway_entrance_source"], regions["main_hallway_entrance_sink"]
    assert all(9.25 <= p["x"] <= 10.75 and 0.0 < p["y"] <= 1.8 for p in source["polygon"] + sink["polygon"])
    assert max(p["y"] for p in sink["polygon"]) < min(p["y"] for p in source["polygon"])
    agent = source["config"]["agent"]
    assert agent["sink_affinity"] == [{"sink": "main_hallway_entrance_sink", "weight": 1.0}]
    visitor = hospital.types[agent["agent_type"]]
    exit_pose = visitor["sequences"]["leave"]["steps"]["exit"]["target_pose"]
    assert 9.4 <= exit_pose["x"] <= 10.6 and 0.1 <= exit_pose["y"] <= 0.6
    assert visitor["sequences"]["seek_thirst"]["then"] == "visit"


def test_a_world_object_moved_in_the_graph_only_is_refused(queue: Site) -> None:
    graph = load(dump(queue.graph))
    fact = next(f for f in graph.facts if f.verb == "at" and f.subject.startswith("obj:waiting_chair"))
    graph.remove_fact(fact.subject, fact.verb, fact.object, fact.params)
    graph.add_fact(fact.subject, fact.verb, fact.object, {**fact.params, "x": 1.0, "y": 1.0})
    with pytest.raises(ValueError, match="is an object of world.yaml"):
        read(attrs.evolve(queue, graph=graph))


def test_builder_reads_how_people_use_objects_from_humansim() -> None:
    scene = build(HOSPITAL, HOSPITAL / "scenarios" / "showcase" / "scenario.yaml")

    def affords(obj: str) -> list[tuple[str, dict[str, Any]]]:
        return [(str(f.object), dict(f.params)) for f in scene.facts if f.subject == obj and f.verb == "affords"]

    assert affords("obj:reception_desk") == [("interaction:QUEUE_USE", {"from": "patient.yaml#default.check_in"})]
    assert ("interaction:TALK_TO", {"from": "doctor.yaml#default.consult", "meet": True}) in affords("obj:exam_chair_1")
    assert ("interaction:SERVICE", {"from": "nurse.yaml#leg_exam.offer", "meet": True, "tag": "nurse_escort", "offer": True}) in affords("obj:exam_chair_1")
    ((kind, params),) = affords("obj:patient_bed_1")
    assert kind == "interaction:SERVICE" and {k: v for k, v in params.items() if k != "from"} == {"meet": True, "tag": "nurse_escort", "seek": True, "offer": True}
    satisfies = {(str(f.object), f.params.get("from")) for f in scene.facts if f.subject == "obj:reception_desk" and f.verb == "satisfies"}
    assert satisfies == {("need:checkin", "patient.yaml#default.check_in")}


def test_service_objects_are_not_sought_and_step_needs_are() -> None:
    result = read(open_site(HOSPITAL, "showcase"), S.load({"count": 2, "synthesize_needs": False}))
    assert result.needs["checkin"].target == "reception_desk" and result.needs["checkin"].kind == "QUEUE_USE"
    assert (result.needs["treatment"].target, result.needs["treatment"].kind, result.needs["treatment"].meet) == ("exam_chair_1", "TALK_TO", True)
    assert all("patient_bed_1" not in p.objects for p in result.needs.values())


@pytest.mark.parametrize(
    ("block", "model", "expected"),
    [
        ({"type": "patient_bed", "capacity": 1}, "Hospital/Isolation_Bed", ("LIE_ON", False, "rule:prone")),
        ({"type": "chair", "capacity": 1}, "", ("SIT_ON", False, "rule:seated")),
        ({"type": "spot", "seats": [{"x": 0.0, "y": 0.0}]}, "", ("SIT_ON", False, "rule:seated")),
        ({"type": "desk", "capacity": 3, "formation": {"type": "line"}}, "Hospital/Reception_Counter", ("QUEUE_USE", False, "rule:formation line")),
        ({"type": "gathering_area", "capacity": 0, "formation": {"type": "f_formation"}}, "Hospital/Waiting_Room_Bench", ("GROUP_CONVERSATION", True, "rule:formation f_formation")),
        ({"type": "table", "capacity": 1}, "Hospital/Bedside_Table", ("USE", False, "rule:capacity")),
        ({"type": "exit"}, "Common/Object/painting_1", None),
    ],
)
def test_rules_follow_humansims_registry(block: dict[str, Any], model: str, expected: tuple[str, bool, str] | None) -> None:
    assert by_rule(block, model) == expected


def test_a_graph_without_affords_reads_by_rule(queue: Site) -> None:
    graph = load(dump(queue.graph))
    for f in [f for f in graph.facts if f.verb == "affords"]:
        graph.remove_fact(f.subject, f.verb, f.object, f.params)
    result = read(attrs.evolve(queue, graph=graph), S.load({"count": 1, "synthesize_needs": True}))
    assert result.needs["thirst"].kind == "QUEUE_USE"
    assert result.needs["visit_gathering_area"].meet


def _world_copy(tmp_path: Path) -> Path:
    root = tmp_path / "worlds" / "hospital_1"
    shutil.copytree(HOSPITAL, root)
    return root


def test_write_replaces_only_generated_scenarios(tmp_path: Path) -> None:
    root = _world_copy(tmp_path)
    site = open_site(root, "queue")
    result = read(site, S.load({"count": 2}))
    out = write(result, "flowedit_queue")
    assert sorted(p.name for p in out.iterdir()) == ["provenance.yaml", "scenario.yaml", "types"]
    assert len(list((out / "types").glob("*.yaml"))) == 3
    write(read(site, S.load({"count": 1})), "flowedit_queue")
    assert len(list((out / "types").glob("*.yaml"))) == 2
    with pytest.raises(WouldOverwrite):
        write(result, "queue")
    provenance = yaml.safe_load((out / "provenance.yaml").read_text())
    assert provenance["world"] == "hospital_1" and provenance["needs"]["thirst"]["objects"] == ["fountain"]


def test_cli_reads_a_world_into_a_scenario(tmp_path: Path) -> None:
    root = _world_copy(tmp_path)
    assert main(["read", "hospital_1", "queue", "--worlds", str(root.parent), "--count", "2", "--write"]) == 0
    text = (root / "scenarios" / "flowedit_queue" / "scenario.yaml").read_text()
    assert "name: fountain" in text and "flow_2" in text and "robots" not in text


def test_cli_reads_an_edited_world_with_the_people_of_its_base(local_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = local_root / "worlds"
    shutil.copytree(HOSPITAL, worlds / "hospital_1")
    assert main(["read", "hospital_1", "queue", "--worlds", str(worlds), "--count", "2", "--write"]) == 0
    assert main(["edit", "hospital_1", "function", "obj:fountain", "--scenario", "queue", "-p", "dwell=12", "--worlds", str(worlds), "--write"]) == 0
    saved = Path(capsys.readouterr().out.strip().splitlines()[-1].removeprefix("wrote "))
    assert saved.parent == worlds
    assert main(["read", saved.name, "--worlds", str(worlds), "--count", "2", "--write"]) == 0
    base = (worlds / "hospital_1" / "scenarios" / "flowedit_queue" / "types" / "flow__flow_1.yaml").read_text()
    edited = (saved / "scenarios" / "flowedit_queue" / "types" / "flow__flow_1.yaml").read_text()
    assert base != edited and base.replace("mean: 5.0, std: 1.5, clip_low: 4.0, clip_high: 6.0", "X").count("X") == 1
    assert yaml.safe_load(edited)["sequences"]["seek_thirst"]["steps"]["use"]["duration"] == {"mean": 12.0}
    assert yaml.safe_load(edited)["sequences"]["idle"] == yaml.safe_load(base)["sequences"]["idle"]
