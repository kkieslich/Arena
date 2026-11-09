from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from arena_simulation_setup.generative.flowedit.build import presets
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.generative.flowedit.edit import DEMAND_VERBS, EDITS, GRAPH, Site, apply, open_site, save
from arena_simulation_setup.kg import Scene, Value, dumps, load
from flowedit_docks import CHARGER, DOCKED, DOCKED_IDS, FACING_NORTH, LOCKER

if TYPE_CHECKING:
    from arena_simulation_setup.shared.entities import Obstacle

pytest.importorskip("numpy")
pytestmark = pytest.mark.usefixtures("local_root")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"

EDITS_OF_EVERY_TYPE: list[tuple[str, str, dict[str, Any]]] = [
    ("move", "obj:fountain", {"x": 15.0, "y": 11.5}),
    ("insert", "obj:fountain_2", {"model": "Hospital/SM_WaterDispenser_01a", "x": 20.0, "y": 5.0, "type": "fountain", "capacity": 1, "satisfies": {"thirst": 80.0}, "dwell": {"mean": 8.0, "std": 2.0}}),
    ("remove", "obj:reception_trashcan_1", {}),
    ("function", "obj:fountain", {"capacity": 2, "dwell": 12.0}),
    ("access", "door:or_door", {"locked": True, "authorized": ["doctor_1"], "unlock_on": "alarm"}),
    ("layout", "door:reception_door", {"offset": [0.0, -0.5]}),
]


@pytest.fixture(scope="module")
def queue() -> Site:
    return open_site(HOSPITAL, "queue")


@pytest.fixture(autouse=True)
def docked_assets(local_root: Path) -> dict[str, Path]:
    """The docked models resolve in every test, as a docked object's model must (`resolve_footprints`)."""
    from flowedit_docks import write_docked_assets

    return write_docked_assets(local_root)


def _facts(scene: Scene, subject: str, verb: str) -> list[tuple[Value, dict[str, Value]]]:
    return [(f.object, dict(f.params)) for f in scene.facts if f.subject == subject and f.verb == verb]


def _rooms(scene: Scene, ref: str) -> list[str]:
    return [f.subject for f in scene.facts if f.verb == "contains" and f.object == ref and f.subject.startswith("room:")]


@pytest.mark.parametrize(("edit_type", "target", "params"), EDITS_OF_EVERY_TYPE, ids=[e[0] for e in EDITS_OF_EVERY_TYPE])
def test_every_type_saves_a_world_that_loads(queue: Site, local_root: Path, edit_type: str, target: str, params: dict[str, Any]) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    path = save(apply(queue, edit_type, target, params))
    assert path.parent == local_root / "worlds"
    WorldIdentifier(path.name).resolve_sync().load()
    record = yaml.safe_load((path / EDITS).read_text())
    assert record["edits"] == [{"id": path.name.split("__")[-1], "parent": "hospital_1", "type": edit_type, "target": target, "params": params}]
    assert load(path / GRAPH).validate() == []


def test_apply_leaves_its_site_alone_and_names_the_edit_by_its_record(queue: Site) -> None:
    before = dumps(queue.graph)
    first = apply(queue, *EDITS_OF_EVERY_TYPE[0])
    assert dumps(queue.graph) == before
    assert not queue.edits
    assert first.name == apply(queue, *EDITS_OF_EVERY_TYPE[0]).name != apply(queue, "move", "obj:fountain", {"x": 15.0}).name


def test_int_and_float_params_name_the_same_edit(queue: Site) -> None:
    assert apply(queue, "move", "obj:fountain", {"x": 15}).name == apply(queue, "move", "obj:fountain", {"x": 15.0}).name


def test_world_names_say_what_the_edit_did_and_end_in_its_id(queue: Site) -> None:
    moved = apply(queue, "move", "obj:fountain", {"x": 15.0, "y": 25.0})
    assert moved.name == f"hospital_1__move_fountain__{moved.edits[-1].id}"
    assert moved.key == f"hospital_1__{moved.edits[-1].id}"
    left = apply(queue, "move", "obj:fountain", {"x": 15.0, "y": 25.0, "leave_demand": True})
    assert left.name == f"hospital_1__move_fountain_leave_demand__{left.edits[-1].id}"
    assert apply(queue, "layout", "door:or_door", {"op": "remove"}).name.startswith("hospital_1__layout_remove_or_door__")


def test_the_readable_name_stays_out_of_the_ids(queue: Site) -> None:
    first = apply(queue, "move", "obj:fountain", {"x": 15.0})
    second = apply(first, "layout", "door:or_door", {"op": "remove"})
    assert second.edits[-1].parent == first.key
    assert second.name.split("__")[0] == "hospital_1" and second.name.count("__") == 2


def test_moved_scenario_object_changes_room_only_in_the_graph(queue: Site) -> None:
    moved = apply(queue, "move", "obj:fountain", {"x": 15.0, "y": 11.5})
    assert _rooms(queue.graph, "obj:fountain") == ["room:central_hallway"]
    assert _rooms(moved.graph, "obj:fountain") == ["room:sub_hallway"]
    assert _facts(moved.graph, "obj:fountain", "at") == [("floor:0", {"x": 15.0, "y": 11.5, "yaw": 1.57})]
    assert all(o.name != "fountain" for o in moved.world.levels["0"].objects)


def test_moved_world_object_moves_in_world_yaml(queue: Site) -> None:
    path = save(apply(queue, "move", "obj:reception_counter_1", {"x": 3.0}))
    objects = yaml.safe_load((path / "world.yaml").read_text())["levels"][0]["objects"]
    assert next(o for o in objects if o["name"] == "reception_counter_1")["pose"][:2] == [3.0, 2.8]
    assert (path / "kg.yaml").read_text() == (HOSPITAL / "kg.yaml").read_text()


def test_function_changes_demand_and_dwell(queue: Site) -> None:
    edited = apply(queue, "function", "obj:fountain", {"capacity": 2, "satisfies": {"thirst": None, "hunger": 30}, "dwell": {"mean": 9, "std": 2}})
    edit = edited.edits[-1]
    assert _facts(edited.graph, "obj:fountain", "serves") == [("service:fountain", {"capacity": 2})]
    assert _facts(edited.graph, "obj:fountain", "satisfies") == [("need:hunger", {"amount": 30.0})]
    assert _facts(edited.graph, "obj:fountain", "dwells") == [(9.0, {"std": 2.0, "from": f"edit:{edit.id}"})]


def test_access_sets_a_gate_and_a_schedule(queue: Site) -> None:
    gated = apply(queue, "access", "door:or_door", {"locked": True, "authorized": ["doctor_1"], "unlock_on": "alarm"})
    assert _facts(gated.graph, "door:or_door", "gated") == [(True, {})]
    assert _facts(gated.graph, "door:or_door", "authorizes") == [("doctor_1", {})]
    assert _facts(gated.graph, "door:or_door", "unlocks_on") == [("regime:alarm", {})]
    opened = apply(gated, "access", "door:or_door", {"remove": True})
    assert _facts(opened.graph, "door:or_door", "gated") == []
    timed = apply(queue, "access", "schedule:fire_alarm", {"windows": [{"start": 60, "end": 90, "value": "on"}]})
    assert _facts(timed.graph, "schedule:fire_alarm", "window") == [(60.0, {"end": 90.0, "value": "on"})]
    assert _facts(timed.graph, "schedule:fire_alarm", "asserts") == [("regime:alarm", {})]


def test_moved_door_closes_its_old_gap(queue: Site) -> None:
    moved = apply(queue, "layout", "door:reception_door", {"offset": [0.0, -0.5]})
    on_x8 = sorted((round(min(w.start.y, w.end.y), 2), round(max(w.start.y, w.end.y), 2)) for w in moved.world.levels["0"].all_walls if w.start.x == w.end.x == 8.0 and max(w.start.y, w.end.y) <= 5.0)
    assert on_x8 == [(0.0, 1.25), (2.75, 3.25), (3.25, 5.0)]


def test_removed_door_drops_its_connection(queue: Site) -> None:
    assert [f.params["door"] for f in queue.graph.facts if f.verb == "connects" and f.params.get("door") == "door:or_door"]
    removed = apply(queue, "layout", "door:or_door", {"op": "remove"})
    assert "door:or_door" not in removed.graph.entities
    assert not [f for f in removed.graph.facts if f.verb == "connects" and f.params.get("door") == "door:or_door"]


def test_door_off_any_wall_is_refused(queue: Site) -> None:
    with pytest.raises(ValueError, match="lies on no wall"):
        apply(queue, "layout", "door:new_door", {"op": "add", "span": [[3.0, 3.0], [4.0, 3.0]]})


def test_wall_paths_are_added_moved_and_removed(queue: Site) -> None:
    walls = len(queue.world.levels["0"].walls)
    added = apply(queue, "layout", "wall:0", {"op": "add", "path": [[18.5, 2.0], [18.5, 4.0]]})
    assert len(added.world.levels["0"].walls) == walls + 1
    moved = apply(added, "layout", f"wall:0/{walls}", {"offset": [0.5, 0.0]})
    assert [p.x for p in moved.world.levels["0"].walls[-1].path] == [19.0, 19.0]
    assert len(apply(moved, "layout", f"wall:0/{walls}", {"op": "remove"}).world.levels["0"].walls) == walls


def test_edits_chain_from_a_saved_world(queue: Site) -> None:
    first = apply(queue, *EDITS_OF_EVERY_TYPE[0])
    reopened = open_site(save(first))
    assert reopened.name == first.name and reopened.scenario == "queue"
    with pytest.raises(ValueError, match="seed"):
        open_site(save(first), seed=1)
    second = apply(reopened, "layout", "door:or_door", {"op": "remove"})
    path = save(second)
    record = yaml.safe_load((path / EDITS).read_text())
    assert [e["parent"] for e in record["edits"]] == ["hospital_1", first.key]
    assert path.name == f"hospital_1__2_edits_layout_remove_or_door__{record['edits'][-1]['id']}"
    assert _rooms(load(path / GRAPH), "obj:fountain") == ["room:sub_hallway"]


def test_cli_prints_the_edit_and_writes_with_write(local_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["edit", "hospital_1", "function", "obj:fountain", "--scenario", "queue", "--worlds", str(WORLDS), "-p", "capacity=2", "-p", "satisfies.thirst=60"]
    assert main(argv) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["edits"][0]["params"] == {"capacity": 2, "satisfies": {"thirst": 60}}
    assert not (local_root / "worlds").exists()
    assert main([*argv, "--write"]) == 0
    assert (local_root / "worlds" / printed["world"] / EDITS).is_file()


def test_function_clear_leaves_an_inert_object(queue: Site) -> None:
    inserted = apply(queue, *EDITS_OF_EVERY_TYPE[1])
    cleared = apply(inserted, "function", "obj:fountain_2", {"clear": True})
    assert not [f for f in cleared.graph.facts if f.subject == "obj:fountain_2" and f.verb in DEMAND_VERBS]
    objects = yaml.safe_load((save(cleared) / "world.yaml").read_text())["levels"][0]["objects"]
    assert not next(o for o in objects if o["name"] == "fountain_2").get("humansim")
    scenario = apply(queue, "function", "obj:fountain", {"clear": True, "type": "fountain", "capacity": 2})
    assert _facts(scenario.graph, "obj:fountain", "serves") == [("service:fountain", {"capacity": 2})]
    assert not _facts(scenario.graph, "obj:fountain", "has_formation")


def _demand(scene: Scene, ref: str) -> set[tuple[str, Value, tuple[tuple[str, Value], ...]]]:
    return {(f.verb, f.object, tuple(sorted(f.params.items()))) for f in scene.facts if f.subject == ref and f.verb in DEMAND_VERBS}


def test_move_leaving_demand_leaves_it_on_a_demand_only_node_at_the_old_spot(queue: Site) -> None:
    moved = apply(queue, "move", "obj:fountain", {"x": 15.0, "y": 11.5, "leave_demand": True})
    assert not _demand(moved.graph, "obj:fountain")
    assert _demand(moved.graph, "obj:fountain_demand") == _demand(queue.graph, "obj:fountain") != set()
    assert _facts(moved.graph, "obj:fountain_demand", "at") == _facts(queue.graph, "obj:fountain", "at")
    assert "model" not in moved.graph.entities["obj:fountain_demand"].props
    assert _rooms(moved.graph, "obj:fountain_demand") == ["room:central_hallway"]
    assert _rooms(moved.graph, "obj:fountain") == ["room:sub_hallway"]
    assert moved.name != apply(queue, "move", "obj:fountain", {"x": 15.0, "y": 11.5}).name
    reopened = open_site(save(moved))
    assert reopened.graph.validate() == []
    assert _demand(reopened.graph, "obj:fountain_demand") == _demand(moved.graph, "obj:fountain_demand")


def test_move_leaving_demand_empties_a_world_objects_humansim_block(queue: Site) -> None:
    inserted = apply(queue, *EDITS_OF_EVERY_TYPE[1])
    moved = apply(inserted, "move", "obj:fountain_2", {"x": 21.0, "leave_demand": True})
    objects = yaml.safe_load((save(moved) / "world.yaml").read_text())["levels"][0]["objects"]
    assert not next(o for o in objects if o["name"] == "fountain_2").get("humansim")
    assert all(o["name"] != "fountain_2_demand" for o in objects)  # the demand-only node lives in the graph only
    assert _demand(moved.graph, "obj:fountain_2_demand") == _demand(inserted.graph, "obj:fountain_2")


def test_a_formation_anchored_at_the_moved_object_follows_its_demand(queue: Site) -> None:
    moved = apply(queue, "move", "obj:gathering_area", {"x": 16.0, "leave_demand": True})
    ((_, params),) = _facts(moved.graph, "obj:gathering_area_demand", "has_formation")
    assert params["anchor_ref"] == "gathering_area_demand"


def test_the_demand_only_node_faces_where_its_model_did_and_is_used_as_it_was(queue: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_simulation_setup.generative.flowedit import generate

    key = (queue.root, str(queue.graph.entities["obj:gathering_area"].props["model"]))
    monkeypatch.setitem(generate._FOOTPRINTS, key, None)  # resolved: the tests have no object assets
    monkeypatch.setitem(generate._FACES, key, "-y")
    moved = apply(queue, "move", "obj:gathering_area", {"x": 16.0, "leave_demand": True})
    ((_, old),) = _facts(queue.graph, "obj:gathering_area", "at")
    ((_, node),) = _facts(moved.graph, "obj:gathering_area_demand", "at")
    assert (node["x"], node["y"]) == (old["x"], old["y"])
    queued = old["yaw"] - math.pi / 2 - math.pi  # the bench's front is -y; the node, without a face, is queued behind its yaw
    assert node["yaw"] == pytest.approx(math.atan2(math.sin(queued), math.cos(queued)), abs=1e-3)
    assert _facts(moved.graph, "obj:gathering_area_demand", "affords") == _facts(queue.graph, "obj:gathering_area", "affords") != []
    assert not _facts(moved.graph, "obj:gathering_area", "affords")


def test_a_docked_object_whose_model_does_not_resolve_cannot_leave_its_demand(queue: Site, docked_assets: dict[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    from arena_simulation_setup.generative.flowedit import generate

    for cache in ("_FOOTPRINTS", "_FACES", "_UNRESOLVED"):
        monkeypatch.setattr(generate, cache, {})  # nothing resolved yet
    shutil.rmtree(docked_assets[LOCKER.model])
    inserted = apply(queue, "insert", f"obj:{LOCKER.name}", {**_docked(LOCKER.model, LOCKER.back, LOCKER.dock), "type": "parcel_locker", "capacity": 1})
    with pytest.raises(ValueError, match="model 'parcel_locker' of a docked object does not resolve"):
        apply(inserted, "move", f"obj:{LOCKER.name}", {"x": 22.0, "leave_demand": True})


def test_move_leaving_demand_needs_demand_and_a_free_name(queue: Site) -> None:
    with pytest.raises(ValueError, match="no demand"):
        apply(queue, "move", "obj:reception_trashcan_1", {"x": 3.0, "leave_demand": True})
    taken = apply(queue, "insert", "obj:fountain_demand", {"model": "Hospital/SM_WaterDispenser_01a", "x": 20.0, "y": 5.0})
    with pytest.raises(ValueError, match="already taken"):
        apply(taken, "move", "obj:fountain", {"x": 15.0, "y": 11.5, "leave_demand": True})


def _world_object(site: Site, name: str) -> Obstacle:
    return next(o for level in site.world.levels.values() for o in level.objects if o.name == name)


def test_move_keeps_an_objects_height_and_insert_takes_one(queue: Site) -> None:
    tray = _world_object(queue, "or_surgical_tray_1").pose
    assert tray.position.z > 0.5  # it stands on a table
    moved = apply(queue, "move", "obj:or_surgical_tray_1", {"x": tray.position.x + 0.1, "yaw": 0.5})
    after = _world_object(open_site(save(moved)), "or_surgical_tray_1").pose
    assert after.position.z == pytest.approx(tray.position.z) and after.orientation.to_yaw() == pytest.approx(0.5)
    model = str(queue.graph.entities["obj:or_surgical_tray_1"].props["model"])
    inserted = apply(queue, "insert", "obj:tray_2", {"model": model, "x": tray.position.x, "y": tray.position.y + 0.3, "z": 0.9})
    assert _world_object(inserted, "tray_2").pose.position.z == 0.9


def test_an_inserted_objects_dwell_names_the_insert_edit(queue: Site) -> None:
    inserted = apply(queue, *EDITS_OF_EVERY_TYPE[1])
    ((_, params),) = _facts(inserted.graph, "obj:fountain_2", "dwells")
    assert params["from"] == f"edit:{inserted.edits[-1].id}"


def test_a_false_flag_names_the_same_edit_as_leaving_it_out(queue: Site) -> None:
    move = {"x": 15.0, "y": 11.5}
    assert apply(queue, "move", "obj:fountain", {**move, "leave_demand": False}).name == apply(queue, "move", "obj:fountain", move).name
    assert apply(queue, "function", "obj:fountain", {"capacity": 2, "clear": False}).name == apply(queue, "function", "obj:fountain", {"capacity": 2}).name
    assert apply(queue, "access", "door:or_door", {"locked": True, "remove": False}).name == apply(queue, "access", "door:or_door", {"locked": True}).name


def test_schedule_access_keeps_remove_out_of_its_preset(queue: Site) -> None:
    timed = apply(queue, "access", "schedule:fire_alarm", {"remove": False, "windows": [{"start": 60, "end": 90}]})
    (params,) = presets(next(s for s in timed.world.schedules if s.name == "fire_alarm").semantics, "schedule")
    # the runtime reads a window's `value`: one written without it opens as "on", the edit record keeps what was asked
    assert "remove" not in params and params["windows"] == [{"start": 60, "end": 90, "value": "on"}]
    assert timed.edits[-1].params["windows"] == [{"start": 60, "end": 90}]
    with pytest.raises(ValueError, match="windows are a list"):
        apply(queue, "access", "schedule:fire_alarm", {"windows": [{"start": 60}]})


def _docked(model: str, back: float, dock: list[list[float]]) -> dict[str, Any]:
    return {"model": model, "x": 20.0, "y": back, "yaw": FACING_NORTH, "dock": dock}


def _posed(x: float, y: float, yaw: float, dock: list[list[float]]) -> list[tuple[float, float]]:
    """`dock` turned by `yaw` and moved to `x`, `y`, to the mm."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [(round(x + c * px - s * py, 3), round(y + s * px + c * py, 3)) for px, py in dock]


def _dock_in_level(site: Site, name: str) -> list[tuple[float, float]] | None:
    corners = site.world.levels["0"].lookup_zone_polygon(name)
    return None if corners is None else [(round(c.x, 3), round(c.y, 3)) for c in corners]


def _saved_objects(path: Path) -> dict[str, dict[str, Any]]:
    return {o["name"]: o for o in yaml.safe_load((path / "world.yaml").read_text())["levels"][0]["objects"]}


@pytest.mark.parametrize(("name", "model", "back", "dock"), DOCKED, ids=DOCKED_IDS)
def test_insert_writes_a_dock_that_makes_the_object_a_goto_target(queue: Site, name: str, model: str, back: float, dock: list[list[float]]) -> None:
    path = save(apply(queue, "insert", f"obj:{name}", _docked(model, back, dock)))
    assert _saved_objects(path)[name]["dock"] == dock
    reopened = open_site(path)
    assert name in reopened.world.levels["0"].zone_ref_names()
    assert _dock_in_level(reopened, name) == _posed(20.0, back, FACING_NORTH, dock)


def test_insert_refuses_a_dock_of_fewer_than_three_points(queue: Site) -> None:
    _, model, back, dock = CHARGER
    with pytest.raises(ValueError, match="at least 3 points"):
        apply(queue, "insert", "obj:charger_1", _docked(model, back, dock[:2]))


@pytest.mark.parametrize(("name", "model", "back", "dock"), DOCKED, ids=DOCKED_IDS)
def test_a_moved_object_takes_its_dock_along_in_its_own_frame(queue: Site, name: str, model: str, back: float, dock: list[list[float]]) -> None:
    inserted = apply(queue, "insert", f"obj:{name}", _docked(model, back, dock))
    # to the east wall, the front turned into the room again
    path = save(apply(inserted, "move", f"obj:{name}", {"x": 25.0 - back, "y": 5.0, "yaw": 3.1416}))
    assert _saved_objects(path)[name]["dock"] == dock
    assert _dock_in_level(open_site(path), name) == _posed(25.0 - back, 5.0, 3.1416, dock)


def test_a_locker_moved_without_its_demand_keeps_its_dock(queue: Site) -> None:
    name, model, back, dock = LOCKER
    inserted = apply(queue, "insert", f"obj:{name}", {**_docked(model, back, dock), "type": "parcel_locker", "capacity": 1, "dwell": 30.0})
    moved = apply(inserted, "move", f"obj:{name}", {"x": 22.0, "leave_demand": True})
    assert _dock_in_level(moved, name) == _posed(22.0, back, FACING_NORTH, dock)
    assert _dock_in_level(moved, f"{name}_demand") is None  # robots dock at the model, people queue at the old spot


@pytest.mark.parametrize(("name", "model", "back", "dock"), DOCKED, ids=DOCKED_IDS)
def test_a_removed_object_takes_its_dock_along(queue: Site, name: str, model: str, back: float, dock: list[list[float]]) -> None:
    inserted = apply(queue, "insert", f"obj:{name}", _docked(model, back, dock))
    path = save(apply(inserted, "remove", f"obj:{name}"))
    assert name not in _saved_objects(path)
    assert name not in open_site(path).world.levels["0"].zone_ref_names()
