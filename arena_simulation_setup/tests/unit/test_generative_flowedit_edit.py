from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.generative.flowedit.edit import EDITS, GRAPH, Site, apply, open_site, save
from arena_simulation_setup.kg import Scene, Value, dumps, load

pytest.importorskip("numpy")

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


@pytest.fixture(autouse=True)
def local_worlds(tmp_path: Path) -> Iterator[Path]:
    previous = os.environ.get("ARENA_ASSETS_DIR_LOCAL")
    os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(tmp_path)
    yield tmp_path / "worlds"
    if previous is None:
        del os.environ["ARENA_ASSETS_DIR_LOCAL"]
    else:
        os.environ["ARENA_ASSETS_DIR_LOCAL"] = previous


@pytest.fixture(scope="module")
def queue() -> Site:
    return open_site(HOSPITAL, "queue")


def _facts(scene: Scene, subject: str, verb: str) -> list[tuple[Value, dict[str, Value]]]:
    return [(f.object, dict(f.params)) for f in scene.facts if f.subject == subject and f.verb == verb]


def _rooms(scene: Scene, ref: str) -> list[str]:
    return [f.subject for f in scene.facts if f.verb == "contains" and f.object == ref and f.subject.startswith("room:")]


@pytest.mark.parametrize(("edit_type", "target", "params"), EDITS_OF_EVERY_TYPE, ids=[e[0] for e in EDITS_OF_EVERY_TYPE])
def test_every_type_saves_a_world_that_loads(queue: Site, local_worlds: Path, edit_type: str, target: str, params: dict[str, Any]) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    path = save(apply(queue, edit_type, target, params))
    assert path.parent == local_worlds
    WorldIdentifier(path.name).resolve_sync().load()
    record = yaml.safe_load((path / EDITS).read_text())
    assert record["edits"] == [{"id": path.name.split("__")[1], "parent": "hospital_1", "type": edit_type, "target": target, "params": params}]
    assert load(path / GRAPH).validate() == []


def test_apply_leaves_its_site_alone_and_names_the_edit_by_its_record(queue: Site) -> None:
    before = dumps(queue.graph)
    first = apply(queue, *EDITS_OF_EVERY_TYPE[0])
    assert dumps(queue.graph) == before
    assert not queue.edits
    assert first.name == apply(queue, *EDITS_OF_EVERY_TYPE[0]).name != apply(queue, "move", "obj:fountain", {"x": 15.0}).name


def test_int_and_float_params_name_the_same_edit(queue: Site) -> None:
    assert apply(queue, "move", "obj:fountain", {"x": 15}).name == apply(queue, "move", "obj:fountain", {"x": 15.0}).name


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
    assert [e["parent"] for e in record["edits"]] == ["hospital_1", first.name]
    assert path.name == f"hospital_1__{record['edits'][-1]['id']}"
    assert _rooms(load(path / GRAPH), "obj:fountain") == ["room:sub_hallway"]


def test_cli_prints_the_edit_and_writes_with_write(local_worlds: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["edit", "hospital_1", "function", "obj:fountain", "--scenario", "queue", "--worlds", str(WORLDS), "-p", "capacity=2", "-p", "satisfies.thirst=60"]
    assert main(argv) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["edits"][0]["params"] == {"capacity": 2, "satisfies": {"thirst": 60}}
    assert not local_worlds.exists()
    assert main([*argv, "--write"]) == 0
    assert (local_worlds / printed["world"] / EDITS).is_file()
