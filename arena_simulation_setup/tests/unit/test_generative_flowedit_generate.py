from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from arena_simulation_setup.generative.flowedit.edit import DEMAND_VERBS, EditType, Site, dump_edits, open_site, save, world_object
from arena_simulation_setup.generative.flowedit.generate import CAPACITY, DEFAULT_DWELL_S, DWELL_S, FIELDS, FLOOR_HEIGHT_M, Generated, dump_manifest, floor_objects, generate
from flowedit_docks import CHARGE_POWER_W, CHARGER, FACING_NORTH, LOCKER, Docked

if TYPE_CHECKING:
    from arena_simulation_setup.generative.flowedit.generate import WallPiece
    from arena_simulation_setup.tree.World.model import Zone
    from arena_simulation_setup.utils.geometry import Position

pytest.importorskip("numpy")
pytestmark = pytest.mark.usefixtures("local_root")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
STEP_ONE = (EditType.REMOVE, EditType.FUNCTION, EditType.ACCESS)


@pytest.fixture(scope="module")
def queue() -> Site:
    return open_site(HOSPITAL, "queue")


def _edits(chains: list[Generated]) -> list[dict[str, Any]]:
    return [dump_edits(chain.site) for chain in chains]


def test_same_seed_gives_the_same_edits(queue: Site) -> None:
    assert _edits(generate(queue, seed=3, types=STEP_ONE)) == _edits(generate(queue, seed=3, types=STEP_ONE))


def test_other_seeds_give_other_edits(queue: Site) -> None:
    runs = {str(_edits(generate(queue, seed=seed, types=STEP_ONE))) for seed in range(5)}
    assert len(runs) > 1


def test_a_shorter_chain_is_a_prefix_of_a_longer_one(queue: Site) -> None:
    short = generate(queue, seed=1, steps=1, types=STEP_ONE)
    long = generate(queue, seed=1, steps=3, types=STEP_ONE)
    assert [c.site.edits[:1] for c in long] == [c.site.edits for c in short]
    assert all(len(c.site.edits) == 3 for c in long)
    for chain in long:
        assert [e.parent for e in chain.site.edits] == [queue.name, *(f"{queue.name}__{e.id}" for e in chain.site.edits[:-1])]


def test_every_chain_starts_with_its_type(queue: Site) -> None:
    chains = generate(queue, seed=0, per_type=2, types=STEP_ONE)
    assert [c.records[0]["type"] for c in chains] == [t.value for t in STEP_ONE for _ in range(2)]


def _needy(site: Site) -> set[str]:
    return {f.subject for f in site.graph.facts if f.verb == "satisfies"}


def test_function_changes_a_plausible_capacity_or_dwell_of_an_object_with_a_need(queue: Site) -> None:
    assert FIELDS == ("capacity", "dwell")
    fields = set()
    for seed in range(12):
        (chain,) = generate(queue, seed=seed, types=[EditType.FUNCTION])
        edit = chain.site.edits[-1]
        assert edit.target in _needy(queue) and chain.records[0]["function_carrying"] is True
        (field,) = edit.params
        fields.add(field)
        kind = next(str(f.object).partition(":")[2] for f in queue.graph.facts if f.subject == edit.target and f.verb == "serves")
        if field == "capacity":
            old = next(f.params.get("capacity", 1) for f in queue.graph.facts if f.subject == edit.target and f.verb == "serves")
            low, high = CAPACITY.get(kind, (1, 4))
            assert low <= edit.params["capacity"] <= high and edit.params["capacity"] != old
        else:
            old = next(f for f in queue.graph.facts if f.subject == edit.target and f.verb == "dwells")
            dwell, scale = edit.params["dwell"], edit.params["dwell"]["mean"] / float(old.object)
            low, high = DWELL_S.get(kind, DEFAULT_DWELL_S)
            assert low <= dwell["mean"] <= high and dwell["mean"] != float(old.object)
            assert {k: dwell[k] for k in ("std", "clip_low", "clip_high")} == {k: round(float(old.params[k]) * scale, 2) for k in ("std", "clip_low", "clip_high")}
    assert fields == {"capacity", "dwell"}


def test_dwell_changes_only_on_the_sole_object_for_its_need(queue: Site) -> None:
    (insert,) = generate(queue, seed=0, types=[EditType.INSERT])
    two = insert.site
    assert len(_needy(two)) == 2  # the inserted copy satisfies the same need
    for chain in generate(two, seed=0, per_type=6, types=[EditType.FUNCTION]):
        assert set(chain.site.edits[-1].params) == {"capacity"}


def test_a_site_without_demand_gives_no_function_edit_and_a_note() -> None:
    notes: list[str] = []
    assert generate(open_site(HOSPITAL), types=[EditType.FUNCTION], notes=notes) == []
    assert notes == ["hospital_1: no function edit, no object with a need"]


def test_no_two_chains_reach_the_same_world_at_a_step(queue: Site) -> None:
    for steps in (1, 2):
        chains = generate(queue, seed=0, per_type=3, steps=steps)
        for step in range(steps):
            worlds = [f"{queue.name}__{c.site.edits[step].id}" for c in chains if len(c.site.edits) > step]
            assert len(worlds) == len(set(worlds))


def test_remove_prefers_objects_with_demand(queue: Site) -> None:
    demand = _needy(queue)
    for seed in range(5):
        (chain,) = generate(queue, seed=seed, types=[EditType.REMOVE])
        assert chain.site.edits[-1].target in demand
        assert chain.records[0]["function_carrying"] is True


def test_every_sampled_edit_saves_a_world_that_loads(queue: Site, local_root: Path) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    for chain in generate(queue, seed=2, steps=2, types=STEP_ONE):
        path = save(chain.site)
        assert path.parent == local_root / "worlds"
        WorldIdentifier(path.name).resolve_sync().load()


def test_manifest_names_each_chain(queue: Site) -> None:
    chains = generate(queue, seed=0, types=STEP_ONE)
    manifest = dump_manifest(queue, chains, seed=0, per_type=1, steps=1)
    assert manifest["base"] == "hospital_1" and manifest["scenario"] == "queue" and manifest["fields"] == ["capacity", "dwell"]
    assert [c["world"] for c in manifest["chains"]] == [c.site.name for c in chains]
    assert [c["edits"][0]["id"] for c in manifest["chains"]] == [c.site.edits[0].id for c in chains]


SHIPPED = sorted(p.name for p in WORLDS.iterdir() if not p.name.startswith(".") and (p / "world.yaml").is_file())


def _zone_of(site: Site, level_id: str, x: float, y: float) -> Zone:
    from arena_simulation_setup.generative.flowedit.build import zones_at
    from arena_simulation_setup.generative.flowedit.generate import allowed_zones
    from arena_simulation_setup.utils.geometry import Position

    names = set(zones_at(site.world.levels[level_id], Position(x, y)))
    return next(z for z in allowed_zones(site.world.levels[level_id]) if z.name in names)


def test_all_types_are_deterministic_and_named_by_scope(queue: Site) -> None:
    first = generate(queue, seed=4)
    assert _edits(first) == _edits(generate(queue, seed=4))
    assert [c.records[0]["type"] for c in first] == [t.value for t in EditType]
    for chain in first:
        record = chain.records[0]
        assert ("scope" in record) == (record["type"] in ("move", "insert"))


def test_moves_reach_their_own_room_and_others_on_valid_spots(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.build import zones_at
    from arena_simulation_setup.generative.flowedit.generate import Spots
    from arena_simulation_setup.utils.geometry import Position

    scopes = set()
    for seed in range(12):
        (chain,) = generate(queue, seed=seed, types=[EditType.MOVE])
        edit, scope = chain.site.edits[-1], chain.records[0]["scope"]
        scopes.add(scope)
        level_id, before = next((str(f.object).partition(":")[2], f.params) for f in queue.graph.facts if f.subject == edit.target and f.verb == "at")
        old_rooms = set(zones_at(queue.world.levels[level_id], Position(float(before["x"]), float(before["y"]))))
        zone = _zone_of(queue, level_id, edit.params["x"], edit.params["y"])
        assert (zone.name in old_rooms) == (scope == "local")
        model = str(queue.graph.entities[edit.target].props["model"])
        assert Spots(queue, level_id).valid(model, zone, edit.params["x"], edit.params["y"], edit.params["yaw"])
    assert scopes == {"local", "room"}


def test_a_moved_object_never_overlaps_its_old_footprint(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import footprint

    local = 0
    for chain in generate(queue, seed=0, per_type=40, types=[EditType.MOVE]):
        edit = chain.site.edits[0]
        before = next(f.params for f in queue.graph.facts if f.subject == edit.target and f.verb == "at")
        model = str(queue.graph.entities[edit.target].props["model"])
        old = footprint(queue.root, model, float(before["x"]), float(before["y"]), float(before["yaw"]))
        new = footprint(queue.root, model, edit.params["x"], edit.params["y"], edit.params["yaw"])
        assert not old.intersects(new)  # the inert_object copy stands there next to the original
        local += chain.records[0]["scope"] == "local"
    assert local > 0


def test_insert_copies_an_objects_demand_onto_a_new_name_at_a_valid_spot(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Spots

    for seed in range(4):
        (chain,) = generate(queue, seed=seed, types=[EditType.INSERT])
        edit = chain.site.edits[-1]
        assert edit.target.startswith("obj:gen_") and edit.target not in queue.graph.entities
        assert "type" in edit.params and "satisfies" in edit.params and chain.records[0]["function_carrying"] is True
        level = str(edit.params["level"])
        zone = _zone_of(queue, level, edit.params["x"], edit.params["y"])
        assert Spots(queue, level).valid(edit.params["model"], zone, edit.params["x"], edit.params["y"], edit.params["yaw"])


def test_moved_and_inserted_objects_stand_against_a_wall_facing_the_room(queue: Site) -> None:
    import math

    import shapely
    import shapely.ops
    from arena_simulation_setup.generative.flowedit.generate import ROUNDING_M, WALL_GAP_M, WALL_SLACK_M, WALL_SNAP_M, footprint, front_angle

    for kind in (EditType.MOVE, EditType.INSERT):
        for chain in generate(queue, seed=0, per_type=8, types=[kind]):
            edit = chain.site.edits[-1]
            level = str(edit.params.get("level", "0"))
            model = str(edit.params.get("model") or queue.graph.entities[edit.target].props["model"])
            x, y, yaw = edit.params["x"], edit.params["y"], edit.params["yaw"]
            walls = shapely.union_all([shapely.LineString([(w.start.x, w.start.y), (w.end.x, w.end.y)]) for w in queue.world.levels[level].all_walls])
            shape = footprint(queue.root, model, x, y, yaw)
            # as close to a wall as the wall gap allows, the zone edge in front of the wall line taking a little more
            assert shape.distance(walls) <= WALL_GAP_M + WALL_SLACK_M + 0.35
            wall = shapely.ops.nearest_points(walls, shapely.Point(x, y))[0]
            front = yaw + front_angle(queue.root, model)
            assert math.cos(front) * (x - wall.x) + math.sin(front) * (y - wall.y) > 0  # the front looks away from the wall
            # and a wall stands right behind the middle of the back
            fx, fy = math.cos(front), math.sin(front)
            back = max(-((vx - x) * fx + (vy - y) * fy) for vx, vy in shape.exterior.coords)
            reach = back + WALL_GAP_M + WALL_SLACK_M + WALL_SNAP_M + ROUNDING_M
            assert shapely.LineString([(x, y), (x - fx * reach, y - fy * reach)]).intersects(walls)


def _spans(pieces: list[WallPiece], x: float) -> list[tuple[float, float]]:
    """The y ranges of the walled pieces on the line x = `x`."""
    ys = [[c[1] for c in p.line.coords] for p in pieces if all(abs(c[0] - x) < 1e-9 for c in p.line.coords)]
    return sorted((round(min(y), 6), round(max(y), 6)) for y in ys)


ROOM = ((0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 4.0))


def test_a_door_gap_at_an_edges_middle_only_cuts_its_wall() -> None:
    import shapely
    from arena_simulation_setup.generative.flowedit.generate import wall_pieces

    walls = [((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (10.0, 1.5)), ((10.0, 2.5), (10.0, 4.0)), ((10.0, 4.0), (0.0, 4.0)), ((0.0, 4.0), (0.0, 0.0))]
    pieces = wall_pieces(shapely.Polygon(ROOM), walls)
    assert _spans(pieces, 10.0) == [(0.0, 1.5), (2.5, 4.0)]
    assert {p.normal for p in pieces if all(c[0] == 10.0 for c in p.line.coords)} == {(-1.0, 0.0)}  # into the room
    assert len(pieces) == 5


def test_a_wall_stub_or_a_wall_meeting_an_edge_puts_no_wall_behind_it() -> None:
    import shapely
    from arena_simulation_setup.generative.flowedit.generate import wall_pieces

    # a 0.3 m stub in the middle of the west edge, a wall meeting it across, another meeting the north edge
    walls = [((0.0, 0.0), (10.0, 0.0)), ((0.0, 1.85), (0.0, 2.15)), ((0.0, 2.0), (-1.0, 2.0)), ((5.0, 4.0), (5.0, 6.0))]
    pieces = wall_pieces(shapely.Polygon(ROOM), walls)
    assert [[tuple(round(c, 6) for c in xy) for xy in p.line.coords] for p in pieces] == [[(0.0, 0.0), (10.0, 0.0)]]


def test_a_wall_off_the_edge_or_turned_away_from_it_counts_only_within_the_snap() -> None:
    import shapely
    from arena_simulation_setup.generative.flowedit.generate import WALL_SNAP_M, wall_pieces

    # 0.2 m behind the south edge, past the snap behind the north one, turned 14 degrees away from the east one
    walls = [((0.0, -0.2), (10.0, -0.2)), ((0.0, 4.0 + WALL_SNAP_M + 0.1), (10.0, 4.0 + WALL_SNAP_M + 0.1)), ((10.0, 0.0), (11.0, 4.0))]
    pieces = wall_pieces(shapely.Polygon(ROOM), walls)
    assert [[tuple(round(c, 6) for c in xy) for xy in p.line.coords] for p in pieces] == [[(0.0, 0.0), (10.0, 0.0)]]
    assert pieces[0].offset(5.0, 0.0) == pytest.approx(-0.2)


def test_the_hallway_wall_cut_by_a_door_at_its_middle_takes_objects_but_its_opening_does_not(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Spots

    spots = Spots(queue, "0")
    hallway = next(z for z in queue.world.levels["0"].zones if z.name == "central_hallway")
    east = _spans(spots.wall_edges(hallway), 12.0)
    assert (13.1, 16.6) in east and (18.1, 21.6) in east  # around the operating room door, the edge's middle (12, 17.1)
    assert not any(lo < 11.5 < hi for lo, hi in east)  # the sub hallway opens there: no wall


def test_door_layouts_move_add_and_remove_without_cutting_a_room_off(queue: Site) -> None:
    ops = set()
    for seed in range(12):
        (chain,) = generate(queue, seed=seed, types=[EditType.LAYOUT])
        edit, edited = chain.site.edits[-1], chain.site
        ops.add(edit.params["op"])
        if edit.params["op"] == "remove":
            rooms = {f.subject for f in queue.graph.facts if f.verb == "connects" and f.params.get("door") == edit.target} | {str(f.object) for f in queue.graph.facts if f.verb == "connects" and f.params.get("door") == edit.target}
            assert all(any(f.verb == "connects" and room in (f.subject, f.object) for f in edited.graph.facts) for room in rooms)
        if edit.params["op"] == "add":
            assert len({f.subject for f in edited.graph.facts if f.verb == "connects" and f.params.get("door") == edit.target} | {str(f.object) for f in edited.graph.facts if f.verb == "connects" and f.params.get("door") == edit.target}) == 2
    assert ops == {"move", "add", "remove"}


def test_every_edit_type_shows_up_on_the_shipped_worlds() -> None:
    seen = {c.records[0]["type"] for name in SHIPPED for c in generate(open_site(WORLDS / name), seed=0)}
    seen |= {c.records[0]["type"] for c in generate(open_site(HOSPITAL, "queue"), seed=0)}
    assert seen == {t.value for t in EditType}


@pytest.mark.parametrize("name", SHIPPED)
def test_every_sampled_edit_of_a_shipped_world_exports_a_world_that_loads(name: str) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    for chain in generate(open_site(WORLDS / name), seed=0):
        path = save(chain.site)
        WorldIdentifier(path.name).resolve_sync().load()


def test_cli_prints_the_manifest_and_writes_worlds_and_manifest(local_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import yaml
    from arena_simulation_setup.generative.flowedit.cli import main

    argv = ["generate", "hospital_1", "--scenario", "queue", "--seed", "2", "--types", "move,function", "--worlds", str(WORLDS)]
    assert main(argv) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert [c["edits"][0]["type"] for c in printed["chains"]] == ["move", "function"]
    assert not (local_root / "worlds").exists()
    assert main([*argv, "--write"]) == 0
    manifest = local_root / "flowedit" / "hospital_1__seed2.yaml"
    written = yaml.safe_load(manifest.read_text())
    assert [c["world"] for c in written["chains"]] == [c["world"] for c in printed["chains"]]
    assert all((Path(c["path"]) / "world.yaml").is_file() and not (Path(c["path"]) / "scenarios").exists() for c in written["chains"])


def test_cli_rejects_unknown_types(capsys: pytest.CaptureFixture[str]) -> None:
    from arena_simulation_setup.generative.flowedit.cli import main

    with pytest.raises(SystemExit):
        main(["generate", "hospital_1", "--types", "teleport"])
    assert "edit types are" in capsys.readouterr().err


def test_access_never_locks_a_room_off(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import spare_doors

    spare = set(spare_doors(queue))
    for seed in range(12):
        (chain,) = generate(queue, seed=seed, types=[EditType.ACCESS])
        edit = chain.site.edits[-1]
        if edit.params.get("locked") is True:
            assert edit.target in spare


def _move_chain(queue: Site) -> Generated:
    for seed in range(20):
        (chain,) = generate(queue, seed=seed, types=[EditType.MOVE])
        if chain.records[0]["function_carrying"]:
            return chain
    raise AssertionError("no function-carrying move in 20 seeds")


def test_a_function_carrying_move_gets_both_geometry_only_controls(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Control, controls

    chain = _move_chain(queue)
    move = chain.site.edits[0]
    made = dict(controls(queue, chain))
    assert set(made) == {Control.INERT_OBJECT, Control.MOVED_WITHOUT_DEMAND}
    inert = made[Control.INERT_OBJECT]
    ref = f"{move.target}_inert"
    at = next(f.params for f in inert.graph.facts if f.subject == ref and f.verb == "at")
    assert (at["x"], at["y"]) == (move.params["x"], move.params["y"])
    assert not [f for f in inert.graph.facts if f.subject == ref and f.verb in DEMAND_VERBS]
    assert [f.verb for f in inert.graph.facts if f.subject == move.target and f.verb == "satisfies"] == ["satisfies"]  # the original keeps its demand where it stood
    moved = made[Control.MOVED_WITHOUT_DEMAND]
    at = next(f.params for f in moved.graph.facts if f.subject == move.target and f.verb == "at")
    assert (at["x"], at["y"]) == (move.params["x"], move.params["y"])
    assert not [f for f in moved.graph.facts if f.subject == move.target and f.verb in DEMAND_VERBS]
    old = next(f.params for f in queue.graph.facts if f.subject == move.target and f.verb == "at")
    left = next(f.params for f in moved.graph.facts if f.subject == f"{move.target}_demand" and f.verb == "at")
    assert left == old  # the demand stays at the old spot
    assert [e.params.get("leave_demand") for e in moved.edits] == [True]


def test_a_function_carrying_insert_gets_only_the_inert_copy(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Control, controls

    for seed in range(20):
        (chain,) = generate(queue, seed=seed, types=[EditType.INSERT])
        if chain.records[0]["function_carrying"]:
            assert [kind for kind, _ in controls(queue, chain)] == [Control.INERT_OBJECT]
            return
    raise AssertionError("no function-carrying insert in 20 seeds")


@pytest.fixture(scope="module")
def edited(queue: Site) -> Site:
    """The queue site after a geometry-only move of the reception counter: an edited world to generate on."""
    from arena_simulation_setup.generative.flowedit.edit import apply

    return apply(queue, "move", "obj:reception_counter_1", {"x": 3.0})


def test_controls_on_an_edited_world_follow_each_chains_own_first_edit(edited: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Control, controls

    chains = generate(edited, seed=0, types=[EditType.MOVE, EditType.INSERT])
    assert [c.records[0]["function_carrying"] for c in chains] == [True, True]
    for chain in chains:
        first = next(e for e in chain.site.edits if e.id == chain.records[0]["id"])
        made = controls(edited, chain)
        assert made
        for kind, control in made:
            (edit,) = control.edits[len(edited.edits) :]
            assert edit.target == (first.target if kind is Control.MOVED_WITHOUT_DEMAND else f"{first.target}_inert")
            assert (edit.params["x"], edit.params["y"]) == (first.params["x"], first.params["y"])


def test_the_manifest_lists_only_a_controls_own_edits_on_an_edited_world(edited: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import controls

    chains = generate(edited, seed=0, types=[EditType.MOVE])
    made = {chain.site.name: controls(edited, chain) for chain in chains}
    manifest = dump_manifest(edited, chains, 0, 1, 1, chain_controls=made)
    listed = [(c["world"], c["ids"]) for entry in manifest["chains"] for c in entry["controls"]]
    assert listed and listed == [(control.name, [control.edits[-1].id]) for chain in chains for _, control in made[chain.site.name]]


def test_controls_refuse_a_site_that_is_not_the_chains_base(queue: Site, edited: Site) -> None:
    from arena_simulation_setup.generative.flowedit.edit import apply
    from arena_simulation_setup.generative.flowedit.generate import controls

    chain = generate(edited, seed=0, types=[EditType.MOVE])[0]
    other = apply(queue, "move", "obj:fountain", {"x": 15.0, "y": 11.5})
    with pytest.raises(ValueError, match="is not the base of"):
        controls(other, chain)


def test_controls_are_saved_and_listed_in_the_manifest(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import controls
    from arena_simulation_setup.tree.World import WorldIdentifier

    chain = _move_chain(queue)
    made = controls(queue, chain)
    worlds = {control.name: save(control) for _, control in made}
    for path in worlds.values():
        WorldIdentifier(path.name).resolve_sync().load()
    manifest = dump_manifest(queue, [chain], 0, 1, 1, worlds, {chain.site.name: made})
    assert [c["kind"] for c in manifest["chains"][0]["controls"]] == ["inert_object", "moved_without_demand"]


def test_geometry_only_edits_get_no_controls(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import controls

    (chain,) = generate(queue, seed=0, types=[EditType.LAYOUT])
    assert controls(queue, chain) == []


def test_cli_writes_the_controls_with_controls(local_root: Path) -> None:
    import yaml
    from arena_simulation_setup.generative.flowedit.cli import main

    assert main(["generate", "hospital_1", "--scenario", "queue", "--seed", "0", "--types", "move,insert", "--controls", "--write", "--worlds", str(WORLDS)]) == 0
    written = yaml.safe_load((local_root / "flowedit" / "hospital_1__seed0.yaml").read_text())
    made = [c for chain in written["chains"] for c in chain.get("controls", [])]
    assert made and all(Path(c["path"]).is_dir() for c in made)


def test_raised_objects_are_neither_moved_nor_copied() -> None:
    site = open_site(HOSPITAL)
    raised = {f"obj:{o.name}" for level in site.world.levels.values() for o in level.objects if o.pose.position.z > FLOOR_HEIGHT_M}
    assert raised and not raised & set(floor_objects(site))
    moved = {chain.site.edits[0].target for chain in generate(site, seed=0, per_type=10, types=[EditType.MOVE])}
    assert moved and not moved & raised


def test_without_a_known_face_people_queue_behind_the_yaw(queue: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    from arena_simulation_setup.generative.flowedit import generate

    monkeypatch.setitem(generate._FACES, (queue.root, "Some/Model"), None)
    monkeypatch.setitem(generate._FACES, (queue.root, "Odd/Model"), "up")
    assert generate.front_angle(queue.root, "Some/Model") == pytest.approx(math.pi)
    assert generate.front_angle(queue.root, "Odd/Model") == pytest.approx(math.pi)


def _docked_site(tmp_path: Path, docked: Docked, humansim: dict[str, Any], **fields: Any) -> Site:
    """hospital_1 without scenario plus `docked` against a wall with `fields`, its `humansim` block the only need."""
    root = tmp_path / "worlds" / "hospital_docked"
    shutil.copytree(HOSPITAL, root, ignore=shutil.ignore_patterns("scenarios"))
    world = yaml.safe_load((root / "world.yaml").read_text())
    world["levels"][0]["objects"].append({"name": docked.name, "model": docked.model, "pose": [20.0, docked.back, FACING_NORTH], "dock": docked.dock, "humansim": humansim, **fields})
    (root / "world.yaml").write_text(yaml.safe_dump(world, sort_keys=False))
    return open_site(root)


@pytest.fixture
def docked_locker(tmp_path: Path) -> Site:
    """hospital_1 without scenario plus a docked parcel locker against a wall, its only object with a need."""
    return _docked_site(tmp_path, LOCKER, {"type": "parcel_locker", "capacity": 1, "satisfies": {"parcel": 50.0}})


@pytest.fixture
def docked_charger(tmp_path: Path) -> Site:
    """hospital_1 without scenario plus a docked charger against a wall, given the only need so the generator draws it."""
    return _docked_site(tmp_path, CHARGER, {"type": "robot_charger", "capacity": 1, "satisfies": {"charge": 50.0}}, charge_power_w=CHARGE_POWER_W)


def _dock(site: Site, obj: str) -> list[Position] | None:
    found = world_object(site.world, obj.partition(":")[2])
    assert found is not None, obj
    level, i = found
    return level.objects[i].dock


def test_insert_copies_an_objects_dock(docked_locker: Site) -> None:
    (chain,) = generate(docked_locker, seed=0, types=[EditType.INSERT])
    edit = chain.site.edits[-1]
    assert edit.params["model"] == "parcel_locker" and edit.params["dock"] == LOCKER.dock
    assert _dock(chain.site, edit.target) == _dock(docked_locker, f"obj:{LOCKER.name}")


@pytest.mark.parametrize("kind", [EditType.MOVE, EditType.INSERT])
def test_the_inert_copy_keeps_the_dock(docked_locker: Site, kind: EditType) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Control, controls

    (chain,) = generate(docked_locker, seed=0, types=[kind])
    inert = dict(controls(docked_locker, chain))[Control.INERT_OBJECT]
    assert _dock(inert, f"{chain.site.edits[0].target}_inert") == _dock(docked_locker, f"obj:{LOCKER.name}")


def _charge_power(site: Site, obj: str) -> float | None:
    found = world_object(site.world, obj.partition(":")[2])
    assert found is not None, obj
    level, i = found
    return level.objects[i].charge_power_w


def test_insert_copies_a_chargers_charge_power(docked_charger: Site) -> None:
    (chain,) = generate(docked_charger, seed=0, types=[EditType.INSERT])
    edit = chain.site.edits[-1]
    assert edit.params["dock"] == CHARGER.dock and edit.params["charge_power_w"] == CHARGE_POWER_W
    assert _charge_power(chain.site, edit.target) == CHARGE_POWER_W
    assert _charge_power(open_site(save(chain.site)), edit.target) == CHARGE_POWER_W


@pytest.mark.parametrize("kind", [EditType.MOVE, EditType.INSERT])
def test_the_inert_copy_keeps_the_charge_power(docked_charger: Site, kind: EditType) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Control, controls

    (chain,) = generate(docked_charger, seed=0, types=[kind])
    inert = dict(controls(docked_charger, chain))[Control.INERT_OBJECT]
    assert _charge_power(inert, f"{chain.site.edits[0].target}_inert") == CHARGE_POWER_W
