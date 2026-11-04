from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from arena_simulation_setup.generative.flowedit.edit import EditType, Site, dump_edits, open_site, save
from arena_simulation_setup.generative.flowedit.generate import CAPACITY, FIELDS, dump_manifest, generate

pytest.importorskip("numpy")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
STEP_ONE = (EditType.REMOVE, EditType.FUNCTION, EditType.ACCESS)


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


def _edits(chains: list) -> list[dict]:
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


def test_function_samples_only_a_plausible_new_capacity(queue: Site) -> None:
    assert FIELDS == ("capacity",)
    for seed in range(8):
        (chain,) = generate(queue, seed=seed, types=[EditType.FUNCTION])
        edit = chain.site.edits[-1]
        assert set(edit.params) == {"capacity"}
        assert edit.target != "obj:gathering_area"  # capacity 0: queues without bound
        kind = next(str(f.object).partition(":")[2] for f in queue.graph.facts if f.subject == edit.target and f.verb == "serves")
        old = next(f.params.get("capacity", 1) for f in queue.graph.facts if f.subject == edit.target and f.verb == "serves")
        low, high = CAPACITY.get(kind, (1, 4))
        assert low <= edit.params["capacity"] <= high and edit.params["capacity"] != old
        assert chain.records[0]["function_carrying"] is True


def test_a_site_without_demand_gives_no_function_edit_and_a_note() -> None:
    notes: list[str] = []
    assert generate(open_site(HOSPITAL), types=[EditType.FUNCTION], notes=notes) == []
    assert notes == ["hospital_1: no function edit, no target"]


def test_remove_prefers_objects_with_demand(queue: Site) -> None:
    demand = {f.subject for f in queue.graph.facts if f.verb == "serves"}
    for seed in range(5):
        (chain,) = generate(queue, seed=seed, types=[EditType.REMOVE])
        assert chain.site.edits[-1].target in demand
        assert chain.records[0]["function_carrying"] is True


def test_every_sampled_edit_saves_a_world_that_loads(queue: Site, local_worlds: Path) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    for chain in generate(queue, seed=2, steps=2, types=STEP_ONE):
        path = save(chain.site)
        assert path.parent == local_worlds
        WorldIdentifier(path.name).resolve_sync().load()


def test_manifest_names_each_chain(queue: Site) -> None:
    chains = generate(queue, seed=0, types=STEP_ONE)
    manifest = dump_manifest(queue, chains, seed=0, per_type=1, steps=1)
    assert manifest["base"] == "hospital_1" and manifest["scenario"] == "queue" and manifest["fields"] == ["capacity"]
    assert [c["world"] for c in manifest["chains"]] == [c.site.name for c in chains]
    assert [c["edits"][0]["id"] for c in manifest["chains"]] == [c.site.edits[0].id for c in chains]


SHIPPED = sorted(p.name for p in WORLDS.iterdir() if not p.name.startswith(".") and (p / "world.yaml").is_file())


def _zone_of(site: Site, level_id: str, x: float, y: float):  # noqa: ANN202 - a world Zone
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
        assert Spots(queue, level_id).valid(edit.target, model, zone, edit.params["x"], edit.params["y"], edit.params["yaw"])
    assert scopes == {"local", "room"}


def test_insert_copies_an_objects_demand_onto_a_new_name_at_a_valid_spot(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import Spots

    for seed in range(4):
        (chain,) = generate(queue, seed=seed, types=[EditType.INSERT])
        edit = chain.site.edits[-1]
        assert edit.target.startswith("obj:gen_") and edit.target not in queue.graph.entities
        assert "type" in edit.params and chain.records[0]["function_carrying"] is True
        level = str(edit.params["level"])
        zone = _zone_of(queue, level, edit.params["x"], edit.params["y"])
        assert Spots(queue, level).valid(edit.target, edit.params["model"], zone, edit.params["x"], edit.params["y"], edit.params["yaw"])


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
def test_every_sampled_edit_of_a_shipped_world_exports_a_world_that_loads(name: str, local_worlds: Path) -> None:
    from arena_simulation_setup.tree.World import WorldIdentifier

    for chain in generate(open_site(WORLDS / name), seed=0):
        path = save(chain.site)
        WorldIdentifier(path.name).resolve_sync().load()


def test_every_sampled_edit_with_a_scenario_exports_it_so_it_loads(queue: Site) -> None:
    import yaml
    from arena_simulation_setup.generative.scenario.formats import check_scenario

    for chain in generate(queue, seed=5, steps=2):
        path = save(chain.site)
        check_scenario(yaml.safe_load((path / "scenarios" / "queue" / "scenario.yaml").read_text()), path)


def test_cli_prints_the_manifest_and_writes_worlds_and_manifest(local_worlds: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import yaml
    from arena_simulation_setup.generative.flowedit.cli import main

    argv = ["generate", "hospital_1", "--scenario", "queue", "--seed", "2", "--types", "move,function", "--worlds", str(WORLDS)]
    assert main(argv) == 0
    printed = yaml.safe_load(capsys.readouterr().out)
    assert [c["edits"][0]["type"] for c in printed["chains"]] == ["move", "function"]
    assert not local_worlds.exists()
    assert main([*argv, "--write"]) == 0
    manifest = local_worlds.parent / "flowedit" / "hospital_1__seed2.yaml"
    written = yaml.safe_load(manifest.read_text())
    assert [c["world"] for c in written["chains"]] == [c["world"] for c in printed["chains"]]
    assert all((Path(c["path"]) / "scenarios" / "queue" / "scenario.yaml").is_file() for c in written["chains"])


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


def _move_chain(queue: Site):  # noqa: ANN202 - a Generated chain
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
    assert not [f for f in inert.graph.facts if f.subject == ref and f.verb in ("serves", "satisfies", "dwells", "has_formation")]
    assert [f.verb for f in inert.graph.facts if f.subject == move.target and f.verb == "serves"] == ["serves"]  # the original keeps its demand where it stood
    stripped = made[Control.MOVED_WITHOUT_DEMAND]
    assert [e.type.value for e in stripped.edits] == ["move", "function"]
    assert not [f for f in stripped.graph.facts if f.subject == move.target and f.verb in ("serves", "satisfies", "dwells", "has_formation")]


def test_controls_are_saved_with_their_scenario_and_listed_in_the_manifest(queue: Site) -> None:
    import yaml
    from arena_simulation_setup.generative.flowedit.generate import controls

    chain = _move_chain(queue)
    made = controls(queue, chain)
    worlds = {control.name: save(control) for _, control in made}
    for _, control in made:
        doc = yaml.safe_load((worlds[control.name] / "scenarios" / "queue" / "scenario.yaml").read_text())
        names = {o["name"] for o in doc["objects"]}
        assert chain.site.edits[0].target.partition(":")[2] in names
    stripped = next(c for k, c in made if k.value == "moved_without_demand")
    doc = yaml.safe_load((worlds[stripped.name] / "scenarios" / "queue" / "scenario.yaml").read_text())
    obj = next(o for o in doc["objects"] if o["name"] == chain.site.edits[0].target.partition(":")[2])
    assert "humansim" not in obj
    manifest = dump_manifest(queue, [chain], 0, 1, 1, worlds, {chain.site.name: made})
    assert [c["kind"] for c in manifest["chains"][0]["controls"]] == ["inert_object", "moved_without_demand"]


def test_geometry_only_edits_get_no_controls(queue: Site) -> None:
    from arena_simulation_setup.generative.flowedit.generate import controls

    (chain,) = generate(queue, seed=0, types=[EditType.LAYOUT])
    assert controls(queue, chain) == []


def test_cli_writes_the_controls_with_controls(local_worlds: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import yaml
    from arena_simulation_setup.generative.flowedit.cli import main

    assert main(["generate", "hospital_1", "--scenario", "queue", "--seed", "0", "--types", "move,insert", "--controls", "--write", "--worlds", str(WORLDS)]) == 0
    written = yaml.safe_load((local_worlds.parent / "flowedit" / "hospital_1__seed0.yaml").read_text())
    made = [c for chain in written["chains"] for c in chain.get("controls", [])]
    assert made and all(Path(c["path"]).is_dir() for c in made)
