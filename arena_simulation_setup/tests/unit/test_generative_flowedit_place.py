from __future__ import annotations

import dataclasses
import itertools
import math
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import attrs
import pytest
import yaml
from arena_simulation_setup.generative.flowedit import place
from arena_simulation_setup.generative.flowedit.edit import EditType, Site, open_site, world_object
from arena_simulation_setup.generative.flowedit.place import CATALOG, Layout, Rule, Verdict, candidate_spots
from flowedit_docks import BAY, CHARGER, LOCKER, Docked

if TYPE_CHECKING:
    from arena_simulation_setup.generative.flowedit.place import Spot

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.usefixtures("local_root")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
MEMBERS = ["charger_1", "charger_2", "bay_1", "bay_2", "locker_1", "loading_point"]
#: The fleet study's chargers (Kolja: 22 Wh batteries, 175 W chargers)
STUDY_CHARGE_POWER_W = 175.0
#: How close the floor raster's corridor width comes to the geometric one: about 1.5 cells, by where walls fall on it
RASTER_M = 1.5 * place.RESOLUTION_M


@pytest.fixture(autouse=True)
def docked_assets(local_root: Path) -> dict[str, Path]:
    """The element models resolve in every test, faced +x, as an element's model must (`resolve_element`)."""
    from flowedit_docks import write_docked_assets

    return write_docked_assets(local_root)


@pytest.fixture(scope="module")
def hospital() -> Site:
    return open_site(HOSPITAL)


def _world(root: Path, objects: list[dict[str, Any]] = (), **level: Any) -> Site:  # type: ignore[assignment]
    """hospital_1 without scenarios at `root`, plus `objects` on its level and the level fields `level`."""
    shutil.copytree(HOSPITAL, root, ignore=shutil.ignore_patterns("scenarios"))
    world = yaml.safe_load((root / "world.yaml").read_text())
    world["levels"][0]["objects"] += list(objects)
    world["levels"][0].update(level)
    (root / "world.yaml").write_text(yaml.safe_dump(world, sort_keys=False))
    return open_site(root)


def _asset(local_root: Path, model: str, annotation: dict[str, Any]) -> None:
    from arena_simulation_setup.shared.entities import Obstacle

    directory = local_root / Obstacle.parse({"name": "probe", "model": model, "pose": [0.0, 0.0, 0.0]}).model.relpath()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "annotation.yaml").write_text(yaml.safe_dump(annotation))


def _valid_layout(site: Site, **kwargs: Any) -> Layout:
    """The first valid spot of each element, in catalog order, that keeps the layout valid."""
    chosen: list[Spot] = []
    for name in CATALOG:
        chosen.append(next(s for s in candidate_spots(site, name, **kwargs) if s.valid and not place.check_layout(site, Layout.of([*chosen, s]))))
    return Layout.of(chosen)


def _centre(polygon: Any) -> tuple[float, float]:  # noqa: ANN401 - a shapely polygon
    return polygon.centroid.x, polygon.centroid.y


@pytest.mark.parametrize("element", list(CATALOG))
def test_spots_are_the_same_for_the_same_site_and_cached(hospital: Site, element: str) -> None:
    first = candidate_spots(hospital, element)
    assert first and candidate_spots(hospital, element) is first
    place.clear_cache()
    again = candidate_spots(open_site(HOSPITAL), element)
    assert [s.id for s in again] == [s.id for s in first]
    assert [(o.x, o.y, o.yaw) for s in again for o in s.objects] == [(o.x, o.y, o.yaw) for s in first for o in s.objects]
    assert [(s.verdicts, s.corridor_m) for s in again] == [(s.verdicts, s.corridor_m) for s in first]


def _by_piece(spots: tuple[Spot, ...]) -> dict[tuple[str, str, int], list[Spot]]:
    out: dict[tuple[str, str, int], list[Spot]] = {}
    for s in spots:
        out.setdefault((s.level, s.zone, s.piece), []).append(s)
    return out


@pytest.mark.parametrize("element", list(CATALOG))
def test_spots_step_one_metre_along_each_wall_piece_centred_on_it(hospital: Site, element: str) -> None:
    pieces = _by_piece(candidate_spots(hospital, element))
    assert any(len(group) > 2 for group in pieces.values())
    for group in pieces.values():
        assert [s.step for s in group] == list(range(len(group)))
        assert all(b.along_m - a.along_m == pytest.approx(place.STEP_M, abs=1e-3) for a, b in itertools.pairwise(group))
        # the objects' footprints move 1 m along the wall from step to step, to the cm their poses are rounded to
        for a, b in itertools.pairwise(group):
            assert math.dist(_centre(a.objects[0].footprint), _centre(b.objects[0].footprint)) == pytest.approx(place.STEP_M, abs=0.02)
        # as much of the piece is left at its start as at its end
        first, last = group[0], group[-1]
        assert first.along_m == pytest.approx(first.piece_m - last.along_m, abs=1e-3)


def test_a_smaller_step_gives_more_spots_on_the_same_pieces(hospital: Site) -> None:
    metre = _by_piece(candidate_spots(hospital, "locker"))
    half = _by_piece(candidate_spots(hospital, "locker", step_m=0.5))
    assert set(half) == set(metre)
    assert all(len(half[key]) in (2 * len(group) - 1, 2 * len(group)) for key, group in metre.items())


@pytest.mark.parametrize("element", ["charger_pair", "bay_pair"])
def test_pair_members_stand_one_metre_apart_on_one_wall_piece(hospital: Site, element: str) -> None:
    spots = candidate_spots(hospital, element)
    assert spots
    for s in spots:
        a, b = s.objects
        assert (a.name, b.name) == CATALOG[element].members
        assert s.piece_m >= 2.0
        assert math.dist(_centre(a.footprint), _centre(b.footprint)) == pytest.approx(place.PAIR_SPACING_M, abs=0.02)
        assert a.yaw == b.yaw
        assert s.frame.depth(a.x, a.y) == pytest.approx(s.frame.depth(b.x, b.y), abs=0.02)
        assert a.dock.intersection(b.dock).area < place.AREA_EPS


def _slanted(root: Path, degrees: float = 30.0, seed: int | None = None) -> Site:
    """A 12 x 6 m hall walled all round, turned by `degrees`, with nothing in it."""
    turn = math.radians(degrees)
    corners = [[round(20.0 + x * math.cos(turn) - y * math.sin(turn), 6), round(20.0 + x * math.sin(turn) + y * math.cos(turn), 6)] for x, y in [(0.0, 0.0), (12.0, 0.0), (12.0, 6.0), (0.0, 6.0)]]
    if not (root / "world.yaml").is_file():
        root.mkdir(parents=True)
        world = {"format": "arena-world/2", "levels": [{"id": "0", "zones": [{"name": "hall", "polygon": corners}], "walls": [{"path": corners, "closed": True}], "objects": []}]}
        (root / "world.yaml").write_text(yaml.safe_dump(world, sort_keys=False))
    return open_site(root, seed=seed)


@pytest.mark.parametrize("element", ["charger_pair", "bay_pair"])
def test_pair_members_stand_exactly_one_metre_apart_on_a_slanted_wall(tmp_path: Path, element: str) -> None:
    """Rounding each member's pose on its own would leave them up to a cm off the spacing on a slanted wall, and a
    pair's docks (and a bay pair's slots) would overlap; the row's middle is rounded once instead."""
    spots = candidate_spots(_slanted(tmp_path / "worlds" / "slanted"), element)
    assert spots and all(abs(s.frame.u[0]) > 0.4 and abs(s.frame.u[1]) > 0.4 for s in spots)  # no wall runs along an axis
    for s in spots:
        a, b = s.objects
        assert math.dist((a.x, a.y), (b.x, b.y)) == pytest.approx(place.PAIR_SPACING_M, abs=2e-4)  # poses to 0.1 mm, as world.yaml keeps them
        assert a.dock.intersection(b.dock).area < place.AREA_EPS and a.footprint.intersection(b.footprint).area < place.AREA_EPS
        span = 2.0
        # a metre clear of the corners: nearer, the side wall bounds the corridor rays at the pair's edge (for review)
        if s.along_m - span / 2.0 > 1.0 and s.piece_m - s.along_m - span / 2.0 > 1.0:
            assert s.valid, s.failed
    assert sum(s.valid for s in spots) >= 10


@pytest.mark.parametrize(("element", "docked"), [("charger_pair", CHARGER), ("locker", LOCKER), ("bay_pair", BAY), ("loading_point", BAY)])
def test_objects_back_onto_the_wall_with_their_docks_in_the_room(hospital: Site, element: str, docked: Docked) -> None:
    import shapely

    spots = [s for s in candidate_spots(hospital, element) if s.verdict[Rule.AGAINST_WALL] is Verdict.PASS]
    assert spots
    for s in spots:
        for o in s.objects:
            # hospital_1's walls run on its zone edges: the origin stands where the docked tests put it off the wall
            assert s.frame.depth(o.x, o.y) == pytest.approx(docked.back, abs=0.015)
            assert min(s.frame.depth(x, y) for x, y in shapely.get_coordinates(o.footprint)) == pytest.approx(0.06, abs=0.01)
            assert s.frame.depth(*_centre(o.dock)) >= s.frame.depth(*_centre(o.footprint)) - 1e-6
            assert o.dock.covers(shapely.Point(o.goto))
            assert math.dist(o.goto, _centre(o.dock)) < 0.02  # a rectangular dock's deepest point is its middle


def test_the_locker_stands_by_its_middle_although_its_origin_is_a_back_corner(hospital: Site) -> None:
    for s in candidate_spots(hospital, "locker"):
        (o,) = s.objects
        assert s.frame.along(*_centre(o.footprint)) == pytest.approx(s.along_m, abs=0.02)
        assert abs(s.frame.along(o.x, o.y) - s.along_m) == pytest.approx(0.52, abs=0.02)


def test_an_object_without_a_bounding_box_blocks_no_spot_but_an_unresolved_one_does(tmp_path: Path, local_root: Path, hospital: Site) -> None:
    """The twin's `link_scan` is a scan of the whole site: its model resolves without a bounding box, as a floor marking's
    does, so it blocks nothing; decor whose model does not resolve keeps the generator's fallback box."""
    _asset(local_root, "Common/Object/site_scan", {"face": "+y", "note": "scene capture, not an obstacle"})
    spot = next(s for s in candidate_spots(hospital, "charger_pair") if s.valid)
    at = [spot.objects[0].x, spot.objects[0].y, 0.0]
    scanned = _world(tmp_path / "worlds" / "scanned", [{"name": "link_scan", "model": "Common/Object/site_scan", "pose": at}])
    decor = _world(tmp_path / "worlds" / "decor", [{"name": "decor", "model": "Common/Object/no_such_decor", "pose": at}])
    from arena_simulation_setup.generative.flowedit.generate import floor_objects

    assert "obj:link_scan" in floor_objects(scanned)  # the generator still counts it
    assert next(s for s in candidate_spots(scanned, "charger_pair") if s.id == spot.id).valid
    assert next(s for s in candidate_spots(decor, "charger_pair") if s.id == spot.id).verdict[Rule.NO_OVERLAP] is Verdict.FAIL


def test_rules_without_site_data_are_unknown_and_filter_nothing(hospital: Site) -> None:
    for name, element in CATALOG.items():
        spots = candidate_spots(hospital, name)
        assert [rule for rule, _ in spots[0].verdicts] == [rule for rule in Rule if rule in element.rules]
        for rule in (Rule.PEDS_ZONES, Rule.LIFT_DISTANCE, Rule.OUTLET_CABLE, Rule.TACTILE_PATH, Rule.EXIT_WIDTH, Rule.DEAD_END_TURNING):
            assert all(s.verdict.get(rule, Verdict.UNKNOWN) is Verdict.UNKNOWN for s in spots), rule
        assert any(s.valid and Verdict.UNKNOWN in s.verdict.values() for s in spots)
    assert Rule.OUTLET_CABLE in CATALOG["charger_pair"].rules and Rule.OUTLET_CABLE in CATALOG["locker"].rules
    assert Rule.DOOR_DISTANCE not in CATALOG["locker"].rules  # §3 asks it of chargers and bays


def test_door_distance_is_checked_on_a_level_with_doors_and_unknown_without(tmp_path: Path, hospital: Site) -> None:
    import shapely

    level = hospital.world.levels["0"]
    spans = shapely.union_all([shapely.LineString([(d.start.x, d.start.y), (d.end.x, d.end.y)]) for d in level.doors])
    spots = candidate_spots(hospital, "charger_pair")
    outcomes = {s.verdict[Rule.DOOR_DISTANCE] for s in spots}
    assert outcomes == {Verdict.PASS, Verdict.FAIL}
    for s in spots:
        near = shapely.union_all([g for o in s.objects for g in (o.footprint, o.dock)])
        assert (s.verdict[Rule.DOOR_DISTANCE] is Verdict.PASS) == (near.distance(spans) >= place.DOOR_DISTANCE_M)
    doorless = _world(tmp_path / "worlds" / "doorless", doors=[])
    assert {s.verdict[Rule.DOOR_DISTANCE] for s in candidate_spots(doorless, "charger_pair")} == {Verdict.UNKNOWN}


def test_spots_in_a_peds_zone_fail_once_the_zones_are_given(hospital: Site) -> None:
    spots = candidate_spots(hospital, "locker", peds_zones=["central_hallway"])
    assert {s.verdict[Rule.PEDS_ZONES] for s in spots if s.zone == "central_hallway"} == {Verdict.FAIL}
    assert Verdict.PASS in {s.verdict[Rule.PEDS_ZONES] for s in spots if s.zone != "central_hallway"}
    with pytest.raises(ValueError, match=r"no zones \['nowhere'\] for the PEDS zones"):
        candidate_spots(hospital, "locker", peds_zones=["nowhere"])


def _across(spots: tuple[Spot, ...], zone: str, axis: int) -> list[Spot]:
    """The spots on the walls of `zone` that face across it along `axis` (0: x, 1: y)."""
    return [s for s in spots if s.zone == zone and abs(s.frame.n[axis]) > 0.99]


def _facing_the_nurse_room(spots: tuple[Spot, ...]) -> list[Spot]:
    """The spots on the central hallway's east wall (the nurse resting room's) south of y 9.5. Across from them the
    west wall has only 1.5 m doors, narrower than a pair, so some ray of each pair meets the wall; further north the
    hallway opens to the 3 m sub hallway, which leaves more room in front."""
    return [s for s in _across(spots, "central_hallway", 0) if s.frame.n[0] < 0 and _centre(s.taken)[1] < 9.5]


def test_corridor_width_is_what_the_element_leaves_of_the_corridor(hospital: Site) -> None:
    spots = candidate_spots(hospital, "charger_pair")
    # the 4 m central hallway less the pair's 1.71 m (wall gap, charger, docks, approach)
    hall = _facing_the_nurse_room(spots)
    assert hall and max(s.corridor_m for s in hall) == pytest.approx(4.0 - 1.71, abs=RASTER_M)
    assert any(s.verdict[Rule.CORRIDOR_WIDTH] is Verdict.PASS for s in hall)
    # the 3 m sub hallway leaves 1.29 m: every spot facing across it fails
    sub = _across(spots, "sub_hallway", 1)
    assert sub and {s.verdict[Rule.CORRIDOR_WIDTH] for s in sub} == {Verdict.FAIL}
    assert max(s.corridor_m for s in sub) == pytest.approx(3.0 - 1.71, abs=RASTER_M)
    # a floor marking blocks only its slot
    bays = _facing_the_nurse_room(candidate_spots(hospital, "bay_pair"))
    assert bays and max(s.corridor_m for s in bays) == pytest.approx(4.0 - 1.26, abs=RASTER_M)


def test_an_object_in_front_fails_the_approach_the_turning_circle_and_the_docks(tmp_path: Path, hospital: Site) -> None:
    spot = next(s for s in _across(candidate_spots(hospital, "charger_pair"), "central_hallway", 0) if s.valid)
    x, y = spot.frame.at(spot.along_m, 1.0)
    kiosk = {"name": "kiosk", "model": CHARGER.model, "pose": [x, y, 0.0, 0.0, 0.0, spot.objects[0].yaw]}
    blocked = next(s for s in candidate_spots(_world(tmp_path / "worlds" / "kiosk", [kiosk]), "charger_pair") if s.id == spot.id)
    assert blocked.verdict[Rule.NO_OVERLAP] is Verdict.PASS  # 0.33 m from the chargers' front
    assert {blocked.verdict[r] for r in (Rule.CLEAR_ZONE, Rule.TURNING_CIRCLE, Rule.DOCKS_ON_FLOOR)} == {Verdict.FAIL}
    assert not blocked.valid


def test_excluded_zones_and_the_zone_list_limit_the_spots(hospital: Site) -> None:
    from arena_simulation_setup.tree.World.model import Level, Zone

    spots = candidate_spots(hospital, "loading_point")
    assert "central_hallway" in {s.zone for s in spots}
    rest = candidate_spots(hospital, "loading_point", exclude=("central_*",))
    assert [s.id for s in rest] == [s.id for s in spots if s.zone != "central_hallway"]
    assert {s.zone for s in candidate_spots(hospital, "loading_point", zones=["sub_hallway"])} == {"sub_hallway"}
    names = ["upper_concourse_north", "ramp", "stair_landing", "side_corridor_east", "side_corridor_west_north", "side_room", "south_arm", "lift_lobby"]
    level = Level(id="0", zones=[Zone(name=n, polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]) for n in names])
    assert place.candidate_zones(level) == ["upper_concourse_north"]
    with pytest.raises(ValueError, match="no zones"):
        place.candidate_zones(level, zones=["nowhere"])


def test_an_element_model_without_a_face_or_that_does_not_resolve_is_refused(tmp_path: Path, local_root: Path) -> None:
    site = _world(tmp_path / "worlds" / "refuse")
    _asset(local_root, "Common/Object/faceless", {"bounding_box": [[-0.2, 0.2], [-0.4, 0.4], [0.0, 1.0]]})
    faceless = place.Element("faceless", "Common/Object/faceless", ("faceless_1",), place.CHARGER_DOCK)
    with pytest.raises(ValueError, match="has no face annotation"):
        candidate_spots(site, faceless)
    with pytest.raises(ValueError, match="'Common/Object/no_such_element' does not resolve"):
        candidate_spots(site, attrs.evolve(faceless, model="Common/Object/no_such_element"))
    _asset(local_root, "Common/Object/marking", {"face": "+x"})
    with pytest.raises(ValueError, match="has no bounding box"):
        candidate_spots(site, attrs.evolve(faceless, model="Common/Object/marking"))
    with pytest.raises(ValueError, match="no element 'teleporter'"):
        candidate_spots(site, "teleporter")


def test_placed_elements_give_the_nav_filter_footprints_dock_points_and_blocking(hospital: Site) -> None:
    from arena_simulation_setup.metrics import navr_filter

    assert place.PlacedElement is navr_filter.PlacedElement  # one definition, the filter's
    assert [f.name for f in dataclasses.fields(place.PlacedElement)] == ["name", "footprint", "dock_point", "blocks_floor"]
    layout = _valid_layout(hospital)
    placed = place.placed_elements(layout)
    assert [p.name for p in placed] == MEMBERS
    by_name = {p.name: p for p in placed}
    assert {n for n, p in by_name.items() if p.blocks_floor} == {"charger_1", "charger_2", "locker_1"}
    for name in ("bay_1", "bay_2", "loading_point"):
        assert by_name[name].footprint.area == pytest.approx(1.2)  # the slot, where the filter parks a jackal
        assert math.dist(by_name[name].dock_point, _centre(by_name[name].footprint)) < 0.02
    import shapely

    for name in ("charger_1", "locker_1"):
        assert not by_name[name].footprint.covers(shapely.Point(by_name[name].dock_point))


def test_the_layout_id_hashes_its_spots_in_any_order(hospital: Site) -> None:
    layout = _valid_layout(hospital)
    assert Layout.of(reversed(layout.spots)).layout_id == layout.layout_id
    assert [s.element.name for s in Layout.of(reversed(layout.spots)).spots] == list(CATALOG)
    other = next(s for s in candidate_spots(hospital, "locker") if s.id != layout.spots[2].id)
    assert Layout.of([*layout.spots[:2], other, layout.spots[3]]).layout_id != layout.layout_id
    with pytest.raises(ValueError, match="one spot per element"):
        Layout.of([layout.spots[0], layout.spots[0]])


def test_a_layout_whose_elements_share_floor_is_refused(hospital: Site) -> None:
    chargers = next(s for s in _across(candidate_spots(hospital, "charger_pair"), "central_hallway", 0) if s.valid)
    clash = next(s for s in candidate_spots(hospital, "locker") if s.taken.intersection(chargers.keep_clear).area > 0.1)
    problems = place.check_layout(hospital, Layout.of([chargers, clash]))
    assert "charger_pair and locker overlap" in problems
    with pytest.raises(ValueError, match="overlap"):
        place.apply_layout(hospital, Layout.of([chargers, clash]))


def test_elements_facing_each_other_across_the_hallway_leave_too_narrow_a_corridor(hospital: Site) -> None:
    chargers = next(s for s in _across(candidate_spots(hospital, "charger_pair"), "central_hallway", 0) if s.valid)
    cx, cy = _centre(chargers.taken)
    facing = [s for s in _across(candidate_spots(hospital, "locker"), "central_hallway", 0) if s.frame.n[0] * chargers.frame.n[0] < 0]
    opposite = min(facing, key=lambda s: abs(_centre(s.taken)[1] - cy))
    assert abs(_centre(opposite.taken)[1] - cy) < 0.6 and abs(_centre(opposite.taken)[0] - cx) > 2.0
    assert chargers.corridor_m >= place.CORRIDOR_MIN_M
    problems = place.check_layout(hospital, Layout.of([chargers, opposite]))
    assert "charger_pair and locker overlap" not in problems
    assert any(p.startswith("charger_pair: the other elements leave a") for p in problems), problems


def test_apply_layout_inserts_every_object_with_its_dock_and_charge_power(hospital: Site) -> None:
    layout = _valid_layout(hospital)
    edited = place.apply_layout(hospital, layout)
    assert [e.type for e in edited.edits] == [EditType.INSERT] * len(MEMBERS)
    assert [e.target for e in edited.edits] == [f"obj:{name}" for name in MEMBERS]
    assert not any(f"obj:{name}" in hospital.graph.entities for name in MEMBERS)  # in memory, on a copy
    for spot in layout.spots:
        for o in spot.objects:
            found = world_object(edited.world, o.name)
            assert found is not None
            level, i = found
            obj = level.objects[i]
            assert (obj.pose.position.x, obj.pose.position.y) == pytest.approx((o.x, o.y), abs=1e-6)
            assert math.remainder(obj.pose.orientation.to_yaw() - o.yaw, math.tau) == pytest.approx(0.0, abs=1e-6)
            dock = obj.dock_polygon()
            assert dock is not None
            assert [(round(p.x, 3), round(p.y, 3)) for p in dock] == [(round(x, 3), round(y, 3)) for x, y in list(o.dock.exterior.coords)[:-1]]
            assert obj.charge_power_w == (STUDY_CHARGE_POWER_W if spot.element.name == "charger_pair" else None)
            assert edited.graph.entities[f"obj:{o.name}"].props["model"] == spot.element.model.rsplit("/", 1)[-1]


def test_apply_layout_moves_placed_objects_and_remove_element_removes_them(hospital: Site) -> None:
    layout = _valid_layout(hospital)
    edited = place.apply_layout(hospital, layout)
    old = layout.spots[2]
    # on the edited site the locker's own object is no obstacle to its spots, the others' objects and docks are
    lockers = candidate_spots(edited, "locker")
    assert old.id in {s.id for s in lockers if s.valid}
    others = Layout.of([s for s in layout.spots if s.element.name != "locker"])
    new = next(s for s in lockers if s.valid and s.id != old.id and not place.check_layout(edited, Layout.of([*others.spots, s])))
    moved = place.apply_layout(edited, Layout.of([new]))
    assert [(e.type, e.target) for e in moved.edits[len(edited.edits) :]] == [(EditType.MOVE, "obj:locker_1")]
    found = world_object(moved.world, "locker_1")
    assert found is not None
    level, i = found
    dock = level.objects[i].dock_polygon()
    assert dock is not None
    assert [(round(p.x, 2), round(p.y, 2)) for p in dock] == [(round(x, 2), round(y, 2)) for x, y in list(new.objects[0].dock.exterior.coords)[:-1]]
    removed = place.remove_element(moved, "locker")
    assert "obj:locker_1" not in removed.graph.entities and world_object(removed.world, "locker_1") is None
    assert removed.edits[-1].type is EditType.REMOVE
    bays = place.remove_element(moved, "bay_pair")
    assert [(e.type, e.target) for e in bays.edits[-2:]] == [(EditType.REMOVE, "obj:bay_1"), (EditType.REMOVE, "obj:bay_2")]


@pytest.fixture
def hospital_copy(local_root: Path) -> Path:
    """hospital_1 in the local worlds root, on the filesystem layouts are saved to, so their assets can be hardlinked."""
    root = local_root / "worlds" / "hospital_copy"
    shutil.copytree(HOSPITAL, root)
    return root


def test_save_layout_writes_a_world_copy_that_opens_with_the_layout(hospital_copy: Path) -> None:
    site = open_site(hospital_copy)
    layout = _valid_layout(site)
    path = place.save_layout(site, layout)
    assert path.name == f"hospital_copy__layout_{layout.layout_id}" == place.layout_name(site, layout)
    reopened = open_site(path)
    assert [e.type for e in reopened.edits] == [EditType.INSERT] * len(MEMBERS)
    for o in layout.objects:
        found = world_object(reopened.world, o.name)
        assert found is not None
        level, i = found
        obj = level.objects[i]
        assert (obj.pose.position.x, obj.pose.position.y) == pytest.approx((o.x, o.y), abs=1e-6)
        assert obj.dock is not None and len(obj.dock) == 4
        assert obj.charge_power_w == (STUDY_CHARGE_POWER_W if o.name.startswith("charger") else None)
    # the same spots on the reopened copy: its own objects are no obstacles to their element
    assert {s.id for s in candidate_spots(reopened, "charger_pair") if s.valid} >= {layout.spots[0].id}
    asset = next(p for p in (hospital_copy / "assets").rglob("*") if p.is_file())
    assert (path / asset.relative_to(hospital_copy)).stat().st_ino == asset.stat().st_ino


def test_insert_params_carry_the_dock_and_a_chargers_power(hospital: Site) -> None:
    layout = _valid_layout(hospital)
    chargers, bays = layout.spots[0], layout.spots[1]
    params = place.insert_params(chargers, chargers.objects[0])
    assert CATALOG["charger_pair"].charge_power_w == STUDY_CHARGE_POWER_W
    assert params["dock"] == [list(p) for p in place.CHARGER_DOCK] and params["charge_power_w"] == STUDY_CHARGE_POWER_W
    assert params["model"] == CHARGER.model and params["level"] == "0"
    assert "charge_power_w" not in place.insert_params(bays, bays.objects[0])
    assert place.insert_params(bays, bays.objects[0])["dock"] == [list(p) for p in place.SLOT]


def test_save_layout_refuses_a_name_without_the_layout_prefix(hospital_copy: Path) -> None:
    """`edit.save` replaces a world of the same name in the writable local root: a layout copy saved as the site's
    own name would delete it."""
    site = open_site(hospital_copy)
    layout = _valid_layout(site)
    for name in ("hospital_copy", "hospital_1", "anything", place.layout_prefix(site)):
        with pytest.raises(ValueError, match="hospital_copy__layout_<suffix>"):
            place.save_layout(site, layout, name)
    assert (hospital_copy / "world.yaml").is_file() and not (hospital_copy / "edits.yaml").exists()
    path = place.save_layout(site, layout, f"{place.layout_prefix(site)}premise_narrow")
    assert path.name == "hospital_copy__layout_premise_narrow"
    assert place.save_layout(site, layout, path.name) == path  # replaces its own layout copy


def test_apply_layout_gives_a_differing_object_the_catalogs_model_dock_and_power(hospital: Site) -> None:
    layout = _valid_layout(hospital)
    chargers = layout.spots[0]
    stale = attrs.evolve(CATALOG["charger_pair"], model="Common/Object/parcel_locker", dock=place.LOCKER_DOCK, charge_power_w=125.0)
    old = place.apply_layout(hospital, Layout.of([attrs.evolve(chargers, element=stale)]), check=False)
    edited = place.apply_layout(old, Layout.of([chargers]))
    assert [(e.type, e.target) for e in edited.edits[len(old.edits) :]] == [(EditType.REMOVE, "obj:charger_1"), (EditType.INSERT, "obj:charger_1"), (EditType.REMOVE, "obj:charger_2"), (EditType.INSERT, "obj:charger_2")]
    for o in chargers.objects:
        found = world_object(edited.world, o.name)
        assert found is not None
        obj = found[0].objects[found[1]]
        assert obj.charge_power_w == STUDY_CHARGE_POWER_W and [(p.x, p.y) for p in obj.dock or []] == list(place.CHARGER_DOCK)
        assert edited.graph.entities[f"obj:{o.name}"].props["model"] == "robot_charger"
    # as the catalog has them, they move
    again = place.apply_layout(edited, Layout.of([chargers]))
    assert [e.type for e in again.edits[len(edited.edits) :]] == [EditType.MOVE, EditType.MOVE]


def test_the_caches_keep_the_last_sites_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    place.clear_cache()
    monkeypatch.setattr(place, "CACHED_SITES", 2)
    root = tmp_path / "worlds" / "slanted"
    sites = [_slanted(root, seed=seed) for seed in (1, 2, 3)]
    first = candidate_spots(sites[0], "locker")
    candidate_spots(sites[1], "locker")
    assert candidate_spots(sites[0], "locker") is first  # used again: now the latest
    candidate_spots(sites[2], "locker")  # drops sites[1]
    assert {key[3] for key in place._SPOTS} == {1, 3} and {key[3] for key in place._CONTEXTS} == {1, 3}
    assert len(place._SITES) == 2
