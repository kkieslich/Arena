from __future__ import annotations

import random

import attrs
import numpy as np
import pytest
import shapely
import shapely.affinity
import yaml

from arena_simulation_setup.metrics.nav_at_r import level_nav_at_r, nav_at_r
from arena_simulation_setup.metrics.navr_filter import JACKAL_RADIUS, WHEELCHAIR_RADIUS, NavrBase, PlacedElement, UnresolvedFootprintError, navr_check, parked_robot
from arena_simulation_setup.shared import Obstacle
from arena_simulation_setup.tree.World import FORMAT, WorldView, world_assets


def _zone(name, x0, y0, x1, y1):
    return {'name': name, 'polygon': [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


def _corridor_world():
    """Room a (5 x 4 m) and room b (3 x 4 m) joined by a 1.2 m wide, 3 m long hall at y 1.4..2.6, and a sealed room c."""
    outline = [[0, 0], [5, 0], [5, 1.4], [8, 1.4], [8, 0], [11, 0], [11, 4], [8, 4], [8, 2.6], [5, 2.6], [5, 4], [0, 4]]
    return {
        'zones': [_zone('a', 0, 0, 5, 4), _zone('hall', 5, 1.4, 8, 2.6), _zone('b', 8, 0, 11, 4), _zone('c', 12, 0, 15, 3)],
        'walls': [{'path': outline, 'closed': True}, {'path': [[12, 0], [15, 0], [15, 3], [12, 3]], 'closed': True}],
    }


def _two_rooms(gap):
    """Room a 4 x 4 m, room b 2 x 4 m, joined through a `gap` wide door centred at y = 2 in the shared wall x = 4."""
    low, high = 2.0 - gap / 2, 2.0 + gap / 2
    return {
        'zones': [_zone('a', 0.0, 0.0, 4.0, 4.0), _zone('b', 4.0, 0.0, 6.0, 4.0)],
        'walls': [
            {'path': [[4.0, 0.0], [0.0, 0.0], [0.0, 4.0], [4.0, 4.0]]},
            {'path': [[4.0, 0.0], [4.0, low]]},
            {'path': [[4.0, high], [4.0, 4.0]]},
            {'path': [[4.0, 0.0], [6.0, 0.0], [6.0, 4.0], [4.0, 4.0]]},
        ],
        'doors': [{'name': 'a_b', 'span': [[4.0, low], [4.0, high]]}],
    }


def _view(tmp_path, level, name='w'):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / 'world.yaml').write_text(yaml.safe_dump({'format': FORMAT, 'levels': [{'id': '0', **level}]}))
    return WorldView(root)


def _level(tmp_path, level, name='w'):
    return _view(tmp_path, level, name).load(validate=False).levels['0']


def _asset(view, model, annotation):
    directory = view.path / 'assets' / 'Common' / 'Object' / model
    directory.mkdir(parents=True)
    (directory / 'annotation.yaml').write_text(yaml.safe_dump(annotation))


def _box(name, x0, y0, x1, y1, dock=None, blocks=True):
    return PlacedElement(name, shapely.box(x0, y0, x1, y1), dock, blocks)


def _bay(name, x, y):
    """A waiting-bay slot (1.2 x 1.0 m floor marking) centred on (x, y), its dock the slot centre."""
    return PlacedElement(name, shapely.box(x - 0.6, y - 0.5, x + 0.6, y + 0.5), (x, y), False)


@pytest.fixture
def corridor(tmp_path):
    return NavrBase(_level(tmp_path, _corridor_world()))


def test_charger_narrowing_the_hall_below_the_wheelchair_is_rejected(corridor):
    # 0.6 m deep against the hall's south wall: 0.6 m stays, under 2 x 0.325 but over 2 x 0.267
    charger = _box('charger_1', 6.25, 1.4, 6.75, 2.0, dock=(6.5, 2.3))
    verdict = navr_check(corridor, [charger])
    assert not verdict.ok
    assert verdict.floor_loss_m2 > 5.0  # room b and the hall's east end
    assert verdict.new_cutoff_zones == ('b',)
    assert verdict.unreachable_docks == ()  # the jackal still passes and reaches the dock
    assert verdict.relabelled == (WHEELCHAIR_RADIUS,)
    assert verdict.nav_at_r[WHEELCHAIR_RADIUS] < corridor.nav_at_r(WHEELCHAIR_RADIUS)
    assert verdict.dominant_m2[JACKAL_RADIUS] > verdict.dominant_m2[WHEELCHAIR_RADIUS] + 5.0
    assert 'leaves the dominant component' in verdict.reasons[0]


def test_charger_against_a_wide_room_wall_is_accepted_without_relabelling(corridor):
    charger = _box('charger_1', 1.75, 0.0, 2.25, 0.6, dock=(2.0, 1.2))
    verdict = navr_check(corridor, [charger], start=(1.0, 2.0))
    assert verdict.ok
    assert verdict.floor_loss_m2 == 0.0
    assert verdict.new_cutoff_zones == ()
    assert verdict.unreachable_docks == ()
    assert verdict.relabelled == ()
    assert verdict.dominant_m2[WHEELCHAIR_RADIUS] < corridor.dominant_m2(WHEELCHAIR_RADIUS)
    assert min(verdict.start_share.values()) > 0.75  # the sealed room c is the rest


def test_floor_walled_in_by_elements_is_lost_without_relabelling(corridor):
    # three lockers close a 2.0 x 1.4 m pocket against room a's south wall: about 1 m2 at r 0.325 strands
    walls = [_box('locker_1', 1.0, 0.0, 1.2, 1.6), _box('locker_2', 3.2, 0.0, 3.4, 1.6), _box('locker_3', 1.0, 1.4, 3.4, 1.6)]
    verdict = navr_check(corridor, walls)
    assert not verdict.ok
    assert verdict.floor_loss_m2 == pytest.approx(1.0, abs=0.2)
    assert verdict.relabelled == ()
    full = navr_check(corridor, walls, relabel=True)
    assert attrs.evolve(verdict, nav_at_r={}) == attrs.evolve(full, relabelled=(), nav_at_r={})
    assert verdict.nav_at_r == pytest.approx(full.nav_at_r)


def test_dock_behind_a_wall_or_off_the_map_is_unreachable(corridor):
    locker = _box('locker_1', 10.4, 1.5, 11.0, 2.5, dock=(12.5, 1.5))  # dock in the sealed room c
    far = _box('charger_1', 1.75, 0.0, 2.25, 0.6, dock=(50.0, 1.0))
    verdict = navr_check(corridor, [locker, far])
    assert not verdict.ok
    assert verdict.floor_loss_m2 == 0.0
    assert verdict.unreachable_docks == ('locker_1', 'charger_1')


def test_parked_jackal_on_a_doorway_bay_cuts_the_room_off_but_its_own_dock_stays_reachable(tmp_path):
    base = NavrBase(_level(tmp_path, _two_rooms(1.0)))
    bay = _bay('bay_1', 4.0, 2.0)
    assert navr_check(base, [bay], park_radius=None).ok  # the marking alone blocks nothing
    verdict = navr_check(base, [bay])
    assert not verdict.ok
    assert verdict.floor_loss_m2 > 3.0
    assert verdict.new_cutoff_zones == ('b',)
    assert verdict.unreachable_docks == ()


def test_bay_pair_keeps_both_docks_reachable_with_the_other_slot_occupied(corridor):
    bays = [_bay('bay_1', 1.5, 0.6), _bay('bay_2', 2.5, 0.6)]
    verdict = navr_check(corridor, bays)
    assert verdict.ok
    assert verdict.unreachable_docks == ()
    assert verdict.floor_loss_m2 == 0.0
    # a slot in open floor, no wall within reach of its own goto check
    assert navr_check(corridor, [_bay('bay_1', 2.5, 2.0)]).ok


def test_dock_covered_by_another_element_is_unreachable(corridor):
    bay = _bay('bay_1', 2.0, 2.0)
    locker = _box('locker_1', 1.6, 1.6, 2.4, 2.4, dock=(2.0, 3.2))
    verdict = navr_check(corridor, [bay, locker])
    assert verdict.unreachable_docks == ('bay_1',)


def test_dock_inside_a_narrow_nook_is_unreachable_for_the_jackal(corridor):
    # two lockers 0.4 m apart against room a's west wall: the nook between them is under 2 x 0.267
    lockers = [_box('locker_1', 0.0, 1.0, 0.8, 1.8, dock=(1.5, 1.4)), _box('locker_2', 0.0, 2.2, 0.8, 3.0, dock=(0.4, 2.0))]
    verdict = navr_check(corridor, lockers)
    assert verdict.unreachable_docks == ('locker_2',)


@pytest.mark.parametrize('footprint', [None, shapely.Polygon()])
def test_element_without_a_footprint_raises_instead_of_counting_as_floor(corridor, footprint):
    with pytest.raises(UnresolvedFootprintError, match="'charger_1'"):
        navr_check(corridor, [PlacedElement('charger_1', footprint, (2.0, 1.2), True)])


def test_base_object_whose_model_does_not_resolve_raises(tmp_path):
    level = _two_rooms(1.0)
    level['objects'] = [{'name': 'charger_1', 'model': 'no_such_model', 'pose': {'x': 1.0, 'y': 1.0, 'z': 0.0}}]
    view = _view(tmp_path, level)
    with pytest.raises(UnresolvedFootprintError, match="'charger_1'"):
        NavrBase(view.load(validate=False).levels['0'], view=view)
    assert nav_at_r(view)['0'].missing == ('charger_1',)


def test_base_object_without_a_bbox_is_floor_and_excluded_objects_are_dropped(tmp_path):
    level = _two_rooms(1.0)
    level['objects'] = [
        {'name': 'scan', 'model': 'scan', 'pose': {'x': 1.0, 'y': 1.0, 'z': 0.0}},
        {'name': 'crate_1', 'model': 'crate', 'pose': {'x': 4.0, 'y': 2.0, 'z': 0.0}},
    ]
    view = _view(tmp_path, level)
    _asset(view, 'scan', {'name': 'scan'})
    _asset(view, 'crate', {'bounding_box': [[-0.3, 0.3], [-0.6, 0.6], [0.0, 0.8]]})
    world = view.load(validate=False)
    blocked = NavrBase(world.levels['0'], view=view, world=world)
    assert blocked.unannotated == 1
    assert blocked.cut_off_zones(WHEELCHAIR_RADIUS) == ('b',)
    assert NavrBase(world.levels['0'], view=view, world=world, exclude={'crate_1'}).cut_off_zones(WHEELCHAIR_RADIUS) == ()


def test_footprint_cells_match_the_full_render(tmp_path):
    level = _level(tmp_path, _corridor_world())
    base = NavrBase(level, radii=())
    for i, poly in enumerate([shapely.box(1.03, 0.0, 1.71, 0.64), shapely.affinity.rotate(shapely.box(6.1, 1.5, 6.9, 2.1), 37, use_radians=False), shapely.Point(9.37, 2.11).buffer(JACKAL_RADIUS, quad_segs=16), shapely.box(10.5, 3.5, 11.6, 4.6)]):
        grid, _ = level.render_grid(resolution=base.resolution, asset_color='black', asset_name_color=None, static_objects=[(f'p{i}', poly)])
        row, col, mask = base.frame.polygon_mask(poly)
        mine = base.occupied.copy()
        mine[row : row + mask.shape[0], col : col + mask.shape[1]] |= mask
        assert np.array_equal(mine, grid != 255), i


def _layouts(rng, count):
    for _ in range(count):
        elements = []
        for k in range(rng.randint(1, 4)):
            x, y = rng.uniform(0.0, 11.0), rng.uniform(0.0, 4.0)
            w, h = rng.uniform(0.3, 1.0), rng.uniform(0.3, 1.0)
            poly = shapely.affinity.rotate(shapely.box(x - w / 2, y - h / 2, x + w / 2, y + h / 2), rng.uniform(0, 180))
            dock = (rng.uniform(0.0, 11.0), rng.uniform(0.0, 4.0))
            elements.append(PlacedElement(f'e{k}', poly, dock, True))
            if rng.random() < 0.4:
                slot = (rng.uniform(0.0, 11.0), rng.uniform(0.0, 4.0))
                elements.append(PlacedElement(f'bay{k}', shapely.Point(slot).buffer(0.5), slot, False))
        yield elements


def _obstacles(elements):
    """The blocking footprints and the jackals parked on the slots, as level_nav_at_r's static objects."""
    return [(e.name, e.footprint) for e in elements if e.blocks_floor] + [(e.name, parked_robot(e)) for e in elements if not e.blocks_floor]


def test_incremental_check_matches_full_relabelling_and_level_nav_at_r(tmp_path):
    level = _level(tmp_path, _corridor_world())
    base = NavrBase(level)
    local = 0
    for elements in _layouts(random.Random(7), 120):
        fast = navr_check(base, elements, start=(1.0, 2.0))
        full = navr_check(base, elements, start=(1.0, 2.0), relabel=True)
        local += not fast.relabelled
        assert attrs.evolve(fast, relabelled=(), nav_at_r={}) == attrs.evolve(full, relabelled=(), nav_at_r={})
        assert fast.nav_at_r == pytest.approx(full.nav_at_r)
        blocking = _obstacles(elements)
        for radius in (WHEELCHAIR_RADIUS, JACKAL_RADIUS):
            ref = level_nav_at_r(level, radius=radius, start=(1.0, 2.0), static_objects=blocking)
            assert fast.nav_at_r[radius] == pytest.approx(ref.value)
            assert fast.dominant_m2[radius] == pytest.approx(ref.largest_share * ref.traversable_area)
            assert fast.start_share[radius] == pytest.approx(ref.start_share)
    assert local > 60  # most layouts need no relabelling


def test_slot_goto_lifts_only_its_own_jackal(tmp_path):
    base = NavrBase(_level(tmp_path, _corridor_world()))
    checked = 0
    for elements in _layouts(random.Random(11), 80):
        verdict = navr_check(base, elements)
        for e in elements:
            if e.blocks_floor:
                continue
            checked += 1
            # the other slots' jackals as plain obstacles, this slot unparked, everything relabelled in full
            others = [o if o.blocks_floor else PlacedElement(f'{o.name}/parked', parked_robot(o), None, True) for o in elements if o is not e]
            without = navr_check(base, [*others, e], park_radius=None, relabel=True)
            assert (e.name in verdict.unreachable_docks) == (e.name in without.unreachable_docks), e.name
    assert checked > 20


def test_level_nav_at_r_takes_in_memory_objects_like_annotated_entities(tmp_path):
    level = _two_rooms(1.0)
    view = _view(tmp_path, level)
    _asset(view, 'crate', {'bounding_box': [[-0.3, 0.3], [-0.6, 0.6], [0.0, 0.8]]})
    world = view.load(validate=False)
    crate = Obstacle.parse({'name': 'crate_1', 'model': 'crate', 'pose': [4.0, 2.0, 0.0]})
    edited = attrs.evolve(world, levels={'0': attrs.evolve(world.levels['0'], objects=[crate])})
    assert nav_at_r(view)['0'].components == 1
    furnished = nav_at_r(view, world=edited)['0']
    assert furnished.components == 2
    assert furnished.cut_off_zones == ('b',)
    in_memory = level_nav_at_r(world.levels['0'], static_objects=[('crate_1', shapely.box(3.7, 1.4, 4.3, 2.6))])
    assert in_memory.value == pytest.approx(furnished.value)
    assert in_memory.cut_off_zones == ('b',)
    with world_assets(view, edited):
        assert level_nav_at_r(edited.levels['0']).value == pytest.approx(furnished.value)
