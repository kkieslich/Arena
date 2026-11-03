from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.kg import Scene
from arena_simulation_setup.shared.judge import zone_polygons
from arena_simulation_setup.tree.World import FORMAT, WorldFormatError, WorldView, contents, derive, parse_world, write_world
from arena_simulation_setup.tree.World.model import World
from arena_simulation_setup.tree.World.view import check_kg
from arena_simulation_setup.utils.geometry import Position

DEMO = Path(__file__).resolve().parents[2] / 'worlds' / 'stairs_demo'

WORLD: dict = {
    'format': FORMAT,
    'layout': 'stacked',
    'levels': [
        {
            'id': '0',
            'zones': [{'name': 'hall', 'polygon': [[0, 0], [20, 0], [20, 10], [0, 10]]}],
            'stairs': [{'name': 'up', 'mouth': [6.0, 2.0], 'to': '1'}],
            'escalators': [{'name': 'ride', 'mouth': [2.0, 5.0], 'size': [14.0, 1.0], 'to': '1', 'style': 'belt'}],
        },
        {'id': '1', 'zones': [{'name': 'gallery', 'polygon': [[0, 0], [20, 0], [20, 10], [0, 10]]}]},
    ],
}


def _spread(data: dict) -> None:
    data['layout'] = 'spread'
    data['levels'][0]['stairs'][0]['exit'] = [3.0, 2.0]
    data['levels'][0]['escalators'][0]['exit'] = [9.0, 5.0]


def _world(change: Callable[[dict], object] | None = None) -> dict:
    data = copy.deepcopy(WORLD)
    if change is not None:
        change(data)
    return data


def _stairs(world: World) -> dict:
    return {stair.name: stair for level in world.all_levels for stair in level.stairs}


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _logged(name: str) -> Iterator[list[str]]:
    handler = _Records()
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)


def _write(directory: Path, data: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'world.yaml').write_text(yaml.safe_dump(data), encoding='utf-8')
    return directory


def test_one_entry_derives_a_bottom_half_and_a_top_half_one_storey_up() -> None:
    world = parse_world(_world())
    assert [(stair.name, stair.kind, stair.top) for stair in world.levels['0'].stairs] == [('up_bottom', 'stairs', False), ('ride_bottom', 'escalator', False)]
    assert [(stair.name, stair.kind, stair.top) for stair in world.levels['1'].stairs] == [('up_top', 'stairs', True), ('ride_top', 'escalator', True)]
    stairs = _stairs(world)
    bottom, top = stairs['up_bottom'], stairs['up_top']
    assert (bottom.group, top.group, bottom.rise, bottom.length) == ('up', 'up', 3.0, 2.6)
    assert bottom.mouth == Position(6.0, 2.0)
    assert (bottom.seam.x, bottom.seam.y) == pytest.approx((top.seam.x, top.seam.y))
    assert top.mouth == Position(11.2, 2.0)
    assert (bottom.mouth_side, top.mouth_side) == ('-x', '+x')
    assert world.validate() == []


def test_levels_take_their_elevation_or_a_storey_each() -> None:
    def tall(data: dict) -> None:
        data['levels'][1]['elevation'] = 4.2

    assert _stairs(parse_world(_world(tall)))['up_bottom'].rise == pytest.approx(4.2)
    world = parse_world(_world())
    assert (world.elevation('0'), world.elevation('1')) == (0.0, 3.0)


def test_stairs_board_both_ways_and_an_escalator_on_its_direction() -> None:
    stairs = _stairs(parse_world(_world()))
    assert (stairs['up_bottom'].to, stairs['up_top'].to) == ('up_top', 'up_bottom')
    assert (stairs['ride_bottom'].to, stairs['ride_top'].to) == ('ride_top', '')
    assert stairs['ride_bottom'].belt > 0.0 > stairs['ride_top'].belt
    down = _stairs(parse_world(_world(_set(0, 'escalators', 0, direction='down'))))
    assert (down['ride_bottom'].to, down['ride_top'].to) == ('', 'ride_bottom')


def test_a_spread_world_places_the_top_half_at_its_exit() -> None:
    stairs = _stairs(parse_world(_world(_spread)))
    assert stairs['up_top'].mouth == Position(3.0, 2.0)
    assert stairs['up_top'].overhang == 0.0
    assert not stairs['up_top'].stacked


def _set(level: int, key: str, index: int, **fields: object) -> Callable[[dict], None]:
    return lambda data: data['levels'][level][key][index].update(fields)


def _append(level: int, key: str, **fields: object) -> Callable[[dict], None]:
    return lambda data: data['levels'][level].setdefault(key, []).append(fields)


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        (_set(0, 'stairs', 0, to='9'), r"levels\[0\]\.stairs\[0\]\.to: no level '9' in this world"),
        (_set(0, 'stairs', 0, to='0'), r"levels\[0\]\.stairs\[0\]\.to: level '0' lies at elevation 0\.0, not above level '0' at 0\.0"),
        (_set(0, 'stairs', 0, exit=[3.0, 2.0]), r'levels\[0\]\.stairs\[0\]\.exit: a stacked world places the top half itself, right after the bottom half'),
        (lambda data: data.update(layout='spread'), r"levels\[0\]\.stairs\[0\]\.exit: a spread world takes the mouth of the top half on level '1', in its frame"),
        (lambda data: data.update(layout='tall'), r'layout: expected stacked or spread, got .tall.'),
        (_set(0, 'stairs', 0, direction='down'), r'levels\[0\]\.stairs\[0\]\.direction: stairs run both ways, only escalators take a direction'),
        (_set(0, 'escalators', 0, direction='sideways'), r"levels\[0\]\.escalators\[0\]\.direction: expected up or down, got 'sideways'"),
        (_append(0, 'stairs', name='up', mouth=[1.0, 8.0], to='1'), r"levels\[0\]\.stairs\[1\]\.name: stair 'up' is already named at levels\[0\]\.stairs\[0\]"),
        (_set(0, 'stairs', 0, style='belt'), r'levels\[0\]\.stairs\[0\]\.style: belt is an escalator style, stairs take steps'),
        (_set(0, 'stairs', 0, size=[0.0, 1.4]), r'levels\[0\]\.stairs\[0\]\.size: expected a positive \[length, width\]'),
        (_set(0, 'stairs', 0, size=[5.2]), r'levels\[0\]\.stairs\[0\]\.size: expected a positive \[length, width\]'),
        (_set(0, 'stairs', 0, mouth_side='up'), r"levels\[0\]\.stairs\[0\]\.mouth_side: expected one of"),
        (_set(0, 'escalators', 0, speed=0.0), r'levels\[0\]\.escalators\[0\]\.speed: must be positive, got 0\.0'),
        (_set(0, 'stairs', 0, tread=0.0), r'levels\[0\]\.stairs\[0\]\.tread: must be positive, got 0\.0'),
        (_set(0, 'stairs', 0, sides={'left': {'height': -1.0}}), r'levels\[0\]\.stairs\[0\]\.sides\.left\.height: must be 0 for no side wall or positive, got -1\.0'),
        (_set(0, 'stairs', 0, sides={'up': {}}), r"levels\[0\]\.stairs\[0\]\.sides: unknown keys \['up'\]"),
        (_set(0, 'stairs', 0, surfaces={'soffit': 'Marble'}), r"levels\[0\]\.stairs\[0\]\.surfaces: unknown keys \['soffit'\]"),
        (_set(0, 'stairs', 0, rails=[{'across': 'middle'}]), r"levels\[0\]\.stairs\[0\]\.rails\[0\]\.across: expected left, right or meters left of the axis within 0\.7, got 'middle'"),
        (_set(0, 'stairs', 0, rails=[{'across': 0.0, 'kind': 'a', 'material': 'Marble'}]), r'levels\[0\]\.stairs\[0\]\.rails\[0\]: takes kind or material, not both'),
        (_set(0, 'stairs', 0, headroom=0.0), r'levels\[0\]\.stairs\[0\]\.headroom: must be positive, got 0\.0'),
        (_set(0, 'escalators', 0, plank=0.0), r'levels\[0\]\.escalators\[0\]\.plank: must be positive, got 0\.0'),
        (_set(0, 'escalators', 0, landing_length=7.0), r'levels\[0\]\.escalators\[0\]\.landing_length: must lie in \[0, 7\.0\), half the length of the stair, got 7\.0'),
        (_set(0, 'stairs', 0, kind='belt_escalator'), r'levels\[0\]\.stairs\[0\]\.style: belt is an escalator style, stairs take steps'),
        (_set(0, 'escalators', 0, top=True), r"(?s)levels\[0\]\.escalators\[0\]: .*unknown keys \['top'\]"),
    ],
)
def test_a_broken_stair_is_named_by_its_yaml_path(change: Callable[[dict], None], message: str) -> None:
    with pytest.raises(WorldFormatError, match=message):
        parse_world(_world(change))


def test_stair_names_are_unique_across_kinds() -> None:
    world = parse_world(_world(_append(0, 'zones', name='up', polygon=[[0, 0], [1, 0], [1, 1]])))
    with pytest.raises(ValueError, match=r"name 'up' is used by a zone and a stair, names are unique across the world"):
        world.validate()


def test_floors_open_over_wells_overhangs_and_escalator_pits() -> None:
    def steps(data: dict) -> None:
        data['levels'][0]['escalators'][0].update(style='steps', size=[6.8, 1.0], mouth=[2.0, 5.0])

    world = parse_world(_world(steps))
    stairs = _stairs(world)
    well = stairs['up_top']
    assert well.overhang == pytest.approx(2.6 - 2.6 * 0.9 / 1.5)
    upper = [(round(f.pos.x - f.x_length / 2, 3), round(f.pos.y - f.y_length / 2, 3), round(f.pos.x + f.x_length / 2, 3), round(f.pos.y + f.y_length / 2, 3)) for f in world.levels['1'].all_floors]
    holes = [hole for stair in world.levels['1'].stairs for hole in stair.floor_openings()]
    assert holes[0] == pytest.approx((6.0 + 2.6 - well.overhang, 1.3, 11.2, 2.7))
    for x0, y0, x1, y1 in upper:
        for hx0, hy0, hx1, hy1 in holes:
            assert x1 <= hx0 + 1e-3 or x0 >= hx1 - 1e-3 or y1 <= hy0 + 1e-3 or y0 >= hy1 - 1e-3
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in upper) == pytest.approx(200.0 - sum((hx1 - hx0) * (hy1 - hy0) for hx0, hy0, hx1, hy1 in holes), abs=1e-2)
    lower = world.levels['0'].all_floors
    pit = stairs['ride_bottom'].floor_openings()[0]
    assert pit == pytest.approx((2.0 - stairs['ride_bottom'].pitch - 0.001, 4.5, 5.4, 5.5))
    assert sum(f.x_length * f.y_length for f in lower) == pytest.approx(200.0 - (pit[2] - pit[0]) * (pit[3] - pit[1]))


def test_the_overhang_is_guarded_on_the_upper_level_only() -> None:
    stairs = _stairs(parse_world(_world()))
    bottom, top = stairs['up_bottom'], stairs['up_top']
    assert bottom.overhang > 0.0
    assert all(bottom.along(point.x, point.y) <= bottom.length + 1e-9 for wall in bottom.walls() for point in (wall.start, wall.end))
    walls = top.walls()
    end = top.length + top.overhang
    guard = [(wall.start, wall.end, wall.bottom) for wall in walls if wall.bottom is None]
    assert any(abs(top.along(start.x, start.y) - end) < 1e-9 and abs(top.along(finish.x, finish.y) - end) < 1e-9 for start, finish, _ in guard)


def test_a_partner_on_a_level_not_loaded_is_a_warning() -> None:
    world = parse_world(_world())
    world.levels = {'0': world.levels['0']}
    assert world.validate(loaded_level_ids={'0'}) == ["stair 'up_bottom' leads to 'up_top' on a level that was not loaded, it will not board", "stair 'ride_bottom' leads to 'ride_top' on a level that was not loaded, it will not board"]
    with pytest.raises(ValueError, match=r"stair 'up_bottom' leads to unknown stair 'up_top'"):
        world.validate()


@pytest.mark.parametrize('spread', [False, True])
def test_written_stairs_are_one_entry_each_and_parse_back_equal(spread: bool) -> None:
    world = parse_world(_world(_spread if spread else None))
    text = write_world(world)
    written = yaml.safe_load(text)
    assert 'stairs' not in written['levels'][1] and 'escalators' not in written['levels'][1]
    stairs = {'name': 'up', 'mouth': [6.0, 2.0], 'size': [5.2, 1.4], 'to': '1'}
    assert written['levels'][0]['stairs'] == [{**stairs, 'exit': [3.0, 2.0]} if spread else stairs]
    again = parse_world(written)
    assert [level.stairs for level in again.all_levels] == [level.stairs for level in world.all_levels]
    assert write_world(again) == text


def test_prefixed_levels_prefix_stair_names_to_and_group() -> None:
    level = parse_world(_world()).levels['1'].prefixed('x_')
    assert [(stair.name, stair.to, stair.group) for stair in level.stairs] == [('x_up_top', 'x_up_bottom', 'x_up'), ('x_ride_top', '', 'x_ride')]


def test_compacted_levels_move_their_stairs() -> None:
    compact = parse_world(_world()).compact_world({'0': (0.0, 0.0), '1': (25.0, 0.0)})
    assert [(stair.name, stair.position.x) for stair in compact.stairs] == [('up_bottom', 7.3), ('ride_bottom', 5.5), ('up_top', 34.9), ('ride_top', 37.5)]


def test_a_stair_resolves_as_a_named_place_by_its_halves_and_its_name() -> None:
    level = parse_world(_world()).levels['0']
    assert level.zone_ref_names() == ['hall', 'up_bottom', 'ride_bottom', 'up', 'ride']
    assert level.lookup_zone_polygon('up') == level.lookup_zone_polygon('up_bottom') == level.stairs[0].corners()
    assert level.refs()['stair'] == ['up_bottom', 'ride_bottom']


def test_the_judge_finds_a_stair_on_both_levels_of_the_compacted_world() -> None:
    compact = parse_world(_world()).compact_world({'0': (0.0, 0.0), '1': (25.0, 0.0)})
    polygons = zone_polygons(compact)
    assert polygons['up_top'].bounds == pytest.approx((33.6, 1.3, 36.2, 2.7))
    assert polygons['up'].area == pytest.approx(2.0 * 2.6 * 1.4)
    assert polygons['up'].bounds == pytest.approx((6.0, 1.3, 36.2, 2.7))


def test_derived_kg_places_stairs_and_connects_their_pairs() -> None:
    world = parse_world(_world())
    scene = derive(world)
    assert sorted(contents(scene, 'hall')) == ['stair:ride_bottom', 'stair:up_bottom']
    facts = {(fact.subject, fact.verb, str(fact.object)) for fact in scene.facts}
    assert {('floor:0', 'contains', 'stair:up_bottom'), ('stair:up_bottom', 'connects_floor', 'stair:up_top'), ('stair:up_top', 'connects_floor', 'stair:up_bottom'), ('stair:ride_bottom', 'connects_floor', 'stair:ride_top')} <= facts
    assert ('stair:ride_top', 'connects_floor', 'stair:ride_bottom') not in facts
    assert scene.validate() == []


def test_a_kg_stair_must_name_a_world_stair() -> None:
    world = parse_world(_world())
    scene = Scene()
    scene.add_entity('stair:up')
    scene.add_entity('stair:up_bottom')
    check_kg(world, scene)
    scene.add_entity('stair:ghost')
    with pytest.raises(ValueError, match=r"stair:ghost: no stair named 'ghost' in world\.yaml"):
        check_kg(world, scene)


@pytest.fixture
def tower(tmp_path: Path) -> Path:
    return _write(tmp_path / 'tower', _world())


def test_levels_linked_under_one_prefix_keep_their_pairs(tower: Path, tmp_path: Path) -> None:
    main = _write(tmp_path / 'main', {'format': FORMAT, 'levels': [{'from': '../tower:0', 'prefix': 't_'}, {'from': '../tower:1', 'prefix': 't_'}]})
    stairs = _stairs(WorldView(main).load())
    assert [(name, stair.to) for name, stair in stairs.items()] == [('t_up_bottom', 't_up_top'), ('t_ride_bottom', 't_ride_top'), ('t_up_top', 't_up_bottom'), ('t_ride_top', '')]


def test_a_linked_to_leaving_the_world_is_dropped_with_a_warning(tower: Path, tmp_path: Path) -> None:
    main = _write(tmp_path / 'main', {'format': FORMAT, 'levels': [{'from': '../tower:0', 'prefix': 't_'}]})
    with _logged('arena_simulation_setup.tree.World.parse') as logged:
        stairs = _stairs(WorldView(main).load())
    assert [(name, stair.to) for name, stair in stairs.items()] == [('t_up_bottom', ''), ('t_ride_bottom', '')]
    assert "level 0 from ../tower:0: stair t_up_bottom loses 't_up_top', it is not in this world, it does not board" in logged


def test_the_demo_world_loads_with_belt_and_step_escalators() -> None:
    notes: list[str] = []
    world = WorldView(DEMO).load(notes=notes)
    assert notes == []
    assert world.layout == 'stacked'
    stairs = _stairs(world)
    assert {name: (stair.kind, stair.style, stair.to) for name, stair in stairs.items()} == {
        'stairs_bottom': ('stairs', 'steps', 'stairs_top'),
        'esc_up_bottom': ('escalator', 'belt', 'esc_up_top'),
        'esc_down_bottom': ('escalator', 'steps', ''),
        'stairs_top': ('stairs', 'steps', 'stairs_bottom'),
        'esc_up_top': ('escalator', 'belt', ''),
        'esc_down_top': ('escalator', 'steps', 'esc_down_bottom'),
    }


def test_a_stacked_world_stacks_every_level_at_its_elevation_and_a_spread_one_refuses() -> None:
    assert parse_world(_world()).stack_offsets() == {'0': (0.0, 0.0, 0.0), '1': (0.0, 0.0, 3.0)}
    with pytest.raises(ValueError, match=r'a spread world gives every level a floor plan of its own'):
        parse_world(_world(_spread)).stack_offsets()


def test_a_stacked_top_half_walls_its_sides_down_to_the_floor_below() -> None:
    stacked = _stairs(parse_world(_world()))['up_top']
    spread = _stairs(parse_world(_world(_spread)))['up_top']
    assert stacked.walls()[0].bottom == -3.0
    assert spread.walls()[0].bottom == -1.5


def _dressed(data: dict) -> None:
    data['levels'][0]['stairs'][0].update(
        surfaces={'steps': 'Oak_Planks', 'risers': 'Marble'},
        sides={'left': {'kind': 'glass_balustrade'}, 'right': {'material': 'Marble', 'height': 1.1}},
        rails=[{'across': 'left', 'material': 'Marble'}, {'across': 0.3, 'kind': 'steel_handrail', 'solid': True}],
    )


def test_the_top_half_takes_the_sides_and_rails_mirrored_and_a_dressed_stair_writes_back_equal() -> None:
    world = parse_world(_world(_dressed))
    stairs = _stairs(world)
    bottom, top = stairs['up_bottom'], stairs['up_top']
    assert (bottom.sides['left'].kind, bottom.sides['right'].height) == ('glass_balustrade', 1.1)
    assert (top.sides['right'].kind, top.sides['left'].height) == ('glass_balustrade', 1.1)
    assert [rail.across for rail in bottom.rails] == pytest.approx([bottom.width / 2.0 - 0.08, 0.3])
    assert [rail.across for rail in top.rails] == pytest.approx([-(bottom.width / 2.0 - 0.08), -0.3])
    assert top.material_of('risers').name == 'Marble'
    text = write_world(world)
    entry = yaml.safe_load(text)['levels'][0]['stairs'][0]
    assert entry['rails'][0] == {'across': 'left', 'material': 'Marble'}
    assert entry['sides'] == {'left': {'kind': 'glass_balustrade'}, 'right': {'material': 'Marble', 'height': 1.1}}
    assert write_world(parse_world(yaml.safe_load(text))) == text


def test_open_walls_reach_the_simulator_only_and_stair_walls_keep_their_height_under_a_ceiling() -> None:
    def ceiling(data: dict) -> None:
        _dressed(data)
        data['levels'][0]['zones'][0]['ceiling'] = {'height': 2.8}

    level = parse_world(_world(ceiling)).levels['0']
    stair_walls = [wall for stair in level.stairs for wall in stair.walls()]
    open_walls = [wall for stair in level.stairs for wall in stair.open_walls()]
    assert open_walls and not any(wall in level.all_walls for wall in open_walls)
    assert any(wall.kind == 'steel_handrail' for wall in level.all_walls)
    closed = asyncio.run(level.closed_walls())
    assert closed[-len(open_walls) :] == open_walls
    assert all(wall in closed for wall in stair_walls)


KINDS = {
    'mall': {
        'speed': 0.7,
        'material': 'Aluminum_Anodized',
        'surfaces': {'landings': 'Marble', 'fixture': 'Concrete_Smooth'},
        'sides': {'left': {'kind': 'glass_balustrade', 'height': 1.1}, 'right': {'material': 'Paint_Gloss'}},
        'rails': [{'across': 'left', 'material': 'Carpet_Charcoal', 'height': 1.0}],
    },
    'broken': {'mouth': [0.0, 0.0]},
}


def _kind(**fields: object) -> Callable[[dict], None]:
    return lambda data: data['levels'][0]['escalators'][0].update(kind='mall', **fields)


def test_a_stair_takes_its_kind_and_its_own_keys_win_key_by_key() -> None:
    world = parse_world(_world(_kind(speed=0.4, surfaces={'fixture': 'Marble'}, sides={'left': {'material': 'Oak_Planks'}})), stair_kind=KINDS.__getitem__)
    ride = _stairs(world)['ride_bottom']
    assert (ride.preset, ride.speed, ride.material.name) == ('mall', 0.4, 'Aluminum_Anodized')
    assert {key: material.name for key, material in ride.surfaces.items()} == {'landings': 'Marble', 'fixture': 'Marble'}
    assert (ride.sides['left'].kind, ride.sides['left'].material.name, ride.sides['left'].height) == ('', 'Oak_Planks', 1.1)
    assert ride.sides['right'].material.name == 'Paint_Gloss'
    assert [(rail.height, rail.material.name) for rail in ride.rails] == [(1.0, 'Carpet_Charcoal')]


def test_a_stair_over_a_kind_writes_back_only_its_own_keys() -> None:
    world = parse_world(_world(_kind(speed=0.5, sides={'left': {'height': 1.2}})), stair_kind=KINDS.__getitem__)
    text = write_world(world)
    entry = yaml.safe_load(text)['levels'][0]['escalators'][0]
    assert entry == {'name': 'ride', 'kind': 'mall', 'mouth': [2.0, 5.0], 'size': [14.0, 1.0], 'to': '1', 'style': 'belt', 'speed': 0.5, 'sides': {'left': {'height': 1.2}}}
    assert write_world(parse_world(yaml.safe_load(text), stair_kind=KINDS.__getitem__)) == text


@pytest.mark.parametrize(
    ('kind', 'message'),
    [
        ('broken', r"levels\[0\]\.escalators\[0\]\.kind: stair kind 'broken': unknown keys \['mouth'\]"),
        ('missing', r"levels\[0\]\.escalators\[0\]\.kind: 'missing'"),
    ],
)
def test_a_broken_or_missing_stair_kind_is_named_by_its_yaml_path(kind: str, message: str) -> None:
    with pytest.raises(WorldFormatError, match=message):
        parse_world(_world(lambda data: data['levels'][0]['escalators'][0].update(kind=kind)), stair_kind=KINDS.__getitem__)


def test_a_world_finds_a_stair_kind_in_its_own_assets_first(tmp_path: Path) -> None:
    kind = tmp_path / 'assets' / 'Common' / 'Stairs' / 'stairs'
    kind.mkdir(parents=True)
    (kind / 'stairs.yaml').write_text(yaml.safe_dump({'tread': 0.3, 'surfaces': {'risers': 'Marble'}}), encoding='utf-8')
    world = WorldView(_write(tmp_path, _world(_set(0, 'stairs', 0, kind='stairs')))).load()
    up = _stairs(world)['up_bottom']
    assert (up.preset, up.tread, up.material_of('risers').name) == ('stairs', 0.3, 'Marble')


@pytest.mark.parametrize('name', ['stairs', 'escalator', 'belt_escalator'])
def test_the_default_stair_kinds_ship_in_the_repo_and_carry_only_stair_keys(name: str) -> None:
    from arena_simulation_setup.tree.Stairs import STAIRS_KIND_KEYS, StairsIdentifier

    assert set(StairsIdentifier(name).resolve_sync()) <= STAIRS_KIND_KEYS


def _modelled(data: dict) -> None:
    data['levels'][0]['escalators'][0].update(
        surfaces={'steps': {'material': 'Aluminum_Anodized', 'model': 'tread_capture'}, 'landings': {'model': 'plate_capture'}, 'fixture': 'Concrete_Smooth'},
        models=[{'model': 'whole_capture', 'at': [7.0, 0.0, 0.0]}, {'model': 'sign', 'at': [1.0, 0.6, 2.2], 'yaw': 1.5708}],
    )


def test_surfaces_take_models_beside_materials_and_models_write_back() -> None:
    world = parse_world(_world(_modelled))
    ride = _stairs(world)["ride_bottom"]
    assert {key: material.name for key, material in ride.surfaces.items()} == {'steps': 'Aluminum_Anodized', 'fixture': 'Concrete_Smooth'}
    assert {key: model.name for key, model in ride.surface_models.items()} == {'steps': 'tread_capture', 'landings': 'plate_capture'}
    assert [(item.model.name, item.at, item.yaw) for item in ride.models] == [('whole_capture', (7.0, 0.0, 0.0), 0.0), ('sign', (1.0, 0.6, 2.2), 1.5708)]
    assert _stairs(world)['ride_top'].surface_models == ride.surface_models
    text = write_world(world)
    entry = yaml.safe_load(text)['levels'][0]['escalators'][0]
    assert entry['surfaces'] == {'steps': {'material': 'Aluminum_Anodized', 'model': 'tread_capture'}, 'landings': {'model': 'plate_capture'}, 'fixture': 'Concrete_Smooth'}
    assert entry['models'] == [{'model': 'whole_capture', 'at': [7.0, 0.0, 0.0]}, {'model': 'sign', 'at': [1.0, 0.6, 2.2], 'yaw': 1.5708}]
    assert write_world(parse_world(yaml.safe_load(text))) == text


@pytest.mark.parametrize(
    ('fields', 'message'),
    [
        ({'surfaces': {'brushes': {'model': 'x'}}}, r'levels\[0\]\.escalators\[0\]\.surfaces\.brushes\.model: only steps, risers, landings, fixture take a model, brushes takes a material'),
        ({'surfaces': {'steps': {'texture': 'x'}}}, r"levels\[0\]\.escalators\[0\]\.surfaces\.steps: unknown keys \['texture'\]"),
        ({'models': [{'at': [0, 0, 0]}]}, r"levels\[0\]\.escalators\[0\]\.models\[0\]: missing keys \['model'\]"),
    ],
)
def test_a_broken_model_is_named_by_its_yaml_path(fields: dict, message: str) -> None:
    with pytest.raises(WorldFormatError, match=message):
        parse_world(_world(lambda data: data['levels'][0]['escalators'][0].update(fields)))
