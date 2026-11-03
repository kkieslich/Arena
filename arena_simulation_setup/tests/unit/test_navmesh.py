from __future__ import annotations

from pathlib import Path

import attrs
import numpy as np
import pytest
import yaml

from arena_simulation_setup.shared.navmesh import NAV_CLIMB, NavLevel, NavMesh, build_navmesh, navmesh_levels
from arena_simulation_setup.shared.world import Floor
from arena_simulation_setup.tree.World import WorldView, parse_world
from arena_simulation_setup.utils.geometry import Position

WORLDS = Path(__file__).resolve().parents[2] / 'worlds'


def _stacked_demo() -> list[NavLevel]:
    world = WorldView(WORLDS / 'stairs_demo').load()
    return navmesh_levels(world, world.stack_offsets())


def _at(mesh: NavMesh, x: float, y: float, z: float | None = None) -> np.ndarray:
    near = np.all(np.abs(mesh.vertices[:, :2] - (x, y)) < 1e-6, axis=1)
    if z is not None:
        near &= np.abs(mesh.vertices[:, 2] - z) < 1e-6
    return np.flatnonzero(near)


def _component(mesh: NavMesh, x: float, y: float, z: float) -> int:
    (index,) = _at(mesh, x, y, z)
    return int(mesh.components()[index])


def _only(levels: list[NavLevel], group: str | None) -> list[NavLevel]:
    return [attrs.evolve(level, stairs=[stair for stair in level.stairs if stair.group == group]) for level in levels]


def test_the_stacked_demo_is_one_surface_from_the_lobby_to_the_gallery() -> None:
    levels = _stacked_demo()
    assert all(stair.joined for level in levels for stair in level.stairs)
    mesh = build_navmesh(levels)
    assert mesh.components().max() == 0
    assert _component(mesh, 15.0, 8.0, 0.0) == _component(mesh, 15.0, 8.0, 3.0)
    assert mesh.vertices[:, 2].min() == pytest.approx(0.0)
    assert mesh.vertices[:, 2].max() < 3.05


@pytest.mark.parametrize('group', ['stairs', 'esc_up', 'esc_down'])
def test_each_stair_and_escalator_alone_joins_the_levels_through_its_footprint(group: str) -> None:
    levels = _only(_stacked_demo(), group)
    mesh = build_navmesh(levels)
    assert _component(mesh, 15.0, 8.0, 0.0) == _component(mesh, 15.0, 8.0, 3.0)
    for level, low, high in ((levels[0], 0.2, 1.4), (levels[1], 1.6, 2.8)):
        (stair,) = level.stairs
        inside = np.array([stair.contains(x, y) for x, y in mesh.vertices[:, :2]])
        heights = mesh.vertices[inside, 2]
        assert np.any((heights > low) & (heights < high))
        assert heights.min() >= -1e-9
        assert heights.max() <= 3.05


def test_without_stairs_the_levels_stay_apart() -> None:
    mesh = build_navmesh(_only(_stacked_demo(), None))
    assert _component(mesh, 15.0, 8.0, 0.0) != _component(mesh, 15.0, 8.0, 3.0)


def test_faces_point_up_and_climb_at_most_one_step() -> None:
    mesh = build_navmesh(_stacked_demo())
    a, b, c = (mesh.vertices[mesh.faces[:, k]] for k in range(3))
    assert np.all(np.cross(b - a, c - a)[:, 2] > 0.0)
    heights = mesh.vertices[mesh.faces, 2]
    assert np.all(heights.max(axis=1) - heights.min(axis=1) <= NAV_CLIMB + 1e-9)


def test_a_flat_world_lies_on_its_floor_and_leaves_out_the_points_under_walls() -> None:
    world = WorldView(WORLDS / 'sbend_corridor').load()
    mesh = build_navmesh(navmesh_levels(world, {'0': (0.0, 0.0, 0.0)}))
    assert np.all(mesh.vertices[:, 2] == 0.0)
    assert len(mesh.vertices) < 101 * 131
    assert mesh.components().max() == 0
    assert len(_at(mesh, 5.0, 0.0)) == 0
    assert len(_at(mesh, 5.0, 3.0)) == 0
    assert len(_at(mesh, 5.0, 5.0)) == 0
    assert len(_at(mesh, 5.0, 4.0)) == 0
    for x, y in ((5.0, 0.1), (5.0, 2.9), (5.0, 5.1), (5.0, 12.9)):
        assert len(_at(mesh, x, y)) == 1


def test_a_shift_moves_the_mesh_by_it() -> None:
    world = WorldView(WORLDS / 'sbend_corridor').load()
    still = build_navmesh(navmesh_levels(world, {'0': (0.0, 0.0, 0.0)}))
    moved = build_navmesh(navmesh_levels(world, {'0': (2.0, -1.0, 0.5)}))
    assert len(moved.vertices) == len(still.vertices)
    assert np.allclose(np.sort(moved.vertices - (2.0, -1.0, 0.5), axis=0), np.sort(still.vertices, axis=0))


def test_the_ply_file_reads_back_with_its_counts_and_values(tmp_path: Path) -> None:
    mesh = build_navmesh(_stacked_demo())
    path = mesh.write_ply(tmp_path / 'demo.ply')
    data = path.read_bytes()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    header = data[:end].decode('ascii').splitlines()
    assert header[:2] == ['ply', 'format binary_little_endian 1.0']
    counts = {line.split()[1]: int(line.split()[2]) for line in header if line.startswith('element')}
    assert counts == {'vertex': len(mesh.vertices), 'face': len(mesh.faces)}
    assert 'property list uchar uint vertex_indices' in header
    body = data[end:]
    assert len(body) == 12 * counts['vertex'] + 13 * counts['face']
    vertices = np.frombuffer(body, dtype='<f4', count=3 * counts['vertex']).reshape(-1, 3)
    faces = np.frombuffer(body[12 * counts['vertex'] :], dtype=[('count', 'u1'), ('index', '<u4', (3,))])
    assert np.allclose(vertices, mesh.vertices, atol=1e-5)
    assert np.all(faces['count'] == 3)
    assert np.array_equal(faces['index'], mesh.faces)


TWO_ROOMS = """\
format: arena-world/2
levels:
- id: '0'
  zones:
  - {name: west, polygon: [[0, 0], [4, 0], [4, 4], [0, 4]], floor: Concrete_Smooth}
  - {name: east, polygon: [[4, 0], [8, 0], [8, 4], [4, 4]], floor: Concrete_Smooth}
  walls:
  - {path: [[0, 0], [8, 0], [8, 4], [0, 4]], closed: true}
  - {path: [[4, 0], [4, 4]]}
  doors:
  - {name: between, span: [[4, 1.5], [4, 2.5]]}
"""


def _two_rooms() -> list[NavLevel]:
    return navmesh_levels(parse_world(yaml.safe_load(TWO_ROOMS)), {})


def test_a_door_opening_joins_both_rooms() -> None:
    mesh = build_navmesh(_two_rooms())
    assert len(_at(mesh, 4.0, 2.0)) == 1
    assert len(_at(mesh, 4.0, 1.0)) == 0
    assert _component(mesh, 2.0, 2.0, 0.0) == _component(mesh, 4.0, 2.0, 0.0) == _component(mesh, 6.0, 2.0, 0.0)


def test_a_shut_door_walls_off_its_opening() -> None:
    mesh = build_navmesh(_two_rooms(), shut={'between'})
    assert len(_at(mesh, 4.0, 2.0)) == 0
    assert _component(mesh, 2.0, 2.0, 0.0) != _component(mesh, 6.0, 2.0, 0.0)


def _step(height: float) -> list[NavLevel]:
    floors = [Floor(name='low', pos=Position(2.0, 2.0, 0.0), x_length=4.0, y_length=4.0), Floor(name='high', pos=Position(6.0, 2.0, height), x_length=4.0, y_length=4.0)]
    return [NavLevel(level='0', floors=floors, walls=[], stairs=[], doors=[])]


def test_a_step_taller_than_the_climb_splits_the_mesh() -> None:
    mesh = build_navmesh(_step(0.15), climb=0.1)
    assert mesh.components().max() == 1
    assert _component(mesh, 2.0, 2.0, 0.0) != _component(mesh, 6.0, 2.0, 0.15)


def test_a_step_within_the_climb_keeps_the_mesh_whole() -> None:
    mesh = build_navmesh(_step(0.15), climb=0.2)
    assert mesh.components().max() == 0
    assert _component(mesh, 2.0, 2.0, 0.0) == _component(mesh, 6.0, 2.0, 0.15)


def test_a_staircase_joins_the_levels_only_for_a_climb_of_its_riser() -> None:
    levels = _only(_stacked_demo(), 'stairs')
    low = build_navmesh(levels, climb=0.1)
    high = build_navmesh(levels, climb=0.2)
    assert _component(low, 15.0, 8.0, 0.0) != _component(low, 15.0, 8.0, 3.0)
    assert _component(high, 15.0, 8.0, 0.0) == _component(high, 15.0, 8.0, 3.0)


def _elevators() -> list[NavLevel]:
    world = WorldView(WORLDS / 'mechanisms_elevator').load()
    return navmesh_levels(world, {'0': (0.0, 0.0, 0.0), '1': (0.0, 0.0, 3.0)})


def test_elevator_cabins_are_walkable_on_every_level_they_serve() -> None:
    mesh = build_navmesh(_elevators())
    for x, y in ((3.0, 11.0), (7.0, 11.0), (11.0, 11.0), (17.0, 5.0)):
        assert _component(mesh, x, y, 0.0) == _component(mesh, 8.0, 5.0, 0.0)
    assert _component(mesh, 21.0, 5.0, 0.0) == _component(mesh, 26.0, 5.0, 0.0) != _component(mesh, 8.0, 5.0, 0.0)
    for x, y in ((3.0, 11.0), (7.0, 11.0), (11.0, 11.0), (13.0, -1.0)):
        assert _component(mesh, x, y, 3.0) == _component(mesh, 8.0, 5.0, 3.0)
    assert _component(mesh, 3.0, 11.0, 0.0) != _component(mesh, 3.0, 11.0, 3.0)


def test_elevator_cabins_open_only_on_their_door_side() -> None:
    mesh = build_navmesh(_elevators())
    assert len(_at(mesh, 3.0, 10.0, 0.0)) == 1
    for x, y in ((2.0, 11.0), (4.0, 11.0), (3.0, 12.0)):
        assert len(_at(mesh, x, y, 0.0)) == 0
    for x, y in ((16.0, 5.0), (22.0, 5.0)):
        assert len(_at(mesh, x, y, 0.0)) == 1
    for x, y in ((18.0, 5.0), (20.0, 5.0)):
        assert len(_at(mesh, x, y, 0.0)) == 0


def test_nearest_finds_the_closest_vertex_and_its_distance() -> None:
    mesh = build_navmesh(_two_rooms())
    index, distance = mesh.nearest(2.04, 2.0, 0.3)
    assert mesh.vertices[index] == pytest.approx((2.0, 2.0, 0.0))
    assert distance == pytest.approx(np.hypot(0.04, 0.3))
    assert NavMesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)).nearest(0.0, 0.0, 0.0) == (-1, np.inf)
