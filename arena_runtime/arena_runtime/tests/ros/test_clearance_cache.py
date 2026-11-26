"""The floor's cached clearance map gives the same available cells as a distance transform per call, forbidden zones
included, and it follows every occupancy change."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

try:
    from arena_runtime.env.world_manager import utils
    from arena_runtime.env.world_manager.utils import WorldLayers, WorldOccupancy
    from arena_runtime.env.world_manager.world_manager import _occupancy_to_available, _outside_rects
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")


def _random_grid(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), WorldOccupancy.EMPTY, dtype=np.uint8)
    for _ in range(int(rng.integers(0, 12))):
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        grid[r : r + int(rng.integers(1, 8)), c : c + int(rng.integers(1, 8))] = WorldOccupancy.FULL
    return grid


def _random_rect(rng: np.random.Generator, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """A forbidden-zone rect as tf_posr2rect gives it, often touching or crossing the grid border."""
    r, c = int(rng.integers(-6, h + 6)), int(rng.integers(-6, w + 6))
    dr, dc = int(rng.integers(0, 9)), int(rng.integers(0, 9))
    return (r - dr, c - dc), (r + dr, c + dc)


def test_cached_cells_with_forbidden_rects_equal_a_distance_transform_per_call() -> None:
    rng = np.random.default_rng(7)
    for _ in range(300):
        h, w = int(rng.integers(5, 60)), int(rng.integers(5, 60))
        grid = _random_grid(rng, h, w)
        layers = WorldLayers(WorldOccupancy(grid.copy()))
        rects = [_random_rect(rng, h, w) for _ in range(int(rng.integers(0, 5)))]
        safe = float(rng.choice([0.0, 0.5, 1.0, 1.5, 2.0, 3.7, 6.0]))
        fork = layers.fork()
        for rect in rects:
            fork.occupy(*rect)
        reference = _occupancy_to_available(fork.grid, safe)
        cached = layers.available(safe)
        got = _outside_rects(cached, rects, safe, layers.shape) if rects and len(cached) else cached
        assert np.array_equal(got, reference), (h, w, rects, safe)


def test_every_occupancy_writer_moves_the_version_and_the_cells_follow() -> None:
    layers = WorldLayers(WorldOccupancy(np.full((30, 30), WorldOccupancy.EMPTY, dtype=np.uint8)))
    seen = [layers.version]

    def check() -> None:
        assert layers.version > seen[-1]
        seen.append(layers.version)
        assert np.array_equal(layers.available(2.0), _occupancy_to_available(layers.grid, 2.0))

    layers.available(2.0)
    mask = np.zeros((30, 30), dtype=bool)
    mask[10:14, 10:14] = True
    layers.obstacle_occupy(mask)
    check()
    layers.forbidden_occupy((2, 2), (6, 9))
    check()
    fork = layers.fork()
    fork.occupy((20, 20), (25, 28))
    fork.commit()
    check()
    layers.forbidden_clear()
    check()
    layers.obstacle_clear()
    check()


def test_two_threads_on_a_cold_cache_compute_the_clearance_once(monkeypatch: pytest.MonkeyPatch) -> None:
    layers = WorldLayers(WorldOccupancy(np.full((40, 40), WorldOccupancy.EMPTY, dtype=np.uint8)))
    computed: list[int] = []
    real = utils.clearance_of

    def slow(grid: np.ndarray) -> np.ndarray:
        computed.append(1)
        time.sleep(0.02)
        return real(grid)

    monkeypatch.setattr(utils, "clearance_of", slow)
    start = threading.Barrier(2)
    got: list[np.ndarray] = []

    def call() -> None:
        start.wait()
        got.append(layers.clearance())

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert computed == [1] and got[0] is got[1]


def test_the_cache_is_bounded_and_read_only() -> None:
    layers = WorldLayers(WorldOccupancy(np.full((20, 20), WorldOccupancy.EMPTY, dtype=np.uint8)))
    for d in range(20):
        layers.available(float(d) / 4)
    assert len(layers._available) <= utils._AVAILABLE_CACHE_SIZE
    with pytest.raises(ValueError):
        layers.available(1.0)[0, 0] = 5
    with pytest.raises(ValueError):
        layers.clearance()[0, 0] = 5


def test_a_writer_between_the_clearance_and_the_cells_leaves_no_stale_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """An occupancy change while the clearance map is computed (the reset's worker thread, a writer on the event loop) must
    not get the old map's cells cached under the new version."""
    layers = WorldLayers(WorldOccupancy(np.full((40, 40), WorldOccupancy.EMPTY, dtype=np.uint8)))
    real = utils.clearance_of
    bumped = [False]

    def computing_while_a_writer_occupies(grid: np.ndarray) -> np.ndarray:
        clearance = real(grid)
        if not bumped[0]:
            bumped[0] = True
            layers.forbidden_occupy((15, 15), (25, 25))  # the writer, after the map was computed from the old grid
        return clearance

    monkeypatch.setattr(utils, "clearance_of", computing_while_a_writer_occupies)
    cells = layers.available(2.0)
    assert bumped[0]
    assert np.array_equal(cells, _occupancy_to_available(layers.grid, 2.0))
    assert layers._available_version == layers.version
    assert np.array_equal(layers.available(2.0), _occupancy_to_available(layers.grid, 2.0))
