"""The env's side executor: its own thread, unaffected by a host executor held in one callback, and a beat that tells a wedged thread."""

from __future__ import annotations

import threading
import time
import typing

NS = "/side_executor_probe"


def _side() -> typing.Any:
    from arena_runtime.env.side_executor import SideExecutor

    return SideExecutor(NS, "env_side")


def _wait(pred: typing.Callable[[], bool], what: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def test_nodes_added_run_on_the_side_thread(rclpy_context: object) -> None:
    import rclpy

    side = _side()
    node = rclpy.create_node("tracker", namespace=NS)
    threads: list[str] = []
    try:
        node.create_timer(0.02, lambda: threads.append(threading.current_thread().name))
        side.add_node(node)
        assert side.node_names() == {f"{NS}/tracker"}
        _wait(lambda: len(threads) >= 3, "the timer to fire")
        assert set(threads) == {"env_side"}
        side.remove_node(node)
        assert side.node_names() == set()
        seen = len(threads)
        time.sleep(0.2)
        assert len(threads) <= seen + 1
    finally:
        side.shutdown()
        node.destroy_node()


def test_a_held_host_executor_does_not_hold_the_side_thread(rclpy_context: object) -> None:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor

    side = _side()
    host, tracker = rclpy.create_node("host", namespace=NS), rclpy.create_node("tracker", namespace=NS)
    host_executor = SingleThreadedExecutor()
    host_executor.add_node(host)
    ticks: list[float] = []
    try:
        host.create_timer(0.01, lambda: time.sleep(2.0))
        tracker.create_timer(0.1, lambda: ticks.append(time.monotonic()))
        side.add_node(tracker)
        threading.Thread(target=host_executor.spin, daemon=True).start()
        time.sleep(1.5)
        assert len(ticks) >= 12, f"{len(ticks)} ticks in 1.5 s"
        assert max(b - a for a, b in zip(ticks, ticks[1:], strict=False)) < 0.3
    finally:
        host_executor.shutdown()
        side.shutdown()
        host.destroy_node()
        tracker.destroy_node()


def test_the_beat_ages_only_while_the_side_thread_is_wedged(rclpy_context: object) -> None:
    import rclpy

    from arena_runtime.env.side_executor import BEAT_S

    side = _side()
    node = rclpy.create_node("wedge", namespace=NS)
    release = threading.Event()
    try:
        _wait(lambda: side.beat_age() < BEAT_S, "a first beat")
        time.sleep(3 * BEAT_S)
        assert side.beat_age() < 2 * BEAT_S
        timer = node.create_timer(0.01, lambda: (node.destroy_timer(timer), release.wait(10.0)))
        side.add_node(node)
        time.sleep(4 * BEAT_S)
        assert side.beat_age() > 2 * BEAT_S
        release.set()
        _wait(lambda: side.beat_age() < BEAT_S, "the beat after the release")
    finally:
        release.set()
        side.shutdown()
        node.destroy_node()


def test_shutdown_stops_the_thread(rclpy_context: object) -> None:
    side = _side()
    side.shutdown()
    assert not side._thread.is_alive()
    side.shutdown()
