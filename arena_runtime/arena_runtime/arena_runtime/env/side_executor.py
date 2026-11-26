"""The env's side executor: a second executor in its own thread for the work that must not wait in the env executor's queue."""

from __future__ import annotations

import contextlib
import threading
import time

import rclpy
import rclpy.clock
import rclpy.executors
import rclpy.node
from arena_rclpy_mixins.spin import create_executor
from rclpy.exceptions import InvalidHandle
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy

#: wall seconds between two beats of the side thread, the grain of `beat_age`
BEAT_S = 0.5


class SideExecutor:
    """An EventsExecutor spun in a thread of its own, beside the env executor.

    The env executor takes every message the env subscribes to (TF, the clock of each node, the crowd frames, odometry)
    and runs the env's timers, one callback after another. When it falls behind (two or more robots driving through a
    dense crowd on a loaded machine), everything queued on it waits: a 10 Hz collision tracker tick or the reply to a
    node-watch ping can wait for seconds. Nodes added here have their own queue and thread; they share only the
    interpreter lock with the rest of the env, which a thread waiting for it gets within a switch interval.

    `node` is a plain node on this executor, without parameter services or rosout, for clients whose replies must not
    queue behind the env executor (the node watch's pings). `beat_age` tells how long ago the thread last ran a callback
    of its own wall-clock beat, so a wedged side thread can be told from a busy one.
    """

    def __init__(self, namespace: str, name: str) -> None:
        self._executor: rclpy.executors.Executor = create_executor()
        self.node = rclpy.node.Node(name, namespace=namespace, use_global_arguments=False, start_parameter_services=False, enable_rosout=False)
        self._beat_at = time.monotonic()
        self._beat = self.node.create_timer(BEAT_S, self._on_beat, clock=rclpy.clock.Clock(clock_type=rclpy.clock.ClockType.STEADY_TIME))
        self._executor.add_node(self.node)
        self._lock = threading.Lock()
        self._nodes: dict[str, rclpy.node.Node] = {}
        self._closed = False
        self._thread = threading.Thread(target=self._spin, name=name, daemon=True)
        self._thread.start()

    def _on_beat(self) -> None:
        self._beat_at = time.monotonic()

    def _spin(self) -> None:
        while rclpy.ok(context=self.node.context) and not self._closed:
            try:
                self._executor.spin()
                return
            except rclpy.executors.ExternalShutdownException:
                return
            except InvalidHandle:
                continue  # a node removed while one of its callbacks was due
            except _rclpy.RCLError:
                if rclpy.ok(context=self.node.context) and not self._closed:
                    raise
                return

    def beat_age(self) -> float:
        """Wall seconds since the side thread last ran its beat, at most about `BEAT_S` while it is not wedged."""
        return time.monotonic() - self._beat_at

    def add_node(self, node: rclpy.node.Node) -> None:
        with self._lock:
            self._nodes[node.get_fully_qualified_name()] = node
        self._executor.add_node(node)

    def remove_node(self, node: rclpy.node.Node) -> None:
        with self._lock:
            self._nodes.pop(node.get_fully_qualified_name(), None)
        with contextlib.suppress(InvalidHandle):
            self._executor.remove_node(node)

    def node_names(self) -> set[str]:
        """Fully qualified names of the nodes added here (not `node` itself)."""
        with self._lock:
            return set(self._nodes)

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop the thread and destroy `node`; the added nodes stay with their owners."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._executor.shutdown(timeout_sec=timeout)
        if self._thread is not threading.current_thread():
            self._thread.join(timeout)
        with contextlib.suppress(Exception):
            self.node.destroy_node()
