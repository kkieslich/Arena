"""Hung node detection for one env: a tally of missed pings and the round that feeds it."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import attrs
from arena_rclpy_mixins.Async import ClientWrapper
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from rcl_interfaces.srv import GetParameters

if TYPE_CHECKING:
    from arena_rclpy_mixins.Async import AsyncNode

    from arena_runtime.env.side_executor import SideExecutor

_PARAM_SERVICE = "/get_parameters"
_STATE_SERVICE = "/get_state"
_HELPER = re.compile(r"(_rclcpp_node|/moveit(_\d+)?)$")


@attrs.define
class MissTally:
    """Pings each node has missed in a row, tracked once it has answered at least once."""

    limit: int
    _misses: dict[str, int] = attrs.field(factory=dict, init=False)

    def tracked(self) -> set[str]:
        return set(self._misses)

    def forget_absent(self, present: set[str]) -> None:
        self._misses = {name: count for name, count in self._misses.items() if name in present}

    def forget(self, names: set[str]) -> None:
        for name in names:
            self._misses.pop(name, None)

    def record(self, answered: set[str]) -> str | None:
        """Count one round and return a node that reached the limit, a round most nodes missed counts for nobody."""
        known = self._misses.keys() | answered
        if 2 * len(answered) < len(known):
            return None
        hung = None
        for name in known:
            self._misses[name] = 0 if name in answered else self._misses[name] + 1
            if self._misses[name] >= self.limit:
                hung = name
                self._misses[name] = 0
        return hung

    def miss(self, missed: set[str]) -> str | None:
        """Count a round that could judge only `missed` (each missed it): the other tracked nodes keep their count."""
        hung = None
        for name in missed & self._misses.keys():
            self._misses[name] += 1
            if self._misses[name] >= self.limit:
                hung = name
                self._misses[name] = 0
        return hung


class NodePinger:
    """Pings every node under ``root`` in other processes: lifecycle nodes must report active, any other node must answer on its parameter service.

    The nodes of this process are not pinged: a ping to a node the pinging node's executor spins needs that executor
    twice (to serve the request, then to take the reply) and so measures its queue, not a hang, and a wedged one stops
    the rounds altogether, which the watchdog's own timer starts. With a ``side`` executor the pings' replies are taken on
    its thread, so a lagging env executor delays no reply, and the nodes the side executor spins (the collision
    trackers) count as answered while its thread beats. A side thread that missed its beats for a whole timeout is
    wedged: no reply could be taken on it, so nobody is pinged and only its own nodes miss.
    """

    def __init__(self, node: AsyncNode, root: str, limit: int, *, side: SideExecutor | None = None) -> None:
        self._node = node
        self._root = root
        self._side = side
        self._tally = MissTally(limit)
        self._clients: dict[str, ClientWrapper] = {}
        # the service each node was last pinged on: a lifecycle node's get_parameters can show up in the graph before its
        # get_state, and an answer on the first must not make it a tracked plain node once the second shows up
        self._pinged_on: dict[str, str] = {}

    def _executor_nodes(self) -> set[str]:
        """The pinging node and every node its executor spins."""
        own = {self._node.get_fully_qualified_name()}
        executor = self._node.executor
        if executor is not None:
            for node in executor.get_nodes():
                name = getattr(node, "get_fully_qualified_name", None)
                if name is not None:
                    own.add(name())
        return own

    def _side_nodes(self) -> set[str]:
        """The nodes under the root the side executor spins."""
        return {name for name in self._side.node_names() if name.startswith(f"{self._root}/")} if self._side is not None else set()

    def _targets(self) -> dict[str, str]:
        """Each node under the root outside this process, and the service to ping it on."""
        services = {name for name, _ in self._node.get_service_names_and_types() if name.startswith(f"{self._root}/")}
        nodes = {name.removesuffix(_PARAM_SERVICE) for name in services if name.endswith(_PARAM_SERVICE)} - self._executor_nodes() - self._side_nodes()
        nodes = {name for name in nodes if not _HELPER.search(name)}
        return {name: name + _STATE_SERVICE if name + _STATE_SERVICE in services else name + _PARAM_SERVICE for name in nodes}

    def _client(self, service: str, timeout: float) -> ClientWrapper:
        """A client on the side executor's node when there is one, its replies taken on the side thread, else on the pinging node."""
        srv_type = GetState if service.endswith(_STATE_SERVICE) else GetParameters
        if self._side is None:
            return self._node.create_client_wrapper(srv_type, service, timeout=timeout)
        return ClientWrapper(self._node, self._side.node.create_client(srv_type, service), timeout=timeout)

    def _destroy(self, client: ClientWrapper) -> None:
        (self._side.node if self._side is not None else self._node).destroy_client(client.client)

    async def _alive(self, service: str, timeout: float) -> bool:
        if service.endswith(_STATE_SERVICE):
            reply = await self._clients[service].call_timeout(GetState.Request(), timeout)
            return reply is not None and reply.current_state.id == State.PRIMARY_STATE_ACTIVE
        return await self._clients[service].call_timeout(GetParameters.Request(), timeout) is not None

    def _sync_clients(self, targets: dict[str, str], timeout: float) -> None:
        for service in self._clients.keys() - targets.values():
            self._destroy(self._clients.pop(service))
        for service in targets.values() - self._clients.keys():
            self._clients[service] = self._client(service, timeout)

    def prepare(self, timeout: float) -> int:
        """Make the clients for every node there is to ping now, and return how many it made. The env calls it at the end
        of each reset: the first round of an episode made them all, about 40 rclpy clients of a 4-robot fleet, holding the
        GIL for seconds right after EPISODE STARTED."""
        before = len(self._clients)
        self._sync_clients(self._targets(), timeout)
        return len(self._clients) - before

    async def round(self, timeout: float, *, new_episode: bool) -> str | None:
        """Ping once and return a node that stopped answering or left its active state, or None."""
        targets = self._targets()
        side = self._side_nodes()
        self._tally.forget({name for name, service in targets.items() if self._pinged_on.get(name, service) != service})
        self._pinged_on = dict(targets)
        if new_episode:
            self._tally.forget_absent(set(targets) | side)
        if self._side is not None and self._side.beat_age() >= timeout:
            # no reply could be taken on the wedged side thread: its own nodes miss, the others are not judged (counting them
            # as answered would clear a frozen node's misses whenever a loaded machine delays two beats)
            return self._tally.miss(side)
        self._sync_clients(targets, timeout)
        names = sorted(targets)
        alive = await asyncio.gather(*(self._alive(targets[name], timeout) for name in names))
        answered = {name for name, up in zip(names, alive, strict=True) if up}
        if self._side is not None and self._side.beat_age() < timeout:
            answered |= side
        return self._tally.record(answered)
