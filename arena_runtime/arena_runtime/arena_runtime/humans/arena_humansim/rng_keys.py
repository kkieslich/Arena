"""Keys of the adapter's episode RNG streams, built from names that are the same in every env and spawn order."""

from __future__ import annotations

import typing

ROOT = "humansim"


class AgentSource(typing.NamedTuple):
    """What the engine reports about where a flow agent came from: its source, release index and name without the env prefix."""

    source_name: str = ""
    source_seq: int = 0
    name: str = ""


def walker_key(name: str) -> tuple[str, ...]:
    """Stream of a scenario pedestrian's sampled agent type, `name` its sim path without the env prefix."""
    return (ROOT, "walker", name)


def seats_key(name: str) -> tuple[str, ...]:
    """Stream of an object's seat poses drawn from its sit slots, `name` its sim path without the env prefix."""
    return (ROOT, "seats", name)


def flow_model_key(agent_id: int, source: AgentSource) -> tuple[str | int, ...]:
    """Stream of a flow agent's human model: its source and release index, else its engine name, else its engine id."""
    if source.source_name:
        return (ROOT, "flow-model", "source", source.source_name, int(source.source_seq))
    if source.name:
        return (ROOT, "flow-model", "name", source.name)
    return (ROOT, "flow-model", "id", int(agent_id))


__all__ = ["AgentSource", "flow_model_key", "seats_key", "walker_key"]
