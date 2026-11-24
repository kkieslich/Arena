"""Which robot each charger charges: the first one parked in its dock, until it moves or leaves."""

from __future__ import annotations

import typing

import attrs
from arena_simulation_setup.shared.conditions import MembershipAtom
from arena_simulation_setup.shared.judge import ParkTimer, atom_holds

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

    from arena_simulation_setup.shared.judge import Sample, Zones

PARK_TIME = 1.0  # sim seconds within the judge's park drift and turn before a robot in a dock counts as parked


@attrs.define
class Chargers:
    """An episode's chargers with their power in W by name, each charging one robot parked in its dock at a time."""

    power: Mapping[str, float] = attrs.field(factory=dict)
    _parks: dict[str, ParkTimer] = attrs.field(factory=dict, init=False)
    _occupants: dict[str, str] = attrs.field(factory=dict, init=False)

    def step(self, sample: Sample, zones: Zones) -> dict[str, float]:
        """Charge power in W of every robot on this sample, 0 for a robot that does not charge."""
        for robot in self._parks.keys() - sample.robots.keys():
            del self._parks[robot]
        parked = {robot for robot, (x, y, yaw) in sample.robots.items() if self._parks.setdefault(robot, ParkTimer()).held_for(x, y, yaw, sample.t) >= PARK_TIME}

        def docked(robot: str, charger: str) -> bool:
            return robot in parked and atom_holds(MembershipAtom(subject='robot', zone=charger), sample, zones, robot) is True

        for charger in self.power:
            occupant = self._occupants.get(charger)
            if occupant is not None and not docked(occupant, charger):
                del self._occupants[charger]
            if charger not in self._occupants:
                taken = set(self._occupants.values())
                arrival = next((robot for robot in sample.robots if robot not in taken and docked(robot, charger)), None)
                if arrival is not None:
                    self._occupants[charger] = arrival
        charging = {robot: self.power[charger] for charger, robot in self._occupants.items()}
        return {robot: charging.get(robot, 0.0) for robot in sample.robots}
