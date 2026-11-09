"""Docked objects the FLOWEDIT tests place against a wall, with their docks in the object frame."""

from __future__ import annotations

import typing


class Docked(typing.NamedTuple):
    name: str
    model: str
    back: float
    dock: list[list[float]]


#: Turned so their front (`face: +x`) faces into the room, origin `back` m off the wall
FACING_NORTH = 1.5708

#: The charger's and the bay's origin is their middle, the parcel locker's a back corner. The charger and the locker have
#: a 1.2 x 1.0 m jackal dock in front; the waiting bay is a floor marking with no collider and no bounding box, one slot
#: per object, its dock the slot around its origin.
CHARGER = Docked("charger_1", "Common/Object/robot_charger", 0.26, [[0.25, -0.5], [1.45, -0.5], [1.45, 0.5], [0.25, 0.5]])
LOCKER = Docked("locker_1", "Common/Object/parcel_locker", 0.06, [[0.55, 0.02], [1.75, 0.02], [1.75, 1.02], [0.55, 1.02]])
BAY = Docked("bay_1", "Common/Object/waiting_bay", 0.65, [[-0.6, -0.5], [0.6, -0.5], [0.6, 0.5], [-0.6, 0.5]])
DOCKED = [CHARGER, LOCKER, BAY]
DOCKED_IDS = [d.model.rsplit("/", 1)[-1] for d in DOCKED]

#: The fleet study's charger power, which a robot in the charger's dock charges with
CHARGE_POWER_W = 125.0
