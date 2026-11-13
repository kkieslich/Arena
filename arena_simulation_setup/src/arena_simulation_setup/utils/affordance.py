"""Affordance slots of placed objects and the claims that keep overlapping ones from being occupied together."""

from __future__ import annotations

import enum
import math
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence

import attrs
import numpy as np

from arena_simulation_setup.utils.geometry import Facing, Pose, Position

Side = tuple[float, float]
Footprint = tuple[Side, Side]
Box = tuple[Side, Side, Side]


class AffordanceKind(enum.StrEnum):
    SIT = "sit"


def affordance_kind(kind: AffordanceKind | str) -> AffordanceKind:
    try:
        return AffordanceKind(kind)
    except ValueError:
        raise ValueError(f"unknown affordance {kind!r}, known: {', '.join(AffordanceKind)}") from None


def has_z(box: Box) -> bool:
    """Whether the box has a z range, an unbuilt slot's is zero-height."""
    return box[2][1] > box[2][0]


def slot_box(value: Iterable[Iterable[float]]) -> Box:
    """A slot bbox from [[x_min, x_max], [y_min, y_max]] with an optional [z_min, z_max], zero-height without one."""
    sides = [typing.cast(Side, tuple(float(bound) for bound in side)) for side in value]
    if len(sides) == 2:
        sides.append((0.0, 0.0))
    if len(sides) != 3 or any(len(side) != 2 for side in sides):
        raise ValueError(f"slot bbox must be [[x_min, x_max], [y_min, y_max]] or with a third [z_min, z_max], got {value}")
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = sides
    if not (x_max > x_min and y_max > y_min) or z_max < z_min:
        raise ValueError(f"slot bbox needs positive x and y extents and z_max >= z_min, got {value}")
    return ((x_min, x_max), (y_min, y_max), (z_min, z_max))


def _optional_box(value: Iterable[Iterable[float]] | None) -> Box | None:
    return None if value is None else slot_box(value)


def _optional_facing(value: Facing | Sequence[float | str] | float | str | None) -> Facing | None:
    return value if value is None or isinstance(value, Facing) else Facing.parse(value)


@attrs.define
class Slot:
    """Seat-surface bbox and allowed facings in the object frame, both absent on a slot the build places."""

    bbox: Box | None = attrs.field(default=None, converter=_optional_box)
    facing: Facing | None = attrs.field(default=None, converter=_optional_facing)

    def __attrs_post_init__(self) -> None:
        if self.bbox is not None and self.facing is None:
            raise ValueError(f"a slot with a bbox needs a yaw, got bbox {self.bbox}")

    @classmethod
    def parse(cls, value: Slot | Mapping[str, typing.Any]) -> Slot:
        """A slot from its annotation entry, {bbox, yaw} with either left out."""
        if isinstance(value, Slot):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(f"slot must be a mapping with bbox and yaw, got {value!r}")
        if unknown := sorted(set(value) - {"bbox", "yaw"}):
            raise ValueError(f"slot keys are bbox and yaw, got unknown {', '.join(unknown)}")
        return cls(bbox=value.get("bbox"), facing=value.get("yaw"))

    def unstructure(self) -> dict[str, object]:
        result: dict[str, object] = {}
        if self.bbox is not None:
            result["bbox"] = [list(side) for side in (self.bbox if has_z(self.bbox) else self.bbox[:2])]
        if self.facing is not None:
            result["yaw"] = self.facing.serialize()
        return result

    def placed(self, kind: AffordanceKind | str, index: int) -> AffordanceSlot:
        """This slot as slot index of its kind, refused while the build has not placed it."""
        if self.bbox is None or self.facing is None:
            raise ValueError(f"{kind} slot {index} is not placed yet, build the asset")
        return AffordanceSlot(kind=affordance_kind(kind), index=index, bbox=self.bbox, facing=self.facing)


Affordances = dict[AffordanceKind, list[Slot]]


def structure_affordances(value: Affordances | Mapping[str, Iterable[typing.Any]] | None) -> Affordances:
    """The affordances block of an annotation, at most one slot per kind left to the build."""
    affordances: Affordances = {affordance_kind(kind): [Slot.parse(slot) for slot in slots or ()] for kind, slots in (value or {}).items()}
    for kind, slots in affordances.items():
        if sum(slot.bbox is None for slot in slots) > 1:
            raise ValueError(f"at most one {kind} slot may leave its bbox to the build, give the others a bbox")
    return affordances


def unstructure_affordances(affordances: Affordances) -> dict[str, list[dict[str, object]]]:
    return {str(kind): [slot.unstructure() for slot in slots] for kind, slots in affordances.items()}


@attrs.frozen
class AffordanceSlot:
    """A placed slot of one affordance kind, its bbox being the box that claims compare."""

    kind: AffordanceKind
    index: int
    bbox: Box
    facing: Facing

    @property
    def center(self) -> tuple[float, float]:
        (x_min, x_max), (y_min, y_max), _ = self.bbox
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)

    def pose(self, rng: np.random.Generator) -> Pose:
        """Object-local pose at the box center facing one allowed yaw drawn from rng."""
        return Pose(Position(*self.center), self.facing.sample(rng))

    def overlaps(self, other: AffordanceSlot) -> bool:
        """Boxes overlap with positive volume, a box without a z range overlaps through all heights."""
        sides = 3 if has_z(self.bbox) and has_z(other.bbox) else 2
        return all(low < other_high and other_low < high for (low, high), (other_low, other_high) in zip(self.bbox[:sides], other.bbox[:sides], strict=True))

    def distance(self, xy: tuple[float, float]) -> float:
        """Distance in the object frame from a point to the box footprint, 0 inside."""
        (x_min, x_max), (y_min, y_max), _ = self.bbox
        x, y = xy
        return math.hypot(max(x_min - x, 0.0, x - x_max), max(y_min - y, 0.0, y - y_max))

    def rank(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Sort key toward an object-frame point: footprint distance, ties going to the nearer box center."""
        return (self.distance(xy), math.dist(self.center, xy))


def placed_slots(affordances: Affordances | Mapping[str, Iterable[typing.Any]]) -> list[AffordanceSlot]:
    """Placed slots of an annotation's affordances block, every kind."""
    return [slot.placed(kind, index) for kind, slots in structure_affordances(affordances).items() for index, slot in enumerate(slots) if slot.bbox is not None]


SEAT_SLOT_REACH = 0.15  # m, a scenario seat this close to a sit slot's footprint takes that slot's height


def seat_slot(seat: Pose, slots: Sequence[AffordanceSlot]) -> AffordanceSlot | None:
    """The sit slot with a z range a scenario seat lies in or next to (within SEAT_SLOT_REACH), the nearest center among overlapping ones."""
    xy = (seat.position.x, seat.position.y)
    near = [slot for slot in slots if has_z(slot.bbox) and slot.distance(xy) <= SEAT_SLOT_REACH]
    return min(near, key=lambda slot: (slot.distance(xy), math.dist(slot.center, xy))) if near else None


def box_reach(seat: Pose, box: Sequence[Sequence[float]], behind: bool = False) -> float:
    """How far ahead of a seat along its facing, or behind it, an object-frame box ends."""
    yaw = seat.orientation.to_yaw() + (math.pi if behind else 0.0)
    reach = math.inf
    for (low, high), at, step in zip(box[:2], (seat.position.x, seat.position.y), (math.cos(yaw), math.sin(yaw)), strict=True):
        if step > 1e-9:
            reach = min(reach, (high - at) / step)
        elif step < -1e-9:
            reach = min(reach, (low - at) / step)
    return max(0.0, reach)


def seat_reach(seat: Pose, slot: AffordanceSlot, behind: bool = False) -> float:
    """How far ahead of a seat along its facing, or behind it, its slot's box ends, in the object frame."""
    return box_reach(seat, slot.bbox, behind)


def seat_approach(seat: Pose, front: float | None, bounding_box: Sequence[Sequence[float]] | None, stand: Callable[[float], float]) -> float:
    """Where a sitter stands ahead of a seat before it sits, `stand` of its slot's front edge `front` ahead, never inside the object's bounding box, 0 for a seat outside every slot."""
    if front is None:
        return 0.0
    return max(stand(front), 0.0 if bounding_box is None else box_reach(seat, bounding_box))


def max_occupancy(slots: Sequence[AffordanceSlot]) -> int:
    """Most of the slots that can be occupied at once, no two of them overlapping."""
    if not slots:
        return 0
    first, rest = slots[0], slots[1:]
    return max(max_occupancy(rest), 1 + max_occupancy([slot for slot in rest if not slot.overlaps(first)]))


def conflicts(slots: Sequence[AffordanceSlot]) -> list[list[int]]:
    """Per slot, the indices of the other slots it overlaps."""
    return [[j for j, other in enumerate(slots) if j != i and slot.overlaps(other)] for i, slot in enumerate(slots)]


class SlotOccupied(ValueError):
    pass


@attrs.define
class AffordanceClaims:
    """Occupied slots of one placed object, no two of them overlapping."""

    slots: list[AffordanceSlot]
    _active: dict[tuple[AffordanceKind, int], AffordanceSlot] = attrs.field(factory=dict, init=False)

    @classmethod
    def parse(cls, affordances: Affordances | Mapping[str, Iterable[typing.Any]]) -> AffordanceClaims:
        """Claims over the placed slots of an annotation's affordances block."""
        return cls(placed_slots(affordances))

    def occupied(self) -> list[AffordanceSlot]:
        return list(self._active.values())

    def blocking(self, slot: AffordanceSlot) -> AffordanceSlot | None:
        return next((active for active in self._active.values() if active.overlaps(slot)), None)

    def available(self, kind: AffordanceKind | str | None = None) -> list[AffordanceSlot]:
        """Slots overlapping no occupied slot, of every kind when kind is None."""
        wanted = None if kind is None else affordance_kind(kind)
        return [slot for slot in self.slots if wanted in (None, slot.kind) and self.blocking(slot) is None]

    def closest(self, xy: tuple[float, float], kind: AffordanceKind | str | None = None) -> AffordanceSlot | None:
        """Available slot nearest to an object-frame point, ties going to the nearer box center."""
        return min(self.available(kind), key=lambda slot: slot.rank(xy), default=None)

    def claim(self, slot: AffordanceSlot) -> AffordanceSlot:
        if (blocker := self.blocking(slot)) is not None:
            raise SlotOccupied(f"{slot.kind} slot {slot.index} overlaps the occupied {blocker.kind} slot {blocker.index}")
        self._active[(slot.kind, slot.index)] = slot
        return slot

    def release(self, slot: AffordanceSlot) -> None:
        if self._active.pop((slot.kind, slot.index), None) is None:
            raise KeyError(f"{slot.kind} slot {slot.index} is not claimed")
