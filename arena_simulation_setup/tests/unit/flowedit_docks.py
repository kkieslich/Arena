"""Docked objects the FLOWEDIT tests place against a wall, with their docks in the object frame."""

from __future__ import annotations

import typing

import yaml

if typing.TYPE_CHECKING:
    from pathlib import Path


class Docked(typing.NamedTuple):
    name: str
    model: str
    back: float
    dock: list[list[float]]


#: The bounding boxes of the docked models' annotations in the public bucket, to the cm, by model; the waiting bay, a
#: floor marking, has none. All three face +x.
BOXES = {"Common/Object/robot_charger": [[-0.2, 0.2], [-0.39, 0.39], [0.0, 1.61]], "Common/Object/parcel_locker": [[0.0, 0.51], [0.0, 1.04], [0.0, 2.0]]}


def write_docked_assets(root: Path) -> dict[str, Path]:
    """The docked models as annotation-only assets under the asset root `root`, so their footprints and fronts resolve
    as a docked object's must (`resolve_footprints`); their directories by model."""
    from arena_simulation_setup.shared.entities import Obstacle

    out: dict[str, Path] = {}
    for docked in DOCKED:
        directory = root / Obstacle.parse({"name": docked.name, "model": docked.model, "pose": [0.0, 0.0, 0.0]}).model.relpath()
        directory.mkdir(parents=True, exist_ok=True)
        box = BOXES.get(docked.model)
        (directory / "annotation.yaml").write_text(yaml.safe_dump({"face": "+x", **({"bounding_box": box} if box is not None else {})}))
        out[docked.model] = directory
    return out


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
