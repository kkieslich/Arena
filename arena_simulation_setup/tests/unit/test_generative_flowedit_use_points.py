"""The FLOWEDIT generator against the runtime's use-point check: refused edits, turns at a spot, copied formations, left-behind demand."""

from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from arena_simulation_setup.generative.flowedit.edit import EditType, Site, apply, open_site
from arena_simulation_setup.generative.flowedit.generate import TURNS, generate, refusal, site_findings, spots_for
from arena_simulation_setup.generative.flowedit.read import read_graph
from arena_simulation_setup.metrics.use_points import object_points
from arena_simulation_setup.shared import Obstacle

pytest.importorskip("scipy")
pytestmark = pytest.mark.usefixtures("local_root")

HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"
#: the annotations of the public bucket, to the cm
COUNTER = {"face": "-y", "bounding_box": [[-0.87, 0.87], [-0.68, 0.68], [0.0, 1.1]]}
DISPENSER = {"face": "+x", "bounding_box": [[-0.2, 0.2], [-0.17, 0.17], [0.0, 1.49]]}
DESK = {"bounding_box": [[-0.5, 0.5], [-0.4, 0.4], [0.0, 0.8]]}
LINE = {"type": "line", "params": {"base_step": 1.0, "front_offset": 0.8}}


def _asset(root: Path, model: str, annotation: dict[str, Any]) -> None:
    directory = root / Obstacle.parse({"name": "probe", "model": model, "pose": [0.0, 0.0, 0.0]}).model.relpath()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "annotation.yaml").write_text(yaml.safe_dump(annotation))


def test_a_move_whose_use_point_lands_in_the_reception_counter_is_refused(tmp_path: Path, local_root: Path) -> None:
    # hospital_1 move_fountain 746a7931 of FLOWEDIT data v2: the fountain faces into the reception, but 0.18 m into the counter
    _asset(local_root, "Hospital/Reception_Counter", COUNTER)
    _asset(local_root, "Hospital/SM_WaterDispenser_01a", DISPENSER)
    root = tmp_path / "hospital_1"
    shutil.copytree(HOSPITAL, root, ignore=lambda directory, names: [n for n in names if Path(directory).name == "scenarios" and n != "queue"])
    site = open_site(root, "queue")
    edited = apply(site, EditType.MOVE, "obj:fountain", {"x": 2.06, "y": 0.26, "yaw": 1.5708})
    why = refusal(site, edited, {"fountain"})
    assert why is not None and why.startswith("fountain: use point at (2.06, 1.06)") and "inside the box of reception_desk" in why
    assert refusal(site, apply(site, EditType.MOVE, "obj:fountain", {"x": 11.0, "y": 10.0, "yaw": 1.57}), {"fountain"}) is None


def _site(tmp_path: Path, walls: list[list[list[float]]] = (), objects: list[dict[str, Any]] = (), *, east: float = 12.0, fountain: tuple[float, float, float] = (11.75, 6.0, math.pi)) -> Site:
    """A 6 x 8 m room west of a hall up to `east` joined by a door, a fountain (by default on the hall's east wall facing west)."""
    root = tmp_path / "w"
    for name, annotation in (("fountain", DISPENSER), ("desk", DESK)):
        _asset(root / "assets", f"Common/{name}", annotation)
    level = {
        "id": "0",
        "zones": [{"name": "room", "polygon": [[0, 0], [6, 0], [6, 8], [0, 8]]}, {"name": "hall", "polygon": [[6, 0], [east, 0], [east, 8], [6, 8]]}],
        "walls": [{"path": [[0, 0], [east, 0], [east, 8], [0, 8]], "closed": True}, {"path": [[6, 0], [6, 8]]}, *({"path": path} for path in walls)],
        "doors": [{"name": "door", "span": [[6, 1.0], [6, 2.5]]}],
        "objects": list(objects),
    }
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [level]}))
    fountain = {"name": "fountain", "model": "Common/fountain", "pose": list(fountain), "humansim": {"type": "fountain", "capacity": 1, "satisfies": {"thirst": 100.0}, "formation": LINE}}
    (root / "scenarios" / "demand").mkdir(parents=True)
    (root / "scenarios" / "demand" / "scenario.yaml").write_text(yaml.safe_dump({"format": "arena-scenario/2", "objects": [fountain]}))
    return open_site(root, "demand")


def _place(site: Site, around: tuple[float, float, float]) -> tuple[float, float, float] | None:
    spots = spots_for(site, "0")
    zone = next(z for z in spots.level.zones if z.name == "room")

    def fits(x: float, y: float, yaw: float) -> bool:
        return refusal(site, apply(site, EditType.MOVE, "obj:fountain", {"x": x, "y": y, "yaw": yaw}), {"fountain"}) is None

    return spots.against_wall(random.Random(0), "Common/fountain", zone, lambda _x, _y: True, around=around, fits=fits)


def test_a_spot_whose_front_is_blocked_turns_the_fountain_along_the_wall(tmp_path: Path) -> None:
    # a desk 0.6 m off the west wall takes the use point of a fountain facing into the room, not that of one facing along the wall
    site = _site(tmp_path, objects=[{"name": "desk", "model": "Common/desk", "pose": [1.5, 4.0, 0.0]}])
    spot = _place(site, (0.0, 4.0, 0.2))
    assert spot is not None and TURNS[1] == math.pi / 2
    x, y, yaw = spot
    assert yaw == pytest.approx(math.pi / 2, abs=1e-3) and abs(y - 4.0) <= 0.2 and x == pytest.approx(0.23, abs=0.02)


def test_a_spot_blocked_in_every_turn_is_given_up_for_another(tmp_path: Path) -> None:
    # the same desk closes a 1.2 m niche of the west wall: no turn of the fountain in it passes, a spot outside does
    niche = [[[0.0, 3.4], [1.0, 3.4]], [[0.0, 4.6], [1.0, 4.6]]]
    site = _site(tmp_path, walls=niche, objects=[{"name": "desk", "model": "Common/desk", "pose": [1.5, 4.0, 0.0]}])
    assert _place(site, (0.0, 4.0, 0.2)) is None
    spot = _place(site, (0.0, 4.0, 3.0))
    assert spot is not None and not 3.4 < spot[1] < 4.6


def test_generated_moves_and_inserts_keep_every_use_point_reachable(tmp_path: Path) -> None:
    site = _site(tmp_path)
    assert site_findings(site) == []
    chains = generate(site, seed=0, per_type=4, types=[EditType.MOVE, EditType.INSERT])
    assert len(chains) == 8
    for chain in chains:
        assert [f for f in site_findings(chain.site) if f.point != "model"] == []
        if chain.site.edits[-1].type is EditType.INSERT:
            # an inserted copy queues as its source does
            assert chain.site.edits[-1].params["formation"] == LINE


def test_a_left_behind_demand_node_is_queued_where_the_model_was(tmp_path: Path) -> None:
    site = _site(tmp_path)
    moved = apply(site, EditType.MOVE, "obj:fountain", {"x": 0.25, "y": 6.0, "yaw": 0.0, "leave_demand": True})
    assert [f for f in site_findings(moved) if f.point != "model"] == []
    node = read_graph(moved.graph).objects["fountain_demand"]
    points = object_points(Obstacle.parse({"name": "fountain_demand", "model": "", "pose": [node.x, node.y, node.yaw], "humansim": node.block()}), None)
    assert [(round(p.x, 3), round(p.y, 3)) for p in points] == [(round(10.95 - k, 3), 6.0) for k in range(4)]


def test_a_lock_that_shuts_the_fountains_room_is_refused(tmp_path: Path) -> None:
    site = _site(tmp_path, east=16.0, fountain=(0.25, 6.0, 0.0))
    assert site_findings(site) == []
    why = refusal(site, apply(site, EditType.ACCESS, "door:door", {"locked": True}), set())
    assert why is not None and why.startswith("fountain: use point")


def test_a_door_move_behind_a_desk_is_refused(tmp_path: Path) -> None:
    desk = {"name": "desk", "model": "Common/desk", "pose": [6.6, 5.75, math.pi / 2]}
    site = _site(tmp_path, objects=[desk], east=16.0, fountain=(0.25, 6.0, 0.0))
    assert site_findings(site) == []
    assert refusal(site, apply(site, EditType.LAYOUT, "door:door", {"op": "move", "offset": [0.0, -0.5]}), set()) is None
    why = refusal(site, apply(site, EditType.LAYOUT, "door:door", {"op": "move", "offset": [0.0, 4.0]}), set())
    assert why is not None and why.startswith("fountain: use point")


def test_an_inert_copy_in_front_of_the_fountain_is_refused(tmp_path: Path) -> None:
    site = _site(tmp_path)
    copy = apply(site, EditType.INSERT, "obj:fountain_inert", {"model": "Common/fountain", "x": 10.95, "y": 6.0, "yaw": 0.0})
    why = refusal(site, copy, {"fountain_inert"})
    assert why is not None and why.startswith("fountain: use point at (10.95, 6.00)") and "inside the box of fountain_inert" in why


@pytest.mark.parametrize("world", ["hallway_school", "hallway_clinic"])
def test_every_door_a_generated_layout_edit_moves_or_adds_is_open_for_the_crowd(tmp_path: Path, world: str) -> None:
    # FLOWEDIT v3 had 16 such doors walled shut in `all_walls` (the crowd's walls) though the generator took them as on a wall
    from arena_simulation_setup.generative.flowedit.edit import EditType

    # a copy of its own: the use-point findings are kept by the site's root and key
    shutil.copytree(Path(__file__).resolve().parents[2] / "worlds" / world, tmp_path / world)
    notes: list[str] = []
    chains = generate(open_site(tmp_path / world, "demand"), 0, 5, 1, (EditType.LAYOUT,), notes)
    assert chains, notes
    assert {chain.site.name: [d.name for level in chain.site.world.levels.values() for d in level.walled_doors()] for chain in chains} == {chain.site.name: [] for chain in chains}


def test_an_insert_whose_queue_slot_the_larger_walkers_stop_off_is_refused(tmp_path: Path) -> None:
    # queue slot 2 of the inserted copy 0.025 m from a desk: a 0.25 m walker stops 0.475 m from it, an adult drawn one sd
    # up (0.27 m, routed 0.55 m off) 0.525 m (FLOWEDIT v3's hallway_lab insert 23f93739: a queue slot by a wall, 0.47 m)
    site = _site(tmp_path, objects=[{"name": "desk", "model": "Common/desk", "pose": [3.05, 4.425, 0.0]}])
    assert [f for f in site_findings(site) if f.point != "model"] == []
    params = {"model": "Common/fountain", "x": 0.25, "y": 4.0, "yaw": 0.0, "type": "fountain", "capacity": 1, "satisfies": {"thirst": 100.0}, "formation": LINE}
    why = refusal(site, apply(site, EditType.INSERT, "obj:fountain_2", params), {"fountain_2"})
    assert why is not None and why.startswith("fountain_2: queue slot 2 at (3.05, 4.00)") and "stops 0.52 m from it" in why
