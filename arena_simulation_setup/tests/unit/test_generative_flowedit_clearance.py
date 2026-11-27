"""The clearance rule (generative/flowedit/clearance.py): every gap an element's floor leaves beside walls, pillars,
fixtures and other elements, by the site's directional base flow. A bay pair's slots beside a pillar (the twin's
travel-optimal b961c4b7: 0.77 m on a wall-side stream), slots sealed flush, a recess, one-way and two-way flow, a
locker's dock, no base flow (conservative), INFRA-FlowNet's syn_straight charger neck, and the rule in `place` on
hospital_1."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import attrs
import pytest
import shapely

pytest.importorskip("numpy")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402
from arena_simulation_setup.generative.flowedit import clearance as C  # noqa: E402
from arena_simulation_setup.generative.flowedit import place  # noqa: E402
from arena_simulation_setup.generative.flowedit.walkway import ONE_WAY_M, TWO_WAY_M  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "clearance"
HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"
#: The base flow's cells and its recording
CELL, SECONDS = 0.5, 3600.0
#: Heading bins: east [0, pi/4) and west [-pi, -3 pi/4)
EAST, WEST = 4, 0


def _ring(cx: float, cy: float, r: float, n: int = 24) -> list[list[list[float]]]:
    pts = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]
    return [[list(pts[k]), list(pts[(k + 1) % n])] for k in range(n)]


#: The pillar's radius; its centre's x, in front of the elements
R, PX = 0.35, 19.0


def _corridor(gap: float, front: float = -3.74) -> dict[str, Any]:
    """A 40 m x 10 m corridor (walls at y = -5 and 5) with a 0.7 m pillar of the twin's colonnade `gap` in front of
    an element on the south wall whose front is at y = `front`: the corridor's stream passes between them."""
    walls = [[[0, -5], [40, -5]], [[40, -5], [40, 5]], [[40, 5], [0, 5]], [[0, 5], [0, -5]]] + _ring(PX, front + gap + R + C.JOIN_M / 2.0, R)
    return {"floor": [[[0, -5], [40, -5], [40, 5], [0, 5]]], "walls": walls, "obstacles": {}}


def _bays() -> dict[str, Any]:
    """A bay pair on the south wall, its slots 1.0 x 1.2 m from the wall gap (y -4.94 to -3.74), x 18 to 20."""
    objs = []
    for k, x in enumerate((18.0, 19.0)):
        slot = [[x, -4.94], [x + 1.0, -4.94], [x + 1.0, -3.74], [x, -3.74]]
        objs.append({"name": f"bay_{k + 1}", "footprint": slot, "dock": slot, "blocks_floor": False})
    return {"layout_id": "t", "elements": [{"element": "bay_pair", "objects": objs}]}


def _tracks(*paths: tuple[float, float, float, float, int], hours: float = 1.0) -> C.Tracks:
    """Walkers walking straight at 1 m/s, sampled every 0.1 s: per path (x0, y0) to (x1, y1), `n` walkers."""
    keys, ts, xs, ys = [], [], [], []
    for p, (x0, y0, x1, y1, n) in enumerate(paths):
        steps = int(math.dist((x0, y0), (x1, y1)) / 0.1)
        for w in range(n):
            for k in range(steps + 1):
                keys.append(f"{p}:{w}")
                ts.append(w * 100.0 + k * 0.1)
                xs.append(x0 + (x1 - x0) * k / steps)
                ys.append(y0 + (y1 - y0) * k / steps)
    return C.Tracks.from_samples(np.array(keys), np.array(ts), np.array(xs), np.array(ys), hours)


def _flow(*bins_rates: tuple[int, float], x0: float = 0.0, y0: float = -5.0, nx: int = 80, ny: int = 20) -> C.BaseFlow:
    """A uniform base flow over the corridor: walkers an hour per metre of cross-section in each heading bin given.
    A stream of q walkers/h per metre at speed v holds q / v / 3600 walkers per m^2: that many pedestrian-seconds
    per m^2 and recorded second."""
    time = np.zeros((ny, nx, C.BINS))
    for b, q in bins_rates:
        time[:, :, b] = q / C.WALK_SPEED_MPS / 3600.0 * CELL * CELL * SECONDS
    return C.BaseFlow(x0, y0, CELL, SECONDS, time)


def test_a_bay_slot_0_77_m_from_a_pillar_on_a_wall_side_stream_fails() -> None:
    # b961c4b7: the bay pair's slots 0.77 m from a pillar, a stream passing between them that carries no shortest route
    g = _corridor(0.77)
    lay = _bays()
    for flow in (None, _flow((EAST, 150.0), (WEST, 150.0)), _flow((EAST, 300.0))):
        (gap,) = [p for p in C.check_record(lay, g, flow) if "bay_pair" in p["what"] and "wall_piece" in p["what"]]
        assert gap["gap_m"] == pytest.approx(0.77, abs=0.01)
        assert gap["gap_m"] < gap["required_m"]
    two = C.check_record(lay, g, _flow((EAST, 150.0), (WEST, 150.0)))
    assert two[0]["required_m"] == TWO_WAY_M and two[0]["walkers_ph"] > 10.0
    # one-way, it is still under a wheelchair's 0.9 m
    one = C.check_record(lay, g, _flow((EAST, 300.0)))
    assert one[0]["required_m"] == ONE_WAY_M


def test_a_slot_sealed_flush_to_the_pillar_passes() -> None:
    # 0.10 m from the pillar: sealed, nobody walks through; the stream goes round both
    assert C.check_record(_bays(), _corridor(0.10), _flow((EAST, 150.0), (WEST, 150.0))) == []
    # touching it
    assert C.check_record(_bays(), _corridor(0.0)) == []
    # 1.8 m from it, two-way or not
    assert C.check_record(_bays(), _corridor(1.85)) == []


def test_the_wall_an_element_stands_against_does_not_hide_the_far_side_of_its_corridor() -> None:
    # a 2.6 m corridor whose walls are one connected piece: the bays stand against its south wall, its north wall is
    # the gap's other side
    walls = [[[0, -1.3], [40, -1.3]], [[40, -1.3], [40, 1.3]], [[40, 1.3], [0, 1.3]], [[0, 1.3], [0, -1.3]]]
    g = {"floor": [[[0, -1.3], [40, -1.3], [40, 1.3], [0, 1.3]]], "walls": walls, "obstacles": {}}
    slots = [[[x, -1.24], [x + 1.0, -1.24], [x + 1.0, -0.04], [x, -0.04]] for x in (18.0, 19.0)]
    lay = {"layout_id": "c", "elements": [{"element": "bay_pair", "objects": [{"name": f"bay_{k + 1}", "footprint": s, "dock": s} for k, s in enumerate(slots)]}]}
    (p,) = C.check_record(lay, g)
    assert p["gap_m"] == pytest.approx(1.3 - 0.025 + 0.04, abs=1e-3) and p["required_m"] == TWO_WAY_M
    assert p["segment"][1][1] == pytest.approx(1.275, abs=1e-3)


def test_a_gap_into_a_pocket_counts_where_base_walkers_crossed_its_line() -> None:
    # a 6 m x 4 m pocket off the corridor, open to the west between y 0 and 2; a bay slot sealed to its upper jamb
    # leaves 0.775 m of the opening
    g = _corridor(6.0)
    g["walls"] += [[[30, 0], [36, 0]], [[36, 0], [36, 4]], [[36, 4], [30, 4]], [[30, 4], [30, 2]]]
    slot = [[29.5, 0.8], [30.5, 0.8], [30.5, 2.0], [29.5, 2.0]]
    lay = {"layout_id": "p", "elements": [{"element": "bay_pair", "objects": [{"name": "bay_1", "footprint": slot, "dock": slot}]}]}
    split = _flow((EAST, 150.0), (WEST, 150.0))
    crossed = attrs.evolve(split, tracks=_tracks((27.0, 0.4, 33.0, 0.4, 3), (33.0, 0.3, 27.0, 0.3, 3)))
    passed_by = attrs.evolve(split, tracks=_tracks((10.0, -2.0, 38.0, -2.0, 50)))
    # walkers crossed the gap's line before the layout: it is theirs to walk through
    (p,) = C.check_record(lay, g, crossed)
    assert p["gap_m"] == pytest.approx(0.775, abs=1e-3) and p["walkers_ph"] == pytest.approx(6.0) and "by tracks" in p["why"]
    # nobody crossed it: a pocket no base walker went into, no passage
    assert C.check_record(lay, g, passed_by) == [] and C.check_record(lay, g, _flow()) == []
    # without tracks the map counts the walkers at the line (here the uniform stream): it counts
    assert len(C.check_record(lay, g, split)) == 1
    # without a base flow the pocket counts (conservative)
    (p,) = C.check_record(lay, g)
    assert p["why"] == C.CONSERVATIVE
    # a nook behind the gap, its floor within 2.5 m of it: with a base flow it counts where walkers crossed its line;
    # without one it is a recess
    nook = _corridor(6.0)
    nook["walls"] += [[[30, 0], [32, 0]], [[32, 0], [32, 2]], [[32, 2], [30, 2]]]
    assert len(C.check_record(lay, nook, crossed)) == 1 and C.check_record(lay, nook, passed_by) == []
    assert C.check_record(lay, nook) == []


def test_tracks_count_each_walker_crossing_a_segment_once_per_direction() -> None:
    seg = ((20.05, -1.0), (20.05, 1.0))  # its left normal points west
    t = _tracks((15.0, 0.0, 25.0, 0.0, 4), (25.0, 0.5, 15.0, 0.5, 2), (15.0, 3.0, 25.0, 3.0, 9), hours=0.5)
    # eastward walkers cross it from its left (back), westward ones from its right (forth); the third path misses it
    assert t.crossing(seg) == (pytest.approx(4.0), pytest.approx(8.0), pytest.approx(12.0))
    seg = ((20.0, -1.0), (20.0, 1.0))
    # a walker going there and back counts once each way, once in all
    back_and_forth = C.Tracks.from_samples(np.array(["a"] * 3), np.array([0.0, 0.1, 0.2]), np.array([19.95, 20.05, 19.95]), np.zeros(3), 1.0)
    assert back_and_forth.crossing(seg) == (1.0, 1.0, 1.0)
    # samples further apart than a step are no step
    gap = C.Tracks.from_samples(np.array(["a", "a"]), np.array([0.0, 2.0]), np.array([19.9, 20.1]), np.zeros(2), 1.0)
    assert gap.crossing(seg) == (0.0, 0.0, 0.0)


def test_the_walkers_an_hour_come_from_the_tracks_where_given() -> None:
    g = _corridor(1.2)  # 1.2 m between the bay slots and the pillar, crossed east-west
    split = _flow((EAST, 150.0), (WEST, 150.0))  # two-way by the map's split
    few = attrs.evolve(split, tracks=_tracks((17.0, -3.1, 21.0, -3.1, 4), (21.0, -3.0, 17.0, -3.0, 4)))
    many = attrs.evolve(split, tracks=_tracks((17.0, -3.1, 21.0, -3.1, 6), (21.0, -3.0, 17.0, -3.0, 6)))
    assert C.check_record(_bays(), g, few) == []  # 8 walkers/h: one-way, 1.2 m >= 0.9 m
    (p,) = C.check_record(_bays(), g, many)  # 12 walkers/h both ways: two-way
    assert p["required_m"] == TWO_WAY_M and p["walkers_ph"] == pytest.approx(12.0)


def test_the_map_estimate_is_calibrated_to_the_tracks_on_the_twin() -> None:
    # an 8 x 8 m window of the v3 twin's site-as-is run (seed 299, 2700 s): its flow cells and walker tracks
    import pandas as pd

    cells = pd.read_parquet(FIXTURES / "site_v3_window_cells.parquet")
    tracks = pd.read_parquet(FIXTURES / "site_v3_window_tracks.parquet")
    flow = C.BaseFlow.from_cells(cells["x"], cells["y"], cells["heading_rad"], cells["ped_seconds"], 0.5, 2700.0)
    walkers = C.Tracks.from_samples(tracks["key"].to_numpy(), tracks["t"].to_numpy(), tracks["x"].to_numpy(dtype=float), tracks["y"].to_numpy(dtype=float), 0.75)
    x0, y0 = float(cells["x"].min()), float(cells["y"].min())
    rng = np.random.default_rng(1)
    ratios = []
    for _ in range(400):
        cx, cy = x0 + 1.5 + rng.random() * 5.0, y0 + 1.5 + rng.random() * 5.0
        half, th = rng.uniform(0.3, 0.9), rng.uniform(0.0, math.pi)
        seg = ((cx - half * math.cos(th), cy - half * math.sin(th)), (cx + half * math.cos(th), cy + half * math.sin(th)))
        mapped, counted = sum(flow.crossing(seg)), walkers.crossing(seg)[2]
        if counted >= C.FLOW_MIN_PED_H and mapped > 0:
            ratios.append(counted / mapped)
    assert len(ratios) >= 100
    # the calibrated map's estimate matches the tracks' count in the median, within the site-wide quartiles
    assert 0.75 <= float(np.median(ratios)) <= 1.25
    assert attrs.evolve(flow, tracks=walkers).walkers(seg)[1] == "tracks"


def test_a_pair_beside_a_room_corner_leaves_a_recess_not_a_passage() -> None:
    # a bay pair on the south wall 1.2 m from the west wall: the 1.2 m wide nook between them is closed by the wall
    g = _corridor(6.0)
    slots = [[[x, -4.94], [x + 1.0, -4.94], [x + 1.0, -3.74], [x, -3.74]] for x in (1.225, 2.225)]
    lay = {"layout_id": "k", "elements": [{"element": "bay_pair", "objects": [{"name": f"bay_{k + 1}", "footprint": s, "dock": s} for k, s in enumerate(slots)]}]}
    assert C.check_record(lay, g) == []
    # measured across the middle of the nook, its closed end is in sight of the through test: not counted at all
    assert not [x for x in C.gaps(C.record_pieces(lay), C.site_obstacles(g), _flow(), all_gaps=True) if x.width < 1.5]


def test_a_recess_between_wall_elements_is_no_passage() -> None:
    g = _corridor(6.0)
    bays = [{"name": f"bay_{k}", "footprint": [[x, 3.74], [x + 1, 3.74], [x + 1, 4.94], [x, 4.94]], "dock": [[x, 3.74], [x + 1, 3.74], [x + 1, 4.94], [x, 4.94]]} for k, x in ((1, 5.0), (2, 6.0))]
    locker = {"name": "locker_1", "footprint": [[8.48, 4.14], [9.52, 4.14], [9.52, 4.94], [8.48, 4.94]], "dock": [[8.5, 2.9], [9.5, 2.9], [9.5, 4.1], [8.5, 4.1]], "blocks_floor": True}
    lay = {"layout_id": "r", "elements": [{"element": "bay_pair", "objects": bays}, {"element": "locker", "objects": [locker]}]}
    # the 1.48 m nook between the bays and the locker opens onto the wall: no through-passage
    assert C.check_record(lay, g) == []
    counted = C.gaps(C.record_pieces(lay), C.site_obstacles(g), all_gaps=True)
    assert not [x for x in counted if "bay_pair" in x.what and "locker_1#footprint" in x.what]


@pytest.mark.parametrize(
    ("flow", "required", "ok"),
    [
        (((EAST, 150.0), (WEST, 150.0)), TWO_WAY_M, False),  # both ways, 300 walkers/h: 1.8 m
        (((EAST, 300.0),), ONE_WAY_M, True),  # one way: 0.9 m
        (((EAST, 290.0), (WEST, 10.0)), ONE_WAY_M, True),  # 3 % the other way: one-way
        (((EAST, 2.0), (WEST, 2.0)), ONE_WAY_M, True),  # both ways but under 10 walkers/h through the gap
        ((), ONE_WAY_M, True),  # nobody crosses it
        (None, TWO_WAY_M, False),  # no base flow: two-way, conservative
    ],
)
def test_one_way_or_two_way_by_the_base_flow(flow: Any, required: float, ok: bool) -> None:  # noqa: ANN401
    g = _corridor(1.2)  # a 1.2 m gap between the slots and the pillar
    lay = _bays()
    base = None if flow is None else _flow(*flow)
    counted = [x for x in C.gaps(C.record_pieces(lay), C.site_obstacles(g), base, all_gaps=True) if "wall_piece" in x.what and x.width < 1.5]
    (gap,) = counted
    assert gap.width == pytest.approx(1.2, abs=0.01)
    assert gap.required == required and gap.ok is ok
    if flow is None:
        assert gap.why == C.CONSERVATIVE and gap.walkers_ph is None


def test_the_base_flow_counts_the_walkers_crossing_a_gap() -> None:
    # 300 walkers/h per metre in the heading bin round pi/8 through a north-south gap 2 m long: 600 cos(pi/8) walkers/h
    # cross it, westward from the gap's direction (north) is forth; the band's round ends hold walkers too, so the flux
    # reads a little more, never fewer, and is then scaled to the tracks' count (`MAP_CALIBRATION`)
    base = _flow((EAST, 300.0))
    forth, back = base.crossing(((20.0, -1.0), (20.0, 1.0)))
    true = 600.0 * math.cos(math.pi / 8)
    assert forth == 0.0 and true <= back / C.MAP_CALIBRATION <= 1.6 * true
    assert base.walkers(((20.0, -1.0), (20.0, 1.0))) == (pytest.approx(back), "map")
    # a gap along the bin's heading: nobody crosses it
    d = (math.cos(math.pi / 8), math.sin(math.pi / 8))
    assert base.crossing(((20.0 - d[0], -d[1]), (20.0 + d[0], d[1]))) == (0.0, 0.0)


def test_a_lockers_dock_keeps_0_9_m_whatever_the_flow() -> None:
    # a locker on the south wall, its dock in front (y -4.1 to -2.9), a pillar in front of the dock
    fp = [[18.48, -4.94], [19.52, -4.94], [19.52, -4.14], [18.48, -4.14]]
    dock = [[18.5, -4.1], [19.5, -4.1], [19.5, -2.9], [18.5, -2.9]]
    locker = {"layout_id": "l", "elements": [{"element": "locker", "objects": [{"name": "locker_1", "footprint": fp, "dock": dock, "blocks_floor": True}]}]}
    two_way = _flow((EAST, 150.0), (WEST, 150.0))
    (p,) = C.check_record(locker, _corridor(0.85, front=-2.9), two_way)
    assert p["docked"] and p["required_m"] == ONE_WAY_M and p["gap_m"] == pytest.approx(0.85, abs=0.01)
    # 0.95 m beside a docked robot is a lane taking turns, on a two-way stream too; the locker itself stands clear
    assert C.check_record(locker, _corridor(0.95, front=-2.9), two_way) == []
    assert C.check_record(locker, _corridor(0.95, front=-2.9)) == []


def test_without_a_base_flow_every_gap_is_two_way_and_the_verdict_says_so() -> None:
    (p,) = [x for x in C.check_record(_bays(), _corridor(1.2)) if "wall_piece" in x["what"]]
    assert p["required_m"] == TWO_WAY_M and p["why"] == C.CONSERVATIVE
    with pytest.raises(ValueError, match="recorded seconds"):
        C.BaseFlow(0.0, 0.0, CELL, 0.0, np.zeros((2, 2, C.BINS)))
    with pytest.raises(ValueError, match="no flow cells"):
        C.BaseFlow.from_cells([], [], [], [], CELL, SECONDS)


def test_the_occupied_slot_is_at_least_1_0_by_1_2() -> None:
    s = C.slot_polygon([[0, 0], [0.6, 0], [0.6, 0.8], [0, 0.8]])
    xs, ys = s.exterior.xy
    assert max(xs) - min(xs) == pytest.approx(1.0) and max(ys) - min(ys) == pytest.approx(1.2)
    assert shapely.Point(0.3, 0.4).within(s)
    big = shapely.box(0, 0, 1.0, 1.2)
    assert C.slot_polygon(big).equals(big)


def test_the_heading_bins_are_the_flow_maps() -> None:
    from arena_simulation_setup.generative.flowedit.forecast import maps

    assert (C.HEADINGS, C.BINS) == (maps.HEADINGS, maps.BINS)
    centres = [maps.bin_centre(b) for b in range(maps.HEADINGS)]
    flow = C.BaseFlow.from_cells([0.25] * 9, [0.25] * 9, [*centres, None], list(range(1, 10)), CELL, SECONDS)
    assert flow.time[0, 0].tolist() == [float(k) for k in range(1, 10)]
    # INFRA-FlowNet's (bins, ny, nx) layout
    assert np.array_equal(C.BaseFlow.from_bins_first(np.moveaxis(flow.time, -1, 0), 0.0, 0.0, SECONDS).time, flow.time)


def test_the_syn_straight_charger_pair_beside_a_colonnade_pillar_fails() -> None:
    # INFRA-FlowNet v2's sampler, run without a crowd, placed the charger pair's docks 0.93 m from a colonnade pillar,
    # where 13-14 walkers got stuck
    fixture = json.loads((FIXTURES / "syn_straight.json").read_text())
    problems = C.check_record(fixture["layout"], fixture["geom"])
    neck = [p for p in problems if "charger_pair" in p["what"] and p["required_m"] == TWO_WAY_M]
    assert neck and 0.85 < neck[0]["gap_m"] < 0.95
    (ax, ay), (bx, by) = neck[0]["segment"]
    assert 131 < ax < 134 and -3.5 < ay < -2.0 and math.dist((bx, by), (133.5, -2.25)) < 0.4
    # the same as INFRA-FlowNet's own guard found
    assert sorted((p["gap_m"], p["required_m"], p["segment"]) for p in problems) == sorted((p["gap_m"], p["required_m"], p["segment"]) for p in fixture["guard"])


# --- the rule in place ----------------------------------------------------------------------------------------------


@pytest.fixture
def hospital(local_root: Path) -> Any:  # noqa: ANN401
    from arena_simulation_setup.generative.flowedit.edit import open_site
    from flowedit_docks import write_docked_assets

    write_docked_assets(local_root)
    return open_site(HOSPITAL)


def test_every_spot_gets_a_clearance_verdict_conservative_without_a_base_flow(hospital: Any) -> None:  # noqa: ANN401
    spots = place.candidate_spots(hospital, "charger_pair")
    assert spots and {s.verdict[place.Rule.CLEARANCE] for s in spots} <= {place.Verdict.PASS, place.Verdict.FAIL}
    assert {s.clearance_basis for s in spots} == {C.CONSERVATIVE}
    failed = [s for s in spots if s.verdict[place.Rule.CLEARANCE] is place.Verdict.FAIL]
    assert failed and all(s.clearance and all(g.width < g.required == TWO_WAY_M for g in s.clearance) for s in failed)
    assert all(not s.clearance for s in spots if s.verdict[place.Rule.CLEARANCE] is place.Verdict.PASS)
    # a base flow nobody walks in asks 0.9 m of every gap: what fails then fails conservatively too
    level = next(iter(hospital.world.levels.values()))
    x0, y0, x1, y1 = shapely.union_all([z.shape() for z in level.zones if len(z.polygon) >= 3]).bounds
    empty = C.BaseFlow(x0, y0, CELL, SECONDS, np.zeros((math.ceil((y1 - y0) / CELL), math.ceil((x1 - x0) / CELL), C.BINS)))
    still = place.candidate_spots(hospital, "charger_pair", facts=place.SiteFacts(base_flow=empty))
    assert {s.clearance_basis for s in still} == {"base flow"} and all(s.base_flow is empty for s in still)
    fail_one = {s.id for s in still if s.verdict[place.Rule.CLEARANCE] is place.Verdict.FAIL}
    assert fail_one <= {s.id for s in failed}
    assert all(g.required == ONE_WAY_M for s in still for g in s.clearance)


def test_check_layout_names_a_gap_two_elements_leave_between_them(hospital: Any) -> None:  # noqa: ANN401
    from arena_simulation_setup.generative.flowedit.place import Layout

    def clear(element: str) -> list[Any]:
        return [s for s in place.candidate_spots(hospital, element) if s.verdict[place.Rule.CLEARANCE] is place.Verdict.PASS]

    # a charger pair and a bay pair facing each other across the central hallway, each clear of the walls alone
    pairs = [(c, b) for c in clear("charger_pair") for b in clear("bay_pair") if c.zone == b.zone == "central_hallway" and 0.5 < c.taken.distance(b.taken) < 1.5 and c.frame.n[0] * b.frame.n[0] + c.frame.n[1] * b.frame.n[1] < -0.9]
    assert pairs
    charger, bays = min(pairs, key=lambda p: p[0].taken.distance(p[1].taken))
    width = charger.taken.distance(bays.taken)
    problems = [p for p in place.check_layout(hospital, Layout.of([charger, bays])) if p.startswith("clearance:")]
    # the gap between them, once, against the conservative two-way width
    assert len(problems) == 1 and "charger_pair" in problems[0] and "bay_pair" in problems[0], problems
    assert f"{width:.2f} m gap" in problems[0] and "needs 1.80 m" in problems[0] and C.CONSERVATIVE in problems[0]
    (gap,) = place.clearance_gaps(hospital, [charger, bays], only=["bay_pair"])
    assert gap.width == pytest.approx(width, abs=1e-3) and gap.required == TWO_WAY_M
    # a base flow nobody walks in asks 0.9 m: a gap of 0.9 m or more passes
    level = next(iter(hospital.world.levels.values()))
    x0, y0, x1, y1 = shapely.union_all([z.shape() for z in level.zones if len(z.polygon) >= 3]).bounds
    empty = C.BaseFlow(x0, y0, CELL, SECONDS, np.zeros((math.ceil((y1 - y0) / CELL), math.ceil((x1 - x0) / CELL), C.BINS)))
    assert all(g.required == ONE_WAY_M for g in place.clearance_gaps(hospital, [charger, bays], empty, all_gaps=True))
