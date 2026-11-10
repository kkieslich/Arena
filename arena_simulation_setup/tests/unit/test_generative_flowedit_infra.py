from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import cli, infra, place, search
from arena_simulation_setup.generative.flowedit import suite as SU
from arena_simulation_setup.generative.flowedit.edit import Site, open_site
from arena_simulation_setup.generative.flowedit.place import CATALOG, Layout, candidate_spots
from arena_simulation_setup.generative.flowedit.split import base_of

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytestmark = pytest.mark.usefixtures("local_root")

HOSPITAL = Path(__file__).resolve().parents[2] / "worlds" / "hospital_1"
SOURCE = [[9.0, 2.0], [11.0, 2.0], [11.0, 3.0], [9.0, 3.0]]
SINK = [[9.0, 30.0], [11.0, 30.0], [11.0, 31.0], [9.0, 31.0]]


@pytest.fixture(autouse=True)
def docked_assets(local_root: Path) -> None:
    from flowedit_docks import write_docked_assets

    write_docked_assets(local_root)


@pytest.fixture
def hospital_copy(local_root: Path) -> Path:
    """hospital_1 in the local worlds root, on the filesystem layouts are saved to, with a crowd scenario `mix`."""
    root = local_root / "worlds" / "hospital_copy"
    shutil.copytree(HOSPITAL, root)
    region = {"rate_profile": [{"t": 0.0, "rate": 0.1}], "max_concurrent": -1, "max_total": -1, "agent": {"agent_type": "adult", "sink_affinity": [{"sink": "far", "weight": 1.0}]}}
    doc = {
        "format": "arena-scenario/2",
        "regions": {
            "near": {"type": "source", "polygon": [{"x": x, "y": y} for x, y in SOURCE], "config": region},
            "far": {"type": "sink", "polygon": [{"x": x, "y": y} for x, y in SINK]},
            "talk": {"type": "area", "polygon": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]},
        },
    }
    (root / "scenarios" / "mix").mkdir(parents=True)
    (root / "scenarios" / "mix" / "scenario.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return root


def _layout(site: Site) -> Layout:
    chosen: list[place.Spot] = []
    for name in CATALOG:
        chosen.append(next(s for s in candidate_spots(site, name) if s.valid and not place.check_layout(site, Layout.of([*chosen, s]))))
    return Layout.of(chosen)


def test_keep_out_reads_the_sources_and_sinks_of_a_scenario(hospital_copy: Path) -> None:
    import shapely

    polygons = infra.keep_out(hospital_copy / "scenarios" / "mix" / "scenario.yaml")
    assert [shapely.get_coordinates(p).tolist()[:4] for p in polygons] == [SOURCE, SINK]


def test_the_reference_suite_runs_the_site_without_robots_at_each_seed() -> None:
    doc = infra.reference_suite("site", [1, 2, 3])
    assert doc["references"] is False and doc["launch"] == infra.LAUNCH and doc["launch"]["human.robots.visible"] == "true"
    assert [s["name"] for s in doc["stages"]] == ["site__s1", "site__s2", "site__s3"]
    for seed, stage in zip([1, 2, 3], doc["stages"], strict=True):
        assert SU.parse_stage_name(stage["name"]) == ("site", seed)
        assert stage["map"] == "site" and stage["seed"] == seed and stage["episodes"] == 1
        assert stage["robot"] == infra.ROBOT == "jackal[4]"
        assert stage["modes"] == {"robots": "stationary", "obstacles": "scenario"}
        assert stage["timeout"] == infra.EPISODE_S == 1500.0
        assert stage["config"] == {"scenario": {"file": infra.SCENARIO}, "stationary": {"pos_x": SU.PARK, "pos_y": SU.PARK}}


def test_the_layout_suite_runs_each_copy_at_each_seed_with_the_fleet_and_its_references() -> None:
    fleet = infra.fleet_config(20.0)
    assert fleet == {"homes": ["bay_1", "bay_2", "charger_1", "charger_2"], "loading": "loading_point", "locker": "locker_1", "rate": 20.0, "pickup_share": 0.3, "load_s": 15.0, "unload_s": 10.0, "charge_below": 30.0, "charge_until": 90.0}
    worlds = ["site__layout_aaaa1111", "site__layout_bbbb2222"]
    doc = infra.layout_suite([*worlds, worlds[0]], fleet, [1, 2])
    assert doc["references"] is True and doc["launch"] == infra.LAUNCH
    assert [s["name"] for s in doc["stages"]] == [f"{w}__s{seed}" for w in worlds for seed in (1, 2)]
    for stage in doc["stages"]:
        world, seed = SU.parse_stage_name(stage["name"])
        assert base_of(world) == "site" and stage["map"] == world and stage["seed"] == seed
        assert stage["modes"] == {"robots": "fleet", "obstacles": "scenario"} and stage["robot"] == "jackal[4]"
        assert stage["config"] == {"scenario": {"file": "study_mix"}, "fleet": fleet}
    with pytest.raises(ValueError, match="charge_below"):
        infra.fleet_config(20.0, charge_below=95.0)
    with pytest.raises(ValueError, match="rate"):
        infra.fleet_config(0.0)


def test_the_rate_asks_for_three_quarters_of_the_slowest_layouts_capacity() -> None:
    # 4 robots, 130 m at 0.26 m/s plus 25 s of holds: 525 s a job
    assert infra.fleet_rate([10.0, 130.0, 50.0], speed_mps=0.26) == pytest.approx(round(0.75 * 4 * 3600 / 525.0, 2))
    assert infra.fleet_rate([130.0]) == pytest.approx(round(0.75 * 4 * 3600 / (130.0 / infra.SPEED_MPS + 25.0), 2))
    with pytest.raises(ValueError):
        infra.fleet_rate([10.0, math.inf])


def test_the_contest_scales_the_battery_and_reads_back_from_its_literal() -> None:
    entries = infra.contest()
    assert entries == [{"name": "fleet_dwb", "mobile": infra.PLANNER, "task.episode.battery.capacity_wh": 22.0}]
    literal = infra.contest_literal(entries)
    assert "\n" not in literal and yaml.safe_load(literal) == entries


def test_write_replaces_only_its_own_suites(tmp_path: Path) -> None:
    out = infra.write(infra.reference_suite("site", [1]), tmp_path / "s.yaml", ["a header"])
    assert out.read_text().startswith(infra.MARKER + "\n# a header\n")
    assert yaml.safe_load(out.read_text())["stages"][0]["name"] == "site__s1"
    infra.write(infra.reference_suite("site", [2]), out)
    (tmp_path / "other.yaml").write_text("stages: []\n")
    with pytest.raises(ValueError, match="not written by flowedit infra"):
        infra.write(infra.reference_suite("site", [1]), tmp_path / "other.yaml")


def test_save_layouts_writes_copies_with_the_crowd_and_hardlinked_assets(hospital_copy: Path) -> None:
    site = open_site(hospital_copy)
    layout = _layout(site)
    scenario = infra.scenario_dir(site, "mix")
    saved = infra.save_layouts(site, [layout, layout], scenario)
    assert list(saved) == [layout.layout_id]
    path = saved[layout.layout_id]
    assert path.name == place.layout_name(site, layout)
    assert (path / "scenarios" / "mix" / "scenario.yaml").read_text() == (scenario / "scenario.yaml").read_text()
    assert infra.hardlinked(hospital_copy, path)
    copy = open_site(path)
    assert {o.name for o in layout.objects} <= {o.name for level in copy.world.levels.values() for o in level.objects}
    with pytest.raises(FileNotFoundError, match="no scenario"):
        infra.scenario_dir(site, "no_such_crowd")


def test_save_layouts_refuses_a_root_it_cannot_hardlink_into(hospital_copy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    def no_link(*_: object) -> None:
        raise OSError(18, "Invalid cross-device link")

    site = open_site(hospital_copy)
    monkeypatch.setattr(os, "link", no_link)
    with pytest.raises(ValueError, match="cannot hardlink"):
        infra.save_layouts(site, [_layout(site)], infra.scenario_dir(site, "mix"))


def test_cli_reference_writes_the_site_suite(hospital_copy: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "suites" / "infra_h1_site.yaml"
    assert cli.main(["infra", "reference", "--site", str(hospital_copy), "--scenario", "mix", "--seeds", "1-3,5", "--out", str(out)]) == 0
    doc = yaml.safe_load(out.read_text())
    assert [s["name"] for s in doc["stages"]] == ["hospital_copy__s1", "hospital_copy__s2", "hospital_copy__s3", "hospital_copy__s5"]
    assert "arena evaluation benchmark --suite infra_h1_site" in capsys.readouterr().out
    assert cli.main(["infra", "reference", "--site", str(hospital_copy), "--scenario", "nope", "--out", str(out)]) == 1


def test_cli_layouts_searches_saves_and_writes_the_fleet_suite(hospital_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    site = open_site(hospital_copy)
    loading = next(s for s in candidate_spots(site, "loading_point") if s.valid and s.zone == "central_hallway")
    level = site.world.levels[loading.level]
    base = NavrBase(level, exclude={o.name for o in level.objects})  # the fake assets have no hospital furniture
    monkeypatch.setattr(search.Search, "base", property(lambda self: base))
    cells = [{"planner": "peds_only", "stage": f"hospital_copy__s{seed}", "flow_world": "hospital_copy", "flow_base": "hospital_copy", "flow_seed": seed, "x": 10.25, "y": 20.25, "cell_m": 0.5, "heading_rad": 0.0, "ped_seconds": 50.0} for seed in (1, 2)]
    pd.DataFrame(cells).to_parquet(tmp_path / "flow_cells.parquet")
    out = tmp_path / "suites" / "infra_h1.yaml"
    x, y = loading.objects[0].goto
    args = ["infra", "layouts", "--site", str(hospital_copy), "--scenario", "mix", "--occupancy", str(tmp_path), "--loading", f"{x},{y}", "--seeds", "1-2", "--samples", "20", "--random", "2", "--out", str(out)]
    assert cli.main([*args, "--dry-run"]) == 0
    assert not out.exists()
    assert cli.main(args) == 0
    assert "&id" not in out.read_text()  # each stage its own fleet block, no YAML aliases
    doc = yaml.safe_load(out.read_text())
    manifest = json.loads(out.with_name("infra_h1.layouts.json").read_text())
    roles = [r["role"] for r in manifest["runs"]]
    assert roles == ["robot_only", "joint", "random", "random"]
    ids = [r["layout_id"] for r in manifest["runs"]]
    assert len(set(ids)) == 4
    assert [s["name"] for s in doc["stages"]] == [f"hospital_copy__layout_{i}__s{seed}" for i in ids for seed in (1, 2)]
    assert doc["references"] is True
    rate = manifest["rate_per_h"]
    assert rate == pytest.approx(infra.fleet_rate([r["travel_m"] for r in manifest["runs"]]), abs=0.01)
    assert all(s["config"]["fleet"]["rate"] == rate for s in doc["stages"])
    for i in ids:
        world = Path(manifest["worlds"][i])
        assert (world / "scenarios" / "mix" / "scenario.yaml").is_file() and infra.hardlinked(hospital_copy, world)
    assert manifest["loading"]["spot"] == loading.id
    assert len(manifest["joint_top"]) <= search.JOINT_TOP
    printed = capsys.readouterr().out
    assert "--suite infra_h1 --contest" in printed and "task.episode.battery.capacity_wh: 22.0" in printed
