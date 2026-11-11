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
SOURCE = [[9.0, 6.0], [11.0, 6.0], [11.0, 7.0], [9.0, 7.0]]
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
    assert doc["launch"]["human.skeleton"] == "false"
    # the recorder keeps the tracks, robots and semantics, not the pedestrians' skeletons, named frames or arena_peds
    assert doc["record"] == infra.RECORD and doc["record"]["exclude"] == ["peds", "animation_states", "tf_humans"]
    import re

    dropped = [re.compile(pattern) for pattern, ms in doc["record"]["tf_frames"] if ms < 0]
    for frame in ("body_env_0_agent_2", "l_p_collar_env_0_agent_2", "root_x_env_0_agent_6", "env_0/flow_1", "env_0/south_adult#12", "10", "107"):
        assert any(p.search(frame) for p in dropped), frame
    for frame in ("env_0/jackal_0/base_link", "env_0/jackal_0/odom", "env_0/jackal/front_left_wheel_link", "map", "odom"):
        assert not any(p.search(frame) for p in dropped), frame
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
    assert doc["references"] == ["unhindered_peds"] and doc["launch"] == infra.LAUNCH
    assert (infra.episodes(doc), infra.sim_hours(doc)) == (8, round(8 * infra.EPISODE_S / 3600.0, 2))
    assert infra.episodes(infra.reference_suite("site", [1, 2, 3])) == 3
    assert [s["name"] for s in doc["stages"]] == [f"{w}__s{seed}" for w in worlds for seed in (1, 2)]
    for stage in doc["stages"]:
        world, seed = SU.parse_stage_name(stage["name"])
        assert base_of(world) == "site" and stage["map"] == world and stage["seed"] == seed
        assert stage["modes"] == {"robots": "fleet", "obstacles": "scenario"} and stage["robot"] == "jackal[4]"
        assert stage["config"] == {"scenario": {"file": infra.SCENARIO}, "fleet": fleet}
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
    assert entries == [{"name": "fleet_dwb", "mobile": infra.PLANNER, "task.episode.battery.capacity_wh": 36.0, "task.episode.battery.soc.min": 40.0, "task.episode.battery.soc.max": 100.0}]
    with pytest.raises(ValueError, match="soc_min"):
        infra.contest(soc_min=80.0, soc_max=50.0)
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
    joint = roles.count("joint")
    assert 1 <= joint <= search.JOINT_TOP and roles == ["robot_only", *["joint"] * joint, "random", "random"]
    ids = list(dict.fromkeys(r["layout_id"] for r in manifest["runs"]))
    assert [r["layout_id"] for r in infra.runs(manifest)] == ids
    assert [s["name"] for s in doc["stages"]] == [f"hospital_copy__layout_{i}__s{seed}" for i in ids for seed in (1, 2)]
    assert doc["references"] == ["unhindered_peds"]
    prefilter = manifest["travel_prefilter"]
    robot = manifest["runs"][0]
    assert prefilter["max_travel_m"] == pytest.approx(search.TRAVEL_SLACK * robot["travel_m"], abs=1e-3)
    assert all(r["travel_m"] <= prefilter["max_travel_m"] + 1e-6 for r in manifest["runs"] if r["role"] == "joint")
    assert all(r["ranking_score"] is not None for r in manifest["runs"])
    assert manifest["episodes"] == 2 * len(doc["stages"]) and manifest["site_config"] is None
    rate = manifest["rate_per_h"]
    assert rate == pytest.approx(infra.fleet_rate([r["travel_m"] for r in manifest["runs"]]), abs=0.01)
    assert all(s["config"]["fleet"]["rate"] == rate for s in doc["stages"])
    for i in ids:
        world = Path(manifest["worlds"][i])
        assert (world / "scenarios" / "mix" / "scenario.yaml").is_file() and infra.hardlinked(hospital_copy, world)
    assert manifest["loading"]["spot"] == loading.id
    assert len(manifest["joint_top"]) <= search.JOINT_TOP
    printed = capsys.readouterr().out
    assert "--suite infra_h1 --contest" in printed and "task.episode.battery.capacity_wh: 36.0" in printed


VISITORS = {
    "rate_per_h": 6.0,
    "target": {"name": "locker_visit", "humansim": {"type": "parcel_locker", "capacity": 1, "formation": {"type": "line", "params": {"base_step": 0.9}}}, "clearance_m": 0.35},
    "regions": {
        "near_locker": {
            "type": "source",
            "polygon": [{"x": x, "y": y} for x, y in [[9.0, 8.0], [11.0, 8.0], [11.0, 9.0], [9.0, 9.0]]],
            "config": {"rate_profile": [{"t": 0.0, "rate": 0.001}], "max_concurrent": -1, "max_total": -1, "agent": {"agent_type": "./types/visitor.yaml", "sink_affinity": [{"sink": "far", "weight": 1.0}]}},
        }
    },
}


@pytest.fixture
def with_visitors(hospital_copy: Path) -> Path:
    """hospital_copy whose `mix` crowd has locker visitors beside it, and their type."""
    folder = hospital_copy / "scenarios" / "mix"
    (folder / infra.VISITORS).write_text(yaml.safe_dump(VISITORS, sort_keys=False))
    (folder / "types").mkdir()
    (folder / "types" / "visitor.yaml").write_text("name: visitor\nextends: adult\n")
    text = (folder / "scenario.yaml").read_text()
    (folder / "scenario.yaml").write_text("# the mix crowd\n# of the copy\n" + text)
    return hospital_copy


def test_the_shipped_twin_config_holds_site_facts_and_pins_the_courier_handoff_point_by_rule() -> None:
    config = infra.site_config("smu_link_full")
    assert config.site == "smu_link_full" and config.scenario == "study_entrances" and config.crowd_side == "left"
    assert config.exclude == ("ramp",)
    facts = config.facts
    assert facts.courier_access == search.TWIN_LOADING_XY and facts.outlets == ()
    assert {e.name for e in facts.recipient_entrances} == {"mrt_bencoolen", "scis1_student", "scis1_staff"}
    assert {e.name for e in facts.entrances} - {e.name for e in facts.recipient_entrances} == {"bras_basah_2", "escalator_1"}
    assert all(e.placeholder for e in facts.entrances)
    # the rules compile to the search's engine: the courier handoff point pinned at the access point, nothing else
    assert config.placements == {"loading_point": search.Placement(pin=search.TWIN_LOADING_XY)}
    assert [(types, share) for types, share in config.flow_split] == [(("mrt",), 0.5), (("building", "shop", "escalator", "stairs"), 0.3), (("lift", "other"), 0.2)]
    record = config.record()
    assert "elements" not in record and record["pins"] == {"loading_point": list(search.TWIN_LOADING_XY)}
    assert record["entrances"][2] == {"name": "mrt_bencoolen", "type": "mrt", "points": [[157.5, 241.5]], "recipients": True, "level": None, "placeholder": True}
    with pytest.raises(FileNotFoundError, match="no INFRA site config"):
        infra.site_config("no_such_site")


FACTS = {"site": "s", "courier_access": [1, 2]}


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"site": "s", "courier_access": [1, 2], "elements": {"locker": {"zones": ["a"]}}}, "not elements"),
        ({"site": "s"}, "no courier_access"),
        ({**FACTS, "courier_access": [1]}, r"courier_access must be \[x, y\]"),
        ({**FACTS, "entrances": [{"name": "e", "type": "portal", "at": [0, 0]}]}, "type 'portal'"),
        ({**FACTS, "entrances": [{"name": "e", "type": "mrt"}]}, "`at`"),
        ({**FACTS, "entrances": [{"name": "e", "type": "mrt", "at": [0, 0], "polygon": [[0, 0], [1, 0], [1, 1]]}]}, "`at`"),
        ({**FACTS, "entrances": [{"name": "e", "type": "mrt", "polygon": [[0, 0], [1, 0]]}]}, "a point or a polygon"),
        ({**FACTS, "entrances": [{"name": "e", "type": "mrt", "at": [0, 0], "users": True}]}, "unknown keys"),
        ({**FACTS, "entrances": [{"name": "e", "type": "mrt", "at": [0, 0]}, {"name": "e", "type": "shop", "at": [1, 1]}]}, "names repeat"),
        ({**FACTS, "outlets": [[1, 2, 3]]}, "an outlet must be"),
        ({**FACTS, "flow_split": [{"types": ["mrt"], "share": 0.6}, {"types": ["other"], "share": 0.3}]}, "add up to 0.9"),
        ({**FACTS, "flow_split": [{"types": ["mrt"], "share": 0.5}, {"types": ["mrt", "other"], "share": 0.5}]}, "in two groups"),
        ({**FACTS, "flow_split": [{"types": ["kiosk"], "share": 1.0}]}, "each group names"),
        ({"courier_access": [1, 2]}, "no site"),
        ({**FACTS, "zones": {}}, "unknown keys"),
        ({**FACTS, "crowd_density": {"density": 0.3, "area_m2": 100.0}}, "needs exactly density, area_m2, mean_time_s"),
        ({**FACTS, "crowd_density": {"density": 0.3, "area_m2": 100.0, "mean_time_s": 0.0}}, "must be > 0"),
    ],
)
def test_a_site_config_refuses_what_is_no_site_fact(doc: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        infra.SiteConfig.parse(doc)


def test_a_site_config_reads_entrances_outlets_and_the_flow_split(tmp_path: Path) -> None:
    path = tmp_path / "site.yaml"
    doc = {
        "site": "s",
        "scenario": "crowd",
        "exclude": ["ramp"],
        "entrances": [{"name": "main", "type": "building", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]], "recipients": True}, {"name": "lift_a", "type": "lift", "at": [5, 5], "level": "1"}],
        "courier_access": [1, 2],
        "outlets": [[3, 4]],
        "flow_split": [{"types": ["building"], "share": 0.7}, {"types": ["lift"], "share": 0.3}],
    }
    path.write_text(yaml.safe_dump(doc))
    config = infra.SiteConfig.load(path)
    assert (config.scenario, config.exclude, config.path) == ("crowd", ("ramp",), path)
    main, lift = config.facts.entrances
    assert (main.recipients, main.anchor, main.level, main.placeholder) == (True, (1.0, 1.0), None, False)
    assert (lift.recipients, lift.anchor, lift.level) == (False, (5.0, 5.0), "1")
    assert config.facts.outlets == ((3.0, 4.0),) and config.facts.courier_access == (1.0, 2.0)
    assert config.placements == {"loading_point": search.Placement(pin=(1.0, 2.0))}
    assert infra.SiteConfig.parse({**FACTS}).flow_split == ()


def test_keep_out_and_the_visitor_sources_read_the_locker_visitors_too(with_visitors: Path) -> None:
    scenario = with_visitors / "scenarios" / "mix"
    assert len(infra.keep_out(scenario / "scenario.yaml")) == 3
    assert infra.visitor_sources(scenario) == ["near_locker"]
    assert infra.visitor_sources(HOSPITAL / "scenarios") == []


def test_layout_copies_get_the_locker_visitors_aimed_at_their_own_locker(with_visitors: Path) -> None:
    import shapely

    site = open_site(with_visitors)
    layout = _layout(site)
    scenario = infra.scenario_dir(site, "mix")
    saved = infra.save_layouts(site, [layout], scenario, visitor_rate_per_h=12.0)
    folder = saved[layout.layout_id] / "scenarios" / "mix"
    assert not (folder / infra.VISITORS).exists() and (folder / "types" / "visitor.yaml").is_file()
    text = (folder / "scenario.yaml").read_text()
    assert text.startswith(infra.MARKER + ": locker visitors of locker_visitors.yaml merged in") and "# the mix crowd\n# of the copy\n" in text
    doc = yaml.safe_load(text)
    assert set(doc["regions"]) == {"near", "far", "talk", "near_locker"}
    # the site's through-walker regions stay as they were, the visitors scaled from 6 to 12 per hour
    original = yaml.safe_load((scenario / "scenario.yaml").read_text())
    assert all(doc["regions"][name] == region for name, region in original["regions"].items())
    assert doc["regions"]["near_locker"]["config"]["rate_profile"] == [{"t": 0.0, "rate": 0.002}]
    (target,) = [o for o in doc["objects"] if o["name"] == "locker_visit"]
    locker = next(o for s in layout.spots if s.element.name == "locker" for o in s.objects)
    x, y, theta = target["pose"]
    assert target["model"] == "" and target["humansim"]["capacity"] == 1
    assert (x, y) == pytest.approx((locker.footprint.centroid.x, locker.footprint.centroid.y), abs=1e-3)
    # the first visitor stands in front of the locker, 0.35 m clear of its footprint, on the side of its dock
    front = target["humansim"]["formation"]["params"]["front_offset"]
    first = shapely.Point(x - front * math.cos(theta), y - front * math.sin(theta))
    assert first.distance(locker.footprint) == pytest.approx(0.35, abs=0.02)
    assert first.distance(locker.dock) < locker.footprint.centroid.distance(locker.dock)
    assert target["humansim"]["formation"]["params"]["base_step"] == 0.9
    # a scenario without visitors is copied as it is
    assert infra.crowd(layout, HOSPITAL / "scenarios" / "default") is None


def test_cli_premise_writes_the_two_most_different_layouts_with_the_fleet(hospital_copy: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    site = open_site(hospital_copy)
    loading = next(s for s in candidate_spots(site, "loading_point") if s.valid and s.zone == "central_hallway")
    level = site.world.levels[loading.level]
    base = NavrBase(level, exclude={o.name for o in level.objects})
    monkeypatch.setattr(search.Search, "base", property(lambda self: base))
    out = tmp_path / "suites" / "infra_h1_premise.yaml"
    x, y = loading.objects[0].goto
    args = ["infra", "premise", "--site", str(hospital_copy), "--scenario", "mix", "--loading", f"{x},{y}", "--out", str(out)]
    assert cli.main([*args, "--dry-run"]) == 0 and not out.exists()
    assert cli.main(args) == 0
    doc = yaml.safe_load(out.read_text())
    manifest = json.loads(infra.manifest_path(out).read_text())
    assert [r["role"] for r in manifest["runs"]] == ["narrow", "wide"]
    ids = [r["layout_id"] for r in manifest["runs"]]
    assert [s["name"] for s in doc["stages"]] == [f"hospital_copy__layout_{i}__s{seed}" for i in ids for seed in infra.PREMISE_SEEDS]
    assert doc["references"] == ["unhindered_peds"] and manifest["episodes"] == 12
    printed = capsys.readouterr()
    assert "15 episodes (§10: 15)" in printed.err and "--suite infra_h1_premise --contest" in printed.out


def test_cli_evaluation_writes_the_two_picks_and_the_site_on_fresh_seeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    worlds = {"aaaa1111": str(tmp_path / "site__layout_aaaa1111"), "bbbb2222": str(tmp_path / "site__layout_bbbb2222"), "cccc3333": str(tmp_path / "site__layout_cccc3333")}
    fleet = infra.fleet_config(9.5)
    selection = tmp_path / "infra_h1.layouts.json"
    runs = [{"role": role, "rank": 1, "layout_id": layout_id, "travel_m": 1.0} for role, layout_id in (("robot_only", "aaaa1111"), ("joint", "bbbb2222"), ("random", "cccc3333"))]
    selection.write_text(json.dumps({"site": "site", "site_root": "/x", "scenario": "/x/scenarios/study_mix_v2", "seeds": [1, 2, 3, 4, 5], "episode_s": 1500.0, "warmup_s": 420.0, "rate_per_h": 9.5, "fleet": fleet, "contest": "[{name: fleet_dwb}]", "runs": runs, "worlds": worlds}))
    pick = tmp_path / "pick.json"
    pick.write_text(json.dumps({"joint_pick": "bbbb2222", "robot_only": "aaaa1111"}))
    out = tmp_path / "suites" / "infra_h1_eval.yaml"
    args = ["infra", "evaluation", "--pick", str(pick), "--selection", str(selection), "--out", str(out)]
    assert cli.main(args) == 0
    doc = yaml.safe_load(out.read_text())
    site = yaml.safe_load(out.with_name("infra_h1_eval_site.yaml").read_text())
    assert [s["name"] for s in doc["stages"]] == [f"site__layout_{i}__s{seed}" for i in ("bbbb2222", "aaaa1111") for seed in infra.EVALUATION_SEEDS]
    assert all(s["config"] == {"scenario": {"file": "study_mix_v2"}, "fleet": fleet} for s in doc["stages"])
    assert doc["references"] == ["unhindered_peds"] and site["references"] is False
    assert [s["name"] for s in site["stages"]] == [f"site__s{seed}" for seed in infra.EVALUATION_SEEDS]
    manifest = json.loads(infra.manifest_path(out).read_text())
    assert manifest["picks"] == {"joint": "bbbb2222", "robot_only": "aaaa1111"} and manifest["episodes"] == 50
    assert "--suite infra_h1_eval --contest '[{name: fleet_dwb}]'" in capsys.readouterr().out
    # the picks must be selection layouts, and the evaluation seeds fresh
    pick.write_text(json.dumps({"joint_pick": "dddd4444", "robot_only": "aaaa1111"}))
    assert cli.main(args) == 1
    pick.write_text(json.dumps({"joint_pick": "bbbb2222", "robot_only": "aaaa1111"}))
    assert cli.main([*args, "--seeds", "5-12", "--force"]) == 1


# --- the crowd from the entrances ---------------------------------------------------------------------------------------


def _entrance_config(**extra: object) -> infra.SiteConfig:
    doc = {
        "site": "s",
        "courier_access": [1, 2],
        "entrances": [
            {"name": "m1", "type": "mrt", "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]},
            {"name": "m2", "type": "mrt", "at": [20, 1], "recipients": True},
            {"name": "b1", "type": "building", "at": [10, 9]},
        ],
        "flow_split": [{"types": ["mrt"], "share": 0.5}, {"types": ["building", "shop"], "share": 0.3}, {"types": ["other"], "share": 0.2}],
        **extra,
    }
    return infra.SiteConfig.parse(doc)


def test_entrance_shares_split_by_type_group_and_spread_the_empty_groups_share() -> None:
    shares = infra.entrance_shares(_entrance_config())
    assert shares == pytest.approx({"m1": 0.3125, "m2": 0.3125, "b1": 0.375})  # 0.5 and 0.3 of 0.8, the mrt share halved
    assert sum(shares.values()) == pytest.approx(1.0)
    loose = infra.SiteConfig.parse({**FACTS, "entrances": [{"name": "l", "type": "lift", "at": [0, 0]}], "flow_split": [{"types": ["mrt"], "share": 1.0}]})
    with pytest.raises(ValueError, match="no flow_split group"):
        infra.entrance_shares(loose)
    even = infra.SiteConfig.parse({**FACTS, "entrances": [{"name": "a", "type": "mrt", "at": [0, 0]}, {"name": "b", "type": "shop", "at": [5, 0]}]})
    assert infra.entrance_shares(even) == {"a": 0.5, "b": 0.5}


def test_the_entrance_crowd_makes_every_entrance_a_source_and_sink_at_the_total_rate() -> None:
    import shapely

    config = _entrance_config()
    floor = shapely.box(-5, -5, 30, 15)
    scenario, visitors_doc, types = infra.entrance_crowd(config, floor, rate=1.2, visitor_rate_per_h=12.0, robots=[{"start": [1, 1, 0], "phases": []}])
    regions = scenario["regions"]
    sources = {n: r for n, r in regions.items() if r["type"] == "source"}
    assert set(sources) == {f"{e}_{g}" for e in ("m1", "m2", "b1") for g, _, _ in infra.GROUPS}
    assert {n for n, r in regions.items() if r["type"] == "sink"} == {"m1_sink", "m2_sink", "b1_sink"}
    assert sum(r["config"]["rate_profile"][0]["rate"] for r in sources.values()) == pytest.approx(1.2, abs=1e-4)
    assert sources["b1_wheelchair"]["config"]["rate_profile"][0]["rate"] == pytest.approx(1.2 * 0.375 * 0.15, abs=1e-6)
    assert sources["b1_long_cane"]["config"]["agent"]["agent_type"] == "blind_white_cane"
    # a walker leaves by another entrance, in proportion to its share
    affinity = {a["sink"]: a["weight"] for a in sources["m1_adult"]["config"]["agent"]["sink_affinity"]}
    assert affinity == pytest.approx({"m2_sink": 0.3125 / 0.6875, "b1_sink": 0.375 / 0.6875}, abs=1e-5)
    # a point entrance gets a square cut to the floor kept clear of the walls; a polygon entrance keeps its polygon
    assert regions["m1_sink"]["polygon"] == [{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.0}, {"x": 2.0, "y": 2.0}, {"x": 0.0, "y": 2.0}]
    square = shapely.Polygon([(p["x"], p["y"]) for p in regions["m2_sink"]["polygon"]])
    assert square.area == pytest.approx(infra.ENTRANCE_SQUARE_M**2, abs=0.01) and square.centroid.distance(shapely.Point(20, 1)) < 0.01
    near_wall = infra.entrance_crowd(config, shapely.box(-5, -5, 30, 9.5))[0]["regions"]["b1_sink"]["polygon"]
    assert max(p["y"] for p in near_wall) == pytest.approx(9.5 - infra.ENTRANCE_WALL_M, abs=0.01)
    assert scenario["robots"] == [{"start": [1, 1, 0], "phases": []}]
    # locker visitors from the recipient entrance only, back out by its sink
    assert set(visitors_doc["regions"]) == {f"m2_locker_{g}" for g, _, _ in infra.GROUPS}
    assert sum(r["config"]["rate_profile"][0]["rate"] for r in visitors_doc["regions"].values()) == pytest.approx(12.0 / 3600, abs=1e-6)
    leave = types["locker_visitor__m2_adult.yaml"]["sequences"]["leave"]["steps"]["exit"]["target_pose"]
    assert (leave["x"], leave["y"]) == pytest.approx((20.0, 1.0), abs=0.01)
    assert visitors_doc["regions"]["m2_locker_adult"]["config"]["agent"]["sink_affinity"] == [{"sink": "m2_sink", "weight": 1.0}]


def test_write_crowd_writes_the_scenario_directory_once(tmp_path: Path) -> None:
    import shapely

    scenario, visitors_doc, types = infra.entrance_crowd(_entrance_config(), shapely.box(-5, -5, 30, 15))
    folder = infra.write_crowd(tmp_path / "crowd", scenario, visitors_doc, types, "the crowd\nof the test")
    text = (folder / "scenario.yaml").read_text()
    assert text.startswith("# the crowd\n# of the test\n") and yaml.safe_load(text)["format"] == "arena-scenario/2"
    assert (folder / infra.VISITORS).is_file() and (folder / "types" / "locker_visitor__m2_adult.yaml").is_file()
    assert len(infra.keep_out(folder / "scenario.yaml")) == 4 * 3 + 3  # 3 sources and a sink per entrance, and the visitors
    with pytest.raises(FileExistsError):
        infra.write_crowd(tmp_path / "crowd", scenario, visitors_doc, types, "again")


def test_the_crowd_density_sets_the_inflow_by_littles_law() -> None:
    import argparse

    from arena_simulation_setup.generative.flowedit import infra_cli

    config = _entrance_config(crowd_density={"density": 0.3, "area_m2": 1000.0, "mean_time_s": 200.0})
    cd = config.crowd_density
    assert (cd.present(), cd.rate(), cd.rate(0.6)) == pytest.approx((300.0, 1.5, 3.0))
    assert config.record()["crowd_density"] == {"density": 0.3, "area_m2": 1000.0, "mean_time_s": 200.0}

    def rate(config: infra.SiteConfig, rate: float | None = None, density: float | None = None) -> float:
        return infra_cli._crowd_rate(argparse.Namespace(rate=rate, density=density), config)[0]

    assert rate(config) == pytest.approx(1.5) and rate(config, density=0.45) == pytest.approx(2.25)
    assert rate(config, rate=0.9) == 0.9  # --rate wins
    with pytest.raises(ValueError, match="not both"):
        rate(config, rate=0.9, density=0.3)
    plain = _entrance_config()
    assert plain.crowd_density is None and rate(plain) == infra.CROWD_RATE
    with pytest.raises(ValueError, match="no crowd_density"):
        rate(plain, density=0.3)
    twin = infra.site_config("smu_link_full").crowd_density
    assert twin.density == 0.1266 and twin.rate() == pytest.approx(0.8, abs=1e-3) and twin.present() == pytest.approx(160.0, abs=0.1)


def test_the_crowd_side_goes_from_the_site_config_into_every_suites_launch_block() -> None:
    assert _entrance_config(crowd_side="left").crowd_side == "left"
    assert _entrance_config(crowd_side="left").record()["crowd_side"] == "left"
    with pytest.raises(ValueError, match="crowd_side"):
        _entrance_config(crowd_side="middle")
    assert infra.reference_suite("s", [1], side="left")["launch"]["humansim.global_planner.lane_side"] == "left"
    assert "humansim.global_planner.lane_side" not in infra.reference_suite("s", [1])["launch"]
    assert infra.layout_suite(["w"], infra.fleet_config(10.0), [1], side="right")["launch"]["humansim.global_planner.lane_side"] == "right"
    assert infra.site_config("smu_link_full").crowd_side == "left"

