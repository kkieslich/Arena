from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import cli, entrance_fleet, infra, place, recipients, search
from arena_simulation_setup.generative.flowedit.demand import EntranceSearch
from arena_simulation_setup.generative.flowedit.edit import Site

# the entrance search's test world, and a copy of it with a crowd between the test entrances
from test_generative_flowedit_demand import assets, base, crowd, entrances, hospital, inner, loading, space  # noqa: F401
from test_generative_flowedit_recipients import site_copy  # noqa: F401

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("shapely")

SERVES = {"locker_1": ("a", "b"), "locker_2": ("c",)}


def test_a_seeds_jobs_go_to_the_locker_that_serves_their_entrance() -> None:
    stream = recipients.jobs(4, ["a", "b", "c"], rate_per_h=120.0, episode_s=900.0)
    entries = entrance_fleet.job_entries(stream, SERVES)
    assert len(entries) == len(stream) > 10
    for job, entry in zip(stream, entries, strict=True):
        release, kind, locker, entrance = entry.split()
        assert (float(release), kind, entrance) == (job.release_s, job.kind, job.entrance) and entrance in SERVES[locker]
    # another layout serves the entrances otherwise: the same jobs at the same times, other lockers
    other = entrance_fleet.job_entries(stream, {"locker_1": ("c",), "locker_2": ("a",), "locker_3": ("b",)})
    assert [e.split()[:2] + e.split()[3:] for e in other] == [e.split()[:2] + e.split()[3:] for e in entries] and other != entries
    with pytest.raises(ValueError, match="which no locker serves"):
        entrance_fleet.job_entries(stream, {"locker_1": ("a",)})


def test_the_fleet_block_runs_the_job_list_by_the_rules_of_the_redesign_and_the_first_studys_block_is_as_it_was() -> None:
    block = entrance_fleet.fleet_config(["locker_1", "locker_2"], ["1.000 delivery locker_2 c"], rate_per_h=9.0)
    assert block == {
        "homes": ["bay_1", "bay_2", "charger_1", "charger_2"],
        "loading": "loading_point",
        "lockers": ["locker_1", "locker_2"],
        "jobs": ["1.000 delivery locker_2 c"],
        "rate": 9.0,
        "pickup_share": 0.3,
        "load_s": 15.0,
        "unload_s": 10.0,
        "charge_until": 90.0,
        "reserve": 10.0,
        "charge_rule": "need",
        "idle_floor": 17.0,
        "one_need_charge": True,
        "charge_at_start": True,
        "stop_empty": True,
        "nearest_charger": True,
        "charge_jam": "retry",
        "top_up": False,
    }
    logged = entrance_fleet.fleet_config(["locker_1"], [], rate_per_h=9.0, charge_log="/logs/x.csv", charge_log_s=30.0)
    assert (logged["soc_log"], logged["soc_log_s"]) == ("/logs/x.csv", 30.0)
    with pytest.raises(ValueError, match="at least one locker"):
        entrance_fleet.fleet_config([], [], rate_per_h=9.0)
    # the first study's block names none of the new parameters, so its stages keep the mode's defaults
    assert set(infra.fleet_config(6.46)) == {"homes", "loading", "locker", "rate", "pickup_share", "load_s", "unload_s", "charge_below", "charge_until"}


def test_the_contest_keeps_each_robots_own_battery_and_starts_full_or_from_charge_states() -> None:
    full = entrance_fleet.contest()
    assert full == [{"name": "fleet_dwb", "mobile": infra.PLANNER, "task.episode.battery.soc.min": 100.0, "task.episode.battery.soc.max": 100.0}]
    drawn = entrance_fleet.contest(soc_file="/suites/x.charge_states.json")
    assert drawn == [{"name": "fleet_dwb", "mobile": infra.PLANNER, "task.episode.battery.soc.file": "/suites/x.charge_states.json"}]
    assert entrance_fleet.contest(start_soc=80.0)[0]["task.episode.battery.soc.max"] == 80.0
    # no capacity override in either; the first study's contest scales the battery to 36 Wh as before
    assert not [key for entry in (*full, *drawn) for key in entry if "capacity" in key]
    assert infra.contest()[0]["task.episode.battery.capacity_wh"] == 36.0 and yaml.safe_load(infra.contest_literal(full)) == full
    with pytest.raises(ValueError, match="0 to 100"):
        entrance_fleet.contest(start_soc=120.0)


def test_a_scenario_without_its_crowd_keeps_everything_but_the_sources() -> None:
    doc = {"format": "arena-scenario/2", "robots": [{"start": [1, 2, 0]}], "regions": {"in": {"type": "source", "polygon": []}, "out": {"type": "sink", "polygon": []}, "talk": {"type": "area"}}, "teleport_zones": [{"name": "stairs"}]}
    bare = entrance_fleet.without_crowd(doc)
    assert list(bare["regions"]) == ["out", "talk"] and {k: v for k, v in bare.items() if k != "regions"} == {k: v for k, v in doc.items() if k != "regions"}
    assert list(doc["regions"]) == ["in", "out", "talk"]


def _log(path: Path, rows: list[str], robots: tuple[str, ...] = ("r0", "r1", "r2", "r3")) -> Path:
    path.write_text("\n".join([",".join(["t_s", *(f"soc:{r}" for r in robots), *(f"home:{r}" for r in robots)]), *rows]) + "\n")
    return path


def test_charge_states_are_the_logged_charges_after_the_run_in_in_the_order_of_the_start_spots(tmp_path: Path) -> None:
    homes = "bay_1,bay_2,charger_1,charger_2"
    log = _log(
        tmp_path / "a.csv",
        [
            f"0.001,90.000,80.000,70.000,60.000,{homes}",
            f"60.001,89.000,79.000,71.000,61.000,{homes}",
            f"120.001,88.000,78.000,72.000,62.000,{homes}",
            # r0 and r2 have swapped homes: the charge on bay_1 is r2's now, the one on charger_1 r0's
            "180.001,87.000,77.000,73.000,63.000,charger_1,bay_2,bay_1,charger_2",
            f"240.001,86.000,,74.000,64.000,{homes}",
            f"300.001,85.000,75.000,75.000,65.000,{homes}",
        ],
    )
    episodes = entrance_fleet.read_charge_log(log)
    assert len(episodes) == 1 and episodes[0]["robots"] == ["r0", "r1", "r2", "r3"] and len(episodes[0]["rows"]) == 6
    assert episodes[0]["rows"][4] == (240.001, [86.0, None, 74.0, 64.0], ["bay_1", "bay_2", "charger_1", "charger_2"])
    doc = entrance_fleet.charge_states([log], from_s=100.0, every_s=60.0)
    # from 100 s on, a state per minute; the row with an unknown charge is left out
    assert doc["states"] == [[88.0, 78.0, 72.0, 62.0], [73.0, 77.0, 87.0, 63.0], [85.0, 75.0, 75.0, 65.0]]
    assert doc["format"] == "infra-charge-states/1" and doc["spots"] == ["bay_1", "bay_2", "charger_1", "charger_2"]
    assert doc["source"] == {"logs": [str(log)], "from_s": 100.0, "every_s": 60.0, "rows": 6, "skipped": 1}
    # states further apart; two episodes in one log and a second log add up
    # at the marks 0, 150, 300 s: the first row at or after each
    assert entrance_fleet.charge_states([log], from_s=0.0, every_s=150.0)["states"] == [[90.0, 80.0, 70.0, 60.0], [73.0, 77.0, 87.0, 63.0], [85.0, 75.0, 75.0, 65.0]]
    log.write_text(log.read_text() * 2)
    assert len(entrance_fleet.read_charge_log(log)) == 2 and len(entrance_fleet.charge_states([log, log], from_s=100.0, every_s=60.0)["states"]) == 12
    # the file is what the runtime reads, and is checked as it checks it
    out = tmp_path / "states.json"
    out.write_text(json.dumps(doc))
    assert entrance_fleet.load_charge_states(out)["states"] == doc["states"]
    out.write_text(json.dumps({**doc, "states": [[90.0], [80.0, 70.0]]}))
    with pytest.raises(ValueError, match="the same number of charges"):
        entrance_fleet.load_charge_states(out)
    out.write_text(json.dumps({"states": doc["states"]}))
    with pytest.raises(ValueError, match="no charge states"):
        entrance_fleet.load_charge_states(out)
    with pytest.raises(ValueError, match="no charge state from 1000 s on"):
        entrance_fleet.charge_states([log], from_s=1000.0)
    with pytest.raises(ValueError, match="fits no header"):
        entrance_fleet.read_charge_log(_log(tmp_path / "b.csv", ["1.0,90.0"]))
    other = _log(tmp_path / "c.csv", ["0.001,90.000,80.000,70.000,60.000,bay_2,bay_1,charger_1,charger_2"])
    with pytest.raises(ValueError, match="an episode starts on"):
        entrance_fleet.charge_states([log, other], from_s=0.0)


def _searched(site: Site, space: EntranceSearch, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Path, dict]:  # noqa: F811
    """A search report on the site copy: the arguments that name the site, the report and its travel-optimal layout."""
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    spot = next(s for s in place.candidate_spots(site, "loading_point") if s.valid and s.zone == "central_hallway")
    level = site.world.levels[spot.level]
    navr = NavrBase(level, exclude={o.name for o in level.objects})
    monkeypatch.setattr(search.Search, "base", property(lambda self: navr))
    x, y = spot.objects[0].goto
    config = tmp_path / "site.yaml"
    config.write_text(yaml.safe_dump({"site": site.key, "scenario": "mix", "courier_access": [x, y], "entrances": [{"name": e.name, "type": e.type, "at": list(e.anchor)} for e in space.entrances]}))
    report = tmp_path / "report.json"
    common = ["--site", str(site.root), "--site-config", str(config)]
    assert cli.main(["infra", "search", *common, "--reach", "12", "--margins", "1.5", "--rate", "6", "--rank", "walk", "--out", str(report)]) == 0
    optimal = next(r for r in json.loads(report.read_text())["reaches"][0]["layouts"] if r["role"] == "travel_optimal")
    return common, report, optimal


def test_cli_evaluation_writes_the_fleet_stages_with_the_seeds_jobs_and_a_fleet_only_run(space: EntranceSearch, site_copy: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    site = site_copy
    common, report, optimal = _searched(site, space, tmp_path, monkeypatch)
    layout_id = optimal["layout_id"]
    world = f"{site.key}__layout_{layout_id}"
    out = tmp_path / "suites" / "infra_rd.yaml"
    args = ["infra", "evaluation", *common, "--report", str(report), "--layouts", f"travel_optimal={layout_id}", "--seeds", "71-73", "--without-robots-seeds", "71", "--rate", "240", "--recipient-offset", "60", "--no-check", "--out", str(out)]
    capsys.readouterr()
    assert cli.main([*args, "--arms", "site,without,fleet", "--seed-part", "h1=71,72", "--seed-part", "h2=73"]) == 0
    printed = capsys.readouterr()
    doc = yaml.safe_load(out.read_text())
    without = yaml.safe_load(out.with_name("infra_rd_without.yaml").read_text())
    site_doc = yaml.safe_load(out.with_name("infra_rd_site.yaml").read_text())
    assert [s["name"] for s in doc["stages"]] == [f"{world}__s{seed}" for seed in (71, 72, 73)]
    assert doc["references"] is False and doc["launch"] == site_doc["launch"] and doc["record"] == site_doc["record"]
    names = sorted(e.name for e in space.entrances)
    for stage in doc["stages"]:
        seed = stage["seed"]
        fleet = stage["config"]["fleet"]
        # the seed's jobs, each to the locker that serves its entrance; the stage runs the scenario with the seed's
        # recipients, the one its run without robots runs
        stream = recipients.jobs(seed, names, rate_per_h=240.0, episode_s=1500.0)
        assert fleet["jobs"] == entrance_fleet.job_entries(stream, optimal["serves"]) and fleet["lockers"] == list(optimal["serves"])
        assert fleet == entrance_fleet.fleet_config(list(optimal["serves"]), fleet["jobs"], rate_per_h=240.0)
        assert stage["modes"] == {"robots": "fleet", "obstacles": "scenario"} and stage["config"]["scenario"] == {"file": f"mix__s{seed}"} and stage["timeout"] == 1500.0
    assert without["stages"][0]["config"]["scenario"] == doc["stages"][0]["config"]["scenario"] and without["stages"][0]["name"] == doc["stages"][0]["name"]
    manifest = json.loads(infra.manifest_path(out).read_text())
    contest = entrance_fleet.contest()
    assert manifest["suite"] == "infra_rd" and yaml.safe_load(manifest["contest"]) == contest and manifest["charge_states"] is None and manifest["start_charge"] == 100.0
    assert manifest["fleet"]["charge_rule"] == "need" and "jobs" not in manifest["fleet"] and manifest["fleet_only"] is False
    # with the fleet's runs written the seeds of the runs without robots are also named for the analysis
    assert manifest["episodes"] == 3 + 1 + 3 and manifest["without_robots_seeds"] == [71] == manifest["crowd_only_seeds"]
    assert [(p["tag"], p["suite"], p["without_suite"], p["episodes"]) for p in manifest["seed_parts"]] == [("h1", "infra_rd_h1", "infra_rd_without_h1", 2 + 1 + 2), ("h2", "infra_rd_h2", None, 2)]
    assert [s["seed"] for s in yaml.safe_load(out.with_name("infra_rd_h1.yaml").read_text())["stages"]] == [71, 72]
    assert f"--suite infra_rd --contest '{manifest['contest']}'" in printed.out and "task.episode.battery.capacity_wh" not in printed.out
    # the start charge from charge states: the file is copied beside the suite and the contest names the copy
    states = tmp_path / "states.json"
    states.write_text(json.dumps({"format": "infra-charge-states/1", "states": [[90.0, 80.0, 70.0, 60.0], [50.0, 40.0, 30.0, 20.0]]}))
    drawn = out.with_name("infra_rdc.yaml")
    assert cli.main([*args[:-1], str(drawn), "--arms", "fleet", "--charge-states", str(states)]) == 0
    copy = drawn.with_name("infra_rdc.charge_states.json")
    manifest = json.loads(infra.manifest_path(drawn).read_text())
    assert json.loads(copy.read_text())["states"] == [[90.0, 80.0, 70.0, 60.0], [50.0, 40.0, 30.0, 20.0]] and manifest["charge_states"] == str(copy)
    assert yaml.safe_load(manifest["contest"]) == entrance_fleet.contest(soc_file=str(copy)) and manifest["site_suite"] is None and manifest["without_suite"] is None
    assert not drawn.with_name("infra_rdc_site.yaml").exists() and manifest["episodes"] == 3 and manifest["without_robots_seeds"] == []
    # a fleet-only run: no walkers, the same jobs, the robots' charge logged per stage
    only = out.with_name("infra_rdf.yaml")
    assert cli.main([*args[:-1], str(only), "--fleet-only", "--charge-log", "/logs/rdf", "--duration", "7200"]) == 0
    doc = yaml.safe_load(only.read_text())
    assert [s["name"] for s in doc["stages"]] == [f"{world}__s{seed}" for seed in (71, 72, 73)] and not only.with_name("infra_rdf_site.yaml").exists()
    for stage in doc["stages"]:
        assert stage["config"]["scenario"] == {"file": "mix__fleet_only"} and stage["timeout"] == 7200.0
        assert stage["config"]["fleet"]["soc_log"] == f"/logs/rdf/{stage['name']}.csv" and stage["config"]["fleet"]["soc_log_s"] == 60.0
        assert stage["config"]["fleet"]["jobs"] == entrance_fleet.job_entries(recipients.jobs(stage["seed"], names, rate_per_h=240.0, episode_s=7200.0), optimal["serves"])
    manifest = json.loads(infra.manifest_path(only).read_text())
    bare = yaml.safe_load((Path(manifest["worlds"][layout_id]) / "scenarios" / "mix__fleet_only" / "scenario.yaml").read_text())
    assert not [r for r in bare["regions"].values() if r["type"] == "source"] and len(bare["regions"]) == len(space.entrances) and "objects" not in bare
    assert manifest["fleet_only"] is True and manifest["visitor_sources"] == [] and manifest["episodes"] == 3 and "without_robots_seeds" not in manifest
    # what does not go together
    assert cli.main([*args, "--arms", "fleet,robots"]) == 1 and cli.main([*args, "--charge-log", "/logs/x"]) == 1
    assert cli.main([*args[:-1], str(out.with_name("infra_rdx.yaml")), "--arms", "fleet", "--charge-states", str(tmp_path / "none.json")]) == 1


def test_cli_charge_states_writes_the_file_the_start_charge_is_drawn_from(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    homes = "bay_1,bay_2,charger_1,charger_2"
    # two robots run down by 1 % per 100 s, two charge at the same rate until they are full (the log's charge is capped)
    log = _log(tmp_path / "run.csv", [f"{t}.001,{90 - t / 100:.3f},{80 - t / 100:.3f},{min(100.0, 70 + t / 100):.3f},{min(100.0, 60 + t / 100):.3f},{homes}" for t in range(0, 7200, 60)])
    out = tmp_path / "states.json"
    assert cli.main(["infra", "charge-states", "--log", str(log), "--from-s", "3600", "--every-s", "600", "--out", str(out)]) == 0
    doc = entrance_fleet.load_charge_states(out)
    assert len(doc["states"]) == 6 and doc["states"][0] == [54.0, 44.0, 100.0, 96.0] and doc["spots"] == homes.split(",")
    assert "6 charge states of 4 robots" in capsys.readouterr().err
    assert cli.main(["infra", "charge-states", "--log", str(log), "--out", str(out)]) == 1
    assert cli.main(["infra", "charge-states", "--log", str(log), "--from-s", "99999", "--out", str(out), "--force"]) == 1


def test_charge_states_are_taken_at_whole_marks_and_do_not_drift(tmp_path: Path) -> None:
    import random

    # rows at the first tick after each 60 s mark, as the fleet mode writes them (the reviewer's probe)
    rng = random.Random(3)
    homes = "bay_1,bay_2,charger_1,charger_2"
    rows = []
    for k in range(14400 // 60 + 1):
        t = k * 60 + rng.uniform(0.0, 0.1)
        rows.append(f"{t:.3f}," + ",".join(f"{100 - t / 300:.3f}" for _ in range(4)) + f",{homes}")
    doc = entrance_fleet.charge_states([_log(tmp_path / "a.csv", rows)], from_s=3600.0, every_s=600.0)
    # 3600, 4200, ..., 14400 s: 19 states, each at its mark (to the tick)
    assert len(doc["states"]) == 19
    assert [round((100.0 - s[0]) * 300.0) for s in doc["states"]] == [3600 + 600 * k for k in range(19)]
