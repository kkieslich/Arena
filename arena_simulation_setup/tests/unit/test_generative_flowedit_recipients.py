from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import cli, demand, infra, place, recipients, search
from arena_simulation_setup.generative.flowedit import suite as SU
from arena_simulation_setup.generative.flowedit.demand import EntranceSearch
from arena_simulation_setup.generative.flowedit.edit import Site, open_site
from arena_simulation_setup.generative.flowedit.search import Search

# the entrance search's test world and search space
from test_generative_flowedit_demand import HOSPITAL, _reach, assets, base, crowd, entrances, hospital, inner, loading, space  # noqa: F401

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
shapely = pytest.importorskip("shapely")

CHARGERS, BAYS, LOCKER = search.SEARCHED
MIX = {"adult": ("adult", 0.7), "wheelchair": ("wheelchair_manual", 0.2), "long_cane": ("blind_white_cane", 0.1)}


def _square(x: float, y: float, half: float = 0.4) -> list[dict[str, float]]:
    return [{"x": x - half, "y": y - half}, {"x": x + half, "y": y - half}, {"x": x + half, "y": y + half}, {"x": x - half, "y": y + half}]


def _scenario(names_at: dict[str, tuple[float, float]]) -> dict:
    """A crowd between the entrances `names_at`: per entrance a sink and a source per walker group, 70/20/10."""
    regions: dict = {}
    for name, (x, y) in names_at.items():
        for group, (agent_type, share) in MIX.items():
            regions[f"{name}_{group}"] = {"type": "source", "polygon": _square(x, y), "config": {"rate_profile": [{"t": 0.0, "rate": share}], "max_concurrent": -1, "max_total": -1, "agent": {"agent_type": agent_type, "sink_affinity": [{"sink": f"{other}_sink", "weight": 1.0} for other in names_at if other != name]}}}
    for name, (x, y) in names_at.items():
        regions[f"{name}_sink"] = {"type": "sink", "polygon": _square(x, y), "config": {"capacity": -1}}
    return {"format": "arena-scenario/2", "regions": regions}


def test_the_jobs_of_a_seed_come_from_the_seed_alone() -> None:
    names = ["a", "b", "c", "d"]
    first = recipients.jobs(71, names, rate_per_h=60.0, episode_s=3600.0)
    assert first == recipients.jobs(71, names, rate_per_h=60.0, episode_s=3600.0) != recipients.jobs(72, names, rate_per_h=60.0, episode_s=3600.0)
    assert [j.index for j in first] == list(range(len(first))) and all(a.release_s < b.release_s for a, b in zip(first, first[1:], strict=False))
    assert all(0.0 < j.release_s < 3600.0 and j.kind in ("delivery", "pickup") and j.entrance in names for j in first)
    # a shorter episode cuts the same list, another rate moves the same jobs in time
    assert recipients.jobs(71, names, rate_per_h=60.0, episode_s=900.0) == tuple(j for j in first if j.release_s < 900.0)
    twice = recipients.jobs(71, names, rate_per_h=120.0, episode_s=1800.0)
    assert [(j.kind, j.entrance) for j in twice] == [(j.kind, j.entrance) for j in first][: len(twice)] and len(twice) >= len(first) - 1
    assert all(j.release_s == pytest.approx(f.release_s / 2.0, abs=1e-3) for j, f in zip(twice, first, strict=False))
    # another pickup share changes kinds only
    more = recipients.jobs(71, names, rate_per_h=60.0, pickup_share=0.9, episode_s=3600.0)
    assert [(j.release_s, j.entrance) for j in more] == [(j.release_s, j.entrance) for j in first] and sum(j.kind == "pickup" for j in more) > sum(j.kind == "pickup" for j in first)
    for bad in ({"rate_per_h": 0.0}, {"rate_per_h": 6.0, "pickup_share": 1.5}, {"rate_per_h": 6.0, "episode_s": 0.0}):
        with pytest.raises(ValueError, match="must be positive"):
            recipients.jobs(1, names, **bad)
    with pytest.raises(ValueError, match="no recipient entrances"):
        recipients.jobs(1, [], rate_per_h=6.0)


def test_the_job_stream_has_the_rate_the_pickup_share_and_equal_demand_per_entrance() -> None:
    names = ["a", "b", "c", "d"]
    stream = [j for seed in range(200) for j in recipients.jobs(seed, names, rate_per_h=40.0, pickup_share=0.3, episode_s=3600.0)]
    assert len(stream) / 200 == pytest.approx(40.0, rel=0.05)
    assert sum(j.kind == "pickup" for j in stream) / len(stream) == pytest.approx(0.3, abs=0.02)
    assert [sum(j.entrance == name for j in stream) / len(stream) for name in names] == pytest.approx([0.25] * 4, abs=0.02)


def test_each_delivery_brings_one_recipient_at_its_entrance_after_the_offset() -> None:
    names = ["a", "b", "c", "d"]
    stream = recipients.jobs(5, names, rate_per_h=120.0, episode_s=1500.0)
    serves = {"locker_1": ("a", "b"), "locker_2": ("c",), "locker_3": ("d",)}
    who = recipients.recipients(5, stream, serves, MIX, offset_s=420.0, episode_s=1500.0, kinds=("delivery",))
    due = [j for j in stream if j.kind == "delivery" and round(round((j.release_s + 420.0) / 0.1) * 0.1, 3) < 1500.0]
    # by default every job brings one (FLEET_DECISIONS 7): pickups too
    assert recipients.RECIPIENT_KINDS == ("delivery", "pickup") and [r.job for r in recipients.recipients(5, stream, serves, MIX, offset_s=420.0, episode_s=1500.0)] == [j.index for j in stream if round(round((j.release_s + 420.0) / 0.1) * 0.1, 3) < 1500.0]
    assert [r.job for r in who] == [j.index for j in due] and len(who) < len(stream)
    by_job = {j.index: j for j in stream}
    for r in who:
        job = by_job[r.job]
        assert r.entrance == job.entrance and r.entrance in serves[r.locker] and r.agent_type == MIX[r.group][0]
        assert r.t_s == pytest.approx(job.release_s + 420.0, abs=0.05 + 1e-9) and round(r.t_s * 10) == pytest.approx(r.t_s * 10) and r.t_s < 1500.0
        assert r.source == f"recipient_{r.job:03d}__{r.entrance}_locker_{r.group}"
    # every job: the pickups' too
    assert len(recipients.recipients(5, stream, serves, MIX, offset_s=0.0, episode_s=1500.0, kinds=("delivery", "pickup"))) == len(stream)
    # another layout, whose lockers serve the entrances otherwise, has the same recipients but for their locker
    other = recipients.recipients(5, stream, {"locker_1": ("d", "c", "b"), "locker_2": ("a",)}, MIX, offset_s=420.0, episode_s=1500.0, kinds=("delivery",))
    assert [(r.job, r.entrance, r.group, r.t_s, r.source) for r in other] == [(r.job, r.entrance, r.group, r.t_s, r.source) for r in who]
    # and a recipient's group does not depend on which other jobs there are
    half = recipients.recipients(5, stream[::2], serves, MIX, offset_s=420.0, episode_s=1500.0, kinds=("delivery",))
    assert {r.job: r.group for r in half}.items() <= {r.job: r.group for r in who}.items()
    with pytest.raises(ValueError, match="which no locker serves"):
        recipients.recipients(5, stream, {"locker_1": ("a",)}, MIX)
    with pytest.raises(ValueError, match="positive shares"):
        recipients.recipients(5, stream, serves, {})


def test_recipients_are_drawn_with_the_crowds_mix() -> None:
    names = ["a", "b"]
    serves = {"locker_1": ("a", "b")}
    groups = [r.group for seed in range(300) for r in recipients.recipients(seed, recipients.jobs(seed, names, rate_per_h=60.0, pickup_share=0.0, episode_s=3600.0), serves, MIX, offset_s=0.0, episode_s=3600.0)]
    assert len(groups) > 15_000
    assert [groups.count(g) / len(groups) for g in MIX] == pytest.approx([0.7, 0.2, 0.1], abs=0.015)


def test_the_crowds_mix_is_read_from_the_scenarios_sources() -> None:
    doc = _scenario({"ward": (1.0, 1.0), "lift": (5.0, 1.0)})
    mix = recipients.crowd_mix(doc)
    assert {g: t for g, (t, _) in mix.items()} == {"adult": "adult", "wheelchair": "wheelchair_manual", "long_cane": "blind_white_cane"}
    assert [share for _, share in mix.values()] == pytest.approx([0.7, 0.2, 0.1])
    # a group without inflow is none of the crowd's
    for name in ("ward_long_cane", "lift_long_cane"):
        doc["regions"][name]["config"]["rate_profile"][0]["rate"] = 0.0
    assert list(recipients.crowd_mix(doc)) == ["adult", "wheelchair"] and recipients.crowd_mix(doc)["adult"][1] == pytest.approx(0.7 / 0.9)
    doc["regions"]["ward_adult"]["config"]["agent"]["agent_type"] = "./types/visitor.yaml"
    with pytest.raises(ValueError, match="no walker group's"):
        recipients.crowd_mix(doc)
    with pytest.raises(ValueError, match="no inflow"):
        recipients.crowd_mix({"regions": {}})


def _picked(space: EntranceSearch, count: int) -> tuple[demand.Placed, dict]:
    pick = demand.travel_optimal(space, _reach(space, count))
    return pick, json.loads(json.dumps(space.record(pick)))


@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_layout_is_read_back_from_its_record(space: EntranceSearch, count: int) -> None:
    pick, record = _picked(space, count)
    report = {"reaches": [{"reach_m": 1.0, "layouts": [{"role": "travel_optimal", **record}, {"role": "practice_based", **record}]}]}
    records = recipients.layout_records(report)
    assert list(records) == [pick.layout_id]
    layout = recipients.layout_of(space.search, records[pick.layout_id])
    assert layout.layout_id == pick.layout_id and [s.id for s in layout.spots] == [s.id for s in pick.layout.spots]
    assert [(o.name, o.x, o.y, o.yaw) for o in layout.objects] == [(o.name, o.x, o.y, o.yaw) for o in pick.layout.objects]
    serves = recipients.serves_of(record, layout)
    assert list(serves) == [f"locker_{i + 1}" for i in range(count)]
    assert sorted(e for names in serves.values() for e in names) == sorted(e.name for e in space.entrances)
    assert not place.check_layout(space.search.site, layout)


def test_a_record_that_does_not_fit_the_search_space_is_refused(space: EntranceSearch) -> None:
    pick, record = _picked(space, 2)
    moved = json.loads(json.dumps(record))
    moved["spots"][2]["spot"] = "00000000"
    with pytest.raises(ValueError, match="the search space has no locker spot 00000000"):
        recipients.layout_of(space.search, moved)
    other = json.loads(json.dumps(record))
    other["layout_id"] = "ffffffff"
    with pytest.raises(ValueError, match="give layout"):
        recipients.layout_of(space.search, other)
    with pytest.raises(ValueError, match="reported twice"):
        recipients.layout_records([record, {**record, "serves": {"locker_1": ["door_0"]}}])
    twice = {**record, "serves": {"locker_1": ["door_0", "door_1"], "locker_2": ["door_1"]}}
    with pytest.raises(ValueError, match="served by two lockers"):
        recipients.serves_of(twice)
    with pytest.raises(ValueError, match="no lockers of the layout"):
        recipients.serves_of({**record, "serves": {"locker_9": ["door_0"]}}, pick.layout)


def _crowd(space: EntranceSearch, count: int, seed: int = 3, rate: float = 240.0) -> tuple[demand.Placed, dict, dict, tuple[recipients.Recipient, ...]]:
    pick, record = _picked(space, count)
    doc = _scenario({e.name: e.anchor for e in space.entrances})
    serves = recipients.serves_of(record, pick.layout)
    who = recipients.recipients(seed, recipients.jobs(seed, sorted(e.name for e in space.entrances), rate_per_h=rate, episode_s=1500.0), serves, MIX, offset_s=60.0, episode_s=1500.0)
    return pick, doc, serves, who


def test_the_crowd_of_a_layout_and_seed_has_a_target_per_locker_and_a_source_per_recipient(space: EntranceSearch) -> None:
    pick, doc, serves, who = _crowd(space, 3)
    assert len(who) > 20 and {r.group for r in who} == set(MIX)
    out, types, header = recipients.recipient_crowd(doc, pick.layout, serves, who, MIX)
    # the site's regions stay as they are; each recipient is one more source that releases one walker at its time
    assert all(out["regions"][name] == region for name, region in doc["regions"].items())
    assert set(out["regions"]) - set(doc["regions"]) == {r.source for r in who} and len({r.source for r in who}) == len(who)
    for r in who:
        region = out["regions"][r.source]
        config = region["config"]
        assert region["type"] == "source" and region["polygon"] == doc["regions"][f"{r.entrance}_sink"]["polygon"]
        assert config["max_total"] == 1 and config["rate_profile"] == [{"t": r.t_s, "rate": 0.0}, {"t": round(r.t_s + 0.05, 3), "rate": 2000.0}]
        assert config["agent"] == {"agent_type": f"./types/recipient__{r.entrance}_{r.group}.yaml", "sink_affinity": [{"sink": f"{r.entrance}_sink", "weight": 1.0}]}
    # a target in front of every locker: the first in line 0.35 m clear of its footprint, on the side of its dock
    targets = {o["name"]: o for o in out["objects"]}
    assert list(targets) == ["locker_visit", "locker_visit_2", "locker_visit_3"]
    lockers = {o.name: o for s in pick.layout.spots if s.element.name.startswith(LOCKER) for o in s.objects}
    for locker, target in zip(("locker_1", "locker_2", "locker_3"), targets.values(), strict=True):
        x, y, theta = target["pose"]
        placed = lockers[locker]
        assert target["model"] == "" and target["humansim"]["type"] == "parcel_locker" and target["humansim"]["capacity"] == 1
        assert (x, y) == pytest.approx((placed.footprint.centroid.x, placed.footprint.centroid.y), abs=1e-3)
        front = target["humansim"]["formation"]["params"]["front_offset"]
        first = shapely.Point(x - front * math.cos(theta), y - front * math.sin(theta))
        assert first.distance(placed.footprint) == pytest.approx(0.35, abs=0.02) and first.distance(placed.dock) < placed.footprint.centroid.distance(placed.dock)
        assert target["humansim"]["formation"]["params"]["base_step"] == 0.9
    # an agent type per served entrance and group: the group's type, bound for the locker that serves the entrance,
    # standing there as the first study's visitors did, then back to the entrance's area
    assert set(types) == {f"recipient__{e.name}_{g}.yaml" for e in space.entrances for g in MIX}
    for locker, names in serves.items():
        for name in names:
            for group, (agent_type, _) in MIX.items():
                body = types[f"recipient__{name}_{group}.yaml"]
                step = body["sequences"]["visit"]["steps"]["pickup"]
                assert body["extends"] == agent_type and body["mode"] == "behavior_tree" and body["initial_sequence"] == "visit"
                assert step["target"] == recipients.target_name(locker) and step["interaction"] == "USE"
                assert step["duration"] == {"mean": 30.0, "std": 10.0, "clip_low": 15.0, "clip_high": 60.0} and step["patience"] == {"mean": 300.0, "std": 0.0}
                leave = body["sequences"]["leave"]["steps"]["exit"]["target_pose"]
                assert shapely.Polygon([(p["x"], p["y"]) for p in doc["regions"][f"{name}_sink"]["polygon"]]).contains(shapely.Point(leave["x"], leave["y"]))
    assert header.startswith(f"parcel recipients: {len(who)} (")
    # no recipients: the targets and types are there, no source is added
    empty, same_types, _ = recipients.recipient_crowd(doc, pick.layout, serves, (), MIX)
    assert set(empty["regions"]) == set(doc["regions"]) and same_types == types and [o["name"] for o in empty["objects"]] == list(targets)
    del doc["regions"]["door_0_sink"]
    with pytest.raises(ValueError, match="no sink 'door_0_sink'"):
        recipients.recipient_crowd(doc, pick.layout, serves, who, MIX)


@pytest.fixture
def site_copy(tmp_path: Path, assets: Path, space: EntranceSearch) -> Site:  # noqa: F811
    """hospital_1 in the local worlds root with a crowd between the test entrances, and the first study's visitors."""
    root = assets / "worlds" / f"hospital_rc_{tmp_path.name}"
    shutil.copytree(HOSPITAL, root)
    folder = root / "scenarios" / "mix"
    folder.mkdir(parents=True)
    (folder / "scenario.yaml").write_text("# the mix crowd\n" + yaml.safe_dump(_scenario({e.name: e.anchor for e in space.entrances}), sort_keys=False))
    (folder / infra.VISITORS).write_text(yaml.safe_dump({"rate_per_h": 6.0, "target": {"name": "locker_visit"}, "regions": {}}))
    (folder / "types").mkdir()
    (folder / "types" / "visitor.yaml").write_text("name: visitor\nextends: adult\n")
    return open_site(root)


def test_a_layout_copy_gets_one_crowd_scenario_per_seed_and_nothing_written_is_replaced(space: EntranceSearch, site_copy: Site) -> None:
    pick, doc, serves, who = _crowd(space, 2)
    crowds = {recipients.scenario_name("mix", seed): recipients.recipient_crowd(doc, pick.layout, serves, who if seed == 3 else (), MIX) for seed in (3, 4)}
    assert list(crowds) == ["mix__s3", "mix__s4"] and recipients.scenario_name("mix", 3, "b") == "mix__b__s3"
    scenario = site_copy.root / "scenarios" / "mix"
    path = recipients.save_layout(site_copy, pick.layout, scenario, crowds)
    assert path.name == f"{site_copy.key}__layout_{pick.layout_id}" and infra.hardlinked(site_copy.root, path)
    world = yaml.safe_load((path / "world.yaml").read_text())
    names = [o["name"] for level in world["levels"] for o in level.get("objects") or []]
    assert {"charger_1", "charger_2", "bay_1", "bay_2", "loading_point", "locker_1", "locker_2"} <= set(names)
    assert {o["name"]: o.get("charge_power_w") for level in world["levels"] for o in level.get("objects") or [] if o["name"].startswith("charger")} == {"charger_1": place.CHARGE_POWER_W, "charger_2": place.CHARGE_POWER_W}
    # the chargers' power is the copy's, not the layout id's: 270 Wh in 4 h on top of the standby draw by default
    slow = recipients.with_charge_power(pick.layout, recipients.CHARGE_POWER_W)
    assert recipients.copy_path(site_copy, slow) == path and slow.layout_id == pick.layout_id and recipients.CHARGE_POWER_W == 109.7 and [o.name for o in slow.objects] == [o.name for o in pick.layout.objects]
    assert sorted({s.element.charge_power_w for s in slow.spots if s.element.charge_power_w is not None}) == [109.7]
    with pytest.raises(ValueError, match="nothing recorded is replaced"):
        recipients.save_layout(site_copy, slow, scenario, crowds)
    assert sorted(p.name for p in (path / "scenarios").iterdir() if p.name.startswith("mix")) == ["mix__s3", "mix__s4"]
    for name, (expected, types, header) in crowds.items():
        folder = path / "scenarios" / name
        text = (folder / "scenario.yaml").read_text()
        assert text.startswith(f"{infra.MARKER}: {header}\n# the mix crowd\n") and yaml.safe_load(text) == expected
        assert not (folder / infra.VISITORS).exists() and (folder / "types" / "visitor.yaml").is_file()
        assert {p.name for p in (folder / "types").iterdir()} == {"visitor.yaml", *types}
        assert yaml.safe_load((folder / "types" / next(iter(types))).read_text()) == next(iter(types.values()))
    # the same call again keeps everything; another seed is added beside
    stamp = (path / "world.yaml").stat().st_mtime_ns
    assert recipients.save_layout(site_copy, pick.layout, scenario, {**crowds, "mix__s5": crowds["mix__s4"]}) == path
    assert (path / "world.yaml").stat().st_mtime_ns == stamp and (path / "scenarios" / "mix__s5" / "scenario.yaml").is_file()
    # a scenario that fails the check is taken away again, and nothing else: here the copy stays
    with pytest.raises(ValueError, match="nothing of it is kept: no floor"):
        recipients.save_layout(site_copy, pick.layout, scenario, {"mix__s6": crowds["mix__s4"]}, check=lambda world, name: "no floor" if name == "mix__s6" else None)
    assert sorted(p.name for p in (path / "scenarios").iterdir() if p.name.startswith("mix")) == ["mix__s3", "mix__s4", "mix__s5"] and (path / "world.yaml").stat().st_mtime_ns == stamp
    assert recipients.save_layout(site_copy, pick.layout, scenario, crowds, check=lambda world, name: None) == path
    # a written scenario is not replaced by another, nor a copy that holds another layout
    with pytest.raises(ValueError, match="a written scenario is not replaced"):
        recipients.save_layout(site_copy, pick.layout, scenario, {"mix__s4": crowds["mix__s3"]})
    (path / "edits.yaml").write_text((path / "edits.yaml").read_text().replace("locker_2", "locker_9"))
    with pytest.raises(ValueError, match="nothing recorded is replaced"):
        recipients.save_layout(site_copy, pick.layout, scenario, crowds)


def test_cli_evaluation_writes_the_layouts_of_a_search_report_without_robots_and_the_site(space: EntranceSearch, site_copy: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    site = site_copy
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
    found = json.loads(report.read_text())["reaches"][0]
    assert found["valid"] is True
    by_role = {r["role"]: r for r in found["layouts"]}
    optimal, practice = by_role["travel_optimal"], by_role["practice_based"]
    capsys.readouterr()
    out = tmp_path / "suites" / "infra_rd" / "suite.yaml"
    ids = f"travel_optimal={optimal['layout_id']}" + (f",practice={practice['layout_id']}" if practice["layout_id"] != optimal["layout_id"] else "")
    count = len(ids.split(","))
    args = ["infra", "evaluation", *common, "--report", str(report), "--layouts", ids, "--seeds", "71-74", "--without-robots-seeds", "71,72", "--rate", "240", "--recipient-offset", "60", "--seed-part", "h1=71,72", "--seed-part", "h2=73-74", "--out", str(out)]
    assert cli.main([*args, "--dry-run"]) == 0 and not SU.suites_dir(out).exists()
    assert "no world or suite written" in capsys.readouterr().err
    # each layout's crowd scenario is checked as the runtime checks it at an episode's start: the test world's
    # furniture resolves in no asset root here, which stops the command before a suite is written
    assert cli.main(args) == 1 and "human.check" in capsys.readouterr().err and not SU.suites_dir(out).exists()
    assert not [p.name for p in site.root.parent.iterdir() if p.name.startswith(f"{site.key}__layout_")]  # the copy that failed is not kept
    args = [*args, "--no-check"]
    assert cli.main(args) == 0
    printed = capsys.readouterr()
    assert not out.exists()  # the runs with the fleet are not written here
    site_doc = yaml.safe_load(SU.sibling(out, "infra_rd_site").read_text())
    without = yaml.safe_load(SU.sibling(out, "infra_rd_without").read_text())
    assert [s["name"] for s in site_doc["stages"]] == [f"{site.key}__s{seed}" for seed in (71, 72, 73, 74)] and site_doc == infra.reference_suite(site.key, [71, 72, 73, 74], scenario="mix")
    worlds = [f"{site.key}__layout_{layout_id.rpartition('=')[2]}" for layout_id in ids.split(",")]
    assert [s["name"] for s in without["stages"]] == [f"{world}__s{seed}" for world in worlds for seed in (71, 72)]
    assert without["references"] is False and without["launch"] == site_doc["launch"] and without["record"] == site_doc["record"]
    for stage in without["stages"]:
        # a crowd-only stage as the site's, on the copy and its scenario of the seed
        twin = next(s for s in site_doc["stages"] if s["seed"] == stage["seed"])
        assert stage == {**twin, "name": stage["name"], "map": stage["map"], "config": {**twin["config"], "scenario": {"file": f"mix__s{stage['seed']}"}}}
    manifest = json.loads(infra.manifest_path(out).read_text())
    copy = yaml.safe_load((Path(manifest["worlds"][optimal["layout_id"]]) / "world.yaml").read_text())
    assert {o["charge_power_w"] for level in copy["levels"] for o in level.get("objects") or [] if o["name"].startswith("charger")} == {109.7} and manifest["charge_power_w"] == 109.7
    assert manifest["suite"] is None and manifest["site_suite"] == "infra_rd_site" and manifest["without_suite"] == "infra_rd_without"
    # the seeds with a crowd-only run are not named `without_robots_seeds`: that key makes the analysis ask every row
    # for a run with the fleet, and none is written here
    assert manifest["seeds"] == [71, 72, 73, 74] and manifest["crowd_only_seeds"] == [71, 72] and "without_robots_seeds" not in manifest and manifest["episodes"] == 4 + 2 * count
    assert manifest["writer_commit"] is None and manifest["picks"] == {"travel_optimal": optimal["layout_id"], **({"practice": practice["layout_id"]} if count == 2 else {})}
    assert list(manifest["worlds"]) == [r["layout_id"] for r in manifest["runs"]] and manifest["runs"][0]["role"] == "travel_optimal" and manifest["runs"][0]["serves"] == optimal["serves"]
    assert manifest["seed_parts"] == [
        {"tag": "h1", "seeds": [71, 72], "site_suite": "infra_rd_site_h1", "without_suite": "infra_rd_without_h1", "episodes": 2 + 2 * count},
        {"tag": "h2", "seeds": [73, 74], "site_suite": "infra_rd_site_h2", "without_suite": None, "episodes": 2},
    ]
    assert [s["seed"] for s in yaml.safe_load(SU.sibling(out, "infra_rd_without_h1").read_text())["stages"]] == [71, 72] * count and not SU.sibling(out, "infra_rd_without_h2").exists()
    # the recipients follow the seeds' jobs, the same in every layout; their sources are the manifest's visitor sources
    names = sorted(e.name for e in space.entrances)
    for seed in (71, 72, 73, 74):
        stream = recipients.jobs(seed, names, rate_per_h=240.0, episode_s=1500.0)
        assert manifest["jobs"][str(seed)] == [j.record() for j in stream]
        per_layout = [manifest["recipients"]["by_layout"][r["layout_id"]][str(seed)] for r in manifest["runs"]]
        assert [x["job"] for x in per_layout[0]] == [j.index for j in stream if round((j.release_s + 60.0) * 10) / 10 < 1500.0]
        assert all([(x["job"], x["entrance"], x["group"], x["t_s"]) for x in other] == [(x["job"], x["entrance"], x["group"], x["t_s"]) for x in per_layout[0]] for other in per_layout)
        for record in manifest["runs"]:
            doc = yaml.safe_load((Path(manifest["worlds"][record["layout_id"]]) / "scenarios" / f"mix__s{seed}" / "scenario.yaml").read_text())
            mine = manifest["recipients"]["by_layout"][record["layout_id"]][str(seed)]
            assert {name for name in doc["regions"] if name.startswith("recipient_")} == {x["source"] for x in mine} <= set(manifest["visitor_sources"])
            assert all(x["entrance"] in record["serves"][x["locker"]] for x in mine)
    assert manifest["recipients"]["offset_s"] == 60.0 and manifest["recipients"]["kinds"] == ["delivery", "pickup"] and manifest["recipients"]["patience_s"] == 300.0 and list(manifest["recipients"]["mix"]) == ["adult", "wheelchair", "long_cane"]
    assert "--suite infra_rd_site " in printed.out and "--suite infra_rd_without " in printed.out and "peds_only" in printed.out

    # again: everything is kept as it is; other settings at the same seeds are refused, with a tag they get new scenarios
    def stamped() -> dict[Path, int]:
        return {p: p.stat().st_mtime_ns for p in SU.suites_dir(out).rglob("*") if p.is_file()}

    stamps = stamped()
    assert cli.main(args) == 0 and stamped() == stamps
    assert cli.main([*args, "--recipient-offset", "90"]) == 1
    assert "written suites and manifests are not replaced" in capsys.readouterr().err and stamped() == stamps
    # another name with the same scenarios: refused before a suite or a manifest of that name is written
    assert cli.main([*args, "--recipient-offset", "90", "--out", str(SU.sibling(out, "infra_rdx"))]) == 1
    assert "a written scenario is not replaced" in capsys.readouterr().err and not list(SU.suites_dir(out).glob("infra_rdx*"))
    assert cli.main([*args, "--recipient-offset", "90", "--tag", "b", "--out", str(SU.sibling(out, "infra_rdb"))]) == 0
    assert yaml.safe_load(SU.sibling(out, "infra_rdb_without").read_text())["stages"][0]["config"]["scenario"] == {"file": "mix__b__s71"}
    assert cli.main([*args, "--tag", "c", "--writer-commit", "0123abcd", "--out", str(SU.sibling(out, "infra_rdc"))]) == 0
    assert json.loads(infra.manifest_path(SU.sibling(out, "infra_rdc")).read_text())["writer_commit"] == "0123abcd"
    # what the command needs, and what belongs to the first study's evaluation
    assert cli.main(["infra", "evaluation", *common, "--report", str(report), "--layouts", "x=ffffffff", "--seeds", "71", "--rate", "6"]) == 1
    assert cli.main(["infra", "evaluation", *common, "--report", str(report), "--layouts", ids, "--seeds", "71"]) == 1
    assert cli.main(["infra", "evaluation", *common, "--report", str(report), "--layouts", ids, "--seeds", "71", "--rate", "6", "--without-robots-seeds", "72"]) == 1
    assert cli.main(["infra", "evaluation", *common, "--report", str(report), "--layouts", ids, "--seeds", "71-72", "--rate", "6", "--seed-part", "h1=71"]) == 1
    assert cli.main(["infra", "evaluation", *common, "--report", str(report), "--layouts", ids, "--seeds", "71", "--rate", "6", "--pick", str(report)]) == 1
    assert cli.main(["infra", "evaluation", *common]) == 1
    # every layout under a role of its own
    from arena_simulation_setup.generative.flowedit import infra_cli

    assert infra_cli._named_layouts("a=11111111, b=22222222") == [("a", "11111111"), ("b", "22222222")]
    for bad in ("11111111", "a=11111111,a=22222222", "a=11111111,b=11111111", "=11111111", "a="):
        with pytest.raises(Exception, match="a layout is <role>=<id>|twice"):
            infra_cli._named_layouts(bad)


def test_the_locker_front_of_the_first_study_is_the_front_of_its_locker_spot(hospital: Site, inner: Search) -> None:  # noqa: F811
    spot = inner.spots[LOCKER][0]
    layout = place.Layout.of([inner.loading, spot])
    assert infra.locker_front(layout, 0.35) == infra.spot_front(spot, 0.35) == infra.spot_front(demand.numbered(spot, 2), 0.35)
    with pytest.raises(ValueError, match="has no locker"):
        infra.locker_front(place.Layout.of([inner.loading]), 0.35)


def test_a_new_copy_whose_scenario_fails_the_check_is_not_kept(space: EntranceSearch, site_copy: Site) -> None:  # noqa: F811
    pick, doc, serves, who = _crowd(space, 1)
    crowds = {"mix__s3": recipients.recipient_crowd(doc, pick.layout, serves, who, MIX)}
    seen: list[tuple[str, str]] = []

    def check(world: Path, name: str) -> str | None:
        seen.append((world.name, name))
        assert (world / "scenarios" / name / "scenario.yaml").is_file()
        return "the first in line stands in a wall"

    with pytest.raises(ValueError, match="nothing of it is kept: the first in line stands in a wall"):
        recipients.save_layout(site_copy, pick.layout, site_copy.root / "scenarios" / "mix", crowds, check=check)
    assert seen == [(recipients.copy_path(site_copy, pick.layout).name, "mix__s3")] and not recipients.copy_path(site_copy, pick.layout).exists()


def test_cli_evaluation_refuses_a_kept_layout_copy_that_holds_a_wall_its_site_config_removes(space: EntranceSearch, site_copy: Site, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from arena_simulation_setup.metrics.navr_filter import NavrBase

    site = site_copy
    spot = next(s for s in place.candidate_spots(site, "loading_point") if s.valid and s.zone == "central_hallway")
    level = site.world.levels[spot.level]
    navr = NavrBase(level, exclude={o.name for o in level.objects})
    monkeypatch.setattr(search.Search, "base", property(lambda self: navr))
    x, y = spot.objects[0].goto
    doc = {"site": site.key, "scenario": "mix", "courier_access": [x, y], "entrances": [{"name": e.name, "type": e.type, "at": list(e.anchor)} for e in space.entrances]}
    config = tmp_path / "site.yaml"
    config.write_text(yaml.safe_dump(doc))
    report = tmp_path / "report.json"
    common = ["--site", str(site.root), "--site-config", str(config)]
    assert cli.main(["infra", "search", *common, "--reach", "12", "--margins", "1.5", "--rate", "6", "--rank", "walk", "--out", str(report)]) == 0
    optimal = next(r for r in json.loads(report.read_text())["reaches"][0]["layouts"] if r["role"] == "travel_optimal")
    out = tmp_path / "suites" / "infra_rd" / "suite.yaml"
    args = ["infra", "evaluation", "--report", str(report), "--layouts", f"travel_optimal={optimal['layout_id']}", "--seeds", "71", "--rate", "240", "--no-check"]
    assert cli.main([*args, *common, "--out", str(out)]) == 0
    copy = next(p for p in site.root.parent.iterdir() if p.name == f"{site.key}__layout_{optimal['layout_id']}")
    # a sign the scan caught stands in the kept copy only (the site was cleaned, the copy not): the copy is reused as it is
    # (its edits.yaml matches), and the writer refuses it instead of writing a suite on it
    text = (copy / "world.yaml").read_text()
    sign = [[x + 1.0, y], [x + 1.4, y], [x + 1.4, y + 0.4], [x + 1.0, y + 0.4]]
    world = yaml.safe_load(text)
    world["levels"][0]["walls"].append({"path": sign, "closed": True, "kind": "scan_boundary"})
    (copy / "world.yaml").write_text(yaml.safe_dump(world, sort_keys=False))
    config.write_text(yaml.safe_dump({**doc, "removed_walls": [{"name": "sign", "kind": "scan_boundary", "bbox": [x + 1.0, y, x + 1.4, y + 0.4]}]}))
    capsys.readouterr()
    other = tmp_path / "suites" / "infra_rdz" / "suite.yaml"
    assert cli.main([*args, *common, "--tag", "z", "--out", str(other)]) == 1
    assert "still holds the walls ['sign']" in capsys.readouterr().err
    assert not list(SU.suites_dir(other).glob("infra_rdz*"))
