from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from arena_simulation_setup.generative.flowedit import demand as D
from arena_simulation_setup.generative.flowedit import demand_propose as DP
from arena_simulation_setup.generative.flowedit import settings as S
from arena_simulation_setup.generative.flowedit.build import build
from arena_simulation_setup.generative.flowedit.cli import main
from arena_simulation_setup.generative.flowedit.edit import DEMAND_VERBS, INERT, apply, open_site
from arena_simulation_setup.generative.flowedit.read import plan_needs, read_graph
from arena_simulation_setup.generative.scenario.propose.client import Reply
from arena_simulation_setup.kg import Scene, Value, dumps, load, write

pytest.importorskip("numpy")

WORLDS = Path(__file__).resolve().parents[2] / "worlds"
HOSPITAL = WORLDS / "hospital_1"
QUEUE = HOSPITAL / "scenarios" / "queue" / "scenario.yaml"
DISPENSER = "obj:reception_water_dispenser_1"
TRASHCAN = "obj:reception_trashcan_1"
#: The models of hospital_1 no class rule covers
OPEN = ["Hospital/Anesthesia_Machine", "Hospital/Autoclave", "Hospital/Defibrillator", "Hospital/Exam_Table", "Hospital/IV_Stand", "Hospital/Instrument_Table", "Hospital/Microscope_Station", "Hospital/Operating_Table", "Hospital/Refrigerated_Medicine_Unit", "Hospital/SM_Scale_01a", "Hospital/Surgical_Table", "Hospital/Ventilator", "Hospital/Vital_Signal_Station"]  # fmt: skip
CLASSES = D.load_classes()


def no_tags(model: str) -> None:
    return None


def _facts(scene: Scene, subject: str, verb: str) -> list[tuple[Value, dict[str, Value]]]:
    return [(f.object, dict(f.params)) for f in scene.facts if f.subject == subject and f.verb == verb]


def _demand(scene: Scene, subject: str) -> list[tuple[str, Value, dict[str, Value]]]:
    return [(f.verb, f.object, dict(f.params)) for f in scene.facts if f.subject == subject and (f.verb in DEMAND_VERBS or f.verb == INERT)]


@pytest.fixture
def queue() -> Scene:
    return build(HOSPITAL, QUEUE)


@pytest.fixture
def bare() -> Scene:
    return build(HOSPITAL)


def _proposals(path: Path, models: dict[str, dict[str, Any]]) -> Path:
    path.write_text(yaml.safe_dump({"format": D.PROPOSALS_FORMAT, "models": models}))
    return path


# the inert fact


def test_an_inert_object_validates_and_survives_a_round_trip(bare: Scene, tmp_path: Path) -> None:
    bare.add_fact(TRASHCAN, INERT, True, {"by": "class", "why": "kind::bin"})
    assert not bare.validate()
    assert not D.check(bare)
    write(bare, tmp_path / "flowgraph.yaml")
    assert _facts(load(tmp_path / "flowgraph.yaml"), TRASHCAN, INERT) == [(True, {"by": "class", "why": "kind::bin"})]


def test_an_object_is_authored_decided_or_undecided(queue: Scene) -> None:
    assert D.source(queue, "obj:fountain") == D.AUTHORED
    assert D.source(queue, TRASHCAN) == D.UNDECIDED
    queue.add_fact(TRASHCAN, INERT, True, {"by": "class"})
    assert D.source(queue, TRASHCAN) == "class"
    assert TRASHCAN not in D.undecided(queue)
    queue.add_fact(TRASHCAN, "serves", "service:bin", {"by": "class"})
    assert D.check(queue) == [f"{TRASHCAN}: is inert and has demand"]


def test_a_function_edit_that_gives_demand_drops_inert(local_root: Path) -> None:
    site = open_site(HOSPITAL, "queue")
    site.graph.add_fact(TRASHCAN, INERT, True, {"by": "class"})
    edited = apply(site, "function", TRASHCAN, {"type": "bin", "capacity": 1})
    assert not _facts(edited.graph, TRASHCAN, INERT)
    assert _facts(edited.graph, TRASHCAN, "serves") == [("service:bin", {"capacity": 1})]
    assert _facts(site.graph, TRASHCAN, INERT)


def test_the_reader_leaves_inert_objects_alone(queue: Scene) -> None:
    plain = plan_needs(read_graph(queue), S.load(), "0", [])
    queue.add_fact(TRASHCAN, INERT, True, {"by": "class"})
    assert plan_needs(read_graph(queue), S.load(), "0", []) == plain


# the class table


def test_the_table_names_only_asa_kinds_and_every_block_builds() -> None:
    from arena_assets.semantic import load_spec

    kinds = set(load_spec().facets["kind"].values_for("object"))
    hoi = set(load_spec().facets["hoi"].values_for("object"))
    for cls in CLASSES.values():
        assert set(cls.kinds) <= kinds, cls.name
        assert set(cls.hoi) <= hoi, cls.name
        assert cls.desc, cls.name
    scene = build(HOSPITAL)
    for i, cls in enumerate(CLASSES.values()):
        obj = D.undecided(scene)[i]
        D._fill(scene, obj, cls, {"by": "class", "why": "test"}, [])
        assert D.source(scene, obj) == "class"
    assert not scene.validate()
    assert CLASSES["inert"].block is None


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"a": {"kinds": ["chair"]}, "b": {"kinds": ["chair"]}}, "kinds 'chair' is already of class 'a'"),
        ({"a": {"dwell": 5.0, "humansim": {"type": "x"}}}, "a dwell needs a need"),
        ({"a": {"capacity": 1}}, "unknown keys ['capacity']"),
    ],
)
def test_a_wrong_table_is_refused(tmp_path: Path, raw: dict[str, Any], message: str) -> None:
    path = tmp_path / "classes.yaml"
    path.write_text(yaml.safe_dump({"format": D.CLASSES_FORMAT, "classes": raw}))
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        D.load_classes(path)


@pytest.mark.parametrize(
    ("model", "tags", "expected"),
    [
        ("Hospital/Thing", ["domain::hospital", "kind::dispenser"], ("fountain", "kind::dispenser")),
        ("Hospital/Thing", ["kind::equipment", "hoi::sit"], ("chair", "hoi::sit")),
        ("Hospital/SM_Chair_01a", ["kind::cabinet"], ("inert", "kind::cabinet")),
        ("Hospital/SM_Chair_01a", ["kind::equipment"], ("chair", "name:chair")),
        ("Hospital/SM_Chair_01a", None, ("chair", "name:chair")),
        ("Hospital/Lab_Bench", None, ("desk", "name:labbench")),
        ("Hospital/Waiting_Room_Bench", None, ("gathering_area", "name:bench")),
        ("Hospital/SM_WheelChair_01a", None, ("inert", "name:wheelchair")),
        ("Hospital/SM_BedSideTable_01b", None, ("inert", "name:sidetable")),
        ("Hospital/SM_HospitalBed_02d", None, ("bed", "name:bed")),
        ("Hospital/SM_ReceptionDesk_01a", None, ("counter", "name:receptiondesk")),
        ("Office/Stand_Table_Work", None, ("desk", "name:tablework")),
        ("Office/SM_ChairOffice", None, ("chair", "name:chair")),
        ("Office/SM_SecretaryDeskA", None, ("desk", "name:desk")),
        ("Office/SM_Keyboard", None, ("inert", "name:board")),
        ("Hospital/IV_Stand", None, None),
        ("Office/SM_DrinksMachine", None, ("fountain", "name:drinksmachine")),
        ("Hospital/Ventilator", None, None),
        ("Hospital/Exam_Table", ["kind::table"], None),
    ],
)
def test_an_asset_is_classed_by_its_kind_then_hoi_then_the_longest_word_of_its_name(model: str, tags: list[str] | None, expected: tuple[str, str] | None) -> None:
    assert D.classify(model, tags, CLASSES) == expected


# the class stage


def test_the_class_stage_leaves_authored_demand_alone(queue: Scene) -> None:
    plain = build(HOSPITAL, QUEUE)
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    for obj in ("obj:fountain", "obj:reception_desk", "obj:gathering_area"):
        assert _demand(queue, obj) == _demand(plain, obj)
        assert D.source(queue, obj) == D.AUTHORED


def test_every_pre_filled_fact_says_who_decided_it_and_why(queue: Scene) -> None:
    plain = {f.key for f in build(HOSPITAL, QUEUE).facts}
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    added = [f for f in queue.facts if f.key not in plain]
    assert added
    assert {f.verb for f in added} <= {*DEMAND_VERBS, INERT}
    assert all(f.params.get("by") == "class" and f.params.get("why") for f in added)
    assert _facts(queue, TRASHCAN, INERT) == [(True, {"by": "class", "why": "name:trashcan"})]
    assert not D.check(queue)


def test_a_pre_filled_object_is_used_as_the_object_that_already_serves_its_need(queue: Scene) -> None:
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    mark = {"by": "class", "why": "name:dispenser"}
    assert _facts(queue, DISPENSER, "serves") == [("service:fountain", {"capacity": 1, **mark})]
    assert _facts(queue, DISPENSER, "satisfies") == [("need:thirst", {"amount": 100.0, **mark})]
    for verb in ("dwells", "affords"):
        assert _facts(queue, DISPENSER, verb) == [(o, {**p, **mark}) for o, p in _facts(queue, "obj:fountain", verb)]
    plan = plan_needs(read_graph(queue), S.load(), "0", [])["thirst"]
    assert plan.target == "fountain"
    assert sorted(plan.objects) == ["fountain", "reception_water_dispenser_1"]


def test_without_an_object_for_its_need_a_class_brings_its_own_dwell_and_use(bare: Scene) -> None:
    D.prefill(bare, [D.Stage.CLASS], no_tags)
    mark = {"by": "class", "why": "name:dispenser"}
    assert _facts(bare, DISPENSER, "dwells") == [(5.0, {"std": 1.5, "clip_low": 4.0, "clip_high": 6.0, "from": "demand_classes.yaml#fountain", **mark})]
    assert _facts(bare, DISPENSER, "affords") == [("interaction:QUEUE_USE", {"from": "rule:formation line", **mark})]
    assert plan_needs(read_graph(bare), S.load(), "0", [])["thirst"].objects == ("reception_water_dispenser_1",)


def test_a_gathering_area_is_anchored_at_itself(bare: Scene) -> None:
    D.prefill(bare, [D.Stage.CLASS], no_tags)
    sofa = next(obj for obj in D.objects(bare) if D.model_of(bare, obj) == "Hospital/Waiting_Room_Sofa")
    ((formation, params),) = _facts(bare, sofa, "has_formation")
    assert formation == "formation:f_formation"
    assert params["anchor_ref"] == sofa.partition(":")[2]


def test_the_class_stage_decides_all_of_hospital_1_but_its_equipment_tables_and_stands(queue: Scene) -> None:
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    assert sorted({D.model_of(queue, obj) for obj in D.undecided(queue)}) == OPEN
    by_source: dict[str, int] = {}
    for row in D.coverage(queue):
        by_source[row.source] = by_source.get(row.source, 0) + row.count
    assert by_source == {"authored": 5, "class": 120, "undecided": 23}
    assert sum(by_source.values()) == len(D.objects(queue))


def test_pre_filling_is_deterministic_and_decided_objects_stay(queue: Scene) -> None:
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    once = dumps(queue)
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    assert dumps(queue) == once
    again = build(HOSPITAL, QUEUE)
    D.prefill(again, [D.Stage.CLASS], no_tags)
    assert dumps(again) == once


def test_a_tag_of_the_asset_decides_before_its_name(queue: Scene, local_root: Path) -> None:
    from arena_simulation_setup.generative.scenario.formats import load_world
    from arena_simulation_setup.shared.entities import Obstacle

    for model, tags in (("Hospital/Ventilator", ["domain::hospital", "kind::dispenser"]), ("Hospital/SM_Chair_01a", ["kind::cabinet"])):
        directory = local_root / Obstacle.parse({"name": "probe", "model": model, "pose": [0.0, 0.0, 0.0]}).model.relpath()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "annotation.yaml").write_text(yaml.safe_dump({"asa": 1, "tags": tags}))
    tags_of = D.asset_tags(HOSPITAL, load_world(HOSPITAL), ["Hospital/Ventilator", "Hospital/SM_Chair_01a", "Hospital/SM_TrashCan"])
    assert tags_of == {"Hospital/Ventilator": ["domain::hospital", "kind::dispenser"], "Hospital/SM_Chair_01a": ["kind::cabinet"], "Hospital/SM_TrashCan": None}
    D.prefill(queue, [D.Stage.CLASS], D.world_tags(HOSPITAL, queue))
    ventilator = next(obj for obj in D.objects(queue) if D.model_of(queue, obj) == "Hospital/Ventilator")
    chair = next(obj for obj in D.objects(queue) if D.model_of(queue, obj) == "Hospital/SM_Chair_01a")
    assert _facts(queue, ventilator, "serves")[0][1]["why"] == "kind::dispenser"
    assert _facts(queue, chair, INERT) == [(True, {"by": "class", "why": "kind::cabinet"})]


# proposals


def test_a_humans_proposal_decides_before_tag_and_name_and_a_models_only_in_the_llm_stage(queue: Scene) -> None:
    proposals = {
        "Hospital/Exam_Table": {"class": "bed"},
        "Hospital/Lab_Bench": {"class": "inert"},
        "Hospital/Ventilator": {"class": "inert", "by": "llm", "model": "gemini-3.5-flash", "prompt_sha": "12a42fdc3c28fdd0", "why": "a machine staff operate"},
    }
    D.prefill(queue, [D.Stage.CLASS], lambda model: ["kind::desk"] if model == "Hospital/Lab_Bench" else None, proposals=proposals)
    table = next(obj for obj in D.objects(queue) if D.model_of(queue, obj) == "Hospital/Exam_Table")
    bench = next(obj for obj in D.objects(queue) if D.model_of(queue, obj) == "Hospital/Lab_Bench")
    ventilator = next(obj for obj in D.objects(queue) if D.model_of(queue, obj) == "Hospital/Ventilator")
    assert _facts(queue, table, "serves") == [("service:patient_bed", {"capacity": 1, "interaction_radius": 1.2, "by": "class", "why": D.REVIEWED})]
    assert _facts(queue, bench, INERT) == [(True, {"by": "class", "why": D.REVIEWED})]
    assert D.source(queue, ventilator) == D.UNDECIDED
    D.prefill(queue, [D.Stage.CLASS, D.Stage.LLM], no_tags, proposals=proposals)
    assert _facts(queue, ventilator, INERT) == [(True, {"by": "llm", "model": "gemini-3.5-flash", "prompt_sha": "12a42fdc3c28fdd0", "why": "a machine staff operate"})]
    assert D.source(queue, ventilator) == "llm"


def test_the_llm_stage_alone_takes_only_a_models_proposals(queue: Scene) -> None:
    D.prefill(queue, [D.Stage.LLM], no_tags, proposals={"Hospital/Ventilator": {"class": "inert", "by": "llm"}, "Hospital/Exam_Table": {"class": "bed"}})
    assert {D.model_of(queue, obj) for obj in D.objects(queue) if D.source(queue, obj) == "llm"} == {"Hospital/Ventilator"}
    assert not [obj for obj in D.objects(queue) if D.source(queue, obj) == "class"]


def test_a_proposal_of_no_class_is_refused(queue: Scene, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="proposed as class 'throne'"):
        D.prefill(queue, [D.Stage.CLASS], no_tags, proposals={"Hospital/Ventilator": {"class": "throne"}})
    with pytest.raises(ValueError, match="needs a class"):
        D.load_proposals(_proposals(tmp_path / "p.yaml", {"Hospital/Ventilator": {"by": "llm"}}))
    assert D.load_proposals(tmp_path / "missing.yaml") == {}


def test_the_shipped_proposals_name_only_classes() -> None:
    assert all(entry["class"] in CLASSES for entry in D.load_proposals().values())


ANSWER = {
    "Hospital/Ventilator": {"class": "inert", "why": "A machine: staff operate it"},
    "Hospital/Exam_Table": {"class": "bed", "why": "Patients lie on it."},
    "Hospital/SM_Scale_01a": {"class": "unknown", "why": "Could be anything."},
    "Hospital/Autoclave": {"class": "throne", "why": "x"},
    "Hospital/SM_TrashCan": {"class": "inert", "why": "x"},
}
ASKED = ["Hospital/Autoclave", "Hospital/Defibrillator", "Hospital/Exam_Table", "Hospital/SM_Scale_01a", "Hospital/Ventilator"]


def test_parse_drops_what_was_not_asked_or_is_no_class_and_names_what_is_missing() -> None:
    proposals, complaints = DP.parse(json.dumps(ANSWER), ASKED, CLASSES, model="m", sha="abc")
    assert [(p.model, p.cls, p.confident) for p in proposals] == [("Hospital/Ventilator", "inert", True), ("Hospital/Exam_Table", "bed", True), ("Hospital/SM_Scale_01a", "unknown", False)]
    assert proposals[0].why == "A machine staff operate it"
    assert (proposals[0].llm, proposals[0].prompt_sha) == ("m", "abc")
    assert complaints == ["Hospital/Autoclave: 'throne' is not a class", "Hospital/SM_TrashCan: not one of the models asked about", "Hospital/Autoclave: no answer returned", "Hospital/Defibrillator: no answer returned"]


@pytest.mark.parametrize(("text", "complaint"), [("not json", "response was not JSON"), ("[1, 2]", "response was list, expected an object")])
def test_parse_refuses_what_is_not_a_json_object(text: str, complaint: str) -> None:
    proposals, complaints = DP.parse(text, ASKED, CLASSES, model="m", sha="abc")
    assert not proposals
    assert complaints[0].startswith(complaint)


def test_parse_reads_a_fenced_answer() -> None:
    proposals, _ = DP.parse("```json\n" + json.dumps({"Hospital/Ventilator": {"class": "inert"}}) + "\n```", ASKED, CLASSES, model="m", sha="abc")
    assert [(p.model, p.cls) for p in proposals] == [("Hospital/Ventilator", "inert")]


def test_merge_writes_confident_proposals_and_never_over_a_humans(tmp_path: Path) -> None:
    path = _proposals(tmp_path / "proposals.yaml", {"Hospital/Exam_Table": {"class": "inert"}})
    proposals, _ = DP.parse(json.dumps(ANSWER), ASKED, CLASSES, model="m", sha="abc")
    written, kept = DP.merge(proposals, path)
    assert written == 1
    assert kept == ["Hospital/Exam_Table: reviewed by a human as 'inert'", "Hospital/SM_Scale_01a: model returned `unknown` - left for a human"]
    assert D.load_proposals(path) == {
        "Hospital/Exam_Table": {"class": "inert"},
        "Hospital/Ventilator": {"class": "inert", "by": "llm", "model": "m", "prompt_sha": "abc", "why": "A machine staff operate it"},
    }
    assert path.read_text().startswith("# FLOWEDIT demand proposals")
    assert DP.merge(proposals, path)[0] == 0


def test_only_undecided_models_without_a_proposal_are_asked_about(queue: Scene) -> None:
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    assert DP.unproposed(queue, {}) == OPEN
    assert DP.unproposed(queue, {"Hospital/Ventilator": {"class": "inert", "by": "llm"}}) == [m for m in OPEN if m != "Hospital/Ventilator"]


def test_the_prompt_names_the_classes_the_decided_models_and_the_evidence(queue: Scene) -> None:
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    head = DP.system(CLASSES)
    assert all(f"  {name}" in head for name in [*CLASSES, DP.UNKNOWN])
    user = DP.build_prompt(queue, ["Hospital/Vital_Signal_Station"], lambda model: ["kind::equipment"], world="hospital_1")
    assert '"Hospital/SM_WaterDispenser_01a": "fountain (thirst)"' in user
    asked = json.loads(user.split("suggests:\n")[1])
    assert list(asked) == ["Hospital/Vital_Signal_Station"]
    assert asked["Hospital/Vital_Signal_Station"]["count"] == 5
    assert asked["Hospital/Vital_Signal_Station"]["tags"] == ["kind::equipment"]
    assert asked["Hospital/Vital_Signal_Station"]["rooms"]


def test_propose_asks_once_and_never_for_nothing(queue: Scene, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def ask(system: str, user: str, *, model: str | None = None, temperature: float = 0.0) -> Reply:
        calls.append((system, user))
        return Reply(text=json.dumps({m: {"class": "inert", "why": "equipment"} for m in OPEN}))

    monkeypatch.setattr(DP, "ask", ask)
    D.prefill(queue, [D.Stage.CLASS], no_tags)
    assert DP.propose(queue, [], CLASSES, no_tags, world="hospital_1") == ([], [])
    proposals, complaints = DP.propose(queue, OPEN, CLASSES, no_tags, world="hospital_1", model="some-model")
    assert len(calls) == 1
    assert not complaints
    assert {(p.cls, p.llm) for p in proposals} == {("inert", "some-model")}
    assert len({p.prompt_sha for p in proposals}) == 1


# the CLI


def test_cli_build_pre_fills_only_when_asked(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["build", "hospital_1", "queue", "--worlds", str(WORLDS)]) == 0
    assert capsys.readouterr().out == dumps(build(HOSPITAL, QUEUE))
    assert main(["build", "hospital_1", "queue", "--worlds", str(WORLDS), "--prefill", "class"]) == 0
    graph = load(yaml.safe_load(capsys.readouterr().out))
    assert D.source(graph, DISPENSER) == "class"
    assert D.source(graph, "obj:fountain") == D.AUTHORED
    with pytest.raises(SystemExit):
        main(["build", "hospital_1", "--worlds", str(WORLDS), "--prefill", "guess"])


def test_cli_demand_reports_the_sources_and_fails_while_objects_are_undecided(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["demand", "hospital_1", "queue", "--worlds", str(WORLDS), "--proposals", str(tmp_path / "proposals.yaml")]
    assert main(argv) == 1
    out = capsys.readouterr().out
    assert "   1  Hospital/SM_WaterDispenser_01a       class      fountain (thirst)         name:dispenser" in out
    assert out.splitlines()[-1] == "5 authored, 120 class, 23 undecided"
    _proposals(tmp_path / "proposals.yaml", {m: {"class": "inert", "by": "llm", "model": "m", "prompt_sha": "abc", "why": "equipment"} for m in OPEN})
    assert main(argv) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "5 authored, 120 class, 23 llm"
    assert main([*argv, "--prefill", "class"]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "5 authored, 120 class, 23 undecided"


def test_cli_demand_propose_prints_the_prompt_and_asks_only_with_write(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def ask(system: str, user: str, *, model: str | None = None, temperature: float = 0.0) -> Reply:
        calls.append(user)
        return Reply(text=json.dumps({m: {"class": "bed" if m.endswith("Exam_Table") else "inert", "why": "equipment"} for m in OPEN}))

    monkeypatch.setattr(DP, "ask", ask)
    path = tmp_path / "proposals.yaml"
    argv = ["demand", "hospital_1", "queue", "--worlds", str(WORLDS), "--proposals", str(path), "--propose"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "Hospital/Vital_Signal_Station" in out
    assert "(dry run" in out
    assert not calls
    assert not path.exists()
    assert main([*argv, "--write"]) == 0
    out = capsys.readouterr().out
    assert len(calls) == 1
    assert f"  13 proposal(s) written to {path}" in out
    assert out.splitlines()[-1] == "5 authored, 120 class, 23 llm"
    assert D.load_proposals(path)["Hospital/Exam_Table"]["class"] == "bed"
    assert main([*argv, "--write"]) == 0
    assert len(calls) == 1
    assert "every undecided model has a proposal already" in capsys.readouterr().out


def test_cli_demand_without_a_model_configured_says_so(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("LLM_API_ENDPOINT", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert main(["demand", "hospital_1", "queue", "--worlds", str(WORLDS), "--proposals", str(tmp_path / "proposals.yaml"), "--propose", "--write"]) == 1
    assert "no LLM configured" in capsys.readouterr().out
