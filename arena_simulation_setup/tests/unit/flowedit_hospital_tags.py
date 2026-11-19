"""The ASA tags of hospital_1's object models as the public bucket annotates them, for the FLOWEDIT demand tests."""

from __future__ import annotations

import typing

import yaml

if typing.TYPE_CHECKING:
    from pathlib import Path

#: The `tags` of each model's annotation.yaml in the bucket `public/models`, as of 2026-11-18. The 24 `SM_*` annotations
#: say `note: Generated with Gemini-2.5`; two are known wrong (`SM_WheelChair_01a`, `Lab_Bench`) and overruled in
#: demand_proposals.yaml.
TAGS: dict[str, list[str]] = {
    "Hospital/Anesthesia_Machine": ["domain::hospital", "kind::equipment", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Autoclave": ["domain::hospital", "kind::equipment", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Biohazard_Container": ["domain::hospital", "kind::safety", "hoi::touch", "color::yellow", "material::plastic"],
    "Hospital/Defibrillator": ["domain::hospital", "kind::equipment", "hoi::touch", "color::white", "color::red", "material::plastic"],
    "Hospital/Display_Board": ["domain::hospital", "kind::sign", "hoi::see", "color::white", "material::steel"],
    "Hospital/Exam_Table": ["domain::hospital", "kind::table", "hoi::lie", "color::blue", "material::steel", "material::leather"],
    "Hospital/IV_Stand": ["domain::hospital", "kind::equipment", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Instrument_Table": ["domain::hospital", "kind::table", "hoi::touch", "color::gray", "material::steel"],
    "Hospital/Isolation_Bed": ["domain::hospital", "kind::bed", "hoi::lie", "color::white", "material::steel", "material::fabric"],
    "Hospital/Lab_Bench": ["domain::hospital", "kind::counter", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Lab_Stool": ["domain::hospital", "kind::stool", "hoi::sit", "color::yellow", "material::steel", "material::fabric"],
    "Hospital/Medical_Stool": ["domain::hospital", "kind::stool", "hoi::sit", "color::yellow", "material::steel", "material::canvas"],
    "Hospital/Microscope_Station": ["domain::hospital", "kind::equipment", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Operating_Table": ["domain::hospital", "kind::table", "hoi::lie", "color::silver", "color::blue", "material::steel", "material::fabric"],
    "Hospital/Patient_Armchair": ["domain::hospital", "kind::chair", "hoi::sit", "color::white", "color::brown", "material::wood", "material::canvas"],
    "Hospital/Pharmacy_Shelf": ["domain::hospital", "kind::shelf", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Privacy_Curtain_Rail": ["domain::hospital", "kind::curtain", "hoi::touch", "color::white", "material::steel", "material::canvas"],
    "Hospital/Reception_Counter": ["domain::hospital", "kind::counter", "hoi::touch", "color::white", "color::brown", "material::wood"],
    "Hospital/Refrigerated_Medicine_Unit": ["domain::hospital", "kind::appliance", "hoi::touch", "color::white", "material::steel"],
    "Hospital/SM_BedSideTable_01b": ["domain::hospital", "kind::table"],
    "Hospital/SM_Chair_01a": ["domain::hospital", "kind::chair"],
    "Hospital/SM_Chair_02a": ["domain::hospital", "kind::chair"],
    "Hospital/SM_Chair_04a": ["domain::hospital", "kind::chair"],
    "Hospital/SM_Desk_04a": ["domain::hospital", "kind::table"],
    "Hospital/SM_HospitalBed_01b": ["domain::hospital", "kind::bed"],
    "Hospital/SM_HospitalBed_02d": ["domain::hospital", "kind::bed"],
    "Hospital/SM_MedShelf_01d": ["domain::hospital", "kind::shelf"],
    "Hospital/SM_MedicalCabinet_01a": ["domain::hospital", "kind::cabinet"],
    "Hospital/SM_MopSet_01b": ["domain::hospital", "kind::equipment"],
    "Hospital/SM_ReceptionDesk_01a": ["domain::hospital", "kind::counter"],
    "Hospital/SM_Scale_01a": ["domain::hospital", "kind::equipment"],
    "Hospital/SM_ShelfSet_01a": ["domain::hospital", "kind::shelf"],
    "Hospital/SM_Shelf_06f": ["domain::hospital", "kind::shelf"],
    "Hospital/SM_SideTable_02a": ["domain::hospital", "kind::table"],
    "Hospital/SM_SupplyCabinet_01c": ["domain::hospital", "kind::cabinet"],
    "Hospital/SM_SupplyCart_01e": ["domain::hospital", "kind::cart"],
    "Hospital/SM_Toilet": ["domain::hospital", "kind::sanitary"],
    "Hospital/SM_ToiletPaper": ["domain::hospital", "kind::sanitary"],
    "Hospital/SM_TrashCan": ["domain::hospital", "kind::bin"],
    "Hospital/SM_UtilitiesTray_01e": ["domain::hospital", "kind::container"],
    "Hospital/SM_Washbasin": ["domain::hospital", "kind::sanitary"],
    "Hospital/SM_WaterDispenser_01a": ["domain::hospital", "kind::dispenser"],
    "Hospital/SM_WheelChair_01a": ["domain::hospital", "kind::chair"],
    "Hospital/Surgical_Table": ["domain::hospital", "kind::table", "hoi::lie", "color::blue", "material::steel", "material::canvas"],
    "Hospital/Ventilator": ["domain::hospital", "kind::equipment", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Vital_Signal_Station": ["domain::hospital", "kind::equipment", "hoi::touch", "color::white", "material::steel"],
    "Hospital/Waiting_Room_Bench": ["domain::hospital", "kind::bench", "hoi::sit", "color::blue", "material::steel", "material::plastic"],
    "Hospital/Waiting_Room_Sofa": ["domain::hospital", "kind::sofa", "hoi::sit", "color::white", "color::brown", "material::wood", "material::fabric"],
}  # fmt: skip


def write_hospital_tags(root: Path) -> dict[str, Path]:
    """The models of `TAGS` as annotation-only assets under the asset root `root`, so their tags resolve as with the
    bucket's assets; their directories by model."""
    from arena_simulation_setup.shared.entities import Obstacle

    out: dict[str, Path] = {}
    for model, tags in TAGS.items():
        directory = root / Obstacle.parse({"name": "probe", "model": model, "pose": [0.0, 0.0, 0.0]}).model.relpath()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "annotation.yaml").write_text(yaml.safe_dump({"asa": 1, "tags": tags}))
        out[model] = directory
    return out
