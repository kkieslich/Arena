"""MuJoCo obstacles take every mesh visual of their SDF, glow visuals included, and prefer the SDF over a Y-up OBJ beside it."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from arena_simulation_setup.tree.assets.Object import ObjectView
from arena_simulation_setup.utils.models import Model, ModelType

_LAMP_SDF = """<sdf version="1.9"><model name="lamp"><link name="link">
<visual name="visual"><geometry><mesh><uri>lamp.dae</uri></mesh></geometry></visual>
<visual name="glow_Shade"><geometry><mesh><uri>file://lamp.glow_0.dae</uri></mesh></geometry><material><diffuse>0.85 0.8 0.7 1</diffuse></material></visual>
<visual name="marker"><geometry><box><size>1 1 1</size></box></geometry></visual>
<collision name="collision"><geometry><mesh><uri>lamp.dae</uri></mesh></geometry></collision>
</link></model></sdf>"""


@pytest.fixture
def mesh_files():
    return pytest.importorskip("arena_runtime.sim.mujoco_simulator")._mesh_files


def test_sdf_lamp_gives_its_body_and_glow_meshes_and_skips_primitives(tmp_path: Path, mesh_files):
    model = Model(type=ModelType.SDF, name="lamp", description=_LAMP_SDF, path=tmp_path / "lamp.sdf")
    assert mesh_files(model) == [("visual", str(tmp_path / "lamp.dae")), ("glow_Shade", str(tmp_path / "lamp.glow_0.dae"))]


def test_obj_is_its_own_single_visual_and_a_pathless_model_has_none(tmp_path: Path, mesh_files):
    assert mesh_files(Model(type=ModelType.OBJ, name="crate", description="", path=tmp_path / "crate.obj")) == [("visual", str(tmp_path / "crate.obj"))]
    assert mesh_files(Model(type=ModelType.SDF, name="lamp", description=_LAMP_SDF, path=None)) == []


def test_a_link_bundle_with_obj_and_sdf_spawns_from_the_sdf(tmp_path: Path):
    mesh_types = pytest.importorskip("arena_runtime.sim.mujoco_simulator")._MESH_MODEL_TYPES
    bundle = tmp_path / "door"
    for suffix, text in ((".obj", "v 0 0 0\n"), (".sdf", _LAMP_SDF)):
        (bundle / f"door{suffix}").mkdir(parents=True)
        (bundle / f"door{suffix}" / f"door{suffix}").write_text(text)
    model = asyncio.run(ObjectView(bundle).model.get(mesh_types))
    assert (model.type, model.path) == (ModelType.SDF, bundle / "door.sdf" / "door.sdf")
