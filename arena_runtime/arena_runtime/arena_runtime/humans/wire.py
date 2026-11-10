"""The arena_peds wire: a ped's body pose (BODY.md) from its 40 semantic joint angles."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence

import numpy as np
from arena_people_msgs.msg import BodyPose, Pedestrian
from arena_simulation_setup.shared.body import BODY_POSE_JOINTS, SEMANTIC_DOFS, semantic_to_body_batch
from geometry_msgs.msg import Quaternion, Vector3

from arena_runtime.humans.animation_manager import BodyLayer, nlerp

_SEMANTIC = frozenset(SEMANTIC_DOFS)

BodyCache = MutableMapping[int, tuple[tuple[int, tuple[float, ...]], BodyPose]]
"""The last layer-free body pose built per ped id, under the stance and semantic angles it was built from."""


def fill_body(ped: Pedestrian, stance: Mapping[str, np.ndarray], layers: Sequence[BodyLayer] = ()) -> None:
    """Move the semantic DOFs of a ped's joint_state into its body pose, joint_state keeping only the driven joints, clip body groups blended over it."""
    fill_bodies([ped], [stance], [layers])


def fill_bodies(peds: Sequence[Pedestrian], stances: Sequence[Mapping[str, np.ndarray]], layers: Sequence[Sequence[BodyLayer]], cache: BodyCache | None = None) -> None:
    """`fill_body` of every ped over its own stance and layers, the body math batched per stance.

    With a cache, a ped without layers whose stance and semantic angles repeat the last ones gets that body pose again
    instead of a new one (the same values, built once).
    """
    work: list[tuple[Pedestrian, Mapping[str, np.ndarray], Sequence[BodyLayer], dict[str, float], tuple[float, ...]]] = []
    for ped, stance, ped_layers in zip(peds, stances, layers, strict=True):
        angles = dict(zip(ped.joint_state.name, ped.joint_state.position, strict=True))
        if not any(name in _SEMANTIC for name in angles):
            continue
        row = tuple(float(angles.get(dof, 0.0)) for dof in SEMANTIC_DOFS)
        driven = {name: value for name, value in angles.items() if name not in _SEMANTIC}
        ped.joint_state.name, ped.joint_state.position = list(driven), list(driven.values())
        if cache is not None and not ped_layers and (hit := cache.get(ped.id)) is not None and hit[0] == (id(stance), row):
            ped.body = hit[1]
            continue
        work.append((ped, stance, ped_layers, angles, row))

    groups: dict[int, list[int]] = {}
    for k, (_ped, stance, _layers, _angles, _row) in enumerate(work):
        groups.setdefault(id(stance), []).append(k)
    for members in groups.values():
        stance = work[members[0]][1]
        quats = semantic_to_body_batch(np.array([work[k][4] for k in members]), stance)
        for k, ped_quats in zip(members, quats, strict=True):
            ped, _stance, ped_layers, _angles, row = work[k]
            ped.body = _body_pose(ped_quats, ped_layers)
            if cache is not None and not ped_layers:
                cache[ped.id] = ((id(stance), row), ped.body)


def _body_pose(quats: np.ndarray, layers: Sequence[BodyLayer]) -> BodyPose:
    """The body pose of one ped's (len(BODY_POSE_JOINTS), 4) quaternions, clip body groups blended over them."""
    if not layers:
        return BodyPose(
            joints=list(BODY_POSE_JOINTS),
            rotations=[Quaternion(x=x, y=y, z=z, w=w) for x, y, z, w in quats.tolist()],
            root_offset=Vector3(x=0.0, y=0.0, z=0.0),
        )
    rotations = dict(zip(BODY_POSE_JOINTS, quats, strict=True))
    pelvis = np.zeros(3)
    for layer in layers:
        for joint, (quat, weight) in layer.rotations.items():
            rotations[joint] = np.asarray(nlerp(rotations[joint].tolist(), quat, weight))
        if layer.pelvis is not None:
            pelvis = (1.0 - layer.weight) * pelvis + layer.weight * layer.pelvis
    return BodyPose(
        joints=list(rotations),
        rotations=[Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])) for q in rotations.values()],
        root_offset=Vector3(x=float(pelvis[0]), y=float(pelvis[1]), z=float(pelvis[2])),
    )
