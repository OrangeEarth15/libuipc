"""A joint force or torque acts on the joints of the geometry that carries it.

``AffineBodyRevoluteJointExternalForce`` and ``AffineBodyPrismaticJointExternalForce``
need not be applied to every joint geometry of their base joint. The scene: two
joints of one kind, each between a fixed box and a free one, in two joint
geometries; no gravity, no contact. The second geometry carries a torque of
2 N m (revolute) or a force of 2 N (prismatic). Its joint moves and the other
one stays, by the same amount as when both geometries carry the constitution
and the first one carries zero.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyPrismaticJoint,
    AffineBodyPrismaticJointExternalForce,
    AffineBodyRevoluteJoint,
    AffineBodyRevoluteJointExternalForce,
)
from uipc.geometry import label_surface, label_triangle_orient, linemesh, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


FRAMES = 10
LOAD = 2.0  # N m or N
BOX = 0.1 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64
)
BOX_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)
KINDS = {
    # base joint, its load, the load's attributes, the joint state it moves
    "revolute": (AffineBodyRevoluteJoint, AffineBodyRevoluteJointExternalForce, "external_torque", "angle"),
    "prismatic": (AffineBodyPrismaticJoint, AffineBodyPrismaticJointExternalForce, "external_force", "distance"),
}


def joints(kind, on_both):
    """(scene, the two joint slots). ``on_both``: both joint geometries carry the load constitution, the
    first with 0; otherwise only the second carries it."""
    base, load, attribute, _ = KINDS[kind]
    config = Scene.default_config()
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-5
    config["newton"]["transrate_tol"] = 1e-5
    config["newton"]["semi_implicit"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1e-10
    scene = Scene(config)
    first = np.zeros(1, np.int32)
    slots = []
    for k, z in enumerate((0.0, 1.0)):
        boxes = []
        for x0, fixed in ((0.0, True), (0.4, False)):
            mesh = tetmesh(BOX + np.array([x0, 0.0, z]), BOX_TETS)
            label_surface(mesh)
            label_triangle_orient(mesh)
            AffineBodyConstitution().apply_to(mesh, 1e8, 1e3)
            view(mesh.instances().find(builtin.is_fixed))[0] = int(fixed)
            boxes.append(scene.objects().create(f"box{k}{int(fixed)}").geometries().create(mesh)[0])
        if kind == "revolute":  # about z, between the boxes
            ends = np.array([[0.2, 0.0, z - 0.5], [0.2, 0.0, z + 0.5]])
        else:  # along x, through the boxes' centres
            ends = np.array([[0.2, 0.0, z], [0.6, 0.0, z]])
        joint = linemesh(ends, np.array([[0, 1]], np.int32))
        label_surface(joint)
        base().apply_to(joint, [boxes[0]], first, [boxes[1]], first, np.array([100.0]))
        if k == 1 or on_both:
            load().apply_to(joint, LOAD if k == 1 else 0.0)
            view(joint.edges().find(f"{attribute}/is_constrained"))[:] = 1
        slots.append(scene.objects().create(f"joint{k}").geometries().create(joint)[0])
    return scene, slots


def joint_states(workspace, kind, on_both):
    """The angles (distances) of the two joints after ``FRAMES`` frames."""
    scene, slots = joints(kind, on_both)
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    for _ in range(FRAMES):
        world.advance()
        assert world.is_valid()
    world.retrieve()
    state = KINDS[kind][3]
    return np.array([float(np.asarray(view(s.geometry().edges().find(state))).reshape(-1)[0]) for s in slots])


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("kind", list(KINDS))
def test_load_on_the_second_geometry_moves_its_joint(tmp_path, kind):
    alone = joint_states(tmp_path / "alone", kind, on_both=False)
    both = joint_states(tmp_path / "both", kind, on_both=True)
    assert abs(both[1]) > 1e-3 and both[0] == 0.0
    assert alone[0] == 0.0
    np.testing.assert_allclose(alone, both, rtol=0, atol=1e-11)
