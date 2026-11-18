"""A joint between a body and itself is refused at ``World.init``.

Both sides of such a joint move together: it has no energy, and the assembly of
its Hessian would put the cross block of its two sides on the body's diagonal
block once, without the transpose. Every joint constitution with two sides
(revolute, prismatic, fixed, spherical) refuses it.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc import Exception as UIPCException
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyFixedJoint,
    AffineBodyPrismaticJoint,
    AffineBodyRevoluteJoint,
    AffineBodySphericalJoint,
)
from uipc.geometry import label_surface, label_triangle_orient, linemesh, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


BOX = 0.1 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64
)
BOX_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)
FIRST = np.zeros(1, np.int32)


def self_link_scene(kind):
    """A fixed box and a free spinning one, joined to itself by one joint of ``kind``."""
    config = Scene.default_config()
    config["contact"]["enable"] = False
    scene = Scene(config)
    slots = []
    for k, (x0, fixed) in enumerate(((0.0, True), (0.4, False))):
        mesh = tetmesh(BOX + np.array([x0, 0.0, 0.0]), BOX_TETS)
        label_surface(mesh)
        label_triangle_orient(mesh)
        AffineBodyConstitution().apply_to(mesh, 1e8, 1e3)
        view(mesh.instances().find(builtin.is_fixed))[0] = int(fixed)
        if not fixed:
            V = np.zeros((4, 4))
            V[:3, :3] = [[0.0, -2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            view(mesh.instances().find(builtin.velocity))[0] = V
        slots.append(scene.objects().create(f"box{k}").geometries().create(mesh)[0])
    free = slots[1]
    if kind in ("revolute", "prismatic"):
        end = [0.4, 0.0, 1.0] if kind == "revolute" else [0.4, 1.0, 0.0]
        joint = linemesh(np.array([[0.4, 0.0, 0.0], end]), np.array([[0, 1]], np.int32))
        label_surface(joint)
        base = AffineBodyRevoluteJoint() if kind == "revolute" else AffineBodyPrismaticJoint()
        base.apply_to(joint, [free], FIRST, [free], FIRST, np.array([100.0]))
    elif kind == "fixed":
        joint = AffineBodyFixedJoint().create_geometry([free], FIRST, [free], FIRST, np.array([100.0]))
    else:
        anchor = np.array([[0.4, 0.0, 0.0]])
        joint = AffineBodySphericalJoint().create_geometry(anchor, [free], FIRST, [free], FIRST, np.array([100.0]))
    scene.objects().create("joint").geometries().create(joint)
    return scene


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("kind", ["revolute", "prismatic", "fixed", "spherical"])
def test_joint_between_a_body_and_itself_is_refused(tmp_path, kind):
    engine = Engine("cuda", str(tmp_path))
    world = World(engine)
    with pytest.raises(UIPCException, match=r"Joint 0 of geometry \d+ links geometry \d+ instance 0 to itself"):
        world.init(self_link_scene(kind))
