"""An external articulation over a prismatic joint needs the joint's right-hand axis.

``ExternalArticulationConstraint`` measures a prismatic joint by the offset of
its anchors along the left-hand and the right-hand axis, and reads the
right-hand axis of a joint given by local positions from ``r_position0`` and
``r_position1``. Equal right-hand end points give no axis: the articulation
would measure half of the joint's displacement and drive the body twice as far
as it is told. They are refused at ``World.init`` with the sentence of an edge
with equal end points.

The scene: a fixed box and one without a kinetic term on a prismatic joint
along y given by local positions, which an articulation holds ``STEP`` along the
joint from the box's initial pose.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc import Exception as UIPCException
from uipc.constitution import AffineBodyConstitution, AffineBodyPrismaticJoint, ExternalArticulationConstraint
from uipc.geometry import affine_body, label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


FRAMES = 10
STEP = 0.01  # m
BOX = 0.1 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64
)
BOX_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)
FIRST = np.zeros(1, np.int32)


def articulated_prismatic_scene(right_end):
    """``right_end`` is the joint's ``r_position1``; its ``r_position0`` is (-0.4, 0, 0). Returns (scene, the
    driven box)."""
    config = Scene.default_config()
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-6
    config["newton"]["transrate_tol"] = 1e-6
    config["newton"]["semi_implicit"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1e-8
    scene = Scene(config)
    slots = []
    for k, x0 in enumerate((0.0, 0.4)):
        mesh = tetmesh(BOX + np.array([x0, 0.0, 0.0]), BOX_TETS)
        label_surface(mesh)
        label_triangle_orient(mesh)
        AffineBodyConstitution().apply_to(mesh, 1e8, 1e3)
        view(mesh.instances().find(builtin.is_fixed))[0] = int(k == 0)
        view(mesh.instances().find(builtin.external_kinetic))[0] = 1
        if k:  # the state the articulation measures its increment from
            ref = mesh.instances().create("ref_dof_prev", np.zeros((12, 1)))
            view(ref)[:] = affine_body.transform_to_q(mesh.transforms().view())
        slots.append(scene.objects().create(f"box{k}").geometries().create(mesh)[0])
    zero = np.zeros((1, 3))
    l1, r0, r1 = np.array([[0.0, 1.0, 0.0]]), np.array([[-0.4, 0.0, 0.0]]), np.array([right_end], np.float64)
    joint = AffineBodyPrismaticJoint().create_geometry(
        zero, l1, r0, r1, [slots[0]], FIRST, [slots[1]], FIRST, np.array([100.0])
    )
    joint_slot, _ = scene.objects().create("joint").geometries().create(joint)
    articulation = ExternalArticulationConstraint().create_geometry([joint_slot], FIRST)
    view(articulation["joint_joint"].find("mass"))[0] = 1e4
    view(articulation["joint"].find("delta_theta_tilde"))[0] = STEP
    scene.objects().create("articulation").geometries().create(articulation)
    return scene, slots[1]


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_articulated_prismatic_axis_is_an_axis(tmp_path):
    """The control: with right-hand end points 1 m apart along y the articulation holds the box ``STEP``
    along the joint from its initial pose in every frame."""
    scene, box = articulated_prismatic_scene([-0.4, 1.0, 0.0])
    engine = Engine("cuda", str(tmp_path))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    for _ in range(FRAMES):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        T = np.array(view(box.geometry().transforms())).reshape(4, 4)
        assert T[1, 3] == pytest.approx(STEP, abs=1e-6)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_articulated_prismatic_axis_with_equal_right_end_points_is_refused(tmp_path):
    scene, _ = articulated_prismatic_scene([-0.4, 0.0, 0.0])
    engine = Engine("cuda", str(tmp_path))
    world = World(engine)
    with pytest.raises(
        UIPCException, match="Edge positions must not be too close: r_position0 and r_position1 of joint 0"
    ):
        world.init(scene)
