"""A prismatic joint given by local positions needs an axis.

``l_position0`` and ``l_position1`` of a prismatic joint created with local
positions give its axis. Equal end points give none, as for an edge with equal
end points, and are refused at ``World.init`` with the sentence of the edge.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc import Exception as UIPCException
from uipc.constitution import AffineBodyConstitution, AffineBodyPrismaticJoint
from uipc.geometry import label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


FRAMES = 10
BOX = 0.1 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64
)
BOX_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)
FIRST = np.zeros(1, np.int32)


def local_prismatic_scene(end):
    """A fixed box at the origin and a free one beside it, with a spin and a velocity, joined by a
    prismatic joint given by local positions: the axis runs from the fixed box's origin to ``end`` in its
    frame (None: no joint). Returns (scene, free box)."""
    config = Scene.default_config()
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-6
    config["newton"]["transrate_tol"] = 1e-6
    config["newton"]["semi_implicit"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1e-8
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
            V[:3, 3] = [0.5, 0.0, 0.3]
            view(mesh.instances().find(builtin.velocity))[0] = V
        slots.append(scene.objects().create(f"box{k}").geometries().create(mesh)[0])
    fixed, free = slots
    if end is not None:
        zero = np.zeros((1, 3))
        l1 = np.array([end], np.float64)
        r0, r1 = np.array([[-0.4, 0.0, 0.0]]), np.array([[-0.4, 1.0, 0.0]])
        joint = AffineBodyPrismaticJoint().create_geometry(
            zero, l1, r0, r1, [fixed], FIRST, [free], FIRST, np.array([100.0])
        )
        scene.objects().create("joint").geometries().create(joint)
    return scene, free


def run(workspace, scene, slot):
    """The transform of ``slot`` after every frame."""
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    out = []
    for _ in range(FRAMES):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        out.append(np.array(view(slot.geometry().transforms())).reshape(4, 4))
    return np.array(out)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_local_prismatic_axis_is_an_axis(tmp_path):
    """The control: with end points 1 m apart along y the joint holds the box on that line."""
    free = run(tmp_path / "free", *local_prismatic_scene(None))
    held = run(tmp_path / "held", *local_prismatic_scene([0.0, 1.0, 0.0]))
    assert np.abs(held - free).max() > 1e-2


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_local_prismatic_axis_with_equal_end_points_is_refused(tmp_path):
    scene, _ = local_prismatic_scene([0.0, 0.0, 0.0])
    engine = Engine("cuda", str(tmp_path))
    world = World(engine)
    with pytest.raises(
        UIPCException, match="Edge positions must not be too close: l_position0 and l_position1 of joint 0"
    ):
        world.init(scene)
