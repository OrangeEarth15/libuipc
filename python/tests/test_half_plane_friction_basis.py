"""Regression: friction on a half plane whose normal points along -x.

The tangent basis of vertex / half-plane friction takes
``e1 = normalize(trial x N)`` with ``trial`` the x axis, or the z axis when
the normal is close to x. The test was ``N . x > 0.9``, which misses the
opposite direction: for ``N = (-1, 0, 0)`` the cross product is zero, both
tangents are zero and the plane has no friction, under IPC and AL-IPC. The
test is now ``|N . x| > 0.9``.

The reference is mirror symmetry: a cube pressed on a wall and pulled along it
gives, for the wall facing -x, the mirror image of its trajectory on the wall
facing +x. Lagged friction does not depend on which orthonormal tangent basis
is taken, so the two runs agree to round-off.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, Transform, World, view
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import halfplane, label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

S = 0.1  # half edge of the cube
CUBE = S * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
CUBE_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]],
                     dtype=np.int32)
FRAMES = 50
DT = 0.01
# the pull along the wall, 2 m/s^2, without friction: the cube's drop in 50 frames
FRICTIONLESS = 2.0 * DT**2 * FRAMES * (FRAMES + 1) / 2


def slide(workspace, sign, constitution, friction=True):
    """The cube's centre after every frame. The cube starts 5 mm from the wall
    x = 0 whose normal is ``(sign, 0, 0)``; gravity presses it on the wall with
    9.8 m/s^2 and pulls it along the wall with 2 m/s^2; friction coefficient
    0.5. ``sign = -1`` is the mirror image of ``sign = 1``."""
    config = Scene.default_config()
    config["gravity"] = [[-9.8 * sign], [-2.0], [0.0]]
    config["contact"]["friction"]["enable"] = friction
    config["contact"]["d_hat"] = 0.01
    config["contact"]["constitution"] = constitution
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)
    mesh = tetmesh(CUBE, CUBE_TETS)
    label_surface(mesh)
    label_triangle_orient(mesh)
    AffineBodyConstitution().apply_to(mesh, 1e8)
    t = Transform.Identity()
    t.translate(np.array([sign * (S + 0.005), 0.0, 0.0]))
    view(mesh.transforms())[0] = t.matrix()
    slot, _ = scene.objects().create("cube").geometries().create(mesh)
    wall = halfplane(np.zeros(3), np.array([sign, 0.0, 0.0]))
    scene.objects().create("wall").geometries().create(wall)

    engine = Engine("cuda", str(workspace))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    out = []
    for _ in range(FRAMES):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        out.append(np.array(view(slot.geometry().transforms())).reshape(4, 4)[:3, 3].copy())
    return np.array(out)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("constitution", ["ipc", "al-ipc"])
def test_mirrored_walls_give_mirrored_trajectories(tmp_path, constitution):
    plus = slide(tmp_path / "plus", 1.0, constitution)
    minus = slide(tmp_path / "minus", -1.0, constitution)
    # friction holds the cube on the wall facing +x: it slides 2 mm (IPC) and
    # 98 mm (AL-IPC) of the 255 mm it falls without friction
    assert -plus[-1, 1] < 0.5 * FRICTIONLESS
    # before the fix the wall facing -x had no friction: 255 mm
    np.testing.assert_allclose(minus * [-1.0, 1.0, 1.0], plus, rtol=0, atol=1e-8)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_without_friction_both_walls_give_the_closed_form(tmp_path):
    for sign in (1.0, -1.0):
        fall = -slide(tmp_path / str(sign), sign, "ipc", friction=False)[-1, 1]
        assert abs(fall - FRICTIONLESS) < 1e-9
