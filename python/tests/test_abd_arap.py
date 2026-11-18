"""A spinning ARAP affine body (constitution UID 2) follows the rigid rotation.

A free cube (``kappa`` 1e8 Pa) starts spinning at one turn per second about z,
without gravity or contact, for 60 BDF1 frames. In the rigid limit each step
takes the rotation nearest to ``2 A_n - A_(n-1)`` (the minimizer of the kinetic
term over the rotations, for a second moment that is isotropic in the plane of
the spin), so the angle increments obey ``theta_0 = atan(omega dt)``,
``theta_(n+1) = atan2(sin theta_n, 2 - cos theta_n)``: 195.2 degrees after 60
frames. This needs the Hessian of the ARAP energy to be the derivative of its
gradient: with a Hessian laid out for the transpose of ``F`` the Newton steps go
astray for every body that has turned.
"""

import math

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc.constitution import AffineBodyConstitution
from uipc.geometry import label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


FRAMES, OMEGA, DT = 60, 2.0 * math.pi, 0.01
CORNERS = 0.15 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    np.float64,
)
CUBE_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)


def rigid_spin_angles(omega, dt, frames):
    theta, total, out = math.atan(omega * dt), 0.0, []
    for _ in range(frames):
        total += theta
        out.append(total)
        theta = math.atan2(math.sin(theta), 2.0 - math.cos(theta))
    return np.array(out)


def spin(workspace):
    """The angle [degrees] about z after every frame, with the Newton loop converged to 1e-5 m/s."""
    config = Scene.default_config()
    config["dt"] = DT
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["newton"]["semi_implicit"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-5
    config["newton"]["transrate_tol"] = 1e-5
    config["linear_system"]["tol_rate"] = 1e-10
    scene = Scene(config)
    mesh = tetmesh(CORNERS, CUBE_TETS)
    label_surface(mesh)
    label_triangle_orient(mesh)
    abd_config = AffineBodyConstitution.default_config()
    abd_config["name"] = "ARAP"
    AffineBodyConstitution(abd_config).apply_to(mesh, 1e8)
    velocity = np.zeros((4, 4))
    velocity[:3, :3] = [[0.0, -OMEGA, 0.0], [OMEGA, 0.0, 0.0], [0.0, 0.0, 0.0]]  # dA/dt = [omega]x A
    view(mesh.instances().find(builtin.velocity))[0] = velocity
    slot, _ = scene.objects().create("cube").geometries().create(mesh)
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    angles = []
    for _ in range(FRAMES):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        A = np.asarray(view(slot.geometry().transforms())).reshape(4, 4)[:3, :3]
        angles.append(math.atan2(A[1, 0], A[0, 0]))
    return np.degrees(np.unwrap(angles))


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_spinning_arap_body_follows_the_rigid_rotation(tmp_path):
    exact = np.degrees(rigid_spin_angles(OMEGA, DT, FRAMES))
    assert exact[-1] == pytest.approx(195.215, abs=1e-3)
    np.testing.assert_allclose(spin(tmp_path), exact, rtol=0, atol=2e-3)
