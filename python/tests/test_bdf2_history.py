"""The BDF2 history: the first step, and the first step after a state accessor's ``copy_from``.

BDF2 (``sym/bdf2.inl``) predicts from two states. The history starts one step
back at the current velocity, ``x_(-1) = x_0 - h v_0`` and ``v_(-1) = v_0``:
then a body that moves without forces moves ``h v_0`` in every step, the first
one included. For a free unknown under a constant acceleration ``a`` the
scheme gives

    x_n - x_0 = n h v_0 + a h^2 (n (n - 1) / 2 + 3/4 - (n / 6 + 3/4) / 3^n)

(the second part is the motion from rest, ``test_bdf2_external_force.py``),
and its velocity stays ``v_0`` while ``a = 0``. A history that takes the
initial state for both states instead moves ``2/3 h v_0`` in the first step.
"""

import numpy as np
import pytest

from uipc import Engine, Matrix4x4, Scene, Vector3, World, builtin, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.core import AffineBodyStateAccessorFeature, FiniteElementStateAccessorFeature
from uipc.geometry import label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


H = 0.01
FRAMES = 6
TOL = 1e-12
TET = 0.2 * np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
V_SOFT = np.array([1.0, 0.0, 0.0])
V_BODY = np.array([0.3, 1.0, -0.5])
GRAVITY = np.array([0.0, -9.8, 0.0])
OBJECTS = {"soft": V_SOFT, "body": V_BODY}


def from_rest(n, h):
    n = np.asarray(n, np.float64)
    return h * h * (n * (n - 1) / 2 + 0.75 - (n / 6 + 0.75) / 3.0**n)


def one_step_back(n, h, v, a=(0.0, 0.0, 0.0)):
    """``x_n - x_0`` for the frames ``n`` from the history ``x_(-1) = x_0 - h v``."""
    n = np.asarray(n, np.float64)
    return np.outer(n * h, v) + np.outer(from_rest(n, h), a)


def thrown(gravity=(0.0, 0.0, 0.0), body="free"):
    """A soft tetrahedron thrown at ``V_SOFT`` and an affine one at ``V_BODY`` (``body``: ``"fixed"`` or
    ``"quasi-static"`` all the same, the affine body and the soft tetrahedron's vertices), under
    BDF2, without contact, solved to round-off."""
    config = Scene.default_config()
    config["integrator"]["type"] = "bdf2"
    config["dt"] = H
    config["gravity"] = [[float(g)] for g in gravity]
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-7
    config["newton"]["transrate_tol"] = 1e-7
    config["newton"]["semi_implicit"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1e-12
    scene = Scene(config)
    slots = {}
    for name, x0 in (("soft", 0.0), ("body", 2.0)):
        mesh = tetmesh(TET + np.array([x0, 0.0, 0.0]), np.array([[0, 1, 2, 3]], np.int32))
        label_surface(mesh)
        label_triangle_orient(mesh)
        if name == "soft":
            StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.4), 1e3)
            view(mesh.vertices().find(builtin.velocity))[:] = V_SOFT.reshape(3, 1)
            if body != "free":
                fixed = mesh.vertices().find(builtin.is_fixed)
                if fixed is None:
                    fixed = mesh.vertices().create(builtin.is_fixed, 0)
                view(fixed)[:] = int(body == "fixed")
                dynamic = mesh.vertices().find(builtin.is_dynamic)
                if dynamic is None:
                    dynamic = mesh.vertices().create(builtin.is_dynamic, 1)
                view(dynamic)[:] = int(body != "quasi-static")
        else:
            AffineBodyConstitution().apply_to(mesh, 1e8, 1e3)
            V = np.zeros((4, 4))
            V[:3, 3] = V_BODY
            view(mesh.instances().find(builtin.velocity))[0] = V
            view(mesh.instances().find(builtin.is_fixed))[0] = int(body == "fixed")
            view(mesh.instances().find(builtin.is_dynamic))[0] = int(body != "quasi-static")
        slots[name] = scene.objects().create(name).geometries().create(mesh)[0]
    return scene, slots


def centres(slots):
    soft = np.asarray(view(slots["soft"].geometry().positions())).reshape(-1, 3).mean(axis=0)
    T = np.asarray(view(slots["body"].geometry().transforms())).reshape(4, 4)
    return {"soft": soft, "body": T[:3, 3].copy()}


def velocities(world):
    """The velocities of the soft tetrahedron's vertices and of the body's translation, from the
    engine."""
    fem = world.features().find(FiniteElementStateAccessorFeature)
    abd = world.features().find(AffineBodyStateAccessorFeature)
    st = fem.create_geometry(0, 4)
    st.vertices().create(builtin.velocity, Vector3.Zero())
    fem.copy_to(st)
    soft = np.asarray(st.vertices().find(builtin.velocity).view()).reshape(-1, 3)
    st = abd.create_geometry(0, 1)
    st.instances().create(builtin.velocity, Matrix4x4.Zero())
    abd.copy_to(st)
    body = np.asarray(st.instances().find(builtin.velocity).view()).reshape(4, 4)[:3, 3]
    return soft, body


class Thrown:
    def __init__(self, workspace, **scene):
        self.scene, self.slots = thrown(**scene)
        self.engine = Engine("cuda", str(workspace))  # a World holds its engine weakly
        self.world = World(self.engine)
        self.world.init(self.scene)
        assert self.world.is_valid()

    def advance(self, frames=FRAMES, check_velocities=None):
        """The displacements of the two centres over ``frames`` frames."""
        self.world.retrieve()
        rows = [centres(self.slots)]
        for _ in range(frames):
            self.world.advance()
            assert self.world.is_valid()
            self.world.retrieve()
            rows.append(centres(self.slots))
            if check_velocities is not None:
                check_velocities(*velocities(self.world))
        return {k: np.array([r[k] for r in rows]) - rows[0][k] for k in OBJECTS}


def assert_follows(r, velocities, a=(0.0, 0.0, 0.0), h=H):
    n = np.arange(len(next(iter(r.values()))))
    for name, v in velocities.items():
        np.testing.assert_allclose(r[name], one_step_back(n, h, v, a), rtol=0, atol=TOL, err_msg=name)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_constant_velocity(tmp_path):
    """Without a force every frame moves ``h v_0``, the first one included, and the velocity stays."""

    def check(soft, body):
        np.testing.assert_allclose(soft, np.tile(V_SOFT, (4, 1)), rtol=0, atol=1e-10)
        np.testing.assert_allclose(body, V_BODY, rtol=0, atol=1e-10)

    r = Thrown(tmp_path).advance(check_velocities=check)
    assert_follows(r, OBJECTS)
    np.testing.assert_allclose(r["soft"][1], H * V_SOFT, rtol=0, atol=TOL)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_constant_acceleration(tmp_path):
    r = Thrown(tmp_path, gravity=GRAVITY).advance()
    assert_follows(r, OBJECTS, GRAVITY)
    assert abs(r["body"][-1, 1] - FRAMES * H * V_BODY[1]) > 1e-2  # gravity is at work


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_dt_written_after_init(tmp_path):
    """The history starts with the step it is used in: ``dt`` halved between init and the first
    frame."""
    run = Thrown(tmp_path)
    view(run.scene.config().find("dt"))[0] = H / 2
    assert_follows(run.advance(), OBJECTS, h=H / 2)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("body", ["fixed", "quasi-static"])
def test_body_without_a_velocity_in_the_prediction(tmp_path, body):
    """A fixed or quasi-static body or vertex given a velocity stays where it is, at rest: its
    history starts at rest too (one step back at the given velocity, BDF2's velocity update would
    report ``-v_0 / 2`` after the first step)."""

    def check(soft, body_velocity):
        np.testing.assert_allclose(soft, 0.0, rtol=0, atol=1e-10)
        np.testing.assert_allclose(body_velocity, 0.0, rtol=0, atol=1e-10)

    r = Thrown(tmp_path, body=body).advance(2, check_velocities=check)
    np.testing.assert_allclose(r["body"], 0.0, rtol=0, atol=TOL)
    np.testing.assert_allclose(r["soft"], 0.0, rtol=0, atol=TOL)
