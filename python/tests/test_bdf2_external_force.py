"""External forces on affine bodies under BDF2 follow the closed form of the scheme.

Under ``integrator/type = "bdf2"`` the affine-body predictor must add the
acceleration of the external forces (``AffineBodyExternalBodyForce``, the joint
forces of ``AffineBodyPrismaticJointExternalForce`` and
``AffineBodyRevoluteJointExternalForce``) to gravity, as its BDF1 sibling and
both finite-element predictors do.

Reference: BDF2 (``sym/bdf2.inl``) for a free unknown under a constant
acceleration ``a`` and step ``h``,

    x_(n+1) = 4/3 x_n - 1/3 x_(n-1) + 2/9 h (4 v_n - v_(n-1)) + 4/9 h^2 a
    v_(n+1) = (3 x_(n+1) - 4 x_n + x_(n-1)) / (2 h),

has from rest (``x_(-1) = x_0``, ``v_(-1) = v_0 = 0``) the solution

    x_n - x_0 = a h^2 (n (n - 1) / 2 + 3/4 - (n / 6 + 3/4) / 3^n),

which converges at first order in ``h`` to ``a t^2 / 2``. BDF1 gives
``x_n - x_0 = a h^2 n (n + 1) / 2``.
"""

import math

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyExternalBodyForce,
    AffineBodyPrismaticJoint,
    AffineBodyPrismaticJointExternalForce,
    AffineBodyRevoluteJoint,
    AffineBodyRevoluteJointExternalForce,
    ElasticModuli,
    FiniteElementExternalForce,
    StableNeoHookean,
)
from uipc.geometry import label_surface, label_triangle_orient, linemesh, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


SIDE, DENSITY = 0.1, 1e3
MASS = DENSITY * SIDE**3  # 1 kg
INERTIA = MASS * 2 * SIDE**2 / 12  # about an axis through the centre
FORCE = 1.0  # N: the body force and the prismatic joint force
TORQUE = 0.01  # N m
TET = SIDE * np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
VERTEX_FORCE = 0.01  # N on each vertex of the soft tetrahedron
VERTEX_MASS = DENSITY * SIDE**3 / 6 / 4
END = 0.1  # s
STEPS = (0.01, 0.005, 0.0025)
CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    np.float64,
)
CUBE_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)


def bdf2_from_rest(n, h, a):
    n = np.asarray(n, np.float64)
    return a * h * h * (n * (n - 1) / 2 + 0.75 - (n / 6 + 0.75) / 3.0**n)


def bdf1_from_rest(n, h, a):
    n = np.asarray(n, np.float64)
    return a * h * h * n * (n + 1) / 2


def test_closed_form_solves_the_recurrence():
    h, a = 0.01, 3.0
    x, v = [0.0, 0.0], [0.0, 0.0]  # x_(-1), x_0 and v_(-1), v_0
    for _ in range(40):
        x.append(4 / 3 * x[-1] - x[-2] / 3 + 2 / 9 * h * (4 * v[-1] - v[-2]) + 4 / 9 * h * h * a)
        v.append((3 * x[-1] - 4 * x[-2] + x[-3]) / (2 * h))
    np.testing.assert_allclose(bdf2_from_rest(np.arange(41), h, a), x[1:], rtol=1e-12, atol=0)


def cube(scene, name, centre, fixed=False):
    """An affine cube of 1 kg whose mesh is centred at its origin, placed at ``centre``."""
    mesh = tetmesh(CORNERS * SIDE / 2, CUBE_TETS)
    label_surface(mesh)
    label_triangle_orient(mesh)
    AffineBodyConstitution().apply_to(mesh, 1e8, DENSITY)
    T = np.eye(4)
    T[:3, 3] = centre
    view(mesh.transforms())[0] = T
    view(mesh.instances().find(builtin.is_fixed))[0] = int(fixed)
    return scene.objects().create(name).geometries().create(mesh)[0]


def axis(p0, p1):
    joint = linemesh(np.array([p0, p1], np.float64), np.array([[0, 1]], np.int32))
    label_surface(joint)
    return joint


def forced_scene(integrator, dt):
    """``body`` under a body force, ``slider`` pushed by a prismatic joint force away from a fixed cube,
    ``spinner`` turned by a revolute joint torque about an axis through its centre, and a soft
    tetrahedron under a vertex force; no gravity, no contact, solved to round-off."""
    config = Scene.default_config()
    config["integrator"]["type"] = integrator
    config["dt"] = dt
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-7
    config["newton"]["transrate_tol"] = 1e-7
    config["newton"]["semi_implicit"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1e-12
    scene = Scene(config)

    slots = {"body": cube(scene, "body", (0.0, 0.0, 0.0))}
    force = np.zeros(12)
    force[0] = FORCE
    body = slots["body"].geometry()
    AffineBodyExternalBodyForce().apply_to(body, force)

    first = np.zeros(1, np.int32)
    anchor = cube(scene, "slider anchor", (0.0, 1.0, 0.0), fixed=True)
    slots["slider"] = cube(scene, "slider", (0.0, 1.0, 0.5))
    joint = axis((0.0, 1.0, 0.5), (0.0, 1.0, 1.0))  # through both centres, anchored at the slider's
    AffineBodyPrismaticJoint().apply_to(joint, [anchor], [slots["slider"]], 100.0)
    AffineBodyPrismaticJointExternalForce().apply_to(joint, FORCE)
    view(joint.edges().find("external_force/is_constrained"))[:] = 1
    scene.objects().create("prismatic").geometries().create(joint)

    anchor = cube(scene, "spinner anchor", (1.0, 1.0, 0.0), fixed=True)
    slots["spinner"] = cube(scene, "spinner", (1.0, 1.0, 0.5))
    joint = axis((1.0, 1.0, 0.0), (1.0, 1.0, 1.0))
    AffineBodyRevoluteJoint().apply_to(joint, [anchor], first, [slots["spinner"]], first, np.array([100.0]))
    AffineBodyRevoluteJointExternalForce().apply_to(joint, TORQUE)
    view(joint.edges().find("external_torque/is_constrained"))[:] = 1
    scene.objects().create("revolute").geometries().create(joint)

    soft = tetmesh(TET + np.array([2.0, 0.0, 0.0]), np.array([[0, 1, 2, 3]], np.int32))
    label_surface(soft)
    label_triangle_orient(soft)
    StableNeoHookean().apply_to(soft, ElasticModuli.youngs_poisson(1e5, 0.4), DENSITY)
    FiniteElementExternalForce().apply_to(soft, np.array([VERTEX_FORCE, 0.0, 0.0]))
    slots["soft"] = scene.objects().create("soft").geometries().create(soft)[0]
    return scene, slots


def state(slots):
    out = {}
    for name in ("body", "slider", "spinner"):
        T = np.asarray(view(slots[name].geometry().transforms())).reshape(4, 4)
        out[name] = T[:3, 3].copy()
        if name == "spinner":
            out["angle"] = math.atan2(T[1, 0], T[0, 0])
    out["soft"] = np.asarray(view(slots["soft"].geometry().positions())).reshape(-1, 3).mean(axis=0)
    return out


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    cache = {}

    def run(integrator, dt):
        """Displacements of every object after every frame up to ``END``."""
        key = (integrator, dt)
        if key not in cache:
            scene, slots = forced_scene(integrator, dt)
            engine = Engine("cuda", str(tmp_path_factory.mktemp(f"{integrator}_{dt}")))
            world = World(engine)
            world.init(scene)
            assert world.is_valid()
            world.retrieve()
            rows = [state(slots)]
            for _ in range(round(END / dt)):
                world.advance()
                assert world.is_valid()
                world.retrieve()
                rows.append(state(slots))
            cache[key] = {k: np.array([r[k] for r in rows]) - rows[0][k] for k in rows[0]}
        return cache[key]

    return run


ACCELERATIONS = {
    "body": (0, FORCE / MASS),
    "slider": (2, FORCE / MASS),
    "soft": (0, VERTEX_FORCE / VERTEX_MASS),
}


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("dt", STEPS)
def test_bdf2_forces_follow_the_closed_form(runs, dt):
    r = runs("bdf2", dt)
    n = np.arange(round(END / dt) + 1)
    for name, (k, a) in ACCELERATIONS.items():
        expected = np.zeros((len(n), 3))
        expected[:, k] = bdf2_from_rest(n, dt, a)
        assert expected[-1, k] > 1e-3
        np.testing.assert_allclose(r[name], expected, rtol=0, atol=1e-11, err_msg=name)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("dt", STEPS)
def test_bdf2_torque_follows_the_closed_form(runs, dt):
    """The angle is the closed form with ``tau / I`` up to the stiffness of the body (``q`` is not an
    angle), and the centre stays on the axis."""
    r = runs("bdf2", dt)
    n = np.arange(round(END / dt) + 1)
    expected = bdf2_from_rest(n, dt, TORQUE / INERTIA)
    assert expected[-1] > 2.5e-2
    np.testing.assert_allclose(r["angle"], expected, rtol=0, atol=2e-6)
    assert np.abs(r["spinner"]).max() < 1e-11


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_bdf2_converges_to_the_continuous_motion(runs):
    """``F t^2 / (2 m)`` and ``tau t^2 / (2 I)`` at ``END``, at first order in the step."""
    for a, end in ((FORCE / MASS, lambda r: r["body"][-1, 0]), (TORQUE / INERTIA, lambda r: r["angle"][-1])):
        exact = a * END**2 / 2
        errors = [exact - end(runs("bdf2", dt)) for dt in STEPS]
        for coarse, fine in zip(errors, errors[1:]):
            assert fine / coarse == pytest.approx(0.5, abs=0.06)
        assert abs(errors[-1]) < 0.03 * exact


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_bdf1_forces_follow_the_closed_form(runs):
    """The control: under BDF1 the same forces act, ``x_n - x_0 = a h^2 n (n + 1) / 2``."""
    dt = STEPS[0]
    r = runs("bdf1", dt)
    n = np.arange(round(END / dt) + 1)
    for name, (k, a) in ACCELERATIONS.items():
        expected = np.zeros((len(n), 3))
        expected[:, k] = bdf1_from_rest(n, dt, a)
        np.testing.assert_allclose(r[name], expected, rtol=0, atol=1e-11, err_msg=name)
    np.testing.assert_allclose(r["angle"], bdf1_from_rest(n, dt, TORQUE / INERTIA), rtol=0, atol=2e-6)
