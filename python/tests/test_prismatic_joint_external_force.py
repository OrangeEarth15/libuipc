"""The external force of a prismatic joint acts at the joint's anchors.

The force ``f`` of ``AffineBodyPrismaticJointExternalForce`` is an internal
force of the two bodies a prismatic joint links: ``-f t`` on one and ``+f t``
on the other. Applied at the joint's anchors, which lie on its axis, the two
forces have one line of action: the pair keeps its linear and its angular
momentum, and its motion does not depend on where each mesh has its origin.
Applied at the origins of the two mesh frames instead, they are a couple
whenever an origin is off the axis.

The scene: two free cubes of 8 kg side by side along x, joined by a prismatic
joint along the x axis whose force of 10 N pushes them apart; no gravity, no
contact, 40 frames. ``height`` is the distance of the cubes' centres from the
axis; each mesh has its origin at its cube's centre plus a shift.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, view
from uipc.constitution import AffineBodyConstitution, AffineBodyPrismaticJoint, AffineBodyPrismaticJointExternalForce
from uipc.geometry import label_surface, label_triangle_orient, linemesh, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


FRAMES, DT = 40, 0.01
FORCE, MASS = 10.0, 8.0
CUBE = 0.1 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float64
)
TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)
CENTRES = (-0.15, 0.15)  # x of the two cubes
ON_AXIS = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
OFF_AXIS = ((0.0, 0.3, 0.0), (0.1, -0.2, 0.05))  # mesh origins relative to the cubes' centres


def run(workspace, height, shifts):
    """Per frame: the transforms of the two bodies and the world positions of the cubes' corners; and
    each body's mass, first and second moment in its mesh frame."""
    config = Scene.default_config()
    config["dt"] = DT
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["newton"]["velocity_tol"] = 1e-6
    config["newton"]["transrate_tol"] = 1e-6
    config["newton"]["semi_implicit"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1e-10
    scene = Scene(config)
    slots = []
    for name, x, shift in zip(("left", "right"), CENTRES, shifts):
        mesh = tetmesh(CUBE - np.asarray(shift), TETS)
        label_surface(mesh)
        label_triangle_orient(mesh)
        AffineBodyConstitution().apply_to(mesh, 1e8)
        T = np.eye(4)
        T[:3, 3] = np.array([x, height, 0.0]) + np.asarray(shift)
        view(mesh.transforms())[0] = T
        slots.append(scene.objects().create(name).geometries().create(mesh)[0])
    joint = linemesh(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), np.array([[0, 1]], np.int32))
    label_surface(joint)
    AffineBodyPrismaticJoint().apply_to(joint, [slots[0]], [slots[1]], 100.0)
    AffineBodyPrismaticJointExternalForce().apply_to(joint, FORCE)
    view(joint.edges().find("external_force/is_constrained"))[:] = 1
    scene.objects().create("joint").geometries().create(joint)
    out = {"T": [], "x": [], "mass": []}
    for slot in slots:
        meta = slot.geometry().meta()
        m, s, S = (np.asarray(meta.find(n).view(), np.float64) for n in ("abd_mass", "abd_mass_x_bar", "abd_mass_x_bar_x_bar"))
        out["mass"].append((float(m.reshape(-1)[0]), s.reshape(3), S.reshape(3, 3)))
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    for frame in range(FRAMES + 1):
        if frame:
            world.advance()
            assert world.is_valid()
            world.retrieve()
        T = [np.asarray(view(slot.geometry().transforms())).reshape(4, 4).copy() for slot in slots]
        out["T"].append(T)
        out["x"].append([(CUBE - np.asarray(sh)) @ t[:3, :3].T + t[:3, 3] for t, sh in zip(T, shifts)])
    return {"T": np.array(out["T"]), "x": np.array(out["x"]), "mass": out["mass"]}


def momenta(r, frame):
    """Linear and angular momentum (about the world origin) of the pair after ``frame``, with the BDF1
    velocities. For a body ``x = p + A x_bar`` of mass ``m``, first moment ``s`` and second moment ``S``:
    ``P = m p' + A' s`` and ``L = m p x p' + p x A' s + A s x p' + axial(A S A'^T)``."""
    P, L = np.zeros(3), np.zeros(3)
    for body, (m, s, S) in enumerate(r["mass"]):
        T1, T0 = r["T"][frame, body], r["T"][frame - 1, body]
        A, p = T1[:3, :3], T1[:3, 3]
        Ad, pd = (T1[:3, :3] - T0[:3, :3]) / DT, (T1[:3, 3] - T0[:3, 3]) / DT
        P += m * pd + Ad @ s
        K = A @ S @ Ad.T
        L += m * np.cross(p, pd) + np.cross(p, Ad @ s) + np.cross(A @ s, pd)
        L += np.array([K[1, 2] - K[2, 1], K[2, 0] - K[0, 2], K[0, 1] - K[1, 0]])
    return P, L


def turn(r):
    """The angle [degrees] about z by which the left cube has turned at the end."""
    edge = r["x"][-1, 0, 1] - r["x"][-1, 0, 0]
    return float(np.degrees(np.arctan2(edge[1], edge[0])))


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("height, shifts", [(0.0, (OFF_AXIS[0], ON_AXIS[1])), (0.0, OFF_AXIS), (0.25, OFF_AXIS)])
def test_isolated_pair_keeps_its_momentum(tmp_path, height, shifts):
    r = run(tmp_path, height, shifts)
    for frame in (1, 10, 20, FRAMES):
        P, L = momenta(r, frame)
        np.testing.assert_allclose(P, 0.0, atol=1e-9)
        np.testing.assert_allclose(L, 0.0, atol=1e-9)
    assert abs(turn(r)) < (1e-4 if height else 1e-9)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_force_through_the_centres_of_mass_only_separates_the_cubes(tmp_path):
    """Each centre follows the BDF1 recurrence of the acceleration ``f / m``."""
    r = run(tmp_path, 0.0, OFF_AXIS)
    slide = FORCE / MASS * DT * DT * FRAMES * (FRAMES + 1) / 2
    for body, sign in ((0, -1.0), (1, 1.0)):
        centre = r["x"][-1, body].mean(axis=0)
        np.testing.assert_allclose(centre, [CENTRES[body] + sign * slide, 0.0, 0.0], rtol=0, atol=1e-6)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("height", [0.0, 0.25])
def test_motion_does_not_depend_on_the_mesh_frames(tmp_path, height):
    centred = run(tmp_path / "centred", height, ON_AXIS)
    shifted = run(tmp_path / "shifted", height, OFF_AXIS)
    assert np.abs(centred["x"][-1] - centred["x"][0]).max() > 0.1
    np.testing.assert_allclose(shifted["x"], centred["x"], rtol=0, atol=1e-10)
