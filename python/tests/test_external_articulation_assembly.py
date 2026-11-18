"""The matrix an external articulation assembles is ``J^T M J`` with all its cross blocks.

``ExternalArticulationConstraint`` has the energy ``1/2 r^T M r`` over the
residuals ``r_i = delta_theta_i - delta_theta_tilde_i`` of its joints. Its
Gauss-Newton matrix is ``J^T M J``: for the joints ``i`` and ``j`` the term
``M_ij (g_i g_j^T + g_j g_i^T)``, ``g_i = d delta_theta_i / dq``, with blocks on
every pair of a body of joint ``i`` and a body of joint ``j``.

The scene: three free cubes, a revolute and a prismatic joint between them and
one articulation over both, no gravity, no contact, at rest with
``delta_theta_tilde = 0``. In the first Newton iteration of the first frame
``r = 0``, so the articulation adds exactly ``J^T M J`` and nothing else, and
the matrices of two runs that differ only in ``M_01`` differ by
``M_01 (g_0 g_1^T + g_1 g_0^T)``. ``extras/debug/dump_linear_system`` gives the
stored upper 3x3 blocks of each run's matrix; ``g_0`` and ``g_1`` come from
differences of ``delta_theta`` as the articulation's documentation defines it
(``docs/specification/constitutions/external_articulation_constraint.md``). The
joints link the bodies (left, right) in three ways: a chain (0, 1), (1, 2); two
joints whose left body 1 lies between their right bodies, (1, 0), (1, 2); and
two joints whose left body 0 has the smallest index, (0, 1), (0, 2).
"""

import glob
import math
import os

import numpy as np
import pytest

from uipc import Engine, Scene, World, view
from uipc.constitution import (
    AffineBodyConstitution,
    AffineBodyPrismaticJoint,
    AffineBodyRevoluteJoint,
    ExternalArticulationConstraint,
)
from uipc.geometry import label_surface, label_triangle_orient, linemesh, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


CORNERS = 0.2 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    np.float64,
)
CUBE_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)
HEIGHTS = (-0.8, 0.0, 0.8)
HINGE = np.array([[-0.5, 0.0, -0.4], [0.5, 0.0, -0.4]])  # revolute joint about x
SLIDER = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.4]])  # prismatic joint along z
# the (left, right) bodies of the revolute and of the prismatic joint
CASES = {
    "chain": ((0, 1), (1, 2)),
    "shared_middle": ((1, 0), (1, 2)),
    "shared_first": ((0, 1), (0, 2)),
}
M_DIAG, M_01 = 64.0, 8.0


def rotation(k):
    """A small rotation of body ``k`` about (1, 0, 0.2): the blocks of the joints are dense."""
    axis = np.array([1.0, 0.0, 0.2]) / math.hypot(1.0, 0.2)
    angle = 0.1 * k
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def transforms():
    out = []
    for k, z in enumerate(HEIGHTS):
        T = np.eye(4)
        T[:3, :3] = rotation(k)
        T[2, 3] = z
        out.append(T)
    return out


def scene_of(case, m01):
    config = Scene.default_config()
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["enable"] = False
    config["extras"]["debug"]["dump_linear_system"] = True
    scene = Scene(config)
    mesh = tetmesh(CORNERS, CUBE_TETS)
    mesh.instances().resize(3)
    label_surface(mesh)
    label_triangle_orient(mesh)
    AffineBodyConstitution().apply_to(mesh, 100e6)
    view(mesh.transforms())[:] = np.array(transforms())
    slot, _ = scene.objects().create("links").geometries().create(mesh)

    def instance(i):
        return np.array([i], np.int32)

    (rl, rr), (pl, pr) = CASES[case]
    edge = np.array([[0, 1]], np.int32)
    hinge = linemesh(HINGE, edge)
    AffineBodyRevoluteJoint().apply_to(hinge, [slot], instance(rl), [slot], instance(rr), np.array([100.0]))
    hinge_slot, _ = scene.objects().create("hinge").geometries().create(hinge)
    slider = linemesh(SLIDER, edge)
    AffineBodyPrismaticJoint().apply_to(slider, [slot], instance(pl), [slot], instance(pr), np.array([100.0]))
    slider_slot, _ = scene.objects().create("slider").geometries().create(slider)
    articulation = ExternalArticulationConstraint().create_geometry(
        [hinge_slot, slider_slot], np.zeros(2, np.int32)
    )
    view(articulation["joint_joint"].find("mass"))[:] = np.array([M_DIAG, m01, m01, M_DIAG])
    scene.objects().create("articulation").geometries().create(articulation)
    return scene


def first_matrix(workspace, case, m01):
    """The stored entries of the matrix of frame 1, Newton iteration 0, as a dense array."""
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene_of(case, m01))
    assert world.is_valid()
    world.advance()
    assert world.is_valid()
    (path,) = glob.glob(os.path.join(str(workspace), "**", "A.1.0.mtx"), recursive=True)
    with open(path) as f:
        lines = [line.split() for line in f if not line.startswith("%")]
    rows, cols, _ = (int(v) for v in lines[0])
    A = np.zeros((rows, cols))
    for i, j, value in lines[1:]:
        A[int(i) - 1, int(j) - 1] += float(value)
    return A


# ---- delta_theta as the documentation defines it, q = (p, rows of A) per body
def point(q, x):
    return q[0] + q[1:] @ x


def vector(q, v):
    return q[1:] @ v


def orthonormal_basis(t):
    """libuipc's frame of an axis (``affine_body/utils``): ``n = t x e``, ``b = t x n``."""
    t = t / np.linalg.norm(t)
    e = np.array([0.0, 1.0, 0.0]) if abs(t[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
    n = np.cross(t, e)
    n /= np.linalg.norm(n)
    b = np.cross(t, n)
    return n, b / np.linalg.norm(b)


def joint_function(kind, left, right):
    """``delta_theta(z, z_prev)`` of a joint between the bodies ``left`` and ``right`` (z: all bodies'
    q, 36 entries)."""
    Ts = transforms()
    TL, TR = Ts[left], Ts[right]
    if kind == "revolute":
        n, b = orthonormal_basis(HINGE[1] - HINGE[0])
        basis = [TL[:3, :3].T @ n, TL[:3, :3].T @ b, TR[:3, :3].T @ n, TR[:3, :3].T @ b]

        def frame(z):
            qk, ql = z.reshape(3, 4, 3)[left], z.reshape(3, 4, 3)[right]
            nk, bk = vector(qk, basis[0]), vector(qk, basis[1])
            nl, bl = vector(ql, basis[2]), vector(ql, basis[3])
            return 0.5 * (nk @ nl + bk @ bl), 0.5 * (bk @ nl - nk @ bl)

        def delta_theta(z, z_prev):
            c, s = frame(z)
            ct, st = frame(z_prev)
            return math.atan2(s * ct - c * st, c * ct + s * st)

        return delta_theta

    t = (SLIDER[1] - SLIDER[0]) / np.linalg.norm(SLIDER[1] - SLIDER[0])
    c = SLIDER[0]
    basis = [
        TL[:3, :3].T @ (c - TL[:3, 3]),
        TL[:3, :3].T @ t,
        TR[:3, :3].T @ (c - TR[:3, 3]),
        TR[:3, :3].T @ t,
    ]

    def theta(z):
        qk, ql = z.reshape(3, 4, 3)[left], z.reshape(3, 4, 3)[right]
        ck, tk = point(qk, basis[0]), vector(qk, basis[1])
        cl, tl = point(ql, basis[2]), vector(ql, basis[3])
        return 0.5 * ((cl - ck) @ tk + (cl - ck) @ tl)

    return lambda z, z_prev: theta(z) - theta(z_prev)


def jacobian(case):
    """``g_0``, ``g_1`` at rest, by central differences."""
    z0 = np.concatenate([np.vstack([T[:3, 3], T[:3, :3]]).reshape(-1) for T in transforms()])
    (rl, rr), (pl, pr) = CASES[case]
    out = []
    for f in (joint_function("revolute", rl, rr), joint_function("prismatic", pl, pr)):
        g = np.zeros(36)
        for k in range(36):
            e = np.zeros(36)
            e[k] = 1e-6
            g[k] = (f(z0 + e, z0) - f(z0 - e, z0)) / 2e-6
        out.append(g)
    return out


def stored_half(M):
    """What the linear system stores of a symmetric matrix: its 3x3 blocks on and above the diagonal."""
    S = M.copy()
    for r in range(M.shape[0] // 3):
        for c in range(r):
            S[3 * r : 3 * r + 3, 3 * c : 3 * c + 3] = 0.0
    return S


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("case", list(CASES))
def test_articulation_assembles_every_cross_block(tmp_path, case):
    with_coupling = first_matrix(tmp_path / "coupled", case, M_01)
    without = first_matrix(tmp_path / "uncoupled", case, 0.0)
    assert with_coupling.shape == (36, 36)
    g0, g1 = jacobian(case)
    cross = M_01 * (np.outer(g0, g1) + np.outer(g1, g0))
    assert np.abs(cross).max() > 1.0
    np.testing.assert_allclose(
        with_coupling - without, stored_half(cross), rtol=0, atol=1e-6 * np.abs(cross).max()
    )
