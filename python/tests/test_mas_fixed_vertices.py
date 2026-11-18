"""Regression: fixed finite-element vertices under the MAS preconditioner.

The finite-element subsystem turns every Hessian block of a fixed vertex into
zero, its diagonal blocks into the identity, and its gradient into zero, so
the row of a fixed vertex stays zero through the PCG and the vertex keeps its
position, as ``is_fixed`` says. The diagonal preconditioner keeps it so. MAS
summed the identity blocks into its coarse levels and wrote the coarse
corrections onto the fixed row of ``z``, so the PCG iterated that row and the
step moved the vertex. MAS is now the preconditioner of the free unknowns: a
fixed vertex adds nothing to any level, and its row of ``z`` is zero.

References: the invariant (a fixed vertex is where it started, bit for bit,
after any number of frames) and the consistent path (with tight PCG and Newton
tolerances the runs with MAS and with the diagonal preconditioner agree in
every frame).
"""

import numpy as np
import pytest

from uipc import Engine, Scene, Transform, World, builtin, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


def block_tets(n, size):
    """A cube of edge ``size`` as (n + 1)^3 vertices and 5 n^3 positively
    oriented tetrahedra."""
    g = np.linspace(0.0, size, n + 1)
    x = np.array([[a, b, c] for a in g for b in g for c in g])

    def idx(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    even = ((0, 4, 2, 1), (3, 1, 2, 7), (5, 4, 1, 7), (6, 2, 4, 7), (1, 2, 4, 7))
    odd = ((1, 0, 3, 5), (2, 3, 0, 6), (4, 5, 6, 0), (7, 6, 5, 3), (0, 3, 5, 6))
    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]
                tets += [[c[v] for v in t] for t in (even if (i + j + k) % 2 == 0 else odd)]
    tets = np.array(tets, dtype=np.int32)
    flip = np.linalg.det(x[tets[:, 1:]] - x[tets[:, :1]]) < 0
    tets[flip, 2], tets[flip, 3] = tets[flip, 3].copy(), tets[flip, 2].copy()
    return x, tets


def hanging_block(workspace, preconditioner, frames, constitution="ipc", with_body=False,
                  tight=False, keep=0):
    """A soft block of 27 vertices that hangs from the nine fixed vertices of
    its top face, no contact. ``with_body``: an affine body far away.
    ``keep``: linear_system/keep_preconditioner. Returns the block's positions
    after every frame (frame 0 first) and the indices of the fixed vertices."""
    config = Scene.default_config()
    config["contact"]["enable"] = False
    config["contact"]["constitution"] = constitution
    config["linear_system"]["fem_preconditioner"] = preconditioner
    config["linear_system"]["keep_preconditioner"] = keep
    if tight:
        config["linear_system"]["tol_rate"] = 1e-10
        config["newton"]["velocity_tol"] = 1e-7
    scene = Scene(config)
    x, tets = block_tets(2, 0.2)
    x[:, 1] += 1.0
    mesh = tetmesh(x, tets)
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.45), 1e3)
    top = np.flatnonzero(x[:, 1] > 1.2 - 1e-9)
    view(mesh.vertices().find(builtin.is_fixed))[top] = 1
    slot, _ = scene.objects().create("block").geometries().create(mesh)
    if with_body:
        body = tetmesh(0.2 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
                       np.array([[0, 1, 2, 3]], dtype=np.int32))
        label_surface(body)
        label_triangle_orient(body)
        AffineBodyConstitution().apply_to(body, 1e8)
        t = Transform.Identity()
        t.translate(np.array([3.0, 1.0, 0.0]))
        view(body.transforms())[0] = t.matrix()
        scene.objects().create("body").geometries().create(body)

    engine = Engine("cuda", str(workspace))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    out = [np.array(view(slot.geometry().positions())).reshape(-1, 3)]
    for _ in range(frames):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        out.append(np.array(view(slot.geometry().positions())).reshape(-1, 3))
    return np.array(out), top


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("preconditioner", ["mas", "diag"])
@pytest.mark.parametrize("constitution", ["ipc", "al-ipc"])
@pytest.mark.parametrize("with_body", [False, True], ids=["fem", "with_body"])
@pytest.mark.parametrize("keep", [0, 1], ids=["assembled", "kept"])
def test_fixed_vertices_keep_their_position(tmp_path, preconditioner, constitution, with_body,
                                            keep):
    x, top = hanging_block(tmp_path, preconditioner, 10, constitution, with_body, keep=keep)
    # before the fix, under MAS: 4.0e-6 m off after the first frame
    np.testing.assert_array_equal(x[:, top], np.broadcast_to(x[0, top], x[:, top].shape))
    # the free vertices do move
    assert np.abs(x[-1] - x[0]).max() > 1e-3


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_mas_and_diag_agree_when_the_solves_are_tight(tmp_path):
    mas, _ = hanging_block(tmp_path / "mas", "mas", 20, tight=True)
    diag, _ = hanging_block(tmp_path / "diag", "diag", 20, tight=True)
    np.testing.assert_allclose(mas, diag, rtol=0, atol=1e-10)
