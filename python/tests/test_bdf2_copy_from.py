"""A state accessor's ``copy_from`` under BDF2 restarts the history of what it writes.

After ``copy_from`` the written bodies and vertices continue from the written
state at the written velocity: their BDF2 history restarts one step back,
``x_(n-1) = x - h v``, ``v_(n-1) = v``, as at the first step
(``test_bdf2_history.py``), so the frames after the write follow the same closed
form from the written state. The rows that are not written keep their history.
"""

import numpy as np
import pytest

from uipc import Matrix4x4, Vector3, builtin, view
from uipc.core import AffineBodyStateAccessorFeature, FiniteElementStateAccessorFeature

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason
from test_bdf2_history import GRAVITY, V_BODY, V_SOFT, Thrown, assert_follows

SHIFT = np.array([0.5, 0.25, 0.0])
SWAPPED = {"soft": V_BODY, "body": V_SOFT}


def write_soft(world, shift, velocity):
    fem = world.features().find(FiniteElementStateAccessorFeature)
    st = fem.create_geometry(0, 4)
    st.vertices().create(builtin.position, Vector3.Zero())
    st.vertices().create(builtin.velocity, Vector3.Zero())
    fem.copy_to(st)
    x = np.array(st.positions().view()).reshape(-1, 3) + shift
    view(st.positions())[:] = x.reshape(-1, 3, 1)
    view(st.vertices().find(builtin.velocity))[:] = np.asarray(velocity).reshape(3, 1)
    fem.copy_from(st)


def write_body(world, shift, velocity):
    abd = world.features().find(AffineBodyStateAccessorFeature)
    st = abd.create_geometry(0, 1)
    st.instances().create(builtin.transform, Matrix4x4.Zero())
    st.instances().create(builtin.velocity, Matrix4x4.Zero())
    abd.copy_to(st)
    T = np.array(st.transforms().view())
    T[0, :3, 3] += shift
    V = np.zeros((1, 4, 4))
    V[0, :3, 3] = velocity
    view(st.transforms())[:] = T
    view(st.instances().find(builtin.velocity))[:] = V
    abd.copy_from(st)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("gravity", [(0.0, 0.0, 0.0), tuple(GRAVITY)])
@pytest.mark.parametrize("velocity_only", [False, True])
def test_restart_after_copy_from(tmp_path, gravity, velocity_only):
    """Both objects thrown again after three frames, moved by ``SHIFT`` unless ``velocity_only``, with
    their velocities exchanged: the frames after the write follow the closed form from it."""
    run = Thrown(tmp_path, gravity=gravity)
    run.advance(3)
    shift = np.zeros(3) if velocity_only else SHIFT
    write_soft(run.world, shift, V_BODY)
    write_body(run.world, shift, V_SOFT)
    assert_follows(run.advance(), SWAPPED, gravity)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_rows_not_written_keep_their_history(tmp_path):
    """Only the body is written: the soft tetrahedron goes on as if nothing happened."""
    run = Thrown(tmp_path, gravity=GRAVITY)
    before = run.advance(3)
    write_body(run.world, SHIFT, V_SOFT)
    after = run.advance()
    soft = np.concatenate([before["soft"], before["soft"][-1] + after["soft"][1:]])
    assert_follows({"soft": soft}, {"soft": V_SOFT}, GRAVITY)
    assert_follows({"body": after["body"]}, {"body": V_SOFT}, GRAVITY)
