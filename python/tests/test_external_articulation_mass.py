"""An external articulation's ``mass`` must be symmetric.

``ExternalArticulationConstraint`` has the energy ``1/2 r^T M r`` over the
residuals of its joints. The gradient of that energy is ``J^T (M + M^T) r / 2``;
the constraint assembles ``J^T M r`` and ``J^T M J``, the derivatives of the
energy only for a symmetric ``M``. A ``mass`` whose mirrored entries differ by
more than rounding (1e-12 of its largest entry), or that is not a number, is
refused: at ``World.init``, and before the frame for which it is written later.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc import Exception as UIPCException
from uipc.constitution import AffineBodyConstitution, AffineBodyRevoluteJoint, ExternalArticulationConstraint
from uipc.geometry import affine_body, label_surface, label_triangle_orient, linemesh, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason


SYMMETRIC = [[0.5, 0.1], [0.1, 0.3]]
ASYMMETRIC = [[0.5, 0.1], [0.14, 0.3]]
NOT_SYMMETRIC = r"'mass' of geometry \d+ is not symmetric: its entries \(0, 1\) and \(1, 0\) are 0.1 and 0.14"
BOX = 0.1 * np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    np.float64,
)
BOX_TETS = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]], np.int32)


def chain(mass):
    """A fixed box and two boxes without a kinetic term in a row, hinged about z, both hinges driven by one
    articulation with the 2 x 2 ``mass``. Returns (scene, the articulation's slot)."""
    config = Scene.default_config()
    config["contact"]["enable"] = False
    scene = Scene(config)
    slots = []
    for k in range(3):
        mesh = tetmesh(BOX + np.array([0.4 * k, 0.0, 0.0]), BOX_TETS)
        label_surface(mesh)
        label_triangle_orient(mesh)
        AffineBodyConstitution().apply_to(mesh, 1e8, 1e3)
        view(mesh.instances().find(builtin.is_fixed))[0] = int(k == 0)
        view(mesh.instances().find(builtin.external_kinetic))[0] = 1
        if k:
            ref = mesh.instances().create("ref_dof_prev", np.zeros((12, 1)))
            view(ref)[:] = affine_body.transform_to_q(mesh.transforms().view())
        slots.append(scene.objects().create(f"box{k}").geometries().create(mesh)[0])
    first = np.zeros(1, np.int32)
    joints = []
    for k in (1, 2):
        axis = np.array([[0.4 * k - 0.2, 0.0, 0.0], [0.4 * k - 0.2, 0.0, 1.0]])
        joint = linemesh(axis, np.array([[0, 1]], np.int32))
        label_surface(joint)
        AffineBodyRevoluteJoint().apply_to(joint, [slots[k]], first, [slots[k - 1]], first, np.array([100.0]))
        joints.append(scene.objects().create(f"joint{k}").geometries().create(joint)[0])
    articulation = ExternalArticulationConstraint().create_geometry(joints, np.zeros(2, np.int32))
    view(articulation["joint_joint"].find("mass"))[:] = np.asarray(mass, np.float64).reshape(-1)
    view(articulation["joint"].find("delta_theta_tilde"))[:] = [0.01, -0.02]
    slot, _ = scene.objects().create("articulation").geometries().create(articulation)
    return scene, slot


def world_of(workspace, scene):
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    return engine, world


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_symmetric_mass_runs(tmp_path):
    within_rounding = np.array(SYMMETRIC)
    within_rounding[1, 0] += 4e-13 * 0.5  # of the largest entry, 0.5
    for k, mass in enumerate((SYMMETRIC, within_rounding)):
        scene, _ = chain(mass)
        engine, world = world_of(tmp_path / str(k), scene)
        assert world.is_valid()
        for _ in range(3):
            world.advance()
            assert world.is_valid()


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_mass_that_is_not_symmetric_is_refused_at_init(tmp_path):
    scene, _ = chain(ASYMMETRIC)
    with pytest.raises(UIPCException, match=NOT_SYMMETRIC):
        world_of(tmp_path / "asymmetric", scene)
    # an infinite entry passed the symmetry test against an infinite bound
    for name, entries in (("inf", {(0, 1): np.inf}),
                          ("inf-mirror", {(0, 1): np.inf, (1, 0): -np.inf}),
                          ("nan", {(0, 1): np.nan})):
        not_finite = np.array(SYMMETRIC)
        for ij, value in entries.items():
            not_finite[ij] = value
        scene, _ = chain(not_finite)
        with pytest.raises(UIPCException, match="is not a finite number"):
            world_of(tmp_path / name, scene)


WRITTEN_LATER = """
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import test_external_articulation_mass as t
from uipc import view
scene, slot = t.chain(t.SYMMETRIC)
engine, world = t.world_of(sys.argv[2], scene)
world.advance()
print("FIRST FRAME", world.is_valid(), flush=True)
view(slot.geometry()["joint_joint"].find("mass"))[:] = np.reshape(t.ASYMMETRIC, -1)
world.advance()
print("SECOND FRAME", world.is_valid(), flush=True)
"""


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_mass_written_between_frames_is_refused_in_its_frame(tmp_path):
    """The mass is read again before every frame and checked there. An error in a frame ends the
    process (the backend's handling of an exception during ``advance``), so it runs in a child."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = subprocess.run(
        [sys.executable, "-c", WRITTEN_LATER, here, str(tmp_path)], capture_output=True, text=True, timeout=600
    )
    out = p.stdout + p.stderr
    assert "FIRST FRAME True" in out
    assert "SECOND FRAME" not in out
    assert p.returncode != 0
    assert "is not symmetric: its entries (0, 1) and (1, 0) are 0.1 and 0.14" in out
