"""Regression: affine-body vertices carry their ``thickness`` into contact.

``AffineBodyShell`` and ``AffineBodyRod`` write the radius ``r`` of the solid
as the ``thickness`` of the vertices (a slab of thickness ``2 r``, a cylinder
of radius ``r``), and the body's mass is that solid's. The finite-element
vertex reporter passes the attribute on to contact; the affine-body vertex
reporter did not, so an affine-body shell or rod touched with its mid-surface
or centre line and two such solids came to rest inside each other.

The reference is independent of the engine: at rest, the distance ``d``
between the rod's centre line and the shell's mid-surface satisfies
``r_rod + r_shell <= d < r_rod + r_shell + d_hat`` (the solids do not overlap,
and the barrier that carries the rod acts only within ``d_hat`` of their
surfaces). Under AL-IPC the dropped rod keeps rocking, so there the invariant
of the collision-free state every frame ends in is checked: the solids never
overlap, and the rod does reach the shell.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, Transform, World, builtin, view
from uipc.constitution import AffineBodyConstitution, AffineBodyRod, AffineBodyShell
from uipc.geometry import (
    ground,
    label_surface,
    label_triangle_orient,
    linemesh,
    tetmesh,
    trimesh,
)

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

R_SHELL, R_ROD, D_HAT = 0.02, 0.03, 0.01
ROD = np.array([[-0.4, 0.0, 0.05], [-0.1, 0.0, 0.05], [0.2, 0.0, 0.05], [0.2, 0.0, -0.25]])


def translation(y):
    t = Transform.Identity()
    t.translate(np.array([0.0, y, 0.0]))
    return t.matrix()


def rod_heights(workspace, frames, constitution):
    """(frames, 4): the heights of the rod's vertices above the shell's
    mid-surface after every frame. The rod (radius 0.03 m) is dropped from
    0.1 m onto a fixed shell (radius 0.02 m), as in sim case 65."""
    config = Scene.default_config()
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = D_HAT
    config["contact"]["constitution"] = constitution
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)

    quad = np.array([[-0.6, 0.0, -0.6], [0.6, 0.0, -0.6], [0.6, 0.0, 0.6], [-0.6, 0.0, 0.6]])
    shell = trimesh(quad, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32))
    label_surface(shell)
    AffineBodyShell().apply_to(shell, 1e8, 1e3, R_SHELL)
    view(shell.transforms())[0] = translation(0.5)
    view(shell.instances().find(builtin.is_fixed))[0] = 1
    scene.objects().create("shell").geometries().create(shell)

    rod = linemesh(ROD, np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32))
    label_surface(rod)
    AffineBodyRod().apply_to(rod, 1e8, 1e3, R_ROD)
    view(rod.transforms())[0] = translation(0.6)
    slot, _ = scene.objects().create("rod").geometries().create(rod)

    engine = Engine("cuda", str(workspace))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    heights = []
    for _ in range(frames):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        T = np.asarray(view(slot.geometry().transforms())).reshape(4, 4)
        heights.append((ROD @ T[:3, :3].T + T[:3, 3])[:, 1] - 0.5)
    return np.array(heights)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_rod_rests_on_the_surface_of_the_shell(tmp_path):
    heights = rod_heights(tmp_path, 150, "ipc")[-1]
    # before the fix: 9.6 mm, the two solids 40 mm inside each other
    assert heights.min() >= R_ROD + R_SHELL
    assert heights.max() < R_ROD + R_SHELL + D_HAT


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_solids_never_overlap_under_al_ipc(tmp_path):
    lowest = rod_heights(tmp_path, 300, "al-ipc").min(axis=1)
    # before the fix: down to d_hat above the mid-surface
    assert lowest.min() >= R_ROD + R_SHELL
    assert lowest.min() < R_ROD + R_SHELL + D_HAT, "the rod reaches the shell"


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("constitution", ["ipc", "al-ipc"])
def test_affine_box_rests_its_thickness_above_the_ground(tmp_path, constitution):
    """An affine cube whose vertices carry a thickness of 0.03 m, dropped
    onto a half plane: its lowest vertex comes to rest at least the thickness
    above the plane and at most the thickness plus ``d_hat``."""
    thickness = 0.03
    config = Scene.default_config()
    config["contact"]["friction"]["enable"] = False
    config["contact"]["d_hat"] = D_HAT
    config["contact"]["constitution"] = constitution
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)

    corners = 0.1 * np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=np.float64)
    tets = np.array([[0, 1, 3, 4], [1, 2, 3, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 3, 4, 6]],
                    dtype=np.int32)
    cube = tetmesh(corners, tets)
    label_surface(cube)
    label_triangle_orient(cube)
    AffineBodyConstitution().apply_to(cube, 1e8)
    cube.vertices().create(builtin.thickness, thickness)
    view(cube.transforms())[0] = translation(0.1 + thickness + 2 * D_HAT)
    slot, _ = scene.objects().create("cube").geometries().create(cube)
    scene.objects().create("ground").geometries().create(ground(0.0))

    engine = Engine("cuda", str(tmp_path))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    lowest = []
    for _ in range(100):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        T = np.asarray(view(slot.geometry().transforms())).reshape(4, 4)
        lowest.append((corners @ T[:3, :3].T + T[:3, 3])[:, 1].min())
    # before the fix: within d_hat of the plane, inside the thickness
    assert min(lowest) >= thickness
    assert min(lowest) < thickness + D_HAT, "the cube reaches the plane"
