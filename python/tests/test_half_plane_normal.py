"""Regression: the normal ``N`` of a half plane is normalized when it is read.

The half plane took ``N`` as stored, so a normal of length ``s`` scaled every
distance to the plane by ``s``: a body came to rest closer to the plane
(``s > 1``) or further from it (``s < 1``) than ``d_hat`` and the barrier say.
The sanity check that rejects a vertex inside its thickness at init took the
same scaled distance. A plane is the same plane for every length of its
normal, so the reference is the scene with the unit normal, which ``ground()``
builds.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, view
from uipc.constitution import ElasticModuli, Particle, StableNeoHookean
from uipc.geometry import halfplane, label_surface, label_triangle_orient, pointcloud, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

FRAMES = 40
S3 = np.sqrt(3) / 2
TET = 0.2 * np.array([[0, 1, 0], [0, 0, 1], [-S3, 0, -0.5], [S3, 0, -0.5]], dtype=np.float64)


def lowest_height(workspace, scale, constitution):
    """The lowest y of a soft tetrahedron dropped from 2 cm onto the plane
    y = 0 with the normal ``(0, scale, 0)``, after every frame."""
    config = Scene.default_config()
    config["contact"]["constitution"] = constitution
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)
    mesh = tetmesh(TET + np.array([0.0, 0.02, 0.0]), np.array([[0, 1, 2, 3]], dtype=np.int32))
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.45), mass_density=1e3)
    slot, _ = scene.objects().create("tet").geometries().create(mesh)
    plane = halfplane(np.zeros(3), np.array([0.0, scale, 0.0]))
    scene.objects().create("plane").geometries().create(plane)

    engine = Engine("cuda", str(workspace))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    ys = []
    for _ in range(FRAMES):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        ys.append(np.array(view(slot.geometry().positions())).reshape(-1, 3)[:, 1].min())
    return np.array(ys)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("constitution", ["ipc", "al-ipc"])
def test_normal_length_does_not_change_the_plane(tmp_path, constitution):
    unit = lowest_height(tmp_path / "unit", 1.0, constitution)
    assert unit.min() > 0.0
    for scale in (2.0, 0.5):
        # before the fix: about 4.4 mm above the plane for scale 2
        scaled = lowest_height(tmp_path / str(scale), scale, constitution)
        np.testing.assert_allclose(scaled, unit, rtol=0, atol=1e-9)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("backend", ["none", "cuda"])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_sanity_check_takes_the_distance_to_the_plane(tmp_path, backend, scale):
    """A particle of radius 1 mm is inside its thickness 0.7 mm above the
    plane, and outside it 1.5 mm above: the sanity check (the CPU one with the
    ``none`` backend, the CUDA backend's own with ``cuda``) rejects the first
    scene and accepts the second, for every length of the normal."""
    for height, valid in ((0.0007, False), (0.0015, True)):
        config = Scene.default_config()
        config["gravity"] = [[0.0], [0.0], [0.0]]
        scene = Scene(config)
        scene.contact_tabular().default_model(0.5, 1e9)
        particle = pointcloud(np.array([[0.0, height, 0.0]]))
        label_surface(particle)
        Particle().apply_to(particle, 1e3, 0.001)
        scene.objects().create("particle").geometries().create(particle)
        plane = halfplane(np.zeros(3), np.array([0.0, scale, 0.0]))
        scene.objects().create("plane").geometries().create(plane)
        engine = Engine(backend, str(tmp_path / f"{height}"))
        world = World(engine)
        world.init(scene)
        # before the fix: a distance scaled by the length of the normal
        assert world.is_valid() == valid, (scale, height)
