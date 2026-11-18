"""Regression: the pair of a codim vertex and a shell vertex.

The point / codim-point candidates pair every surface vertex ``V`` with every
codim vertex ``W`` (a vertex of dimension <= 1: a particle or a rod vertex).
A pair of two codim vertices is found twice, as ``(V, W)`` and ``(W, V)``, so
the candidate predicate drops the copy with ``V >= W``. The predicate tested
``dimension(V) <= 2`` instead of ``<= 1``, so it also dropped the pair of a
shell vertex and a codim vertex, which is found once, whenever the shell
vertex came later in the vertex order (always, in a scene of finite elements).

The reference is the same contact with a vertex of a volume mesh: a particle
above the apex of a fixed tetrahedron and above the shared vertex of a fixed
shell fan with the same three triangles meets the same pairs, so the two
scenes have the same number of point-point contact pairs.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc.constitution import (
    ElasticModuli,
    ElasticModuli2D,
    NeoHookeanShell,
    Particle,
    StableNeoHookean,
)
from uipc.core import ContactSystemFeature
from uipc.geometry import (
    Geometry,
    label_surface,
    label_triangle_orient,
    pointcloud,
    tetmesh,
    trimesh,
)

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

# the three ring vertices around the shared vertex at the origin, 4 cm below it
RING = np.array([[0.2 * np.cos(k * 2 * np.pi / 3), -0.04, 0.2 * np.sin(k * 2 * np.pi / 3)]
                 for k in range(3)])
FAN = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3]], dtype=np.int32)


def point_point_pairs(workspace, apex):
    """The point-point contact pairs after one frame of a particle 6 mm above
    the shared vertex of three fixed triangles: those of a shell fan, or with
    ``apex``, those of a tetrahedron."""
    config = Scene.default_config()
    config["gravity"] = [[0.0], [0.0], [0.0]]
    config["contact"]["friction"]["enable"] = False
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)
    x = np.vstack([np.zeros(3), RING])
    if apex:
        mesh = tetmesh(x, np.array([[0, 1, 2, 3]], dtype=np.int32))
        label_surface(mesh)
        label_triangle_orient(mesh)
        StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e6, 0.3), 200.0)
    else:
        mesh = trimesh(x, FAN)
        label_surface(mesh)
        NeoHookeanShell().apply_to(mesh, ElasticModuli2D.youngs_poisson(1e6, 0.3), 200.0, 0.001)
    view(mesh.vertices().find(builtin.is_fixed))[:] = 1
    scene.objects().create("fixed").geometries().create(mesh)
    particle = pointcloud(np.array([[0.0, 0.006, 0.0]]))
    label_surface(particle)
    Particle().apply_to(particle, 1e3, 0.001)
    scene.objects().create("particle").geometries().create(particle)

    engine = Engine("cuda", str(workspace))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    world.advance()
    assert world.is_valid()
    pairs = Geometry()
    world.features().find(ContactSystemFeature).contact_energy("PP+N", pairs)
    return pairs.instances().size()


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_shell_vertex_meets_a_particle_as_a_volume_vertex_does(tmp_path):
    volume = point_point_pairs(tmp_path / "apex", apex=True)
    shell = point_point_pairs(tmp_path / "fan", apex=False)
    assert volume > 0
    # before the fix the shell fan had one pair fewer: the point / codim-point
    # pair of the shared vertex and the particle was dropped
    assert shell == volume
