"""Regression: a recover under AL-IPC continues the dumped run.

An AL-IPC frame starts from its active set: the constraint pairs, their
multipliers and their inactive counts (it takes its friction pairs from them,
linearizes them at the collision-free positions and merges the new candidates
into them). The dump held none of it, so a recover kept the active set of the
frame the world had reached, and a new world started from an empty one: the
recovered run was another run. The dump now holds the active set and the
collision-free positions, as the other systems dump what their next frame
starts from, and a recover loads them.

The reference is the straight run, which the recovered run must follow from
the frame after the dump, in the world that dumped and in a new world.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, Transform, Vector3, World, builtin, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import ground, label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

FRAMES, AT = 30, 10
S3 = np.sqrt(3) / 2
TET = 0.2 * np.array([[0, 1, 0], [0, 0, 1], [-S3, 0, -0.5], [S3, 0, -0.5]], dtype=np.float64)


def new_scene():
    config = Scene.default_config()
    config["contact"]["constitution"] = "al-ipc"
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)
    return scene


def thrown_tet():
    """A soft tetrahedron thrown at the ground: it lands in the first frames,
    then slides with friction (vertex / half-plane constraints)."""
    scene = new_scene()
    mesh = tetmesh(TET + np.array([0.0, 0.03, 0.0]), np.array([[0, 1, 2, 3]], dtype=np.int32))
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.45), mass_density=1e3)
    velocity = mesh.vertices().find(builtin.velocity)
    if velocity is None:
        velocity = mesh.vertices().create(builtin.velocity, Vector3.Zero())
    view(velocity)[:] = np.array([1.0, -2.0, 0.0]).reshape(3, 1)
    slot, _ = scene.objects().create("tet").geometries().create(mesh)
    scene.objects().create("ground").geometries().create(ground(0.0))
    return scene, slot


def stacked_tets():
    """An affine tetrahedron dropped tip-first onto a fixed one
    (point-triangle constraints)."""
    scene = new_scene()
    obj = scene.objects().create("tets")
    slots = []
    for y, fixed in ((0.35, 0), (0.0, 1)):
        mesh = tetmesh(1.5 * TET, np.array([[0, 1, 2, 3]], dtype=np.int32))
        label_surface(mesh)
        label_triangle_orient(mesh)
        AffineBodyConstitution().apply_to(mesh, 100e6)
        t = Transform.Identity()
        t.translate(Vector3.UnitY() * y)
        view(mesh.transforms())[0] = t.matrix()
        view(mesh.instances().find(builtin.is_fixed))[0] = fixed
        slots.append(obj.geometries().create(mesh)[0])
    return scene, slots[0]


SCENES = {"thrown_tet": thrown_tet, "stacked_tets": stacked_tets}


def world_positions(slot):
    geo = slot.geometry()
    T = np.asarray(view(geo.transforms())).reshape(4, 4)
    return np.asarray(view(geo.positions())).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]


def frames_after_the_dump(workspace, build, mode):
    """World positions after the frames ``AT + 1 .. FRAMES``. ``mode``:
    ``straight``; ``same world``, which dumps after frame ``AT``, runs five
    more frames and recovers the dump; ``new world``, which recovers the dump
    in a new world on the same workspace."""
    scene, slot = build()
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    for _ in range(AT):
        world.advance()
        assert world.is_valid()
        world.retrieve()
    if mode != "straight":
        assert world.dump()
        if mode == "same world":
            for _ in range(5):
                world.advance()
        else:
            scene, slot = build()
            engine = Engine("cuda", str(workspace))
            world = World(engine)
            world.init(scene)
            assert world.is_valid()
        assert world.recover(AT)
        assert world.frame() == AT
    out = []
    for _ in range(AT, FRAMES):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        out.append(world_positions(slot))
    return np.array(out)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("mode", ["same world", "new world"])
def test_recovered_run_is_the_straight_run(tmp_path, scene, mode):
    build = SCENES[scene]
    straight = frames_after_the_dump(tmp_path / "straight", build, "straight")
    # before the fix: the thrown tetrahedron 1.1e-4 m (same world) and
    # 1.9e-2 m (new world) from the straight run
    recovered = frames_after_the_dump(tmp_path / "recovered", build, mode)
    np.testing.assert_allclose(recovered, straight, rtol=0, atol=1e-9)
