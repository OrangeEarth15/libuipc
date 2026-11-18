"""Regression: a ``dt`` written after ``world.init``.

The frames read ``dt`` from the scene config every time, so a ``dt`` written
between frames is the step of the next frame. What depends on it must follow:
a scene whose ``dt`` is halved right after init must run as the scene that was
initialised with the halved ``dt``.

The contact stiffness corridor ``[base / dt^2, 100 base / dt^2]`` bounds the
product ``kappa dt^2`` that the barrier takes. It was computed once, from the
``dt`` of init, and the contact tabular was resolved against it once (a default
model the user did not set takes the lower bound, other resistances are
clamped into it, the adaptive stiffness of every frame is clamped to it).
Now the tabular is resolved again when ``dt`` changes.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, Transform, Vector3, World, builtin, view
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import ground, label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

S3 = np.sqrt(3) / 2
TET = 0.2 * np.array([[0, 1, 0], [0, 0, 1], [-S3, 0, -0.5], [S3, 0, -0.5]], dtype=np.float64)

# how the stiffness of the default model depends on the corridor: left as the
# scene creates it (the lower bound), set by the user (clamped into the
# corridor), adaptive (clamped in every frame)
STIFFNESS = {"lower bound": None, "clamped": 1e9, "adaptive": builtin.adaptive}


def new_scene(dt, resistance, constitution):
    config = Scene.default_config()
    config["dt"] = dt
    config["contact"]["constitution"] = constitution
    if constitution == "al-ipc":
        config["contact"]["friction"]["enable"] = False
    scene = Scene(config)
    if resistance is not None:
        scene.contact_tabular().default_model(0.5, resistance)
    return scene


def thrown_tet(dt, resistance, constitution):
    """A soft tetrahedron thrown at the ground: it lands in the first frames,
    then slides."""
    scene = new_scene(dt, resistance, constitution)
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


def stacked_tets(dt, resistance, constitution):
    """An affine tetrahedron dropped tip-first onto a fixed one."""
    scene = new_scene(dt, resistance, constitution)
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


def run(workspace, build, dt_init, dt, resistance, constitution="ipc", frames=30, recover_at=0):
    """World positions of the moving body after ``frames`` frames at ``dt``,
    written right after init when it differs from ``dt_init``.
    ``recover_at``: the run dumps after that frame, goes on for five frames and
    recovers the dump before it continues."""
    scene, slot = build(dt_init, resistance, constitution)
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    if dt != dt_init:
        view(scene.config().find("dt"))[0] = dt
    for frame in range(1, frames + 1):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        if frame == recover_at:
            assert world.dump()
            for _ in range(5):
                world.advance()
            assert world.recover(recover_at)
    geo = slot.geometry()
    T = np.asarray(view(geo.transforms())).reshape(4, 4)
    return np.asarray(view(geo.positions())).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("stiffness", STIFFNESS)
@pytest.mark.parametrize("recover_at", [0, 10], ids=["straight", "recovered"])
def test_contact_stiffness_follows_dt(tmp_path, scene, stiffness, recover_at):
    build, resistance = SCENES[scene], STIFFNESS[stiffness]
    halved_at_init = run(tmp_path / "init", build, 0.005, 0.005, resistance, recover_at=recover_at)
    # before the fix: 2.8e-4 to 8.6e-3 m apart after 30 frames
    halved_after_init = run(tmp_path / "after", build, 0.01, 0.005, resistance, recover_at=recover_at)
    np.testing.assert_allclose(halved_after_init, halved_at_init, rtol=0, atol=1e-9)


def halved_after_15_frames(workspace, build, resistance, by_recover):
    """World positions of the moving body after 15 frames at dt = 0.01 and 15
    at 0.005. ``by_recover``: the second half is run by a new world,
    initialised with dt = 0.005, that recovers the dump of frame 15; else the
    world that ran the first half takes the halved ``dt``."""
    scene, slot = build(0.01, resistance, "ipc")
    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    for _ in range(15):
        world.advance()
        assert world.is_valid()
        world.retrieve()
    if by_recover:
        assert world.dump()
        scene, slot = build(0.005, resistance, "ipc")
        engine = Engine("cuda", str(workspace))
        world = World(engine)
        world.init(scene)
        assert world.is_valid()
        assert world.recover(15)
    else:
        view(scene.config().find("dt"))[0] = 0.005
    for _ in range(15):
        world.advance()
        assert world.is_valid()
        world.retrieve()
    geo = slot.geometry()
    T = np.asarray(view(geo.transforms())).reshape(4, 4)
    return np.asarray(view(geo.positions())).reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("scene", SCENES)
@pytest.mark.parametrize("stiffness", STIFFNESS)
def test_contact_stiffness_follows_a_dt_written_between_frames(tmp_path, scene, stiffness):
    """``dt`` halved after frames in contact: the run continues as a world
    initialised with the halved ``dt`` continues it from a dump."""
    build, resistance = SCENES[scene], STIFFNESS[stiffness]
    recovered = halved_after_15_frames(tmp_path / "recovered", build, resistance, True)
    changed = halved_after_15_frames(tmp_path / "changed", build, resistance, False)
    np.testing.assert_allclose(changed, recovered, rtol=0, atol=1e-9)
