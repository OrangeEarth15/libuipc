"""Regression: an adaptive default contact model with more than one contact
element.

``default_model(mu, adaptive)`` makes the default model adaptive: its
resistance -1 marks it, and every frame the adaptive strategy computes the
stiffness it takes. The pairs of contact elements without a model of their
own take the default model, but the strategy adapted only the entry of the
default element with itself: every other pair kept the marker -1 as its
stiffness, a barrier of negative stiffness that attracts. Now every pair that
takes the adaptive default model is adaptive.

The reference is the scene with one contact element: a tetrahedron on a
contact element of its own, whose pair with the ground's element has no
model, rests where it rests when both are on the default element.
"""

import numpy as np
import pytest

from uipc import Engine, Scene, World, builtin, view
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.geometry import ground, label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

FRAMES = 10
TET = 0.2 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)


def run(workspace, written, second_element):
    """A soft tetrahedron resting 5 mm above the ground, its contact adaptive.
    ``written``: the adaptive model is the default model (``default_model``)
    or a model of the default element with itself (``insert``).
    ``second_element``: the tetrahedron is on a contact element of its own.
    Returns the positions, the resistance of model 0 after every frame and
    whether every frame converged."""
    config = Scene.default_config()
    config["contact"]["friction"]["enable"] = False
    # the configured bounds rule, wide enough that the stiffness is the one
    # the strategy projects
    config["contact"]["adaptive"]["kappa_eval_scale"] = 1.0
    config["contact"]["adaptive"]["min_kappa"] = 1.0
    config["contact"]["adaptive"]["max_kappa"] = 1e20
    config["newton"]["velocity_tol"] = 1e-6
    config["newton"]["max_iter"] = 64
    config["newton"]["semi_implicit"]["enable"] = False
    config["linear_system"]["tol_rate"] = 1e-8
    scene = Scene(config)
    tabular = scene.contact_tabular()
    if written == "insert":
        tabular.insert(tabular.default_element(), tabular.default_element(), 0.0, builtin.adaptive)
    else:
        tabular.default_model(0.0, builtin.adaptive)
    mesh = tetmesh(TET + np.array([0.0, 0.005, 0.0]), np.array([[0, 1, 2, 3]], dtype=np.int32))
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.4), 1e3)
    if second_element:
        tabular.create("second").apply_to(mesh)
    slot, _ = scene.objects().create("tet").geometries().create(mesh)
    scene.objects().create("ground").geometries().create(ground(0.0))

    engine = Engine("cuda", str(workspace))
    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    resistance, converged = [], True
    for _ in range(FRAMES):
        world.advance()
        assert world.is_valid()
        world.retrieve()
        models = scene.contact_tabular().contact_models()
        resistance.append(float(np.asarray(view(models.find("resistance"))).reshape(-1)[0]))
        converged = converged and bool(engine.frame_stats()["converged"])
    x = np.array(view(slot.geometry().positions())).reshape(-1, 3)
    return x, np.array(resistance), converged


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_pairs_without_a_model_take_the_adaptive_default(tmp_path):
    x1, kappa1, converged1 = run(tmp_path / "one", "default_model", False)
    assert converged1
    assert np.all(kappa1 > 0.0)
    assert x1[:, 1].min() > 3e-3
    # before the fix: the pair kept the stiffness -1, every frame ran to
    # newton/max_iter and the tetrahedron sank to micrometres above the ground
    x2, kappa2, converged2 = run(tmp_path / "two", "default_model", True)
    assert converged2
    np.testing.assert_allclose(kappa2, kappa1, rtol=1e-9)
    np.testing.assert_allclose(x2, x1, rtol=0, atol=1e-9)


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_an_inserted_adaptive_model_stays_its_own_pair(tmp_path):
    """``insert`` makes the pair of the default element with itself adaptive
    and leaves the default model as it was: a pair without a model then sits
    at the lower bound, and the adaptive entry, never in contact, keeps
    ``contact/adaptive/init_kappa``."""
    x, kappa, converged = run(tmp_path, "insert", True)
    assert converged
    np.testing.assert_array_equal(kappa, 1e9)
    assert x[:, 1].min() > 0.0
