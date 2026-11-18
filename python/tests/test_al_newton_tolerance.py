"""Regression: the Newton tolerance of the AL-IPC pipeline tests the step just
solved.

The AL-IPC frame asked its tolerance checkers right after the solve and
collected the vertex displacements only afterwards, so ``MaxTranslationChecker``
read the displacement of the Newton iteration before (or of the last
collision-free sweep) while ``ABDToleranceChecker`` read this iteration's step.
The IPC frame collects first and tests after; the AL-IPC frame now does too.
The verdict decides whether the line search accepts a trial whose energy rose
and whether a backtracked step updates the multipliers.

The reference is independent of the engine: a soft tetrahedron in free fall
from rest. The energy of a frame is quadratic in a rigid translation, so the
first Newton step of frame ``f`` goes the whole way to the prediction,
``f g dt^2`` under BDF1, and every later step of the frame is round-off. With
``newton/velocity_tol`` 0.05 m/s the tolerance is 5e-4 m: the first step of
every frame fails it, the others pass. The tolerance report of every Newton
iteration, which the engine logs at the info level, must say so.
"""

import os
import re
import sys
import tempfile

import numpy as np
import pytest

from uipc import Engine, Logger, Scene, World
from uipc.constitution import ElasticModuli, StableNeoHookean
from uipc.geometry import ground, label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

G, DT, VELOCITY_TOL = 9.8, 0.01, 0.05
FRAMES = 4
REPORT = re.compile(
    r"Frame (\d+) Newton Iteration (\d+) =+\n"
    r"    \[(.)\] <\S*MaxTranslationChecker> Residual/AbsTol: (\S+)/(\S+)")


class CapturedStdout:
    """The engine's log (spdlog writes to the process's stdout), captured at
    the file-descriptor level."""

    def __enter__(self):
        sys.stdout.flush()
        self._file = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(1)
        os.dup2(self._file.fileno(), 1)
        return self

    def __exit__(self, *exc):
        sys.stdout.flush()
        os.dup2(self._saved, 1)
        os.close(self._saved)
        self._file.seek(0)
        self.text = re.sub(r"\x1b\[[0-9;]*m", "", self._file.read().decode(errors="replace"))
        self._file.close()
        return False


def reports(workspace, constitution):
    """``{frame: [(passed, residual) of every Newton iteration]}`` from the
    tolerance reports of ``FRAMES`` frames of a tetrahedron falling from
    0.5 m, far from the ground."""
    config = Scene.default_config()
    config["contact"]["constitution"] = constitution
    config["linear_system"]["tol_rate"] = 1e-12
    config["linear_system"]["check_interval"] = 1
    config["newton"]["velocity_tol"] = VELOCITY_TOL
    scene = Scene(config)
    scene.contact_tabular().default_model(0.5, 1e9)
    tet = 0.2 * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    mesh = tetmesh(tet + np.array([0.0, 0.5, 0.0]), np.array([[0, 1, 2, 3]], dtype=np.int32))
    label_surface(mesh)
    label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.4), 1e3)
    scene.objects().create("tet").geometries().create(mesh)
    scene.objects().create("ground").geometries().create(ground(0.0))

    engine = Engine("cuda", str(workspace))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    with CapturedStdout() as log:
        Logger.set_level(Logger.Level.Info)
        try:
            for _ in range(FRAMES):
                world.advance()
        finally:
            Logger.set_level(Logger.Level.Warn)
    assert world.is_valid()
    out = {}
    for frame, newton, mark, residual, tol in REPORT.findall(log.text):
        assert float(tol) == pytest.approx(VELOCITY_TOL * DT, rel=1e-12)
        rows = out.setdefault(int(frame), [])
        assert int(newton) == len(rows), "one report per Newton iteration, in order"
        rows.append((mark == "*", float(residual)))
    return out


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
@pytest.mark.parametrize("constitution", ["al-ipc", "ipc"])
def test_the_report_is_of_the_step_just_solved(tmp_path, constitution):
    r = reports(tmp_path, constitution)
    assert sorted(r) == list(range(1, FRAMES + 1))
    for frame, rows in r.items():
        # before the fix, under AL-IPC: iteration 0 passed with the step of the
        # frame before (0 in frame 1), and iteration 1 failed with f g dt^2
        passed, residual = rows[0]
        assert not passed, (frame, rows)
        np.testing.assert_allclose(residual, frame * G * DT * DT, rtol=1e-6)
        assert len(rows) >= 2, (frame, rows)
        for passed, residual in rows[1:]:
            assert passed, (frame, rows)
            assert residual < 1e-9, (frame, rows)
