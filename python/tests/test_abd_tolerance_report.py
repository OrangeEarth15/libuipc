"""Regression: the ``ABDToleranceChecker`` line of the Newton tolerance report.

Every Newton iteration logs the verdict of each tolerance checker. The
affine-body checker printed ``Tol: < <tolerance>`` also when its check failed:
the line took its sign from a member the check never wrote, while a local of
the same name took the verdict. The line now prints ``>`` when a transform
rate exceeds the tolerance, as the mark of the same line (``[x]``) says.
"""

import os
import re
import sys
import tempfile

import numpy as np
import pytest

from uipc import Engine, Logger, Scene, World
from uipc.constitution import AffineBodyConstitution, ElasticModuli, StableNeoHookean
from uipc.geometry import ground, label_surface, label_triangle_orient, tetmesh

from conftest import skip_cuda_on_macos, skip_cuda_on_macos_reason

S3 = np.sqrt(3) / 2
LINE = re.compile(r"\[(.)\] <\S*ABDToleranceChecker> Tol: (.) (\S+)")


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


@pytest.mark.cuda
@pytest.mark.skipif(skip_cuda_on_macos, reason=skip_cuda_on_macos_reason)
def test_the_sign_follows_the_verdict(tmp_path):
    """An affine tetrahedron that lands on an edge and tips over: Newton steps
    in which its transform rate fails the tolerance and steps in which it
    passes."""
    scene = Scene(Scene.default_config())
    scene.contact_tabular().default_model(0.5, 1e9)
    v = 0.2 * np.array([[0, 1, 0], [0, 0, 1], [-S3, 0, -0.5], [S3, 0, -0.5]], dtype=np.float64)
    c, s = np.cos(0.6), np.sin(0.6)
    tilted = v @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    tilted += np.array([0.6, 0.012 - tilted[:, 1].min(), 0.0])
    obj = scene.objects().create("tets")
    for x, affine in ((v + np.array([0.0, 0.03, 0.0]), False), (tilted, True)):
        mesh = tetmesh(x, np.array([[0, 1, 2, 3]], dtype=np.int32))
        label_surface(mesh)
        label_triangle_orient(mesh)
        if affine:
            AffineBodyConstitution().apply_to(mesh, 1e8)
        else:
            StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(1e5, 0.45), 1e3)
        obj.geometries().create(mesh)
    scene.objects().create("ground").geometries().create(ground(0.0))

    engine = Engine("cuda", str(tmp_path))

    world = World(engine)
    world.init(scene)
    assert world.is_valid()
    with CapturedStdout() as log:
        Logger.set_level(Logger.Level.Info)
        try:
            for _ in range(6):
                world.advance()
        finally:
            Logger.set_level(Logger.Level.Warn)
    assert world.is_valid()
    lines = LINE.findall(log.text)
    marks = {mark for mark, _, _ in lines}
    assert marks == {"*", "x"}, "the scene has both verdicts"
    for mark, sign, tol in lines:
        assert float(tol) == pytest.approx(0.1 * 0.01, rel=1e-12)
        # before the fix: "[x] ... Tol: < 0.001" for a failed check
        assert sign == ("<" if mark == "*" else ">"), (mark, sign)
