"""The parity report collector and the per-device physics tolerances.

Both modules are torch-free, so this file runs without the Torch install.
"""

from __future__ import annotations

import copy

import parity_report
import pytest
import tolerances


@pytest.fixture(autouse=True)
def _isolated_results():
    saved = copy.deepcopy(parity_report.RESULTS)
    parity_report.reset()
    yield
    parity_report.RESULTS.clear()
    parity_report.RESULTS.update(saved)


def test_record_keeps_the_worst_error():
    parity_report.record("c", "m", [1.0, 2.0], [1.1, 2.0], tol="atol=1")
    parity_report.record("c", "m", [1.0, 2.0], [1.0, 2.02])
    r = parity_report.RESULTS[("c", "m")]
    assert r["max_abs"] == pytest.approx(0.1)
    assert r["max_rel"] == pytest.approx(0.1)
    assert r["tol"] == "atol=1"
    parity_report.record("c", "m", [1.0], [1.5])
    assert parity_report.RESULTS[("c", "m")]["max_abs"] == pytest.approx(0.5)


def test_relative_error_ignores_values_near_zero():
    max_abs, max_rel = parity_report.errors([1e-6, 2.0], [1e-4, 2.002])
    assert max_abs == pytest.approx(0.002)
    assert max_rel == pytest.approx(1e-3)
    assert parity_report.errors([0.0], [1e-3]) == (pytest.approx(1e-3), 0.0)
    assert parity_report.errors([], []) == (0.0, 0.0)


def test_table_lists_every_check():
    assert parity_report.table() == "no parity errors recorded"
    parity_report.record("task_math", "signals.foo", [1.0], [1.0], tol="rtol=1e-05 atol=1e-05")
    parity_report.record_value("physics", "qvel median", 2.5e-3, tol="atol=0.01")
    lines = parity_report.table().splitlines()
    assert len(lines) == 3
    assert lines[0].split()[:2] == ["check", "metric"]
    assert lines[1].startswith("physics")
    assert "0.0025" in lines[1] and "atol=0.01" in lines[1]
    assert lines[2].startswith("task_math")
    assert "signals.foo" in lines[2]


def test_physics_tolerance_follows_the_device():
    assert tolerances.physics("cpu") is tolerances.PHYSICS_CPU
    assert tolerances.physics("cuda:0") is tolerances.PHYSICS_CUDA
