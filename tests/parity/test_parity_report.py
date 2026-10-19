"""The parity report collector.

parity_report is torch-free, so this file runs without the Torch install.
"""

from __future__ import annotations

import copy

import parity_report
import pytest


@pytest.fixture(autouse=True)
def _isolated_results():
    saved = copy.deepcopy(parity_report.RESULTS)
    parity_report.reset()
    yield
    parity_report.RESULTS.clear()
    parity_report.RESULTS.update(saved)


def test_relative_error_ignores_values_near_zero():
    max_abs, max_rel = parity_report.errors([1e-6, 2.0], [1e-4, 2.002])
    assert max_abs == pytest.approx(0.002)
    assert max_rel == pytest.approx(1e-3)
    assert parity_report.errors([0.0], [1e-3]) == (pytest.approx(1e-3), 0.0)
    assert parity_report.errors([], []) == (0.0, 0.0)


def _rows():
    """Return the table's data rows as {(check, metric): (max abs, max rel, tol)}."""
    rows = {}
    for line in parity_report.table().splitlines()[1:]:
        check, metric, max_abs, max_rel, tol = line.split(maxsplit=4)
        rows[(check, metric)] = (float(max_abs), float(max_rel), tol)
    return rows


def test_the_table_shows_the_worst_error_of_each_check():
    assert parity_report.table() == "no parity errors recorded"
    parity_report.record("c", "m", [1.0, 2.0], [1.1, 2.0], tol="atol=1")
    parity_report.record("c", "m", [1.0, 2.0], [1.0, 2.02])
    parity_report.record_value("physics", "qvel", 2.5e-3, tol="atol=0.01")
    rows = _rows()
    assert rows.keys() == {("c", "m"), ("physics", "qvel")}
    max_abs, max_rel, tol = rows[("c", "m")]
    assert max_abs == pytest.approx(0.1)
    assert max_rel == pytest.approx(0.1)
    assert tol == "atol=1"
    assert rows[("physics", "qvel")] == (pytest.approx(2.5e-3), 0.0, "atol=0.01")
    parity_report.record("c", "m", [1.0], [1.5])
    assert _rows()[("c", "m")][0] == pytest.approx(0.5)
