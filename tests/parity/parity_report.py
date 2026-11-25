"""The worst error of each parity check, collected across a pytest run.

Tests call record() or record_value() for every compared quantity. conftest
prints table() at the end of the run when pytest gets --parity-report. This
module imports only numpy and the standard library.
"""

from __future__ import annotations

import numpy as np

# Relative errors skip elements whose expected value is smaller than this.
_REL_FLOOR = 1e-3

RESULTS: dict = {}


def errors(expected, actual) -> tuple[float, float]:
    """Return the max absolute and the max relative error of actual."""
    e, a = np.broadcast_arrays(
        np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    )
    diff = np.abs(a - e)
    max_abs = float(diff.max()) if diff.size else 0.0
    mask = np.abs(e) >= _REL_FLOOR
    rel = diff[mask] / np.abs(e[mask])
    max_rel = float(rel.max()) if rel.size else 0.0
    return max_abs, max_rel


def _keep(check, metric, max_abs, max_rel, tol):
    old = RESULTS.get((check, metric))
    if old is not None:
        max_abs = max(max_abs, old["max_abs"])
        max_rel = max(max_rel, old["max_rel"])
        tol = tol if tol is not None else old["tol"]
    RESULTS[(check, metric)] = {"max_abs": max_abs, "max_rel": max_rel, "tol": tol}


def tol_text(tol) -> str:
    """The report's tolerance cell, for example 'rtol=1e-05 atol=1e-05'."""
    return " ".join(f"{k}={v:g}" for k, v in tol.items())


def record(check, metric, expected, actual, tol=None):
    """Keep the worst errors of actual against expected for (check, metric)."""
    max_abs, max_rel = errors(expected, actual)
    _keep(check, metric, max_abs, max_rel, tol)


def record_value(check, metric, value, tol=None):
    """Keep a precomputed scalar error, such as a physics median, as max_abs."""
    _keep(check, metric, float(value), 0.0, tol)


def table() -> str:
    """Return one line per (check, metric), sorted."""
    if not RESULTS:
        return "no parity errors recorded"
    rows = [("check", "metric", "max abs", "max rel", "tol")]
    for (check, metric), r in sorted(RESULTS.items()):
        rows.append(
            (
                str(check),
                str(metric),
                f"{r['max_abs']:.3g}",
                f"{r['max_rel']:.3g}",
                r["tol"] if r["tol"] is not None else "-",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in rows
    )


def reset():
    """Clear RESULTS."""
    RESULTS.clear()
