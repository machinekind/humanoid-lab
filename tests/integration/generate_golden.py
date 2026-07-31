"""Record tests/data/golden/*.npz.

    .venv/bin/python tests/integration/generate_golden.py

Overwrites every golden. Run it only when a change to stock env behavior has
been agreed and reviewed -- test_golden_baseline.py failing is not by itself
a reason to run this. See that module's docstring.

Run on CPU. The goldens are compared bit for bit and XLA's GPU kernels do not
reproduce the CPU backend's reduction orders.
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from golden_rollout import CASES, GOLDEN_DIR, rollout


def main() -> int:
    if jax.default_backend() != "cpu":
        print(
            f"refusing to record on backend {jax.default_backend()!r}: the goldens are "
            "CPU bit patterns. Re-run with JAX_PLATFORMS=cpu.",
            file=sys.stderr,
        )
        return 1

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        arrays = rollout(case)
        np.savez(case.path, **arrays)
        print(f"wrote {case.path.relative_to(GOLDEN_DIR.parents[2])} ({case.path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
