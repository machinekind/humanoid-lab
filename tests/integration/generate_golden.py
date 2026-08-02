"""Record tests/data/golden/*.npz.

    .venv/bin/python tests/integration/generate_golden.py [CASE ...]

With no arguments, overwrites every golden. With one or more case names
(golden_rollout.py's CASE_IDS, e.g. `toy_robot__pd_test`), overwrites only
those and leaves the rest byte-untouched.

Run it only when a change to stock env behavior has been agreed and
reviewed -- test_golden_baseline.py failing is not by itself a reason to run
this. See that module's docstring.

Naming cases is the safer default when the reason to re-record is a change
to ONE robot: re-recording only what changed means the others cannot be
overwritten by accident and then "verified" against themselves.

Run on CPU. The goldens are compared bit for bit and XLA's GPU kernels do not
reproduce the CPU backend's reduction orders.
"""

from __future__ import annotations

import sys

import jax
import numpy as np

from golden_rollout import CASES, GOLDEN_DIR, rollout


def main(argv: list[str]) -> int:
    if jax.default_backend() != "cpu":
        print(
            f"refusing to record on backend {jax.default_backend()!r}: the goldens are "
            "CPU bit patterns. Re-run with JAX_PLATFORMS=cpu.",
            file=sys.stderr,
        )
        return 1

    by_name = {case.name: case for case in CASES}
    selected = list(CASES)
    if argv:
        unknown = [name for name in argv if name not in by_name]
        if unknown:
            print(
                f"unknown case(s) {unknown}; known cases: {sorted(by_name)}",
                file=sys.stderr,
            )
            return 1
        selected = [by_name[name] for name in argv]

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for case in selected:
        arrays = rollout(case)
        np.savez(case.path, **arrays)
        print(
            f"wrote {case.path.relative_to(GOLDEN_DIR.parents[2])} "
            f"({case.path.stat().st_size} bytes)"
        )
    skipped = [c.name for c in CASES if c not in selected]
    if skipped:
        print(f"left untouched: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
