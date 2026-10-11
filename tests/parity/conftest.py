"""Shared setup for tests/parity, the JAX-against-Torch checks.

The suite runs in its own pytest process (`./run.sh test-parity`, and the
second process of `./run.sh test-all`), because JAX_PLATFORMS must be set
before the first jax import. This file sets it to cpu for a bare pytest run
too. It imports no torch: each test module calls
`pytest.importorskip("torch")` itself, so the suite skips cleanly when the
optional Torch install is missing.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
