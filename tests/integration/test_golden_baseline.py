"""The stock-rollout bit-exactness gate.

With every optional mechanism off, an env rollout has to stay bit-exact
against the recorded baseline. Any commit that shifts a single float in a
stock rollout fails here.

A failure is not a "regenerate the goldens" prompt. It means a
default-off mechanism changed behavior while off -- usually by consuming an
RNG key outside its feature flag. Regenerating is correct only when a
deliberate, documented change to stock behavior has been agreed.

The goldens were recorded on a CPU backend, so the test skips elsewhere:
XLA's GPU kernels use different reduction orders and would not match bit for
bit. See golden_rollout.py for the rollout itself.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from golden_rollout import CASE_IDS, CASES, rollout

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason=(
        f"goldens are CPU-recorded bit patterns; this host's default backend is "
        f"{jax.default_backend()!r}. Re-run with JAX_PLATFORMS=cpu."
    ),
)

# Arrays compared whole, by npz key.
_EXACT_KEYS = (
    "reward",
    "done",
    "obs_state",
    "obs_privileged_state",
    "final_rng",
    "final_phase",
    "final_command",
)


@pytest.fixture(scope="module")
def _recorded():
    """Cache each case's rollout so the four envs are built at most once."""
    return {}


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_rollout_is_bit_exact_against_the_recorded_golden(case, _recorded):
    if not case.path.exists():
        pytest.fail(
            f"missing golden {case.path}. Record it with "
            "`.venv/bin/python tests/integration/generate_golden.py`."
        )
    golden = np.load(case.path)

    if case.name not in _recorded:
        _recorded[case.name] = rollout(case)
    fresh = _recorded[case.name]

    for key in _EXACT_KEYS:
        np.testing.assert_array_equal(fresh[key], golden[key], err_msg=key)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_reward_metrics_are_bit_exact_against_the_pre_port_golden(case, _recorded):
    """Every reward term the golden recorded still produces the same numbers.

    Only the golden's own keys are checked. A later mechanism that adds a
    reward term adds a key here, and that key arrives with its own tests; it
    is not this gate's business. A key that *disappears* is a failure, because
    something the pre-port env computed no longer exists.
    """
    if not case.path.exists():
        pytest.fail(
            f"missing golden {case.path}. Record it with "
            "`.venv/bin/python tests/integration/generate_golden.py`."
        )
    golden = np.load(case.path)

    if case.name not in _recorded:
        _recorded[case.name] = rollout(case)
    fresh = _recorded[case.name]

    fresh_by_name = dict(zip(fresh["metric_names"].tolist(), fresh["metric_values"].T))
    golden_names = golden["metric_names"].tolist()

    missing = [n for n in golden_names if n not in fresh_by_name]
    assert not missing, f"reward terms the golden recorded are gone: {missing}"

    for i, name in enumerate(golden_names):
        np.testing.assert_array_equal(
            fresh_by_name[name], golden["metric_values"][:, i], err_msg=name
        )
