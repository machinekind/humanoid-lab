"""The origin/main JAX gate.

tests/data/fixtures/ holds what origin/main (042f018) computes on the JAX
path: train-layout and respawn rollouts, DR draws, single-world mechanisms,
the pure command draws, the composed PPO params and a tiny brax training run.
Each test reruns the same procedure on the current code and compares bit for
bit.

A failure means the code change is wrong. It is never a reason to re-record.
Re-recording is allowed only from the base tree, with generate_fixtures.py.
See reference_rollouts.py for the rollouts and generate_fixtures.py for the
procedure.

The fixtures are CPU bit patterns recorded on one platform, so the tests skip
on any other.
"""

from __future__ import annotations

import json
import platform
import sys

import jax
import numpy as np
import pytest

from reference_rollouts import (
    BRAX_TRAIN_PATH,
    CASE_IDS,
    CASES,
    PPO_PARAMS_PATH,
    brax_train,
    case_path,
    ppo_params,
    rollout,
)

# The base tree has no ppo_config, and generate_fixtures' non-destructive
# check runs this file against base code.
try:
    from humanoid_lab.ppo_config import REPO_PPO_KEYS
except ImportError:
    REPO_PPO_KEYS = frozenset()

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "cpu",
    reason=(
        f"fixtures are CPU-recorded bit patterns; this host's default backend is "
        f"{jax.default_backend()!r}. Re-run with JAX_PLATFORMS=cpu."
    ),
)

_RECORD_HINT = (
    "Record it from the origin/main worktree with tests/integration/generate_fixtures.py "
    "(see its docstring). Never record it from this tree."
)


def _skip_if_other_platform(platform_, machine):
    if (str(platform_), str(machine)) != (sys.platform, platform.machine()):
        pytest.skip(
            f"fixture recorded on {platform_} {machine}; this host is "
            f"{sys.platform} {platform.machine()}"
        )


def _load(path):
    if not path.exists():
        pytest.fail(f"missing fixture {path}. {_RECORD_HINT}")
    data = np.load(path)
    _skip_if_other_platform(data["_platform"], data["_machine"])
    return {k: data[k] for k in data.files if not k.startswith("_")}


def _assert_same(fresh, recorded):
    assert sorted(fresh) == sorted(recorded), (
        f"key sets differ: only fresh {sorted(set(fresh) - set(recorded))}, "
        f"only recorded {sorted(set(recorded) - set(fresh))}"
    )
    for key in sorted(recorded):
        # assert_array_equal ignores dtype, and the fixtures pin bit patterns.
        assert np.asarray(fresh[key]).dtype == recorded[key].dtype, (
            key,
            np.asarray(fresh[key]).dtype,
            recorded[key].dtype,
        )
        np.testing.assert_array_equal(np.asarray(fresh[key]), recorded[key], err_msg=key)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_reference_fixture(case):
    recorded = _load(case_path(case.name))
    _assert_same(rollout(case), recorded)


def test_ppo_params_match_origin_main():
    if not PPO_PARAMS_PATH.exists():
        pytest.fail(f"missing fixture {PPO_PARAMS_PATH}. {_RECORD_HINT}")
    recorded = json.loads(PPO_PARAMS_PATH.read_text())
    meta = recorded.pop("_meta")
    _skip_if_other_platform(meta["_platform"], meta["_machine"])
    fresh = ppo_params()
    for name, expected in recorded.items():
        assert name in fresh, f"experiment {name} is gone"
        brax_facing = {k: v for k, v in fresh[name].items() if k not in REPO_PPO_KEYS}
        assert brax_facing == expected, f"ppo params of {name} changed"
        # Every recorded experiment trains with the symmetry loss off.
        assert fresh[name].get("sym_coef", 0.0) == 0.0, name


def test_brax_train_matches_origin_main():
    recorded = _load(BRAX_TRAIN_PATH)
    _assert_same(brax_train(), recorded)
