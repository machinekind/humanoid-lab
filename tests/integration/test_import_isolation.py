"""Import the shared and core modules in a fresh interpreter.

The shared task modules import only array_api_compat, numpy, the standard
library and each other. The JAX-side core runs without torch. Each check
blocks packages in sys.modules before the first import, so it runs in its
own Python process. tests/unit starts no subprocess, so the checks live
here. tests/unit/test_import_guards.py scans the source for the same rules.
"""

import os
import subprocess
import sys

# Keep in step with SHARED in tests/unit/test_import_guards.py.
SHARED = (
    "humanoid_lab.envs.constants",
    "humanoid_lab.envs.signals",
    "humanoid_lab.envs.progress",
    "humanoid_lab.envs.gait_state",
    "humanoid_lab.envs.push",
    "humanoid_lab.envs.symmetry",
    "humanoid_lab.rewards.terms",
    "humanoid_lab.rewards.compose",
    "humanoid_lab.dr.transform",
)

CORE = (
    "humanoid_lab.envs.joystick",
    "humanoid_lab.envs.wrappers",
    "humanoid_lab.dr.randomize",
    "humanoid_lab.registry",
    "humanoid_lab.train",
    "humanoid_lab.ppo_config",
    "humanoid_lab.policy_io",
    "humanoid_lab.eval.battery",
    "humanoid_lab.eval.video",
    "humanoid_lab.export.policy",
    "humanoid_lab.deploy_contract",
    "humanoid_lab.run_record",
)

# Packages a shared module must never load.
HEAVY = ("jax", "torch", "mujoco", "brax", "mujoco_playground", "ml_collections")


def _python(code):
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    return subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True, env=env)


def _block(names):
    return "import sys\n" + "".join(f"sys.modules[{n!r}] = None\n" for n in names)


def _imports(modules):
    return "".join(f"import {m}\n" for m in modules)


def test_shared_modules_import_with_jax_torch_mujoco_and_brax_blocked():
    proc = _python(_block(HEAVY) + _imports(SHARED))
    assert proc.returncode == 0, (
        "a shared module imports a package it must not (see the module "
        f"docstrings of envs/constants.py and plan section 2):\n{proc.stderr}"
    )


def test_shared_modules_load_no_heavy_package_even_behind_a_try():
    code = (
        "import sys\n"
        + _imports(SHARED)
        + f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        + "assert not bad, bad\n"
    )
    proc = _python(code)
    assert proc.returncode == 0, f"importing the shared modules loaded {proc.stderr}"


def test_core_modules_import_with_torch_blocked():
    proc = _python(_block(("torch",)) + _imports(CORE))
    assert proc.returncode == 0, (
        "a JAX-side core module needs torch. Only humanoid_lab/torch_warp "
        f"and tests/parity may import it:\n{proc.stderr}"
    )
