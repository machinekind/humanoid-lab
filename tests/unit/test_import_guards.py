"""Guard the import direction between the shared modules, the core and Torch.

The shared task modules import only array_api_compat, numpy, the standard
library and each other. The JAX-side core runs without torch. Only
src/humanoid_lab/torch_warp/ and tests/parity import torch or torch_warp, so
`./run.sh test`, `./run.sh test-slow` and every JAX tool keep working on an
install without torch.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

UNIT_DIR = Path(__file__).resolve().parent
SRC_DIR = UNIT_DIR.parent.parent / "src" / "humanoid_lab"

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
    "humanoid_lab.policy_io",
    "humanoid_lab.eval.battery",
    "humanoid_lab.eval.video",
    "humanoid_lab.export.policy",
    "humanoid_lab.deploy_contract",
    "humanoid_lab.run_record",
)

# Written in pieces so the static scan below does not match this file.
_TORCH = "tor" + "ch"
_WARP = _TORCH + "_warp"

# Packages a shared module must never load.
HEAVY = ("jax", _TORCH, "mujoco", "brax", "mujoco_playground", "ml_collections")

TORCH_IMPORTS = {
    "import torch": rf"^\s*import\s+{_TORCH}(\.\w+)*\s*(,|$|\s+as\b|#)",
    "from torch import": rf"^\s*from\s+{_TORCH}(\.\w+)*\s+import\b",
    "dynamic torch import": rf"(import_module|importorskip|__import__)\(\s*['\"]{_TORCH}['\".]",
    "import torch_warp": rf"^\s*(import|from)\s+humanoid_lab\.{_WARP}\b",
    "from humanoid_lab import torch_warp": rf"^\s*from\s+humanoid_lab\s+import\s+[^\n]*\b{_WARP}\b",
    "relative torch_warp import": rf"^\s*from\s+\.+\s*{_WARP}\b|^\s*from\s+\.+\s+import\s+[^\n]*\b{_WARP}\b",
    "dynamic torch_warp import": rf"import_module\(\s*['\"]humanoid_lab\.{_WARP}\b",
}


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
    proc = _python(_block((_TORCH,)) + _imports(CORE))
    assert proc.returncode == 0, (
        "a JAX-side core module needs torch. Only humanoid_lab/torch_warp "
        f"and tests/parity may import it:\n{proc.stderr}"
    )


def _offenders(paths):
    out = {}
    for path in paths:
        source = path.read_text()
        hits = [label for label, pat in TORCH_IMPORTS.items() if re.search(pat, source, re.MULTILINE)]
        if hits:
            out[str(path)] = hits
    return out


def test_no_torch_import_outside_torch_warp_and_tests_parity():
    warp_dir = SRC_DIR / _WARP
    src = [p for p in sorted(SRC_DIR.rglob("*.py")) if warp_dir not in p.parents]
    unit = sorted(UNIT_DIR.rglob("*.py"))
    integration = sorted((UNIT_DIR.parent / "integration").rglob("*.py"))
    assert len(src) > 20 and len(unit) > 20 and len(integration) > 20
    offenders = _offenders(src + unit + integration)
    assert not offenders, (
        f"torch or torch_warp imported outside src/humanoid_lab/{_WARP}/ and "
        f"tests/parity: {offenders}. The core, tests/unit and tests/integration "
        "must run without torch."
    )


def test_the_scan_catches_each_import_form(tmp_path):
    forms = [
        f"import {_TORCH}\n",
        f"import {_TORCH}.nn as nn\n",
        f"    from {_TORCH} import nn\n",
        f"x = pytest.importorskip('{_TORCH}')\n",
        f"from humanoid_lab.{_WARP} import env\n",
        f"from humanoid_lab import {_WARP}\n",
        f"from ..{_WARP} import ppo\n",
        f"importlib.import_module('humanoid_lab.{_WARP}.env')\n",
    ]
    for i, form in enumerate(forms):
        p = tmp_path / f"m{i}.py"
        p.write_text(form)
        assert _offenders([p]), form
    clean = tmp_path / "clean.py"
    clean.write_text(f"# the {_TORCH} runner lives in humanoid_lab.{_WARP}\nimport {_WARP}_free\n")
    assert not _offenders([clean])
