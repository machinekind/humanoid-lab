"""Guard the import direction between the shared modules, the core and Torch.

The shared task modules import only array_api_compat, numpy, the standard
library and each other. The JAX-side core runs without torch. Only
src/humanoid_lab/torch_warp/ and tests/parity import torch or torch_warp, so
`./run.sh test`, `./run.sh test-slow` and every JAX tool keep working on an
install without torch.

These checks read source only. tests/integration/test_import_isolation.py
imports the same modules in a fresh interpreter with the packages blocked.
"""

import ast
import re
import sys
from pathlib import Path

UNIT_DIR = Path(__file__).resolve().parent
SRC_DIR = UNIT_DIR.parent.parent / "src" / "humanoid_lab"

# Keep in step with SHARED in tests/integration/test_import_isolation.py.
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

# Written in pieces so the static scan below does not match this file.
_TORCH = "tor" + "ch"
_WARP = _TORCH + "_warp"

# Top-level packages a shared module may import besides the standard library
# and the other shared modules.
SHARED_MAY_IMPORT = {"__future__", "array_api_compat", "numpy"}

TORCH_IMPORTS = {
    "import torch": rf"^\s*import\s+{_TORCH}(\.\w+)*\s*(,|$|\s+as\b|#)",
    "from torch import": rf"^\s*from\s+{_TORCH}(\.\w+)*\s+import\b",
    "dynamic torch import": rf"(import_module|importorskip|__import__)\(\s*['\"]{_TORCH}['\".]",
    "import torch_warp": rf"^\s*(import|from)\s+humanoid_lab\.{_WARP}\b",
    "from humanoid_lab import torch_warp": rf"^\s*from\s+humanoid_lab\s+import\s+[^\n]*\b{_WARP}\b",
    "relative torch_warp import": rf"^\s*from\s+\.+\s*{_WARP}\b|^\s*from\s+\.+\s+import\s+[^\n]*\b{_WARP}\b",
    "dynamic torch_warp import": rf"import_module\(\s*['\"]humanoid_lab\.{_WARP}\b",
}


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


def _shared_files():
    """{path: package} for each shared module and each package __init__ it runs."""
    files = {}
    for module in SHARED:
        parts = module.split(".")
        files[SRC_DIR.joinpath(*parts[1:-1], parts[-1] + ".py")] = ".".join(parts[:-1])
        for i in range(1, len(parts)):
            files[SRC_DIR.joinpath(*parts[1:i], "__init__.py")] = ".".join(parts[:i])
    return files


def _imported_names(node, package):
    """The module names one import statement loads, relative ones resolved."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    base = node.module or ""
    if node.level:
        base = ".".join([package.rsplit(".", node.level - 1)[0], *([base] if base else [])])
    if base in SHARED or not base.startswith("humanoid_lab"):
        return [base]
    return [f"{base}.{alias.name}" for alias in node.names]


def _shared_offenders(path, package):
    """Imports in one file, at any depth, that a shared module must not make."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            if getattr(node.func, "attr", getattr(node.func, "id", None)) in ("import_module", "__import__"):
                bad.append(f"line {node.lineno}: dynamic import")
            continue
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        for name in _imported_names(node, package):
            root = name.split(".")[0]
            if root == "humanoid_lab":
                ok = name in SHARED
            else:
                ok = root in SHARED_MAY_IMPORT or root in sys.stdlib_module_names
            if not ok:
                bad.append(f"line {node.lineno}: {name}")
    return bad


def test_shared_modules_import_only_numpy_array_api_compat_and_each_other():
    files = _shared_files()
    assert all(path.is_file() for path in files), [str(p) for p in files if not p.is_file()]
    offenders = {str(path): bad for path, package in files.items() if (bad := _shared_offenders(path, package))}
    assert not offenders, (
        "a shared module imports a package it must not (see the module "
        f"docstring of envs/constants.py): {offenders}"
    )


def test_the_shared_scan_catches_each_import_form(tmp_path):
    forms = [
        "import jax\n",
        "import jax.numpy as jp\n",
        "from brax import envs\n",
        "try:\n    import mujoco\nexcept ImportError:\n    mujoco = None\n",
        "def f():\n    from ml_collections import config_dict\n",
        "from humanoid_lab import registry\n",
        "from humanoid_lab.envs import joystick\n",
        "import humanoid_lab.envs.joystick\n",
        "from . import base\n",
        "from ..dr import randomize\n",
        "importlib.import_module('jax')\n",
        f"import {_TORCH}\n",
    ]
    for i, form in enumerate(forms):
        p = tmp_path / f"m{i}.py"
        p.write_text(form)
        assert _shared_offenders(p, "humanoid_lab.envs"), form
    clean = tmp_path / "clean.py"
    clean.write_text(
        "from __future__ import annotations\n"
        "import math\n"
        "import numpy as np\n"
        "from array_api_compat import array_namespace\n"
        "from humanoid_lab.envs import progress, symmetry\n"
        "from humanoid_lab.envs.constants import xp_of\n"
        "from . import push\n"
        "from ..rewards import terms\n"
    )
    assert not _shared_offenders(clean, "humanoid_lab.envs")
