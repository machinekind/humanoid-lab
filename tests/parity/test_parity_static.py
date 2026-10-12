"""Static checks of src/humanoid_lab/torch_warp, read as source.

torch_warp reads an env only through public names: config,
resolved_constants(), mj_model and the other properties. It must also run on
native Windows, so it uses no POSIX-only module and no torch.compile. These
tests parse the files and import nothing, so they run without torch.
"""

from __future__ import annotations

import ast
import re

from humanoid_lab import paths

TORCH_WARP = paths.REPO_ROOT / "src" / "humanoid_lab" / "torch_warp"

PRIVATE_ENV_READ = re.compile(r"\b_?j?env\._")

# Modules that are missing or broken on native Windows.
POSIX_ONLY_MODULES = {"signal", "fcntl", "resource"}

# Dotted names that are missing or broken on native Windows.
POSIX_ONLY_CALLS = {"torch.compile", "os.fork"}


def _sources():
    files = sorted(TORCH_WARP.rglob("*.py"))
    assert files, f"no .py files under {TORCH_WARP}"
    return [(p, p.read_text(encoding="utf-8")) for p in files]


def _dotted(node):
    """Return the dotted name of a Name or Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def _windows_breakers(text):
    """Return each POSIX-only import and each use of a POSIX-only name in text."""
    hits = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.split(".")[0] in POSIX_ONLY_MODULES]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.split(".")[0] in POSIX_ONLY_MODULES:
                hits.append(node.module)
            hits += [f"{node.module}.{a.name}" for a in node.names if f"{node.module}.{a.name}" in POSIX_ONLY_CALLS]
        elif isinstance(node, ast.Attribute) and _dotted(node) in POSIX_ONLY_CALLS:
            hits.append(_dotted(node))
    return hits


def _private_reads(text):
    return [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1) if PRIVATE_ENV_READ.search(line)]


def test_torch_warp_reads_no_private_env_attribute():
    hits = [
        f"{p.relative_to(paths.REPO_ROOT).as_posix()}:{i}: {line}"
        for p, text in _sources()
        for i, line in _private_reads(text)
    ]
    assert not hits, "torch_warp reads a private env attribute:\n" + "\n".join(hits)


def test_torch_warp_uses_no_posix_only_module_or_torch_compile():
    hits = [
        f"{p.relative_to(paths.REPO_ROOT).as_posix()}: {name}"
        for p, text in _sources()
        for name in _windows_breakers(text)
    ]
    assert not hits, "torch_warp uses a name that native Windows lacks:\n" + "\n".join(hits)
