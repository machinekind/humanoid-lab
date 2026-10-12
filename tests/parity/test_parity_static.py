"""Static checks of src/humanoid_lab/torch_warp, read as text.

torch_warp reads an env only through public names: config,
resolved_constants(), mj_model and the other properties. It must also run on
native Windows, so it uses no POSIX-only module and no torch.compile. These
tests read the files and import nothing, so they run without torch.
"""

from __future__ import annotations

import re

from humanoid_lab import paths

TORCH_WARP = paths.REPO_ROOT / "src" / "humanoid_lab" / "torch_warp"

PRIVATE_ENV_READ = re.compile(r"\b_?j?env\._")

# label -> pattern. Each one is missing or broken on native Windows.
NOT_ON_WINDOWS = {
    "torch.compile": re.compile(r"\btorch\.compile\b"),
    "import signal": re.compile(r"^\s*(import|from)\s+signal\b", re.MULTILINE),
    "fcntl": re.compile(r"\bfcntl\b"),
    "import resource": re.compile(r"^\s*(import|from)\s+resource\b", re.MULTILINE),
    "os.fork": re.compile(r"\bos\.fork\b"),
}
SYNC = re.compile(r"\btorch\.cuda\.synchronize\b")


def _sources():
    files = sorted(TORCH_WARP.rglob("*.py"))
    assert files, f"no .py files under {TORCH_WARP}"
    return [(p, p.read_text(encoding="utf-8")) for p in files]


def _private_reads(text):
    return [(i, line.strip()) for i, line in enumerate(text.splitlines(), 1) if PRIVATE_ENV_READ.search(line)]


def test_torch_warp_reads_no_private_env_attribute():
    hits = [
        f"{p.relative_to(paths.REPO_ROOT).as_posix()}:{i}: {line}"
        for p, text in _sources()
        for i, line in _private_reads(text)
    ]
    assert not hits, "torch_warp reads a private env attribute:\n" + "\n".join(hits)


def test_the_scan_catches_private_reads(tmp_path):
    for text in ("env._config", "self.jenv._consts", "self._jenv._x"):
        f = tmp_path / "probe.py"
        f.write_text(f"x = {text}\n", encoding="utf-8")
        assert _private_reads(f.read_text(encoding="utf-8")), text
    for text in ("environment._x", "self._env_count", "self.jenv.config", "physics_env._x"):
        assert not _private_reads(f"x = {text}\n"), text


def test_torch_warp_uses_no_posix_only_module_or_torch_compile():
    hits = []
    for p, text in _sources():
        rel = p.relative_to(paths.REPO_ROOT).as_posix()
        hits += [f"{rel}: {label}" for label, pat in NOT_ON_WINDOWS.items() if pat.search(text)]
        if p.name != "bench.py" and SYNC.search(text):
            hits.append(f"{rel}: torch.cuda.synchronize outside bench.py")
    assert not hits, "\n".join(hits)


def test_the_windows_scan_catches_each_form():
    for label, text in (
        ("torch.compile", "f = torch.compile(f)"),
        ("import signal", "import signal"),
        ("import signal", "from signal import SIGKILL"),
        ("fcntl", "import fcntl"),
        ("import resource", "import resource"),
        ("os.fork", "pid = os.fork()"),
    ):
        assert NOT_ON_WINDOWS[label].search(text), text
    for label, text in (
        ("import signal", "from humanoid_lab.envs import signals"),
        ("import signal", "from humanoid_lab.envs import backend, signals"),
        ("import resource", "# the resource budget"),
    ):
        assert not NOT_ON_WINDOWS[label].search(text), text
