"""run.sh: the Windows venv fallback and the Torch+MJWarp verbs.

Each case copies run.sh into a temporary dir with a fake interpreter that
logs its arguments. The fake sits where a Windows venv puts python, so the
fallback runs on any host with bash. Python's timeout stands in for the
timeout binary, which macOS lacks.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

if shutil.which("bash") is None:
    pytest.skip("bash is not on PATH", allow_module_level=True)

RUN_SH = Path(__file__).resolve().parents[2] / "run.sh"


def _fake_python(path, log):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'#!/usr/bin/env bash\necho "{path.name} $*" >> "{log}"\n')
    path.chmod(0o755)


def _repo(tmp_path, posix=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(RUN_SH, repo / "run.sh")
    log = tmp_path / "calls.log"
    _fake_python(repo / ".venv" / "Scripts" / "python.exe", log)
    if posix:
        _fake_python(repo / ".venv" / "bin" / "python", log)
    return repo, log


def _run(repo, *args):
    p = subprocess.run(
        ["bash", "run.sh", *args],
        cwd=repo,
        timeout=60,
        capture_output=True,
        text=True,
        check=False,
    )
    assert p.returncode == 0, p.stderr
    return p


def test_train_warp_runs_the_torch_entry_point_through_the_windows_venv(tmp_path):
    repo, log = _repo(tmp_path)
    _run(repo, "train-warp", "a=b")
    assert log.read_text().splitlines() == [
        "python.exe -m humanoid_lab.torch_warp.ppo a=b"
    ]


def test_bench_warp_runs_the_bench(tmp_path):
    repo, log = _repo(tmp_path)
    _run(repo, "bench-warp", "bench.iters=1")
    assert log.read_text().splitlines() == [
        "python.exe -m humanoid_lab.torch_warp.bench bench.iters=1"
    ]


def test_the_posix_venv_wins_when_both_exist(tmp_path):
    repo, log = _repo(tmp_path, posix=True)
    _run(repo, "train-warp", "a=b")
    assert log.read_text().splitlines() == ["python -m humanoid_lab.torch_warp.ppo a=b"]
