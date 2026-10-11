"""jobs/video_watch.sh: runs the watcher with a fake PY in a temporary
repo-shaped dir.

The fake PY records each render call and touches the clip. It hands `-c`
calls to the real interpreter, so the train_log fields still print. Each case
waits out INTERVAL=1 and MAX_IDLE=2, about 3 s.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

if shutil.which("bash") is None:
    pytest.skip("bash is not on PATH", allow_module_level=True)

SCRIPT = (Path(__file__).resolve().parents[2] / "jobs" / "video_watch.sh").resolve()


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("")
    return repo


def _fake_py(tmp_path, code=0):
    calls = tmp_path / "calls.log"
    fake = tmp_path / "fake_py"
    fake.write_text(
        "#!/usr/bin/env bash\n"
        f'if [ "${{1:-}}" = "-c" ]; then exec "{sys.executable}" "$@"; fi\n'
        f'echo "$*" >> "{calls}"\n'
        "prev=\n"
        'for a in "$@"; do\n'
        '    [ "$prev" = "--out" ] && touch "$a"\n'
        '    prev="$a"\n'
        "done\n"
        f"exit {code}\n"
    )
    fake.chmod(0o755)
    return fake, calls


def _ckpt(run_dir, step):
    d = run_dir / "checkpoints" / f"{step:012d}"
    d.mkdir(parents=True)
    (d / "ppo_network_config.json").write_text("{}")


def _run(repo, py=None, **env_extra):
    env = {k: v for k, v in os.environ.items() if k != "PY"}
    env.update(RUN_NAME="r", INTERVAL="1", MAX_IDLE="2")
    if py is not None:
        env["PY"] = str(py)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _calls(calls):
    return calls.read_text() if calls.exists() else ""


def test_py_is_required(tmp_path):
    repo = _repo(tmp_path)
    p = _run(repo)
    assert p.returncode != 0
    assert "PY" in p.stderr


def test_run_name_must_be_a_plain_name(tmp_path):
    repo = _repo(tmp_path)
    fake, calls = _fake_py(tmp_path)
    p = _run(repo, fake, RUN_NAME="../x")
    assert p.returncode != 0
    assert _calls(calls) == ""


def test_the_latest_checkpoint_renders_with_its_step(tmp_path):
    repo = _repo(tmp_path)
    fake, calls = _fake_py(tmp_path)
    run = repo / "runs" / "r"
    _ckpt(run, 100)
    _ckpt(run, 200)
    (run / "run.json").write_text("{}")
    p = _run(repo, fake)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "VIDEO step=200" in p.stdout
    assert "IDLE" in p.stdout
    log = _calls(calls)
    assert "-m humanoid_lab.eval.video" in log
    assert "--checkpoint 200" in log
    assert "--scenario walk_ramp" in log
    assert "--checkpoint 100" not in log
    assert (run / "videos" / "step_000000000200.mp4").exists()


def test_a_failed_render_prints_fail_and_exits_nonzero(tmp_path):
    repo = _repo(tmp_path)
    fake, _ = _fake_py(tmp_path, code=1)
    run = repo / "runs" / "r"
    _ckpt(run, 200)
    (run / "run.json").write_text("{}")
    p = _run(repo, fake)
    assert "FAIL step=200" in p.stdout
    assert "IDLE" in p.stdout
    assert p.returncode == 1


def test_nothing_renders_before_run_json_exists(tmp_path):
    repo = _repo(tmp_path)
    fake, calls = _fake_py(tmp_path)
    run = repo / "runs" / "r"
    _ckpt(run, 200)
    p = _run(repo, fake)
    assert "NORUN" in p.stdout
    assert p.returncode == 0
    assert "-m" not in _calls(calls)
    assert not (run / "videos").exists()


def test_done_ends_the_loop_after_the_last_clip(tmp_path):
    repo = _repo(tmp_path)
    fake, _ = _fake_py(tmp_path)
    run = repo / "runs" / "r"
    _ckpt(run, 200)
    (run / "run.json").write_text("{}")
    (run / "DONE").write_text("")
    p = _run(repo, fake, MAX_IDLE="30")
    assert "VIDEO" in p.stdout
    assert "DONE" in p.stdout
    assert p.stdout.index("VIDEO") < p.stdout.index("DONE")
    assert "IDLE" not in p.stdout
    assert p.returncode == 0


def test_train_log_fields_are_optional(tmp_path):
    repo = _repo(tmp_path)
    fake, _ = _fake_py(tmp_path)
    run = repo / "runs" / "r"
    _ckpt(run, 200)
    (run / "run.json").write_text("{}")
    (run / "train_log.jsonl").write_text('{"sps": 5}\n')
    p = _run(repo, fake)
    video = [line for line in p.stdout.splitlines() if line.startswith("VIDEO")]
    assert len(video) == 1
    assert video[0].endswith(" sps=5")
    assert "reward_x1000" not in video[0]
