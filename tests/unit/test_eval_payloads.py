"""The payloads that measure trained runs: jobs/eval_runs.sh, the eval stage
of jobs/train.sh, and jobs/train_chain.sh's handling of a phase whose evals
failed.

A stub `python3` first on PATH records each call's arguments and the
JAX_PLATFORMS and MUJOCO_GL it ran with, plays the module it was called as,
and exits with a scripted code. Nothing trains or loads a model. The stub's
training writes run.json and a complete checkpoint when asked, and its evals
write their default outputs, so the payloads' file tests see real files. An
inline `python3 -` script runs on the real interpreter. A stub `timeout`
records its arguments and runs the command, or exits 124 like an expired
bound. Like GNU timeout, it moves into a new process group unless given
--foreground.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from humanoid_lab import paths
from humanoid_lab.eval.courses import runner

JOBS_DIR = paths.REPO_ROOT / "jobs"
PAYLOADS = [JOBS_DIR / name for name in ("eval_runs.sh", "train.sh", "train_chain.sh")]
BASH = shutil.which("bash")
# Step dir names as brax writes them: zero-padded to 12 digits. An 8 or a 9
# makes a name an invalid octal literal, so a script that reads it without
# 10# fails on it.
OLDER = "000052428800"
STEP = "000104857600"
EVAL_MODULES = ("humanoid_lab.eval.courses", "humanoid_lab.eval.battery", "humanoid_lab.eval.report")

# Scripted exit codes: STUB_FAIL holds `<what>[:<run>]=<rc>` words, where
# <what> is resolve, train, courses, check (courses --check), battery or
# report and <run> limits the word to one run. Any other call exits 0.
# STUB_WRITE_RUN names the runs a training call writes; `@` is the call's
# own run_name, or train.py's <task>_<timestamp> name when it has none. Each
# gets one complete checkpoint, the STUB_STEP step dir. STUB_HANG=<eval>
# makes that eval write its pid to hang.pid and sleep; it takes 0.2 s to
# exit on SIGTERM, so only a caller that waits for it sees it gone.
STUB_PY = r'''
import json
import os
import signal
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
stub = Path(os.environ["STUB_DIR"])
with open(stub / "calls.jsonl", "a") as f:
    f.write(json.dumps({
        "args": argv,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }) + "\n")

if argv[:1] == ["-"]:
    os.execv(sys.executable, [sys.executable, *argv])


def scripted(what, run=""):
    for word in os.environ.get("STUB_FAIL", "").split():
        key, rc = word.rsplit("=", 1)
        name, _, member = key.partition(":")
        if name == what and member in ("", run):
            return int(rc)
    return 0


def flag(name):
    return argv[argv.index(name) + 1] if name in argv else None


module, rest = argv[1], argv[2:]
run_names = [a.split("=", 1)[1] for a in rest if a.startswith("run_name=")]
if module == "humanoid_lab.train" and rest[:3] == ["--cfg", "job", "--resolve"]:
    resolved = stub / "resolved"
    if resolved.exists():
        print(resolved.read_text(), end="")
    else:
        print("run_name:", os.environ.get("STUB_RUN_NAME") or (run_names or ["null"])[-1])
    sys.exit(scripted("resolve"))
if module == "humanoid_lab.train":
    own = run_names[-1] if run_names else "joystick_20261006_120000"
    steps = int(os.environ["STUB_STEP"])
    for name in os.environ.get("STUB_WRITE_RUN", "").split():
        name = own if name == "@" else name
        run_dir = Path("runs") / name
        step = run_dir / "checkpoints" / os.environ["STUB_STEP"]
        step.mkdir(parents=True, exist_ok=True)
        (step / "ppo_network_config.json").write_text("{}")
        run = {
            "run_name": name, "task": "joystick", "num_timesteps": steps,
            "early_stopped": False, "stopped_at_steps": steps,
            "checkpoint_dir": str(run_dir.resolve() / "checkpoints"),
            "hydra_config": {"run_name": None, "seed": 0},
        }
        extra = stub / f"run_{name}.json"
        if extra.exists():
            run.update(json.loads(extra.read_text()))
        (run_dir / "run.json").write_text(json.dumps(run))
    sys.exit(scripted("train", own))
if module == f"humanoid_lab.eval.{os.environ.get('STUB_HANG')}":
    signal.signal(signal.SIGTERM, lambda *_: (time.sleep(0.2), sys.exit(143)))
    (stub / "hang.tmp").write_text(str(os.getpid()))
    os.replace(stub / "hang.tmp", stub / "hang.pid")
    time.sleep(60)
run = Path(flag("--run") or ".").name
if module == "humanoid_lab.eval.courses":
    if "--check" in rest:
        rc = scripted("check", run)
        print("current: stub" if rc == 0 else f"not current: runs/{run}/courses.json: stub reason")
        sys.exit(rc)
    rc = scripted("courses", run)
    if rc == 0:
        out = Path(flag("--out") or f"runs/{run}/courses.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"schema": 1}\n')
    sys.exit(rc)
if module == "humanoid_lab.eval.battery":
    rc = scripted("battery", run)
    if rc == 0:
        steps = [p.name for p in (Path("runs") / run / "checkpoints").iterdir() if p.name.isdigit()]
        doc = {"run": run, "checkpoint": max(steps, key=int)}
        (Path("runs") / run / "battery.json").write_text(json.dumps(doc, indent=2))
    sys.exit(rc)
if module == "humanoid_lab.eval.report":
    rc = scripted("report", run)
    if rc == 0:
        (Path("runs") / run / "eval_report.md").write_text("# stub\n")
    sys.exit(rc)
sys.exit(f"stub python3: unexpected call {argv}")
'''

STUB_TIMEOUT_PY = r'''
import os
import sys

args = sys.argv[1:]
with open(os.path.join(os.environ["STUB_DIR"], "timeout.log"), "a") as f:
    f.write(" ".join(args) + "\n")
if os.environ.get("STUB_TIMEOUT_RC"):
    sys.exit(int(os.environ["STUB_TIMEOUT_RC"]))
foreground = False
while args[0].startswith("-"):
    foreground |= args.pop(0) == "--foreground"
args.pop(0)
if not foreground:
    os.setpgid(0, 0)
os.execvp(args[0], args)
'''


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture(scope="session")
def stubs(tmp_path_factory):
    """bin/ holds the python3 stub and tbin/ the timeout stub. They are
    written once per session: on macOS the first run of a newly written
    executable took about 0.35 s, and a whole stubbed payload run after it
    0.08 s."""
    top = tmp_path_factory.mktemp("stubs")
    (top / "bin").mkdir()
    (top / "stub.py").write_text(STUB_PY)
    _write_exe(top / "bin" / "python3", f'#!/bin/sh\nexec "{sys.executable}" -S "{top / "stub.py"}" "$@"\n')
    (top / "tbin").mkdir()
    (top / "timeout.py").write_text(STUB_TIMEOUT_PY)
    _write_exe(top / "tbin" / "timeout", f'#!/bin/sh\nexec "{sys.executable}" -S "{top / "timeout.py"}" "$@"\n')
    return SimpleNamespace(bin=top / "bin", tbin=top / "tbin")


@pytest.fixture
def jobs(tmp_path, stubs):
    """A repo-root stand-in (pyproject.toml, the real configs/ and jobs/) and
    the stubs' state dir. `jobs.run(payload, **env)` runs a payload from the
    root and returns its exit code, its recorded calls and its output.
    `jobs.start(payload, **env)` starts one as the leader of a new process
    group and returns its Popen. Teardown kills what a test left running."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("")
    (root / "configs").symlink_to(paths.CONFIGS_DIR)
    # train.sh and train_chain.sh call jobs/ scripts by their relative path.
    (root / "jobs").symlink_to(JOBS_DIR)
    stub = tmp_path / "stub"
    stub.mkdir()
    started = []

    def environ(path: str | None, env) -> dict[str, str]:
        for name in ("calls.jsonl", "timeout.log", "hang.pid"):
            (stub / name).unlink(missing_ok=True)
        return {
            "PATH": path or os.pathsep.join([str(stubs.bin), str(stubs.tbin), "/usr/bin", "/bin"]),
            "HOME": str(root),
            "STUB_DIR": str(stub),
            "STUB_STEP": STEP,
            **env,
        }

    def run(payload: str, path: str | None = None, **env) -> SimpleNamespace:
        done = subprocess.run(
            [BASH, str(JOBS_DIR / payload)],
            cwd=root, env=environ(path, env), capture_output=True, text=True, check=False, timeout=120,
        )
        log = stub / "calls.jsonl"
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        timeouts = (stub / "timeout.log").read_text().splitlines() if (stub / "timeout.log").exists() else []
        return SimpleNamespace(
            code=done.returncode, calls=calls, out=done.stdout + done.stderr, timeouts=timeouts
        )

    def start(payload: str, **env) -> subprocess.Popen:
        proc = subprocess.Popen(
            [BASH, str(JOBS_DIR / payload)], cwd=root, env=environ(None, env),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        started.append(proc)
        return proc

    yield SimpleNamespace(root=root, stub=stub, bin=stubs.bin, run=run, start=start)
    for proc in started:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    hang = stub / "hang.pid"
    if hang.exists():
        pid = int(hang.read_text())
        if _alive(pid):
            os.kill(pid, signal.SIGKILL)


def _make_run(root: Path, name: str, *, steps=(STEP,), complete=True, run_json=True,
              run=None, battery=None) -> Path:
    """runs/<name>, an earlier training's run dir: numeric step dirs, each
    complete except the newest when `complete` is false, and a run.json an
    hour old. `battery` names the step its battery.json is for."""
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (run_dir / "checkpoints" / step).mkdir(parents=True, exist_ok=True)
    for step in steps:
        if complete or step != max(steps, key=int):
            (run_dir / "checkpoints" / step / "ppo_network_config.json").write_text("{}")
    if run_json:
        (run_dir / "run.json").write_text(json.dumps({"run_name": name, **(run or {})}))
        _age(run_dir / "run.json")
    if battery is not None:
        (run_dir / "battery.json").write_text(json.dumps({"run": name, "checkpoint": battery}, indent=2))
    return run_dir


def _age(path: Path, seconds: float = 3600.0) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t))


def _alive(pid: int) -> bool:
    """True while the process exists, a zombie included."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _hung_pid(jobs, proc) -> int:
    """The pid of the eval STUB_HANG named, once it sleeps."""
    pid_file = jobs.stub / "hang.pid"
    deadline = time.monotonic() + 30
    while not pid_file.exists():
        assert proc.poll() is None, f"the payload exited rc={proc.returncode} before the eval started"
        assert time.monotonic() < deadline, "the eval never started"
        time.sleep(0.02)
    return int(pid_file.read_text())


def _module(call) -> str | None:
    return call["args"][1] if call["args"][:1] == ["-m"] else None


def _evals(calls) -> list[tuple[str, str]]:
    """(eval, run) of every eval call, in order."""
    out = []
    for call in calls:
        module = _module(call)
        if module in EVAL_MODULES:
            args = call["args"]
            out.append((module.rsplit(".", 1)[1], Path(args[args.index("--run") + 1]).name))
    return out


def _courses_args(call):
    assert _module(call) == "humanoid_lab.eval.courses"
    return runner.parser().parse_args(call["args"][2:])


def _train_overrides(call) -> list[str]:
    rest = call["args"][2:]
    return rest[3:] if rest[:3] == ["--cfg", "job", "--resolve"] else rest


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


def _code(path: Path) -> str:
    """The payload with its comments dropped."""
    return "\n".join(line.split("#")[0] for line in path.read_text().splitlines())


def _files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix() for p in root.rglob("*")
        if p.is_file() and p.relative_to(root).parts[0] not in ("configs", "jobs")
    }


# -- text ------------------------------------------------------------------------


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_every_payload_parses(path):
    done = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_every_payload_is_executable(path):
    """The header examples run each payload as ./jobs/<name>."""
    assert os.access(path, os.X_OK), f"{path.name} is not executable"


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_no_payload_reaches_another_host(path):
    code = _code(path)
    for tool in ("ssh", "scp", "rsync"):
        assert not re.search(rf"\b{tool}\b", code), f"{path.name} runs {tool}"


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_every_variable_a_payload_reads_is_in_its_header(path):
    """A parameter is every uppercase variable the code reads and does not
    set itself. The header documents each one."""
    text = path.read_text()
    header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
    code = _code(path)
    params = set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)="\$\{\1[:?-]', code, re.MULTILINE))
    params |= set(re.findall(r'^: "\$\{([A-Z_][A-Z0-9_]*):[?=]', code, re.MULTILINE))
    assert params
    own = set(re.findall(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=", code, re.MULTILINE)) - params
    used = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", code))
    for name in sorted(params | (used - own - {"PWD"})):
        assert re.search(rf"^#\s+{name}\s", header, re.MULTILINE), f"{path.name}: {name} undocumented"


def _examples(path: Path) -> list[dict[str, str]]:
    """The environment of each worked example in a payload's header: a
    header line that ends in ./jobs/<payload>, continuation lines joined."""
    header = "\n".join(line[1:] for line in path.read_text().splitlines() if line.startswith("#"))
    header = re.sub(r"\\\n", " ", header)
    examples = []
    for line in header.splitlines():
        if line.rstrip().endswith(f"./jobs/{path.name}"):
            pairs = [word.split("=", 1) for word in shlex.split(line)[:-1]]
            assert all(len(pair) == 2 for pair in pairs), line
            examples.append(dict(pairs))
    return examples


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_every_header_example_runs(jobs, path):
    """Each worked example runs against fixture run dirs and exits 0. Every
    courses call parses with the courses CLI's parser, battery and report
    get the run alone, every training override composes, and only the evals
    run on CPU."""
    for name in ("walk_a", "walk_b"):
        _make_run(jobs.root, name, battery=STEP)
    examples = _examples(path)
    assert examples, f"{path.name} has no worked example"
    for env in examples:
        res = jobs.run(path.name, STUB_WRITE_RUN="@", **env)
        assert res.code == 0, (env, res.out)
        assert res.calls, env
        for call in res.calls:
            module = _module(call)
            if module == "humanoid_lab.train":
                _compose(_train_overrides(call))
                assert call["jax_platforms"] is None
            elif module == "humanoid_lab.eval.courses":
                _courses_args(call)
                assert call["jax_platforms"] == "cpu"
            elif module in EVAL_MODULES:
                assert call["args"][2:3] == ["--run"] and len(call["args"]) == 4
                assert call["jax_platforms"] == "cpu"
            else:
                assert call["args"][:1] == ["-"], call


# -- train.sh: the eval stage ------------------------------------------------------


def test_train_measures_the_run_it_trained(jobs):
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@")
    assert res.code == 0, res.out
    resolve, train, *evals = res.calls
    assert _train_overrides(resolve) == _train_overrides(train)
    assert _evals(evals) == [("courses", "r"), ("battery", "r"), ("report", "r")]
    assert [c["jax_platforms"] for c in res.calls] == [None, None, "cpu", "cpu", "cpu"]
    args = _courses_args(evals[0])
    assert (str(args.run), args.ground, args.seeds, args.seed_base, args.out) == ("runs/r", "flat", 8, 0, None)
    # REDO=true: a reused run dir must not keep an earlier training's files.
    assert not args.skip_if_current and not args.check
    # --foreground keeps the stage in the payload's process group.
    assert res.timeouts == ["--foreground 1800 bash jobs/eval_runs.sh"]
    assert {"courses.json", "battery.json", "eval_report.md"} <= {p.name for p in (jobs.root / "runs/r").iterdir()}
    assert json.loads((jobs.root / "runs/r/battery.json").read_text())["checkpoint"] == STEP
    assert not list((jobs.root / "runs").glob(".train_started*"))


@pytest.mark.parametrize(
    "evals",
    [{}, {"STUB_FAIL": "train=3 courses=1"}, {"EVAL_TIMEOUT": "60", "STUB_TIMEOUT_RC": "124"}],
    ids=["evals-pass", "evals-fail", "evals-time-out"],
)
def test_a_failed_training_that_wrote_run_json_is_measured_and_keeps_its_code_whatever_the_evals_do(jobs, evals):
    env = {"STUB_FAIL": "train=3", **evals}
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", **env)
    assert res.code == 3, res.out
    assert "completes them" not in res.out
    if "STUB_TIMEOUT_RC" not in evals:
        assert _evals(res.calls) == [("courses", "r"), ("battery", "r"), ("report", "r")]


def test_a_failed_training_without_run_json_makes_two_calls_and_keeps_its_code(jobs):
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_FAIL="train=3")
    assert (res.code, len(res.calls)) == (3, 2)


def test_the_resolved_run_name_wins_over_run_name(jobs):
    """RUN_NAME=a with run_name=b in RUN_ARGS trains runs/b: Hydra keeps the
    last value, and the resolve prints it."""
    res = jobs.run(
        "train.sh", ROBOT="roboto_origin", RUN_NAME="a", RUN_ARGS="run_name=b", STUB_WRITE_RUN="b",
    )
    assert res.code == 0, res.out
    assert _compose(_train_overrides(res.calls[1])).run_name == "b"
    assert _evals(res.calls) == [("courses", "b"), ("battery", "b"), ("report", "b")]
    # Any other source of the resolved name, such as an experiment's own.
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="a", STUB_RUN_NAME="c", STUB_WRITE_RUN="c")
    assert _evals(res.calls)[0] == ("courses", "c")


def test_a_quoted_resolved_run_name_is_unquoted(jobs):
    """The resolve prints a run_name that YAML would read as a date quoted,
    as run_name: '2026-10-06'. The run dir is runs/2026-10-06."""
    printed = OmegaConf.to_yaml(_compose(["robot=roboto_origin", "run_name=2026-10-06"]), resolve=True)
    assert "run_name: '2026-10-06'" in printed.splitlines()
    res = jobs.run(
        "train.sh", ROBOT="roboto_origin", RUN_NAME="2026-10-06", STUB_RUN_NAME="'2026-10-06'",
        STUB_WRITE_RUN="@",
    )
    assert res.code == 0, res.out
    assert _evals(res.calls) == [("courses", "2026-10-06"), ("battery", "2026-10-06"), ("report", "2026-10-06")]


def test_a_named_run_without_a_fresh_run_json_exits_75(jobs):
    _make_run(jobs.root, "r")
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r")
    assert (res.code, len(res.calls)) == (75, 2)
    assert "no fresh runs/r/run.json" in res.out


def test_an_unnamed_run_is_found_by_its_fresh_run_json(jobs):
    """train.py names the run <task>_<timestamp>. An older run.json is not
    this training's."""
    _make_run(jobs.root, "old")
    res = jobs.run("train.sh", ROBOT="roboto_origin", STUB_WRITE_RUN="@")
    assert res.code == 0, res.out
    assert {run for _, run in _evals(res.calls)} == {"joystick_20261006_120000"}


def _fresh_runs(jobs, *, seeds, **env):
    """A resolve that prints a config with seed 0, and an unnamed training
    that writes one run per seed, as when another job trains in the same
    checkout."""
    (jobs.stub / "resolved").write_text("task:\n  name: joystick\nrun_name: null\nseed: 0\n")
    names = ("joystick_a", "joystick_b")[: len(seeds)]
    for name, seed in zip(names, seeds, strict=True):
        hydra = {"task": {"name": "joystick"}, "run_name": None, "seed": seed}
        (jobs.stub / f"run_{name}.json").write_text(json.dumps({"hydra_config": hydra}))
    return jobs.run("train.sh", ROBOT="roboto_origin", STUB_WRITE_RUN=" ".join(names), **env)


def test_of_two_fresh_run_jsons_the_one_with_this_config_is_measured(jobs):
    res = _fresh_runs(jobs, seeds=(1, 0))
    assert res.code == 0, res.out
    inline = res.calls[2]["args"]
    assert inline[0] == "-" and re.fullmatch(r"runs/\.train_started\.\w+\.cfg", inline[1])
    assert sorted(inline[2:]) == ["runs/joystick_a", "runs/joystick_b"]
    assert {run for _, run in _evals(res.calls)} == {"joystick_b"}
    assert not list((jobs.root / "runs").glob(".train_started*"))


@pytest.mark.parametrize(
    "env, code", [({}, 75), ({"STUB_FAIL": "train=3"}, 3)], ids=["training-ok", "training-failed"],
)
def test_two_fresh_run_jsons_with_this_config_are_not_measured_and_exit_75_or_the_training_code(jobs, env, code):
    res = _fresh_runs(jobs, seeds=(0, 0), **env)
    assert res.code == code, res.out
    assert _evals(res.calls) == []
    assert "none or several match this config" in res.out


def test_a_failed_unnamed_training_does_not_take_another_jobs_lone_run_json(jobs):
    """A training that died before train.py wrote its run.json leaves none
    of its own, and the only fresh run.json can be another job's, at
    another config."""
    res = _fresh_runs(jobs, seeds=(1,), STUB_FAIL="train=3")
    assert res.code == 3, res.out
    assert _evals(res.calls) == []
    assert "training rc=3; of the fresh run.json in runs/joystick_a, none or several match this config" in res.out


def test_a_failed_unnamed_training_that_wrote_its_own_run_json_is_measured(jobs):
    """A crash after train.py wrote run.json: the lone fresh run.json carries
    this config."""
    res = _fresh_runs(jobs, seeds=(0,), STUB_FAIL="train=3")
    assert res.code == 3, res.out
    assert _evals(res.calls) == [("courses", "joystick_a"), ("battery", "joystick_a"), ("report", "joystick_a")]


@pytest.mark.parametrize("unnamed", ["null", "absent"])
def test_no_run_name_and_no_fresh_run_json_exits_with_the_training_code(jobs, unnamed):
    """A resolve whose run_name is null, or that prints no run_name line at
    all, leaves the run unnamed whatever RUN_NAME says."""
    env = {}
    if unnamed == "absent":
        (jobs.stub / "resolved").write_text("")
        env["RUN_NAME"] = "r"
    res = jobs.run("train.sh", ROBOT="roboto_origin", **env)
    assert (res.code, len(res.calls)) == (0, 2)
    assert "nothing to measure" in res.out


@pytest.mark.parametrize("name", ["all", "sweep/s0", "a b", ".."])
@pytest.mark.parametrize("eval_", ["true", "false"])
def test_a_run_name_eval_runs_cannot_address_stops_before_the_training(jobs, name, eval_):
    """jobs/eval_runs.sh takes all as every run and refuses a nested name,
    and its RUNS splits on whitespace. The caller's own pass needs the same
    name, so EVAL=false refuses it too."""
    res = jobs.run("train.sh", ROBOT="roboto_origin", STUB_RUN_NAME=name, EVAL=eval_)
    assert (res.code, len(res.calls)) == (1, 1), res.out
    assert f"run_name '{name}' must be one dir name under runs/" in res.out
    assert not (jobs.root / "runs").exists()


def test_eval_false_runs_the_training_only(jobs):
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", EVAL="false")
    assert (res.code, len(res.calls)) == (0, 2)
    assert sorted(p.name for p in (jobs.root / "runs").iterdir()) == ["r"]


def test_an_eval_failure_after_a_good_training_exits_75(jobs):
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", STUB_FAIL="courses=1")
    assert res.code == 75
    assert _evals(res.calls) == [("courses", "r"), ("battery", "r"), ("report", "r")]
    assert "RUNS=r ./jobs/eval_runs.sh completes them" in res.out


def test_an_expired_eval_timeout_exits_75(jobs):
    res = jobs.run(
        "train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", EVAL_TIMEOUT="60",
        STUB_TIMEOUT_RC="124",
    )
    assert res.code == 75
    assert res.timeouts == ["--foreground 60 bash jobs/eval_runs.sh"]
    assert "rc=124" in res.out


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGKILL], ids=["TERM", "KILL"])
def test_a_kill_to_the_payloads_group_during_the_stage_leaves_no_eval_running(jobs, sig):
    """A deadline stop signals the payload's process group. The stub timeout
    leaves that group unless given --foreground, as GNU timeout does, and an
    eval outside the group would outlive the kill."""
    proc = jobs.start("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", STUB_HANG="courses")
    pid = _hung_pid(jobs, proc)
    os.killpg(proc.pid, sig)
    proc.wait(timeout=30)
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _alive(pid)


def test_without_timeout_on_path_the_stage_runs_unbounded_and_says_so(jobs, tmp_path):
    """A PATH holding every system tool but timeout."""
    bare = tmp_path / "bare"
    bare.mkdir()
    for bin_dir in ("/usr/bin", "/bin"):
        for tool in os.listdir(bin_dir):
            if tool not in ("timeout", "gtimeout") and not (bare / tool).exists():
                (bare / tool).symlink_to(Path(bin_dir) / tool)
    res = jobs.run(
        "train.sh", path=os.pathsep.join([str(jobs.bin), str(bare)]),
        ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@",
    )
    assert res.code == 0, res.out
    assert "NOTE: no timeout on PATH; the eval stage runs unbounded" in res.out
    assert len(_evals(res.calls)) == 3


def test_eval_timeout_0_runs_the_stage_without_a_bound(jobs):
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", EVAL_TIMEOUT="0")
    assert res.code == 0
    assert res.timeouts == []
    assert len(_evals(res.calls)) == 3


@pytest.mark.parametrize(
    "env", [{"EVAL": "yes"}, {"EVAL_TIMEOUT": "30m"}, {"EVAL_WORKERS": "0"}, {"EVAL_WORKERS": "auto"}],
)
def test_a_bad_eval_parameter_stops_before_anything_runs(jobs, env):
    res = jobs.run("train.sh", ROBOT="roboto_origin", **env)
    assert (res.code, res.calls) == (1, [])


def test_a_config_that_fails_to_compose_stops_before_anything_runs(jobs):
    res = jobs.run("train.sh", ROBOT="roboto_origin", STUB_FAIL="resolve=1")
    assert (res.code, len(res.calls)) == (1, 1)
    assert not (jobs.root / "runs").exists()


def test_resolve_and_train_get_identical_overrides(jobs):
    res = jobs.run(
        "train.sh", ROBOT="roboto_origin", EXPERIMENT="yolo_v4", RUN_NAME="r", SEED="2",
        RUN_ARGS="++ppo.num_timesteps=3e8", STUB_WRITE_RUN="@",
    )
    resolve, train = res.calls[:2]
    assert resolve["args"][2:5] == ["--cfg", "job", "--resolve"]
    assert _train_overrides(resolve) == _train_overrides(train)
    cfg = _compose(_train_overrides(train))
    assert (cfg.run_name, cfg.seed, cfg.ppo.num_timesteps) == ("r", 2, 3e8)


def test_exported_eval_parameters_cannot_redirect_the_stage(jobs):
    """The stage pins the canonical measurement whatever the caller exported
    for another purpose."""
    res = jobs.run(
        "train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", RUNS="other",
        TAG="x", SEEDS="4", SEED_BASE="8", EVALS="courses", REDO="false", CHECK="true",
        WORKERS="auto", MUJOCO_GL="egl",
    )
    assert res.code == 0, res.out
    assert _evals(res.calls) == [("courses", "r"), ("battery", "r"), ("report", "r")]
    args = _courses_args(res.calls[2])
    assert (args.seeds, args.seed_base, args.out, args.skip_if_current, args.check) == (8, 0, None, False, False)
    assert args.workers is None
    assert [c["mujoco_gl"] for c in res.calls[2:]] == ["disabled"] * 3
    assert not (jobs.root / "runs/r/eval").exists()


def test_eval_workers_sets_the_stages_course_lanes(jobs):
    res = jobs.run("train.sh", ROBOT="roboto_origin", RUN_NAME="r", STUB_WRITE_RUN="@", EVAL_WORKERS="3")
    assert res.code == 0, res.out
    assert _courses_args(res.calls[2]).workers == 3


# -- train_chain.sh -----------------------------------------------------------------

CHAIN = {
    "ROBOT": "roboto_origin", "RUN_NAME": "c", "PHASE_A_EXPERIMENT": "yolo_v4",
    "PHASE_B_EXPERIMENT": "yolo_chain_sym_b", "STUB_WRITE_RUN": "@",
}


def test_chain_reports_phase_a_evals_missing_after_phase_b_succeeds(jobs):
    res = jobs.run("train_chain.sh", STUB_FAIL="courses:c_a=1", **CHAIN)
    assert res.code == 75, res.out
    assert [run for what, run in _evals(res.calls) if what == "courses"] == ["c_a", "c"]
    assert "phase A (c_a) trained but is not measured. RUNS=c_a ./jobs/eval_runs.sh" in res.out


def test_chain_exits_with_phase_b_code_over_phase_a_missing_evals(jobs):
    res = jobs.run("train_chain.sh", STUB_FAIL="courses:c_a=1 train:c=3", **CHAIN)
    assert res.code == 3


def test_chain_phase_a_75_after_an_early_stop_exits_143(jobs):
    (jobs.stub / "run_c_a.json").write_text(json.dumps({"early_stopped": True, "stopped_at_steps": 500}))
    res = jobs.run("train_chain.sh", STUB_FAIL="courses:c_a=1", **CHAIN)
    assert res.code == 143, res.out
    assert [run for what, run in _evals(res.calls) if what == "courses"] == ["c_a"]


def test_chain_phase_a_75_without_a_fresh_run_json_exits_1(jobs):
    """train.sh exits 75 when a training that exited 0 left no fresh
    run.json. The chain reads that as an exit 0 without a run.json."""
    res = jobs.run("train_chain.sh", **{**CHAIN, "STUB_WRITE_RUN": ""})
    assert res.code == 1, res.out
    assert "phase A (c_a) exited rc=75 without writing runs/c_a/run.json" in res.out


@pytest.mark.parametrize(
    ("record", "rc"),
    [
        # Killed by a signal: train.py installs no handler, so the record
        # stays as written before training.
        ({"status": "running", "early_stopped": None, "stopped_at_steps": None}, 137),
        # Raised (an OOM): train.py records the failure with the last eval's
        # step count, which alone would read as a finished budget.
        (
            {"status": "failed", "early_stopped": None, "stopped_at_steps": 1000,
             "error": "XlaRuntimeError('RESOURCE_EXHAUSTED')"},
            1,
        ),
    ],
)
def test_chain_phase_a_that_did_not_return_does_not_start_phase_b(jobs, record, rc):
    """train.py writes run.json before training and fills early_stopped when
    training returns. A phase A that never returned leaves it null. Its
    stage measures it, and phase B does not start."""
    (jobs.stub / "run_c_a.json").write_text(json.dumps(record))
    res = jobs.run("train_chain.sh", STUB_FAIL=f"train:c_a={rc}", **CHAIN)
    assert res.code == rc, res.out
    assert (
        f"phase A (c_a) exited rc={rc} and runs/c_a/run.json records a training that did not return"
        in res.out
    )
    assert f"status={record['status']}" in res.out
    assert "continuing to phase B" not in res.out
    trained = [
        a for c in res.calls if _module(c) == "humanoid_lab.train" and "--cfg" not in c["args"]
        for a in c["args"] if a.startswith("run_name=")
    ]
    assert trained == ["run_name=c_a"]
    assert [run for what, run in _evals(res.calls) if what == "courses"] == ["c_a"]


def test_chain_leaves_the_task_to_each_experiment_unless_task_is_set(jobs):
    """A group passed on the command line overrides an experiment's own
    choice, so the chain passes task= only when TASK is set."""
    res = jobs.run("train_chain.sh", **CHAIN)
    assert res.code == 0, res.out
    trains = [c for c in res.calls if _module(c) == "humanoid_lab.train"]
    assert trains
    assert not [a for c in trains for a in c["args"] if a.startswith("task=")]
    res = jobs.run("train_chain.sh", TASK="joystick", **CHAIN)
    assert res.code == 0, res.out
    trains = [c for c in res.calls if _module(c) == "humanoid_lab.train"]
    assert all("task=joystick" in c["args"] for c in trains)


def test_chain_measures_both_phases(jobs):
    """Each phase's stage takes the caller's EVAL_TIMEOUT and EVAL_WORKERS."""
    res = jobs.run("train_chain.sh", EVAL_TIMEOUT="60", EVAL_WORKERS="3", **CHAIN)
    assert res.code == 0, res.out
    assert [run for what, run in _evals(res.calls) if what == "courses"] == ["c_a", "c"]
    assert res.timeouts == ["--foreground 60 bash jobs/eval_runs.sh"] * 2
    courses = [_courses_args(c).workers for c in res.calls if _module(c) == "humanoid_lab.eval.courses"]
    assert courses == [3, 3]


def test_chain_eval_false_turns_the_stage_off_for_both_phases(jobs):
    """A caller who measures on a CPU host turns the stage off for both
    phases."""
    res = jobs.run("train_chain.sh", EVAL="false", **CHAIN)
    assert res.code == 0, res.out
    assert _evals(res.calls) == []
    assert res.timeouts == []


@pytest.mark.parametrize("name", ["all", "sweep/s0"])
def test_chain_run_name_eval_runs_cannot_address_stops_before_phase_a(jobs, name):
    """Phase A's all_a would pass jobs/train.sh's check, and phase B's all
    would fail it only after phase A's budget."""
    res = jobs.run("train_chain.sh", **{**CHAIN, "RUN_NAME": name})
    assert (res.code, res.calls) == (1, []), res.out
    assert not (jobs.root / "runs").exists()


# -- eval_runs.sh ---------------------------------------------------------------------


def test_a_named_run_without_run_json_is_skipped_and_the_others_measured(jobs):
    _make_run(jobs.root, "a")
    _make_run(jobs.root, "b", run_json=False)
    res = jobs.run("eval_runs.sh", RUNS="a b runs/c/")
    assert res.code == 1
    assert "SKIPPED: runs/b has no run.json" in res.out
    assert "SKIPPED: runs/c has no run.json" in res.out
    assert {run for _, run in _evals(res.calls)} == {"a"}
    assert "measured [a], skipped [b c], failed []" in res.out


def test_a_run_whose_newest_step_dir_is_incomplete_is_skipped(jobs):
    """An older complete step does not stand in for an incomplete newest."""
    _make_run(jobs.root, "a", steps=(OLDER, STEP), complete=False)
    _make_run(jobs.root, "b", steps=())
    res = jobs.run("eval_runs.sh", RUNS="a b")
    assert (res.code, res.calls) == (1, [])
    assert "SKIPPED: runs/a has no complete newest checkpoint" in res.out
    assert "SKIPPED: runs/b has no complete newest checkpoint" in res.out


def test_run_status_does_not_gate_the_measurement(jobs):
    """train.py writes run.json with status running before training, so a
    killed run's says running. The run is measured on its newest checkpoint
    whatever the status says."""
    _make_run(jobs.root, "a", steps=(OLDER, STEP), run={"status": "running"})
    res = jobs.run("eval_runs.sh", RUNS="a")
    assert res.code == 0, res.out
    assert _evals(res.calls) == [("courses", "a"), ("battery", "a"), ("report", "a")]
    assert json.loads((jobs.root / "runs/a/battery.json").read_text())["checkpoint"] == STEP


def test_runs_all_takes_every_run_dir_with_a_run_json_in_order(jobs):
    for name in ("b", "a"):
        _make_run(jobs.root, name)
    _make_run(jobs.root, "no_json", run_json=False)
    (jobs.root / "runs/.train_started.abc").write_text("")
    res = jobs.run("eval_runs.sh", RUNS="all", EVALS="courses")
    assert res.code == 0, res.out
    assert _evals(res.calls) == [("courses", "a"), ("courses", "b")]


def test_tag_routes_the_course_output_and_keeps_the_runs_own_files(jobs):
    _make_run(jobs.root, "a")
    res = jobs.run("eval_runs.sh", RUNS="a", TAG="seeds8", EVALS="courses", SEED_BASE="8", SEEDS="4")
    assert res.code == 0, res.out
    (call,) = res.calls
    args = _courses_args(call)
    assert (str(args.out), args.seeds, args.seed_base) == ("runs/a/eval/seeds8/courses.json", 4, 8)
    assert (jobs.root / "runs/a/eval/seeds8/courses.json").exists()
    assert not (jobs.root / "runs/a/courses.json").exists()


@pytest.mark.parametrize(
    "env",
    [
        {"TAG": "x"},
        {"TAG": "x", "EVALS": "courses report"},
        {"TAG": "x", "EVALS": "battery"},
        {"SEED_BASE": "8"},
        {"SEEDS": "4"},
        {"EVALS": "courses video"},
        {"REDO": "yes"},
        {"CHECK": "1"},
        {"WORKERS": "0"},
        {"TAG": "a/b", "EVALS": "courses"},
        {"TAG": "..", "EVALS": "courses"},
        {"RUNS": "../a"},
        {"RUNS": "a/b"},
        {"RUNS": "all a"},
        {"RUNS": ""},
    ],
)
def test_a_bad_parameter_exits_1_before_any_call(jobs, env):
    _make_run(jobs.root, "a")
    res = jobs.run("eval_runs.sh", **{"RUNS": "a", **env})
    assert (res.code, res.calls) == (1, [])
    assert "ERROR" in res.out or "RUNS" in res.out


def test_redo_false_skips_current_courses_through_the_cli(jobs):
    _make_run(jobs.root, "a")
    res = jobs.run("eval_runs.sh", RUNS="a", EVALS="courses", WORKERS="3")
    assert _courses_args(res.calls[0]).skip_if_current
    assert _courses_args(res.calls[0]).workers == 3
    res = jobs.run("eval_runs.sh", RUNS="a", EVALS="courses", REDO="true")
    assert not _courses_args(res.calls[0]).skip_if_current


@pytest.mark.parametrize(
    "battery, older, redo, runs",
    [
        (STEP, False, "false", False),
        (STEP, False, "true", True),
        (OLDER, False, "false", True),
        (STEP, True, "false", True),
        (None, False, "false", True),
    ],
    ids=["current", "redo", "older-step", "older-file", "missing"],
)
def test_the_battery_is_skipped_only_when_it_is_for_the_newest_checkpoint(jobs, battery, older, redo, runs):
    """Current: battery.json names the newest step dir and is not older than
    its ppo_network_config.json, which catches a reused run dir."""
    run_dir = _make_run(jobs.root, "a", steps=(OLDER, STEP), battery=battery)
    if older:
        _age(run_dir / "battery.json")
    res = jobs.run("eval_runs.sh", RUNS="a", EVALS="battery", REDO=redo)
    assert res.code == 0, res.out
    assert (_evals(res.calls) == [("battery", "a")]) is runs


def test_one_runs_failure_does_not_stop_the_next(jobs):
    for name in ("a", "b"):
        _make_run(jobs.root, name)
    res = jobs.run("eval_runs.sh", RUNS="a b", STUB_FAIL="courses:a=1")
    assert res.code == 1
    assert _evals(res.calls) == [
        ("courses", "a"), ("battery", "a"), ("report", "a"),
        ("courses", "b"), ("battery", "b"), ("report", "b"),
    ]
    assert "WARN: courses for a failed rc=1; continuing" in res.out
    assert "measured [b], skipped [], failed [a:courses]" in res.out


def test_check_runs_no_eval_and_exits_0_when_everything_is_current(jobs):
    for name in ("a", "b"):
        _make_run(jobs.root, name, battery=STEP)
    res = jobs.run("eval_runs.sh", RUNS="all", CHECK="true")
    assert res.code == 0, res.out
    assert [c["args"][-1] for c in res.calls] == ["--check", "--check"]
    assert [str(_courses_args(c).run) for c in res.calls] == ["runs/a", "runs/b"]
    assert all(c["jax_platforms"] == "cpu" for c in res.calls)
    assert "CURRENT: runs/a" in res.out and "CURRENT: runs/b" in res.out


def test_check_lists_every_gap_and_exits_1(jobs):
    _make_run(jobs.root, "a", battery=STEP)
    _make_run(jobs.root, "b", battery=STEP)
    _make_run(jobs.root, "c", battery=OLDER)
    _make_run(jobs.root, "d", battery=STEP)
    _make_run(jobs.root, "orphan", run_json=False)
    _make_run(jobs.root, "partial", run_json=False, complete=False)
    res = jobs.run("eval_runs.sh", RUNS="all", CHECK="true", STUB_FAIL="check:b=3 check:d=2")
    assert res.code == 1
    assert _evals(res.calls) == [("courses", "a"), ("courses", "b"), ("courses", "c"), ("courses", "d")]
    assert "STALE: runs/b: not current: runs/b/courses.json: stub reason" in res.out
    assert f"STALE: runs/c: runs/c/battery.json is for checkpoint '{OLDER}', not {STEP}" in res.out
    # The CLI's 2 is a refusal, not a verdict: the line names the code, not
    # the CLI's last line.
    assert "STALE: runs/d: courses --check failed rc=2" in res.out
    assert "runs/d/courses.json: stub reason" not in res.out
    assert "UNMEASURABLE: runs/orphan has checkpoints but no run.json" in res.out
    assert "partial" not in res.out
    assert "current [a], stale [b c d], skipped [], unmeasurable [orphan]" in res.out


def test_an_unmeasurable_dir_alone_makes_check_exit_1(jobs):
    _make_run(jobs.root, "a", battery=STEP)
    _make_run(jobs.root, "orphan", run_json=False)
    _make_run(jobs.root, "partial", run_json=False, complete=False)
    res = jobs.run("eval_runs.sh", RUNS="all", CHECK="true")
    assert res.code == 1, res.out
    assert [str(_courses_args(c).run) for c in res.calls] == ["runs/a"]
    assert "current [a], stale [], skipped [], unmeasurable [orphan]" in res.out


def test_check_of_named_runs_lists_only_those_runs(jobs):
    _make_run(jobs.root, "a", battery=STEP)
    _make_run(jobs.root, "orphan", run_json=False)
    res = jobs.run("eval_runs.sh", RUNS="a", CHECK="true")
    assert res.code == 0, res.out
    res = jobs.run("eval_runs.sh", RUNS="orphan", CHECK="true")
    assert (res.code, res.calls) == (1, [])
    assert "SKIPPED: runs/orphan has no run.json" in res.out


def test_check_with_a_tag_audits_the_tagged_file(jobs):
    _make_run(jobs.root, "a")
    res = jobs.run("eval_runs.sh", RUNS="a", CHECK="true", TAG="seeds8", EVALS="courses", SEED_BASE="8")
    assert res.code == 0, res.out
    (call,) = res.calls
    args = _courses_args(call)
    assert (args.check, str(args.out), args.seed_base) == (True, "runs/a/eval/seeds8/courses.json", 8)


@pytest.mark.parametrize(
    "sig, code", [(signal.SIGTERM, 143), (signal.SIGINT, 130), (signal.SIGHUP, 129)], ids=["TERM", "INT", "HUP"],
)
def test_a_signal_ends_the_running_eval_before_eval_runs_exits(jobs, sig, code):
    """An expired timeout --foreground signals eval_runs.sh alone. Its eval
    is gone by the time it exits. bash starts that eval with SIGINT ignored,
    so an INT reaches it only as the script's TERM. The eval takes 0.2 s to
    exit, so it is gone at once only because the script waits for it."""
    _make_run(jobs.root, "a")
    proc = jobs.start("eval_runs.sh", RUNS="a", STUB_HANG="courses")
    pid = _hung_pid(jobs, proc)
    proc.send_signal(sig)
    assert proc.wait(timeout=30) == code
    assert not _alive(pid)


def test_every_eval_runs_on_cpu_and_the_payloads_never_export_it(jobs):
    """JAX_PLATFORMS=cpu goes on each eval call, so a caller's own value and
    the training call are untouched."""
    _make_run(jobs.root, "a")
    res = jobs.run("eval_runs.sh", RUNS="a", JAX_PLATFORMS="cuda")
    assert res.code == 0, res.out
    assert [(c["jax_platforms"], c["mujoco_gl"]) for c in res.calls] == [("cpu", "disabled")] * 3
    res = jobs.run("eval_runs.sh", RUNS="a", MUJOCO_GL="egl", EVALS="report")
    assert res.calls[0]["mujoco_gl"] == "egl"
    for path in PAYLOADS:
        assert not re.search(r"export\s+JAX_PLATFORMS", _code(path)), path.name


def test_eval_runs_writes_only_under_runs(jobs):
    for name in ("a", "b"):
        _make_run(jobs.root, name)
    before = _files(jobs.root)
    jobs.run("eval_runs.sh", RUNS="a b")
    jobs.run("eval_runs.sh", RUNS="a", TAG="t", EVALS="courses")
    jobs.run("eval_runs.sh", RUNS="all", CHECK="true")
    new = _files(jobs.root) - before
    assert new == {
        "runs/a/courses.json", "runs/a/battery.json", "runs/a/eval_report.md",
        "runs/b/courses.json", "runs/b/battery.json", "runs/b/eval_report.md",
        "runs/a/eval/t/courses.json",
    }
