"""The jobs/ payloads against their contract (jobs/README.md).

Each payload parses, names no remote tool, documents every variable it
reads, and writes only under runs/. A stub `python3` first on PATH records
each command a payload runs and the allocator setting it ran with, and
exits with a scripted code. Nothing trains, builds or checks. An inline
`python3 -` script is the payload's own logic and runs on the real
interpreter. The recorded Hydra overrides are then composed, so a payload
that passes an override Hydra refuses fails here.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from humanoid_lab import check_terrain, paths, registry
from humanoid_lab.envs.terrain_joystick import JAX_BOX_LIMIT
from humanoid_lab.eval import terrain_scan
from humanoid_lab.eval.courses import runner
from humanoid_lab.terrain import scene

JOBS_DIR = paths.REPO_ROOT / "jobs"
PAYLOADS = sorted(JOBS_DIR.glob("*.sh"))
BASH = shutil.which("bash")
# The one step dir the stub's training writes, as brax names it.
STUB_STEP = "000000000001"

# Records one call per block: the allocator variable, then each argument.
# STUB_RCS lists exit codes, one per call in order. Calls past the list
# exit 0. With STUB_WRITE_RUN set, a training resolve prints the run_name
# its overrides set (else null), and a training writes runs/<run_name>/
# with a full-budget run.json and one complete checkpoint. Without a
# run_name it writes runs/stub_run/.
STUB_PYTHON = """#!/bin/sh
n=$(cat "$STUB_DIR/count" 2>/dev/null || echo 0)
echo $((n + 1)) > "$STUB_DIR/count"
{
  printf 'PREALLOCATE=%s\\n' "${XLA_PYTHON_CLIENT_PREALLOCATE-<unset>}"
  for a in "$@"; do printf 'ARG=%s\\n' "$a"; done
  printf 'END\\n'
} >> "$STUB_DIR/calls"
if [ "${1-}" = "-" ]; then exec "$STUB_REAL_PYTHON" "$@"; fi
if [ -n "${STUB_WRITE_RUN-}" ] && [ "${1-}" = "-m" ] && [ "${2-}" = "humanoid_lab.train" ]; then
  name=null
  for a in "$@"; do
    case "$a" in run_name=*) name="${a#run_name=}" ;; esac
  done
  if [ "${3-}" = "--cfg" ]; then
    printf 'run_name: %s\\n' "$name"
  else
    [ "$name" = null ] && name=stub_run
    mkdir -p "runs/$name/checkpoints/$STUB_STEP"
    printf '{}\\n' > "runs/$name/checkpoints/$STUB_STEP/ppo_network_config.json"
    printf '{"run_name": "%s", "num_timesteps": 1, "early_stopped": false, "stopped_at_steps": 1}\\n' \\
      "$name" > "runs/$name/run.json"
  fi
fi
set -- ${STUB_RCS:-}
i=0
for rc in "$@"; do
  if [ "$i" -eq "$n" ]; then exit "$rc"; fi
  i=$((i + 1))
done
exit 0
"""


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def workdir(tmp_path):
    """A repo-root stand-in: pyproject.toml, the real configs/ and jobs/, and
    a bin/ holding the python3 stub and an nvidia-smi that samples nothing.
    train.sh and train_chain.sh call jobs/ scripts by their relative path."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "pyproject.toml").write_text("")
    (root / "configs").symlink_to(paths.CONFIGS_DIR)
    (root / "jobs").symlink_to(JOBS_DIR)
    stub = tmp_path / "stub"
    stub.mkdir()
    _write_exe(stub / "python3", STUB_PYTHON)
    _write_exe(stub / "nvidia-smi", "#!/bin/sh\nexit 0\n")
    return root, stub


def _run(workdir, payload: str, **env) -> tuple[int, list[dict]]:
    """(exit code, recorded calls) of `payload` run from the stand-in root."""
    root, stub = workdir
    for name in ("count", "calls"):
        (stub / name).unlink(missing_ok=True)
    full = {
        "PATH": f"{stub}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "HOME": str(root),
        "STUB_DIR": str(stub),
        "STUB_REAL_PYTHON": sys.executable,
        "STUB_STEP": STUB_STEP,
        **env,
    }
    done = subprocess.run(
        [BASH, str(JOBS_DIR / payload)],
        cwd=root, env=full, capture_output=True, text=True, check=False, timeout=60,
    )
    calls = []
    log = stub / "calls"
    if log.exists():
        call = {"args": []}
        for line in log.read_text().splitlines():
            if line == "END":
                calls.append(call)
                call = {"args": []}
            elif line.startswith("PREALLOCATE="):
                call["preallocate"] = line.split("=", 1)[1]
            else:
                call["args"].append(line.split("=", 1)[1])
    return done.returncode, calls


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        return compose(config_name="config", overrides=overrides)


def _train_overrides(call) -> list[str]:
    """The Hydra overrides of a recorded `python3 -m humanoid_lab.train`."""
    args = call["args"]
    assert args[:2] == ["-m", "humanoid_lab.train"]
    rest = args[2:]
    if rest[:3] == ["--cfg", "job", "--resolve"]:
        rest = rest[3:]
    return rest


def _code(path: Path) -> str:
    """The payload with its comments dropped."""
    return "\n".join(line.split("#")[0] for line in path.read_text().splitlines())


# -- text ------------------------------------------------------------------------


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_every_payload_parses(path):
    done = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_no_payload_uses_a_remote_tool_or_a_scheduler_directive(path):
    """Moving files and reaching hosts belong to the caller. A directive line
    such as `#XYZ --flag` is a scheduler header, whatever its scheduler."""
    text = path.read_text()
    code = _code(path)
    for tool in ("ssh", "scp", "rsync"):
        assert not re.search(rf"\b{tool}\b", code), f"{path.name} runs {tool}"
    assert not re.search(r"^#[A-Z$]{1,8}\s+-", text, re.MULTILINE), f"{path.name} has a directive"


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_every_variable_a_payload_reads_is_in_its_header(path):
    """The contract: the header documents each parameter and its default.
    A parameter is every uppercase variable the code reads and does not set
    itself: a defaulted one, a required one, and any read inline, nested in
    a default or inside arithmetic. A scheduler variable cannot slip in
    undocumented."""
    text = path.read_text()
    header = "\n".join(line for line in text.splitlines() if line.startswith("#"))
    code = _code(path)
    params = set(re.findall(r'^\s*([A-Z][A-Z0-9_]*)="\$\{\1[:?-]', code, re.MULTILINE))
    params |= set(re.findall(r'^: "\$\{([A-Z_][A-Z0-9_]*):\?', code, re.MULTILINE))
    assert params
    own = set(re.findall(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=", code, re.MULTILINE)) - params
    used = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]*)", code))
    for body in re.findall(r"\$\(\((.*?)\)\)", code):
        used |= set(re.findall(r"\b([A-Z][A-Z0-9_]*)\b", body))
    read = params | (used - own - {"PWD", "SECONDS"})
    for name in sorted(read):
        documented = re.search(rf"^#\s+{name}\s", header, re.MULTILINE)
        assert documented, f"{path.name}: {name} undocumented"


def _examples(path: Path) -> list[dict[str, str]]:
    """The environment of each worked example in a payload's header. An
    example is a header line that ends in ./jobs/<payload>, its continuation
    lines joined. `N` stands for a measured number and becomes 64."""
    header = "\n".join(line[1:] for line in path.read_text().splitlines() if line.startswith("#"))
    header = re.sub(r"\\\n", " ", header)
    examples = []
    for line in header.splitlines():
        if not line.rstrip().endswith(f"./jobs/{path.name}"):
            continue
        pairs = [word.split("=", 1) for word in shlex.split(line)[:-1]]
        assert all(len(pair) == 2 for pair in pairs), line
        examples.append({name: re.sub(r"\bN\b", "64", value) for name, value in pairs})
    return examples


def _check_call(call) -> None:
    """Fails unless the called module accepts the recorded flags and Hydra
    composes the recorded overrides. An inline script ran for real."""
    if call["args"][:1] == ["-"]:
        return
    module, rest = call["args"][1], call["args"][2:]
    if module == "humanoid_lab.train":
        _compose(_train_overrides(call))
    elif module == "humanoid_lab.check_terrain":
        _, overrides = check_terrain.parser().parse_known_args(rest)
        assert not [a for a in overrides if a.startswith("-")], overrides
        check_terrain.compose(overrides)
    elif module == "humanoid_lab.eval.terrain_scan":
        terrain_scan.parser().parse_args(rest)
    elif module == "humanoid_lab.eval.courses":
        runner.parser().parse_args(rest)
    elif module in ("humanoid_lab.eval.battery", "humanoid_lab.eval.report"):
        assert rest[:1] == ["--run"] and len(rest) == 2, rest
    else:
        raise AssertionError(f"unexpected module {module}")


def _make_run(root: Path, name: str) -> None:
    """runs/<name>, an earlier training's run: a run.json an hour old, so no
    training's eval stage takes it for its own, one complete checkpoint and
    a battery.json for it."""
    run_dir = root / "runs" / name
    step = run_dir / "checkpoints" / STUB_STEP
    step.mkdir(parents=True)
    (step / "ppo_network_config.json").write_text("{}")
    (run_dir / "run.json").write_text(json.dumps({"run_name": name}))
    hour_ago = time.time() - 3600
    os.utime(run_dir / "run.json", (hour_ago, hour_ago))
    battery = {"run": name, "checkpoint": STUB_STEP}
    (run_dir / "battery.json").write_text(json.dumps(battery, indent=2))


@pytest.mark.parametrize("path", PAYLOADS, ids=[p.name for p in PAYLOADS])
def test_every_header_example_runs(workdir, path):
    """jobs/README.md promises a worked example in each header. Each one
    runs, and each command it runs is accepted. The examples that measure
    runs name walk_a and walk_b, and a training leaves its run behind."""
    for name in ("walk_a", "walk_b"):
        _make_run(workdir[0], name)
    examples = _examples(path)
    assert examples, f"{path.name} has no worked example"
    for env in examples:
        code, calls = _run(workdir, path.name, STUB_WRITE_RUN="1", **env)
        assert code == 0, env
        assert calls, env
        for call in calls:
            _check_call(call)


def test_payloads_pass_experiment_without_a_plus():
    """config.yaml's defaults list holds `experiment: null`, so
    `+experiment=` fails to compose."""
    for path in PAYLOADS:
        assert "+experiment=" not in _code(path), path.name
    with pytest.raises(Exception, match="experiment"):
        _compose(["+experiment=roboto_terrain_v1"])


# -- group variables and the compose guard ---------------------------------------


@pytest.mark.parametrize(
    "payload, extra, task",
    [
        ("train.sh", {"TASK": "sizing"}, "sizing"),
        ("preflight_sizing.sh", {"TASK": "sizing", "SIZES_LIST": "16"}, "sizing"),
        # check_terrain.sh reads no TASK. check-terrain gates task=terrain.
        ("check_terrain.sh", {"ARENAS": "train"}, "terrain"),
    ],
)
def test_a_set_group_variable_wins_over_the_experiment_pin(workdir, payload, extra, task):
    """Every documented terrain launch sets EXPERIMENT and ACTUATORS together."""
    code, calls = _run(
        workdir, payload, EXPERIMENT="roboto_terrain_v1", ROBOT="asimov_v1", ACTUATORS="deploy_pd", **extra
    )
    assert code == 0
    if payload == "check_terrain.sh":
        cfg = check_terrain.compose(check_terrain.parser().parse_known_args(calls[-1]["args"][2:])[1])
    else:
        cfg = _compose(_train_overrides(calls[-1]))
    names = (cfg["task"]["name"], cfg["robot"]["name"], cfg["actuators"]["name"])
    assert names == (task, "asimov_v1", "deploy_pd")


@pytest.mark.parametrize("payload", ["train.sh", "preflight_sizing.sh"])
def test_a_config_that_fails_to_compose_stops_before_anything_runs(workdir, payload):
    """A typo fails in seconds. A sweep would otherwise report it as an OOM
    ceiling at every size."""
    root, _ = workdir
    code, calls = _run(workdir, payload, ROBOT="roboto_origin", STUB_RCS="1")
    assert (code, len(calls)) == (1, 1)
    assert calls[0]["args"][2:5] == ["--cfg", "job", "--resolve"]
    assert not (root / "runs").exists()


# -- train.sh --------------------------------------------------------------------


def test_train_lets_the_experiment_pick_robot_task_and_actuators(workdir):
    """Unset group variables reach Hydra as nothing, so the experiment's own
    pins hold. A CLI `task=` would win over them."""
    code, calls = _run(workdir, "train.sh", EXPERIMENT="roboto_terrain_v1")
    assert code == 0
    assert len(calls) == 2
    resolve, train = calls
    assert resolve["args"][2:5] == ["--cfg", "job", "--resolve"]
    overrides = _train_overrides(train)
    assert _train_overrides(resolve) == overrides
    assert "experiment=roboto_terrain_v1" in overrides
    assert not [a for a in overrides if a.split("=")[0] in ("robot", "task", "actuators")]
    cfg = _compose(overrides)
    assert (cfg.task.name, cfg.robot.name, cfg.actuators.name) == (
        "terrain", "roboto_origin", "sizing_ideal"
    )
    assert cfg.ppo.num_envs == 32768


def test_train_passes_the_group_variables_that_are_set(workdir):
    code, calls = _run(
        workdir, "train.sh", ROBOT="asimov_v1", TASK="sizing", ACTUATORS="deploy_pd", WANDB="false"
    )
    assert code == 0
    cfg = _compose(_train_overrides(calls[-1]))
    names = (cfg.task.name, cfg.robot.name, cfg.actuators.name)
    assert names == ("sizing", "asimov_v1", "deploy_pd")
    assert cfg.wandb.enable is False


def test_train_passes_the_sizes_seed_and_run_name(workdir):
    """The header's terrain launch, with a seed and a run name. Every value
    differs from its default, so a dropped variable shows."""
    code, calls = _run(
        workdir, "train.sh", EXPERIMENT="roboto_terrain_v1", ACTUATORS="deploy_pd", NUM_ENVS="8192",
        BATCH="256", SEED="1", RUN_NAME="r",
    )
    assert (code, len(calls)) == (0, 2)
    resolve, train = calls
    overrides = _train_overrides(train)
    assert _train_overrides(resolve) == overrides
    cfg = _compose(overrides)
    assert (cfg.ppo.num_envs, cfg.ppo.batch_size, cfg.seed, cfg.run_name, cfg.actuators.name) == (
        8192, 256, 1, "r", "deploy_pd"
    )


def test_train_needs_a_robot_or_an_experiment(workdir):
    code, calls = _run(workdir, "train.sh")
    assert code == 1
    assert calls == []


def test_train_exits_with_the_training_exit_code(workdir):
    code, calls = _run(workdir, "train.sh", ROBOT="roboto_origin", STUB_RCS="0 3")
    assert (code, len(calls)) == (3, 2)


# -- preflight_sizing.sh ---------------------------------------------------------


def test_preflight_sizing_composes_every_slice_and_writes_under_runs(workdir):
    """terrain_cpu pins 20,000 timesteps and config.yaml turns wandb on, so
    a dropped STEPS or WANDB default shows."""
    root, _ = workdir
    code, calls = _run(
        workdir, "preflight_sizing.sh", EXPERIMENT="terrain_cpu", SIZES_LIST="16 32", TAG="t",
        STEPS="1000", SEED="2",
    )
    assert code == 0
    assert len(calls) == 3
    for call, envs in zip(calls[1:], (16, 32), strict=True):
        overrides = _train_overrides(call)
        assert f"run_name=sizing_terrain_cpu_e{envs}_t" in overrides
        cfg = _compose(overrides)
        assert cfg.task.name == "terrain"
        assert (cfg.ppo.num_envs, cfg.ppo.batch_size) == (envs, envs // 32)
        assert (cfg.ppo.num_timesteps, cfg.seed, cfg.wandb.enable) == (1000, 2, False)
    assert [p.name for p in (root / "runs").iterdir()] == ["sizing-t"]


def test_preflight_sizing_fails_only_when_every_slice_fails(workdir):
    code, _ = _run(workdir, "preflight_sizing.sh", ROBOT="roboto_origin", SIZES_LIST="16 32",
                   STUB_RCS="0 1 0")
    assert code == 0
    code, _ = _run(workdir, "preflight_sizing.sh", ROBOT="roboto_origin", SIZES_LIST="16",
                   STUB_RCS="0 1")
    assert code == 1


# -- check_terrain.sh ------------------------------------------------------------


def test_check_terrain_gates_both_arenas_on_warp_by_default(workdir):
    root, _ = workdir
    code, calls = _run(workdir, "check_terrain.sh", EXPERIMENT="roboto_terrain_v1", TAG="t")
    assert code == 0
    assert [c["args"][:2] for c in calls] == [["-m", "humanoid_lab.check_terrain"]] * 2
    for call, arena in zip(calls, ("train", "eval"), strict=True):
        assert call["preallocate"] == "false"
        args, overrides = check_terrain.parser().parse_known_args(call["args"][2:])
        assert (args.backend, args.require_warp, args.engine) == ("warp", True, "mjx")
        assert (args.arena, args.seed) == (arena, 0)
        assert args.out == f"runs/check_terrain/t/{arena}.json"
        assert overrides == ["experiment=roboto_terrain_v1"]
        cfg = check_terrain.compose(overrides)
        assert (cfg["task"]["name"], cfg["robot"]["name"]) == ("terrain", "roboto_origin")
    assert (root / "runs" / "check_terrain" / "t").is_dir()


@pytest.mark.parametrize("rcs, expected", [("1 0", 1), ("1 2", 2), ("2 1", 2)])
def test_check_terrain_runs_every_arena_and_exits_with_the_worst(workdir, rcs, expected):
    """The largest exit code wins, whichever arena returned it. A failed
    arena does not stop the next one."""
    code, calls = _run(workdir, "check_terrain.sh", EXPERIMENT="roboto_terrain_v1", STUB_RCS=rcs)
    assert (code, len(calls)) == (expected, 2)


def test_check_terrain_off_warp_drops_require_warp(workdir):
    code, calls = _run(
        workdir, "check_terrain.sh", ROBOT="roboto_origin", BACKEND="jax", ARENAS="train",
        NUM_ENVS="8", STEPS="5", SEED="3", RUN_ARGS="--strict",
    )
    assert code == 0
    args, overrides = check_terrain.parser().parse_known_args(calls[0]["args"][2:])
    assert (args.backend, args.require_warp, args.num_envs, args.steps, args.seed, args.strict) == (
        "jax", False, 8, 5, 3, True
    )
    assert overrides == ["robot=roboto_origin"]


def test_check_terrain_header_counts_the_arenas_jax_refuses():
    """Each count is the arena's boxes and its aprons. jax refuses an arena
    over JAX_BOX_LIMIT, so only terrain_cpu's train arena runs there."""
    lines = (JOBS_DIR / "check_terrain.sh").read_text().splitlines()
    header = " ".join(line.lstrip("#").strip() for line in lines if line.startswith("#"))

    def ground_boxes(experiment, kind):
        cfg = check_terrain.compose([f"experiment={experiment}"])
        env_overrides = registry.env_args_from_config(cfg).env_overrides
        arena = check_terrain.effective_arena(env_overrides, check_terrain.arena_block(kind))
        return len(arena.boxes) + len(scene.APRON_GEOMS)

    full_eval = ground_boxes("roboto_terrain_v1", "eval")
    full_train = ground_boxes("roboto_terrain_v1", "train")
    cpu_train = ground_boxes("terrain_cpu", "train")
    assert f"more than {JAX_BOX_LIMIT} ground boxes" in header
    assert f"The eval arena has {full_eval}." in header
    assert f"roboto_terrain_v1's train arena has {full_train}." in header
    assert f"terrain_cpu's train arena has {cpu_train}." in header
    assert cpu_train <= JAX_BOX_LIMIT < min(full_eval, full_train)


def test_check_terrain_needs_an_experiment_or_a_robot(workdir):
    code, calls = _run(workdir, "check_terrain.sh")
    assert (code, calls) == (1, [])


# -- terrain_scan.sh -------------------------------------------------------------


def test_terrain_scan_writes_inside_the_run(workdir):
    code, calls = _run(workdir, "terrain_scan.sh", RUN_DIR="runs/r")
    assert code == 0
    (call,) = calls
    assert call["preallocate"] == "false"
    assert call["args"][:2] == ["-m", "humanoid_lab.eval.terrain_scan"]
    args = terrain_scan.parser().parse_args(call["args"][2:])
    # auto resolves to jax off a GPU, where the scan refuses the suite's arena.
    assert (str(args.run), args.backend, args.eval_seed) == ("runs/r", "auto", 0)
    assert str(args.out) == "runs/r/terrain_scan.json"
    assert (args.cells, args.speeds) == (None, None)


def test_terrain_scan_tags_a_partial_scan(workdir):
    code, calls = _run(
        workdir, "terrain_scan.sh", RUN_DIR="runs/r", CELLS="a,b", SPEEDS="0.3", EVAL_SEED="2",
        BACKEND="warp", TAG="part", RUN_ARGS="--njmax 64",
    )
    assert code == 0
    args = terrain_scan.parser().parse_args(calls[0]["args"][2:])
    assert str(args.out) == "runs/r/terrain_scan_part.json"
    assert (args.cells, args.speeds, args.eval_seed, args.njmax, args.backend) == (
        "a,b", "0.3", 2, 64, "warp"
    )


@pytest.mark.parametrize("run_dir", ["", "/tmp/r", "elsewhere/r", "runs/../r"])
def test_terrain_scan_refuses_a_run_dir_outside_runs(workdir, run_dir):
    code, calls = _run(workdir, "terrain_scan.sh", RUN_DIR=run_dir)
    assert (code, calls) == (1, [])


def test_terrain_scan_exits_with_the_scan_exit_code(workdir):
    code, _ = _run(workdir, "terrain_scan.sh", RUN_DIR="runs/r", STUB_RCS="2")
    assert code == 2
