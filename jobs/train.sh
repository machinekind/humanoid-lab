#!/usr/bin/env bash
# One Hydra-configured Brax PPO training run.
#
# This is a payload. jobs/README.md states the contract it is written
# against. Whatever launches it has already put the checkout in place, made
# the repo root the working directory, activated a venv with the training
# deps, configured the compile caches, and made the requested GPUs visible.
# Brax PPO shards envs across every visible device and psums the gradients,
# so one process uses the whole machine.
#
# Parameters, all optional unless marked:
#   ROBOT       robot config             (REQUIRED unless EXPERIMENT is set)
#   TASK        task config              (default unset, config.yaml's joystick)
#   ACTUATORS   actuator preset          (default unset, config.yaml's sizing_ideal)
#   EXPERIMENT  hydra experiment preset  (default unset, no experiment)
#   RUN_NAME    run dir under runs/      (default train.py's <task>_<timestamp>)
#               The resolved run_name, from any source, must be one dir
#               name: no '/', no whitespace, not all, '.' or '..'. Any
#               other exits 1 before the training starts.
#   NUM_ENVS    parallel envs            (default 32768)
#   BATCH       ppo batch size           (default 1024)
#   SEED        training seed            (default 0)
#   WANDB       set false to turn wandb off (default true)
#   RUN_ARGS    extra hydra overrides, space separated
#   EVAL        true measures the run once the training call returns, with
#               any exit code: jobs/eval_runs.sh's courses, battery and
#               report, on CPU, on this allocation (default true)
#   EVAL_TIMEOUT  seconds the eval stage may take, 0 for no bound. Needs
#               GNU timeout(1) on PATH; without it the stage runs unbounded
#               and says so (default 1800)
#   EVAL_WORKERS  course lanes the stage runs at once; never changes a
#               number (default the courses CLI's: min(8, CPUs))
#
# ROBOT, TASK and ACTUATORS reach Hydra only when set. An experiment pins
# its own robot and task, and sometimes its actuators. A set variable wins
# over the experiment's pin, because Hydra applies command-line group
# choices after the experiment's defaults. An unset one leaves the
# experiment's pin, or config.yaml's default, in place.
#
# The NUM_ENVS/BATCH defaults are an untested starting point for a
# multi-GPU box: measure with jobs/preflight_sizing.sh on the real node
# class and scale the two together.
#
# A full run:
#   ROBOT=roboto_origin SEED=0 NUM_ENVS=32768 BATCH=1024 \
#     RUN_ARGS="++ppo.num_timesteps=3e8" ./jobs/train.sh
#
# A terrain run. A terrain recipe on warp refuses to build until its
# task.env.sim budgets are set. jobs/check_terrain.sh measures them
# (docs/terrain.md). A terrain training run holds MJWarp's CCD scratch
# outside the XLA pool. The scratch is naccdmax_per_env x NUM_ENVS slots.
# Roboto Origin's slot is 5,480 bytes. 128 slots per env hold 23 GB at
# 32768 envs and 5.7 GB at the 8192 below.
#   EXPERIMENT=roboto_terrain_v1 ACTUATORS=deploy_pd NUM_ENVS=8192 BATCH=256 \
#     ./jobs/train.sh
#
# WANDB=true turns the trainer's logging on and nothing else. Where the run
# files land is the environment's business. On a host with no route out,
# export WANDB_MODE=offline and point WANDB_DIR at a directory that outlives
# the job, then sync from somewhere with a network later. An offline run
# whose WANDB_DIR disappears with the job logs nothing.
#
# The eval stage measures the run this training wrote. The resolved config's
# run_name names it: RUN_NAME, a run_name= in RUN_ARGS (the last one wins),
# or the experiment's own. Without one, train.py names the run
# <task>_<timestamp>, and the stage takes the run.json this training wrote.
# A run.json is fresh when it is not older than the training's start. After
# a training that exited 0, the stage takes the only fresh one. After a
# nonzero exit, or when there are several, it takes the one whose
# hydra_config is this config, run_name aside. The stage pins the
# canonical measurement: every eval, the run's own files, 8 course seeds
# from seed 0, outputs recomputed even when current. EVAL_WORKERS, not an
# exported WORKERS, sets its course lanes.
#
# Training only, measured afterwards by the caller on a CPU host:
#   EVAL=false ROBOT=roboto_origin RUN_NAME=walk_a ./jobs/train.sh
#
# Partial-failure policy:
#   - A training that exits nonzero ends the script with its exit code.
#     When it left a fresh run.json (train.py writes one before training
#     starts), the stage measures the run's newest checkpoint before the
#     script exits. Without a run name, that run.json must carry this
#     config.
#   - A training that exits 0 followed by successful evals exits 0. The run
#     dir then holds courses.json, battery.json and eval_report.md.
#   - A training that exits 0 followed by a failed or timed-out eval exits
#     75. The run and its checkpoints are complete. Only the measurement is
#     missing, and RUNS=<run> ./jobs/eval_runs.sh completes it. 75 is
#     sysexits' EX_TEMPFAIL. Python and Hydra do not exit with it on their
#     own, so it cannot be read as a training code.
#   - A training that exits 0 but leaves no fresh runs/<run_name>/run.json,
#     or several fresh run.json files that this config cannot tell apart,
#     exits 75 with nothing measured.
#   - With no run name and no fresh run.json, the script warns and exits
#     with the training's code. train.py writes run.json before training
#     starts, so only a training that died before that leaves none.
#   - A SIGTERM or SIGKILL to this script's process group ends it before
#     the stage. The caller measures that run (jobs/README.md).
#   - The same kill during the stage ends the stage too. No eval outlives
#     the script, so the caller's own pass on the run never races one.
#   - EVAL=false skips the stage, for example when the caller measures on a
#     CPU host.
#
# The steps/s this prints compares only against another run at the same
# SEED (see CLAUDE.md's facts list). A single reading is not a throughput
# figure for the config.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

ROBOT="${ROBOT:-}"
TASK="${TASK:-}"
ACTUATORS="${ACTUATORS:-}"
EXPERIMENT="${EXPERIMENT:-}"
if [ -z "$ROBOT" ] && [ -z "$EXPERIMENT" ]; then
    echo "ERROR: set ROBOT to a configs/robot/ name, e.g. roboto_origin, or set EXPERIMENT" >&2
    exit 1
fi
RUN_NAME="${RUN_NAME:-}"
NUM_ENVS="${NUM_ENVS:-32768}"
BATCH="${BATCH:-1024}"
SEED="${SEED:-0}"
WANDB="${WANDB:-true}"
RUN_ARGS="${RUN_ARGS:-}"
EVAL="${EVAL:-true}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
EVAL_WORKERS="${EVAL_WORKERS:-}"
case "$EVAL" in
    true | false) ;;
    *) echo "ERROR: EVAL must be true or false, got '$EVAL'" >&2; exit 1 ;;
esac
case "$EVAL_TIMEOUT" in
    '' | *[!0-9]*) echo "ERROR: EVAL_TIMEOUT must be whole seconds, got '$EVAL_TIMEOUT'" >&2; exit 1 ;;
esac
case "$EVAL_WORKERS" in
    *[!0-9]* | 0*) echo "ERROR: EVAL_WORKERS must be a positive whole number, got '$EVAL_WORKERS'" >&2; exit 1 ;;
esac

if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="wandb.enable=true"
else
    WANDB_FLAG="wandb.enable=false"
fi

# Expanded below as ${hydra_args[@]:+...}. Under set -u, bash before 4.4
# calls an empty array's [@] an unbound variable, and the defaults can
# leave this array empty. `experiment=` takes no `+`: config.yaml's
# defaults list already holds `experiment: null`, and `+experiment=` fails
# to compose.
hydra_args=()
[ -n "$ROBOT" ] && hydra_args+=("robot=$ROBOT")
[ -n "$TASK" ] && hydra_args+=("task=$TASK")
[ -n "$ACTUATORS" ] && hydra_args+=("actuators=$ACTUATORS")
[ -n "$EXPERIMENT" ] && hydra_args+=("experiment=$EXPERIMENT")
[ -n "$RUN_NAME" ] && hydra_args+=("run_name=$RUN_NAME")

# ++ = add-or-override. The root `ppo:` block is an empty dict in config.yaml
# and in every task preset, so plain `ppo.foo=` fails hydra's struct check for
# keys the preset did not already set.
overrides=(
    seed="$SEED"
    "++ppo.num_envs=$NUM_ENVS"
    "++ppo.batch_size=$BATCH"
    "$WANDB_FLAG"
)
# shellcheck disable=SC2206
overrides+=(${hydra_args[@]:+"${hydra_args[@]}"} $RUN_ARGS)

# Compose the config and exit, before anything expensive starts. A typo in a
# preset name or an override fails here in seconds instead of after the
# environment has been built and the training has run for an hour.
echo "== resolving config =="
# Captured, not discarded: its run_name names the run this training writes.
resolved="$(python3 -m humanoid_lab.train --cfg job --resolve "${overrides[@]}")"
name="$(printf '%s\n' "$resolved" | sed -n 's/^run_name: *//p' | head -n 1)"
case "$name" in
    \'*\' | \"*\") name="${name:1:${#name}-2}" ;;
esac
# jobs/eval_runs.sh addresses a run by one dir name under runs/, and its
# RUNS splits on whitespace. A run it cannot address would stay unmeasured,
# by the stage and by the caller's own pass alike, so it is refused here.
case "$name" in
    all | . | .. | */* | *[[:space:]]*)
        echo "ERROR: run_name '$name' must be one dir name under runs/: no '/', no whitespace, not all, '.' or '..'" >&2
        exit 1 ;;
esac

# A run.json not older than this marker was written by this training.
# Not-older rather than newer, because bash 3.2 compares mtimes in whole
# seconds.
started=""
if [ "$EVAL" = "true" ]; then
    mkdir -p runs
    started="$(mktemp runs/.train_started.XXXXXX)"
    trap 'rm -f "$started" "$started.cfg"' EXIT
fi

echo "== $(date -Iseconds) $(hostname) =="
echo "== running: python3 -m humanoid_lab.train ${overrides[*]}"
rc=0
python3 -m humanoid_lab.train "${overrides[@]}" || rc=$?
echo "== $(date -Iseconds) done rc=$rc =="

[ "$EVAL" = "true" ] || exit "$rc"
# The exit code when this run cannot be measured: the training's own, else 75.
unmeasured="$rc"
[ "$rc" -ne 0 ] || unmeasured=75

case "$name" in
    '' | null | '~')
        # train.py names the run <task>_<timestamp>; find its fresh run.json.
        fresh=()
        for j in runs/*/run.json; do
            if [ -f "$j" ] && ! [ "$j" -ot "$started" ]; then
                fresh+=("${j%/run.json}")
            fi
        done
        if [ "${#fresh[@]}" -eq 0 ]; then
            echo "WARN: training rc=$rc wrote no run.json; nothing to measure" >&2
            exit "$rc"
        fi
        if [ "$rc" -eq 0 ] && [ "${#fresh[@]}" -eq 1 ]; then
            run="${fresh[0]#runs/}"
        else
            # A fresh run.json can be another job's: one besides this
            # training's, or the only one when this training exited nonzero
            # before writing its own. Keep the run whose hydra_config is
            # this resolve's, run_name aside.
            printf '%s\n' "$resolved" > "$started.cfg"
            if ! run="$(python3 - "$started.cfg" "${fresh[@]}" <<'PY'
import json
import sys
from pathlib import Path

import yaml


def config(cfg):
    cfg = dict(cfg)
    cfg.pop("run_name", None)
    return cfg


want = config(yaml.safe_load(Path(sys.argv[1]).read_text()))
hits = []
for run_dir in map(Path, sys.argv[2:]):
    try:
        have = json.loads((run_dir / "run.json").read_text()).get("hydra_config")
    except (OSError, ValueError, AttributeError):
        continue
    if isinstance(have, dict) and config(have) == want:
        hits.append(run_dir.name)
if len(hits) != 1:
    sys.exit(1)
print(hits[0])
PY
            )"; then
                echo "ERROR: training rc=$rc; of the fresh run.json in ${fresh[*]}, none or several match this config; not measured" >&2
                exit "$unmeasured"
            fi
        fi ;;
    *)  run="$name"
        if ! [ -f "runs/$run/run.json" ] || [ "runs/$run/run.json" -ot "$started" ]; then
            echo "ERROR: training rc=$rc left no fresh runs/$run/run.json; not measured" >&2
            exit "$unmeasured"
        fi ;;
esac

# --foreground keeps timeout and the stage in this script's process group.
# Without it GNU timeout moves itself and the evals into a new group, out of
# reach of a kill aimed at this payload's group. On expiry it then signals
# eval_runs.sh alone, which ends its running eval before it exits.
tmo=()
if [ "$EVAL_TIMEOUT" -gt 0 ]; then
    if command -v timeout >/dev/null 2>&1; then
        tmo=(timeout --foreground "$EVAL_TIMEOUT")
    else
        echo "NOTE: no timeout on PATH; the eval stage runs unbounded" >&2
    fi
fi
echo "== $(date -Iseconds) evaluating runs/$run =="
erc=0
RUNS="$run" EVALS="courses battery report" TAG= SEEDS=8 SEED_BASE=0 REDO=true CHECK=false \
    WORKERS="$EVAL_WORKERS" MUJOCO_GL=disabled \
    ${tmo[@]:+"${tmo[@]}"} bash jobs/eval_runs.sh || erc=$?
echo "== $(date -Iseconds) evals done rc=$erc =="
[ "$rc" -eq 0 ] || exit "$rc"
if [ "$erc" -ne 0 ]; then
    echo "ERROR: runs/$run trained; its evals failed rc=$erc (124 = EVAL_TIMEOUT). RUNS=$run ./jobs/eval_runs.sh completes them" >&2
    exit 75
fi
exit 0
