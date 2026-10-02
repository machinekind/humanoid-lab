#!/usr/bin/env bash
# Two trainings back to back: phase A from scratch, then phase B warm-started
# from phase A's latest checkpoint.
#
# This is a payload. jobs/README.md states the contract it is written
# against. Whatever launches it has already put the checkout in place, made
# the repo root the working directory, activated a venv with the training
# deps, configured the compile caches, and made the requested GPUs visible.
# Each phase is one jobs/train.sh run, so everything that file says about
# sizing, wandb and steps/s applies to both phases.
#
# Phase B restores through the stock `restore=` key, relative to the repo
# root (train.py resolves it against the project dir). A restore carries the
# observation normalizer and the policy and value params. The optimizer
# state and the step counter start fresh, so PHASE_B_STEPS is phase B's own
# budget, not a total.
#
# Parameters:
#   ROBOT               robot config                (REQUIRED, e.g. roboto_origin)
#   RUN_NAME            phase B's run dir under runs/ (REQUIRED); phase A
#                       runs as ${RUN_NAME}_a. One dir name: no '/', no
#                       whitespace, not all, '.' or '..'
#   PHASE_A_EXPERIMENT  hydra experiment preset for phase A (REQUIRED)
#   PHASE_B_EXPERIMENT  hydra experiment preset for phase B (REQUIRED)
#   PHASE_A_STEPS       phase A ppo.num_timesteps   (default 5.0e8)
#   PHASE_B_STEPS       phase B ppo.num_timesteps   (default 5.5e8)
#   ACTUATORS           actuator preset             (default sizing_ideal)
#   SEED                training seed, both phases  (default 0)
#   NUM_ENVS            parallel envs, both phases  (default 32768)
#   BATCH               ppo batch size, both phases (default 1024)
#   WANDB               set false to turn wandb off (default true)
#   EXTRA_ARGS          extra hydra overrides for both phases, space
#                       separated, applied after the budget and restore keys
#                       so they win
#   TASK                task config, both phases    (default joystick)
#   EVAL                jobs/train.sh's eval stage, both phases; each
#                       phase reads it from the environment (default true)
#   EVAL_TIMEOUT        the stage's bound in seconds, both phases, read the
#                       same way (default 1800)
#   EVAL_WORKERS        the stage's course lanes at once, both phases, read
#                       the same way (default the courses CLI's: min(8, CPUs))
#
# Each phase's jobs/train.sh measures the run it trained (EVAL), so both
# phases get courses.json, battery.json and eval_report.md. A phase that
# trained but whose evals failed exits 75 from jobs/train.sh. For phase A
# that still counts as trained, so the rules below decide on its run.json as
# they would after an exit 0. When phase B then exits 0, the chain exits 75
# and names phase A's run: RUNS=${RUN_NAME}_a ./jobs/eval_runs.sh measures
# it. A SIGTERM to the process group ends the chain and any eval stage it
# is running. The caller then runs
# RUNS="${RUN_NAME}_a ${RUN_NAME}" ./jobs/eval_runs.sh and names only the
# run dirs that exist.
#
# A chain:
#   ROBOT=roboto_origin ACTUATORS=deploy_pd SEED=0 NUM_ENVS=4096 BATCH=128 \
#     RUN_NAME=chain_sym PHASE_A_EXPERIMENT=yolo_v4 \
#     PHASE_B_EXPERIMENT=yolo_chain_sym_b ./jobs/train_chain.sh
# runs runs/chain_sym_a/ for 5.0e8 steps, then runs/chain_sym/ for 5.5e8
# steps restored from runs/chain_sym_a/checkpoints/<largest step>.
#
# The checkpoint phase B restores from is the largest numeric step dir under
# runs/${RUN_NAME}_a/checkpoints/ (brax names them by zero-padded step
# count) that holds a ppo_network_config.json written since this phase A
# started. brax writes that file after the params, so a step dir cut short
# mid-save is skipped, and a leftover checkpoint from an earlier phase A in
# the same run dir is never picked.
#
# Before phase A starts, phase B's config is composed with the same
# overrides jobs/train.sh will pass it (a placeholder restore path stands in
# for the checkpoint). A typo in PHASE_B_EXPERIMENT or an override that only
# phase B rejects fails here, in seconds, instead of after phase A's budget.
#
# Partial-failure policy: phase B needs a phase A that ran its full budget.
# The evidence is runs/${RUN_NAME}_a/run.json, written since this phase A
# started, with early_stopped false, or with stopped_at_steps at or past
# num_timesteps. Whatever phase A's exit code, the chain decides on that
# file:
#   - No fresh run.json, or one whose early_stopped and stopped_at_steps
#     are still null (a signal, OOM, a crash before training returned):
#     phase B does not run; the chain exits with phase A's code, or 1 if
#     that code was 0 or 75. train.py writes run.json before training
#     starts and fills those fields when training returns. It installs no
#     signal handler, so a SIGTERM aimed at the trainer alone ends it with
#     the fields null, and phase B never starts.
#   - A fresh run.json cut short of its budget: phase B does not run; the
#     chain exits with phase A's code, or 143 if that code was 0 or 75.
#     train.py writes such a run.json only after a plateau stop, which
#     needs early_stop.enable on (off by default).
#   - A full-budget run.json and a nonzero exit (a crash at interpreter
#     exit, wandb or GPU teardown, or 75 from failed evals): phase B runs.
#   - A full-budget run.json but no complete checkpoint: the chain exits 1.
# Otherwise the chain exits with phase B's code, or with 75 when phase B
# exits 0 and phase A's evals failed. A SIGTERM to this script's
# process group stops this bash as well, so no phase starts after it.
# Phase A's run dir and checkpoints stay in place in every case.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

: "${ROBOT:?set ROBOT to a configs/robot/ name, e.g. roboto_origin}"
: "${RUN_NAME:?set RUN_NAME to the phase B run dir name, phase A runs as <RUN_NAME>_a}"
: "${PHASE_A_EXPERIMENT:?set PHASE_A_EXPERIMENT to a configs/experiment/ name}"
: "${PHASE_B_EXPERIMENT:?set PHASE_B_EXPERIMENT to a configs/experiment/ name}"
: "${PHASE_A_STEPS:=5.0e8}"
: "${PHASE_B_STEPS:=5.5e8}"
: "${ACTUATORS:=sizing_ideal}"
: "${SEED:=0}"
: "${NUM_ENVS:=32768}"
: "${BATCH:=1024}"
: "${WANDB:=true}"
: "${EXTRA_ARGS:=}"

: "${TASK:=joystick}"

# jobs/train.sh refuses a run_name jobs/eval_runs.sh cannot address. Phase
# A's name can pass where phase B's fails (all_a and all), so RUN_NAME is
# checked before phase A spends its budget.
case "$RUN_NAME" in
    all | . | .. | */* | *[[:space:]]*)
        echo "ERROR: RUN_NAME '$RUN_NAME' must be one dir name under runs/: no '/', no whitespace, not all, '.' or '..'" >&2
        exit 1 ;;
esac

RUN_A="${RUN_NAME}_a"
CKPT_DIR_A="runs/$RUN_A/checkpoints"
START_MARKER="runs/$RUN_A/chain_phase_a.started"
RUN_JSON_A="runs/$RUN_A/run.json"

# Prints the largest numeric step dir under $1 whose ppo_network_config.json
# is not older than $2. Returns 1 when there is none. brax writes that file
# last in a save, so its presence marks a complete checkpoint. Not-older
# rather than newer, because bash 3.2 compares mtimes in whole seconds. 10#
# reads a zero-padded name as decimal, not octal, and compares numerically,
# so the pick does not depend on the padding width.
latest_checkpoint() {
    local dir="$1" marker="$2" best="" best_step=-1 d name step
    [ -d "$dir" ] || return 1
    for d in "$dir"/*/; do
        [ -d "$d" ] || continue
        d="${d%/}"
        name="${d##*/}"
        case "$name" in
            '' | *[!0-9]*) continue ;;
        esac
        if [ ! -f "$d/ppo_network_config.json" ] || [ "$d/ppo_network_config.json" -ot "$marker" ]; then
            continue
        fi
        step=$((10#$name))
        if [ "$step" -gt "$best_step" ]; then
            best_step="$step"
            best="$d"
        fi
    done
    [ -n "$best" ] || return 1
    printf '%s\n' "$best"
}

# Prints "<early_stopped 0|1> <stopped_at_steps> <num_timesteps>" from the
# run.json at $1.
run_json_budget() {
    python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    d = json.load(f)
print(int(bool(d["early_stopped"])), int(d["stopped_at_steps"]), int(d["num_timesteps"]))
PY
}

# run_phase <run name> <experiment> <run args>
run_phase() {
    ROBOT="$ROBOT" TASK="$TASK" ACTUATORS="$ACTUATORS" SEED="$SEED" \
        NUM_ENVS="$NUM_ENVS" BATCH="$BATCH" WANDB="$WANDB" \
        RUN_NAME="$1" EXPERIMENT="$2" RUN_ARGS="$3" \
        bash jobs/train.sh
}

# resolve_phase <run name> <experiment> <run args>
# Composes the config jobs/train.sh would build for these arguments and
# exits. The override list mirrors jobs/train.sh's.
resolve_phase() {
    local wandb_flag=wandb.enable=false
    [ "$WANDB" = "true" ] && wandb_flag=wandb.enable=true
    # shellcheck disable=SC2206
    local overrides=(
        robot="$ROBOT" task="$TASK" actuators="$ACTUATORS"
        seed="$SEED"
        "++ppo.num_envs=$NUM_ENVS"
        "++ppo.batch_size=$BATCH"
        "$wandb_flag"
        "experiment=$2" "run_name=$1"
        $3
    )
    python3 -m humanoid_lab.train --cfg job --resolve "${overrides[@]}" >/dev/null
}

echo "== chain: resolving phase B config ($PHASE_B_EXPERIMENT) =="
if ! resolve_phase "$RUN_NAME" "$PHASE_B_EXPERIMENT" \
    "restore=$CKPT_DIR_A/0 ++ppo.num_timesteps=$PHASE_B_STEPS $EXTRA_ARGS"; then
    echo "ERROR: phase B config ($PHASE_B_EXPERIMENT) does not compose; phase A ($RUN_A) not started" >&2
    exit 1
fi

mkdir -p "runs/$RUN_A"
touch "$START_MARKER"

echo "== chain phase A: run_name=$RUN_A experiment=$PHASE_A_EXPERIMENT steps=$PHASE_A_STEPS =="
rc_a=0
run_phase "$RUN_A" "$PHASE_A_EXPERIMENT" \
    "++ppo.num_timesteps=$PHASE_A_STEPS $EXTRA_ARGS" || rc_a=$?
# jobs/train.sh's 75: phase A trained, and only its evals failed.
eval_missing_a=0
if [ "$rc_a" -eq 75 ]; then
    eval_missing_a=1
fi
# fail_a <code if phase A exited 0 or 75> <message>
fail_a() {
    echo "ERROR: phase A ($RUN_A) exited rc=$rc_a $2; phase B ($RUN_NAME) not started" >&2
    if [ "$rc_a" -ne 0 ] && [ "$eval_missing_a" -eq 0 ]; then
        exit "$rc_a"
    fi
    exit "$1"
}

if ! [ -f "$RUN_JSON_A" ] || [ "$RUN_JSON_A" -ot "$START_MARKER" ]; then
    fail_a 1 "without writing $RUN_JSON_A"
fi
if ! budget="$(run_json_budget "$RUN_JSON_A")"; then
    fail_a 1 "and $RUN_JSON_A has no readable early_stopped, stopped_at_steps and num_timesteps"
fi
read -r early_stopped stopped_at budget_steps <<<"$budget"
if [ "$early_stopped" = 1 ] && [ "$stopped_at" -lt "$budget_steps" ]; then
    fail_a 143 "after an early stop at $stopped_at of $budget_steps steps (a plateau stop with early_stop.enable on)"
fi
if [ "$eval_missing_a" -eq 1 ]; then
    echo "WARNING: phase A ($RUN_A) trained its full budget but its evals failed (rc=75); continuing to phase B" >&2
elif [ "$rc_a" -ne 0 ]; then
    echo "WARNING: phase A ($RUN_A) exited rc=$rc_a after writing a full-budget $RUN_JSON_A; training finished, continuing to phase B" >&2
fi

if ! ckpt="$(latest_checkpoint "$CKPT_DIR_A" "$START_MARKER")"; then
    echo "ERROR: phase A ($RUN_A) finished (rc=$rc_a) but $CKPT_DIR_A/ has no complete numeric step dir (with ppo_network_config.json) written since $START_MARKER; phase B ($RUN_NAME) not started" >&2
    exit 1
fi

echo "== chain phase B: run_name=$RUN_NAME experiment=$PHASE_B_EXPERIMENT steps=$PHASE_B_STEPS restore=$ckpt =="
rc_b=0
run_phase "$RUN_NAME" "$PHASE_B_EXPERIMENT" \
    "restore=$ckpt ++ppo.num_timesteps=$PHASE_B_STEPS $EXTRA_ARGS" || rc_b=$?
echo "== chain done: phase A rc=$rc_a, phase B rc=$rc_b =="
if [ "$eval_missing_a" -eq 1 ]; then
    echo "ERROR: phase A ($RUN_A) trained but is not measured. RUNS=$RUN_A ./jobs/eval_runs.sh measures it" >&2
    if [ "$rc_b" -eq 0 ]; then
        exit 75
    fi
fi
exit "$rc_b"
