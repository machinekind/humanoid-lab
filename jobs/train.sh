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
# Parameters, all optional:
#   ROBOT       robot config             (REQUIRED, e.g. roboto_origin)
#   TASK        task config              (default joystick)
#   ACTUATORS   actuator preset          (default sizing_ideal)
#   EXPERIMENT  hydra experiment preset  (default unset, no experiment override)
#   RUN_NAME    run dir under runs/      (default train.py's <task>_<timestamp>)
#   NUM_ENVS    parallel envs            (default 32768)
#   BATCH       ppo batch size           (default 1024)
#   SEED        training seed            (default 0)
#   WANDB       set false to turn wandb off (default true)
#   RUN_ARGS    extra hydra overrides, space separated
#
# The NUM_ENVS/BATCH defaults are an untested starting point for a
# multi-GPU box: measure with jobs/preflight_sizing.sh on the real node
# class and scale the two together.
#
# A full run:
#   ROBOT=roboto_origin SEED=0 NUM_ENVS=32768 BATCH=1024 \
#     RUN_ARGS="++ppo.num_timesteps=3e8" ./jobs/train.sh
#
# WANDB=true turns the trainer's logging on and nothing else. Where the run
# files land is the environment's business. On a host with no route out,
# export WANDB_MODE=offline and point WANDB_DIR at a directory that outlives
# the job, then sync from somewhere with a network later. An offline run
# whose WANDB_DIR disappears with the job logs nothing.
#
# Partial-failure policy: there is none to have. This runs one training and
# exits with its exit code.
#
# The steps/s this prints compares only against another run at the same
# SEED (see CLAUDE.md's facts list). A single reading is not a throughput
# figure for the config.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

: "${ROBOT:?set ROBOT to a configs/robot/ name, e.g. roboto_origin}"
TASK="${TASK:-joystick}"
ACTUATORS="${ACTUATORS:-sizing_ideal}"
EXPERIMENT="${EXPERIMENT:-}"
RUN_NAME="${RUN_NAME:-}"
NUM_ENVS="${NUM_ENVS:-32768}"
BATCH="${BATCH:-1024}"
SEED="${SEED:-0}"
WANDB="${WANDB:-true}"
RUN_ARGS="${RUN_ARGS:-}"

if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="wandb.enable=true"
else
    WANDB_FLAG="wandb.enable=false"
fi

# Expanded below as ${hydra_args[@]:+...}. Under set -u, bash before 4.4
# calls an empty array's [@] an unbound variable, and both defaults leave
# this array empty.
hydra_args=()
[ -n "$EXPERIMENT" ] && hydra_args+=("+experiment=$EXPERIMENT")
[ -n "$RUN_NAME" ] && hydra_args+=("run_name=$RUN_NAME")

# ++ = add-or-override. The root `ppo:` block is an empty dict in config.yaml
# and in every task preset, so plain `ppo.foo=` fails hydra's struct check for
# keys the preset did not already set.
overrides=(
    robot="$ROBOT" task="$TASK" actuators="$ACTUATORS"
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
python3 -m humanoid_lab.train --cfg job --resolve "${overrides[@]}" >/dev/null

echo "== $(date -Iseconds) $(hostname) =="
echo "== running: python3 -m humanoid_lab.train ${overrides[*]}"
rc=0
python3 -m humanoid_lab.train "${overrides[@]}" || rc=$?
echo "== $(date -Iseconds) done rc=$rc =="
exit "$rc"
