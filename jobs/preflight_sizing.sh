#!/usr/bin/env bash
# Bounded training slices at several env counts, to size a full-budget launch.
# The two outputs are peak GPU memory and steps/s per env count.
#
# This is a payload. jobs/README.md states the contract it is written
# against. Whatever launches it has already put the checkout in place, made
# the repo root the working directory, activated a venv with the training
# deps, configured the compile caches, and made the requested GPUs visible.
#
# Run this on the same hardware the real run will use. Nothing measured
# elsewhere transfers. A different machine, card or driver gets its own
# measurement: a 24 GB card's env ceiling says nothing about an H100's.
#
# steps/s compares across the slices of one invocation, because they share a
# seed. Across invocations it compares only at a matched SEED. Two runs
# differing only in seed have reported 1,343,166 and 777,859 steps/s. Spawns
# and falls differ, so a different number of geoms sit in contact and the
# contact accumulator overflows at a different rate.
#
# Each slice is a throwaway training of the preset a real run will use, domain
# randomization included, so the measured rate is the launch's early-run cost.
# Training is most expensive while the policy still falls.
#
# Parameters, all optional:
#   ROBOT       robot config             (default asimov_v1)
#   TASK        task config              (default joystick)
#   ACTUATORS   actuator preset          (default sizing_ideal)
#   EXPERIMENT  hydra experiment preset  (default unset, no experiment override)
#   SIZES_LIST  env counts               (default "8192 16384 32768")
#   STEPS       timestep budget per slice, a plain integer (default 30000000)
#   SEED        seed shared by every slice (default 0)
#   TAG         label for this invocation's output dir (default a timestamp)
#   WANDB       set true to log the slices (default false)
#   RUN_ARGS    extra hydra overrides applied to every slice, space separated
#
# List SIZES_LIST ascending. An OOM at a large size then cannot block the
# smaller measurements.
#
# The slices are throwaway runs, and dozens of them bury the real runs in a
# wandb project, which is why WANDB defaults to false here and to true in
# jobs/train.sh.
#
# A default sweep:
#   ./jobs/preflight_sizing.sh
#
# A wider one:
#   SIZES_LIST="8192 16384 32768 65536" ./jobs/preflight_sizing.sh
#
# Batch size per slice is num_envs/32. The playground Go1 PPO config this repo
# builds on sets num_minibatches=32, and brax wants batch_size*num_minibatches
# divisible by num_envs. The ratio matches jobs/train.sh's 32768/1024 defaults.
#
# Partial-failure policy: a slice that dies is a measurement, not an error.
# That is the OOM ceiling this script exists to find, so a failed slice is
# recorded as verdict=FAILED and the sweep continues. The exit code is 0 when
# at least one slice succeeded and 1 when none did, because a sweep where
# everything failed measured nothing.
#
# Reading the results. peak_mem is the per-GPU maximum nvidia-smi saw during
# the slice, sampled every 5s, and reads "unavailable" where nvidia-smi is not
# installed. steps/s comes from the training progress lines in this script's
# own output. Take a steady-state line rather than the first, whose interval
# includes the multi-minute MJX compile. The elapsed time per slice is wall
# clock including that compile, so it is a floor on cost rather than a
# throughput figure.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

ROBOT="${ROBOT:-asimov_v1}"
TASK="${TASK:-joystick}"
ACTUATORS="${ACTUATORS:-sizing_ideal}"
EXPERIMENT="${EXPERIMENT:-}"
SIZES_LIST="${SIZES_LIST:-8192 16384 32768}"
STEPS="${STEPS:-30000000}"
SEED="${SEED:-0}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
WANDB="${WANDB:-false}"
RUN_ARGS="${RUN_ARGS:-}"

# Without this every slice measures the same number: nvidia-smi reports the
# preallocated XLA pool rather than real usage. It is also mandatory under the
# warp backend, which sim.backend=auto selects on a GPU host, because warp
# allocates its EPA scratch outside the XLA pool.
export XLA_PYTHON_CLIENT_PREALLOC=false

if [ "$WANDB" = "true" ]; then
    WANDB_FLAG="wandb.enable=true"
else
    WANDB_FLAG="wandb.enable=false"
fi

# See jobs/train.sh: an empty array's [@] is an unbound variable under set -u
# before bash 4.4, and EXPERIMENT is unset by default.
hydra_args=()
[ -n "$EXPERIMENT" ] && hydra_args+=("+experiment=$EXPERIMENT")

# Compose the config once before the sweep starts. Without this a typo in a
# preset name fails every slice, and the sweep reports it as an OOM ceiling at
# every size.
echo "== resolving config =="
check_args=(
    robot="$ROBOT" task="$TASK" actuators="$ACTUATORS" seed="$SEED"
    "$WANDB_FLAG"
)
# shellcheck disable=SC2206
check_args+=(${hydra_args[@]:+"${hydra_args[@]}"} $RUN_ARGS)
python3 -m humanoid_lab.train --cfg job --resolve "${check_args[@]}" >/dev/null

out_dir="runs/sizing-$TAG"
mkdir -p "$out_dir"

# The memory poller is a background process per slice. An abort between
# starting it and the kill below would otherwise leave it writing for as long
# as the shell lives.
poller=""
trap 'kill "$poller" 2>/dev/null || true' EXIT

succeeded=0
for envs in $SIZES_LIST; do
    batch=$((envs / 32))
    mem_log="$out_dir/mem-e${envs}.csv"
    poller=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 5 \
            > "$mem_log" &
        poller=$!
    fi
    echo "== SIZING slice: num_envs=$envs batch=$batch steps=$STEPS seed=$SEED =="
    started=$SECONDS
    slice_args=(
        robot="$ROBOT" task="$TASK" actuators="$ACTUATORS"
        seed="$SEED"
        "++ppo.num_envs=$envs"
        "++ppo.batch_size=$batch"
        "++ppo.num_timesteps=$STEPS"
        "run_name=sizing_${ROBOT}_e${envs}_${TAG}"
        "$WANDB_FLAG"
    )
    # shellcheck disable=SC2206
    slice_args+=(${hydra_args[@]:+"${hydra_args[@]}"} $RUN_ARGS)
    if python3 -m humanoid_lab.train "${slice_args[@]}"; then
        verdict="OK"
        succeeded=$((succeeded + 1))
    else
        verdict="FAILED"
    fi
    elapsed=$((SECONDS - started))
    peak="unavailable"
    if [ -n "$poller" ]; then
        kill "$poller" 2>/dev/null || true
        wait "$poller" 2>/dev/null || true
        # The numeric guards drop the partial line nvidia-smi can leave when
        # it is killed mid-write.
        peak=$(awk -F', ' '$1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ {if ($2+0 > m[$1]) m[$1]=$2} END {for (g in m) printf "gpu%s:%sMiB ", g, m[g]}' "$mem_log")
        [ -n "$peak" ] || peak="nothing sampled"
    fi
    echo "== SIZING RESULT envs=$envs batch=$batch verdict=$verdict elapsed=${elapsed}s peak_mem: $peak"
done
echo "== SIZING DONE, $succeeded slice(s) succeeded, logs in $out_dir =="

if [ "$succeeded" -eq 0 ]; then
    echo "ERROR: every slice failed, so this sweep measured nothing" >&2
    exit 1
fi
