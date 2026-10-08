#!/usr/bin/env bash
# The warp overflow gate for a terrain recipe: ./run.sh check-terrain on
# warp, once per arena. Each report gives the recipe's contact, CCD and row
# budgets. docs/terrain.md gives the order of work around it.
#
# This is a payload. jobs/README.md states the contract it is written
# against. Whatever launches it has already put the checkout in place, made
# the repo root the working directory, activated a venv with the training
# deps, configured the compile caches, and made the requested GPUs visible.
#
# Run this on a CUDA host of the node class the training run will use.
# MJWarp's buffers exist only on warp. The default backend is warp with
# --require-warp. --require-warp refuses a host where jax has no GPU, with
# exit 2.
#
# Parameters, all optional unless marked:
#   EXPERIMENT  hydra experiment preset  (REQUIRED unless ROBOT is set,
#                                         e.g. roboto_terrain_v1)
#   ROBOT       robot config             (default unset)
#   ACTUATORS   actuator preset          (default unset)
#   ARENAS      arenas to gate, in order (default "train eval")
#   NUM_ENVS    warp worlds per arena    (default unset, check-terrain's 1024)
#   STEPS       control steps per regime (default unset, check-terrain's 200)
#   SEED        check seed               (default 0)
#   BACKEND     warp, or auto or jax     (default warp)
#   TAG         label for this invocation's output dir (default a timestamp)
#   RUN_ARGS    extra check-terrain flags and hydra overrides, space separated
#
# ROBOT and ACTUATORS reach Hydra only when set. A set one wins over the
# experiment's pin. check-terrain puts `task=terrain` on its command line.
# That choice wins over an experiment's task pin. An experiment that pins
# another task is gated as task=terrain, as jobs/train.sh trains it with
# TASK=terrain. Only a `task=` in RUN_ARGS replaces it. A composed task
# other than terrain exits 2.
#
# `train` is the arena the recipe trains on. Its report sizes the recipe's
# task.env.sim budgets. `eval` is the terrain scan suite's arena. Its report
# sizes the budgets jobs/terrain_scan.sh passes in RUN_ARGS.
#
# BACKEND=warp adds --require-warp. Any other value runs without it.
# BACKEND=auto resolves to warp where jax runs on a GPU and MJWarp imports,
# and to jax elsewhere. On jax, check-terrain refuses an arena of more than
# 128 ground boxes, with exit 2. The eval arena has 1139.
# roboto_terrain_v1's train arena has 1904. A jax run of the default ARENAS
# therefore exits 2. terrain_cpu's train arena has 37. A jax run of it alone
# (EXPERIMENT=terrain_cpu ARENAS=train) reports `unverified` and exits 0.
# Only a warp run on a GPU verifies a recipe.
#
# The first gate of a recipe with no task.env.sim budgets:
#   EXPERIMENT=roboto_terrain_v1 ACTUATORS=deploy_pd ./jobs/check_terrain.sh
#
# The gate at explicit budgets, train arena only:
#   EXPERIMENT=roboto_terrain_v1 ACTUATORS=deploy_pd ARENAS=train \
#     RUN_ARGS="--naconmax-per-env N --naccdmax-per-env N --njmax N" \
#     ./jobs/check_terrain.sh
#
# Flags apply only where the recipe sets no task.env.sim budget. A recipe
# that sets them is gated at its own budgets.
#
# Reports go to runs/check_terrain/<TAG>/<arena>.json. A report's `status`
# is the verdict. Its `recommend` block holds the budgets that clear the
# measured peaks. Its `ccd_scratch` block projects the training run's CCD
# scratch. A recommendation marked `lower_bound` needs another gate at the
# recommended values.
#
# Partial-failure policy: every arena runs, whatever the arena before it
# returned. A failed gate is a measurement and still writes its report and
# its recommendations. The exit code is the largest of the arenas' exit
# codes. An arena exits 0 when it passed or was unverified, 1 when it failed
# or raised an exception, and 2 when it was refused.
#
# MJWarp allocates its CCD scratch outside the XLA pool. A preallocated XLA
# pool leaves that scratch only what the pool fraction leaves over, so this
# payload turns preallocation off. jaxlib 0.9.2 reads
# XLA_PYTHON_CLIENT_PREALLOCATE.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

EXPERIMENT="${EXPERIMENT:-}"
ROBOT="${ROBOT:-}"
if [ -z "$ROBOT" ] && [ -z "$EXPERIMENT" ]; then
    echo "ERROR: set EXPERIMENT to a configs/experiment/ name, e.g. roboto_terrain_v1, or set ROBOT" >&2
    exit 1
fi
ACTUATORS="${ACTUATORS:-}"
ARENAS="${ARENAS:-train eval}"
NUM_ENVS="${NUM_ENVS:-}"
STEPS="${STEPS:-}"
SEED="${SEED:-0}"
BACKEND="${BACKEND:-warp}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ARGS="${RUN_ARGS:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# See jobs/train.sh: an empty array's [@] is an unbound variable under set -u
# before bash 4.4, and the defaults can leave this array empty.
check_args=()
[ -n "$EXPERIMENT" ] && check_args+=("experiment=$EXPERIMENT")
[ -n "$ROBOT" ] && check_args+=("robot=$ROBOT")
[ -n "$ACTUATORS" ] && check_args+=("actuators=$ACTUATORS")
[ -n "$NUM_ENVS" ] && check_args+=(--num-envs "$NUM_ENVS")
[ -n "$STEPS" ] && check_args+=(--steps "$STEPS")
check_args+=(--seed "$SEED" --backend "$BACKEND")
[ "$BACKEND" = "warp" ] && check_args+=(--require-warp)

out_dir="runs/check_terrain/$TAG"
mkdir -p "$out_dir"

echo "== $(date -Iseconds) $(hostname) =="
worst=0
for arena in $ARENAS; do
    args=("${check_args[@]}" --arena "$arena" --out "$out_dir/$arena.json")
    # shellcheck disable=SC2206
    args+=($RUN_ARGS)
    echo "== CHECK arena=$arena: python3 -m humanoid_lab.check_terrain ${args[*]}"
    rc=0
    python3 -m humanoid_lab.check_terrain "${args[@]}" || rc=$?
    echo "== CHECK RESULT arena=$arena rc=$rc report=$out_dir/$arena.json"
    if [ "$rc" -gt "$worst" ]; then
        worst=$rc
    fi
done
echo "== $(date -Iseconds) done, reports in $out_dir, rc=$worst =="
exit "$worst"
