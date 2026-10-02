#!/usr/bin/env bash
# One terrain scan of a trained run: ./run.sh terrain-scan on warp. The run's
# checkpoint crosses every cell of its robot's scan suite, and the report
# lands inside the run's directory. docs/terrain.md gives the order of work
# around it.
#
# This is a payload. jobs/README.md states the contract it is written
# against. Whatever launches it has already put the checkout in place, made
# the repo root the working directory, activated a venv with the training
# deps, configured the compile caches, and made the requested GPUs visible.
# The run directory, its run.json and its checkpoints are in place under
# runs/.
#
# The full suite is 3072 worlds per speed on an arena of 1135 boxes. The jax
# backend refuses more than 128 ground boxes, so the scan runs on warp.
# BACKEND=auto resolves to warp where jax runs on a GPU and MJWarp imports.
# On a host without a GPU it resolves to jax. The scan then refuses the
# suite's arena with exit 2, before it builds anything. An explicit
# BACKEND=warp on a host without a GPU runs MJWarp on the CPU device.
#
# Parameters, all optional unless marked:
#   RUN_DIR     run directory under runs/ (REQUIRED, e.g. runs/roboto_terrain_v1)
#   CELLS       comma-separated cell names (default unset, every cell)
#   SPEEDS      comma-separated subset of the suite's speeds (default unset,
#               every speed)
#   EVAL_SEED   observation noise draw    (default 0)
#   BACKEND     auto, warp or jax         (default auto)
#   TAG         report name suffix        (default unset)
#   RUN_ARGS    extra terrain-scan flags, space separated
#
# The report is $RUN_DIR/terrain_scan.json, or
# $RUN_DIR/terrain_scan_<TAG>.json when TAG is set. A scan with CELLS,
# SPEEDS or a nonzero EVAL_SEED should carry a TAG, so it does not replace
# the run's full scan.
#
# The suite's warp budgets are 256 contacts and 2048 rows per world, and
# the naconmax pool for CCD. All three are untuned. jobs/check_terrain.sh's
# eval-arena report recommends measured ones. Pass them in RUN_ARGS.
#
# A full scan:
#   RUN_DIR=runs/roboto_terrain_v1 ./jobs/terrain_scan.sh
#
# A scan at measured budgets and another noise draw:
#   RUN_DIR=runs/roboto_terrain_v1 EVAL_SEED=1 TAG=seed1 \
#     RUN_ARGS="--naconmax-per-env N --naccdmax-per-env N --njmax N" \
#     ./jobs/terrain_scan.sh
#
# Reading the result. The scan exits 0 once it finishes, whatever its gate
# verdict. The verdict is the report's `gate.absolute.verdict`.
# `physics_clean` says whether MJWarp dropped anything. A scan that is not
# physics-clean keeps its numbers, and its verdict is `invalid`.
#
# Partial-failure policy: there is none to have. This runs one scan and
# exits with its exit code: 0 for a finished scan, 2 for a refused request,
# 1 for an exception.
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

: "${RUN_DIR:?set RUN_DIR to the directory of a trained run under runs/, e.g. runs/roboto_terrain_v1}"
case "$RUN_DIR" in
    *..*)
        echo "ERROR: RUN_DIR '$RUN_DIR' climbs with '..', and a payload writes only under runs/" >&2
        exit 1
        ;;
    runs/* | ./runs/* | "$PWD"/runs/*) ;;
    *)
        echo "ERROR: RUN_DIR '$RUN_DIR' is not under runs/, and a payload writes only there" >&2
        exit 1
        ;;
esac
CELLS="${CELLS:-}"
SPEEDS="${SPEEDS:-}"
EVAL_SEED="${EVAL_SEED:-0}"
BACKEND="${BACKEND:-auto}"
TAG="${TAG:-}"
RUN_ARGS="${RUN_ARGS:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

out="$RUN_DIR/terrain_scan.json"
[ -n "$TAG" ] && out="$RUN_DIR/terrain_scan_$TAG.json"

scan_args=(--run "$RUN_DIR" --backend "$BACKEND" --eval-seed "$EVAL_SEED" --out "$out")
[ -n "$CELLS" ] && scan_args+=(--cells "$CELLS")
[ -n "$SPEEDS" ] && scan_args+=(--speeds "$SPEEDS")
# shellcheck disable=SC2206
scan_args+=($RUN_ARGS)

echo "== $(date -Iseconds) $(hostname) =="
echo "== running: python3 -m humanoid_lab.eval.terrain_scan ${scan_args[*]}"
rc=0
python3 -m humanoid_lab.eval.terrain_scan "${scan_args[@]}" || rc=$?
echo "== $(date -Iseconds) done rc=$rc report=$out =="
exit "$rc"
