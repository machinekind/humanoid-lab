#!/usr/bin/env bash
# The Torch+MJWarp acceptance run on a CUDA host. It runs the unit suite, the
# parity suite on CPU and on CUDA, a smoke training with a resume, a bounded
# training of the real recipe, the bench, and battery on the result. It
# writes a step table at the end. Everything it writes lands under runs/.
#
# This is a payload. jobs/README.md states the contract it is written
# against. Whatever launches it has already put the checkout in place, made
# the repo root the working directory, activated a venv with the training
# deps, configured the compile caches, and made the requested GPUs visible.
#
# The venv the caller prepares holds the base dependencies. The Torch trainer
# needs two more: the dev and torch_warp extras, and on Linux a CUDA torch
# from the cu126 index, which the extra cannot name (see
# src/humanoid_lab/torch_warp/README.md, "Linux"). This payload installs
# both into the active venv when they are missing. INSTALL_TORCH=false skips
# the torch install and fails instead.
#
# Parameters, all optional:
#   TAG            label for this invocation; names its runs   (default a timestamp)
#   EXPERIMENT     hydra experiment preset                     (default roboto_gait_v40)
#   REAL_MINUTES   wall-clock budget of the real-recipe run    (default 15, 0 skips it)
#   NUM_ENVS       env count for the real run and the bench    (default unset: the preset's)
#   SEED           seed for every run                          (default 0)
#   UNIT_TESTS     run tests/unit first                        (default true)
#   INSTALL_TORCH  install CUDA torch when it is missing       (default true)
#   TORCH_INDEX    wheel index for that install                (default the cu126 index)
#   ONLY_PARITY    stop after the two parity steps             (default false)
#
# Outputs, under runs/torch_warp_validate_<TAG>/:
#   environment.json   interpreter, library and GPU versions, from the box
#   unit.log, parity_cpu.log, parity_cuda.log   the pytest output, with --parity-report
#   smoke.log, resume.log, real.log, bench.log, battery.log, video.log
#   bench.json         the bench's JSON
#   steps.tsv          one line per step: name, exit code, seconds
# The trainings land in runs/<TAG>_smoke/ and runs/<TAG>_real/.
#
# A default run:
#   ./jobs/torch_warp_validate.sh
#
# A longer real run at a chosen env count:
#   REAL_MINUTES=60 NUM_ENVS=4096 ./jobs/torch_warp_validate.sh
#
# The parity suite alone, on CPU and on CUDA, with the environment report:
#   ONLY_PARITY=true UNIT_TESTS=false ./jobs/torch_warp_validate.sh
#
# Partial-failure policy: every step runs whatever the earlier ones did, so
# one table shows the whole state. The exit code is nonzero when any step but
# the video render failed. The video needs a GL library the image may lack.

set -uo pipefail

: "${GPUS:=1}"
: "${ON_FAILURE:=runs every step and exits nonzero if any step but the video render failed}"
: "${TAG:=$(date +%Y%m%d_%H%M%S)}"
: "${EXPERIMENT:=roboto_gait_v40}"
: "${REAL_MINUTES:=15}"
: "${NUM_ENVS:=}"
: "${SEED:=0}"
: "${UNIT_TESTS:=true}"
: "${INSTALL_TORCH:=true}"
: "${TORCH_INDEX:=https://download.pytorch.org/whl/cu126}"
: "${ONLY_PARITY:=false}"
# JAX stays on the CPU here. torch and warp own the GPU, and the Torch
# trainer refuses a GPU JAX backend.
# MUJOCO_GL is left alone: an exported `egl` makes `import mujoco` fail on an
# image without libEGL, and only the video step renders. eval/video.py sets
# egl itself on Linux.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

OUT="runs/torch_warp_validate_$TAG"
mkdir -p "$OUT"
STEPS="$OUT/steps.tsv"
: > "$STEPS"
failed=0

# step NAME LOG CMD...: runs CMD with its output in LOG, records the exit
# code and the seconds, and prints the log's tail when it failed.
step() {
    local name="$1" log="$2" rc start
    shift 2
    echo "== $(date -Iseconds) $name: $*"
    start=$(date +%s)
    "$@" > "$log" 2>&1
    rc=$?
    printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - start ))" >> "$STEPS"
    if [ "$rc" -ne 0 ]; then
        echo "== $name FAILED rc=$rc; last lines of $log:"
        tail -n 30 "$log"
    else
        echo "== $name ok, $(( $(date +%s) - start ))s"
    fi
    return "$rc"
}

# finish: prints the step table and exits with the run's status.
finish() {
    echo "== steps (name, rc, seconds) =="
    cat "$STEPS"
    echo "== $(date -Iseconds) done failed=$failed =="
    exit "$failed"
}

PY=.venv/bin/python
[ -x "$PY" ] || PY="$(command -v python3)"

# -- dependencies ------------------------------------------------------------
pip_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PY" "$@"
    else
        "$PY" -m pip install "$@"
    fi
}
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    if [ "$INSTALL_TORCH" = "true" ]; then
        step install_torch "$OUT/install.log" pip_install torch --index-url "$TORCH_INDEX" || exit 1
    else
        echo "ERROR: no CUDA torch in the venv and INSTALL_TORCH=false" >&2
        exit 1
    fi
fi
if ! "$PY" -c "import pytest, psutil" 2>/dev/null; then
    step install_extras "$OUT/install_extras.log" pip_install -e ".[dev,torch_warp]" || exit 1
fi

# -- environment report ------------------------------------------------------
"$PY" - "$OUT/environment.json" <<'PY'
import json, platform, subprocess, sys
import jax, mujoco, torch, warp
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip().split(", ")
    gpu = dict(zip(("name", "memory_total", "driver_version", "compute_cap"), smi))
except Exception as e:  # noqa: BLE001
    gpu = {"error": str(e)}
report = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "warp": warp.__version__,
    "warp_cuda_available": warp.is_cuda_available(),
    "jax": jax.__version__,
    "jax_backend": jax.default_backend(),
    "mujoco": mujoco.__version__,
    "gpu": gpu,
}
json.dump(report, open(sys.argv[1], "w"), indent=2)
print(json.dumps(report, indent=2))
PY

# -- tests -------------------------------------------------------------------
if [ "$UNIT_TESTS" = "true" ]; then
    step unit "$OUT/unit.log" ./run.sh test || failed=1
fi
step parity_cpu "$OUT/parity_cpu.log" ./run.sh test-parity --parity-report || failed=1
step parity_cuda "$OUT/parity_cuda.log" ./run.sh test-parity --parity-device cuda:0 --parity-report || failed=1
[ "$ONLY_PARITY" = "true" ] && finish

# -- trainings ---------------------------------------------------------------
common=(experiment="$EXPERIMENT" wandb.enable=false seed="$SEED")
smoke=("${common[@]}" smoke=true run_name="${TAG}_smoke" ++ppo.num_timesteps=1e6 torch_warp.ckpt_minutes=1)
step smoke "$OUT/smoke.log" ./run.sh train-warp "${smoke[@]}" torch_warp.max_minutes=2 || failed=1
step resume "$OUT/resume.log" ./run.sh train-warp "${smoke[@]}" torch_warp.max_minutes=4 \
    ++ppo.num_timesteps=2e6 torch_warp.resume=true || failed=1

real=("${common[@]}" run_name="${TAG}_real")
[ -n "$NUM_ENVS" ] && real+=("++ppo.num_envs=$NUM_ENVS")
if [ "$REAL_MINUTES" != "0" ]; then
    step real "$OUT/real.log" ./run.sh train-warp "${real[@]}" torch_warp.ckpt_minutes=5 \
        torch_warp.max_minutes="$REAL_MINUTES" || failed=1
fi

# -- bench -------------------------------------------------------------------
bench=(experiment="$EXPERIMENT" seed="$SEED" bench.out="$OUT/bench.json")
[ -n "$NUM_ENVS" ] && bench+=("++ppo.num_envs=$NUM_ENVS")
step bench "$OUT/bench.log" ./run.sh bench-warp "${bench[@]}" || failed=1

# -- evaluation of the real run ------------------------------------------------
if [ "$REAL_MINUTES" != "0" ] && [ -f "runs/${TAG}_real/run.json" ]; then
    step battery "$OUT/battery.log" ./run.sh battery --run "runs/${TAG}_real" || failed=1
    step video "$OUT/video.log" ./run.sh eval --run "runs/${TAG}_real" || true
fi

finish
