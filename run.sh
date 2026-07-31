#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

case "${1:-}" in
  train) shift; "$PY" -m humanoid_lab.train "$@" ;;
  smoke) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.train smoke=true wandb.enable=false "$@" ;;
  build) shift; "$PY" -m humanoid_lab.build_model "$@" ;;
  check) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.check_model "$@" ;;
  # Warp contact/constraint budget measurement (port item 2.2). Forced onto
  # CPU like `check`: it is a sizing probe, and the jax backend is where the
  # active-contact count is readable without a GPU box. Passthrough args:
  # --robot NAME --preset NAME [--steps N] [--seeds N] [--seed N] [--out path.json].
  check-contacts) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.check_contacts "$@" ;;
  # The split (port item 0.1, tests/unit/test_suite_split.py guards it):
  # `test` is the edit-loop suite -- model-free, runs in seconds. `test-slow`
  # builds models and steps MJX. `test-all` is both, for CI and pre-merge.
  test)  shift; "$PY" -m pytest tests/unit -q "$@" ;;
  # JAX_COMPILATION_CACHE_DIR makes the MJX compiles persist across runs, so
  # a re-run of the slow suite skips the tens of seconds of XLA compilation.
  # The MIN_COMPILE_TIME_SECS=0 override stores every compile, not just the
  # ones over jax's default 1 s threshold.
  test-slow)
    shift
    export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-.jax_cache}"
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
    "$PY" -m pytest tests/integration -q "$@"
    ;;
  test-all)
    shift
    export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-.jax_cache}"
    export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
    "$PY" -m pytest tests/unit tests/integration -q "$@"
    ;;
  # Single-purpose verb: just the checkpoint rollout -> runs/<name>/sizing_data.npz.
  # Passthrough args: --run runs/<name> [--episodes N] [--steps N] [--seed N].
  sizing-collect) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.sizing.collect "$@" ;;
  # PLAN.md build step 7's gate verb: collect (CPU, skipped if runs/<name>/
  # sizing_data.npz already exists, unless --recollect forces a fresh rollout)
  # then report. --run is required; --episodes/--steps/--seed route to
  # collect, --motors to report -- both CLIs use strict argparse (unknown
  # flags error), so this verb splits them itself rather than passing "$@"
  # through to both.
  sizing-report)
    shift
    run="" episodes="" steps="" seed="" motors="" recollect=0
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --run) run="$2"; shift 2 ;;
        --episodes) episodes="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --motors) motors="$2"; shift 2 ;;
        --recollect) recollect=1; shift ;;
        *) echo "sizing-report: unknown arg $1" >&2; exit 1 ;;
      esac
    done
    if [[ -z "$run" ]]; then echo "sizing-report: --run is required" >&2; exit 1; fi

    collect_args=(--run "$run")
    [[ -n "$episodes" ]] && collect_args+=(--episodes "$episodes")
    [[ -n "$steps" ]] && collect_args+=(--steps "$steps")
    [[ -n "$seed" ]] && collect_args+=(--seed "$seed")
    if [[ "$recollect" == "1" || ! -f "$run/sizing_data.npz" ]]; then
      JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.sizing.collect "${collect_args[@]}"
    fi

    report_args=(--run "$run")
    [[ -n "$motors" ]] && report_args+=(--motors "$motors")
    "$PY" -m humanoid_lab.sizing.report "${report_args[@]}"
    ;;
  # PLAN.md build step 10: fixed eval battery -> runs/<name>/battery.json.
  # Passthrough args: --run runs/<name> [--out path.json]
  # [--alpha A] [--lag-tau TAU] [--torque-envelope OMEGA_B,OMEGA_0].
  #
  # The last three are the robustness grid's eval-only plant perturbations
  # (port item 4.4, see src/humanoid_lab/eval/grid.py). Every grid cell
  # passes --out: a perturbed measurement must never overwrite the run's
  # canonical battery.json. Cell filenames come from eval/grid.py's
  # cell_name, which is what `grid-report` aggregates, e.g.
  #   ./run.sh battery --run runs/r --alpha 1.58 --lag-tau 0.005 \
  #     --out runs/r/grid/battery_a1.58_lag5ms_envnone.json
  battery) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.eval.battery "$@" ;;
  # Aggregates runs/<name>/grid/*.json into a markdown table with PASS/FAIL
  # per cell (port item 4.4). Passthrough args:
  # --runs runs/<name> [runs/<other> ...] [--out path.md].
  grid-report) shift; "$PY" -m humanoid_lab.eval.grid_report "$@" ;;
  # Renders runs/<name>/eval_report.md from battery.json (run `battery`
  # first). If runs/<name>/sizing_data.npz also exists, additionally runs
  # sizing-report's report half -- a separate decoupled invocation (not a
  # merged report; eval/report.py never imports sizing/report.py), so
  # `report` gives a one-stop "everything this run has" view without
  # coupling the two report modules together. Passthrough args:
  # --run runs/<name> [--out path.md].
  report)
    shift
    run=""
    args=()
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --run) run="$2"; args+=(--run "$2"); shift 2 ;;
        *) args+=("$1"); shift ;;
      esac
    done
    if [[ -z "$run" ]]; then echo "report: --run is required" >&2; exit 1; fi
    "$PY" -m humanoid_lab.eval.report "${args[@]}"
    if [[ -f "$run/sizing_data.npz" ]]; then
      "$PY" -m humanoid_lab.sizing.report --run "$run"
    fi
    ;;
  # Renders one battery scenario rollout to MP4 (mujoco Renderer,
  # offscreen). darwin: uses the default GL backend (CGL); linux:
  # eval/video.py sets MUJOCO_GL=egl unless already exported -- see its
  # module docstring. Only exercised on darwin so far in this repo; treat
  # the linux/egl path as untested until a GPU-box run confirms it.
  # Rollouts are push-free by default (the battery's measurement
  # convention); --push restores the run's own random pushes.
  # Passthrough args: --run runs/<name> [--scenario name] [--steps N]
  # [--out path.mp4] [--seed N] [--plot-torque] [--plot-joints]
  # [--joint NAME] [--push].
  eval) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.eval.video "$@" ;;
  *)
    echo "usage: run.sh {train|smoke|build|check|check-contacts|test|test-slow|test-all|sizing-collect|sizing-report|battery|grid-report|report|eval} [args]"
    exit 1
    ;;
esac
