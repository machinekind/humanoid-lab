#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

case "${1:-}" in
  train) shift; "$PY" -m humanoid_lab.train "$@" ;;
  smoke) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.train smoke=true wandb.enable=false "$@" ;;
  build) shift; "$PY" -m humanoid_lab.build_model "$@" ;;
  check) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.check_model "$@" ;;
  test)  shift; "$PY" -m pytest tests -q "$@" ;;
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
  # Passthrough args: --run runs/<name> [--out path.json].
  battery) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.eval.battery "$@" ;;
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
  # Passthrough args: --run runs/<name> [--scenario name] [--steps N] [--out path.mp4].
  eval) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.eval.video "$@" ;;
  *)
    echo "usage: run.sh {train|smoke|build|check|test|sizing-collect|sizing-report|battery|report|eval} [args]"
    exit 1
    ;;
esac
