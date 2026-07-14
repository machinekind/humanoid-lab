#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

case "${1:-}" in
  train) shift; "$PY" -m humanoid_lab.train "$@" ;;
  smoke) shift; JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.train smoke=true wandb.enable=false "$@" ;;
  test)  shift; "$PY" -m pytest tests -q "$@" ;;
  *) echo "usage: run.sh {train|smoke|test} [args]"; exit 1 ;;
esac
