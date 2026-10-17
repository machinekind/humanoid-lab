#!/usr/bin/env bash
# Renders a CPU clip from each new complete checkpoint of a run while the run
# trains.
#
# This is a payload. jobs/README.md states the contract it is written
# against. Whatever launches it has already put the checkout in place and
# made the repo root the working directory. It runs next to a training and
# needs no GPU. Each clip comes from humanoid_lab.eval.video on the JAX env
# with JAX_PLATFORMS=cpu. It runs in bash 3.2 and in Git Bash on Windows.
#
# Parameters:
#   RUN_NAME   run dir name under runs/   (REQUIRED, e.g. roboto_walk_v1)
#   PY         the python that has humanoid_lab installed
#              (REQUIRED, no default, e.g. .venv/bin/python)
#   INTERVAL   seconds between checks     (default 600)
#   SCENARIO   eval.video scenario        (default walk_ramp)
#   MAX_IDLE   seconds without a new checkpoint before it gives up
#              (default 7200, 0 means no limit)
#
# A watcher that gives up after an hour without a new checkpoint:
#   RUN_NAME=roboto_walk_v1 PY=.venv/bin/python MAX_IDLE=3600 bash jobs/video_watch.sh
#
# Each check prints one line:
#   VIDEO step=<N> path=<mp4> [log fields]  a clip rendered. The log fields
#                                           are reward_x1000, falls_per_1M and
#                                           sps from the last train_log.jsonl
#                                           line, each one only when present.
#   FAIL step=<N> log=<log>                 the render failed. Its log says why.
#   NORUN step=<N>                          a new checkpoint exists but
#                                           run.json does not yet.
#   NOCKPT                                  no new complete checkpoint.
#   DONE                                    the trainer wrote DONE and its last
#                                           checkpoint has been handled.
#   IDLE seconds=<MAX_IDLE>                 no new checkpoint for MAX_IDLE s.
#
# Outputs: runs/$RUN_NAME/videos/step_<12-digit step>.mp4, with the render's
# output in a .log of the same name next to each clip.
#
# Each clip is pinned to the step it is named after through eval.video's
# --checkpoint flag. A checkpoint is complete once its step dir holds
# ppo_network_config.json, which is written last.
#
# Nothing renders before runs/$RUN_NAME/run.json exists, because eval.video
# reads the run's config from it. The Torch trainer writes run.json at start,
# so its runs get clips while they train. The Brax trainer writes run.json at
# the end, so a Brax run gets its clip once training finishes. Brax writes no
# DONE, so MAX_IDLE ends a watcher on a Brax run.
#
# Partial-failure policy: a failed render prints FAIL with its log path and
# the loop goes on. A failed step is not retried. The script exits 1 at the
# end when any render failed, and 0 otherwise.

set -uo pipefail
# No -e: the loop must survive a failed render.

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

: "${RUN_NAME:?set RUN_NAME to a run dir name under runs/, e.g. roboto_walk_v1}"
: "${PY:?set PY to the python that has humanoid_lab installed, e.g. .venv/bin/python}"
: "${INTERVAL:=600}"
: "${SCENARIO:=walk_ramp}"
: "${MAX_IDLE:=7200}"

# Outputs stay under runs/.
case "$RUN_NAME" in
    */* | ..*)
        echo "ERROR: RUN_NAME must be a plain run dir name; got '$RUN_NAME'" >&2
        exit 1
        ;;
esac
RUN="runs/$RUN_NAME"

# Largest numeric step dir that holds ppo_network_config.json, or -1.
latest_step() {
    local best=-1 d name step
    for d in "$RUN"/checkpoints/*/; do
        [ -d "$d" ] || continue
        d="${d%/}"
        name="${d##*/}"
        case "$name" in '' | *[!0-9]*) continue ;; esac
        [ -f "$d/ppo_network_config.json" ] || continue
        step=$((10#$name))
        [ "$step" -gt "$best" ] && best=$step
    done
    echo "$best"
}

# key=value for each of reward_x1000, falls_per_1M and sps in the last
# train_log.jsonl line. Prints nothing when the file is missing or unreadable.
last_log() {
    [ -r "$RUN/train_log.jsonl" ] || return 0
    tail -n 1 "$RUN/train_log.jsonl" 2>/dev/null | "$PY" -c "import json, sys
try:
    r = json.loads(sys.stdin.read())
except Exception:
    sys.exit(0)
if isinstance(r, dict):
    print(' '.join(f'{k}={r[k]}' for k in ('reward_x1000', 'falls_per_1M', 'sps') if k in r))
" 2>/dev/null
}

rendered=-1
seen=-1
failed=0
idle_since=$SECONDS

while true; do
    step="$(latest_step)"
    if [ "$step" -gt "$seen" ]; then
        # A new checkpoint resets the idle clock even when nothing renders.
        seen=$step
        idle_since=$SECONDS
    fi

    if [ "$step" -gt "$rendered" ] && [ -f "$RUN/run.json" ]; then
        mkdir -p "$RUN/videos"
        out="$RUN/videos/step_$(printf %012d "$step").mp4"
        log="${out%.mp4}.log"
        if JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.eval.video --run "$RUN" --checkpoint "$step" \
            --scenario "$SCENARIO" --out "$out" >"$log" 2>&1; then
            fields="$(last_log)"
            echo "VIDEO step=$step path=$out${fields:+ $fields}"
        else
            echo "FAIL step=$step log=$log"
            failed=$((failed + 1))
        fi
        rendered=$step
    elif [ -f "$RUN/DONE" ] && [ "$step" -le "$rendered" ]; then
        echo "DONE"
        break
    elif [ "$step" -gt "$rendered" ]; then
        echo "NORUN step=$step"
    else
        echo "NOCKPT"
    fi

    # Checked after the branches, so a run that never writes run.json still ends.
    if [ "$MAX_IDLE" -gt 0 ] && [ $((SECONDS - idle_since)) -ge "$MAX_IDLE" ]; then
        echo "IDLE seconds=$MAX_IDLE"
        break
    fi
    sleep "$INTERVAL"
done

if [ "$failed" -gt 0 ]; then
    exit 1
fi
exit 0
