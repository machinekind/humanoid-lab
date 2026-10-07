#!/usr/bin/env bash
# Every INTERVAL seconds: if the run has a newer complete checkpoint, render
# a clip from it on the CPU (humanoid_lab.eval.video, the original JAX env)
# and print one line per event:
#   VIDEO step=<N> path=<mp4> reward_x1000=<last logged> falls_per_1M=<...>
#   NOCKPT          (no new complete checkpoint since the last clip)
#   FAIL step=<N>   (render failed; log next to the clip)
#   DONE            (the trainer wrote DONE and its last checkpoint has a clip)
# Usage, from the repo root:
#   RUN=runs/loco_warp_v1 INTERVAL=600 bash wtrain/video_watch.sh
set -uo pipefail
: "${RUN:?set RUN=runs/<name>}"
INTERVAL="${INTERVAL:-600}"
SCENARIO="${SCENARIO:-walk_ramp}"
PY="${PY:-/d/hlv/Scripts/python.exe}"
mkdir -p "$RUN/videos"
last=-1

# Largest numeric step dir with ppo_network_config.json (written last, so it
# marks a complete save) -- jobs/train_chain.sh's latest_checkpoint rule.
latest_step() {
    local best=-1 d name step
    for d in "$RUN"/checkpoints/*/; do
        [ -d "$d" ] || continue
        d="${d%/}"; name="${d##*/}"
        case "$name" in '' | *[!0-9]*) continue ;; esac
        [ -f "$d/ppo_network_config.json" ] || continue
        step=$((10#$name))
        [ "$step" -gt "$best" ] && best=$step
    done
    echo "$best"
}

last_log() {
    tail -n 1 "$RUN/train_log.jsonl" 2>/dev/null | "$PY" -c "import sys,json
try:
    r=json.loads(sys.stdin.read()); print(f\"reward_x1000={r['reward_x1000']} falls_per_1M={r['falls_per_1M']} sps={r['sps']}\")
except Exception: print('reward_x1000=- falls_per_1M=-')"
}

while true; do
    step="$(latest_step)"
    if [ "$step" -gt "$last" ]; then
        out="$RUN/videos/step_$(printf %012d "$step").mp4"
        if JAX_PLATFORMS=cpu "$PY" -m humanoid_lab.eval.video --run "$RUN" --scenario "$SCENARIO" \
            --out "$out" >"${out%.mp4}.log" 2>&1; then
            echo "VIDEO step=$step path=$out $(last_log)"
        else
            echo "FAIL step=$step"
        fi
        last=$step
    elif [ -f "$RUN/DONE" ]; then
        echo "DONE"
        exit 0
    else
        echo "NOCKPT"
    fi
    sleep "$INTERVAL"
done
