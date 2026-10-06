#!/usr/bin/env bash
# The standard evals of trained runs: the course benchmark, the battery and
# the report. jobs/train.sh runs this on the run it trains (EVAL). The
# caller runs it after every training payload ends, by any path, on each run
# the payload trained: a payload killed with its process group cannot
# measure its own run (jobs/README.md). Run it directly for runs that
# predate that, after a catalogue change, for a replicate at another seed
# base, or with CHECK=true to list what is not measured.
#
# This is a payload. jobs/README.md states the contract it is written
# against. The caller has put the checkout in place, made the repo root the
# working directory and activated the venv. Every eval here runs on CPU
# (JAX_PLATFORMS=cpu on each call), where course lanes reproduce bit for
# bit and their seed noise bands are measured. No GPU is needed.
#
# A run is measurable when runs/<run>/run.json exists and the largest
# numeric step dir under runs/<run>/checkpoints/ holds
# ppo_network_config.json, which brax writes last in a save. train.py
# writes run.json only when training returns or stops early, so a
# still-running training is SKIPPED. A killed run is measured on its newest
# checkpoint once its caller has written a run.json. A status field in
# run.json, when present, is not read.
#
# Parameters:
#   RUNS       run dir names under runs/, space separated, or all for every
#              runs/*/ holding a run.json (REQUIRED). A leading runs/ is
#              dropped. A name is one dir name: no '/', not all, '.' or '..'
#   EVALS      evals to run, in order        (default "courses battery report")
#   TAG        empty: each run's own courses.json, battery.json and
#              eval_report.md. Set: runs/<run>/eval/<TAG>/courses.json, and
#              EVALS must be courses          (default empty)
#   SEEDS      course seeds per row; other values need TAG (default 8)
#   SEED_BASE  first course seed; nonzero needs TAG       (default 0)
#   WORKERS    course lanes run at once; never changes a number
#              (default the courses CLI's: min(8, CPUs))
#   REDO       true re-runs evals whose output is already current
#              (default false)
#   CHECK      true runs no eval. It lists every run whose courses.json is
#              missing or not current or whose battery.json is not for its
#              newest checkpoint, and exits 1 when it lists any. With
#              RUNS=all it also lists every runs/*/ with a complete
#              checkpoint but no run.json (default false)
#   MUJOCO_GL  (default disabled) nothing here renders
#
# Current. The courses CLI decides whether a courses.json is current
# (--skip-if-current, --check): the same catalogue, checkpoint files, seeds
# and rows. battery.json is current when its "checkpoint" names the newest
# step dir and it is not older than that dir's ppo_network_config.json. The
# report loads no model and is rendered on every pass. CHECK audits the
# courses and battery words of EVALS, and with TAG set the tagged file.
#
# Two runs:
#   RUNS="walk_a walk_b" ./jobs/eval_runs.sh
# After a training payload ends, by any path, for each run it trained (both
# phases of a chain):
#   RUNS=walk_a ./jobs/eval_runs.sh
# One replicate of the course scores, at seed base 8, beside the canonical
# one. A noise band takes bases 8, 16 and 24 (docs/configuration.md):
#   RUNS=walk_a SEED_BASE=8 TAG=seeds8 EVALS=courses ./jobs/eval_runs.sh
# Every run on this host that lacks current results, then the audit:
#   RUNS=all ./jobs/eval_runs.sh
#   RUNS=all CHECK=true ./jobs/eval_runs.sh
#
# Partial-failure policy: every run is attempted, and one run's failure does
# not stop the next. A run named in RUNS with no run.json, or with no
# complete newest checkpoint, is SKIPPED and counts as a failure. The last
# line lists measured, skipped and failed runs. The exit code is 0 only
# when every eval of every run succeeded or was already current, otherwise
# 1. With CHECK=true it is 0 only when nothing was listed. A bad parameter
# exits 1 before any eval runs. A SIGTERM, SIGINT or SIGHUP ends the running
# eval before the script exits, with 143, 130 or 129.

set -euo pipefail

if [ ! -f pyproject.toml ] || [ ! -d configs ]; then
    echo "ERROR: run this from the repo root; '$PWD' has no pyproject.toml and configs/" >&2
    exit 1
fi

: "${RUNS:?set RUNS to run dir names under runs/, or all}"
EVALS="${EVALS:-courses battery report}"
TAG="${TAG:-}"
SEEDS="${SEEDS:-8}"
SEED_BASE="${SEED_BASE:-0}"
WORKERS="${WORKERS:-}"
REDO="${REDO:-false}"
CHECK="${CHECK:-false}"
MUJOCO_GL="${MUJOCO_GL:-disabled}"
export MUJOCO_GL

bad() {
    echo "ERROR: $1" >&2
    exit 1
}

# read -a splits on whitespace without expanding globs.
read -r -a eval_words <<<"$EVALS"
[ "${#eval_words[@]}" -gt 0 ] || bad "EVALS is empty; name courses, battery or report"
for w in "${eval_words[@]}"; do
    case "$w" in
        courses | battery | report) ;;
        *) bad "EVALS: '$w' is not courses, battery or report" ;;
    esac
done
case "$REDO" in true | false) ;; *) bad "REDO must be true or false, got '$REDO'" ;; esac
case "$CHECK" in true | false) ;; *) bad "CHECK must be true or false, got '$CHECK'" ;; esac
case "$SEEDS" in '' | *[!0-9]* | 0*) bad "SEEDS must be a positive whole number, got '$SEEDS'" ;; esac
case "$SEED_BASE" in '' | *[!0-9]* | 0?*) bad "SEED_BASE must be a whole number, got '$SEED_BASE'" ;; esac
case "$WORKERS" in *[!0-9]* | 0*) bad "WORKERS must be a positive whole number, got '$WORKERS'" ;; esac
if [ -n "$TAG" ]; then
    case "$TAG" in
        . | .. | *[!A-Za-z0-9._-]*) bad "TAG '$TAG' must be one dir name of letters, digits, '.', '_' and '-'" ;;
    esac
    # The battery has no seed knob, and the report renders the run's own
    # files, so a tagged pass measures courses only.
    [ "${eval_words[*]}" = courses ] || bad "TAG=$TAG needs EVALS=courses, got '$EVALS'"
elif [ "$SEEDS" != 8 ] || [ "$SEED_BASE" != 0 ]; then
    # The run's own courses.json is the full measurement: 8 seeds from seed
    # 0, the courses CLI's defaults. Anything else goes to a tagged file.
    bad "SEEDS=$SEEDS SEED_BASE=$SEED_BASE needs a TAG: the run's own courses.json is 8 seeds from seed 0"
fi

read -r -a run_words <<<"$RUNS"
[ "${#run_words[@]}" -gt 0 ] || bad "RUNS is empty"
members=()
if [ "${run_words[*]}" = all ]; then
    for j in runs/*/run.json; do
        [ -f "$j" ] || continue
        j="${j%/run.json}"
        members+=("${j#runs/}")
    done
else
    for w in "${run_words[@]}"; do
        n="${w#runs/}"
        n="${n%/}"
        case "$n" in
            '' | . | .. | */* | all) bad "RUNS: '$w' is not one run dir name under runs/ (all stands alone)" ;;
        esac
        case " ${members[*]:-} " in
            *" $n "*) ;;
            *) members+=("$n") ;;
        esac
    done
fi

# Prints the largest numeric step dir under $1/checkpoints/, the one the
# battery and the courses CLI measure. Returns 1 when there is none. 10#
# reads a zero-padded name as decimal, not octal.
newest_step() {
    local best="" best_step=-1 d name step
    for d in "$1"/checkpoints/*/; do
        [ -d "$d" ] || continue
        d="${d%/}"
        name="${d##*/}"
        case "$name" in
            '' | *[!0-9]*) continue ;;
        esac
        step=$((10#$name))
        if [ "$step" -gt "$best_step" ]; then
            best_step="$step"
            best="$d"
        fi
    done
    [ -n "$best" ] || return 1
    printf '%s\n' "$best"
}

# Returns 0 when $1/checkpoints/ holds any complete numeric step dir.
has_complete_step() {
    local d name
    for d in "$1"/checkpoints/*/; do
        d="${d%/}"
        name="${d##*/}"
        case "$name" in
            '' | *[!0-9]*) continue ;;
        esac
        [ -f "$d/ppo_network_config.json" ] && return 0
    done
    return 1
}

# Prints why runs/$1/battery.json is not for the step dir $2, and nothing
# when it is. battery.json is written with indent=2, so its top-level
# "checkpoint" key sits on its own line after two spaces. The mtime test
# catches a reused run dir whose new training wrote the same step names.
battery_stale() {
    local b="runs/$1/battery.json" have
    if [ ! -f "$b" ]; then
        echo "$b does not exist"
        return 0
    fi
    have="$(sed -n 's/^  "checkpoint": "\([^"]*\)".*$/\1/p' "$b" | head -n 1)"
    if [ "$have" != "${2##*/}" ]; then
        echo "$b is for checkpoint '${have}', not ${2##*/}"
    elif [ "$b" -ot "$2/ppo_network_config.json" ]; then
        echo "$b is older than $2/ppo_network_config.json"
    fi
}

# The courses CLI's arguments for run $1, before any mode flag.
courses_args() {
    args=(-m humanoid_lab.eval.courses --run "runs/$1" --ground flat
          --seeds "$SEEDS" --seed-base "$SEED_BASE")
    if [ -n "$TAG" ]; then
        args+=(--out "runs/$1/eval/$TAG/courses.json")
    fi
}

# An eval runs in the background and the script waits for it. A trapped
# signal interrupts that wait at once, where a foreground call would delay
# the trap until the eval returned. The trap ends the eval and waits for it,
# so no eval outlives the script. bash starts a background command with
# SIGINT ignored when job control is off, so the trap sends SIGTERM.
child=""
stop() {
    if [ -n "$child" ]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit "$1"
}
trap 'stop 143' TERM
trap 'stop 130' INT
trap 'stop 129' HUP

# run_eval <eval> <run> <python3 args...>: one eval call, on CPU. A failure
# is recorded and the next eval runs.
run_eval() {
    local what="$1" m="$2" rc=0
    shift 2
    echo "== $what runs/$m =="
    JAX_PLATFORMS=cpu python3 "$@" &
    child=$!
    wait "$child" || rc=$?
    child=""
    if [ "$rc" -ne 0 ]; then
        echo "WARN: $what for $m failed rc=$rc; continuing" >&2
        failed+=("$m:$what")
    fi
}

# check_run <run> <newest step dir>: prints a STALE line for each audited
# output of run <run> that is not current. Returns 1 when it printed any.
check_run() {
    local m="$1" ckpt="$2" w why rc listed=0
    for w in "${eval_words[@]}"; do
        case "$w" in
            courses)
                courses_args "$m"
                rc=0
                why="$(JAX_PLATFORMS=cpu python3 "${args[@]}" --check)" || rc=$?
                if [ "$rc" -eq 3 ]; then
                    echo "STALE: runs/$m: ${why##*$'\n'}"
                    listed=1
                elif [ "$rc" -ne 0 ]; then
                    echo "STALE: runs/$m: courses --check failed rc=$rc"
                    listed=1
                fi ;;
            battery)
                why="$(battery_stale "$m" "$ckpt")"
                if [ -n "$why" ]; then
                    echo "STALE: runs/$m: $why"
                    listed=1
                fi ;;
        esac
    done
    [ "$listed" -eq 0 ]
}

measured=()
skipped=()
failed=()
current=()
stale=()
unmeasurable=()
for m in ${members[@]:+"${members[@]}"}; do
    echo "== $(date -Iseconds) runs/$m =="
    if [ ! -f "runs/$m/run.json" ]; then
        echo "SKIPPED: runs/$m has no run.json"
        skipped+=("$m")
        continue
    fi
    if ! ckpt="$(newest_step "runs/$m")" || [ ! -f "$ckpt/ppo_network_config.json" ]; then
        echo "SKIPPED: runs/$m has no complete newest checkpoint"
        skipped+=("$m")
        continue
    fi

    if [ "$CHECK" = true ]; then
        if check_run "$m" "$ckpt"; then
            current+=("$m")
            echo "CURRENT: runs/$m"
        else
            stale+=("$m")
        fi
        continue
    fi

    n_failed="${#failed[@]}"
    for w in "${eval_words[@]}"; do
        case "$w" in
            courses)
                courses_args "$m"
                if [ -n "$WORKERS" ]; then
                    args+=(--workers "$WORKERS")
                fi
                if [ "$REDO" = false ]; then
                    args+=(--skip-if-current)
                fi
                run_eval courses "$m" "${args[@]}" ;;
            battery)
                why="$(battery_stale "$m" "$ckpt")"
                if [ "$REDO" = false ] && [ -z "$why" ]; then
                    echo "== battery runs/$m: battery.json is current for ${ckpt##*/}; skipped =="
                else
                    run_eval battery "$m" -m humanoid_lab.eval.battery --run "runs/$m"
                fi ;;
            report)
                run_eval report "$m" -m humanoid_lab.eval.report --run "runs/$m" ;;
        esac
    done
    if [ "${#failed[@]}" -eq "$n_failed" ]; then
        measured+=("$m")
    fi
done

if [ "$CHECK" = true ]; then
    if [ "${run_words[*]}" = all ]; then
        for d in runs/*/; do
            d="${d%/}"
            [ -d "$d" ] && [ ! -f "$d/run.json" ] || continue
            if has_complete_step "$d"; then
                echo "UNMEASURABLE: $d has checkpoints but no run.json"
                unmeasurable+=("${d#runs/}")
            fi
        done
    fi
    echo "== EVAL_RUNS CHECK: current [${current[*]:-}], stale [${stale[*]:-}]," \
        "skipped [${skipped[*]:-}], unmeasurable [${unmeasurable[*]:-}] =="
    [ "${#stale[@]}" -eq 0 ] && [ "${#skipped[@]}" -eq 0 ] && [ "${#unmeasurable[@]}" -eq 0 ] || exit 1
    exit 0
fi

echo "== EVAL_RUNS SUMMARY: measured [${measured[*]:-}], skipped [${skipped[*]:-}]," \
    "failed [${failed[*]:-}] =="
[ "${#skipped[@]}" -eq 0 ] && [ "${#failed[@]}" -eq 0 ] || exit 1
exit 0
