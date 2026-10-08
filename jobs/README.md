# jobs/

Payloads for remote training runs. A payload is a plain shell script that
assumes a prepared environment and reads its parameters from environment
variables. It knows about training and knows nothing about the machine it
runs on.

Everything that knows about a particular machine lives in a private ops repo:
scheduler headers, hostnames, storage layout, environment setup, and the
sync. `CLAUDE.local.md` says where that repo is. Nothing in this directory
names a scheduler, a host, or a cluster, and nothing here may grow such a
name.

`train.sh` runs one training. `train_chain.sh` runs two trainings back to
back through `train.sh`, the second restored from the first's latest
checkpoint. `preflight_sizing.sh` runs bounded slices at
several env counts and reports peak GPU memory and steps/s per size, so a
full-budget launch is sized from measurements. `check_terrain.sh` gates a
terrain recipe against MJWarp's contact, CCD and row buffers on each arena
and recommends its budgets. `terrain_scan.sh` scores a trained run on its
robot's terrain scan suite. `docs/terrain.md` gives the order the terrain
payloads run in. Each script's header documents its parameters, their
defaults, and a worked example.

`EXPERIMENT` names a file under `configs/experiment/`. The payloads pass it
as `experiment=<name>`. `+experiment=` fails to compose, because
`configs/config.yaml`'s defaults list already holds `experiment: null`.
`train.sh`, `preflight_sizing.sh` and `check_terrain.sh` pass `ROBOT`,
`TASK` and `ACTUATORS` to Hydra only when set. A set one wins over the
experiment's own pin. `check_terrain.sh` reads no `TASK`. `train_chain.sh`
requires `ROBOT` and defaults `TASK` to `joystick` and `ACTUATORS` to
`sizing_ideal`, so both its phases pass all three.

## The contract

The caller guarantees:

- The checkout is in place and the repo root is the working directory.
- A venv with the training dependencies is active, and the compile caches
  are configured.
- The requested GPUs are visible.
- `XLA_CLIENT_MEM_FRACTION` is unset. `humanoid_lab.train` sets
  `XLA_PYTHON_CLIENT_MEM_FRACTION`, and jaxlib 0.9.2 raises when both are
  set. `humanoid_lab.check_terrain` and `humanoid_lab.eval.terrain_scan`
  import `humanoid_lab.train`, so they set it too.
- stdout and stderr are captured, and a sentinel exit file is written when
  the payload exits.

A payload promises:

- It runs from the repo root and writes its outputs only under `runs/`.
- It reads its parameters only from environment variables. Its header
  documents each one, its default, and a worked example.
- It names no scheduler and no machine. No scheduler environment variables,
  no ssh, no rsync, no hardcoded hostnames. Printing the host it landed on is
  fine, and `train.sh` does it.
- It exits nonzero on failure, and its header states its partial-failure
  policy.

## Measuring every model

Every trained run gets the standard evals: `courses.json`, `battery.json`
and `eval_report.md`. `eval_runs.sh` runs them on CPU for the runs named in
`RUNS`. It needs no GPU.

`train.sh` measures the run it trains whenever the training call returns,
whatever its exit code. `EVAL=false` turns that off. When the training
exits 0 and its evals then fail or time out, `train.sh` exits 75.
`train_chain.sh` exits 75 when phase B succeeds after a phase A that exited
75.

A payload killed with its process group cannot measure its own run. A
deadline stop kills it that way. The caller therefore has one more duty.
After every training payload ends, by any path, it runs
`RUNS=<runs> ./jobs/eval_runs.sh` for every run dir the payload trained.
For `train_chain.sh` those are `<RUN_NAME>_a` and `<RUN_NAME>`. A phase B
that never started has no `run.json`. It is SKIPPED, and a SKIPPED run makes
the call exit 1. So the caller names only the run dirs that exist. A kill
during `train.sh`'s eval stage ends the stage too, so no eval of the payload
outlives it to race that call. The call also completes a run after a 75.
The run needs a `run.json` first. When the trainer died before writing one,
the caller writes it. When `train.sh` already measured the run, the call
finds every file current and exits in seconds.

`RUNS=all CHECK=true ./jobs/eval_runs.sh` is the audit. It runs no eval. It
lists every run whose `courses.json` is missing or not current, and every
`battery.json` that is not for its run's newest checkpoint. It also lists
every run dir with checkpoints but no `run.json`. It exits 1 when it lists
anything. `RUNS=all ./jobs/eval_runs.sh` measures every run that lacks
current results.

`eval_runs.sh` and the eval stage in `train.sh` write only under `runs/`.
Neither submits a job.
