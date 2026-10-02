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

`train.sh` runs one training. `preflight_sizing.sh` runs bounded slices at
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
`ROBOT`, `TASK` and `ACTUATORS` reach Hydra only when set. A set one wins
over the experiment's own pin.

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
