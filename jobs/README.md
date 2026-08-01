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
full-budget launch is sized from measurements. Each script's header documents
its parameters, their defaults, and a worked example.

## The contract

The caller guarantees:

- The checkout is in place and the repo root is the working directory.
- A venv with the training dependencies is active, and the compile caches
  are configured.
- The requested GPUs are visible.
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

Remote job submission is a human-authorized action. Agents never submit on
their own.
