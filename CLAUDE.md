# humanoid-lab

MuJoCo MJX + Brax PPO locomotion training for the Asimov v1 and Roboto Origin
humanoids. Configuration is Hydra: `configs/` holds robot, task, actuators, dr,
network and experiment groups.

## Local commands

`./run.sh` wraps `.venv/bin/python` and takes a verb:

| verb | what it does |
|---|---|
| `train` | Hydra training run |
| `smoke` | short CPU training run, wandb off |
| `build` / `check` | build and check a robot's MJX model |
| `test` | `pytest tests -q` |
| `battery` / `report` / `eval` | eval battery, report, video |
| `sizing-collect` / `sizing-report` | actuator sizing rollout and report |

Overrides pass straight through, e.g. `./run.sh train ppo.num_timesteps=3e8 run_name=asimov_v1`.

## Cluster

Training runs on HPC (remote host `<hpc-host>`, `<gpu-partition>` partition, multi-GPU
nodes). Per-person paths live in untracked local config.

- `./hpc.sh [command]` runs a command on the remote host, or opens a shell
  with no arguments. It resolves the ssh destination from `make -s hpc-dest`
  and uses `ssh -o BatchMode=yes`, which fails fast instead of hanging on a
  password prompt. Use it rather than reading the config yourself: the
  per-person files are gitignored and agent sessions cannot read them.
- `make push` / `make pull` rsync the tree and bring `runs/` and `logs/` back.
  `make queue` and `make logs JOB=<id>` wrap queue-status and log tails.
- `hpc/local.mk` (gitignored) holds your tree name as `HPC_TREE = <name>`.
  `hpc/local.env` (gitignored, cluster-side) holds `STORE_DIR` and the other
  per-person paths that batch jobs read; `hpc/local.env.example` is the
  template.
- `hpc/train.job` is the job and `hpc/_common.sh` the shared setup.
  `hpc/README.md` covers the details.

Nobody runs `submit` without Marcin's explicit go. This is a standing rule
carried over from w01-tek.

## Facts that cost time to learn

- `warp-lang==1.13.0` is a mandatory pin. Newer versions break
  `mujoco-mjx`'s warp backend on this cluster.
- The venv lives on remote storage (`STORE_DIR`), not `$HOME`.
- steps/s is comparable only between runs at the same seed. Two w01-tek runs
  differing only in seed reported 1,343,166 and 777,859 steps/s.
