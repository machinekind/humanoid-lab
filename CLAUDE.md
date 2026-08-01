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
| `check-contacts` | measure the warp contact and constraint budgets a preset needs |
| `test` | `pytest tests/unit -q` — model-free, seconds, the edit loop |
| `test-slow` / `test-all` | `tests/integration` (builds and steps MJX) / both, before a merge |
| `battery` / `grid-report` / `report` / `eval` | eval battery, robustness-grid table, report, video |
| `sizing-collect` / `sizing-report` | actuator sizing rollout and report |
| `export` | deploy artifacts (`policy.npz`, `policy_meta.json`) from a checkpoint |

Overrides pass straight through, e.g. `./run.sh train ppo.num_timesteps=3e8 run_name=asimov_v1`.

docs/configuration.md's `run.sh` verbs table is the authoritative one and
lists every flag each verb takes.

## Remote training

Remote training execution lives in a private ops repo; see CLAUDE.local.md
(gitignored) for where it is. `jobs/` holds the payloads that repo calls, and
`jobs/README.md` states the contract between the two.

Remote job submission is a human-authorized action. Agents never submit on
their own.

## Facts that cost time to learn

- `warp-lang==1.13.0` is a mandatory pin. Newer versions break
  `mujoco-mjx`'s warp backend on the training hosts.
- steps/s is comparable only between runs at the same seed. Two runs differing
  only in seed reported 1,343,166 and 777,859 steps/s.
- Memory ceilings and throughput do not transfer between node classes. Measure
  on the hardware the real run will use.
