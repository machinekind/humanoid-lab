# Training on Windows

This runbook takes a native Windows machine with an NVIDIA GPU from a fresh
clone to a trained Roboto Origin policy. The trainer is the Torch+MJWarp
trainer in `src/humanoid_lab/torch_warp/`. Its
[README](../src/humanoid_lab/torch_warp/README.md) holds the validated
versions, the refusal ledger, the checkpoint format and the parity
tolerances.

## What runs on Windows

On Windows, pip installs the CPU build of jax 0.9.2. JAX never touches the
GPU there.

| Runs | Does not run |
|---|---|
| `train-warp` and `bench-warp` on the GPU | Brax training on the GPU. `./run.sh train` runs on CPU JAX only. |
| `test`, `test-parity` and `test-parity --parity-device cuda:0` | The goldens and reference fixtures. They are darwin arm64 bit patterns and skip. `tests/parity` is the check. |
| `build`, `check`, `check-contacts` | The sizing task in the Torch trainer. `sizing-collect` stays a CPU JAX rollout. |
| `battery`, `report`, `eval`, `export`, `sizing-collect` on CPU JAX | `export` of a policy with `heading_err` on its actor list, which includes `roboto_gait_v40`. |

`eval` sets `MUJOCO_GL=egl` only on Linux. On Windows it uses MuJoCo's
default GL. The video path has been exercised only on darwin.

## Prerequisites

| Item | Requirement |
|---|---|
| GPU and driver | An NVIDIA GPU. The driver must meet the minimum of the validated torch build and of warp 1.13. The README's "Validated versions" table names it. |
| Python | 3.11 or 3.12. `pyproject.toml` requires `>=3.11,<3.13`. |
| Shells | PowerShell for install, training and Python entry points. Git Bash, from Git for Windows, for `run.sh` and `jobs/*.sh`. |
| Line endings | `.gitattributes` keeps every `.sh` at LF. A clone made before that file existed checks `run.sh` out with CRLF, and bash fails on it. Fix it with `git add --renormalize .` or a fresh clone. |
| VRAM and disk | The repo states no figure. Read `memory` in `train_log.jsonl` or in the bench JSON. Memory and throughput do not transfer between GPUs, so measure on the card the real run uses. |

## Install

In PowerShell, from the repo root:

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch==<V> --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -e ".[torch_warp,dev]"
.venv\Scripts\python -c "import torch, warp; print(torch.cuda.is_available(), warp.is_cuda_available())"
```

- `<V>` is the torch version in the README's "Validated versions" table. The
  cu126 index is an example until the validated build is named.
- Install torch first. PyPI's Windows torch is a CPU-only build, and an
  extra cannot name an index. Pip then keeps the CUDA build, because
  `torch==<V>` matches `<V>+cu126`.
- The extra install resolves jax and jaxlib 0.9.2 (CPU), mujoco and
  mujoco-mjx 3.10.0, and warp-lang 1.13.0, which is a CUDA-enabled wheel.
  warp-lang 1.13.0 is a mandatory pin. mujoco-mjx 3.10.0 vendors the MJWarp
  that needs exactly that version.
- The last line must print `True True`.
- The install must be editable. The trainer finds `configs/` relative to its
  source file, and a built wheel ships no `configs/`.

## Commands in each shell

`run.sh` runs in Git Bash and picks `.venv/Scripts/python.exe` when
`.venv/bin/python` is missing. In PowerShell, call the module directly. JAX
is CPU-only on Windows, so the `JAX_PLATFORMS=cpu` that `run.sh` sets
changes nothing there.

| Git Bash | PowerShell |
|---|---|
| `./run.sh test` | `.venv\Scripts\python -m pytest tests/unit -q` |
| `./run.sh test-parity` | `.venv\Scripts\python -m pytest tests/parity -q` |
| `./run.sh train-warp <overrides>` | `.venv\Scripts\python -m humanoid_lab.torch_warp.ppo <overrides>` |
| `./run.sh bench-warp <overrides>` | `.venv\Scripts\python -m humanoid_lab.torch_warp.bench <overrides>` |
| `./run.sh battery --run runs/<name>` | `.venv\Scripts\python -m humanoid_lab.eval.battery --run runs/<name>` |
| `./run.sh eval --run runs/<name>` | `.venv\Scripts\python -m humanoid_lab.eval.video --run runs/<name>` |
| `./run.sh export --run runs/<name>` | `.venv\Scripts\python -m humanoid_lab.export.policy --run runs/<name>` |

`./run.sh report` also runs the sizing report when `runs/<name>/sizing_data.npz`
exists. Its PowerShell form is `.venv\Scripts\python -m humanoid_lab.eval.report --run runs/<name>`.

Overrides are Hydra overrides. A key the yaml already sets takes a plain
override, such as `torch_warp.max_minutes=600` or `ppo.sym_coef=1.0`. A key
absent from the yaml needs `++`, such as `++ppo.num_timesteps=3e8` and
`++ppo.num_envs=1024`. A new key under `actuators.overrides` takes `+`.

## Smoke checks

Run these in order on a fresh clone before a real run. They are also the
Windows acceptance checklist for PR #11.

1. The install one-liner prints `True True`.
2. `./run.sh test`. Then `./run.sh test-parity --parity-report`, which
   checks the CPU tolerances. Then
   `./run.sh test-parity --parity-device cuda:0 --parity-report`, which runs
   the Torch side of the task-math, lockstep, physics, DR, RNG and
   closed-loop tests on the GPU.
3. A smoke run on the GPU. `smoke=true` sets 64 envs, caps episodes at 200
   steps and skips the contact probe. `++ppo.num_timesteps=1e6` raises its
   100k-step budget so the run outlasts one checkpoint interval:
   ```
   ./run.sh train-warp experiment=roboto_gait_v40 smoke=true wandb.enable=false run_name=smoke_w1 ++ppo.num_timesteps=1e6 torch_warp.ckpt_minutes=1
   ```
   Run it once from PowerShell and once from Git Bash, with different
   `run_name`s. Let it pass one wall-clock checkpoint, then press Ctrl+C.
   Ctrl+C writes a final checkpoint and run.json with `early_stopped: true`.
4. Physics moves under CUDA graph capture. A `train_log.jsonl` line within
   the first episode shows a nonzero `done_ep_len` or `falls_per_1M`. An
   empty capture raises "captured physics graph did nothing".
5. Resume the smoke run. Repeat its overrides and add
   `torch_warp.resume=true`. `++ppo.num_timesteps=2e6` extends it.
6. Warm start from a Brax checkpoint with
   `restore=runs/<brax_run>/checkpoints/<step>` and a new `run_name`. The
   trainer prints the `task.env`, `actuators` and `ppo` leaves that differ
   from the source run.
7. Run battery, `eval --scenario long`, `eval --compare`,
   `eval --checkpoint <step>` and export on the result. See
   [After training](#after-training).
8. Run `jobs/video_watch.sh` against a training run. See
   [Watching a run](#watching-a-run).
9. `./run.sh bench-warp experiment=roboto_gait_v40`. Keep the JSON.

## A real run

The preset `configs/experiment/roboto_gait_v40.yaml` restates the v40 gait
recipe from [roboto-gait.md](lessons/roboto-gait.md). It selects
`roboto_origin`, the `deploy_pd` actuators and the `large` network. It sets
no `ppo.num_timesteps` and no `ppo.num_envs`. Those come from playground's
Go1 PPO config unless you override them. The v40 chain warm-started each run
from the one before, so a run from scratch does not repeat that lineage.

From scratch:

```
./run.sh train-warp experiment=roboto_gait_v40 run_name=roboto_gait_t1 ++ppo.num_timesteps=3e8 torch_warp.max_minutes=600
```

Warm start from a checkpoint, Brax or Torch:

```
./run.sh train-warp experiment=roboto_gait_v40 run_name=roboto_gait_t2 restore=runs/roboto_gait_t1/checkpoints/000300000000
```

Resume a Torch run after a stop. Repeat every original override and add
`torch_warp.resume=true`:

```
./run.sh train-warp experiment=roboto_gait_v40 run_name=roboto_gait_t1 ++ppo.num_timesteps=3e8 torch_warp.max_minutes=600 torch_warp.resume=true
```

| Key | Default | Effect |
|---|---|---|
| `torch_warp.device` | `cuda:0` | `cpu` runs without graph capture and is very slow. |
| `torch_warp.ckpt_minutes` | `10` | Wall-clock minutes between checkpoints. |
| `torch_warp.max_minutes` | `null` | Stops the run after a checkpoint once this many minutes pass. |
| `torch_warp.critic_warmup_iters` | `0` | Iterations at the start that train only the value network. |
| `wandb.enable` | `true` | `false` trains without wandb. A failed `wandb.init` prints `wandb disabled` and training continues. |
| `contact_preflight` | `true` | A 1-world probe measures contact peaks before training and records them in run.json. |

A resume may change only `ppo.num_timesteps`, `torch_warp.max_minutes`,
`torch_warp.ckpt_minutes` and `torch_warp.resume`. A resume restarts the
envs with a reset, so it is not bit-exact. The README's
"Train, warm start and resume" section has the full rules.

### Outputs

Everything lands in `runs/<run_name>/`:

| File | Content |
|---|---|
| `run.json` | Written at start with null results, and again at the end. `final_reward` is `null`. |
| `train_log.jsonl` | One line on the first iteration and every 5 iterations: `steps`, `sps`, `reward_x1000`, `falls_per_1M`, `done_ep_len`, `losses`, `terms`, `memory`, `budget`. |
| `checkpoints/<12-digit step>/` | Brax-format checkpoints: step 0, one every `ckpt_minutes`, one at the end. |
| `trainer_state.pt` | Networks, Adam, normalizers and RNG state. A resume loads it. |
| `trainer.pid` | The trainer's process id. |
| `DONE` | Written at the end. It holds the last step. |

The README's "run.json and train_log.jsonl" section defines every field.
steps/s compares only between runs at the same seed on the same GPU.

### Watching a run

The console prints one line per logged iteration with steps, reward, falls,
losses, steps/s and RSS. `train_log.jsonl` holds the same values.

`jobs/video_watch.sh` renders a CPU clip from each new complete checkpoint.
It starts once run.json exists, which the Torch trainer writes at start. Run
it in a second Git Bash window from the repo root. `PY` is required and has
no default. On Windows it is `.venv/Scripts/python.exe`:

```
RUN_NAME=roboto_gait_t1 PY=.venv/Scripts/python.exe MAX_IDLE=3600 bash jobs/video_watch.sh
```

Clips land in `runs/<run_name>/videos/step_<12-digit step>.mp4`, each with a
`.log` of the render next to it. `INTERVAL` (default 600 s) sets the check
period and `SCENARIO` (default `walk_ramp`) the scenario. The watcher prints
`DONE` and exits after the trainer writes `DONE`.

## After training

Each of these runs on CPU JAX and reads a Torch run like a Brax run.

```
./run.sh battery --run runs/roboto_gait_t1
./run.sh report --run runs/roboto_gait_t1
./run.sh eval --run runs/roboto_gait_t1 --scenario long
./run.sh eval --run runs/roboto_gait_t1 --compare
./run.sh export --run runs/roboto_gait_t1
```

- `battery` writes `runs/<name>/battery.json`. `--checkpoint STEP` measures
  an earlier checkpoint.
- `report` writes `runs/<name>/eval_report.md` from `battery.json`.
- `eval` writes an MP4. `--compare` puts the open-loop and heading-closed
  rollouts side by side. `--checkpoint STEP` renders an earlier checkpoint.
- `export` writes `policy.npz` and `policy_meta.json` into
  `runs/<name>/deploy/`. It refuses a `roboto_gait_v40` policy, because
  `heading_err` is on its actor list and the robot has no source for it.

The repo records no CPU timing for these on Windows.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `humanoid_lab.torch_warp needs torch` | torch is not installed. | Install CUDA torch from the cu126 index, then the `torch_warp` extra. |
| `torch_warp.device=cuda:0 needs a CUDA build of torch and a visible GPU` | PyPI's CPU-only torch is installed, usually because the extra was installed first. | `.venv\Scripts\python -m pip uninstall torch`, then install torch from the cu126 index. |
| `warp finds no CUDA device. Check the NVIDIA driver.` | warp's CUDA runtime finds no GPU. | Install or update the NVIDIA driver. |
| MJWarp model construction fails after a package upgrade | warp-lang, mujoco or mujoco-mjx moved off its pin. mujoco-mjx 3.10.0 needs exactly warp-lang 1.13.0. | Re-run `.venv\Scripts\python -m pip install -e ".[torch_warp,dev]"`, which restores the pins. |
| bash fails on `run.sh` or `jobs/*.sh` | CRLF checkout from a clone older than `.gitattributes`. | `git add --renormalize .`, or a fresh clone. |
| `JAX runs on <backend>, so it would take the GPU memory torch and warp need` | JAX was imported on a GPU backend before `humanoid_lab.torch_warp`. Only a Linux jax-cuda install can do this. | Set `JAX_PLATFORMS=cpu` before the first jax import. |
| `the Torch trainer refuses this config` | A set key has a value the Torch loop does not implement. The error lists every one. | Change the listed keys, or train with `./run.sh train`. The README's "Supported subset and refusals" lists each refusal. |
| `<key>: unknown key, the Torch trainer does not implement it` | A `ppo` or `network_factory` key outside the ledger. | Remove the key. |
| Hydra fails to compose an override | The key is absent from the yaml. | Prefix it with `++`. |
| `batch_size * num_minibatches = ... must be a multiple of num_envs` | `++ppo.num_envs` does not divide the batch. | Pick a `num_envs` that divides `batch_size * num_minibatches`. |
| `runs/<name>/run.json exists. Continue it with torch_warp.resume=true, or pick another run_name.` | The run dir already holds a run. | Add `torch_warp.resume=true`, or change `run_name`. |
| `torch_warp.resume=true: ... not found` | The run dir lacks run.json or `trainer_state.pt`. | Start a new run, or warm-start from a checkpoint. |
| `torch_warp.resume=true: the composed config differs from ...` | An override differs from the original run outside the four keys a resume may change. The error names each key. | Repeat the original overrides exactly. |
| `... was written by 'brax_ppo', not by torch_warp_ppo` | Only a Torch run resumes. | Warm-start with `restore=runs/<run>/checkpoints/<step>` and a new `run_name`. |
| `network config mismatch with ...` | The composed network differs from the checkpoint's `ppo_network_config.json`. | Use the network group of the source run. |
| A warm start refuses an observation change | Observation lists may grow only by appending at the end. | Append new components at the end of `obs.state` or `obs.privileged`. |
| `MJWarp needs contact budgets` | The robot has no `sim_budget` block. | Run `./run.sh check-contacts` and record `naconmax_per_env` and `njmax` in `robot.yaml`. |
| `WARNING: the probe already reached a warp buffer ceiling` | The contact probe hit a warp buffer limit. Warp drops the overflow silently. | Raise `task.env.sim.naconmax_per_env` or `njmax`. |
| `captured physics graph did nothing: the clock did not advance` | The first CUDA graph launch left the physics still. | No fix is documented. `torch_warp.device=cpu` skips capture and isolates the fault. |
| `actor observation(s) ['heading_err'] have no source on the robot` | `export` of a policy with `heading_err` on its actor list. | Such a policy cannot be exported yet. See [Refusals](deploy.md#refusals). |
