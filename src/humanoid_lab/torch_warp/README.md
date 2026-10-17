# torch_warp: the Torch PPO trainer on MJWarp

## What this is

This package is the Torch PPO trainer on MJWarp physics. It is for hosts
without JAX-CUDA, such as native Windows. JAX/Brax stays the primary
trainer.

On Linux the `torch_warp` extra installs no torch. A Linux user who wants
the Torch trainer or the parity suite installs torch by hand from the cu126
index, which matches jax's CUDA 12. See [Linux](#linux).

On a GPU host, jax-cuda preallocates most of the VRAM that torch and warp
need. The Torch trainer runs jax on the CPU.

The trainer reads the same Hydra configs as the Brax trainer. Its primary
config is `configs/torch_warp.yaml`, which is `configs/config.yaml` plus the
Torch-only `torch_warp` and `bench` blocks. The task math lives in the shared
modules both trainers call. The trainer writes Brax-format checkpoints and
the same run.json as the Brax trainer. So the battery, the video tool and
the exporter read its runs like Brax runs.

## Contents

- `physics.py`: the MJWarp model and data, stepped from Torch tensors, with per-world DR fields and CUDA graph capture.
- `env.py`: `WarpJoystick`, the joystick task on Torch tensors, built on the shared task modules, with the brax episode counter and the Torch RNG streams.
- `ppo.py`: the refusal ledger, the PPO loop and `main`, the training run.
- `policy_io.py`: the networks, the observation normalizer, Brax checkpoint conversion and `trainer_state.pt`.
- `bench.py`: a benchmark of physics, `env.step` and full PPO iterations, written to JSON.

## Env protocol

`ppo.Trainer` takes any object with this protocol. `WarpJoystick`'s
docstring states the same contract.

- Attributes: `num_envs`, `action_size`, `actor_obs_size`,
  `privileged_obs_size`, `device`, `term_names`, `actor_obs_names` and
  `env_config`.
- `reset() -> (state, priv)`: float32 tensors of shape
  `(N, actor_obs_size)` and `(N, privileged_obs_size)`.
- `step(action) -> (state, priv, reward, done, truncation, terms)`. `action`
  is `(N, action_size)`, already squashed into [-1, 1]. `reward` is float32
  `(N,)`. `done` and `truncation` are bool `(N,)`. `terms` maps each name in
  `term_names` to its unscaled `(N,)` reward term.

## Install on Windows

[docs/windows.md](../../../docs/windows.md#install) has the install steps.

## Validated versions

| Item | Version |
|---|---|
| torch build and CUDA index | Pending the PR author's Windows run. |
| torch the parity suite ran against here | 2.14.1, CPU build, macOS arm64. |
| Minimum NVIDIA driver | The higher of the validated torch build's and warp 1.13's minimum. Pending. |
| GPU | Pending. |
| Windows build | Pending. |
| Python | Pending. |
| mujoco | 3.10.0 |
| mujoco-mjx | 3.10.0. It vendors MJWarp. |
| warp-lang | 1.13.0. This pin is mandatory. |
| jax | 0.9.2, CPU. |

## Linux

The `torch_warp` extra installs no torch on Linux, because PyPI's Linux
torch brings CUDA 13. To run the trainer or the parity suite on Linux,
install torch by hand from the cu126 index, which matches jax's CUDA 12:

```
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## Train, warm start and resume

`torch_warp.device` defaults to `cuda:0`. A CPU run happens only when you
set `torch_warp.device=cpu`. A CPU run skips graph capture and is very slow,
so use it for smoke checks only.

From scratch. The networks start from brax's own initializer at the run
seed:

```
python -m humanoid_lab.torch_warp.ppo experiment=roboto_gait_v40 run_name=roboto_gait_t1
./run.sh train-warp experiment=roboto_gait_v40 run_name=roboto_gait_t1
```

A smoke run on CPU:

```
python -m humanoid_lab.torch_warp.ppo smoke=true wandb.enable=false torch_warp.device=cpu
./run.sh train-warp smoke=true wandb.enable=false torch_warp.device=cpu
```

Warm start from a Brax or Torch checkpoint. `restore=` names the checkpoint
step directory:

```
python -m humanoid_lab.torch_warp.ppo experiment=roboto_gait_v40 run_name=roboto_gait_t2 restore=runs/roboto_gait_t1/checkpoints/000300000000
./run.sh train-warp experiment=roboto_gait_v40 run_name=roboto_gait_t2 restore=runs/roboto_gait_t1/checkpoints/000300000000
```

The trainer prints the `task.env`, `actuators` and `ppo` leaves that differ
from the source run, and records them in run.json's `warm_start` block.

Resume a Torch run. Repeat the original run's overrides and add
`torch_warp.resume=true`:

```
python -m humanoid_lab.torch_warp.ppo experiment=roboto_gait_v40 run_name=roboto_gait_t1 torch_warp.resume=true
./run.sh train-warp experiment=roboto_gait_v40 run_name=roboto_gait_t1 torch_warp.resume=true
```

- A resume may change only `ppo.num_timesteps`, `torch_warp.max_minutes`,
  `torch_warp.ckpt_minutes` and `torch_warp.resume`. Any other change is
  refused, and the error names each changed key.
- A resume loads `trainer_state.pt`. It appends to `train_log.jsonl` and adds
  an entry to the trainer block's `resumes`.
- A resume is not bit-exact, because the envs restart with a reset.
- A run without `torch_warp.resume=true` refuses a run dir that already holds
  run.json.
- A run written by the old Torch runner, or by the Brax trainer, cannot be
  resumed. Warm-start from its last checkpoint with `restore=`.

`torch_warp.max_minutes` stops the run after a checkpoint. Ctrl+C writes a
final checkpoint. Both write run.json with `early_stopped: true`.

## Override syntax

Overrides are Hydra overrides, as for `./run.sh train`. Under Hydra struct
mode, a key that is absent from the yaml needs `++`. For example
`task.env.command.deadband=0.15` fails to compose, and
`++task.env.command.deadband=0.15` works. A key the yaml already sets, such
as `ppo.sym_coef` or `torch_warp.max_minutes`, takes a plain override. A new
key under `actuators.overrides` takes `+`.

## Checkpoint format

Everything goes under `runs/<run_name>/`:

| File | What it holds |
|---|---|
| `checkpoints/<step>/` | A Brax checkpoint, one per save. The first is step 0, then one every `torch_warp.ckpt_minutes`, then one at the end. |
| `trainer_state.pt` | The networks, Adam, both normalizers, the step and iteration counts, the loop generator and the env draw generators. Written to a `.tmp` file and moved with `os.replace`. |
| `run.json` | The run record. See below. |
| `train_log.jsonl` | One line per logged iteration. See below. |
| `trainer.pid` | The trainer's own process id. |
| `DONE` | Written at the end. It holds the last step. |

- Each checkpoint is written through brax's own `ppo_checkpoint.save`. Its
  `ppo_network_config.json` is written last and carries the saved network's
  true `observation_size`.
- The battery, the video tool and the exporter read these checkpoints like
  Brax ones.
- A save never overwrites an existing step directory. Orbax would delete the
  directory first, which fails on Windows while a reader holds a file in it.
- A warm start may grow the observation only by appending components at the
  end of `obs.state` or `obs.privileged`. The new normalizer columns start at
  mean 0 and std 0.2. A change in the middle of a list is refused.
- The network kwargs in the checkpoint's `ppo_network_config.json` must equal
  the composed ones.

## run.json and train_log.jsonl

run.json has the same keys as a Brax run's. See
[run.json](../../../docs/configuration.md#runjson). The Torch run differs in
these ways:

- It is written at start with null results and again at the end.
- `final_reward` is `null`, because there is no eval loop.
  `stopped_at_steps` is the last training step.
- `env_config` records `sim.backend: warp` and `sim.num_envs` as the run
  used them.

The `trainer` block holds `name: torch_warp_ppo`, `physics: warp`, the
package versions and the device, plus these fields:

| Field | Meaning |
|---|---|
| `torch_warp` | The run's `torch_warp` block. |
| `constants_env_overrides` | The overrides of the one-world JAX env the trainer reads its constants from: `sim.backend: jax`, `sim.num_envs: 1`. |
| `ignored` | Each ignored ledger key that the config sets, with the reason. |
| `budget_peaks` | The highest logged `nacon_pool_max` (contacts in the whole pool) and `nefc_max` (constraint rows per world). |
| `train_reward_last` | The reward per step at the last logged iteration. |
| `resumes` | One `{step, time}` entry per resume. |

Each line of `train_log.jsonl` holds:

| Key | Meaning |
|---|---|
| `it` | The PPO iteration. |
| `steps` | Env steps so far. |
| `sps` | Env steps per second since the last line. |
| `minutes` | Wall-clock minutes since this process started training. |
| `reward_x1000` | Mean reward per control step since the last line, times 1000. |
| `falls_per_1M` | Falls per million env steps since the last line. |
| `done_ep_len` | Mean length of the episodes that ended, or `null`. |
| `losses` | `policy_loss`, `v_loss`, `entropy`, `approx_kl`, `clipfrac` and `sym_loss`. |
| `terms` | The mean of each unscaled reward term. |
| `memory` | MB: `rss`, plus `private` where psutil has it (Windows), `torch_max_allocated` and `warp_mempool_high` on CUDA. |
| `budget` | `nacon_pool_max` and `nefc_max` at this line. |

A line is written on the first iteration of each process and every 5
iterations.

## Supported subset and refusals

The trainer checks every config key against the ledger in `ppo.py`. A key is
implemented, ignored with a recorded reason, or refused when it is set to a
value the Torch loop does not implement. All refusals are listed in one
error. The tests named here are in `tests/parity/`, unless a path is given.

- **Refused PPO keys** unless at brax's default: `action_repeat`,
  `reward_scaling`, `clipping_epsilon_value`, `normalize_observations_mode`,
  `normalize_observations_std_eps`, `normalize_until_count`,
  `bootstrap_on_timeout`, `use_distributional_critic`,
  `learning_rate_schedule` (so ADAPTIVE_KL), `vision`, `augment_pixels`,
  `restore_value_fn` and `max_devices_per_host`. `normalize_observations`
  and `normalize_advantage` must be true. Tests:
  `test_parity_refusals.py::test_each_refused_ppo_key_is_named` and
  `test_adaptive_kl_is_refused`.
- **Refused network keys**: `activation` at any value, a distribution other
  than tanh_normal, `layer_norm`, any kernel init function or kwargs, the
  distributional critic, and swapped observation keys. Only brax's default
  activation (swish) is implemented. A Hydra override gives a string, and
  make_ppo_networks needs a callable. Tests:
  `test_parity_refusals.py::test_each_refused_network_key_is_named` and
  `test_an_activation_string_is_refused`.
- **Unknown keys**: an unknown `ppo` or `network_factory` key is refused.
  Test: `test_parity_refusals.py::test_an_unknown_ppo_key_is_refused`.
- **`early_stop.enable=true`** is refused, because it needs an eval reward.
  Test: `test_parity_refusals.py::test_early_stop_is_refused`.
- **`task.env.sim.backend=jax`** is refused. The trainer steps MJWarp.
  Tests: `test_parity_refusals.py::test_sim_backend_jax_is_refused` and
  `test_sim_backend_jax_is_refused_by_the_ledger`.
- **One device per process.** `max_devices_per_host` other than its default
  is refused. Test:
  `test_parity_refusals.py::test_each_refused_ppo_key_is_named[max_devices_per_host]`.
- **No eval loop.** The eval keys are ignored, and `final_reward` is `null`.
  Tests: `test_parity_refusals.py::test_ignored_keys_come_back_with_their_reason`
  and `test_parity_run_record.py::test_from_scratch_writes_a_brax_checkpoint_and_the_record`.
- **No per-step metrics.** The Torch env emits none, so the sizing task is
  refused. Sizing telemetry stays a JAX rollout. Test:
  `test_parity_refusals.py::test_the_sizing_task_is_refused`.
- **Contact budgets** must be resolved for the robot. Test:
  `test_parity_refusals.py::test_missing_budgets_are_refused`.
- **`domain_rand=false` with an enabled dr field** is refused. Test:
  `test_parity_glue_env.py::test_dr_off_with_an_enabled_field_is_refused`.
- **heading_err export** is refused (owner decision 1). A policy with
  `heading_err` on the actor list trains, but it cannot be exported. See
  [Refusals](../../../docs/deploy.md#refusals). Tests:
  `tests/integration/test_deploy_contract_env.py::test_a_heading_err_policy_is_refused_at_export`
  and `tests/unit/test_deploy_runtime.py::test_a_heading_err_layout_is_refused`.
- **CPU runs** are slow. They run only when `torch_warp.device=cpu` is set.
  A `cuda` device without a CUDA torch build, a GPU or warp's CUDA runtime is
  refused, and the error names the PyTorch index. Test:
  `test_parity_refusals.py::test_a_cuda_device_without_cuda_is_refused`.
- **Warm-start observation growth** only at the end of a list. Test:
  `test_parity_checkpoint.py::test_mid_list_growth_is_refused`.
- **Checkpoint network kwargs** must match the composed ones. Test:
  `test_parity_checkpoint.py::test_network_kwargs_must_match_the_checkpoint`.

The normalizer implements brax's Welford mode with `std_eps` 0 only.

## Parity and the bench

`tests/parity` checks the Torch+MJWarp runner against the JAX env. Torch's
info is never copied from JAX. The lockstep checks replay only the physics
arrays and the random draws, and WarpJoystick keeps its own bookkeeping.

| File | What it checks |
|---|---|
| `test_parity_task_math.py` | The shared task math on JAX arrays and on Torch tensors, from the same values. |
| `test_parity_env.py` | The lockstep run. WarpJoystick runs on the wrapped JAX env's physics and draws for 8 worlds, 400 steps and 5 configs. Every info key, obs, reward, done, truncation, term and ctrl must match. |
| `test_parity_physics.py` | One control step of MJX on JAX against MJWarp, from 100 identical states along a rollout with falls, respawns and foot contacts. |
| `test_parity_dr.py` | Identical DR draws give identical per-world model fields, exact in float32. |
| `test_parity_rng.py` | Torch draws repeat for a fixed seed. Toggling an optional draw leaves the other streams alone, on both backends. Every draw lies in its closed range, and the statistics match their definitions. |
| `test_parity_closed_loop.py` | The real WarpJoystick on MJWarp and TorchDraws, retained recipe, 16 worlds, 500 steps. Outputs stay finite, falls and timeouts occur, and a respawn restores the first pose, `yaw_ref` and the landing state. |
| `test_parity_glue_env.py` | The env glue: the draw structure against the JAX adapters, done and truncation on timeouts and falls, and the refusal of DR off with an enabled field. |
| `test_parity_glue_physics.py` | The WarpPhysics glue: the respawn writes only the masked worlds, and CUDA graph capture. |
| `test_parity_checkpoint.py` | Checkpoints round-trip between Brax and Torch. |
| `test_parity_ppo_math.py` | The Torch GAE, log-prob and entropy against brax's. |
| `test_parity_refusals.py` | Every refusal fails at construction or in the config ledger. |
| `test_parity_trainer_loop.py` | The Torch PPO loop. |
| `test_parity_run_record.py` | `ppo.main` end to end: run outputs, the from-scratch weights against brax `ppo.train`, warm start, resume and run.json. |
| `test_parity_static.py` | `torch_warp` read as source: public env names only, no POSIX-only module and no `torch.compile`. |
| `test_parity_bench.py` | The JSON `bench.py` writes. |
| `replay.py` | The helpers the lockstep checks share: the JAX recorder, `ReplayPhysics` and `ReplayDraws`. |
| `parity_report.py` | Collects the worst error of each check for `--parity-report`. `test_parity_report.py` tests it. |

The tolerances are in `tests/parity/tolerances.py`. The measured values come
from this suite on macOS arm64 with torch 2.14.1, CPU build.

| Check | Tolerance | Worst case measured here |
|---|---|---|
| Task math and the lockstep run | `rtol=1e-5`, `atol=1e-5` | Task math: 3.9e-3 abs on `compose.compute_rewards`, where the relative error is 2.5e-7. Lockstep: 1.0 abs on `terms.joint_acc`, whose values are about 1e6, at a relative error of 4.5e-7. Outside the terms the worst lockstep error is 1.9e-6 abs. |
| Physics on CPU | qpos `atol=1e-4`, qvel `atol=1e-2` | qpos 2.6e-5, qvel 5.2e-3. |
| Physics on CUDA | 10 times the CPU values | Not measured. The value is provisional until the PR author's GPU run. |
| DR fields | exact | 0 in every field. |
| Bools and ints | exact | exact. |

Run the parity suite on CPU JAX:

```
./run.sh test-parity
./run.sh test-parity --parity-report
python -m pytest tests/parity --parity-device cuda:0 --parity-report
```

`./run.sh test-parity` sets `JAX_PLATFORMS=cpu` itself.
`tests/parity/conftest.py` sets it too, so the bare pytest command works in
PowerShell. `--parity-report` prints the worst error of each check at the end
of the run.
`--parity-device cuda:0` runs the Torch side on a GPU with the CUDA physics
tolerance. The files that follow it are `test_parity_task_math.py`,
`test_parity_env.py`, `test_parity_physics.py`, `test_parity_dr.py`,
`test_parity_rng.py` and `test_parity_closed_loop.py`. The trainer,
checkpoint, run-record and glue tests keep Torch on CPU. JAX stays on CPU. The whole suite takes about two minutes
on an Apple silicon Mac with a warm compile cache. On a machine without CUDA
the only skipped test is `test_cuda_graph_steps_match_cpu`.

The goldens in `tests/data/golden/` and the fixtures in
`tests/data/fixtures/` are bit patterns recorded on darwin arm64. Their tests
skip on every other platform. On Windows, `tests/parity` is the check.

Run the bench:

```
python -m humanoid_lab.torch_warp.bench experiment=roboto_gait_v40
./run.sh bench-warp experiment=roboto_gait_v40
```

It writes JSON to `bench.out`, or to `runs/bench/bench_<timestamp>.json`
when `bench.out` is null. `bench.py`'s docstring lists every field. Bench
numbers compare only between runs at the same seed on the same hardware.

## Windows acceptance checklist

The checklist is [Smoke checks](../../../docs/windows.md#smoke-checks) in docs/windows.md. The PR author runs it at the commit-4 SHA and at the PR head and pastes the output into the PR.

## wtrain flags

The PR's `wtrain/train.py` took command-line flags. Each one maps to an
override of the composed config, or is dropped.

| wtrain flag | Override |
|---|---|
| `--src` | `restore=runs/roboto_walk_v1/checkpoints/000300000000` |
| `--name` | `run_name=roboto_walk_v2` |
| `--steps` | `++ppo.num_timesteps=3e8` |
| `--num-envs` | `++ppo.num_envs=4096` |
| `--seed` | `seed=1` |
| `--ckpt-minutes` | `torch_warp.ckpt_minutes=20` |
| `--lr` | `++ppo.learning_rate=1e-4` |
| `--sym-coef` | `ppo.sym_coef=1.0` |
| `--critic-warmup-iters` | `torch_warp.critic_warmup_iters=50` |
| `--max-minutes` | `torch_warp.max_minutes=600` |
| `--src-ckpt` | dropped: `restore=` names the checkpoint step directory itself |
| `--resume` | `torch_warp.resume=true run_name=roboto_walk_v2`, repeating the original run's overrides |
| `--reset-keyframe` | `task.env.reset_keyframe=home_arms_down` |
| `--act-scale` | `+actuators.overrides.groups.knee.action_scale_rad=0.5` |
| `--warp-reward` | The four kept terms: `++task.env.reward.scales.step_length=1.0` and `++task.env.reward.step_length.pair_min=true`, and the same keys for the others: a scale for foot_reach, knee_extend and forward_lean, and pair_min for foot_reach and knee_extend. The 14 deleted terms are dropped, because PR #11 removed them. |
| `--pose-l1` | `++task.env.reward.pose_l1_weights.knee=0.5` |
| `--scale` | `++task.env.reward.scales.feet_phase=1.0` |
| `--warp-cmd-ramp` | `++task.env.command.ramp.lin=0.25 ++task.env.command.ramp.yaw=0.6` |
| `--cmd-straight` | `++task.env.command.straight_prob=0.3` |
| `--warp-lift` | `++task.env.gait.lift.v_full=0.5 ++task.env.gait.lift.min_frac=0.25` |
| `--env` | `++task.env.command.deadband=0.15` |
