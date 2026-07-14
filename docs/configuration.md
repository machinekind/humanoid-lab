# Configuration

humanoid-lab uses Hydra. `configs/config.yaml` is the entry point and every
run composes from the config groups under `configs/`. `run.sh train` wraps
`python -m humanoid_lab.train`. Every Hydra override syntax works after it:
`group=value`, `key=value`, `+key=value`.

## Resolve before running

Print the fully composed config and exit, without touching JAX or building
any model:

```bash
./run.sh train --cfg job --resolve
```

Run this before every GPU job. It costs seconds. A misconfigured GPU run
costs real money.

## Hydra axes

`configs/config.yaml`'s `defaults` list selects one entry from each group
below. Override any of them with `group=name`.

| Axis | Group dir | Default | Selects |
|---|---|---|---|
| `robot` | `configs/robot/` | `asimov_v1` | Which `robots/<name>/` directory supplies `robot.yaml`, `actuators/`, and the vendored MJCF. |
| `task` | `configs/task/` | `joystick` | Selects the task's env class and its reward and observation overlay. `joystick` tracks commanded velocity. `sizing` is `joystick` with sharpened torque and energy penalties, plus per-step tau/omega/power telemetry. |
| `actuators` | `configs/actuators/` | `sizing_ideal` | Which named actuator preset to inject. Also available: `encos_datasheet`, `deploy_pd`. |
| `network` | `configs/network/` | `default` | Policy/value MLP layer sizes, merged into the PPO network factory. |
| `dr` | `configs/dr/` | `default` | The domain-randomization switch block described below. Only one group, `default`, exists today. |

A `configs/experiment/` group directory exists for `+experiment=<name>`
overlays. It currently holds no presets, only `.gitkeep`. A preset added
there needs `+experiment=<name>` on the CLI, since it is not part of the
`defaults` list.

Each robot config is a plain pointer to a robot directory:

```yaml
# configs/robot/asimov_v1.yaml
name: asimov_v1
dir: robots/asimov_v1
```

`train.py` reads `cfg.robot.dir` and loads `robot.yaml` from that directory
at construction time. Adding a robot means adding a matching
`configs/robot/<name>.yaml` pointer, not restructuring this axis.

## Top-level keys

| Key | Default | Meaning |
|---|---:|---|
| `run_name` | `null` | Output goes to `runs/<run_name>`. Unset resolves to `<task>_<timestamp>`, prefixed `smoke_` under `smoke=true`. |
| `seed` | `0` | PPO random seed. |
| `smoke` | `false` | Shrinks PPO to a tiny CPU-sized budget (100k steps, 64 envs) and caps episode length at 200 steps. `run.sh smoke` also forces `JAX_PLATFORMS=cpu` and `wandb.enable=false`. |
| `restore` | `null` | Checkpoint directory to warm-start from. Relative paths resolve against the repo root. |
| `domain_rand` | `false` | Gates the whole `dr` block. `false` with any `dr.*.enable=true` raises at startup rather than silently ignoring the request. |
| `wandb.enable` | `true` | Log to Weights & Biases if import/login succeeds. |
| `wandb.project` | `humanoid-lab` | W&B project name. |
| `ppo` | `{}` | Global PPO overrides, applied after the task's own `task.ppo` block. CLI `ppo.foo=...` wins over both. |

## Actuator presets: the name pointer

`configs/actuators/<name>.yaml` carries one line, a name pointer:

```yaml
# configs/actuators/sizing_ideal.yaml
name: sizing_ideal
```

The Hydra axis only selects a name string, `cfg.actuators.name`. The kp, kd,
effort limit, velocity limit, armature, and frictionloss values for each
joint group live in `robots/<robot>/actuators/<name>.yaml`, loaded at
env-construction time from `cfg.robot.dir` plus that name. A preset name
only works with a robot that ships a matching file under its own
`actuators/` directory. `sizing_ideal`, `encos_datasheet`, and `deploy_pd`
exist under `robots/asimov_v1/actuators/`. A new robot starts with none of
them and must write its own.

A preset also picks the actuator model. Implemented models are `pd` and
`ideal_torque`. `dc_motor_speed_saturation` and `delayed` are registered but
raise `NotImplementedError`. A `pd` preset also sets `soft_limit_factor` and
`action_scale_factor`. See `src/humanoid_lab/robot/presets.py` for the full
field contract.

## Domain randomization (`dr`)

All five switches default `enable: false`. Setting `domain_rand=true` alone
reproduces the original fixed distribution: floor friction, base and link
mass scale, and one shared gain and kd scale. Each switch below adds an
independent randomization on top of that, gated by its own `enable`.

| Switch | Tunables | Default range |
|---|---|---|
| `dr.com_offset` | `xy`, `z` (m) | `0.02`, `0.01` |
| `dr.joint_gains` | `gain_pct`, `kd_pct` | `0.2`, `0.2` |
| `dr.dof` | `damping`, `armature`, `frictionloss` (multiplicative) | `[0.9, 1.1]` each |
| `dr.foot_friction` | `range` (multiplicative, per foot geom) | `[0.8, 1.2]` |
| `dr.motor_strength` | `range` (multiplicative, per actuator forcerange) | `[0.5, 1.1]` |

## `run.sh` verbs

Read from `run.sh` as it stands today:

| Verb | Runs | Notes |
|---|---|---|
| `train` | `python -m humanoid_lab.train` | Full Hydra CLI available after it. |
| `smoke` | `JAX_PLATFORMS=cpu python -m humanoid_lab.train smoke=true wandb.enable=false` | CPU pipeline check. |
| `build` | `python -m humanoid_lab.build_model` | `--robot NAME --preset NAME [--out PATH]`. Writes `robots/<robot>/mjx/<preset>.xml`. |
| `check` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_model` | `--robot NAME --preset NAME [--steps N] [--xml PATH] [--skip-mjx] [--max-qvel N]`. Gate-checks every keyframe for NaN and for `|qvel|` blowup. |
| `test` | `python -m pytest tests -q` | Runs the test suite. |
| `sizing-collect` | `JAX_PLATFORMS=cpu python -m humanoid_lab.sizing.collect` | `--run runs/<name> [--episodes N] [--steps N] [--seed N]`. Rolls the checkpoint out on CPU and writes `<run>/sizing_data.npz`. |
| `sizing-report` | `sizing.collect` then `python -m humanoid_lab.sizing.report` | `--run runs/<name> [--episodes N] [--steps N] [--seed N] [--motors NAME] [--recollect]`. Skips the collect step if `<run>/sizing_data.npz` already exists, unless `--recollect` is passed. Writes `<run>/sizing_report.md` and `<run>/sizing_scatter.png`. |

## Configs compose only from the editable install

`train.py` uses `@hydra.main(config_path="../../configs", ...)`. Hydra
resolves that path relative to `train.py`'s own file location on disk. The
wheel only packages `src/humanoid_lab`, per `pyproject.toml`'s
`tool.hatch.build.targets.wheel.packages`. `configs/` never ships inside a
built wheel. Config discovery only works from an editable install
(`pip install -e .`) of a full checkout. Only then does `../../configs`,
relative to the installed module, resolve back to the repo root's
`configs/`. A wheel installed somewhere else will fail to find any config
group.

## Verified command examples

Default config, resolved:

```bash
$ ./run.sh train --cfg job --resolve
robot:
  name: asimov_v1
  dir: robots/asimov_v1
task:
  name: joystick
  ...
actuators:
  name: sizing_ideal
network: {}
dr:
  com_offset: {enable: false, xy: 0.02, z: 0.01}
  ...
domain_rand: false
wandb: {enable: true, project: humanoid-lab}
```

Switch task and actuator preset:

```bash
$ ./run.sh train robot=asimov_v1 task=sizing actuators=encos_datasheet --cfg job --resolve
task:
  name: sizing
  env: {}
  ppo: {}
actuators:
  name: encos_datasheet
```

Switch the network group:

```bash
$ ./run.sh train network=large --cfg job --resolve
network:
  policy_hidden_layer_sizes: [512, 256, 128]
  value_hidden_layer_sizes: [512, 256, 128]
```

Enable one DR switch:

```bash
$ ./run.sh train domain_rand=true dr.foot_friction.enable=true --cfg job --resolve
dr:
  foot_friction: {enable: true, range: [0.8, 1.2]}
```

Resolve a smoke run's config. This does not train: `--cfg job` exits before
any model is built.

```bash
$ ./run.sh smoke --cfg job --resolve
smoke: true
wandb: {enable: false, project: humanoid-lab}
```

Inspect a CLI verb's own flags:

```bash
$ ./run.sh build --help
usage: build_model.py [-h] --robot ROBOT --preset PRESET [--out OUT]

$ ./run.sh check --help
usage: check_model.py [-h] --robot ROBOT --preset PRESET [--steps STEPS]
                      [--xml XML] [--skip-mjx] [--max-qvel MAX_QVEL]
```
