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
| `robot` | `configs/robot/` | `asimov_v1` | Which `robots/<name>/` directory supplies `robot.yaml`, `actuators/`, and the vendored MJCF. Also an `@package _global_` overlay that patches robot-specific `task`/`dr` tuning. See "Robot configs own robot-specific tuning" below. |
| `task` | `configs/task/` | `joystick` | Selects the task's env class and its reward and observation overlay. `joystick` tracks commanded velocity. `sizing` is `joystick` with sharpened torque and energy penalties, plus per-step tau/omega/power telemetry. |
| `actuators` | `configs/actuators/` | `sizing_ideal` | Which named actuator preset to inject. Also available: `encos_datasheet`, `deploy_pd`. |
| `network` | `configs/network/` | `default` | Policy/value MLP layer sizes, merged into the PPO network factory. |
| `dr` | `configs/dr/` | `default` | The domain-randomization switch block described below. Only one group, `default`, exists today. |
| `experiment` | `configs/experiment/` | `null` | An experiment overlay, selected with `experiment=<name>`. Composed last, after `_self_`. See "Experiments" below. |

The `experiment` group selects a file under `configs/experiment/` by name:
`experiment=<name>`. See "Experiments" below for what one contains and how
it composes.

Each robot config is a `robot.dir` pointer plus an optional training-tuning
overlay:

```yaml
# configs/robot/asimov_v1.yaml
# @package _global_
robot:
  name: asimov_v1
  dir: robots/asimov_v1
task:
  env:
    command: {...}
    obs_noise: {...}
```

`train.py` reads `cfg.robot.dir` and loads `robot.yaml` from that directory
at construction time. Adding a robot means adding a matching
`configs/robot/<name>.yaml` pointer, not restructuring this axis.

### Robot configs own robot-specific tuning

Reward weights, DR ranges, obs-noise scales, and command envelopes differ
per robot. A robot config carries them as an `@package _global_` overlay.
The overlay lives in a `task:` and/or `dr:` section next to the
`robot: {name, dir}` pointer, and it patches the shared task/dr base
configs.

`configs/config.yaml`'s defaults list is `task, actuators, network, dr,
robot, _self_`. `robot` composes after `task` and `dr`, so a robot's
overlay wins over the task/dr base for any key it sets. A CLI override,
such as `task.env.obs_noise.joint_vel=0.5` or `dr.dof.armature=...`, wins
over the robot overlay in turn: Hydra applies command-line overrides after
the whole defaults list composes. `_self_` placement only governs where
`config.yaml`'s own keys merge relative to the groups. Hydra's
defaults-list composition merges nested dicts recursively, so a robot
overlay only needs to name the keys it actually changes. An untouched key
keeps its task/dr base value.

`task.env.reward` composes through one extra step. `configs/task/
joystick.yaml` defines no `reward` key. The reward defaults live in
`envs/joystick.py`'s `default_config()`. A robot overlay's `reward:`
section is therefore the only `task.env.reward` content in the resolved
config. At env construction, `make_env` deep-merges the resolved `task.env`
onto `default_config()` entry by entry (`registry._apply_overrides`). A
partial reward overlay changes exactly the entries it names, and every
unlisted entry keeps its Python default. Pin only the changed entries.
`configs/robot/roboto_origin.yaml`'s `reward:` section is the worked
example.

`configs/robot/asimov_v1.yaml` is the plain case. It pins a command
envelope and obs-noise scales, cited to asimov's own published docs.
`configs/robot/roboto_origin.yaml` is the fuller case. It maps reward
weights, DR ranges, obs-noise scales, and a command envelope from
RoboParty's own upstream training config, each with a provenance comment.
It also carries a "not ported" comment block that records upstream values
with no equivalent switch or term in this repo yet.

## Experiments

An experiment is one training arm recorded as one file.
`configs/experiment/<name>.yaml` is the complete record of what ran. Run it
with `./run.sh train experiment=<name>`.

Every experiment pins its robot and task in its own defaults list:
`defaults: [override /robot: ..., override /task: ..., ...]`. It also pins
`actuators` when the arm's conclusion depends on the actuator gains. A robot
overlay such as `configs/robot/roboto_origin.yaml` pins its own reward
scales. Running an unpinned experiment against a different robot pulls in
that robot's own reward overlay instead. The result is neither arm.
`configs/experiment/asimov_gentle_penalties.yaml` pins `robot: asimov_v1`,
`task: joystick`, and `actuators: sizing_ideal` for this reason:

```yaml
# configs/experiment/asimov_gentle_penalties.yaml
# @package _global_
defaults:
  - override /robot: asimov_v1
  - override /task: joystick
  - override /actuators: sizing_ideal

task:
  env:
    reward:
      scales:
        orientation: -1.0
        lin_vel_z: -0.2
        action_rate: -0.02
        action_accel: -0.02
```

The `experiment` group is the last entry in `config.yaml`'s defaults list,
composed after `_self_`. An experiment overlay wins over `config.yaml`'s own
keys and over the robot/task/dr overlays. A CLI override still wins over the
experiment. Full wins order, weakest to strongest: task/dr base, robot
overlay, `config.yaml`'s own keys, experiment overlay, CLI. `+experiment=<name>`
errors. Hydra's `+` prefix only adds a key that isn't already in the
defaults list, and `experiment` already carries a `null` entry there.

An experiment PR only adds files: its own yaml under `configs/experiment/`,
optionally a new actuator preset, optionally a new reward term. It never
edits `configs/task/`, `configs/dr/`, `configs/robot/`, or an env's
`default_config()`. Promoting a winning experiment's values into those
shared bases is a separate graduation PR.

A new reward term lands in three places: `src/humanoid_lab/rewards/terms.py`,
the env's `_compute_rewards`, and a `0.0` scale in `default_config()`. The
`0.0` scale keeps the term inert for every run that doesn't opt in. An
experiment yaml turns the term on by setting a nonzero value under
`task.env.reward.scales`.

`wandb.group` defaults to the selected experiment's name, so its A/B arms
group together in W&B. An explicit `wandb.group`, in the experiment yaml or
on the CLI, wins over that default.

`tests/unit/test_experiments.py` composes every file under `configs/experiment/`
in CI. It checks that the file pins robot and task, that its `task.env`
overlay applies onto the task's `default_config()`, and that its actuator
preset resolves against the pinned robot with any inline overrides applied.
A typo'd reward key in the overlay raises there. A broken experiment fails
CI before it costs GPU time.

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
| `wandb.group` | `null` | W&B run group. Defaults to the selected experiment's name, or stays `null` if no experiment is selected. An explicit value wins over that default. |
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
exist under `robots/asimov_v1/actuators/`. `robots/roboto_origin/actuators/`
ships `deploy_pd` and `sizing_ideal`, so both work with
`robot=roboto_origin`; `encos_datasheet` remains asimov-only. A new robot
starts with no presets and must write its own.

A preset also picks the actuator model. Implemented models are `pd` and
`ideal_torque`. `dc_motor_speed_saturation` and `delayed` are registered but
raise `NotImplementedError`. A `pd` preset also sets `soft_limit_factor` and
`action_scale_factor`. See `src/humanoid_lab/robot/presets.py` for the full
field contract.

`load_actuator_preset` (`src/humanoid_lab/robot/presets.py`) deep-merges
`cfg.actuators.overrides` onto the loaded preset yaml before validating it.
train.py, the envs, eval, sizing, and the `build`/`check` CLIs all route
through this one function, so an override resolves the same way everywhere.
An experiment yaml sets `actuators.overrides` keys directly:

```yaml
actuators:
  overrides:
    groups:
      knee:
        kp: 80
```

From the CLI, use `+actuators.overrides.groups.<group>.<param>=<value>`. The
merged dict is schema-checked against the preset's known top-level and
per-group keys. A typo like `kp_` raises `ValueError` there, before any
model builds. `run.json` records the resolved `cfg.actuators` block,
overrides included, so `eval/battery.py` and `sizing/collect.py`
reconstruct the same model from a finished run.

## Velocity-tracking kernels (`task.env.reward`)

The two tracking terms, `tracking_lin_vel` and `tracking_ang_vel`, are
`exp(-err²/tracking_sigma)` kernels by default. The switches below reshape
them. Every one is off at its default, and off reproduces the legacy kernel
exactly.

| Key | Default | Meaning |
|---|---:|---|
| `tracking_sigma` | `0.25` | Width of the absolute kernel, in (m/s)² and (rad/s)². |
| `tracking_product` | `false` | Multiply the two kernels into each other: `k_lin, k_ang = k_lin*k_ang, k_ang*k_lin`. Additive tracking pays the easy half of a command — a robot that ignores a pure-spin command still earns the full `tracking_lin_vel`, since standing still tracks the zero linear command perfectly (w01-tek measured that at about 63% of an ideal spin's payout). With the product, full pay needs the whole command tracked. |
| `tracking_relative` | `false` | Score the fraction of the command tracked instead of the absolute error: the width becomes `tracking_rel_sigma * max(\|cmd\|, floor)²`. The absolute kernel pays only within about `√tracking_sigma` of the target whatever the target's size, so a fast command's reward cliff is out of exploration's reach — w01-tek's policy reached 0.70 m/s under a 0.8 command and 0.00 m/s under a 1.0 one. |
| `tracking_rel_sigma` | `0.25` | Dimensionless width of the relative kernel. A w01-tek quadruped starting point: its terrain presets later widened this to `0.5` because the narrow kernel rounded partial tracking to zero. |
| `tracking_rel_floor_lin` | `0.3` | Floor on the linear relative denominator, m/s. Keeps a near-zero command from sharpening the kernel to a point and dividing by zero. |
| `tracking_rel_floor_ang` | `0.4` | Floor on the angular relative denominator, rad/s. Same role. w01-tek's terrain presets later widened this to `0.7`. |
| `tracking_far_weight` | `0.0` | Mix a wide exponential into both kernels: `(1-w)*kernel + w*exp(-err²/tracking_far_sigma)`. Applies in the absolute and the relative branch alike, and the far kernel stays absolute in both. `exp(-err²/σ)` is gradient-free a few sigma out, so a capability the policy never explored gets no pull toward the command; the wide kernel keeps a usable gradient at range without moving the optimum or leaving `[0, 1]`. **This term alone creates a standing deadlock**: at a yaw rate error of 0.8 rad/s it pays `0.25*exp(-0.64/2.5)`, about 19% of the maximum angular reward, for standing still, and that gradient is weaker than the penalties a pivot attempt incurs. Turn it on only together with `tracking_product` or `tracking_relative`. |
| `tracking_far_sigma` | `2.5` | Width of the far kernel, in (m/s)² and (rad/s)². Ten times `tracking_sigma`. |
| `shaping_tracking_gate` | `false` | Multiply the positive gait-shaping terms by the linear tracking kernel, post-product when `tracking_product` is on. Those terms otherwise pay on a commanded env whether or not it translates, which made stand-and-lift the top income under a command in w01-tek's `terrain_blind_v3`: standing with one leg raised earned about 1.8 reward per step against honest walking's 0.25. Gated set: `feet_air_time`, plus `feet_apex` when port item 1.7 lands. `feet_phase` stays ungated — it is the clock-following gradient and has to survive at zero tracking, because stepping is how tracking starts. Stand-still penalties keep their `~moving` mask and are untouched. |

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
| `build` | `python -m humanoid_lab.build_model` | `--robot NAME --preset NAME [--out PATH] [--set PATH=VALUE ...]`. Writes `robots/<robot>/mjx/<preset>.xml`. `--set` requires `--out`, so an ad-hoc override build never overwrites the canonical preset build. |
| `check` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_model` | `--robot NAME --preset NAME [--steps N] [--xml PATH] [--skip-mjx] [--max-qvel N] [--set PATH=VALUE ...]`. Gate-checks every keyframe for NaN and for `|qvel|` blowup. `--set` forces an in-memory build even if a prebuilt XML exists, and is mutually exclusive with `--xml`. |
| `test` | `python -m pytest tests/unit -q` | The fast suite: model-free, runs in seconds. `tests/unit/test_suite_split.py` fails if a test here builds or steps a model. |
| `test-slow` | `python -m pytest tests/integration -q` | The slow suite: builds models, steps MJX. Exports `JAX_COMPILATION_CACHE_DIR` (default `.jax_cache`) so re-runs skip XLA compilation. |
| `test-all` | `python -m pytest tests/unit tests/integration -q` | Both suites. Same compile cache as `test-slow`. Use before merging. |
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
task:
  name: joystick
  env:
    ...
    command: {vx: [-0.8, 0.8], vy: [-0.6, 0.6], wz: [-0.6, 0.6]}
    obs_noise: {gyro: 0.01, joint_pos: 0.01, joint_vel: 0.1}
    ...
  ppo: {}
actuators:
  name: sizing_ideal
  overrides: {}
network: {}
dr:
  com_offset: {enable: false, xy: 0.02, z: 0.01}
  ...
robot:
  name: asimov_v1
  dir: robots/asimov_v1
domain_rand: false
wandb: {enable: true, project: humanoid-lab, group: null}
```

`task`, `actuators`, `network`, `dr`, and `robot` appear in that order
because that is the defaults-list order in `configs/config.yaml`. `command`
and `obs_noise` show up under `task.env` because `robot: asimov_v1`'s
overlay patches them in. That overlay composes after `task` and `dr` in the
same defaults list.

Switch task and actuator preset:

```bash
$ ./run.sh train robot=asimov_v1 task=sizing actuators=encos_datasheet --cfg job --resolve
task:
  name: sizing
  env:
    command: {vx: [-0.8, 0.8], vy: [-0.6, 0.6], wz: [-0.6, 0.6]}
    obs_noise: {gyro: 0.01, joint_pos: 0.01, joint_vel: 0.1}
  ppo: {}
actuators:
  name: encos_datasheet
  overrides: {}
```

`configs/task/sizing.yaml`'s own `env:` is empty. `robot: asimov_v1`'s
overlay still patches `command`/`obs_noise` in, because the overlay applies
regardless of task. The values match what `envs/sizing.py`'s
`default_config()` already carries as Python literals. `envs/sizing.py`'s
`default_config()` starts from `envs/joystick.py`'s `default_config()`. The
resolved config text changes here. The env's runtime behavior does not.

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
wandb: {enable: false, project: humanoid-lab, group: null}
```

Inspect a CLI verb's own flags:

```bash
$ ./run.sh build --help
usage: build_model.py [-h] --robot ROBOT --preset PRESET [--out OUT]
                      [--set PATH=VALUE]

$ ./run.sh check --help
usage: check_model.py [-h] --robot ROBOT --preset PRESET [--steps STEPS]
                      [--xml XML] [--skip-mjx] [--max-qvel MAX_QVEL]
                      [--set PATH=VALUE]
```
