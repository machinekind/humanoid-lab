# Configuration

humanoid-lab uses Hydra. `configs/config.yaml` is the entry point and every
run composes from the config groups under `configs/`. `run.sh train` wraps
`python -m humanoid_lab.train`. Every Hydra override syntax works after it:
`group=value`, `key=value`, `+key=value`.

Which of these keys reach a deployed robot is a separate question, answered
in [deploy.md](deploy.md).

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
| `restore` | `null` | Checkpoint directory to warm-start from. Relative paths resolve against the repo root. `run.json` stores it relative to the project dir, in POSIX form, and records the source run in its `warm_start` block (see [run.json](#runjson)). |
| `domain_rand` | `false` | Gates the whole `dr` block. `false` with any `dr.*.enable=true` raises at startup rather than silently ignoring the request. |
| `contact_preflight` | `true` | Measure the warp contact/constraint peaks on a short probe before training and record them in `run.json`. Skipped automatically under `smoke=true`. See [Warp contact budgets](#warp-contact-budgets-taskenvsim). |
| `wandb.enable` | `true` | Log to Weights & Biases if import/login succeeds. |
| `wandb.project` | `humanoid-lab` | W&B project name. |
| `wandb.group` | `null` | W&B run group. Defaults to the selected experiment's name, or stays `null` if no experiment is selected. An explicit value wins over that default. |
| `ppo` | `{sym_coef: 0.0}` | Global PPO overrides, applied after the task's own `task.ppo` block. CLI `++ppo.foo=...` wins over both. Repo-owned keys (`sym_coef`) are popped before brax's `ppo.train`. |
| `ppo.sym_coef` | `0.0` | Weight of the mirror-symmetry loss on the policy mean, `mean((loc(M s) - M loc(s))^2)`. The maps come from `envs/symmetry.policy_mirror_maps`. The Brax trainer patches brax's PPO loss only while the value is nonzero, and the eval line then prints `sym <loss>`. The Torch trainer applies the same loss. A robot without a `JOINT_SIGNS` entry refuses a nonzero value. This differs from `task.env.symmetry.enable`, the env-side mirroring (see [Mirror augmentation](#mirror-augmentation-taskenvsymmetry)). |
| `early_stop.enable` | `false` | End the run once the eval reward has plateaued. See [Early stopping](#early-stopping-early_stop). |
| `early_stop.min_evals` | `10` | No stop verdict before this many evals exist. |
| `early_stop.patience` | `6` | Consecutive evals with no new best that end the run. |
| `early_stop.min_delta` | `0.5` | A new best must beat the running best by more than this. **Calibrate above the eval noise.** |

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
`action_scale_factor`, or a flat `action_scale_rad` (radians per unit
action, every joint the same) that replaces the factor formula. Non-pd
models ignore both scale fields. A group may set its own `action_scale_rad`.
Under `pd` it beats the preset value for that group's joints, and other
models ignore it. roboto_origin's `deploy_pd` pins upstream's 0.25 rad.
See `src/humanoid_lab/robot/presets.py` for the full field contract.

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
| `tracking_product` | `false` | Multiply the two kernels into each other: `k_lin, k_ang = k_lin*k_ang, k_ang*k_lin`. Additive tracking pays the easy half of a command — a robot that ignores a pure-spin command still earns the full `tracking_lin_vel`, since standing still tracks the zero linear command perfectly (measured at about 63% of an ideal spin's payout). With the product, full pay needs the whole command tracked. |
| `tracking_relative` | `false` | Score the fraction of the command tracked instead of the absolute error: the width becomes `tracking_rel_sigma * max(\|cmd\|, floor)²`. The absolute kernel pays only within about `√tracking_sigma` of the target whatever the target's size, so a fast command's reward cliff is out of exploration's reach — one measured policy reached 0.70 m/s under a 0.8 command and 0.00 m/s under a 1.0 one. |
| `tracking_rel_sigma` | `0.25` | Dimensionless width of the relative kernel. A quadruped starting point; a narrow kernel rounds partial tracking to zero, and on terrain this had to widen to `0.5`. |
| `tracking_rel_floor_lin` | `0.3` | Floor on the linear relative denominator, m/s. Keeps a near-zero command from sharpening the kernel to a point and dividing by zero. |
| `tracking_rel_floor_ang` | `0.4` | Floor on the angular relative denominator, rad/s. Same role; on terrain this had to widen to `0.7`. |
| `tracking_far_weight` | `0.0` | Mix a wide exponential into both kernels: `(1-w)*kernel + w*exp(-err²/tracking_far_sigma)`. Applies in the absolute and the relative branch alike, and the far kernel stays absolute in both. `exp(-err²/σ)` is gradient-free a few sigma out, so a capability the policy never explored gets no pull toward the command; the wide kernel keeps a usable gradient at range without moving the optimum or leaving `[0, 1]`. **This term alone creates a standing deadlock**: at a yaw rate error of 0.8 rad/s it pays `0.25*exp(-0.64/2.5)`, about 19% of the maximum angular reward, for standing still, and that gradient is weaker than the penalties a pivot attempt incurs. Turn it on only together with `tracking_product` or `tracking_relative`. |
| `tracking_far_sigma` | `2.5` | Width of the far kernel, in (m/s)² and (rad/s)². Ten times `tracking_sigma`. |
| `shaping_tracking_gate` | `false` | Multiply the positive gait-shaping terms by the linear tracking kernel, post-product when `tracking_product` is on. Those terms otherwise pay on a commanded env whether or not it translates, which has made stand-and-lift the top income under a command on a quadruped run: standing with one leg raised earned about 1.8 reward per step against honest walking's 0.25. Gated set: `feet_air_time`, `feet_apex` and `feet_apex_min`. `feet_phase` stays ungated — it is the clock-following gradient and has to survive at zero tracking, because stepping is how tracking starts. Stand-still penalties keep their `~moving` mask and are untouched. |

## Orientation tolerance cone (`task.env.reward`)

The `orientation` penalty is `sum(gravity_xy²)`, which is `sin²` of the
base's tilt from vertical. `orientation_tol_deg` puts a tolerance cone around
upright: the penalty becomes `max(sin²(tilt) - sin²(tol), 0)`, exactly zero
inside the cone and rising continuously from its edge with the legacy
penalty's own slope.

Tilt here is measured against **gravity**, not against the local surface. A
flat-referenced penalty therefore taxes the body pitch that locomotion needs
— leaning into an acceleration, or climbing — while a real nosedive stays far
outside any cone worth setting. 20 degrees is a workable cone; 10 was
measured too tight for that reason.

| Key | Default | Meaning |
|---|---:|---|
| `orientation_tol_deg` | `0.0` | Half-angle of the cone, degrees. `sin²` of it is precomputed at construction, so `0` leaves the legacy penalty bit-exact and a live env never re-reads the key — change it by config, not by mutating a built env. |

## Swing shaping (`task.env.reward`)

Two terms shape what a swing looks like, both at weight 0 by default.

`feet_apex` pays each completed swing, once, at touchdown, for how close its
**peak** clearance came to `apex_target`. The env tracks that peak in
`info["swing_apex"]`: a running maximum while the foot is airborne, read at
first contact, cleared afterwards. Duration-averaged clearance terms —
`feet_phase` here — tolerate a long 1.5 to 2 cm skim
that collects nearly as much as a crisp arc, so the optimizer skims. Pricing
the peak has measured 3 to 5 cm swings and 30 to 70% better grip. The term is
in the `shaping_tracking_gate` set.

`feet_landing` is a penalty on downward foot speed weighted by closeness to
the floor, `sum(min(vz, 0)² * clip(1 - clearance/glide_height, 0, 1))`. It is
measured **before** contact on purpose: a penalty read at contact under-reads
impacts, because the solver has already absorbed the hit within the control
step it becomes visible. The gate makes the gradient read "decelerate as you
approach" — 1 at the floor, 0 at `glide_height` and above — so stance feet
score about zero and a swing high above the floor scores zero at any speed.
The physical reference for touchdown softness is free fall over the band:
`sqrt(2*9.81*0.03) ≈ 0.77 m/s`. It is **not** in the shaping gate: gating a
penalty on the tracking kernel would relax it exactly when tracking is
failing, which is when feet are being slammed into the floor.

"At the floor" and "at `glide_height`" are physical heights:
`_foot_clearance` reads the sole's height above the floor (a planted foot
~0), per [docs/lessons/foot-clearance.md](lessons/foot-clearance.md).

| Key | Default | Meaning |
|---|---:|---|
| `scales.feet_apex` | `0.0` | Weight of the per-swing apex reward. `0` = off. |
| `scales.feet_apex_min` | `0.0` | Weight of the two-foot apex reward: each landing pays `clip(min(own peak, other foot's last completed peak) / apex_target, 0, 1)`, so a one-leg gait earns nothing from its lifting leg. Needs exactly two feet. `0` = off, and then no `reward/feet_apex_min` metric exists. In the `shaping_tracking_gate` set. |
| `scales.feet_landing` | `0.0` | Weight of the soft-landing penalty (negative when on). `0` = off. |
| `apex_target` | `0.05` | Swing peak the apex reward asks for, m. Clipped at: the term prices reaching the target, not exceeding it. **Re-derive for this leg** — a quadruped starting value, and our own `gait.swing_height` asks for 0.08 m. |
| `glide_height` | `0.03` | Height band the landing penalty acts in, m. **Re-derive** with `apex_target`. |

`reward.stand_still_vel_weight` (default `0.2`) sets stand_still's
velocity-damping share against a position share of 1; only the ratio
matters, since `scales.stand_still` prices the sum.

**Ported robolab terms** (`rewards/terms.py`; same shapes as upstream, so
the upstream weights transfer): `scales.pose_l1` with
`reward.pose_l1_weights` (a joint_group -> weight map, `null` = 1.0 for
every joint), `scales.joint_pos_limits`, `scales.joint_vel`,
`scales.joint_acc`, `scales.upward`, `scales.feet_distance` /
`scales.knee_distance` with `feet_distance_range` / `knee_distance_range`
(m, base frame), and `scales.feet_contact_without_cmd`. All default to
`0.0` (off). `configs/robot/roboto_origin.yaml` pins the upstream weights.

`task.env.push` has three optional knobs on top of the planar kick:
`interval_steps_range` (random gap to the next push, steps), `vel_z`
(vertical kick, m/s), and `ang_vel_rp` / `ang_vel_yaw` (angular kicks,
rad/s). Off, the legacy fixed schedule is bit-identical.

## Speed-scaled lift (`task.env.gait.lift`)

The lift scale shrinks the swing targets at slow commands. A slow command
then asks for small steps instead of a high-knee march in place. The scale is

    lift = clip(cmd_speed / v_full, min_frac, 1)

where `cmd_speed` is the speed the gait clock serves (planar speed plus the
weighted yaw rate).

| Key | Default | Meaning |
|---|---:|---|
| `lift.v_full` | `0.0` | Commanded speed at which the targets reach full height, m/s. `0` = off. |
| `lift.min_frac` | `1.0` | Floor of the scale. |

The scale multiplies two targets: the `feet_phase` clearance target from the
gait clock and the `feet_apex` target `reward.apex_target`. `feet_apex_min`
keeps the unscaled `apex_target`. Every other term is unchanged.

With `v_full` at `0` the targets reach the terms untouched, so the rewards
are bit-identical to a config without the block. `compute_rewards` reads
`gait.lift` at call time, like `reward.*`.

The lift scale is training-only. The phase observation reads neither key,
and both are in `TRAINING_ONLY_KEYS` (`deploy_contract.py`).

## Frozen clock observation (`task.env.gait.frozen_phase_obs`)

With `gait.freq` at `(0, 0)` the gait clock never moves. This key picks what
the actor observes as `phase` then.

| Value | Phase observation |
|---|---|
| `"cos_sin"` (default) | cos/sin of the frozen leg phases, `[1, -1, 0, 0]` for two feet. The legacy observation. |
| `"zeros"` | `zeros(2F)` for F feet. |

A frozen `cos_sin` phase is a constant side-dependent input, which lets the
policy specialise one leg. `"zeros"` removes that left/right marker. Any
other value is refused at construction. With a running clock the key has no
effect.

The key ships in the deploy contract as `gait_clock.phase_obs`, and it is in
`CONSUMED_KEYS`. A `"zeros"` contract is schema 2, which an old vendored
runtime refuses. See docs/deploy.md.

## Landing and lean terms (`task.env.reward`)

Four terms ported from the Torch+MJWarp runner. Each is off at its default
scale of `0.0`. A term is computed only when its scale in `reward.scales` is
nonzero. With all four off, no reward key, metric, reading or `info` leaf is
added.

| Scale | What it pays |
|---|---|
| `step_length` | Each landing, for how far the landing foot is ahead of the other foot. |
| `foot_reach` | Each landing, for how far ahead of the base the foot lands. |
| `knee_extend` | An open knee in late swing, plus an optional bonus at touchdown. |
| `forward_lean` | A body pitch that grows with the forward command. |

All four pay only while the command walks forward:
`cmd_vx > max(0.1, command.deadband)`. A stand, a turn in place or a
backward command pays 0.

The landing terms read each foot's position in the base frame. The base
position is `xpos[root_body]`, where `root_body` is the body of the free
joint. It comes from the same forward-kinematics stage as the foot sites and
the base orientation. Forward of the base is +x.

### `reward.step_length`

| Key | Default | Meaning |
|---|---:|---|
| `target` | `"clock"` | `"clock"`: target = `cmd_vx / (2 * freq)`, with `freq` the gait clock's frequency at the commanded speed. `"linear"`: target = `target_a + target_b * max(cmd_vx, 0)`. |
| `target_a` | `None` | Target at zero speed, m. Required for `"linear"`. |
| `target_b` | `None` | Target growth per m/s, s. Required for `"linear"`. |
| `power` | `1.0` | Exponent on the fraction. |
| `pair_min` | `False` | See below. |

The fraction is `clip(ahead / target, 0, 1) ** power`, where `ahead` is the
landing foot's x minus the other foot's x. The target is clamped to at least
0.05 m. A foot landing behind the other one earns 0. Power 1 pays the same
per second at any cadence for a given speed. Power 2 makes the pay per second
grow with the step itself.

### `reward.foot_reach`

| Key | Default | Meaning |
|---|---:|---|
| `target_a` | `None` | Target at zero speed, m. Required. |
| `target_b` | `None` | Target growth per m/s, s. Required. |
| `power` | `2.0` | Exponent on the fraction. |
| `x_min` | `0.0` | Foot x that pays 0, m. Must stay below 0.02. |
| `pair_min` | `False` | See below. |

The target is `target_a + target_b * max(cmd_vx, 0)`, clamped to at least
0.02 m. The fraction is `clip((foot_x - x_min) / (target - x_min), 0, 1) **
power`. A foot landing at `x_min` or further back earns 0.

### `reward.knee_extend`

| Key | Default | Meaning |
|---|---:|---|
| `flex_lo` | `None` | Knee angle that counts as fully extended, rad. Required. |
| `flex_hi` | `None` | Knee angle that counts as not extended, rad. Required. |
| `x_split` | `0.0` | Base-frame foot x past which an airborne foot is in late swing, m. |
| `td_min_air` | `0.2` | Shortest swing that earns the touchdown bonus, s. |
| `td_mult` | `0.0` | Weight of the touchdown bonus. `0` = no bonus. |
| `pair_min` | `False` | See below. Applies to the touchdown bonus. |

The extension is `clip((flex_hi - knee_q) / (flex_hi - flex_lo), 0, 1)`. It
pays while a foot is in the air and ahead of `x_split`. The touchdown bonus
pays `td_mult` times the extension on the landing step. Only a real step
earns it: a swing of at least `td_min_air` seconds that lands ahead of the
base. The term needs a `knee` joint group with one joint per foot.

### `reward.forward_lean`

| Key | Default | Meaning |
|---|---:|---|
| `max_deg` | `None` | Lean at full speed, degrees. Required. |
| `vx_max` | `None` | Forward command at which the lean reaches `max_deg`, m/s. Required. |
| `sigma` | `None` | Width of the Gaussian. Required. |

The target is `sin(radians(max_deg) * clip(cmd_vx / vx_max, 0, 1))`. The pay is
`exp(-(gravity_x - target)^2 / sigma)`, where `gravity_x` is the x component
of the body-frame gravity direction. It is 1 at the target.

### `pair_min` and `land_frac`

With `pair_min` on, a landing pays the lower of its own fraction and the other
foot's last landing fraction. One leg's good step then cannot carry a
shuffling partner.

The last fractions live in `info["land_frac"]`, shape `(F, 3)` per world. The
slot order is `[foot_reach, step_length, knee_extend]`. A slot whose term is
off stays 0. A landing foot records its fractions after the rewards have read
the previous record. Each fraction is recorded before the pair-min.

`land_frac` exists only when some landing term has a nonzero scale and
`pair_min` on. It restarts at 0 on respawn through the `land` layer of
`envs/gait_state.py` (`LandReseedWrapper` in `envs/wrappers.py`). A new
episode's first landing is then not compared with the dead episode's last
one.

### Construction refusals

The env refuses to build when a term with a nonzero scale cannot run:

- a required key above is left `None`. The error names each full key.
- `step_length.target` is neither `"clock"` nor `"linear"`.
- `step_length.target` is `"clock"` and `gait.freq[0]` is 0. The clock runs at
  `freq[0]` at a zero command, so the target would be 0/0.
- `foot_reach.x_min` is 0.02 or more. The fraction would divide by zero or a
  negative number.
- `step_length`, `foot_reach` or `knee_extend` is on and the robot does not
  have exactly two feet.
- `knee_extend` is on and the robot has no `knee` joint group with one joint
  per foot.

All four terms are training-only. Their keys sit under the `reward.` prefix,
which `TRAINING_ONLY_PREFIXES` in `deploy_contract.py` covers.

## Settled pose anchor (`task.env.real_pose_ref`)

Off by default. `pose` and `stand_still` both score a deviation from
`_default_pose` — the reset keyframe's **commanded** joint values. Under
gravity and finite gains the robot comes to rest below that command, so the
deviation never reaches zero and both terms charge a floor no policy can
remove. `roboto_origin`'s `home` keyframe settles 0.065 rad off
its command (0.015 rad at the knee), which at the stock `scales.stand_still`
of `-0.5` is 0.032 of standing penalty per step that exists only because the
anchor is wrong. Each actuator preset sags differently, so the floor also
moves with a config axis that has nothing to do with the task.

On, the env settles a **quasi-rigid copy** of the model once at construction
and anchors `pose`, `stand_still` and the reset pose on the result. Every
actuator on the copy becomes the same stiff position servo — kp 400, kd 20,
force cap removed, timestep 5e-4, `implicitfast` — held at the keyframe
targets for two simulated seconds. The pose that comes out does not depend
on the runtime gains or the actuator model: at a given `soft_limit_factor`,
two gain sets and two actuator models settle to bit-identical anchors.
Two things are deliberately not factored out. A preset that changes joint
armature (roboto's presets do) moves the two-second settle, measured at about
5e-5 rad there — armature is a property of the mechanism, not a gain. And a
preset that changes `soft_limit_factor` moves the clip below, and with it the
anchor: roboto's home pose is inside its 0.9 soft limits and 0.029 rad
outside its 0.8 ones. The runtime envelope is preset policy, not an actuator
detail. Compensating the real plant's sag to reach that pose
is the policy's job. Cost is about 0.2 s of plain CPU MuJoCo, and only when
the flag is on.

**The ctrl anchor does not move.** `_default_pose` stays what
`ctrl_from_action` centers on and what the `joint_pos` observation subtracts.
Only the reward reference and the reset pose change. Re-centering the action
space on a sagged pose would silently change what a zero action commands and
what the policy reads back.

Two details are load-bearing:

- The settle targets are clipped to the preset's **soft joint limits**
  (`joint range × soft_limit_factor`, in radians), not to the model's raw
  `ctrlrange` and not to `_ctrl_lo`/`_ctrl_hi`. `step()` clips a `pd`
  preset's motor targets to those soft limits, so a pose settled past them is
  one the policy can never command. A `pd` preset's raw ctrlrange is `[0, 0]`
  besides — those actuators are deliberately `ctrllimited=False`. And
  `_ctrl_lo`/`_ctrl_hi` are the soft limits only for a `pd` preset: for an
  `ideal_torque` one they are the actuator forcerange in N·m, so clipping a
  radian target against them does nothing at all. Reading the angle envelope
  directly is what keeps the two models on the same anchor.
- The settle forces each actuator's `gaintype`/`biastype`/`ctrllimited` as
  well as its gain and bias parameters, because a torque preset's params are
  not a servo's. An `ideal_torque` preset injects
  `biastype NONE` actuators whose ctrl is a torque; overwriting the parameter
  arrays alone would leave `force = 400*ctrl` and settle a different robot.

**Construction raises if the settle does not end standing still**, naming the
robot and the settled height. Two conditions: the settled base height must
clear `fall.min_height`, and the robot must have come to rest (max `|qvel|`
under 1e-2). A biped needs both. `asimov_v1`'s keyframes satisfy neither —
held rigid, `home` topples backward within a second (its CoM sits about 2 cm
behind the heel) and comes to rest at 0.111 m, while `knees_bent` is still
above the fall floor at the two-second cut but moving at 0.15 rad/s and on
the floor by four seconds. A height check alone would have anchored on that
snapshot. **Turn this on only for a robot whose reset keyframe is a standing
equilibrium**; `roboto_origin`'s `home` is (base height flat to 1e-5 m out to
ten simulated seconds), and asimov_v1's are not, pending a balanced keyframe.

This repo has no height command, so one settle is enough. If a height
command ever lands here, the extension is a grid: settle a rung per
commanded stand height and interpolate the anchor on the command.

| Key | Default | Meaning |
|---|---:|---|
| `real_pose_ref` | `false` | Anchor `pose`, `stand_still` and the reset pose on the settled pose instead of the keyframe pose. `false` is the legacy anchor, bit-exact, and runs no settle. |

## No-progress termination (`task.env.no_progress`)

Off by default. When on, an env whose measured progress keeps falling short
of its command is terminated probabilistically, CaT-style
([arXiv 2403.18765](https://arxiv.org/abs/2403.18765)). It closes the
reward-landscape hole where ignoring the command indefinitely is profitable:
forfeiting the rest of the episode is the whole penalty. **No reward term is
attached** — `rewards["termination"]` stays fall-only, and enabling this adds
no key to `reward.scales`.

Per control step, with the command that drove the step:

```
served  = dot(linvel_xy, cmd_xy)/max(|cmd_xy|, 1e-6) + 0.3*gyro_z*sign(cmd_wz)
ema    ←  (1 - dt/ema_sec)*ema + (dt/ema_sec)*served
ratio   = ema / max(demand, 1e-6)          demand = |cmd_xy| + 0.3*|cmd_wz|
hazard  = p_max * clip((risk_below - ratio)/risk_below, 0, 1)
cut     ~ bernoulli(hazard)  when armed, else 0
```

`served` is a projection, not a magnitude, so moving against the command
reads negative — worse than standing still — and moving across it scores
zero. The cut arms only when `demand > 0.05` and `steps_since_cmd*dt >=
grace_sec`. The EMA reseeds to the new demand (ratio 1) on every command
resample, so a robot is never billed for the previous command's shortfall.
The math is `src/humanoid_lab/envs/progress.py`; the env wires it in
`envs/joystick.py`.

Two metrics appear while it is on and exist nowhere otherwise:
`no_progress_cut` (episode sum, 1 exactly when the episode ended on the cut)
and `progress_ratio_per_step` (per-step mean of the ratio, clipped to
`[0, 2]`).

| Key | Default | Meaning |
|---|---:|---|
| `enable` | `false` | Off changes nothing: no info state, no metrics, and no RNG key is split, so a rollout stays bit-exact (`tests/integration/test_golden_baseline.py`). |
| `grace_sec` | `2.0` | No hazard for this long after a reset or a command resample. **Re-derive for a biped.** A quadruped starting value, and the one most likely wrong here: turning a two-legged gait around takes longer than turning a 0.21 m four-bar quadruped's. |
| `ema_sec` | `1.0` | Smoothing horizon of the progress measure, seconds. Long enough that one bad stride does not arm the cut. |
| `risk_below` | `0.5` | The hazard starts below this fraction of the commanded speed. **Re-derive for a biped**, together with `grace_sec`: 50% of demand may be a lot to ask of a humanoid inside the grace window. |
| `p_max` | `0.02` | Per-step hazard at zero progress. Expected survival at a dead stop is `1/p_max` control steps — 50 steps, 1 s at `ctrl_dt=0.02`. |

The meter is also reseeded on every **respawn**, by a wrapper rather than by
the env. `wrap_for_brax_training`, the trainer's own wrapping, ends in
`BraxAutoResetWrapper(full_reset=False)`: on done it restores `data` and
`obs` from the cached first state and returns `state.info` untouched. So
`info` survives every termination, and a cut env would come back carrying the
dying episode's shortfall and a `steps_since_cmd` well past `grace_sec` —
armed on its first step, and dead again within a second. `envs/wrappers.py`'s
`ProgressReseedWrapper` puts `progress_ema` back at the command's demand and
`steps_since_cmd` back to 0 on done, and `train.py` layers it on exactly when
`no_progress.enable` is set. With the cut off, the trainer's `wrap_env_fn` is
`wrap_for_brax_training` itself, unchanged. Reseeding only the EMA and
carrying the counter over would re-arm the cut on the respawn's first step,
so the counter is zeroed too. Any other wrapper that restarts an episode in
place owns the same reseed.

## Mirror augmentation (`task.env.symmetry`)

Off by default. When on, each env draws a flag at reset with probability
`mirror_prob`, and a flagged env presents the policy a world mirrored about
the body xz-plane: both observation vectors are mirrored on the way out and
the action is mirrored back on the way in. Physics, rewards and termination
stay in the real frame, so one policy has to walk both chiralities. Under
the trainer's `BraxAutoResetWrapper(full_reset=False)` the flag lives in
`info`, which survives every respawn, so it holds per env for the whole run.

The maps are in `src/humanoid_lab/envs/symmetry.py`: left/right twins swap,
and each joint's sign comes from its MJCF axis (the module docstring holds
the derivation table). Signs exist for `roboto_origin` only; any other robot
refuses at construction. The battery, eval video and export envs force
`enable` off (`eval/battery.py`), and the key is training-only in the deploy
contract.

| Key | Default | Meaning |
|---|---:|---|
| `enable` | `false` | Off changes nothing: no info key, no RNG key split, no trace change, so a rollout stays bit-exact (`tests/integration/test_golden_baseline.py`). |
| `mirror_prob` | `0.5` | Fraction of envs that present the mirrored world. |

## Pure command draws (`task.env.command`)

The command sampler draws `(vx, vy, wz)` from one uniform box. That box
almost never produces a clean corner: a backward command arrives with random
lateral and yaw contamination attached, and under `tracking_product` or
`tracking_relative` a contaminated corner pays about nothing however well the
robot serves it. The skill is then never profitable to learn, and the policy
settles on refusing it — a quadruped policy trained this way held 0.000 m/s
under a commanded -0.4 backward, and five isolating probes confirmed the
refusal was learned rather than mechanical.

The six draws below rewrite the base sample into a clean command with the
given probability. They apply in the order `wz, vy, slow, fast, back,
straight`, a later draw overwriting an earlier one, and all of them run
before `zero_prob`, which stays the sampler's last word: standing still
overrides every draw. `straight`'s place in that order does not change the
result. Its zeros are `+0.0`, like every other draw's.

Each draw is gated on its static probability and keys off
`jax.random.fold_in(rng, 0x100 + idx)` with an index of its own — `1 wz,
2 vy, 3 slow, 4 fast, 5 back, 6 straight`, fixed. `straight` keys off
`jax.random.fold_in(rng, 0x106)`. So a draw at probability 0 does not
exist in the trace, all six off leave the sampler bit-identical to the
pre-1.6 one (`tests/integration/test_golden_baseline.py`), and enabling one
draw does not move another draw's samples
(`tests/integration/test_pure_command_draws.py`).

The `0x100` offset is load-bearing. `fold_in(key, i)` is bit-identical to
`split(key, n)[i]` for every `i < n`, so a raw table index would fold in one
of the sampler's own base split keys — index 1 would *be* the `vy` uniform
key. The offset puts every draw's key out of reach of any split of `rng`,
whatever width that split later grows to.

Every range is a **starting value to re-derive**, taken from this repo's own
envelope (`vx ±0.8`, `vy ±0.6`, `wz ±0.6`).

| Key | Default | Meaning |
|---|---:|---|
| `pure_wz_prob` | `0.0` | Keep the drawn `wz`, zero the linear part: spin-in-place training. |
| `pure_vy_prob` | `0.0` | Keep the drawn `vy`, zero `vx` and `wz`: pure-strafe training. |
| `pure_slow_prob` | `0.0` | Redraw `vx` from `slow_vx`, zero `vy` and `wz`: clean slow straight walking, so the gait learns to scale down instead of having one speed. |
| `slow_vx` | `(0.1, 0.35)` | Range of the slow redraw, m/s. Sits inside our `vx` range. |
| `pure_fast_prob` | `0.0` | Redraw `vx` from `fast_vx`, zero `vy` and `wz`: clean fast straight walking. |
| `fast_vx` | `(0.5, 0.8)` | Range of the fast redraw, m/s. Tops out at `0.8`, the top of our commanded `vx` box. Setting `fast_vx` **above** the box is a known way to pull a policy past a speed it deadlocks at. Our envelope is capped pending sysid, so commanding past it is a decision for later, not a default. |
| `pure_back_prob` | `0.0` | Redraw `vx` from `back_vx`, zero `vy` and `wz`: clean backward walking, the refusal described above. |
| `back_vx` | `(-0.8, -0.2)` | Range of the backward redraw, m/s. Sits inside asimov_v1's negative `vx` range. **Not inside roboto_origin's**, whose overlay narrows `vx` to `[-0.6, 1.0]` — arming `pure_back_prob` there without narrowing `back_vx` is refused at construction (see below). |
| `straight_prob` | `0.0` | Keep the drawn `vx`, zero `vy` and `wz`: straight walking. |

Every armed redraw is checked against the **composed** `command` box when the
env is constructed: `command.<range>` must sit inside `command.<axis>`, or
`Joystick.__init__` raises a `ValueError` naming the draw, the range and the
box. A redraw is a redraw of the axis, not a widening of it — a policy that
trained on commands outside its own box would ship a contract that
understates what it saw. `deploy_contract.py` refuses the same configuration
at export; the construct-time check just moves the failure to before the GPU
hours. Draws at probability `0.0` are not checked, so the shipped all-off
defaults validate nothing.

## Command ramp (`task.env.command.ramp`)

The ramp limits how fast the served command can change. A sampled command
goes to `info["cmd_target"]`, and each control step `info["command"]` moves
toward it by at most rate · dt per axis. The policy observes and is rewarded
on `info["command"]`.

| Key | Default | Meaning |
|---|---:|---|
| `ramp.lin` | `0.0` | Rate limit on `vx` and `vy`, m/s². |
| `ramp.yaw` | `0.0` | Rate limit on `wz`, rad/s². |

Both at `0.0` is off. When the ramp is on, both rates must be > 0. A config
with only one rate set, or a negative rate, is refused when the env is
constructed (`check_command_ramp` in `envs/joystick.py`).

At reset the command starts at zero and `cmd_target` holds the sampled
command. The command then slews toward it. A respawn keeps the command where
it was, like every other piece of info the wrappers do not clear.

`info["cmd_target"]` exists only when the ramp is on. With the ramp off the
info pytree, the trace and the RNG stream are the same as without the key.

With the no-progress cut also on, `progress_ema` seeds from the served
command, which is zero at reset.

The measurement env forces the ramp off, so the battery, video and export
all run without it (`battery._measurement_env_overrides`). The battery
writes its scripted command into `info["command"]`, and a ramp would slew it
toward the random reset target. Sizing forces the ramp off too
(`sizing.collect.sizing_env_overrides`). It measures the trained gait under
its own commands. The ramp would start each episode at a zero command and
lag every resample.

The ramp is training-only. The runtime does not ramp commands, so the robot
side should rate-limit operator commands (see `docs/deploy.md`).

## Domain randomization (`dr`)

All five switches default `enable: false`. Setting `domain_rand=true` alone
enables the `dr.base` set: floor friction, base and link mass scale, and
one shared gain and kd scale (`gain_fallback`, used only while
`dr.joint_gains` is off). Each switch below it adds an independent
randomization on top, gated by its own `enable`.

| Switch | Tunables | Default range |
|---|---|---|
| `dr.base` | `floor_friction`, `base_mass`, `link_mass`, `gain_fallback` (multiplicative; the always-on set, no `enable`) | `[0.6, 1.2]`, `[0.7, 1.3]`, `[0.9, 1.1]`, `[0.8, 1.2]` |
| `dr.com_offset` | `xy`, `z` (m) | `0.02`, `0.01` |
| `dr.joint_gains` | `gain_pct`, `kd_pct` | `0.2`, `0.2` |
| `dr.dof` | `damping`, `armature`, `frictionloss` (multiplicative) | `[0.9, 1.1]` each |
| `dr.foot_friction` | `range` (multiplicative, per foot geom) | `[0.8, 1.2]` |
| `dr.motor_strength` | `range` (multiplicative, per actuator forcerange) | `[0.5, 1.1]` |
| `dr.base_mass_add` | `kg` (additive, +-kg on the base body) | `1.0` |

**Friction draws and MuJoCo's combine rule.** Two equal-priority geoms
contact at the element-wise **max** of their frictions. So while
`foot_friction` is off, a `dr.base.floor_friction` draw below the foot
geoms' own friction never reaches a foot contact — only the part of the
range above the foot value randomizes anything there. Enabling
`foot_friction` gives the foot geoms contact priority 1, which makes the
foot's draw the contact friction outright (and makes the floor draw
irrelevant at foot contacts, whatever its value). `./run.sh check-friction
--robot <name> --preset <name>` proves this end to end on the box's own
backend: it compares the friction inside each settled foot-floor contact
against that env's draw and exits nonzero on any mismatch. Run it on a GPU
host before trusting a slip-randomized training run to warp.

"Independent" is a property of the RNG plumbing, not a wish. The fixed
distribution draws from `r1..r5 = jax.random.split(rng, 5)`; each switch
above draws from `jax.random.fold_in(rng, 0x100 + idx)` with an index of its
own — `1 joint_gains, 2 com_offset, 3 dof, 4 foot_friction, 5
motor_strength, 6 base_mass_add`, fixed. The `0x100` offset is the same load-bearing constant
as the pure command draws use, for the same reason: `fold_in(key, i)` is
bit-identical to `split(key, n)[i]` for every `i < n`, so a raw table index
keys off one of the five base keys. Before the offset landed, `com_offset`
sampled straight off `r3`, the link-mass key, and `foot_friction` sampled
straight off `r5`, the kd key — measured correlation 1.0 between the COM
offset and the link-mass scale, and between the first foot's friction scale
and the kd scale. Two axes were one axis wearing two names.

Fixing that **changed the DR sampling streams**: a run at a given seed now
draws different worlds than it did before. Nothing published depends on it —
DR is training-only, and the goldens roll out with DR off.
`tests/integration/test_randomize.py` pins both the index domain and the
decorrelation.

## Warp contact budgets (`task.env.sim`)

Only the warp backend reads these. `envs/backend.py`'s `make_data_fn` passes
them to `mjx.make_data` on the warp branch and calls `make_data(mjx_model)`
with no kwargs on the jax branch, so changing either one cannot move a jax
rollout by a bit.

| Key | Default | Meaning |
|---|---:|---|
| `sim.backend` | `auto` | `auto` picks warp on a CUDA host and jax elsewhere. `jax` and `warp` pass through. |
| `sim.naconmax_per_env` | `None` | Contact budget per world. `None` defers to the robot's `sim_budget` block in its robot.yaml. Warp allocates ONE pool for the batch, sized `naconmax_per_env * num_envs`. |
| `sim.njmax` | `None` | Constraint-row budget per world, same `None` fallback. Never multiplied by the env count. |
| `sim.num_envs` | `1` | Batch size the pool is sized for. `train.py` overwrites it with the larger of `ppo.num_envs` and `ppo.num_eval_envs`. |

Both overflows are silent. Contacts past `naconmax` are dropped; rows past
`njmax` apply no force, and nothing warns anywhere — no counter reports it and
no exception is raised, so a run just trains against a robot whose feet half
pass through the floor. That is why the budgets are fail-closed: a warp env
whose robot records no `sim_budget` (and whose sim config sets none) refuses
to construct.

The budgets are measurements OF a robot's collision geometry, so they live
with the robot: each `robots/<name>/robot.yaml` records the
`./run.sh check-contacts` worst case times the 7× headroom rule, with the
measurement provenance in a comment. The headroom rule itself is inherited
from the quadruped predecessor and has not been re-derived for these
robots' fallen-regime contact patterns.

Two facts for the debugging that follows a resize. The pool is a real
device-memory line item: 224 per env at 4096 envs is a 917,504-contact
allocation, and a 4096-env job has run out of device memory on a 256 pool.
And one MJX step is not batch-shape invariant on the jax CPU backend, so a
batched-versus-sequential parity check has to compare integer outcomes,
never floats.

`tests/integration/test_check_contacts.py` discovers every robot directory
and fails if new collision geometry outgrows the recorded budgets.

### The `contacts` block

`run.json` and `battery.json` both carry a `contacts` block with identical
keys on every backend, so a remote GPU run and a local CPU run diff without
branching:

| Field | Meaning |
|---|---|
| `backend` | `jax` or `warp`, as resolved for that run. |
| `nacon_max` | Peak contacts in one world, or `null` if nothing measured it. |
| `nefc_max` | Peak constraint rows in one world. Always `null` on jax: its `_impl.nefc` is a static buffer size, not a count. |
| `naconmax_per_env`, `njmax`, `num_envs` | The budgets the run was configured with. |
| `pool` | `naconmax_per_env * num_envs`, the device allocation. |
| `overflow` | `backend == "warp"` and `nacon_max >= naconmax_per_env`. |
| `rows_overflow` | `backend == "warp"` and `nefc_max >= njmax`. |

`battery.json`'s peaks come from the battery's own rollouts, sampled every
step. `run.json`'s come from the `contact_preflight` probe: brax's PPO loop is
one jitted scan, so no Python-side code holds an `mjx.Data` while training
runs and the live counters are unreachable from there. The probe measures the
same per-world peaks on the same backend, before the job spends GPU hours.

## Eval battery metrics (`battery.json`)

`./run.sh battery` writes one entry per scenario. Every field added by port
items 4.1 to 4.3 is **additive**: no pre-existing field changed meaning, and
none of the new numbers folds into a score, a gate, or the `fell` / `steps`
logic. They are raw readings.

### The rollout record

`rollout` in `eval/battery.py` returns one record per scenario: a dict of
numpy arrays with one row per control step. `RECORD_FIELDS` in the same file
is the one schema for it. In the shapes below, `F` is the foot count, `nu` is
the actuator count and `n_act` is the actuated-joint count.

| Key | Per-step shape | dtype | Frame | Quaternion source | What it holds |
|---|---|---|---|---|---|
| `cmd` | `(3,)` | float32 | command | - | The scripted `[vx, vy, wz]` command for the step. |
| `vx` | `()` | float64 | body | IMU | Forward base velocity, m/s. |
| `vy` | `()` | float64 | body | IMU | Lateral base velocity, m/s. |
| `wz` | `()` | float64 | body | - | Body gyro z channel, rad/s. |
| `height` | `()` | float64 | world | - | Base height from the free joint's qpos, m. |
| `qvel` | `(n_act,)` | float32 | joint | - | Actuated joint velocities. |
| `contact` | `(F,)` | bool | foot | - | Each foot touches the floor. |
| `foot_speed` | `(F,)` | float32 | world | - | Horizontal foot speed, m/s. |
| `tau` | `(nu,)` | float32 | actuator | - | Actuator force. |
| `foot_clear` | `(F,)` | float32 | world | - | Sole height above the floor, m. |
| `foot_vz` | `(F,)` | float32 | world | - | Vertical foot velocity, m/s. |
| `ctrl` | `(nu,)` | float32 | actuator | - | The servo setpoint. |
| `qpos` | `(n_act,)` | float32 | joint | - | Actuated joint angles. |
| `base_pos` | `(3,)` | float32 | world | - | Position of the free-joint body, from its `xpos`. |
| `base_quat` | `(4,)` | float32 | world | IMU | Base orientation, wxyz. |
| `foot_pos` | `(F, 3)` | float32 | world | - | Foot site positions, from `site_xpos`. |
| `knee_q` | `(F,)` | float32 | joint | - | Each foot's knee angle, in foot order. Optional. |

Row `i` is the state after control step `i`. When a step trips `done`, the
record keeps that step's row and the rollout stops. A metric that needs a
standing robot therefore scores `rec[settle:fell_at]`.

The IMU quaternion is the robot's quat sensor, read through the env's
`_quat`. A robot that declares no quat sensor uses the free joint's qpos
instead. `base_pos`, `base_quat` and `foot_pos` come from the same
forward-kinematics stage.

`knee_q` is the only optional key. It is absent on a robot with no knee group
or with foot sites that carry no side token. Its sign is robot-specific, so
metrics read its magnitude. Per-run constants travel in `rollout`'s second
return value, `meta`. It carries `dt` and `foot_sides`.

`rollout` checks the record against `RECORD_FIELDS` before it returns.
`scenario_result` checks it again and names every missing required key.
`tests/integration/test_battery_record.py` runs the real rollout and checks
each key's shape and dtype. The evals read only the record. They do not
depend on which trainer produced the checkpoint.

### The settle window

`eval/battery.py`'s `SETTLE_SEC = 1.0` is the reset transient every new
metric drops. `settle_steps(dt)` converts it to control steps, which is 50
steps at a `ctrl_dt` of 0.02. `rollout` starts recording on the first step
after reset, and the opening steps are the robot falling into its pose
against a command it has not had time to answer. The gait-shape metrics also
stop before the fall row: they score `rec[settle:fell_at]`. The pre-4.1
metrics (`vel_err_*`, `vibration`, `foot_slip`, `height_*`,
`torque_sat_frac`, `mech_power_mean`, `antiphase_score`) still score the
whole record: narrowing their window would change what an existing field
means.

A scenario that ends inside the window therefore reports `null` (or
`swings: 0`) for every new metric while the older ones still print numbers.
That is the expected reading for an early checkpoint that falls in under a
second, not a broken feature — a 100k-step smoke policy falls at about step
50 on `asimov_v1`, right on the boundary.

### Spin probes

Two scenarios, `spin_left` and `spin_right`, hold a pure yaw command —
`wz = +0.5` and `-0.5` rad/s, no translation — for 6 s. They sit inside the
`±0.6` yaw box with headroom, so a row that fails cannot be excused as a
command-envelope corner the policy was never trained near.

| Field | Meaning |
|---|---|
| `yaw_progress_deg` | Body gyro z integrated over the post-settle window, degrees, signed (`+` is CCW / left). `null` when the row did not outlive the settle window. |
| `yaw_cmd_deg` | The commanded yaw rate integrated over the same window: the row's own denominator. |
| `completed` | The scenario held its full scripted duration. |

Both yaw fields are written on **every** scenario row, not just the two spin
ones — they read the same body gyro every row already records, and `turn`'s
yaw budget is worth the same look. The two spin rows are the ones that exist
to be read side by side.

Why per-direction rows: a policy that turns 140 degrees left and 12 right
averages to a healthy-looking 76. A policy has shipped unable to spin right
because every scenario that turned at all turned left.

The frame is the body gyro, not world yaw. Integrating the rate needs no
unwrapping, so a multi-turn spin cannot alias, and a robot that is not
upright gets the honest number — it cannot spin about an axis it is not
standing on. At `ctrl_dt` 0.02 the post-settle window asks for 2.5 rad
(143 deg), short of a full revolution.

**Not built:** a second probe world that replays the DR-patched contact
physics (feet at `geom_priority = 1`) to tell "the policy unlearned turning"
from "the policy turns only in the physics it trained in". That distinction
only exists once foot-friction DR is actually on in a keeper run.

### Gait KPIs

`eval/gait.py`'s `gait_metrics`, pure numpy over the per-foot clearance and
vertical velocity the rollout records. A **swing** is a contiguous run of
clearance above 5 mm in the post-settle window.

| Field | Meaning |
|---|---|
| `swings` | Scorable swings found, pooled over every foot. The sample count behind all four medians below. |
| `swing_apex_med_m` | Median peak clearance of a swing, metres. |
| `swing_apex_p90_m` | 90th percentile of the same. |
| `touchdown_v_med` | Median downward vertical speed on the last airborne step, m/s. |
| `touchdown_softness_med` | Median of `touchdown_v / sqrt(2*g*apex)`: the touchdown speed over what free fall from that swing's own apex would have delivered. `1.0` is a foot dropped like a brick; lower is a foot flown in. |

Three runs are not counted as swings: shorter than 2 steps (contact noise,
not a step taken), still airborne when the record ends (no touchdown to
measure), and already airborne at the first measured step (the settle window
may have cut the apex off, so the number would be a floor rather than a
peak). With no scorable swing the row reads `swings: 0` and `null` for every
median — an unmeasured apex written as `0.0` would average into a keeper
comparison as though a foot had been measured lying on the floor.

The metrics work for any foot count. They fold into nothing: velocity
tracking error scores both a skimming gait and a stand-and-lift farm as
healthy, which is the gap these two numbers close.

Apexes and the 5 mm airborne band are physical heights above the floor
(`_foot_clearance` reads sole height; see
`docs/lessons/foot-clearance.md`).

### Gait shape

`eval/gait.py`'s `gait_shape`, pure numpy over the record's base pose,
command, foot contacts, foot positions and knee angles. Every key below is on
every scenario row with at least 10 steps. A key reads `null` when nothing
was measured. `<side>` is the foot's side token, described below.

| Field | Meaning |
|---|---|
| `speed_frac` | Displacement along the start heading over the commanded distance. 1.0 is a robot that covered exactly the commanded distance. `null` below 0.1 m of commanded travel. |
| `heading_drift_deg` | Yaw change minus the commanded yaw change, degrees. The yaw is unwrapped over the window, so a missed spin keeps its full size and sign. |
| `lateral_drift_m` | Displacement across the start heading, metres. Positive is left. |
| `support_frac_0`, `support_frac_1`, `support_frac_2` | Share of rows with exactly 0, 1 or 2 feet in contact. |
| `roll_std_deg` | Standard deviation of the roll angle, degrees, read from the body-frame gravity vector. |
| `pelvis_sway_cm` | Standard deviation of the base's lateral position, centimetres, after its fitted straight line is removed. The line takes out a steady drift. `null` with fewer than 3 rows. |
| `step_length_m` | Mean of the landing foot's x minus the other foot's x, metres, over every scored landing. It needs exactly two feet. `null` without a scored landing. |
| `landing_x_m_<side>` | Mean forward position of the foot at its scored landings, metres, in the base frame. `null` without a scored landing. |
| `landing_y_m_<side>` | Mean lateral position of the foot at its scored landings, metres, in the base frame. Signed, with left as positive. A sign change shows a foot crossing the midline. `null` without a scored landing. |
| `knee_td_rad_<side>` | Mean knee angle magnitude at the foot's scored landings, radians. `null` without a knee group or without a scored landing. |

The window is `rec[settle:fell_at]`: it starts after the settle window and
ends at the step before a fall. The travel numbers measure from the first
scored row to the last one. The travel and heading metrics are `null` with
fewer than 2 scored rows. The support and roll metrics are `null` for an
empty window.

The heading comes from the IMU quaternion. The base position is the
free-joint body's `xpos`. Foot positions are read relative to the base
position and rotated into the base frame by the same quaternion.

A landing is a contact rising edge: a row where a foot is in contact and was
not on the row before. Each foot's first landing in the window is dropped,
because it lands from the reset pose. The rest are its scored landings. This
detector is separate from the Gait KPIs' clearance detector, and the two
counts can differ.

The side tokens come from the `foot_sites` names in `robot.yaml`. A foot is
`left` or `right` when exactly one of the two words occurs in its site name.
The match is case-sensitive. If any name has neither or both, or two feet
share a side, the feet are named `foot0`, `foot1`, and so on.

`knee_td_rad_<side>` takes the magnitude because knee signs differ by robot.
Asimov's right knee flexes negative, for example.

These metrics fold into nothing. No score, gate or PASS/ATTENTION line reads
them.

Speed fraction, drift and sway read cleanest on straight commands. On a turn
the path curves away from the start heading, and the curve dominates the
sway. On strafe the lateral drift is the commanded sideways travel, and the
speed fraction is null because no forward distance is commanded. The
battery's scenarios are scripted, with ramps and stops, so their numbers
differ from those of a probe that holds one constant speed.

### Servo tracking error

| Field | Meaning |
|---|---|
| `tracking_err_rms` | RMS of `\|ctrl - qpos\|` over the post-settle window, pooled across steps and actuated joints. |
| `tracking_err_p95` | 95th percentile of the same. |

`ctrl` is the setpoint the servo was asked to hold, `qpos` the angle the
joint reached. This is what actuator stiffness work targets, and it is
invisible to every velocity metric: a policy can hit its commanded body
velocity with every joint sagging behind its setpoint. The p95 says whether
the error is spread evenly or lives in a few joints — an RMS over twelve
joints hides one that has given up. Both are `null` when the row did not
outlive the settle window.

Only a position-servo preset makes this a servo error. Under `ideal_torque`,
`ctrl` is a torque in Nm and the subtraction is dimensionally meaningless.
`run.json`'s `actuator_gains.model` is what tells a reader which preset
produced the run. Every preset shipped today resolves to `pd` — both robots'
`deploy_pd` and `sizing_ideal`, and asimov's `encos_datasheet` — so the
caveat is future-proofing, not a live footnote.

### The `actuator_gains` block (`run.json`)

`run.json` carries two actuator records. `actuators` is what the config asked
for: `cfg.actuators` verbatim, preset name and `overrides` included.
`actuator_gains` is what the **built model got**, read back off its actuator
params after preset loading and after `actuators.overrides` merging.

| Field | Meaning |
|---|---|
| `preset` | The preset name, `cfg.actuators.name`. |
| `model` | The actuator model the preset resolved to: `pd`, `ideal_torque`, … |
| `joints` | Actuated joints in canonical order. This is also the column order of `kp` and `kd`. |
| `kp` | Per actuator, `actuator_gainprm[:, 0]`. |
| `kd` | Per actuator, `-actuator_biasprm[:, 2]`. |

The two blocks differ whenever an override patches a gain: an
`actuators.overrides` entry never appears in the preset yaml, so a stamp read
from the yaml would record numbers the run never used.

For a `pd` preset those params **are** the PD gains —
`actuators/models.py`'s `PositionPD.inject` writes `gainprm = (kp, 0, 0)` and
`biasprm = (0, -kp, -kd)`. For `ideal_torque` they are not gains at all:
`gainprm[0]` is `1.0` and there is no bias term, so the block reads `1.0`
and `0.0`. It is stamped anyway. `model` is what makes the numbers readable,
and a `run.json` whose shape depended on the actuator model would need
branching at every reader.

There are no runtime `pd_kp` / `pd_kd` override knobs. The actuator-preset
axis already covers that: a different stiffness is a different preset, or an
`actuators.overrides` entry on the group that needs it, and both land in this
block.

## Eval videos

`./run.sh eval --run runs/<name>` renders one scenario to MP4. A battery
scenario uses the same scripted command trajectory `battery.json` measures.
Its clip and its battery row describe the same trajectory.

`long` is a video-only scenario. It runs for 40 s: a stand, speed-ups to 0.6
and then 0.9 of the vx maximum, a slow-down, a left turn and a right turn at
0.5 rad/s, a walk and a stop. Its vx values scale with the robot's
`command.vx` maximum. Its wz stays 0.5 rad/s, clipped to the robot's
`command.wz` range. It has no battery row.

`--compare` renders the open-loop rollout next to the heading-closed one in
one clip. It loads the checkpoint once and runs two rollouts at the same
seed, scenario and checkpoint. The left half runs open-loop. The right half
runs the outer heading loop at `--heading-gain`, which defaults to 1.5 here.
The labels in the top-left corners read "open loop" and "heading loop, gain
G". The shorter half repeats its last frame, so the side that keeps walking
stays visible after the other one falls. The default output is
`runs/<name>/<scenario>_compare.mp4`. The loop's added wz is a turn command.
A policy trained with `heading_err` observed re-anchors its `yaw_ref` to the
measured yaw while the command's wz exceeds 0.05 rad/s. So in the right half
that policy's own `heading_err` observation reads near zero.

| Flag | Default | Effect |
|---|---|---|
| `--scenario NAME` | `walk_ramp` | Any `eval/battery.py::battery_scenarios` name, or `long`. |
| `--steps N` | the scenario's own length | Truncates the rollout. |
| `--video-size WxH` | `640x480` | Rendered frame size. When the model's offscreen buffer is smaller, `eval/render.py`'s SceneView widens a private model copy, so any size works without touching the robot XML. The stacked `--plot-*` panels follow the frame width. |
| `--overlay-torque` | off | A signed bar per actuator drawn into the frame itself (`eval/overlays.py`), normalized by that joint's own cap, one colour per joint group, red lines at ±1. The instantaneous view of the same signal `--plot-torque` traces over the episode: saturation is visible at the moment it happens. |
| `--plot-torque` | off | A normalized-torque strip under the render: every joint's torque over its own actuator cap, one colour per joint group, dashed lines at ±1. Normalized because this robot's per-joint force ranges are heterogeneous — a single N·m cap line across a hip and an ankle means nothing. |
| `--plot-joints` | off | A per-joint target-vs-state grid: one row per joint group, one column per side, achieved position solid and the policy's target dashed. |
| `--joint NAME` | — | Swaps the grid for a single-joint zoom panel. Implies `--plot-joints`. |
| `--push` | off | Restores the run's own random pushes. |
| `--heading-gain G` | `0`, or `1.5` with `--compare` | The default 0 renders open-loop. Otherwise an outer heading loop adds gain times the wrapped heading error to the scenario's wz, clipped to 0.5 rad/s. Yaw comes from the IMU quaternion through `signals.base_yaw`. The loop's added wz is a turn command, so a heading-trained policy re-anchors its `yaw_ref` to the measured yaw while the command's wz exceeds 0.05 rad/s. |
| `--compare` | off | Renders the open-loop rollout and the heading-closed one side by side in one clip. The gain comes from `--heading-gain`, default 1.5. `--compare --heading-gain 0` is a usage error. |
| `--checkpoint STEP` | the latest checkpoint | Renders the checkpoint saved at this step. Pass the step a battery measured to render that same policy. The final print line names the checkpoint that loaded. |

**Rows of the joint grid share a y-range** (`sharey="row"`). That is what
makes left/right asymmetry readable: per-axes autoscaling would rescale each
column to fill its own box, and a left knee swinging four times as far as
the right would look identical to it.

**Rollouts are push-free by default.** A mid-video kick reads as a policy
failure to anyone watching the clip, and the battery disables pushes for the
same reason. `eval/video.py` states this itself rather than inheriting it,
so the two conventions can be changed independently.

Panels are opt-in: a plain render pays none of their per-step device
transfers. Each is drawn ONCE for the whole episode and replayed with a
moving cursor column stamped in, so video assembly costs one matplotlib pass
per panel instead of one per frame.

**GL backend.** `eval/video.py` sets `MUJOCO_GL=egl` on **linux only** (via
`setdefault`, so an exported value wins). macOS has no EGL and forcing it
there breaks offscreen rendering, so darwin keeps its default (CGL). Only
the darwin path has been exercised in this repo; treat linux/egl as untested
until a GPU-box run confirms it.

## Early stopping (`early_stop`)

Off by default. When on, the trainer ends a run whose eval reward has stopped
climbing. The rule is `plateau_stop` in `src/humanoid_lab/train.py`, a pure
function of the eval rewards seen so far:

- A reward is a new best only when it beats the running best by **more** than
  `min_delta`. A gain of exactly `min_delta` does not count.
- A plateau is `patience` consecutive evals with no new best.
- The rule returns no verdict until `max(min_evals, patience + 1)` evals
  exist.

The progress callback appends each eval reward to a list and raises
`EarlyStop` when the rule fires. `main()` catches it around the `ppo.train`
call. Brax writes a checkpoint at every eval, so the newest checkpoint in
`runs/<name>/checkpoints` is the early-stopped policy, and the reported
metrics come from the last completed eval.

`run.json` carries two fields whether or not the feature is on:
`early_stopped` (bool) and `stopped_at_steps` (the last eval's step count,
which on a completed run is the final eval's).

Patience counts evals, not steps, so `ppo.num_evals` sets how much training
each unit of patience buys. At the default 100M-step budget with brax's
`num_evals`, one eval is several million steps.

**Calibrate `min_delta` above the eval noise before trusting it.** A
`min_delta` inside the noise band lets noise reset the patience clock and
the run never stops. The defaults are uncalibrated starting numbers:
measure the eval noise first by evaluating one checkpoint repeatedly, and
raise `patience` for overnight runs.

## run.json

Every training run writes `runs/<run_name>/run.json`. Both trainers build it
with `run_record.build_run_record` and write it with `run_record.write_run`.
Every tool that reads it goes through `run_record.load_run`.

**Keys**, in order: `run_name`, `task`, `num_timesteps`, `early_stopped`,
`stopped_at_steps`, `final_reward`, `checkpoint_dir`, `contacts`,
`env_config`, `ppo_config`, `hydra_config`, `actuators`, `actuator_gains`,
`trainer`, `warm_start`. The first thirteen are the keys origin/main wrote.

- `hydra_config` is the composed, resolved config of this run.
- `env_config` comes from the built env.
- `ppo_config` holds the PPO params, `network_factory` and the repo-owned
  keys such as `sym_coef`.
- `contacts` is described in [The `contacts` block](#the-contacts-block).
- `actuators` and `actuator_gains` are described in
  [The `actuator_gains` block](#the-actuator_gains-block-runjson).

**The `trainer` block.**

| Field | Meaning |
|---|---|
| `name` | `brax_ppo` or `torch_warp_ppo`. |
| `physics` | The physics backend the run stepped: `jax` or `warp` for Brax, `warp` for Torch. |
| `versions` | The installed versions of brax, jax, torch, warp-lang, mujoco and mujoco-mjx. A package that is not installed reads `null`. |
| `device` | The device the trainer ran on, for example `TFRT_CPU_0` or `cuda:0`. |

The Torch trainer adds its own fields: the `torch_warp` block (critic
warm-up, checkpoint cadence, `max_minutes`), live contact peaks,
`train_reward_last` and `resumes`. See
[Torch trainer](#torch-trainer-configstorch_warpyaml).

**The `warm_start` block.** It is `null` for a run without `restore`.
Otherwise it holds:

| Field | Meaning |
|---|---|
| `checkpoint` | The restored checkpoint, as a portable path. |
| `source_run` | The run dir above `checkpoints/<step>`, as a portable path. |
| `source_results` | The source run's `final_reward`, `stopped_at_steps`, `early_stopped` and `num_timesteps`. |
| `config_diff` | The leaves of `task.env`, `actuators` and `ppo` that differ from the source run's `hydra_config`, as `{"<dotted key>": [old, new]}`. A leaf on one side only reads `null` on the other. |

When the source run cannot be read, the Brax trainer still trains and sets
`source_run`, `source_results` and `config_diff` to `null`.

**Results.** `early_stopped`, `stopped_at_steps` and `final_reward` describe
this run only.

- Brax reads them from the last eval. `final_reward` is brax's
  `eval/episode_reward`. `stopped_at_steps` is the last eval's step count.
  `early_stopped` is true when [early stopping](#early-stopping-early_stop)
  cut the run.
- Torch has no eval, so `final_reward` is `null`. `stopped_at_steps` is its
  last training step. `early_stopped` is true when `max_minutes` or Ctrl+C
  ended the run.

**Write timing.**

- Brax writes run.json once, at the end of training or after an early stop.
  `jobs/train_chain.sh` reads a fresh run.json as proof that phase A
  finished.
- Torch writes it at start with null results and again at the end.

Both write `run.json.tmp` and move it over `run.json` with `os.replace`, so
a reader never sees a half-written file.

**Paths.** `checkpoint_dir` and `hydra_config.restore` are stored relative to
the project dir, in POSIX form, when they lie inside it. A path outside it
is stored absolute, still in POSIX form. `load_run` resolves a relative
`checkpoint_dir` against the project dir. The file on disk is not rewritten.

**No `warp_*` keys.** Neither trainer writes a top-level `warp_*` key. Every
env knob lives under `hydra_config.task.env`. A run written by the old Torch
trainer carries `warp_*` keys, and `load_run` migrates them into
`hydra_config.task.env` (see [Export](deploy.md#the-export)).

## Torch trainer (`configs/torch_warp.yaml`)

The Torch trainer in `src/humanoid_lab/torch_warp/` composes
`configs/torch_warp.yaml`. That file is `config.yaml` plus two Torch-only
blocks. Every other key, every config group and every `experiment=` preset
works as it does for the Brax trainer. `config.yaml` rejects both blocks, so
the Brax trainer never sees them.

**The `torch_warp` block.**

| Key | Default | Meaning |
|---|---|---|
| `device` | `cuda:0` | The Torch and warp device. A CPU run happens only with `cpu` set, and it is slow. |
| `ckpt_minutes` | `10` | Wall-clock minutes between checkpoints. |
| `max_minutes` | `null` | Stop after a checkpoint once this many minutes have passed. `null` runs to `ppo.num_timesteps`. |
| `critic_warmup_iters` | `0` | Iterations at the start that train only the value network. |
| `resume` | `false` | Continue `runs/<run_name>` from its `trainer_state.pt`. |

**The `bench` block.** Only `python -m humanoid_lab.torch_warp.bench`
(`./run.sh bench-warp`) reads it.

| Key | Default | Meaning |
|---|---|---|
| `iters` | `3` | Timed PPO iterations. |
| `physics_steps` | `100` | Timed physics-only steps. |
| `out` | `null` | The JSON output path. `null` writes `runs/bench/bench_<timestamp>.json`. |

The [package README](../src/humanoid_lab/torch_warp/README.md) covers the
Windows install, the train, warm-start and resume commands, the refusal
ledger for the `ppo` and `network` keys, and the parity suite.

**run.json.** A Torch run writes the same keys as a Brax run (see
[run.json](#runjson)). It differs in these ways:

- `trainer.name` is `torch_warp_ppo` and `trainer.physics` is `warp`.
- The trainer block adds `torch_warp`, `constants_env_overrides`, `ignored`,
  `budget_peaks`, `train_reward_last` and `resumes`.
- run.json is written at start with null results and again at the end.
- `final_reward` is `null`, because the Torch trainer has no eval loop.
- `env_config` records `sim.backend: warp` and the run's `sim.num_envs`.

## `run.sh` verbs

Read from `run.sh` as it stands today. `python` below is `.venv/bin/python`.
When that is missing and `.venv/Scripts/python.exe` exists, as in a Windows
venv under Git Bash, run.sh uses that one.

| Verb | Runs | Notes |
|---|---|---|
| `train` | `python -m humanoid_lab.train` | Full Hydra CLI available after it. |
| `smoke` | `JAX_PLATFORMS=cpu python -m humanoid_lab.train smoke=true wandb.enable=false` | CPU pipeline check. |
| `train-warp` | `python -m humanoid_lab.torch_warp.ppo` | The Torch+MJWarp trainer. It composes `configs/torch_warp.yaml`, which is `config.yaml` plus the `torch_warp` and `bench` blocks. Full Hydra CLI available after it. Needs the `torch_warp` extra. See `src/humanoid_lab/torch_warp/README.md`. |
| `bench-warp` | `python -m humanoid_lab.torch_warp.bench` | Times the Torch+MJWarp env and PPO loop and writes one JSON file to `bench.out`, or `runs/bench/bench_<timestamp>.json` when it is null. Same config as `train-warp`. |
| `build` | `python -m humanoid_lab.build_model` | `--robot NAME --preset NAME [--out PATH] [--set PATH=VALUE ...]`. Writes `robots/<robot>/mjx/<preset>.xml`. `--set` requires `--out`, so an ad-hoc override build never overwrites the canonical preset build. |
| `check` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_model` | `--robot NAME --preset NAME [--steps N] [--xml PATH] [--skip-mjx] [--max-qvel N] [--set PATH=VALUE ...]`. Gate-checks every keyframe for NaN and for `|qvel|` blowup. `--set` forces an in-memory build even if a prebuilt XML exists, and is mutually exclusive with `--xml`. |
| `check-contacts` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_contacts` | `--robot NAME --preset NAME [--steps N] [--seeds N] [--seed N] [--out PATH]`. Measures the per-world contact and constraint-row peaks over three regimes and prints the budgets they need. See [Warp contact budgets](#warp-contact-budgets-taskenvsim). |
| `check-friction` | `python -m humanoid_lab.check_friction` | `--robot NAME --preset NAME [--backend auto\|warp\|jax] [--num-envs N] [--range LO HI]`. Verifies end to end, on the box's own backend, that a `dr.foot_friction` draw is the friction inside each foot-floor contact. Exits nonzero on any mismatch. See [Domain randomization](#domain-randomization-dr). |
| `test` | `python -m pytest tests/unit -q` | The fast suite: model-free, runs in seconds. `tests/unit/test_suite_split.py` fails if a test here builds or steps a model. |
| `test-slow` | `python -m pytest tests/integration -q` | The slow suite: builds models, steps MJX. Exports `JAX_COMPILATION_CACHE_DIR` (default `.jax_cache`) so re-runs skip XLA compilation. |
| `test-parity` | `JAX_PLATFORMS=cpu python -m pytest tests/parity -q` | `[--parity-report] [--parity-device cuda:0]`. Checks the Torch+MJWarp runner against the JAX env. It runs the shared task math on Torch tensors. It runs WarpJoystick in lockstep with the wrapped JAX env, with the physics and the random draws replayed from JAX. It steps MJX on JAX and MJWarp one control step from identical states. It checks that identical draws give identical DR fields, and that each backend's draws repeat, stay independent and stay in range. It runs WarpJoystick in a closed loop on MJWarp. The tolerances are in `tests/parity/tolerances.py`. `--parity-report` prints the worst error of each check. `--parity-device cuda:0` runs the Torch side of the task-math, lockstep, physics, DR, RNG and closed-loop tests on a GPU, with the provisional CUDA physics tolerance. The trainer, checkpoint, run-record and glue tests keep Torch on CPU, and JAX stays on CPU. Skips when torch is not installed. Same compile cache as `test-slow`. |
| `test-all` | `python -m pytest tests/unit tests/integration -q`, then `JAX_PLATFORMS=cpu python -m pytest tests/parity -q` | All three suites. Parity runs as a second process, because `JAX_PLATFORMS` must be set before the first jax import. Arguments go to the first process only. Same compile cache as `test-slow`. Use before merging. |
| `sizing-collect` | `JAX_PLATFORMS=cpu python -m humanoid_lab.sizing.collect` | `--run runs/<name> [--episodes N] [--steps N] [--seed N]`. Rolls the checkpoint out on CPU and writes `<run>/sizing_data.npz`. |
| `sizing-report` | `sizing.collect` then `python -m humanoid_lab.sizing.report` | `--run runs/<name> [--episodes N] [--steps N] [--seed N] [--motors NAME] [--recollect]`. Skips the collect step if `<run>/sizing_data.npz` already exists, unless `--recollect` is passed. Writes `<run>/sizing_report.md` and `<run>/sizing_scatter.png`. |
| `battery` | `JAX_PLATFORMS=cpu python -m humanoid_lab.eval.battery` | `--run runs/<name> [--out PATH] [--set BLOCK.KEY=VALUE ...] [--checkpoint STEP]`. Writes `<run>/battery.json` unless `--out` says otherwise. `--set` re-scores the checkpoint under a changed `task.env` value (e.g. `obs_noise.joint_vel=1.75`), merged one level deep over the measurement env; VALUE is read as YAML. It requires an `--out` other than `<run>/battery.json`, so a re-scored variant never overwrites the run's own table, and the variant records the overrides under `env_overrides`. `--checkpoint STEP` measures the checkpoint saved at that step instead of the latest, and `battery.json`'s `checkpoint` field names the one measured. |
| `report` | `python -m humanoid_lab.eval.report`, then `sizing.report` if `<run>/sizing_data.npz` exists | `--run runs/<name> [--out PATH]`. Renders `<run>/eval_report.md` from `battery.json`. |
| `eval` | `JAX_PLATFORMS=cpu python -m humanoid_lab.eval.video` | `--run runs/<name> [--scenario NAME] [--steps N] [--out PATH] [--seed N] [--video-size WxH] [--overlay-torque] [--plot-torque] [--plot-joints] [--joint NAME] [--push] [--heading-gain G] [--compare] [--checkpoint STEP]`. Renders one battery scenario, or the video-only `long`, to MP4. `--compare` puts the open-loop and heading-closed rollouts side by side. `--checkpoint STEP` renders the checkpoint saved at that step instead of the latest. See [Eval videos](#eval-videos). |
| `export` | `JAX_PLATFORMS=cpu python -m humanoid_lab.export.policy` | `--run runs/<name> [--out DIR]`. Writes `policy.npz` and `policy_meta.json` into `<run>/deploy` unless `--out` says otherwise. Both round-trip validations run before either file is placed. See [deploy.md](deploy.md). |

## Configs compose only from the editable install

`train.py` uses `@hydra.main(config_path="../../configs", ...)`.
`torch_warp/ppo.py` and `torch_warp/bench.py` use `"../../../configs"`, one
level deeper. Hydra resolves that path relative to the module's own file
location on disk. The wheel only packages `src/humanoid_lab`, per
`pyproject.toml`'s `tool.hatch.build.targets.wheel.packages`. `configs/`
never ships inside a built wheel. Config discovery only works from an
editable install (`pip install -e .`) of a full checkout. Only then does the
relative path resolve back to the repo root's `configs/`. A wheel installed
somewhere else will fail to find any config group.

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
