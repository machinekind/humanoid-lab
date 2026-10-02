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
| `restore` | `null` | Checkpoint directory to warm-start from. Relative paths resolve against the repo root. A checkpoint whose critic list differs from the env's restores through `restore.py`. Shared critic columns keep their statistics and weights. Added columns get zero weights and their component's prior in the normalizer. The height scan's prior is mean 0.014 m and std 0.066 m, measured on the default arena (`envs/height_scan.py`). A component without one starts at mean 0 and std 1. The prior carries the weight of the source's whole sample count, so it holds long into the run. Removed columns are dropped. The actor list must match. The source's `network.policy_obs_key` must match this run's, and so must its `value_obs_key` unless `restore_value=false`. A `ppo.normalize_observations_mode` other than `welford`, in this run or the source's, refuses: brax's checkpoint load always rebuilds a Welford normalizer. `run.json`'s `restore` block records the plan and the priors. |
| `restore_value` | `true` | Restore the checkpoint's value network with its policy. `false` starts a fresh critic and keeps the restored normalizer. |
| `domain_rand` | `false` | Gates the whole `dr` block. `false` with any `dr.*.enable=true` raises at startup rather than silently ignoring the request. |
| `contact_preflight` | `true` | Measure the warp contact/constraint peaks on a short probe before training and record them in `run.json`. Skipped automatically under `smoke=true`. See [Warp contact budgets](#warp-contact-budgets-taskenvsim). |
| `wandb.enable` | `true` | Log to Weights & Biases if import/login succeeds. |
| `wandb.project` | `humanoid-lab` | W&B project name. |
| `wandb.group` | `null` | W&B run group. Defaults to the selected experiment's name, or stays `null` if no experiment is selected. An explicit value wins over that default. |
| `ppo` | `{}` | Global PPO overrides, applied after the task's own `task.ppo` block. CLI `ppo.foo=...` wins over both. |
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
| `tracking_product` | `false` | Multiply the two kernels into each other: `k_lin, k_ang = k_lin*k_ang, k_ang*k_lin`. Additive tracking pays the easy half of a command — a robot that ignores a pure-spin command still earns the full `tracking_lin_vel`, since standing still tracks the zero linear command perfectly (measured at about 63% of an ideal spin's payout). With the product, full pay needs the whole command tracked. |
| `tracking_relative` | `false` | Score the fraction of the command tracked instead of the absolute error: the width becomes `tracking_rel_sigma * max(\|cmd\|, floor)²`. The absolute kernel pays only within about `√tracking_sigma` of the target whatever the target's size, so a fast command's reward cliff is out of exploration's reach — one measured policy reached 0.70 m/s under a 0.8 command and 0.00 m/s under a 1.0 one. |
| `tracking_rel_sigma` | `0.25` | Dimensionless width of the relative kernel. A quadruped starting point; a narrow kernel rounds partial tracking to zero, and on terrain this had to widen to `0.5`. |
| `tracking_rel_floor_lin` | `0.3` | Floor on the linear relative denominator, m/s. Keeps a near-zero command from sharpening the kernel to a point and dividing by zero. |
| `tracking_rel_floor_ang` | `0.4` | Floor on the angular relative denominator, rad/s. Same role; on terrain this had to widen to `0.7`. |
| `tracking_far_weight` | `0.0` | Mix a wide exponential into both kernels: `(1-w)*kernel + w*exp(-err²/tracking_far_sigma)`. Applies in the absolute and the relative branch alike, and the far kernel stays absolute in both. `exp(-err²/σ)` is gradient-free a few sigma out, so a capability the policy never explored gets no pull toward the command; the wide kernel keeps a usable gradient at range without moving the optimum or leaving `[0, 1]`. **This term alone creates a standing deadlock**: at a yaw rate error of 0.8 rad/s it pays `0.25*exp(-0.64/2.5)`, about 19% of the maximum angular reward, for standing still, and that gradient is weaker than the penalties a pivot attempt incurs. Turn it on only together with `tracking_product` or `tracking_relative`. |
| `tracking_far_sigma` | `2.5` | Width of the far kernel, in (m/s)² and (rad/s)². Ten times `tracking_sigma`. |
| `shaping_tracking_gate` | `false` | Multiply the positive gait-shaping terms by the linear tracking kernel, post-product when `tracking_product` is on. Those terms otherwise pay on a commanded env whether or not it translates, which has made stand-and-lift the top income under a command on a quadruped run: standing with one leg raised earned about 1.8 reward per step against honest walking's 0.25. Gated set: `feet_air_time` and `feet_apex`. `feet_phase` stays ungated — it is the clock-following gradient and has to survive at zero tracking, because stepping is how tracking starts. Stand-still penalties keep their `~moving` mask and are untouched. |

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
| `scales.feet_landing` | `0.0` | Weight of the soft-landing penalty (negative when on). `0` = off. |
| `apex_target` | `0.05` | Swing peak the apex reward asks for, m. Clipped at: the term prices reaching the target, not exceeding it. **Re-derive for this leg** — a quadruped starting value, and our own `gait.swing_height` asks for 0.08 m. |
| `glide_height` | `0.03` | Height band the landing penalty acts in, m. **Re-derive** with `apex_target`. |

`reward.stand_still_vel_weight` (default `0.2`) sets stand_still's
velocity-damping share against a position share of 1; only the ratio
matters, since `scales.stand_still` prices the sum.

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
the env. On a flat config the trainer's wrapping is `wrap_for_brax_training`.
It ends in `BraxAutoResetWrapper(full_reset=False)`. On done it restores
`data` and `obs` from the cached first state and returns `state.info`
untouched. A terrain config gets `envs/terrain_wrapper.py`'s stack. Its
auto-reset rewrites only the curriculum's info keys. `progress_ema` and
`steps_since_cmd` survive a respawn on both stacks. A cut env would
therefore come back carrying the dying episode's shortfall and a
`steps_since_cmd` well past `grace_sec`. It would be armed on its first step
and dead again within a second. `envs/wrappers.py`'s `ProgressReseedWrapper`
puts `progress_ema` back at the command's demand and `steps_since_cmd` back
to 0 on done. `train.py` layers `ProgressReseedWrapper` on whichever stack
applies, exactly when `no_progress.enable` is set. A flat config with the
cut off gets `wrap_for_brax_training` itself, unchanged. Reseeding only the
EMA and carrying the counter over would re-arm the cut on the respawn's first
step, so the counter is zeroed too. Any other wrapper that restarts an
episode in place owns the same reseed.

## Pure command draws (`task.env.command`)

The command sampler draws `(vx, vy, wz)` from one uniform box. That box
almost never produces a clean corner: a backward command arrives with random
lateral and yaw contamination attached, and under `tracking_product` or
`tracking_relative` a contaminated corner pays about nothing however well the
robot serves it. The skill is then never profitable to learn, and the policy
settles on refusing it — a quadruped policy trained this way held 0.000 m/s
under a commanded -0.4 backward, and five isolating probes confirmed the
refusal was learned rather than mechanical.

The five draws below rewrite the base sample into a clean single-axis
command with the given probability. They apply in the order `wz, vy, slow,
fast, back`, a later draw overwriting an earlier one, and all of them run
before `zero_prob`, which stays the sampler's last word: standing still
overrides every draw.

Each draw is gated on its static probability and keys off
`jax.random.fold_in(rng, 0x100 + idx)` with an index of its own — `1 wz,
2 vy, 3 slow, 4 fast, 5 back`, fixed. So a draw at probability 0 does not
exist in the trace, all five off leave the sampler bit-identical to the
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

Every armed redraw is checked against the **composed** `command` box when the
env is constructed: `command.<range>` must sit inside `command.<axis>`, or
`Joystick.__init__` raises a `ValueError` naming the draw, the range and the
box. A redraw is a redraw of the axis, not a widening of it — a policy that
trained on commands outside its own box would ship a contract that
understates what it saw. `deploy_contract.py` refuses the same configuration
at export; the construct-time check just moves the failure to before the GPU
hours. Draws at probability `0.0` are not checked, so the shipped all-off
defaults validate nothing.

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

On a terrain model the floor is every ground geom: the heightfield, the
arena boxes and the aprons. Each world draws one `floor_friction` value and
writes it to all of them. The ground geoms copy the floor plane's priority
0, so `foot_friction`'s priority 1 still wins. `--task terrain` runs the
same probe on the CPU terrain arena.

"Independent" is a property of the RNG plumbing, not a wish. The fixed
distribution draws from `r1..r5 = jax.random.split(rng, 5)`; each switch
above draws from `jax.random.fold_in(rng, 0x100 + idx)` with an index of its
own — `1 joint_gains, 2 com_offset, 3 dof, 4 foot_friction, 5
motor_strength`, fixed. The `0x100` offset is the same load-bearing constant
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
with no kwargs on the jax branch, so changing any of the three budgets
cannot move a jax rollout by a bit.

| Key | Default | Meaning |
|---|---:|---|
| `sim.backend` | `auto` | `auto` picks warp on a CUDA host and jax elsewhere. `jax` and `warp` pass through. |
| `sim.naconmax_per_env` | `None` | Contact budget per world. `None` defers to the robot's `sim_budget` block in its robot.yaml. Warp allocates ONE pool for the batch, sized `naconmax_per_env * num_envs`. |
| `sim.njmax` | `None` | Constraint-row budget per world, same `None` fallback. Never multiplied by the env count. |
| `sim.naccdmax_per_env` | `None` | CCD scratch slots per world for convex pairs. Heightfield pairs and box-box pairs are convex. Warp allocates ONE pool for the batch, sized `naccdmax_per_env * num_envs`. It allocates it on every collision call, outside the XLA pool. `sim_budget.ccd_slot_bytes` gives the bytes per slot. `None` sizes it to the naconmax pool. It has no robot.yaml fallback. It must not exceed `naconmax_per_env`. MJWarp refuses a larger value in `make_data`, and the terrain task refuses it at construction. A pair past the pool is dropped, and MJWarp prints `CCD overflow`. Neither robot's flat model has a convex pair, so a flat run allocates no CCD scratch. |
| `sim.num_envs` | `1` | Batch size the pool is sized for. `train.py` overwrites it with the larger of `ppo.num_envs` and `ppo.num_eval_envs`. |

Neither overflow raises. Contacts and broadphase candidate pairs past the
`naconmax` pool are dropped. Rows past `njmax` apply no force, so a run trains
against a robot whose feet half pass through the floor. MJWarp checks both
budgets when a step advances time. A counter past its buffer then prints
`narrowphase overflow`, `broadphase overflow` or `nefc overflow` from the
device to file descriptor 1. Python's `sys.stdout` never sees that text, so a
run log shows it only when fd 1 is captured. `fd_capture.py` captures it, and
`check-terrain` counts it. A bare `mjx.forward` checks neither budget. The
counters keep counting past their buffers. Because no overflow raises, the
budgets are fail-closed: a warp env whose robot records no `sim_budget` (and
whose sim config sets none) refuses to construct.

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

### Terrain budgets (`check-terrain`)

A robot's `sim_budget` is a flat-floor measurement. A terrain scene adds
heightfield and box contacts. A terrain run on warp therefore refuses to
start unless the recipe sets `task.env.sim.naconmax_per_env` and
`task.env.sim.njmax`. `./run.sh check-terrain` measures those budgets and
gates a recipe against them. No yaml file declares a `sim` block, so Hydra
refuses the plain `task.env.sim.njmax=N` override. Set a budget with
`++task.env.sim.<key>=N`.

The verb composes the recipe with `task=terrain` first, so `experiment=` and
later overrides still apply. A composed task other than terrain exits 2. The
env is the training env, built by `registry.env_args_from_config` and
`make_env`. Push, the no-progress cut, command resampling and spawn grace
are off. The gate budgets are the recipe's `task.env.sim` values. Where the
recipe sets none, `--naconmax-per-env` (512), `--njmax` (4096) and
`--naccdmax-per-env` (default: the naconmax pool) apply. The report's
`budgets.source` names which. The contact and row counters count past
their buffers. A CCD, broadphase or narrowphase overflow still drops work
before a counter sees it. A contact or CCD default below the recipe's
demand therefore gives a `lower_bound` recommendation and a failed gate.
Rerun at the recommended budgets until `lower_bound` clears. For Roboto
Origin at 512 per env and the default 1024 warp worlds, the gate's own CCD
scratch is 2.9 GB. A robot without box colliders has a 4,996-byte slot,
which gives 2.6 GB. `--arena eval` swaps in the terrain scan's arena.

The worlds spread over the arena's tiles, hardest first. Each base sits on
its tile's feature band, between the spawn pad and the tile edge, along one
of 8 headings. Placement reads the dilated spawn grid, so no collider starts
inside a box. Flat-row tiles use the pad. Three regimes each roll `--steps`
control steps (default 200):

- `stand`: zero action. Neither robot stays up under it.
- `walk`: `check-contacts`' full-amplitude sinusoid.
- `fallen`: `check-contacts`' three attitudes, dropped 0.25 m. World `i`
  takes attitude `(i mod n_tiles + i // n_tiles) mod 3`. Each lap of the
  tiles moves a tile on to its next attitude. A tile sees all three only
  when `num_envs` is at least 3 × the tile count. The proxy's default of
  one world per tile gives each tile one attitude.

One warm-up rollout pays the compile. Each regime then runs once, and that
run is the measurement. At every physics step, `n_substeps` per control
step, it records warp's pool-wide `nacon` and `ncollision`, the largest
per-world `nefc`, and whether every `qpos` is finite. On jax it records the
penetrating contacts per world instead. Each control step keeps its worst
physics step. Training telemetry and `check-contacts` sample once per
control step, at its last physics step, so their peaks can read below the
gate's. Each regime runs under a capture of fd 1 that counts MJWarp's
messages:

| Message | Gates | Meaning |
|---|---|---|
| `height field collision overflow` | yes | A heightfield pair collected 50 prism hits. Its later prisms go untested. |
| `CCD overflow` | yes | A convex pair past the `naccdmax` pool is dropped. |
| `narrowphase overflow`, `broadphase overflow` | yes | Contacts or candidate pairs past the naconmax pool are dropped. |
| `nefc overflow`, `njmax_nnz overflow` | yes | Rows past the budget apply no force. |
| `EPA horizon` | no | EPA's 24-entry horizon ran out for one pair. No budget enlarges it. |

| Engine | Condition | Status | Exit |
|---|---|---|---|
| mjx on warp | a gating message, pool or row fill at `--max-fill` (0.9), or a non-finite `qpos` | `fail` | 1 |
| mjx on warp | otherwise | `pass` | 0 |
| mjx on jax | `--require-warp` | `unverified` | 2 |
| mjx on jax | a non-finite `qpos` | `fail` | 1 |
| mjx on jax | otherwise | `unverified` | 0 |
| mujoco | a C reset (bad qpos, qvel or qacc) | `fail` | 1 |
| mujoco | a collider at the per-pair cap | `proxy_at_risk` | 1 with `--strict`, else 0 |
| mujoco | otherwise | `proxy_clear` | 0 |
| any | a refused request: an unknown flag or regime, an empty `--regimes`, `--steps` below 1, a task other than terrain, a missing eval arena, an arena of more than 128 boxes on jax, or `--require-warp` with `--engine mujoco` | `error` | 2 |
| any | an exception | `error`, with the traceback | 1 |

The recommendations divide pool peaks by the world count. The pool is
shared, so a per-env budget covers the batch's total demand, not the worst
world.

- `naconmax_per_env`: the larger pool peak of `nacon` and `ncollision`, per
  world, times 2.0, rounded up to 8.
- `naccdmax_per_env`: the `ncollision` pool peak per world, times 2.0,
  rounded up to 8, and never above `naconmax_per_env`. `ncollision` counts
  every candidate pair, so it bounds each pair type's CCD slots.
- `njmax`: the per-world `nefc` peak times 2.0, rounded up to 32.
- `floor_4x_colliders`: 4 contacts per ground-pairing collider. MJWarp
  writes at most 4 per heightfield pair. It is reported for comparison.
- A CCD, broadphase or narrowphase overflow drops work before a counter sees
  it. The recommendation is then marked `lower_bound`.

The three headrooms and `--max-fill` are untuned. `ccd_scratch` projects the
CCD scratch a training run of the recipe holds outside the XLA pool. It is
the recommended `naccdmax_per_env` (else the gate's budget) times the
training batch times `sim_budget.ccd_slot_bytes`. The training batch is the
larger of `ppo.num_envs` and `ppo.num_eval_envs`. Roboto Origin's slot is
5,480 bytes.

| Check | Where |
|---|---|
| Composition, env build, spawn table, regimes, fd capture, report schema, exit rules, budget arithmetic | any host, through the tests |
| jax run on the CPU arena (`experiment=terrain_cpu`) | any host. Status `unverified`. |
| C per-pair cap proxy, any arena | any host, `--engine mujoco` |
| The gate: messages, pool and row fill, recommendations, CCD demand, throughput | a CUDA host only, `--backend warp --require-warp`, launched by a human |

jax has no per-pair cap, prints no message and has no live counters. Its
`hfield_convex` misses the prisms under a yawed or rolled box's low-side
corners, so box colliders get partial heightfield contact there. The terrain
env's `max_contact_points` caps the contacts jax keeps. The jax counts are
therefore lower bounds, and the report says so under `jax_lower_bound`. The
jax backend refuses an arena of more than 128 ground boxes, so the default
arena runs on warp or on the C proxy.

The C proxy steps plain MuJoCo over the same spawns, regimes and ctrl, one
world at a time. The default is one world per tile. It counts each robot
collider's contacts with the heightfield at every physics step. C stops at
50 contacts per pair, so 50 is a cap hit. C writes a contact for every prism
hit, and MJWarp writes at most 4 per pair. A clear proxy is early warning,
never a pass. On a 4 cm heightfield a resting 9.3 × 9.0 cm box cell gets 17
to 30 contacts in C, depending on where it sits on the grid. Centred on a
node it gets 30. The whole 18.6 × 27.0 cm base box reaches 50 at any
placement. On the default arena, 80 worlds and 200 steps per regime,
Roboto Origin under `deploy_pd` reaches no cap. Its highest counts are 44,
on a thigh and a shin capsule while fallen. C resets a diverging world to
`qpos0` inside `mj_step`. The proxy stops counting that world at the reset
and lists it under `diverged`.

The report goes to `--out`, by default
`runs/check_terrain/<robot>_<preset>_<arena>_<fingerprint[:12]>_<engine>.json`.
`<engine>` is `mujoco`, `mjx-warp` or `mjx-jax`.

| Key | Meaning |
|---|---|
| `schema`, `status`, `reasons` | Report version 1 and the verdict. |
| `engine`, `backend` | `mjx` on the env's backend, or `mujoco` with a null backend. |
| `provenance` | Git commit and dirty flag, package versions, device. |
| `robot`, `preset`, `actuator_overrides` | What was built. |
| `action_window` | Each joint's reachable target range for a position-servo preset, inside the env's clip. Null for any other actuator model. |
| `arena` | Kind, generator version, fingerprint, params, grid size, cell size, box count. |
| `model` | `ngeom`, ground geoms, robot colliders, rows per contact, the jax contact cap. |
| `num_envs`, `steps`, `seed` | The run's size. |
| `compile_s` | On mjx, the seconds of the warm-up rollout that pays the compile. Null on mujoco. |
| `budgets` | The gate budgets, the pool and each budget's source. |
| `regimes` | On mjx: `nacon_pool_max`, `ncollision_pool_max`, `nefc_max`, `active_max`, `peak_step`, `finite`, `messages`, `steady_s`, `env_steps_per_s`. On mujoco: `at_cap`, `max_pair_count`, `finite`, `diverged`, `steady_s`. On mujoco `finite` means no world had a C reset, and `diverged` lists `[world, control step]` for each reset. |
| `jax_lower_bound` | On jax: whether the robot has box colliders, and the contact cap. |
| `fill` | Pool and row demand over capacity. |
| `messages` | MJWarp's messages, summed over the regimes. |
| `recommend` | Budgets that clear the peaks, their headrooms, `floor_4x_colliders` and `lower_bound`. |
| `ccd_scratch` | The training run's CCD scratch projection. |
| `proxy` | The C cap, cap hits and peak pair counts per collider. |
| `error`, `timestamp` | The refusal or traceback, and when the report was written. |

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

### The settle window

`eval/battery.py`'s `SETTLE_STEPS = 50` is the reset transient every new
metric drops — 1 s at asimov's `ctrl_dt` of 0.02, and a step count rather
than a duration. `rollout` starts recording on the first step after reset,
and the opening steps are the robot falling into its pose against a command
it has not had time to answer. The pre-4.1 metrics (`vel_err_*`,
`vibration`, `foot_slip`, `height_*`, `torque_sat_frac`, `mech_power_mean`,
`antiphase_score`) still score the whole record: narrowing their window would
change what an existing field means.

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

## Terrain scan (`terrain_scan.json`)

`./run.sh terrain-scan --run runs/<name>` scores a checkpoint on its robot's
terrain scan suite. `eval/terrain_suite.py` defines the suite. Roboto Origin
has one. A run of any other robot is refused before anything is built. A run
of a task other than joystick or terrain is refused.

The suite's arena has six rows, at difficulties 0.2 to 1.2, and one tile of
every terrain type per row. Each tile is a cell. A cell is named by its
realized dimension, for example `pyramid_stairs_9.8cm`. `--list-cells`
prints the 48 cells with their rows, values and bars, and builds nothing.

A run is one forward crossing from the tile's pad:

- The robot starts on the pad in the reset pose, offset along its heading
  and facing it. The base sits at the pad height plus the reset height.
- It stands at zero command for the settle, `protocol.settle_steps` = 50
  control steps, 1 s.
- It then walks at the constant command `[v, 0, 0]`. The policy first sees
  that command one control step after the settle ends.
- It passes when its base reaches Chebyshev `r_out` from the tile centre
  after the settle, without a fall, before its deadline.

`r_out` is 1.75 m on stair tiles and 1.85 m on every other tile. Each cell
gets 64 runs per speed: 8 headings, 4 offsets and 2 draws. Two draws of one
start differ only in the observation noise. The policy acts
deterministically. The speeds are 0.3 and 0.6 m/s. A run's deadline is the
settle plus 1.6 times its distance over the commanded speed.

The env is the run rebuilt on the terrain task with the suite's arena, so a
joystick run scans too. Pushes, command resampling, the no-progress cut and
the command bias are off. Base contact is on at 1 cm. A fall is the env's
`done`: base height, tilt or base contact. The scan refuses an arena whose
params or fingerprint differ from the suite's. Scores compare only within
one suite version.

Every selected cell rolls in one batch per speed. The full suite is 3072
worlds. The jax backend refuses an arena of more than 128 ground boxes. The
suite's arena has 1139: its 1135 boxes and the 4 aprons. The full suite
therefore runs on warp on a CUDA host. A human launches it. The tests scan a
tiny suite on the CPU arena on jax. A world that has stopped is parked every
step: rewritten in the reset pose with its soles 2 m above the arena's
highest point, at rest. It makes no contact, so a fallen robot cannot fill
the contact pool.

`--cells` and `--speeds` scan a subset, and the warnings then say the scan
is partial. `--eval-seed N` folds N into every cell's key. It redraws the
observation noise on the same course. Seed 0 is the default stream. Run r of
a cell gets the same key in any batch.

Each cell entry carries its type, row, difficulty, value, unit, `r_out`,
`bar`, `threshold` (runs of 64) and `provenance`, and one result per
scanned speed:

| Field | Meaning |
|---|---|
| `passed`, `of`, `rate`, `ci95` | Runs that finished without a fall, out of 64, their rate and its Wilson 95% interval. |
| `falls`, `falls_in_settle` | Runs that fell, and those of them that fell during the settle. |
| `timeouts` | Runs that neither finished nor fell before their deadline. |
| `progress_mean` | The largest `(d - d0) / (r_out - d0)` a run reached after the settle, clipped to [0, 1]. `d` is the base's Chebyshev distance from the tile centre and `d0` the start's. |
| `track_err` | Mean `\|v - body vx\|`, m/s. |
| `saturation` | The fraction of actuator samples whose force exceeds 0.95 of that actuator's cap, as the battery counts it. |
| `clearance` | Mean terrain-relative foot clearance, m. |
| `measured` | Runs with a step after the settle. The four metrics above average over them, and read 0 when there are none. |
| `steps_max` | The longest run's control steps. |

The absolute gate reads every gated cell at every scanned speed:

| Verdict | When |
|---|---|
| `invalid` | The scan is not physics-clean. |
| `fail` | A gated cell passed fewer runs than its threshold. |
| `incomplete` | Fewer (cell, speed) pairs were checked than the full suite gates. |
| `pass` | Otherwise. |

Twelve cells carry bars: rough ground, both slopes and waves on the three
easiest rows. Their thresholds are 61, 52 and 39 of 64. Every bar is
provisional. Stairs, obstacles and rubble are tracked and never gated. A
finished scan exits 0 whatever its verdict. A refused request exits 2.

On warp each dispatch runs under a capture of fd 1. The scan is
physics-clean when no gating MJWarp message printed, no pool or row fill
reached 1, and no running world's `qpos` went non-finite. The counters are
sampled at each control step's last physics step. A scan that is not
physics-clean keeps its numbers. jax prints no message and has no live
counters, so there physics-clean covers non-finite states only.

The suite's warp budgets are 256 contacts and 2048 rows per world, and the
naconmax pool for CCD. All three are untuned. `--naconmax-per-env`,
`--njmax` and `--naccdmax-per-env` override them. `check-terrain --arena
eval` measures what the eval arena needs. At 3072 worlds and 256 slots per
world, Roboto Origin's CCD scratch is 4.3 GB outside the XLA pool.

| Key | Meaning |
|---|---|
| `schema` | Report version 1. |
| `suite` | Robot, suite version, arena fingerprint and generator version. |
| `run`, `checkpoint`, `trained_task` | What was scanned. |
| `robot`, `preset`, `actuator_overrides`, `action_window` | What was built. `action_window` is `check-terrain`'s. |
| `engine` | The backend and the jax, mujoco, mujoco_mjx and warp_lang versions. |
| `protocol` | The suite's course and protocol constants, and the control step. |
| `eval_seed` | The observation noise draw. |
| `cells` | Per cell and speed, above. |
| `contacts` | Per speed, the pool and row peaks against the budgets, their fills and the overflow flags. The peaks are null on jax. |
| `nonfinite_runs` | Per speed, the running worlds whose `qpos` went non-finite. |
| `messages` | MJWarp's messages, summed over the speeds. Null on jax. |
| `physics_clean`, `gate`, `warnings` | The verdicts and what the scan flags. |
| `perf` | Wall seconds with the compile, env steps, env steps per second, loop iterations per speed and the world count. |
| `ccd_scratch` | Bytes per slot, slots and bytes of the scan's CCD scratch. |
| `provenance` | Git commit and dirty flag, package versions, device, and when the scan started. `wandb_run_id` is null. |
| `timestamp` | When the scan finished. |

## Eval videos

`./run.sh eval --run runs/<name>` renders one battery scenario to MP4 —
the same scripted command trajectories `battery.json` measures, so a clip
and its battery row describe the same trajectory.

| Flag | Default | Effect |
|---|---|---|
| `--scenario NAME` | `walk_ramp` | Any `eval/battery.py::battery_scenarios` name. |
| `--steps N` | the scenario's own length | Truncates the rollout. |
| `--video-size WxH` | `640x480` | Rendered frame size. When the model's offscreen buffer is smaller, `eval/render.py`'s SceneView widens a private model copy, so any size works without touching the robot XML. The stacked `--plot-*` panels follow the frame width. |
| `--overlay-torque` | off | A signed bar per actuator drawn into the frame itself (`eval/overlays.py`), normalized by that joint's own cap, one colour per joint group, red lines at ±1. The instantaneous view of the same signal `--plot-torque` traces over the episode: saturation is visible at the moment it happens. |
| `--plot-torque` | off | A normalized-torque strip under the render: every joint's torque over its own actuator cap, one colour per joint group, dashed lines at ±1. Normalized because this robot's per-joint force ranges are heterogeneous — a single N·m cap line across a hip and an ankle means nothing. |
| `--plot-joints` | off | A per-joint target-vs-state grid: one row per joint group, one column per side, achieved position solid and the policy's target dashed. |
| `--joint NAME` | — | Swaps the grid for a single-joint zoom panel. Implies `--plot-joints`. |
| `--push` | off | Restores the run's own random pushes. |

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

train.py writes `run.json` before training starts. Its `status` is then
`running`. `early_stopped`, `stopped_at_steps` and `final_reward` are null at
that point. The `progress` block is rewritten after every eval. At the end
the whole record is rewritten. `status` becomes `finished`, `early_stopped`
or `failed`. On a finished or early-stopped run, `early_stopped` is a bool.
`stopped_at_steps` is the last eval's step count. On a completed run that is
the final eval's step count. On a failed run, `early_stopped` and
`final_reward` stay null. Its `error` field holds the exception. A run that
dies without a Python exception, as on SIGKILL, keeps `status: running`.

Patience counts evals, not steps, so `ppo.num_evals` sets how much training
each unit of patience buys. At the default 100M-step budget with brax's
`num_evals`, one eval is several million steps.

**Calibrate `min_delta` above the eval noise before trusting it.** A
`min_delta` inside the noise band lets noise reset the patience clock and
the run never stops. The defaults are uncalibrated starting numbers:
measure the eval noise first by evaluating one checkpoint repeatedly, and
raise `patience` for overnight runs.

## `run.sh` verbs

Read from `run.sh` as it stands today:

| Verb | Runs | Notes |
|---|---|---|
| `train` | `python -m humanoid_lab.train` | Full Hydra CLI available after it. |
| `smoke` | `JAX_PLATFORMS=cpu python -m humanoid_lab.train smoke=true wandb.enable=false` | CPU pipeline check. |
| `build` | `python -m humanoid_lab.build_model` | `--robot NAME --preset NAME [--out PATH] [--set PATH=VALUE ...]`. Writes `robots/<robot>/mjx/<preset>.xml`. `--set` requires `--out`, so an ad-hoc override build never overwrites the canonical preset build. |
| `check` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_model` | `--robot NAME --preset NAME [--steps N] [--xml PATH] [--skip-mjx] [--max-qvel N] [--set PATH=VALUE ...]`. Gate-checks every keyframe for NaN and for `|qvel|` blowup. `--set` forces an in-memory build even if a prebuilt XML exists, and is mutually exclusive with `--xml`. |
| `check-contacts` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_contacts` | `--robot NAME --preset NAME [--steps N] [--seeds N] [--seed N] [--out PATH]`. Measures the per-world contact and constraint-row peaks over three regimes and prints the budgets they need. See [Warp contact budgets](#warp-contact-budgets-taskenvsim). |
| `check-friction` | `python -m humanoid_lab.check_friction` | `--robot NAME --preset NAME [--task joystick\|terrain] [--backend auto\|warp\|jax] [--num-envs N] [--range LO HI]`. Verifies end to end, on the box's own backend, that a `dr.foot_friction` draw is the friction inside each foot-floor contact. `--task terrain` replaces the floor plane with the CPU terrain arena and stands each world on a flat-row pad. A foot contact with any ground geom counts. Exits nonzero on any mismatch. See [Domain randomization](#domain-randomization-dr). |
| `check-terrain` | `python -m humanoid_lab.check_terrain` | `[--engine mjx\|mujoco] [--backend auto\|warp\|jax] [--arena train\|eval] [--num-envs N] [--steps N] [--regimes stand,walk,fallen] [--naconmax-per-env N] [--naccdmax-per-env N] [--njmax N] [--max-fill F] [--require-warp] [--strict] [--seed N] [--out PATH] [hydra overrides...]`. Gates a terrain recipe against MJWarp's contact, CCD and row buffers and recommends its `task.env.sim` budgets. Not forced onto CPU. The gate itself is `--backend warp --require-warp` on a CUDA host. A host without CUDA runs only the jax check (status `unverified`) and the C proxy (`--engine mujoco`). Neither verifies a recipe. `--num-envs` defaults to 1024 on warp, 16 on jax and one world per tile on the proxy. See [Terrain budgets](#terrain-budgets-check-terrain). |
| `terrain-scan` | `python -m humanoid_lab.eval.terrain_scan` | `--run runs/<name> [--cells a,b] [--speeds 0.3,0.6] [--backend auto\|warp\|jax] [--naconmax-per-env N] [--naccdmax-per-env N] [--njmax N] [--eval-seed N] [--out PATH] [--list-cells]`. Scores the checkpoint on its robot's terrain scan suite. Writes `<run>/terrain_scan.json` unless `--out` says otherwise. Not forced onto CPU. The full suite runs on warp on a CUDA host. `--list-cells` prints the cells and builds nothing. A refused request exits 2. See [Terrain scan](#terrain-scan-terrain_scanjson). |
| `test` | `python -m pytest tests/unit -q` | The fast suite: model-free, runs in seconds. `tests/unit/test_suite_split.py` fails if a test here builds or steps a model. |
| `test-slow` | `python -m pytest tests/integration -q` | The slow suite: builds models, steps MJX. Exports `JAX_COMPILATION_CACHE_DIR` (default `.jax_cache`) so re-runs skip XLA compilation. |
| `test-all` | `python -m pytest tests/unit tests/integration -q` | Both suites. Same compile cache as `test-slow`. Use before merging. |
| `sizing-collect` | `JAX_PLATFORMS=cpu python -m humanoid_lab.sizing.collect` | `--run runs/<name> [--episodes N] [--steps N] [--seed N]`. Rolls the checkpoint out on CPU and writes `<run>/sizing_data.npz`. |
| `sizing-report` | `sizing.collect` then `python -m humanoid_lab.sizing.report` | `--run runs/<name> [--episodes N] [--steps N] [--seed N] [--motors NAME] [--recollect]`. Skips the collect step if `<run>/sizing_data.npz` already exists, unless `--recollect` is passed. Writes `<run>/sizing_report.md` and `<run>/sizing_scatter.png`. |
| `battery` | `JAX_PLATFORMS=cpu python -m humanoid_lab.eval.battery` | `--run runs/<name> [--out PATH]`. Writes `<run>/battery.json` unless `--out` says otherwise. |
| `report` | `python -m humanoid_lab.eval.report`, then `sizing.report` if `<run>/sizing_data.npz` exists | `--run runs/<name> [--out PATH]`. Renders `<run>/eval_report.md` from `battery.json`. |
| `eval` | `JAX_PLATFORMS=cpu python -m humanoid_lab.eval.video` | `--run runs/<name> [--scenario NAME] [--steps N] [--out PATH] [--seed N] [--video-size WxH] [--overlay-torque] [--plot-torque] [--plot-joints] [--joint NAME] [--push]`. Renders one battery scenario to MP4. See [Eval videos](#eval-videos). |
| `export` | `JAX_PLATFORMS=cpu python -m humanoid_lab.export.policy` | `--run runs/<name> [--out DIR]`. Writes `policy.npz` and `policy_meta.json` into `<run>/deploy` unless `--out` says otherwise. Both round-trip validations run before either file is placed. See [deploy.md](deploy.md). |

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
