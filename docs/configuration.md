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
| `restore` | `null` | Checkpoint directory to warm-start from. Relative paths resolve against the repo root. |
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
`action_scale_factor`, or a flat `action_scale_rad` (radians per unit
action, every joint the same) that replaces the factor formula. Non-pd
models ignore both scale fields. roboto_origin's `deploy_pd` pins
upstream's 0.25 rad. See `src/humanoid_lab/robot/presets.py` for the full
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

### The settle window

`eval/battery.py`'s `SETTLE_SEC = 1.0` is the reset transient every new
metric drops. It is a duration: `settle_steps(dt)` converts it at the run's
`ctrl_dt`, 50 steps at 0.02. `rollout` starts recording on the first step
after reset, and the opening steps are the robot falling into its pose
against a command it has not had time to answer. The older metrics
(`vel_err_*`, `vibration`, `foot_slip`, `height_*`, `torque_sat_frac`,
`mech_power_mean`, `antiphase_score`) still score the whole record:
narrowing their window would change what an existing field means.

A scenario that ends inside the window therefore reports `null` (or
`swings: 0`) for every new metric while the older ones still print numbers.
That is the expected reading for an early checkpoint that falls in under a
second, not a broken feature — a 100k-step smoke policy falls at about step
50 on `asimov_v1`, right on the boundary.

### Spin probes

Two scenarios, `spin_left` and `spin_right`, hold a pure yaw command with no
translation for 6 s. The rate is `0.8 wz_max` of the run's own yaw box:
1.256 rad/s on Roboto's `±1.57` and 0.48 rad/s on Asimov's `±0.6`. That
headroom means a row that fails cannot be excused as a command-envelope
corner the policy was never trained near.

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
standing on. The post-settle window is 5 s. It asks for 6.28 rad (360 deg)
on Roboto and 2.4 rad (138 deg) on Asimov.

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

## Course benchmark (`courses.json`)

`./run.sh courses --run runs/<name>` asks whether a policy can walk a given
path. The battery drives open-loop commands and reads the gait. Courses close
the loop. A frozen follower turns the robot's pose into the `[vx, 0, wz]`
command the policy tracks. Each row scores how faithfully the base followed.

### Method

Each row is a path plus a commanded speed, or a held spin. There are 20
rows in five families.

| Family | Rows | What varies |
|---|---|---|
| geometry | `straight_10m`, `arc_r3_90deg`, `circle_r2`, `circle_tight`, `figure_eight_r15`, `square_3m`, `slalom_05m`, `u_turn` | the shape only |
| speed | `straight_slow`, `straight_fast`, `circle_r2_fast`, `speed_steps_straight` | the commanded speed |
| floor | `straight_slippery`, `circle_r2_slippery` | floor and foot friction, mu 0.25 |
| disturbance | `straight_push`, `straight_push_fast` | one lateral kick at 5 m |
| spin | `spin_left`, `spin_right`, `spin_slow`, `spin_fast` | spin direction and rate |

The geometry rows and `spin_left` run at the nominal: flat floor, the
model's friction, `v_nom`, no kick. Every other row names a `baseline` and
differs from it in one variable. `tests/unit/test_courses.py` checks that.

A lane is one row at one seed. It resets the env from the seed's key and
settles for 1 s at zero command (`battery.SETTLE_SEC`). The course is laid
out from the settled pose, with its origin at the base and +x along the
base's heading. The follower then runs until the goal, a fall, non-finite
physics or the time budget. The budget is 2.5 times the ideal time plus
2 s.

The follower is pure pursuit with a 0.40 m lookahead. It never commands
`vy`, so a robot cannot crab through a turn. Above 60 deg of heading error it
spins in place. It walks again below 20 deg. Its yaw command is clipped at
the yaw cap. A path completes inside 0.25 m of its end, once the follower's
progress is within 0.80 m of the end. A closed course therefore cannot
complete at its start. The goal is tested on every pose, including the pose
after the last budget step. A spin completes after one full turn of world
yaw in its direction.

### Robot inputs and derived parameters

Six numbers per robot drive the whole catalogue. They live in
`eval/courses/spec.py`'s `ROBOT_INPUTS`. A unit or integration test fails
when the robot's overlay or model moves under one of them.

| Input | `roboto_origin` | `asimov_v1` | Source |
|---|---:|---:|---|
| `vx_max` (m/s) | 1.0 | 0.8 | overlay `task.env.command.vx[1]` |
| `wz_max` (rad/s) | 1.57 | 0.6 | overlay `min(-wz[0], wz[1])` |
| `stance_halfwidth_m` | 0.0725 | 0.1075 | half the lateral distance between the foot sites at the home keyframe |
| `nominal_height_m` | 0.750 | 0.636 | base z at the home keyframe |
| `push_vel` (m/s) | 0.5 | 0.4 | the planar kick the robot trains against |
| `obs_noise` (gyro, joint_pos, joint_vel) | 0.01, 0.03, 1.75 | 0.01, 0.01, 0.1 | overlay `task.env.obs_noise` over `configs/task/joystick.yaml` |

The overlay values are used, not a run's resolved ones. Every run of one
robot therefore shares one catalogue and one sensor model. Asimov's 0.636 m
is the keyframe height, at which the feet just touch the floor. Its overlay
says it stands at about 0.72-0.75 m. Height cannot bind (see Scores), so no
score moves on it.

Speeds and rates are frozen fractions of the command box. They sit below
1.0 with headroom, as the battery's scenarios do.

| Parameter | Rule | `roboto_origin` | `asimov_v1` |
|---|---|---:|---:|
| `v_nom` | 0.5 `vx_max` | 0.50 | 0.40 |
| `v_slow` | 0.2 `vx_max` | 0.20 | 0.16 |
| `v_fast` | 0.9 `vx_max` | 0.90 | 0.72 |
| speed steps, one per 2.5 m | (0.5, 0.9, 0.5, 0.7) `vx_max` | 0.50/0.90/0.50/0.70 | 0.40/0.72/0.40/0.56 |
| `yaw_cap` | 0.8 `wz_max` | 1.256 | 0.48 |
| spin rates (nominal, slow, fast) | `max(f yaw_cap, 0.333)`, f = 0.8, 0.4, 1.2 | 1.005, 0.502, 1.507 | 0.384, 0.333, 0.576 |
| `r_tight` | `max(1.5 v_nom / yaw_cap, 0.75 m)` | 0.75 | 1.25 |
| slalom wavelength | peak yaw demand at `v_nom` of 0.5 `yaw_cap`, amplitude 0.5 m | 3.964 m | 5.736 m |

The 0.333 rad/s spin floor keeps every spin command at least twice the stand
threshold. Commanded speed is `norm(vx, vy) + 0.3 |wz|`, and below 0.05 the
gait clock freezes (`envs/progress.py`). The floor binds on Asimov's
`spin_slow`. One radius, `r_tight`, serves every tight turn: `circle_tight`,
the u-turn's half circle and the square's rounded corners. The 0.75 m floor
binds on Roboto.

No path row asks for the full yaw cap. Pure pursuit cuts curves to the
inside, and a robot whose yaw lags is pushed back out. Where the follower's
own error is as large as a policy's, a lagging robot outscores a perfect
one. This was computed with the follower driving a simulated unicycle on
Roboto's rows. With the u-turn and the slalom at the full cap and sharp
square corners, a unicycle that executes the follower's commands exactly
scored 3.09, 2.95 and 2.45 on tracking. A 0.30 s first-order yaw lag lifted
its u-turn to 5.44. With `r_tight` and the half-cap slalom, that perfect
unicycle scores at least 5.8 on every Roboto path row and 16.6 on every
Asimov one. `tests/unit/test_courses.py` holds every path row at 4.0 or
more on both robots. The cap is still exercised: `spin_fast` asks for 0.96
`wz_max`.

### The catalogue

Roboto Origin. The `perfect unicycle` column is what a unicycle that
executes the follower's `(vx, wz)` exactly earns: cross-track RMS, the
tracking score that gives, and time over ideal. It goes into the JSON as
`perfect_unicycle`. It is not a ceiling, because a robot whose yaw lags can
exceed it.

| Row | Baseline | Path | Speed | Length m | Ideal s | Budget steps | Perfect unicycle: xte cm / tracking / t:ideal |
|---|---|---|---|---:|---:|---:|---|
| `straight_10m` | | `line(10)` | 0.50 | 10.000 | 20.00 | 2600 | 0 / 1000 / 0.97 |
| `arc_r3_90deg` | | 1 m lead-in, a 90 deg arc of radius 3 m | 0.50 | 5.712 | 11.42 | 1528 | 0.22 / 33.7 / 0.96 |
| `circle_r2` | | lead-in, a full circle of radius 2 m | 0.50 | 13.565 | 27.13 | 3491 | 0.19 / 38.1 / 0.99 |
| `circle_tight` | | lead-in, a full circle of radius `r_tight` | 0.50 | 5.712 | 11.42 | 1528 | 0.92 / 7.9 / 0.99 |
| `figure_eight_r15` | | lead-in, two full circles of radius 1.5 m, opposite ways | 0.50 | 19.846 | 39.69 | 5062 | 0.39 / 18.4 / 1.00 |
| `square_3m` | | lead-in, a 3 m square with corners rounded at `r_tight` | 0.50 | 12.461 | 24.92 | 3215 | 1.26 / 5.8 / 0.99 |
| `slalom_05m` | | lead-in, three sine wavelengths of amplitude 0.5 m | 0.50 | 14.582 | 29.16 | 3746 | 1.22 / 5.9 / 0.99 |
| `u_turn` | | lead-in, 3 m, a half circle of radius `r_tight`, 3 m | 0.50 | 9.356 | 18.71 | 2439 | 1.00 / 7.2 / 0.99 |
| `straight_slow` | `straight_10m` | `line(10)` | 0.20 | 10.000 | 50.00 | 6350 | 0 / 1000 / 0.97 |
| `straight_fast` | `straight_10m` | `line(10)` | 0.90 | 10.000 | 11.11 | 1489 | 0 / 1000 / 0.98 |
| `circle_r2_fast` | `circle_r2` | as `circle_r2` | 0.90 | 13.565 | 15.07 | 1984 | 0.24 / 30.4 / 0.99 |
| `speed_steps_straight` | `straight_10m` | `line(10)` in four 2.5 m blocks | 0.50/0.90/0.50/0.70 | 10.000 | 16.35 | 2144 | 0 / 1000 / 0.98 |
| `straight_slippery` | `straight_10m` | as `straight_10m`, mu 0.25 | 0.50 | 10.000 | 20.00 | 2600 | as `straight_10m` |
| `circle_r2_slippery` | `circle_r2` | as `circle_r2`, mu 0.25 | 0.50 | 13.565 | 27.13 | 3491 | as `circle_r2` |
| `straight_push` | `straight_10m` | as `straight_10m`, kick at 5 m | 0.50 | 10.000 | 20.00 | 2600 | as `straight_10m` |
| `straight_push_fast` | `straight_fast` | as `straight_10m`, kick at 5 m | 0.90 | 10.000 | 11.11 | 1489 | as `straight_fast` |
| `spin_left` | | one turn | +1.005 rad/s | | 6.25 | 882 | |
| `spin_right` | `spin_left` | one turn | -1.005 rad/s | | 6.25 | 882 | |
| `spin_slow` | `spin_left` | one turn | +0.502 rad/s | | 12.51 | 1663 | |
| `spin_fast` | `spin_left` | one turn | +1.507 rad/s | | 4.17 | 621 | |

Asimov v1 has the same rows, names and baselines. The values that differ:

| Row | Speed | Path change | Length m | Ideal s | Budget steps | Perfect unicycle |
|---|---|---|---:|---:|---:|---|
| `straight_10m`, `straight_slippery`, `straight_push` | 0.40 | | 10.000 | 25.00 | 3225 | 0 / 1000 / 0.97 |
| `arc_r3_90deg` | 0.40 | | 5.712 | 14.28 | 1885 | 0.21 / 51.2 / 0.96 |
| `circle_r2`, `circle_r2_slippery` | 0.40 | | 13.565 | 33.91 | 4339 | 0.18 / 59.8 / 0.99 |
| `circle_tight` | 0.40 | radius 1.25 m | 8.853 | 22.13 | 2867 | 0.32 / 33.7 / 0.98 |
| `figure_eight_r15` | 0.40 | | 19.846 | 49.62 | 6302 | 0.39 / 27.5 / 1.00 |
| `square_3m` | 0.40 | corners at 1.25 m | 12.102 | 30.26 | 3882 | 0.65 / 16.6 / 0.98 |
| `slalom_05m` | 0.40 | wavelength 5.736 m | 19.432 | 48.58 | 6173 | 0.60 / 17.8 / 0.99 |
| `u_turn` | 0.40 | radius 1.25 m | 10.926 | 27.32 | 3514 | 0.41 / 26.1 / 0.98 |
| `straight_slow` | 0.16 | | 10.000 | 62.50 | 7912 | 0 / 1000 / 0.97 |
| `straight_fast`, `straight_push_fast` | 0.72 | | 10.000 | 13.89 | 1836 | 0 / 1000 / 0.98 |
| `circle_r2_fast` | 0.72 | | 13.565 | 18.84 | 2455 | 0.22 / 49.9 / 0.99 |
| `speed_steps_straight` | 0.40/0.72/0.40/0.56 | | 10.000 | 20.44 | 2655 | 0 / 1000 / 0.98 |
| `spin_left`, `spin_right` | ±0.384 rad/s | | | 16.36 | 2145 | |
| `spin_slow` | 0.333 rad/s | | | 18.85 | 2456 | |
| `spin_fast` | 0.576 rad/s | | | 10.91 | 1464 | |

`python -m humanoid_lab.eval.courses --list` prints both catalogues.

The floor rows set the sliding friction of the floor and of every foot geom
to 0.25. MuJoCo combines equal-priority friction by element-wise max, so
both have to change. Every Roboto experiment preset trains with
`dr.foot_friction`, which puts its contacts on friction 0.30-1.60. 0.25 sits
just below that range. Asimov trains without foot friction DR, at 1.0. The
other rows walk on the model's own friction, 0.9 on Roboto and 1.0 on
Asimov, and each row records its effective value as `friction`.

The push rows kick the base once, by `push_vel` to the left of its heading.
The kick lands when the follower's progress first reaches 5.0 m. The policy
acts on the observation from before the kick.

### Scores

Each axis divides a physical reference by a measured error. 1.0 means the
error is as large as the reference, and higher is better.

| Axis | Formula | Rows |
|---|---|---|
| `tracking` | `stance_halfwidth_m / rms(cross-track error)` | path |
| `speed` | `mean(cmd vx) / rms(cmd vx - forward speed)`, over steps with `cmd vx` above 0.05 m/s | path |
| `grip` (diagnostic) | base distance / foot slip distance | path |
| `rotation` | `abs(wz cmd) / rms(wz cmd - gyro z)`, the spin-up included | spin |
| `drift` | `stance_halfwidth_m / max(planar distance from the settled pose)` | spin |
| `height` (diagnostic) | `nominal_height_m / rms(base height - nominal_height_m)` | both |
| `smoothness` | `1 / vibration_index(joint velocities, 5 Hz)` | both |

A seed's score is its weakest axis when it completed the course, else 0.
`binding` names that axis. A sub-score is capped at 1000, and a non-finite
error scores 0. A record shorter than 1 s, or a lane whose physics went
non-finite, has no sub-scores. A row reports the median and the worst over
its seeds.

Height and grip are diagnostics. They stay in the min but cannot bind while
the other axes are healthy. On a record that never fell, height stays above
`h_nom / (h_nom - fall.min_height)`: 2.5 on Roboto and 3.4 on Asimov. On the
trained Roboto policy below, over the 601 completed seeds of four seed bases,
grip read 13-38 and height 63.6 or more. Neither bound.

`raw` holds the measured errors. `speed_ratio` is the delivered forward
speed over the commanded one. The speed axis divides by the command, so a
steady 23% shortfall still scores 4.3. Read the ratio beside it. Gait KPIs
from `eval/gait.py` are reported under `gait` and never scored.

The vibration cutoff is the battery's 5 Hz. A gait faster than 1.67 Hz puts
its third harmonic above the cutoff, where it counts as vibration.

A score compares across policies on one row of one robot. Rows differ in
difficulty and in their perfect-unicycle numbers. There is therefore no
overall score.

`vs_baseline` holds a row's median minus its baseline's. On the floor rows
it also holds `slip_ratio`, the row's median slip distance over its
baseline's. Compare a pair by medians, never seed by seed. Every row uses
the same seeds, so a row and its baseline start from the same reset states.
Contact chaos still decorrelates the two rollouts within a few hundred
steps.

### Measurement noise

Every run of a robot is measured under that robot's pinned `obs_noise`,
whatever noise the run trained with. The runner merges it over the run's own
`task.env`. The battery instead measures each run under its own training
noise. Courses pin it because smoothness binds on some rows and moves with
the noise. On 4 seeds of `straight_10m`, the trained Roboto policy below read
smoothness 3.45-3.75 under `joint_vel` noise 0.2 and 3.29-3.44 under 1.75.
`--set obs_noise.joint_vel=0.2` with another `--out` re-scores under another
noise.

### Flags and output

| Flag | Default | Effect |
|---|---|---|
| `--run DIR` | required unless `--list` | The run. Its newest checkpoint is measured. |
| `--ground CLASS` | `flat` | The catalogue partition and the output name. Only `flat` exists. |
| `--out FILE` | `<run>/courses.json` | The JSON. Artifacts go to `<out dir>/courses/`. |
| `--seeds N` | 8 | Rollouts per row. |
| `--seed-base N` | 0 | The first seed. Seed k of every row uses `PRNGKey(seed_base + k)`. |
| `--only NAME ...` | every row | A subset. An unknown name exits 2 and prints the catalogue. |
| `--set BLOCK.KEY=VALUE` | none | A `task.env` override merged one level deep over the measurement env and the pinned noise, recorded as `env_overrides`. A `sim` or `terrain` block exits 2. A `command.resample_steps` below 2 exits 2, because below 2 the env replaces the held command on every step. |
| `--workers N` | `min(8, CPUs)` | Lane threads. It never changes a number. |
| `--skip-if-current` | off | Exit 0 without loading a model when `--out` is current. |
| `--check` | off | Exit 0 when `--out` is current and 3 when it is not, printing why. Loads no model. |
| `--video`, `--video-size WxH` | off, `640x480` | One MP4 per row from its first seed, rendered after the lanes from the recorded joint positions. A renderer that fails warns and keeps the numbers. |
| `--overlay-torque` | off | Torque bars in the `--video` frames, from the recorded torques. |
| `--paths` | off | One overhead PNG per path row, with every seed's base trail in the course frame, labelled by seed number. |
| `--list [--robot NAME]` | | Prints the derived params and the catalogue, and exits. Loads no model. |

The run's own file holds the full measurement only: every row, 8 seeds from
seed 0, the pinned noise and no `--set`. `--only`, `--set`, `--seeds` other
than 8 and a nonzero `--seed-base` each need another `--out`. The report
reads the run's own file. Any two runs' own files for one robot and
catalogue therefore compare. `canonical` in the JSON says whether the
request was the full measurement.

`--out` is current when it holds the same schema, ground class, catalogue
version and fingerprint, robot, checkpoint name and checkpoint sha256,
seeds, seed base, row set and user `--set`. The sha256 covers every file of
the checkpoint dir. A file measured from inputs read off the model is never
current.

The newest checkpoint is the largest numeric step dir, the battery's pick.
Brax writes `ppo_network_config.json` last, so a step dir without it is an
incomplete save, and the runner refuses it. The runner needs the run's
run.json. train.py writes it when training returns or stops early. A
training that is still running, or was killed before that, has none, and the
runner exits 1. A run.json written after a kill, by whatever ran the
training, makes the run measurable. The newest complete checkpoint is then
measured. No run status gates the measurement.

Exit codes:

- 0: written, or current under `--skip-if-current` or `--check`.
- 2: refused. That covers a bad flag, the canonical guard, an unknown row, a
  refused `--set`, a ground class other than `flat`, a jax backend other
  than CPU, and an incomplete newest checkpoint.
- 3: `--check` found the file missing or not current.
- 1: any other error.

The JSON is written atomically, so a crash leaves the previous file in place.
A NaN or an infinity anywhere in the result is an error that names its
field, and nothing is written.

`courses.json` is schema 1 and holds flat-ground rows only.

| Key | Holds |
|---|---|
| `run`, `run_status` | run.json's `run_name` and `status`. `run_status` is `null` when run.json has no `status`, and train.py writes none. |
| `checkpoint`, `checkpoint_step`, `checkpoint_sha256` | the step dir measured, its step, and the sha256 over its files |
| `robot`, `preset`, `seeds`, `seed_base`, `canonical`, `env_overrides`, `budget_cap` | the request. `budget_cap` is `null` unless a test cut every lane short, and a file with one is never current. |
| `catalogue` | The version, the fingerprint, `params_source`, the six inputs and the derived params. `follower` holds the follower constants and the yaw cap. `protocol` holds the protocol constants, `ctrl_dt`, the nominal friction and the noise the lanes ran under. `derivation` holds the fractions that turn the inputs into the params. |
| `engine` | backend, platform, workers, machine, CPU model, `XLA_FLAGS`, and the jax, jaxlib, mujoco and mjx versions |
| `warnings` | rows that command outside the run's own resolved command box, which run as asked |
| `summary` | lane counts by outcome, and the rows every seed completed |
| `courses` | one entry per row, in catalogue order: the row's definition and `spec_hash`, `perfect_unicycle`, the median and worst, the counts of completions, falls, timeouts and non-finite lanes, the median sub-scores, raw metrics and gait KPIs, `binding`, `vs_baseline` and `per_seed` |
| `contacts`, `messages`, `physics_clean`, `nonfinite_lanes` | `contacts` and `messages` hold warp's counters and are `null` on jax. `physics_clean` is true when no lane went non-finite, and `nonfinite_lanes` counts the lanes that did. |
| `perf` | workers, lanes, env-steps, and the env build, compile, lane, perfect-unicycle and wall seconds |
| `provenance` | the git commit and dirty flag, package versions, device, start time and run.json's seed |

A seed's `outcome` is `nonfinite`, `settle_fell`, `fell`, `completed` or
`timed_out`, decided in that order. Path rows report `progress_m` per seed,
and spin rows `progress_rad`.

A robot with no entry in `ROBOT_INPUTS` still gets courses. The runner then
measures the stance and the height on the model, reads the box, the push and
the noise from the run's resolved config, sets `params_source` to
`measured` and adds a warning.

`./run.sh report` appends a `## Courses` section to `eval_report.md` when
`courses.json` exists. It flags a checkpoint that differs from
`battery.json`'s. It carries no PASS or ATTENTION line, because nothing
about a course score is calibrated yet.

### Seeds, determinism and the noise band

Lanes are bit-identical across reruns on one CPU model. That holds with the
same jax, jaxlib, mujoco and mjx versions, the same `XLA_FLAGS` and the same
catalogue fingerprint. The thread count and the lane order do not matter.
`--only` compiles the same program as the full catalogue, so a row's numbers
do not depend on which other rows ran. `tests/integration/test_course_lanes.py`
pins both. XLA:CPU compiles for the host's instruction set, so two CPU
models may differ. `engine.cpu` records the model. arm64 Linux kernels print
no model name, so there it records the core's implementer, variant, part and
revision. Across hosts the agreement is distributional only.

Re-running the same seeds reproduces the same numbers, so a replicate needs
a disjoint `--seed-base`. A row's noise band is the sample SD of its
`score_median` across seed bases 0, 8, 16 and 24, and likewise for
`score_worst`. A difference between two policies on one row below about
twice the band is noise.

### A trained Roboto policy

Measured on the `yolo_v4` preset's run, checkpoint 000786432000, with
`./run.sh courses` at seed bases 0, 8, 16 and 24, 8 seeds each. The table
holds seed base 0, plus each row's band over the four bases.

| Row | Median | Worst | Binding | Done | Error (median) | Speed ratio | Δ median | Band: median | Band: worst |
|---|---:|---:|---|---|---|---:|---:|---:|---:|
| `straight_10m` | 1.92 | 1.53 | tracking | 8/8 | xte 3.8 cm | 1.06 | | 0.23 | 0.17 |
| `arc_r3_90deg` | 2.27 | 1.39 | tracking | 8/8 | xte 3.2 cm | 1.05 | | 0.40 | 0.50 |
| `circle_r2` | 2.60 | 1.89 | tracking | 8/8 | xte 2.8 cm | 1.14 | | 0.33 | 0.47 |
| `circle_tight` | 1.70 | 1.23 | tracking | 8/8 | xte 4.3 cm | 1.06 | | 0.24 | 0.24 |
| `figure_eight_r15` | 2.45 | 1.91 | tracking | 8/8 | xte 3.0 cm | 1.10 | | 0.05 | 0.07 |
| `square_3m` | 1.66 | 1.38 | tracking | 8/8 | xte 4.4 cm | 1.09 | | 0.18 | 0.19 |
| `slalom_05m` | 2.23 | 1.56 | tracking | 8/8 | xte 3.2 cm | 1.08 | | 0.06 | 0.24 |
| `u_turn` | 1.64 | 1.35 | tracking | 8/8 | xte 4.4 cm | 1.05 | | 0.16 | 0.17 |
| `straight_slow` | 0.00 | 0.00 | tracking | 0/8 | xte 13.2 cm | 0.15 | -1.92 | 0.00 | 0.00 |
| `straight_fast` | 3.16 | 2.88 | smoothness | 8/8 | xte 2.0 cm | 0.77 | +1.24 | 0.04 | 0.42 |
| `circle_r2_fast` | 3.23 | 3.21 | smoothness | 8/8 | xte 1.5 cm | 0.82 | +0.63 | 0.01 | 0.03 |
| `speed_steps_straight` | 2.52 | 1.70 | tracking | 8/8 | xte 2.9 cm | 0.93 | +0.60 | 0.17 | 0.44 |
| `straight_slippery` | 2.10 | 0.91 | tracking | 8/8 | xte 3.5 cm | 1.07 | +0.18 | 0.34 | 0.44 |
| `circle_r2_slippery` | 1.91 | 1.05 | tracking | 8/8 | xte 3.8 cm | 1.14 | -0.70 | 0.09 | 0.28 |
| `straight_push` | 0.64 | 0.26 | tracking | 8/8 | xte 11.4 cm | 1.03 | -1.28 | 0.06 | 0.25 |
| `straight_push_fast` | 0.63 | 0.23 | tracking | 8/8 | xte 11.6 cm | 0.74 | -2.53 | 0.02 | 0.18 |
| `spin_left` | 0.38 | 0.32 | drift | 8/8 | wz 0.277 rad/s | | | 0.03 | 0.03 |
| `spin_right` | 0.35 | 0.26 | drift | 8/8 | wz 0.165 rad/s | | -0.03 | 0.03 | 0.03 |
| `spin_slow` | 0.38 | 0.20 | drift | 8/8 | wz 0.170 rad/s | | +0.00 | 0.07 | 0.04 |
| `spin_fast` | 0.72 | 0.41 | drift | 8/8 | wz 0.456 rad/s | | +0.34 | 0.05 | 0.05 |

What the four bases show:

- Tracking binds on most path rows. Speed and smoothness each bind on a
  few seeds of the arc and circle rows. Smoothness binds on most seeds of the
  fast rows. Drift binds on every spin.
- The policy cannot walk slowly. On `straight_slow` its median delivery is
  9-17% of the 0.20 m/s command, and it times out on all 32 seeds.
- At `v_fast` it delivers 75-78% of the command, yet the speed axis reads
  3.6. `speed_ratio` is the column that shows the shortfall.
- The kick can tip it over. It fell on 3 of 32 seeds of `straight_push` and
  on 4 of 32 of `straight_push_fast`, all at seed bases 8 and 24.
- `straight_slow` timed out on every seed, so its band is 0. That band is a
  floor, not a noise estimate. The other median bands run from 0.01
  (`circle_r2_fast`) to 0.40 (`arc_r3_90deg`). Base 0 reads below the other
  three on six of the eight geometry rows. Every row of one base starts from
  the same reset states.

`circle_r2_slippery` against `circle_r2` clears twice the band. The two
bands are 0.09 and 0.33, so twice the larger is 0.67. The median dropped by
0.70 at base 0, 1.53 at base 8, 1.42 at base 16 and 1.41 at base 24. Base 0
clears it only just. The straight pair shows no effect. `straight_slippery`
minus `straight_10m` read +0.18, -0.16, -0.72 and +0.32, and its sign flips
between bases. The feet slide about twice as far at mu 0.25 on both pairs:
`slip_ratio` read 2.0-2.2 on the straight pair and 2.3-2.4 on the circle
pair.

### Cost

One full measurement is 160 lanes. For the Roboto policy above it took
186-188k env-steps, 51k of them on the 8 timeouts of `straight_slow`. A
Roboto policy that timed out on every row would take 406k env-steps.

The numbers below cover six runs on a 10-core Apple M2 Pro with 8 threads:
the four seed bases and two reruns of base 0.

| Part | Roboto |
|---|---:|
| env build and checkpoint load (s) | 1.8-2.4 |
| compile, one lane program per friction group, two groups (s) | 21-29 |
| lanes (s) | 43-57 |
| perfect-unicycle numbers (s) | 0.1 |
| wall (s) | 67-84 |
| env-steps | 186-188k |
| ms per env-step | 0.23-0.30 |

Asimov's cost is not measured here. `perf` in the JSON records these numbers
for every run.

### Changing the catalogue

The catalogue is frozen. A change to a row, a shared constant, a normalizer
or a follower constant invalidates every recorded score.
`families.catalogue_fingerprint` hashes every row of a ground class with
every frozen constant and derived parameter. `tests/unit/test_courses.py`
pins the hash per robot, so any change fails that test. Whoever changes it
decides whether `CATALOGUE_VERSION` moves. Bump it when an existing row's
meaning changes. A new row changes the fingerprint and not the version. A
row's name plus its `spec_hash` is its identity.

To add a row, append it to its family's `courses(p)` in
`eval/courses/families/`, or add a family module and list it in
`FAMILY_MODULES`. Name its `baseline` when it differs from another row in
one variable. Then update the documented rows and the pinned fingerprints in
`tests/unit/test_courses.py`. Old `courses.json` files then fail `--check`,
and a re-run recomputes them.

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

## `run.sh` verbs

Read from `run.sh` as it stands today:

| Verb | Runs | Notes |
|---|---|---|
| `train` | `python -m humanoid_lab.train` | Full Hydra CLI available after it. |
| `smoke` | `JAX_PLATFORMS=cpu python -m humanoid_lab.train smoke=true wandb.enable=false` | CPU pipeline check. |
| `build` | `python -m humanoid_lab.build_model` | `--robot NAME --preset NAME [--out PATH] [--set PATH=VALUE ...]`. Writes `robots/<robot>/mjx/<preset>.xml`. `--set` requires `--out`, so an ad-hoc override build never overwrites the canonical preset build. |
| `check` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_model` | `--robot NAME --preset NAME [--steps N] [--xml PATH] [--skip-mjx] [--max-qvel N] [--set PATH=VALUE ...]`. Gate-checks every keyframe for NaN and for `|qvel|` blowup. `--set` forces an in-memory build even if a prebuilt XML exists, and is mutually exclusive with `--xml`. |
| `check-contacts` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_contacts` | `--robot NAME --preset NAME [--steps N] [--seeds N] [--seed N] [--out PATH]`. Measures the per-world contact and constraint-row peaks over three regimes and prints the budgets they need. See [Warp contact budgets](#warp-contact-budgets-taskenvsim). |
| `check-friction` | `python -m humanoid_lab.check_friction` | `--robot NAME --preset NAME [--backend auto\|warp\|jax] [--num-envs N] [--range LO HI]`. Verifies end to end, on the box's own backend, that a `dr.foot_friction` draw is the friction inside each foot-floor contact. Exits nonzero on any mismatch. See [Domain randomization](#domain-randomization-dr). |
| `test` | `python -m pytest tests/unit -q` | The fast suite: model-free, runs in seconds. `tests/unit/test_suite_split.py` fails if a test here builds or steps a model. |
| `test-slow` | `python -m pytest tests/integration -q` | The slow suite: builds models, steps MJX. Exports `JAX_COMPILATION_CACHE_DIR` (default `.jax_cache`) so re-runs skip XLA compilation. |
| `test-all` | `python -m pytest tests/unit tests/integration -q` | Both suites. Same compile cache as `test-slow`. Use before merging. |
| `sizing-collect` | `JAX_PLATFORMS=cpu python -m humanoid_lab.sizing.collect` | `--run runs/<name> [--episodes N] [--steps N] [--seed N]`. Rolls the checkpoint out on CPU and writes `<run>/sizing_data.npz`. |
| `sizing-report` | `sizing.collect` then `python -m humanoid_lab.sizing.report` | `--run runs/<name> [--episodes N] [--steps N] [--seed N] [--motors NAME] [--recollect]`. Skips the collect step if `<run>/sizing_data.npz` already exists, unless `--recollect` is passed. Writes `<run>/sizing_report.md` and `<run>/sizing_scatter.png`. |
| `battery` | `JAX_PLATFORMS=cpu python -m humanoid_lab.eval.battery` | `--run runs/<name> [--out PATH] [--set BLOCK.KEY=VALUE ...]`. Writes `<run>/battery.json` unless `--out` says otherwise. `--set` re-scores the checkpoint under a changed `task.env` value (e.g. `obs_noise.joint_vel=1.75`), merged one level deep over the measurement env; VALUE is read as YAML. It requires an `--out` other than `<run>/battery.json`, so a re-scored variant never overwrites the run's own table, and the variant records the overrides under `env_overrides`. |
| `report` | `python -m humanoid_lab.eval.report`, then `sizing.report` if `<run>/sizing_data.npz` exists | `--run runs/<name> [--out PATH]`. Renders `<run>/eval_report.md` from `battery.json`, and appends the course section when `<run>/courses.json` exists. A run with `courses.json` and no `battery.json` gets the course section alone. |
| `eval` | `JAX_PLATFORMS=cpu python -m humanoid_lab.eval.video` | `--run runs/<name> [--scenario NAME] [--steps N] [--out PATH] [--seed N] [--video-size WxH] [--overlay-torque] [--plot-torque] [--plot-joints] [--joint NAME] [--push]`. Renders one battery scenario to MP4. See [Eval videos](#eval-videos). |
| `courses` | `JAX_PLATFORMS=cpu python -m humanoid_lab.eval.courses` | `--run runs/<name> [--ground flat] [--out PATH] [--seeds N] [--seed-base N] [--only NAME ...] [--set BLOCK.KEY=VALUE ...] [--workers N] [--skip-if-current \| --check] [--video] [--video-size WxH] [--overlay-torque] [--paths]`, or `--list [--robot NAME]`. Runs the path-following course benchmark on the run's newest checkpoint and writes `<run>/courses.json` unless `--out` says otherwise. See [Course benchmark](#course-benchmark-coursesjson). |
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
