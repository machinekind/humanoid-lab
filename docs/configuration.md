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
| `tracking_product` | `false` | Multiply the two kernels into each other: `k_lin, k_ang = k_lin*k_ang, k_ang*k_lin`. Additive tracking pays the easy half of a command — a robot that ignores a pure-spin command still earns the full `tracking_lin_vel`, since standing still tracks the zero linear command perfectly (w01-tek measured that at about 63% of an ideal spin's payout). With the product, full pay needs the whole command tracked. |
| `tracking_relative` | `false` | Score the fraction of the command tracked instead of the absolute error: the width becomes `tracking_rel_sigma * max(\|cmd\|, floor)²`. The absolute kernel pays only within about `√tracking_sigma` of the target whatever the target's size, so a fast command's reward cliff is out of exploration's reach — w01-tek's policy reached 0.70 m/s under a 0.8 command and 0.00 m/s under a 1.0 one. |
| `tracking_rel_sigma` | `0.25` | Dimensionless width of the relative kernel. A w01-tek quadruped starting point: its terrain presets later widened this to `0.5` because the narrow kernel rounded partial tracking to zero. |
| `tracking_rel_floor_lin` | `0.3` | Floor on the linear relative denominator, m/s. Keeps a near-zero command from sharpening the kernel to a point and dividing by zero. |
| `tracking_rel_floor_ang` | `0.4` | Floor on the angular relative denominator, rad/s. Same role. w01-tek's terrain presets later widened this to `0.7`. |
| `tracking_far_weight` | `0.0` | Mix a wide exponential into both kernels: `(1-w)*kernel + w*exp(-err²/tracking_far_sigma)`. Applies in the absolute and the relative branch alike, and the far kernel stays absolute in both. `exp(-err²/σ)` is gradient-free a few sigma out, so a capability the policy never explored gets no pull toward the command; the wide kernel keeps a usable gradient at range without moving the optimum or leaving `[0, 1]`. **This term alone creates a standing deadlock**: at a yaw rate error of 0.8 rad/s it pays `0.25*exp(-0.64/2.5)`, about 19% of the maximum angular reward, for standing still, and that gradient is weaker than the penalties a pivot attempt incurs. Turn it on only together with `tracking_product` or `tracking_relative`. |
| `tracking_far_sigma` | `2.5` | Width of the far kernel, in (m/s)² and (rad/s)². Ten times `tracking_sigma`. |
| `shaping_tracking_gate` | `false` | Multiply the positive gait-shaping terms by the linear tracking kernel, post-product when `tracking_product` is on. Those terms otherwise pay on a commanded env whether or not it translates, which made stand-and-lift the top income under a command in w01-tek's `terrain_blind_v3`: standing with one leg raised earned about 1.8 reward per step against honest walking's 0.25. Gated set: `feet_air_time` and `feet_apex`. `feet_phase` stays ungated — it is the clock-following gradient and has to survive at zero tracking, because stepping is how tracking starts. Stand-still penalties keep their `~moving` mask and are untouched. |

## Orientation tolerance cone (`task.env.reward`)

The `orientation` penalty is `sum(gravity_xy²)`, which is `sin²` of the
base's tilt from vertical. `orientation_tol_deg` puts a tolerance cone around
upright: the penalty becomes `max(sin²(tilt) - sin²(tol), 0)`, exactly zero
inside the cone and rising continuously from its edge with the legacy
penalty's own slope.

Tilt here is measured against **gravity**, not against the local surface. A
flat-referenced penalty therefore taxes the body pitch that locomotion needs
— leaning into an acceleration, or climbing — while a real nosedive stays far
outside any cone worth setting. w01-tek runs 20 degrees and rejected 10 for
that reason.

| Key | Default | Meaning |
|---|---:|---|
| `orientation_tol_deg` | `0.0` | Half-angle of the cone, degrees. `sin²` of it is precomputed at construction, so `0` leaves the legacy penalty bit-exact and a live env never re-reads the key — change it by config, not by mutating a built env. |

## Swing shaping (`task.env.reward`)

Two terms shape what a swing looks like, both at weight 0 by default.

`feet_apex` pays each completed swing, once, at touchdown, for how close its
**peak** clearance came to `apex_target`. The env tracks that peak in
`info["swing_apex"]`: a running maximum while the foot is airborne, read at
first contact, cleared afterwards. Duration-averaged clearance terms —
`feet_phase` here, `high_step` in w01-tek — tolerate a long 1.5 to 2 cm skim
that collects nearly as much as a crisp arc, so the optimizer skims. Pricing
the peak got w01-tek 3 to 5 cm swings and 30 to 70% better grip. The term is
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

"At the floor" and "at `glide_height`" are clearance readings, not physical
heights: `_foot_clearance` is referenced to the reset keyframe, which floats
the feet a few mm, so both this band and `apex_target` sit about 5 mm
(asimov) or 3 mm (roboto) below the physical height they name. The
measurement and the deferred fix are in
[docs/lessons/foot-clearance.md](lessons/foot-clearance.md).

| Key | Default | Meaning |
|---|---:|---|
| `scales.feet_apex` | `0.0` | Weight of the per-swing apex reward. `0` = off. |
| `scales.feet_landing` | `0.0` | Weight of the soft-landing penalty (negative when on). `0` = off. |
| `apex_target` | `0.05` | Swing peak the apex reward asks for, m. Clipped at: the term prices reaching the target, not exceeding it. **Re-derive for asimov's leg** — this is w01-tek's number for a 0.21 m four-bar leg, and our own `gait.swing_height` asks for 0.08 m. |
| `glide_height` | `0.03` | Height band the landing penalty acts in, m. **Re-derive** with `apex_target`; w01-tek's number, same leg. |

## Settled pose anchor (`task.env.real_pose_ref`)

Off by default. `pose` and `stand_still` both score a deviation from
`_default_pose` — the reset keyframe's **commanded** joint values. Under
gravity and finite gains the robot comes to rest below that command, so the
deviation never reaches zero and both terms charge a floor no policy can
remove. w01-tek measured 0.343 rad of summed sag, about 97% of its entire
standing residual. `roboto_origin`'s `home` keyframe settles 0.065 rad off
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
what the policy reads back; w01-tek keeps the same separation between its ctrl
anchor and its pose reference.

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
  well as its gain and bias parameters. This diverges from w01-tek, whose
  actuators are always position servos. An `ideal_torque` preset injects
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

w01-tek's version settles one rung per commanded stand height and interpolates
the anchor on the command. This repo has no height command, so the table
degenerates to the single settle above. If a height command ever lands here,
that table is the extension.

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
| `grace_sec` | `2.0` | No hazard for this long after a reset or a command resample. **Re-derive for a biped.** w01-tek's number, and the one most likely wrong here: turning a two-legged gait around takes longer than turning a 0.21 m four-bar quadruped's. |
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
`wrap_for_brax_training` itself, unchanged. w01-tek does this in its terrain
respawn wrapper, and reseeds only the EMA; zeroing the counter, so the grace
window comes back too, is a deliberate improvement. Any other wrapper that
restarts an episode in place owns the same reseed.

## Pure command draws (`task.env.command`)

The command sampler draws `(vx, vy, wz)` from one uniform box. That box
almost never produces a clean corner: a backward command arrives with random
lateral and yaw contamination attached, and under `tracking_product` or
`tracking_relative` a contaminated corner pays about nothing however well the
robot serves it. The skill is then never profitable to learn, and the policy
settles on refusing it — w01-tek's `terrain_blind_v2c` held 0.000 m/s under a
commanded -0.4 backward, and five isolating probes confirmed the refusal was
learned rather than mechanical.

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
envelope (`vx ±0.8`, `vy ±0.6`, `wz ±0.6`), not from w01-tek's quadruped.

| Key | Default | Meaning |
|---|---:|---|
| `pure_wz_prob` | `0.0` | Keep the drawn `wz`, zero the linear part: spin-in-place training. |
| `pure_vy_prob` | `0.0` | Keep the drawn `vy`, zero `vx` and `wz`: pure-strafe training. |
| `pure_slow_prob` | `0.0` | Redraw `vx` from `slow_vx`, zero `vy` and `wz`: clean slow straight walking, so the gait learns to scale down instead of having one speed. |
| `slow_vx` | `(0.1, 0.35)` | Range of the slow redraw, m/s. w01-tek's own numbers, which sit inside our `vx` range unchanged. |
| `pure_fast_prob` | `0.0` | Redraw `vx` from `fast_vx`, zero `vy` and `wz`: clean fast straight walking. |
| `fast_vx` | `(0.5, 0.8)` | Range of the fast redraw, m/s. Tops out at `0.8`, the top of our commanded `vx` box. w01-tek deliberately set its own `fast_vx` to `(0.8, 1.2)` — **above** its box — to pull the policy past the speed it deadlocked at. Our envelope is capped pending sysid, so commanding past it is a decision for later, not a default. |
| `pure_back_prob` | `0.0` | Redraw `vx` from `back_vx`, zero `vy` and `wz`: clean backward walking, the refusal w01-tek actually measured. |
| `back_vx` | `(-0.8, -0.2)` | Range of the backward redraw, m/s. Sits inside our negative `vx` range. |

## Mirror augmentation (`task.env.symmetry`)

Off by default. On, each env draws a coin at reset. The ones that come up
heads see every observation mirrored left-to-right, and their action is
un-mirrored before it reaches the physics — so the world they live in is the
ordinary one, and physics, rewards and termination all run in the real frame.

**What it is for.** Cancelling the simulator's own lateral bias: engine
contact ordering, and residual model asymmetry. Both robots carry some.
asimov_v1's vendored CAD export mirrors to 1.0e-4 m and 4.6e-4 kg (its right
ankle-pitch link sits 0.1 mm off in y); roboto_origin's to 5.9e-7 m and
1.5e-3 kg. Averaged over a training batch, half the envs meet that bias from
each side. It costs one bernoulli draw per reset and three gathers per step
(the action, the actor observation, the privileged one).

**What it is not.** It is not a fix for asymmetric behavior. The flag is
drawn once at reset, and under brax's
auto-reset (`BraxAutoResetWrapper(full_reset=False)`, the same wrapper the
no-progress reseed works around) `info` survives every respawn, so in
practice the flag is fixed per env for the whole run. A fixed flag is
unobservable to the learner: for any policy, however asymmetric, a mirrored
env's policy-frame `(obs, action, reward)` stream is *identical* to a plain
env's, so PPO gets exactly zero gradient toward `pi(mirror s) = mirror pi(s)`.
World mirroring cancels the chirality of the **world**; it cannot symmetrize
the **policy**. That is not a guess: w01-tek's `terrain_blind_v3` trained with
this on and provably correct maps — mirror-wrapping the checkpoint swapped
its spin scores exactly, `-33/+124` degrees became `+127/-35` — and still
could not turn one way. Its asymmetric turning was fixed by the reward
mechanics of port items 1.1 to 1.6, and its v4 turns 360 degrees both ways
with no equivariant network. `tests/unit/test_symmetry.py`'s
`test_fixed_flag_world_mirror_is_invisible_to_the_policy` pins the algebra.

**The maps.** `src/humanoid_lab/envs/symmetry.py`, built once at construction
and only when `enable` is set. The joint pairing comes from `robot.yaml`'s
`symmetry` map, which must cover **every** actuated joint: a centerline joint
is written as its own partner (`torso_joint: torso_joint`), and an unlisted
joint raises, because leaving it in place with sign `+1` is a wrong mirror
rather than a missing one.

The joint **signs are derived numerically**, never read off axis names. For
each pair the derivation perturbs the left joint by 0.05 rad at the reset
keyframe, mirrors its foot site's displacement about the robot's xz-plane,
and asks which sign of the right joint reproduces it; the winner must fit
within 1e-4 m and beat the loser by 100x, which makes the probe double as
proof that the robot is mirror-symmetric under this pairing. It measures
displacements rather than positions so a static left-right offset (asimov's
0.1 mm) cannot swamp the comparison. Joints on no foot chain (roboto's arms
and torso) probe the subtree centre of mass below the joint instead — a
child body's own origin is where MuJoCo puts the hinge and does not move at
all when the joint turns.

The probe runs at the reset **keyframe**, and it also checks that the
keyframe is its own mirror — it is the anchor the `joint_pos` observation
subtracts, so a pose that is not mirror-symmetric would break that
observation's map too. With `real_pose_ref` on, the pose the episode starts
from is the settled one instead, which inherits the keyframe's symmetry only
up to the model's own asymmetry worked through the settle: measured on
roboto_origin (the only robot whose keyframe settles), the keyframe is its
own mirror exactly and the settled pose to 1.7e-6 rad. The two switches
compose.

No naming rule would work. asimov_v1 mirrors with `-1` on all twelve leg
joints, because its left and right pitch hinges carry opposite local y-axes.
roboto_origin needs `+1` on the thigh pitch, knee and ankle pitch and `-1` on
the thigh yaw/roll, ankle roll and torso, because its two sides carry
identical axes — including a compound 60-degree thigh yaw/roll pair. A rule
fitted to either robot is wrong on the other. Both tables are pinned in
`tests/integration/test_symmetry_env.py`, which exists to catch silent
drift.

The observation map is assembled from the env's **resolved** actor and
privileged lists and validated against the sizes that env's own catalog
produces, so an obs list change or a robot with a different joint or foot
count fails at construction, naming the component. Vectors mirror as
`(x,-y,z)`, angular rates as `(-x,y,-z)`, the command as `(vx,-vy,-wz)`; the
gait clock's two feet swap in both the cos and the sin half.

Observations are noised in the real frame and mirrored afterwards, the order
w01-tek uses. The noise is i.i.d. within each component with one scale per
component, and the mirror never moves a value across a component boundary, so
mirroring the noisy vector samples the same distribution as noising the
mirrored one.

**The deployment-frame rule.** `eval/battery.py` and `sizing/collect.py`
force `symmetry.enable=false` when they rebuild an env from a run config, and
`eval/video.py` inherits it by rebuilding through the battery. A measurement
describes the frame the robot is deployed in; a battery that drew the coin
the other way would report a policy's `spin_left` as its `spin_right`, and a
sizing rollout would bill the left leg's torques to the right motor. Only
`enable` is overridden — `mirror_prob` and every other recorded value
survive. Any future training-only stochastic augmentation stored in run
config gets the same treatment; the rule lives in one function,
`envs/symmetry.py`'s `deployment_frame_overrides`.

| Key | Default | Meaning |
|---|---:|---|
| `enable` | `false` | Off changes nothing: no maps are built, no `mirror` key enters `info`, and no RNG key is split, so a rollout stays bit-exact (`tests/integration/test_golden_baseline.py`). |
| `mirror_prob` | `0.5` | Probability that an env draws the mirrored frame at reset. `0.5` splits the batch evenly, which is what makes the bias cancel. |

**If asymmetry ever appears anyway**, and survives a healthy reward landscape
(items 1.1 to 1.6 on, tracking honest, no refused command corners), the
escalation is a mirror-equivariant *network*, not a bigger augmentation:

```
pi(s) = (f(s) + mirror_act(f(mirror_obs(s)))) / 2
```

wired through the network factory, using the same two maps this module
already derives. That couples the two frames inside the learner, which is
exactly what the world mirror cannot do. It is documented here and
deliberately **not built** — w01-tek needed no such thing once its rewards
were right.

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
with no kwargs on the jax branch, so changing either one cannot move a jax
rollout by a bit.

| Key | Default | Meaning |
|---|---:|---|
| `sim.backend` | `auto` | `auto` picks warp on a CUDA host and jax elsewhere. `jax` and `warp` pass through. |
| `sim.naconmax_per_env` | `224` | Contact budget per world. Warp allocates ONE pool for the batch, sized `naconmax_per_env * num_envs`. |
| `sim.njmax` | `1120` | Constraint-row budget per world. Never multiplied by the env count. |
| `sim.num_envs` | `1` | Batch size the pool is sized for. `train.py` overwrites it with the larger of `ppo.num_envs` and `ppo.num_eval_envs`. |

Both overflows are silent. Contacts past `naconmax` are dropped; rows past
`njmax` apply no force, and nothing warns anywhere — no counter reports it and
no exception is raised, so a run just trains against a robot whose feet half
pass through the floor.

**Measured 2026-07-31**, `./run.sh check-contacts --robot R --preset P`, 200
control steps × 5 seeds per regime. Per-world peaks, contacts / constraint
rows:

| robot / preset | standing | walking | fallen |
|---|---:|---:|---:|
| `asimov_v1` / `sizing_ideal` | 32 / 157 | 30 / 149 | 22 / 117 |
| `asimov_v1` / `deploy_pd` | 31 / 153 | 29 / 145 | 22 / 117 |
| `roboto_origin` / `sizing_ideal` | 12 / 118 | 9 / 100 | 13 / 124 |
| `roboto_origin` / `deploy_pd` | 12 / 118 | 12 / 118 | 13 / 124 |

Worst case 32 contacts and 157 rows, both on `asimov_v1`, whose 20 foot geoms
put up to two contacts each on the floor plane the moment both feet are flat.
At w01-tek's ~7× headroom that is 224 and 1120. The previous defaults, 32 and
320, carried from w01-tek's quadruped, put `naconmax_per_env` exactly **on**
asimov's standing peak.

The fallen regime does not dominate here, unlike w01-tek's. A neutral action
holds neither robot's home keyframe up under the stock preset gains, so
"standing" is itself a collapse and its opening steps carry the highest count.

Two facts for the debugging that follows a resize. The pool is a real
device-memory line item: 224 at 4096 envs is a 917,504-contact allocation, and
w01-tek ran a 4096-env job out of device memory on a 256 pool. And one MJX step
is not batch-shape invariant on the jax CPU backend, so a batched-versus-
sequential parity check has to compare integer outcomes, never floats.

`tests/integration/test_check_contacts.py` fails if new collision geometry
outgrows the configured budgets.

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

**Calibrate `min_delta` above the eval noise before trusting it.** w01-tek's
eval noise was about ±1.5 reward, so `min_delta=0.5` let noise reset the
patience clock, and they raised it to 1.0. They also raised `patience` to 8
for overnight runs. The defaults here are w01-tek's starting numbers and have
not been calibrated against our eval noise. Measure that noise first by
evaluating one checkpoint repeatedly.

## `run.sh` verbs

Read from `run.sh` as it stands today:

| Verb | Runs | Notes |
|---|---|---|
| `train` | `python -m humanoid_lab.train` | Full Hydra CLI available after it. |
| `smoke` | `JAX_PLATFORMS=cpu python -m humanoid_lab.train smoke=true wandb.enable=false` | CPU pipeline check. |
| `build` | `python -m humanoid_lab.build_model` | `--robot NAME --preset NAME [--out PATH] [--set PATH=VALUE ...]`. Writes `robots/<robot>/mjx/<preset>.xml`. `--set` requires `--out`, so an ad-hoc override build never overwrites the canonical preset build. |
| `check` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_model` | `--robot NAME --preset NAME [--steps N] [--xml PATH] [--skip-mjx] [--max-qvel N] [--set PATH=VALUE ...]`. Gate-checks every keyframe for NaN and for `|qvel|` blowup. `--set` forces an in-memory build even if a prebuilt XML exists, and is mutually exclusive with `--xml`. |
| `check-contacts` | `JAX_PLATFORMS=cpu python -m humanoid_lab.check_contacts` | `--robot NAME --preset NAME [--steps N] [--seeds N] [--seed N] [--out PATH]`. Measures the per-world contact and constraint-row peaks over three regimes and prints the budgets they need. See [Warp contact budgets](#warp-contact-budgets-taskenvsim). |
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
