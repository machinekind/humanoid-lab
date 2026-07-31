# Port details: implementation briefs

Companion to [/PORT.md](../PORT.md). Each section is a self-contained brief
for one implementing agent.

The source repo is `~/git/machinekind/w01-tek`, a clone of [machinekind/w01-tek](https://github.com/machinekind/w01-tek). The repo
was renamed, so both names mean the same project. w01-tek is the four-bar
quadruped this repo's training core was forked from on 2026-07-14. File paths
like `training/wojtek_rl/env.py` are relative to that clone. Commit hashes
resolve there with `git show`. Read the source before implementing. Several of
w01-tek's comments and docstrings are wrong. Each brief's Traps subsection
lists the ones we verified. Trust the code.

House rules for every item:

- Strict TDD. Write the failing test and commit it. Then implement. Unit
  tests are model-free. Integration tests step a real env.
- Every mechanism is off by default. While off, it changes no observation, no
  reward, no termination, and no RNG stream. Gate every new
  `jax.random.split` or `fold_in` behind the feature flag. w01-tek's pattern:
  `env.py` splits `r_term` only inside `if self._config.no_progress.enable`.
- Config keys go through the normal Hydra layering and get documented in
  `docs/configuration.md`.
- Treat w01-tek's tuned numbers as starting points. They were tuned for a
  0.21 m four-bar leg. Each brief flags what to re-derive.

---

## 0.1 Test split

**What.** Split `tests/` into `tests/unit/` and `tests/integration/`. Unit
tests are model-free and run in seconds. Integration tests build models and
step MJX. Add a guard test that enforces the split mechanically. Add `test`,
`test-slow`, and `test-all` verbs to the runner.

**Why.** TDD needs a fast loop. w01-tek's battery test was silently broken for
three days while the whole suite took ten minutes. The fast split caught the
same failure in seconds.

**Source.** Commit `21dc490`. The guard test is
`training/tests/unit/test_suite_split.py`. Its forbidden-pattern list covers
env instantiation, `make_env`, `mjx.put_model`, `MjModel.from_xml_*`,
`build_model.build`, and `load_checkpoint_policy`. Adapt the names to this
repo. w01-tek gives the slow suite a persistent JAX compile cache via
`JAX_COMPILATION_CACHE_DIR` and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`.

**TDD.** The guard test is the first test. Write it, watch it fail on the
current flat `tests/`, and move files until it passes.

---

## 1.1 tracking_product

**What.** Add `reward.tracking_product`, default false. When true, the two
tracking kernels gate each other:
`k_lin, k_ang = k_lin*k_ang, k_ang*k_lin`. The reassignment is simultaneous,
so each side uses the other's pre-product value.

**Why.** Additive tracking pays full linear reward to a robot that ignores
rotation. A standing robot tracks a zero linear command perfectly, and w01-tek
measured that this is worth about 63% of an ideal spin's payout. The header of
`stiff_h2_scratch.yaml` records the measurement. With the product, ignoring
part of the command collapses the whole payout. w01-tek's H2 arm carried this
mechanism and was the first policy there to complete pivots in both
directions.

**Source.** `training/wojtek_rl/env.py:1184-1187`. Commit `fa3c899`.

**Here.** The kernels live in `src/humanoid_lab/rewards/terms.py` and combine
in `envs/joystick.py::_compute_rewards`. Apply the product in the env after
both terms are computed and before scales.

**TDD tests first.**
- Off: the reward dict is bit-exact against current behavior.
- On: a pure-spin command with perfect zero-linear tracking and bad angular
  tracking scores near zero.
- The reassignment is simultaneous. A sequential assignment fails the test.

---

## 1.2 tracking_relative

**What.** Add `reward.tracking_relative`, default false, with
`tracking_rel_sigma=0.25`, `tracking_rel_floor_lin=0.3`, and
`tracking_rel_floor_ang=0.4`. The kernel becomes
`exp(-err² / (rel_sigma * denom²))` with `denom = max(|cmd|, floor)`.

**Why.** The absolute kernel pays only within about √sigma of the target,
whatever the target size. Fast commands therefore have no reachable gradient.
w01-tek measured the cliff. Its policy reached 0.70 m/s at a 0.8 command and
0.00 m/s at a 1.0 command.

**Source.** `training/wojtek_rl/env.py:340-343` and `1161-1174`. Commit
`fa3c899`. First adopted in `stiff_h3_landing.yaml`. The terrain presets later
widened `rel_sigma` to 0.5 and `floor_ang` to 0.7 because the narrow kernel
rounded partial tracking to zero. Keep the keys tunable and treat the defaults
as starting points.

**TDD tests first.**
- Off: bit-exact.
- On: tracking at 80% of target pays the same kernel value at command 0.4 and
  command 0.8.
- A zero command divides by the floor and never by zero.

---

## 1.3 tracking_far

**What.** Add `tracking_far_weight`, default 0.0, and
`tracking_far_sigma=2.5`. The reward becomes
`(1-w)*exp(-err²/σ) + w*exp(-err²/far_σ)`. Implement one helper and apply it
to both kernels in both the absolute and the relative branch. w01-tek names the
helper `_far_blend(kernel, err_sq)`.

**Why.** The wide kernel gives gradient toward the target where the sharp
kernel is flat. This term kept turning alive in w01-tek's F through H
generations.

**Traps, verified.**
- w01-tek first wired the blend only into the absolute branch. Under
  `tracking_relative` it was silently inert. Commit `042ada4` fixed it. Port
  the regression test `test_tracking_far_is_live_under_relative_kernels` from
  `training/tests/integration/test_env.py:151`.
- The far term alone creates a stable standing deadlock. At wz error 0.8 it
  pays `0.25*exp(-0.64/2.5)`, about 19% of the maximum angular reward, for
  standing still, and its gradient is weaker than the penalties a pivot
  attempt incurs. The math is in `stiff_h_scratch.yaml:5-8`. Use this term
  only together with 1.1 or 1.2, and say so in the config comment.
- The correct history: terrain v3 kept the far blend off by deliberate
  choice, one-sided turning returned, and v4 restored the blend. The wiring
  bug was real but did not cause v3's collapse.

**Source.** `training/wojtek_rl/env.py:1152-1156` and `1179-1180`. Commits
`fa3c899` and `042ada4`.

**TDD tests first.**
- Off at weight 0: bit-exact.
- On: the blend is present in the absolute branch.
- On: the blend is present in the relative branch. This is the ported
  regression test.

---

## 1.4 shaping_tracking_gate

**What.** Add `reward.shaping_tracking_gate`, default false. When true, the
positive gait-shaping rewards are multiplied by the linear tracking kernel.
The gate uses the post-product value when 1.1 is also on. In this repo the
gated set is `feet_air_time`, plus `feet_apex` once 1.7 lands. `feet_phase`
stays ungated, as it does in w01-tek: it is the clock-following gradient, and
it has to survive at zero tracking because stepping is how tracking starts.
w01-tek also gates `high_step`, which is quadruped-only and not ported.
Stand-still penalties keep their command gate and are not touched.

**Why.** w01-tek's v3 forensics found that standing with one leg raised earned
about 1.8 reward per step while honest walking earned about 0.25. The shaping
terms paid on the command alone. With the gate, the stand-and-lift strategy
pays nothing.

**Source.** `training/wojtek_rl/env.py:1191-1195`. Commit `3756a15`. The
motivating numbers are in w01-tek
`docs/plans/terrain-training/2026-07-29-terrain-blind-v4-plan.md:6-13`.

**TDD tests first.**
- Off: bit-exact.
- On with zero tracking: the gated terms contribute zero and `stand_still` is
  unchanged.
- Integration: with 1.1 also on, the gate uses the post-product kernel.

---

## 1.5 no_progress

**What.** A probabilistic termination for envs that ignore their command,
after CaT (arXiv 2403.18765). Config block `no_progress` with `enable=false`,
`grace_sec=2.0`, `ema_sec=1.0`, `risk_below=0.5`, `p_max=0.02`.

Mechanics, mirroring `env.py:963-1024`:
- `served = dot(local_linvel_xy, cmd_xy)/max(|cmd_xy|,1e-6) + 0.3*gyro_z*sign(cmd_wz)`.
  Motion against the command reads negative, which is worse than standing.
- A one-second EMA smooths `served`. `progress_ratio = ema / max(demand, 1e-6)`.
  `demand` is the same commanded-speed blend the env uses elsewhere.
- The per-step hazard ramps linearly from zero at `ratio >= risk_below` to
  `p_max` at zero progress. The cut is a bernoulli draw.
- The cut arms only when `demand > 0.05` and `steps_since_cmd*dt >= grace_sec`.
- The EMA reseeds to ratio 1 on every command resample.
- The cut is a true termination with a zero bootstrap value. It is not a
  truncation. No reward term is attached. Losing the rest of the episode is
  the penalty.
- The extra RNG key is split only when the feature is enabled.

**Why.** This closes the reward-landscape hole where ignoring the command
indefinitely is profitable. Together with 1.4 it took w01-tek's terrain scan
from 0 of 2752 to 872-880 of 2752 against a flat-keeper baseline of 587. The
asimov flat-curve hypothesis in
`configs/experiment/asimov_gentle_penalties.yaml` is the same disease family.

**Source.** Commit `3756a15`. `training/wojtek_rl/env.py:260-279` and
`963-1024`. w01-tek also reseeds the EMA in its terrain respawn wrapper. That
part is deferred with terrain. Leave a comment for whoever builds a curriculum
wrapper later.

**Correction (landed).** The respawn reseed was NOT deferrable. This brief
missed `wojtek_rl/env.py:269-272`, which says a flat run turning the cut on
needs the same reseed in its auto-reset path — and our trainer already had
one: `wrap_for_brax_training` ends in `BraxAutoResetWrapper(full_reset=False)`,
which keeps `info` across a respawn. `envs/wrappers.py` carries the reseed,
applied only when the flag is on.

**TDD tests first.**
- Off: bit-exact, including the RNG stream.
- No cut is possible within `grace_sec` of a reset or a command change.
- A zero command never arms the cut.
- Backward motion under a forward command yields negative `served`.
- The hazard is zero at `ratio >= risk_below` and `p_max` at ratio zero.
- After a resample the ratio starts at 1.
- The cut sets `done` without the truncation flag and adds no reward key.

---

## 1.6 Pure command draws

**What.** Bernoulli-gated redraws after the base uniform command sample. Each
draw produces a clean single-axis command and has its own
`jax.random.fold_in(rng, idx)` index. Keys: `pure_wz_prob`, `pure_vy_prob`,
`pure_slow_prob` with `slow_vx`, `pure_fast_prob` with `fast_vx`, and
`pure_back_prob` with `back_vx`. All probabilities default to 0.0. The order
is wz, vy, slow, fast, back. The existing `zero_prob` overwrite stays last.

**Why.** A uniform box rarely samples a clean corner of the command space.
Under 1.1 and 1.2 a contaminated corner pays nothing, so the skill becomes a
learned refusal. w01-tek's terrain run v2c refused every backward command and
held 0.000 m/s under a commanded -0.4. Five isolating probes confirmed it. A
biped will hit backward refusal too.

**Source.** `training/wojtek_rl/env.py:147-172` and `567-616`. Commit
`fa3c899` added slow and fast. Commit `ef5ce5b` added back. The wz and vy
draws predate the fork. Our `_sample_command` in `envs/joystick.py:209` has
only `zero_prob` today, so all five draws are new here.

**Here.** Derive the ranges from this repo's command envelope in
`configs/task/joystick.yaml` and the robot overlays. `back_vx` must sit inside
the robot's negative vx range.

**TDD tests first.**
- All probabilities zero: bit-exact, RNG stream unchanged.
- Enabling one draw leaves the samples of every other draw unchanged.
- Each pure draw zeroes the other axes. The zero overwrite still applies
  last.
- With probability 1.0 every sample is in range and clean.

---

## 1.7 feet_apex and feet_landing

**What.** Two reward terms, both default weight 0.0.
- `feet_apex` tracks each swing's peak clearance in `info["swing_apex"]`. The
  tracker takes the running maximum while the foot is airborne and resets to
  zero on contact. On first contact the term pays
  `sum(clip(swing_apex/apex_target, 0, 1) * first_contact) * moving`.
  `apex_target` is a config key. w01-tek uses 0.05 m. Re-derive for our leg
  length.
- `feet_landing` is a penalty:
  `sum(clip(-foot_vz, 0, inf)² * clip(1 - foot_clearance/glide_height, 0, 1)) * moving`.
  `glide_height` is a config key. w01-tek uses 0.03 m. The term prices
  downward foot speed weighted by closeness to the ground, so it acts before
  contact.

**Why.** Duration-averaged clearance terms tolerate a 1.5 to 2 cm skim that
scores almost as well as a crisp arc. Paying the swing's peak got w01-tek 3 to
5 cm swings and 30 to 70% better grip. A penalty measured at contact
under-reads impacts, because the solver has already absorbed the hit within
the control step. The physical reference for touchdown softness is free-fall
speed over `glide_height`, which is `sqrt(2*9.81*0.03) ≈ 0.77 m/s`.

**Source.** `training/wojtek_rl/env.py:932-940` and `1247-1269`. Commit
`fa3c899`. Tuning history in `stiff_h3_landing.yaml` and `stiff_h6_apex.yaml`.

**Here.** The terms go in `rewards/terms.py`. The swing-apex state goes in
the env info dict, next to the existing air-time and first-contact tracking.
`feet_apex` must respect the 1.4 gate when both are on.

**TDD tests first.**
- Weight zero: bit-exact.
- The apex tracker rises during a swing, pays once at first contact, and
  resets afterward.
- The landing penalty is zero when the foot is above `glide_height` or moving
  up. It grows quadratically with downward speed near the floor.

---

## 1.8 orientation_tol_deg

**What.** Add `reward.orientation_tol_deg`, default 0. The orientation
penalty becomes `max(sin²(tilt) - sin²(tol), 0)`. Our current term is
`orientation(gravity_xy) = sum(gravity_xy²)`, which equals `sin²(tilt)`.
Precompute `sin²(radians(tol))` at construction. A value of 0 must reproduce
the legacy penalty bit-exact.

**Why.** A flat-referenced tilt penalty taxes body pitch that locomotion
needs, such as leaning into acceleration. The cone leaves real nosedives
penalized. w01-tek uses 20 degrees. w01-tek rejected 10 degrees because tilt is
measured against gravity rather than the local surface.

**Source.** `training/wojtek_rl/env.py:487-495` and `1200-1202`. Commit
`ef5ce5b`.

**TDD tests first.**
- tol=0: bit-exact.
- Inside the cone the penalty is exactly zero. At the cone edge it is
  continuous.
- Outside the cone it equals the legacy penalty minus `sin²(tol)`.

---

## 1.9 real_pose_ref

**What.** Add `env.real_pose_ref`, default false. At construction, settle a
quasi-rigid copy of the model and use the settled pose as the anchor for
`pose`, `stand_still`, and reset-pose sampling. The legacy keyframe anchor
stays the default.

Mechanics, from `env.py:620-680` and `496-516`:
- Copy the model. Set kp=400 and kd=20 through gainprm and biasprm. Set
  forcerange to ±1e6, the timestep to 5e-4, and the integrator to
  `mjINT_IMPLICITFAST`.
- Clamp the settle ctrl to the runtime target bounds, which include our soft
  limits. Do not clamp to raw ctrlrange.
- Step two simulated seconds per rung and record the settled qpos.
- Keep only the strictly increasing height prefix, `cut = argmax + 1`. Raise
  `ValueError` if fewer than three rungs survive.

**Why.** Our `pose` and `stand_still` anchor on `_default_pose`, which is the
keyframe's commanded joint values. See `envs/base.py:123-126` and
`envs/joystick.py:389,394`. Under real PD gains the settled pose sags below
the commanded one. w01-tek measured 0.343 rad of summed sag, about 97% of the
standing residual. That sag is a penalty floor the policy can never remove.
The settled anchor is gain-invariant. That matters here because actuator
presets are a whole config axis.

**Here.** This repo has no height command, so the table degenerates to one
settled pose per keyframe. Implement the single-pose version. Keep the table
machinery only if a height command is planned, and note the extension either
way.

**Traps, verified.** w01-tek's docstrings say kp 2000 and kd 100. The code
uses 400 and 20. An inline comment there notes that kp 400 leaves about
5e-3 rad of residual sag, which is the accepted tolerance.

**Source.** Commit `fa3c899`.

**TDD tests first.**
- Off: bit-exact.
- The settled pose differs from the keyframe pose under gravity with finite
  gains.
- The settled anchor is identical under two different runtime gain sets.
- The settle ctrl is clipped to the runtime target bounds.

---

## 2.1 Early stopping

**What.** Config block `early_stop` with `enable=false`, `min_evals=10`,
`patience=6`, `min_delta=0.5`. A pure function
`plateau_stop(rewards, min_evals, patience, min_delta)` decides the stop. A
reward is a new best only if it beats the running best by more than
`min_delta`. A plateau is `patience` consecutive evals without a new best.
The check runs only once `len(rewards) >= max(min_evals, patience+1)`. Wire
it into the progress callback with an `EarlyStop` exception caught around
`train_fn`. On stop, use the last completed eval's metrics. Record
`early_stopped` and `stopped_at_steps` in run.json.

**Why.** Plateaued runs waste GPU hours. Stopping is safe because Brax
checkpoints at every eval, so the latest checkpoint is the early-stopped
policy. Calibrate `min_delta` above the eval noise. w01-tek's noise was ±1.5
reward, so `min_delta=0.5` reset the patience clock on noise, and they raised
it to 1.0. They raised patience to 8 for overnight runs. Measure our eval
noise before trusting any default.

**Source.** `training/wojtek_rl/train.py:34-48`, `286`, `308-321`.
`training/tests/unit/test_early_stop.py`. Commit `fa3c899`.

**Here.** `src/humanoid_lab/train.py`. The progress callback is at
`train.py:192`.

**TDD tests first.** Port w01-tek's unit tests. Noise below `min_delta` does
not reset patience. Monotonic improvement never stops. A plateau of exactly
`patience` evals stops. The length guard holds.

---

## 2.2 MJWarp preflight

**What.** Two pieces of warp homework that gate the first GPU run.
`envs/backend.py` picks warp on CUDA.

1. A contact-budget audit with self-reporting.
   - Our joystick config carries `naconmax_per_env=32` and `njmax=320` from
     w01-tek's quadruped. The config comment already says to revisit them.
     Asimov's contact set is different. It has capsule feet plus toes, more
     geoms per foot, and injected collision primitives.
   - Measure the peak `nacon` and `nefc` per env on the built model in
     standing, walking, and fallen states. Fallen robots dominate contact
     counts, and early training is mostly fallen robots.
   - Size both budgets from the measurement with stated headroom. w01-tek used
     about 7x the measured peak. Record the measurement in the config
     comment.
   - Make training and eval record their `nacon_max` and `nefc_max` peaks and
     flag overflow in run output. The pattern is `terrain_scan.py:947-959`.
     Know the failure modes. `naconmax` overflow drops contacts silently.
     `njmax` overflow drops constraint rows with no warning anywhere, per
     `terrain_scan.py:528-529`.
   - Warp allocates its contact pool at `make_data` time, sized by env count.
     The budget times the env count is a real device-memory line item. A
     4096-env job ran w01-tek's 256 pool out of device memory.

2. The DR vmap composition test. `geom_priority` and other numpy-typed
   `mjx.Model` fields are static pytree metadata. Patching one after deriving
   `in_axes` changes the treedef, and `jax.vmap` fails with a pytree-prefix
   mismatch. The failure is fatal on the jax backend and absent on warp.
   w01-tek shipped the bug twice on parallel branches, commits `53d7436` and
   `3eb7444`. Our `dr/randomize.py` has the fix and
   `docs/lessons/asimov_v1.md` has the lesson. The missing piece is a test
   that calls `jax.vmap` with the DR wrapper's real contract. Port
   `test_in_axes_is_usable_as_vmap_in_axes`, parametrized over DR configs
   including `foot_friction.enable=true`.

**Also know, for later debugging.** One MJX step is not batch-shape invariant
on the jax CPU backend, so batched-vs-sequential parity checks must compare
integer outcomes. Throughput numbers are only comparable at a matched seed.

**Source.** Commits `53d7436`, `3eb7444`, `d605a59`, `ba73ed8`, `e758669`,
`e094d9c`. `training/wojtek_rl/terrain_scan.py:71-80`, `528-529`, `947-959`.

**TDD tests first.**
- The composition test. It passes on current code and guards the regression.
- A schema test for the self-reported budget fields.
- An integration check that builds the model, rolls standing, walking, and
  fallen states, and asserts the measured peaks fit the configured budgets
  with the stated headroom. This test fails when someone adds collision
  geometry without resizing the budgets.

---

## 3.1 Symmetry

**What.** Mirror augmentation as bias insurance, plus the deployment-frame
measurement rule.

- Build mirror maps for observations and actions from `robot.yaml`'s symmetry
  map. For a biped the map swaps left and right joints and flips the sign of
  roll and yaw joints. Vectors mirror as `(x,-y,z)`. Pseudo-vectors mirror as
  `(-x,y,-z)`. The command mirrors as `(vx,-vy,-wz)`. The gait-clock legs
  swap.
- Config `symmetry.enable=false` and `mirror_prob=0.5`. The flag is drawn
  once per env at reset. Incoming actions are un-mirrored before physics.
  Outgoing observations are mirrored. Physics, rewards, and termination run
  in the real frame.
- The deployment-frame rule: battery, eval, and video reconstruction force
  `symmetry.enable=false`. Apply the same rule to any future training-only
  stochastic augmentation stored in run config.

**Why.** The mechanism is easy to overclaim, so read this carefully. w01-tek
proved that a fixed per-episode world mirror is invisible to the learner. PPO
gets zero gradient toward symmetric behavior from it. The proof is pinned by
the test `test_fixed_flag_world_mirror_is_invisible_to_the_policy`. The
mirror's whole value is that it cancels simulator lateral bias, such as
engine contact ordering and residual model asymmetry, at near-zero cost.
w01-tek's asymmetric turning was fixed by the reward mechanics in items 1.1 to
1.6. Its v4 turns 360 degrees both ways with no equivariant network. If
asymmetry appears here and survives a healthy reward landscape, the
escalation is a mirror-equivariant network,
`π(s) = (f(s) + mirror_act(f(mirror_obs(s))))/2`, wired through the network
factory. Document that option. Do not build it.

**Traps, verified.** w01-tek's docstring claims the env validates the mirror
maps against its real observation sizes. It validates only against a static
table. Do better here. Validate the assembled maps against the env's actual
observation layout at construction and raise a named error on mismatch.

**Source.** `training/wojtek_rl/symmetry.py`, `env.py:807-814`, `877-879`,
`1055-1059`. Commits `fa3c899` and `797dc67`. The post-mortem is
`docs/plans/terrain-training/2026-07-29-symmetry-postmortem-and-fix-plan.md`.

**TDD tests first.**
- Off: bit-exact, including RNG.
- Mirror involution: `mirror(mirror(x)) == x` for both maps.
- Reward invariance: a mirrored env's reward for the mirrored action equals
  the real env's reward for the real action.
- The ported invisibility test.
- Battery and eval reconstruction force symmetry off even when run.json says
  it was on.

---

## 4.1 Spin probes

**What.** Two battery scenarios, `spin_left` and `spin_right`. Each holds a
pure spin command inside the robot's command envelope and reports yaw
progress in degrees, completion, and falls for its direction.

**Why.** w01-tek's stiff_b keeper shipped unable to spin right because nothing
tested that direction. Chirality bugs are invisible without per-direction
rows, and bipeds are where they bite. These probes are also the sensor for
the escalation decision in item 3.1.

**Source.** `training/probe_spin_worlds.py` and `courses/families/spin.py`.
Commit `99d2e7a`. w01-tek's second probe world replays the DR-patched contact
physics. Add a flag for that only if foot-friction DR is enabled here.

**Here.** Extend `eval/battery.py::battery_scenarios` additively. Existing
scenario JSON must not change meaning. If we treat the current battery as
frozen, add the spins through an `eval_scenarios = battery + extras` layer,
following w01-tek commit `eb8bad2`, or version the battery explicitly.

**TDD tests first.** The scenario command traces are pure functions. Test
direction, magnitude, and duration before wiring. Add a JSON schema test for
the per-direction fields.

---

## 4.2 Gait KPIs

**What.** A `gait_metrics(rec)` function over recorded per-foot clearance and
vertical velocity. A swing is a contiguous run of clearance above 5 mm. Drop
swings shorter than two steps and swings truncated by the end of the record.
For each swing record the apex and the touchdown speed, which is the downward
vertical speed on the last airborne step.
`softness = touchdown_v / sqrt(2*g*apex)`, where 1.0 means the foot fell like
a brick. Report `swing_apex_med_m`, `swing_apex_p90_m`, `swings`,
`touchdown_v_med`, and `touchdown_softness_med`. These are raw metrics. They
never fold into existing scores or gates.

**Why.** Tracking error cannot see a skimming gait or a stand-and-lift farm.
These two numbers made both visible in w01-tek. The function is pure numpy and
works for any foot count.

**Source.** `training/wojtek_rl/courses/scoring.py:34-77` and
`tests/unit/test_gait_metrics.py`. Commit `dda80b0`.

**Here.** `eval/battery.py` already records foot clearance and contact.
Extend the record with per-foot vertical velocity if missing. Add the metrics
to `scenario_result` additively.

**TDD tests first.** Port w01-tek's unit tests. Use synthetic clearance and
velocity arrays with known swings. Cover the two-step and truncation filters.
A pure free-fall touchdown has softness near 1.0.

---

## 4.3 tracking_error

**What.** A servo KPI: the RMS and p95 of `|ctrl - qpos|` over actuated
joints, excluding the first 50 steps as reset transient. Also stamp the
effective gains in run.json, read back from the built model after injection,
`actuator_gainprm[0,0]` and `-biasprm[0,2]`. Do not stamp the preset yaml
values.

**Why.** This measures whether the PD loop held the commanded position, which
is what actuator work optimizes. The effective-gain stamp matters because
gains pass through preset loading and `actuators.overrides` merging in
`robot/presets.py`. Record what the model actually got.

**Source.** `training/wojtek_rl/battery.py:159-174`. Commit `e274701`. The
other half of that commit added runtime pd_kp and pd_kd overrides. Our
actuator-preset axis already covers that. Do not duplicate it.

**TDD tests first.** A pure-function test with synthetic ctrl and qpos
histories. The settle window is excluded. A run.json schema test covers the
effective-gain fields.

---

## 4.4 Robustness grid

**What.** An eval-only perturbation sweep. It changes no training code.
- `--alpha` scales the effective kp, kd, and torque cap together in place on
  the built model. This models a firmware Kt miscalibration. Operate on
  post-injection values so it works with any preset.
- `--lag-tau` switches the rollout to an explicit per-substep PD loop.
  `tau_pd = kp*(ctrl-qpos) - kd*qvel`, clipped to the cap, then a first-order
  lag on the applied torque:
  `tau += (1-exp(-dt_sub/lag_tau))*(tau_pd - tau)`. The lag state persists
  across control steps and zeroes at reset.
- `--torque-envelope OMEGA_B,OMEGA_0` caps driving torque by speed. The cap
  is flat to OMEGA_B and ramps linearly to zero at OMEGA_0. Braking torque is
  exempt, tested by `tau*qvel >= 0`. The envelope composes with alpha and
  forces the explicit-PD path.
- `--out` keeps grid cells from overwriting the canonical battery.json.
- A grid report aggregates cells against gates. w01-tek gated on no falls,
  velocity error below 0.20, vibration below 1.3x a reference, and saturation
  below 0.05. Re-derive our gates.

**Why.** The grid quantifies the policy's margin to actuator miscalibration,
bandwidth, and back-EMF droop before hardware. It also feeds motor sizing.
The alpha and envelope axes answer how much actuator margin the policy needs,
which complements `task=sizing`.

**Honesty requirement, verified.** The baseline cell takes the native rollout
path. w01-tek has no shared code path through a tiny lag value, and the claim
that it does is false. The required property is different: the explicit-PD
path reproduces the native pipeline within a stated tolerance as lag goes to
zero. w01-tek measured under 1% on track_err_rms. Test that property and
document the branch honestly.

**Source.** `training/wojtek_rl/battery.py:585-950`, `grid_report.py`,
`hpc/stiff_grid.job`. Commits `d47c471` and `935b697`.

**TDD tests first.** Pure-function tests for the lag coefficient and update.
The envelope is flat, then ramps, and exempts braking. Alpha scales the cap
and the gains together. The explicit-PD path matches native within tolerance
at small lag. `--out` never touches battery.json.

---

## 4.5 Video QoL

**What.** Three eval video improvements.
- A per-joint target-vs-state grid. Solid lines show achieved qpos. Dashed
  lines show the policy's target. Rows share a y-range so left-right
  asymmetry is visible. A `--joint <name>` flag swaps in a single-joint zoom.
  The target signal is the issued motor target from info when action delay
  exists, and `data.ctrl` otherwise.
- A `--plots` flag that takes a comma list of panels, `label,torques,grid`,
  with `none` for a bare render. Unknown names exit with the valid list.
- Push-free video rollouts by default, with `--push` to restore them. The
  battery already disables pushes at `eval/battery.py:266`. Videos should
  match.

**Source.** `training/wojtek_rl/eval.py`. Commits `f757024`, `b3223eb`,
`b751fe9`. Commit `f757024` also made the `MUJOCO_GL=egl` default Linux-only
because macOS has no egl. Check our `run.sh`.

**TDD.** Panel parsing and unknown-name rejection are unit tests. Grid
rendering gets an integration smoke test.

---

## 5.1 Deploy contract

**What.** A fail-closed policy metadata contract built from a live env.
- `build_contract(env, run, checkpoint)` produces `policy_meta.json`. Every
  field is a resolved value: the obs layout, the action scale vector, the
  anchor pose, the command box, the PD gains, the torque caps, and the ctrl
  bounds. The deploy side re-derives nothing.
- Two sets, `CONSUMED_KEYS` and `TRAINING_ONLY_KEYS`, plus
  `check_config_covered()`. Every env config key must appear in one set. An
  unclassified key raises and blocks export. Each entry carries a one-line
  comment saying why the key does or does not reach the robot.
- Constraints travel with the policy. Deploy docs state facts. w01-tek's
  commit `ba572ef` reworded imperatives into facts because a future reader
  inherits an imperative as a prohibition nobody decided.

**Why.** Hand-written deploy blocks drift. w01-tek's PR #30 proved it. The
fail-closed ledger makes an unclassified env option unable to ship. Build
this before the first keeper publishes. Our `policy_io.py` and `export/`
stub predate all of it.

**Source.** `training/wojtek_rl/deploy_contract.py`, 251 lines. Read the
whole file. Commit `70243b2`. Ledger evolution examples: `3c7e1e0`,
`9835775`, `45e6b48`, `df8ab80`.

**Here.** Seed the ledger from `envs/joystick.py::default_config()`. Classify
every key on day one. Keeper layout follows the existing <hf-org> flat
layout convention from PLAN.md.

**TDD tests first.**
- An env config with an unknown key makes the contract raise. This test is
  the feature.
- A completeness test walks every default-config key and fails when someone
  adds an env option without classifying it.
- Contracts built from two presets carry different resolved values in the
  same schema.

---

## 5.2 Exporter

**What.** An exporter with two independent round-trip validations. Both run
before any artifact reaches its destination.
1. The numpy MLP forward pass matches the jitted Brax inference function.
2. The actual deploy runtime, loaded from artifacts written to a temp dir,
   matches a reference pipeline computed from the env's resolved fields. The
   obs assembly order, command fill, anchor, scale, clip bounds, and filter
   state all round-trip.
The tolerance is on the 1e-4 rad scale. w01-tek measured float32 reassociation
up to 3.9e-5.

**Why.** The checks catch every class of exporter-runtime drift before a
broken artifact exists.

**Trap, verified.** w01-tek's exporter writes `policy.npz` and
`policy_meta.json` first and validates after. A failure leaves bad artifacts
on disk, and the docstring claims otherwise. We validate against temp-dir
artifacts and then move them into place.

**Source.** `training/wojtek_rl/export_policy.py`. Commit `70243b2`.

**TDD tests first.** A corrupted weight matrix fails validation and leaves
nothing at the destination. A clean export passes and its written meta
round-trips through the loader.

---

## 6.1 Makefile export

**What.** Audit every `submit --export` list in our Make targets.
- Never join `$(if ...)` branches with a literal `\,`. Make cannot escape a
  comma there. w01-tek's list expanded to a lone backslash, and a job was
  submitted with zero variables. The job was caught in the queue and
  cancelled. Use a `comma := ,` variable with `foreach` and `subst`, as in
  w01-tek's `Makefile:49-53`.
- Check completeness. Every env var a batch script consumes must appear in
  the export list. w01-tek added `FLAT_ROW` to a script and forgot the list.
  Add a scripted audit that compares vars referenced in `hpc/*.job` with
  vars exported in the Makefile.

**Source.** Commits `629f510` and `adb0515`.

---

## 6.2 batch hardening

**What.** Four fixes in our `hpc/` templates.
- `run_main "$@" || handler` suspends `set -e` inside the function, so a
  crash's exit code is discarded. Capture and return `$?` explicitly.
  w01-tek's job reported success with two crashed trainings before the fix.
- Probe cache and log directories for writability and fall back to TMPDIR,
  warming from the store copy where one exists.
- Keep wandb on by default with the offline-dir pattern. Never hardcode
  `wandb.enable=false` in a template.
- Note in the template header that throughput numbers are only comparable at
  a matched seed. w01-tek measured 1.34M against 0.78M steps per second on a
  seed change alone.

**Source.** Commits `a61eb9d`, `76aefe4`, `e094d9c`.
`training/hpc/_common.sh`.

---

## 6.3 hpc.sh

**What.** A repo-root `hpc.sh` that loads `.env` itself with
`set -a; source ./.env` and execs `ssh -o BatchMode=yes "$HPC_USER" "$@"`.
BatchMode fails fast instead of hanging on a password prompt. Agent sessions
cannot read the gitignored `.env`, and without the wrapper they misread the
denial as an unreachable cluster. Document the wrapper in CLAUDE.md and the
cluster-hpc skill. Our HPC Makefile from commit `3f5bbf1` may cover part of
this. Check it and align.

**Source.** w01-tek `hpc.sh`. Commit `c5920e8`.

---

## 6.4 Preflight sizing

**What.** Optional. A sizing job that measures peak GPU memory and steps per
second at several env counts on the real node class before a real launch.
w01-tek's header states the rule: nothing measured on other hardware
transfers.

**Source.** `training/hpc/terrain_sizing.job`. Commit `76aefe4`.

---

## Deferred: terrain

Start terrain only after a flat keeper exists. The full system lives in
w01-tek:

- The procedural tiled arena on one shared heightfield with a lookup grid, in
  `terrain.py`.
- The legged_gym-style promote and demote curriculum with the demote
  projection fix, in `terrain_env.py:240-290`, commit `a1b8aef`.
- The teleport auto-reset wrapper, in `terrain_wrapper.py`.
- The flat recovery row, commit `a0df2a9`.
- Three arena kinds with fingerprint guards.
- Chebyshev measurement radii, commit `61da321`.
- Terrain-relative height, clearance, and contact, commits `37816dd` and
  `4e78a47`.
- The MJWarp heightfield contact cap of 50 per geom pair. Large flat
  colliders must be decomposed into small cells. A humanoid pelvis or torso
  box will hit this cap hard. Commits `01f177e` and `ff17d58`. The gate is
  fd-level printf capture, `check_terrain.py:69-101`.

Corrections to carry when reading w01-tek: `spawn_level` pins initial spawns
only, and respawns still move. The smoke grep for overflow was removed as
dead code, and `check-terrain --backend warp` is the real gate. The
run-report template to copy is the v2 first-run report. The v4 report uses a
different structure.
