# First proper roboto_origin locomotion run

Goal: train `roboto_walk_v1`, the first roboto_origin joystick policy trained
to a real budget, by reproducing RoboParty's own flat velocity recipe as
closely as this pipeline permits. Their recipe walks and runs on the physical
robot, so it is the reference; every place we cannot match it is recorded
below as an accepted delta, not silently dropped.

Reference: `Roboparty/robolab` at commit `6b1c3d9988497c8961dcba77892de32edc1770e1`
(2026-07-26), files `robolab/tasks/direct/base/{rpo_env_cfg.py,base_config.py,
agents/rpo_agent_cfg.py}`. This is the source for every weight and range in
this plan. Note it is newer than the snapshot commit
`robots/roboto_origin/PROVENANCE.md` pins; where the two disagree (the
additive base-mass draw shrank from +-3 kg to +-1 kg) this plan follows the
newer robolab values, and the port commits should update the overlay's
"not ported" comments to cite them.

## Where we start

Already in place, no work needed:

- `actuators/deploy_pd.yaml` carries upstream's hardware PD gains verbatim.
- `configs/robot/roboto_origin.yaml` pins the upstream command envelope,
  joint noise scales, and the ten reward weights whose shape matches ours.
- The same file's "not ported" block records every upstream term we lack,
  with weights. That block is this plan's work list.

The existing `runs/hpc_train_roboto` (1e8 steps, final reward negative) was
a pipeline check and says nothing about the recipe. Upstream trains for
about 1.2e9 env steps (4096 envs x 24 steps x 12000 iterations at 50 Hz);
the budget below matches that, not the old run.

## Code changes, in commit order

1. **Reward terms.** Add the missing upstream terms to `rewards/terms.py`
   and `envs/joystick.py`, pin their weights in the roboto overlay:
   per-group L1 pose deviation (hip yaw/roll -0.03; torso plus arm roll,
   arm yaw, elbow pitch, elbow yaw -1.0; arm pitch -0.06; leg pitch chain
   -0.01), `feet_distance` 0.1 (y-separation of the ankle roll bodies,
   band 0.16 to 0.50 m), `knee_distance` 0.1 (band 0.18 to 0.35 m),
   `dof_pos_limits` -1.0, `joint_vel_l2` -2e-4, `dof_acc_l2` -2.5e-7,
   `upward` 0.4, `feet_contact_without_cmd` 0.1. The torso and arm
   deviation weight is what holds the arms still; the distance bands are
   what prevent leg crossing. Keep our existing shapes where a same-name
   upstream term differs (`feet_slip`, `stand_still`; run 1 also kept
   `feet_air_time`, whose upstream shape run 2 ports -- see Run 2);
   the overlay comments already record those deltas.
2. **Action scale.** Add a flat-radians mode to the ActuatorPreset schema
   and set `deploy_pd` to upstream's flat 0.25 rad. Our
   `action_scale_factor` formula gives each joint a different span, so the
   upstream action-rate and smoothness weights currently scale a different
   quantity; the flat mode makes those weights transfer 1:1. The schema is
   ours to change.
3. **DR.** Add an additive base-mass switch (upstream: +-1 kg on the
   torso, on top of our unconditional 0.7 to 1.3 base scale) and expose
   the push schedule in yaml (upstream: one push every 10 to 15 s, xy
   +-0.5 m/s, z +-0.2 m/s, roll/pitch +-0.52 rad/s, yaw +-0.78 rad/s; our
   current push block is planar-only and not yaml-visible). Set
   `dr.foot_friction` to span upstream's 0.3 to 1.6 static-friction
   envelope and verify the draw with `check-friction --backend warp` on
   the training box.
4. **PPO.** Match upstream where brax has the knob: `ppo.discounting=0.994`,
   `ppo.gae_lambda=0.9`, `ppo.entropy_cost=0.005`,
   `ppo.learning_rate=1e-4`, `network=large` (their [512, 256, 128] ELU).
   `build_ppo_params` starts from playground's Go1 config; any of these
   keys it does not carry gets added there rather than overridden blind.

## Accepted deltas for run one

- **Termination.** Upstream terminates on torso/thigh contact. Our model
  has collision geoms only on the feet, so height/tilt termination
  (min_height 0.45, the overlay's values) substitutes.
- **Contact-force terms.** `undesired_contacts`, `feet_force`,
  `feet_stumble`, `feet_height` need contact sensors or foot scanners we
  do not have. Skipped, weights stay recorded in the overlay.
- **Observation history.** Upstream's actor sees a 10-frame observation
  history plus 3 past actions; ours is single-frame plus `last_action`.
  Accepted for run one; revisit before any sim2real attempt, together
  with the action-delay machinery the joystick docstring already defers.
- **Gait clock.** Upstream has none; our joystick's antiphase clock and
  `feet_phase` term stay, since they are this env's stepping mechanism
  and the battery and goldens depend on them.
- **LR schedule.** Upstream uses an adaptive-KL learning rate; brax PPO
  has none. Fixed 1e-4.
- **Heading command.** Upstream steers yaw through a heading controller;
  we sample `wz` directly inside the same +-1.57 envelope.

## Run 2: make stepping pay

Run 1 (`roboto_walk_v1`, the full 1.209e9 budget) executed the recipe
without failures and reached reward 17.6, but never stepped:
`feet_air_time` stayed at 0 for the whole run, the battery shows every
command served by leaning and foot-shuffling, and every episode still
ends in a fall. Evidence: the [run report](https://claude.ai/code/artifact/69344fbf-f826-4a54-be31-325b5d5fc16a)
and wandb run cdoixkux.

The defect is the one upstream shape run 1 kept unported. Their
`feet_air_time_positive_biped` pays every step of single stance, so the
gradient toward lifting a foot exists from the first policy. Our
`feet_air_time` pays only at landing, which a zero-swing policy never
reaches, and `feet_phase` at `phase_sigma` 0.002 is flat at zero
clearance.

Run 2 changes exactly two things, both pinned in the roboto overlay:

1. `feet_air_time_biped`, the upstream shape ported into
   `rewards/terms.py`: weight 0.25, threshold 0.4 s. The landing-event
   `feet_air_time` goes to 0 so a swing is not priced twice.
2. `phase_sigma` 0.002 -> 0.01, so a partial lift scores.

`feet_apex` and `shaping_tracking_gate` stay off: they are our own
measured knobs, not upstream's, and they only enter if run 2 still
shuffles. Everything else -- budget, PPO, DR, actuators, network -- is
run 1 verbatim, and the experiment preset is renamed `roboto_walk_v2`.

Gate verdict (2026-08-03): PASS on the v1-gate sizing. Reward reaches
+8.8 by 20M steps (v1 gate: +5.8 by 12M) and swings appear from zero:
`feet_air_time_biped` earns from the first epoch and landing events
multiply (short swings under 0.1 s; lengthening them is the full
budget's job). Full evidence, the 2x2 attribution matrix, and the
sizing lesson below:
[gate report](https://claude.ai/code/artifact/c6c62b9f-9260-461c-ade8-442622083058).

Full-run verdict (2026-08-03, job job-01,
wandb 7n62pp8j,
[full report](https://claude.ai/code/artifact/ffb483c5-7a12-4a0f-86b5-a608b7f9ed1d)):
reward 21.8 (v1: 17.6) but still no walking. The policy moved from
v1's zero contact breaks to rhythmic micro-steps: flight times around
0.1 s, single-stance income earned, but clearance stays under the
battery's 5 mm swing threshold, and command serving is bit-for-bit
v1's leaning (spin still dead). The new terms pay for lift-off and
single support, none pays for HEIGHT, and the optimizer collected them
with millimeter lifts. Run 3 candidate: enable `feet_apex` (prices the
swing's peak against a 5 cm target at landing; now that landings exist
it has events to pay), optionally `shaping_tracking_gate`. Not a
keeper; the deploy pair is exported but unpublished.

## Run 3: price the swing height

Run 2 proved the air-time gradient works and exposed what it cannot buy:
the optimizer collected lift-off and single-stance income with millimeter
lifts, because no term prices clearance HEIGHT. Run 3 changes exactly one
thing, pinned in the roboto overlay:

1. `feet_apex: 5.0` (was 0). Ours, not upstream's: pays each completed
   swing once, at touchdown, for how close its peak clearance came to
   `apex_target`. Run 2's rhythmic landings give it events to pay from
   the first epoch. The weight is the w01-tek stiff-ladder value for this
   exact term, where it measured 3 to 5 cm swings; the term is sparse
   (once per landing) so it needs an order more weight than the per-step
   terms.
2. `apex_target: 0.08` (was the 0.05 default), the re-derivation the
   default's own comment asks for: this env's gait clock demands 0.08 m
   of swing (`gait.swing_height`), and 0.05 is the quadruped leg's
   number. One reward delta, both knobs of the same term. The first
   gate submission (job-02) died in a node failure before logging a
   step; the gate re-ran as job job-03 with this pin in place, so the
   gate and the full run share the target.

An adversarial review of this plan (2026-08-04) pinned down what run 3
does and does not test: feet_apex is translation-neutral (a stomper and
a walker collect the same apex income; only the tracking differential,
about 0.6-0.7/step under a 0.5 m/s command, favors walking, and v2
proved that differential alone does not buy translation). So run 3
tests "does the robot lift," not "does it walk", and its verdict
criterion is `vel_err`, not reward: walk_ramp vx error must beat v2's
0.494, and battery swings must appear (median apex over 2 cm). Run 4 is
pre-registered now so a stomping v3 costs one ladder step, not a
redesign cycle: `shaping_tracking_gate: true`, the measured anti-stomp
knob (under an unserved command k_lin ~ 0.37, so the gate scales
shaping ~3x in walking's favor without zeroing the gradient).

Budget stays 1.209e9: v1 and v2 both plateau by ~400M (v2 peaks at
+22.7 @ 403M and is flat for the last 800M), so steps were not the
binding constraint; if the v3 curve is still climbing at cutoff, an
extension becomes its own ladder step. The preset is renamed
`roboto_walk_v3`.

Gate verdict (2026-08-04, job job-03,
wandb jghk30b9): PASS. `feet_apex` income per
step rises monotonically from 10M on (0.06 to 1.15e-3, accelerating
over the last three points), reward -5.5 to ~+8 by 30M (v2 gate: +8.8
by 20M), episodes 70 to ~690 steps, KL stable. The late v_loss growth
matches the known non-blocking signature (the blocking one is
huge-at-epoch-0 plus flat reward, absent here). Evidence:
[gate report](https://claude.ai/code/artifact/e3f01b4a-3943-4ab2-8a23-b32731c2d717).
Full run up as job job-04 (16384/512, 4 GPU, v2 sizing verbatim).

Full-run verdict (2026-08-05, train job job-04, eval job-05,
wandb ob4mbu2s,
[full report](https://claude.ai/code/artifact/00289d8d-d306-4323-a9f7-81e84016b5b9)):
FAIL on the pre-registered criteria. walk_ramp vx error 0.496 vs v2's
0.494, battery swings 0 everywhere (clearance never crosses 5 mm),
spin still dead (-3.6 deg on a -360 command); final reward 22.09 (v2:
21.79) -- which is exactly why the verdict criterion is vel_err, not
reward. The new fact is a SIZING PARADOX: at the gate sizing
(4096/128) feet_apex income per step reached 1.15e-3 by 30M, while the
full sizing (16384/512) ends 1.2e9 at 0.19e-3, six times lower, having
collapsed to 0.04e-3 right after the dead phase -- the 16384/512 dead
phase appears to select the leaning basin, not merely delay learning.
Run 4 recommendation: the full budget at the gate sizing (4096 envs,
batch 128, rewards untouched) -- a clean test of the sizing hypothesis
with two independent evidence points (both v2's and v3's gates learn
richer gaits than their full runs on identical code), and faithful to
upstream's own 4096 x 12000 recipe. The pre-registered
`shaping_tracking_gate` moves to run 5, for a policy that steps tall
but ignores commands. Not a keeper; the deploy pair is exported
(max |diff| 6.6e-07) but unpublished.

## Run 4: the sizing hypothesis

Same code, same rewards, same 1.209e9 budget as run 3; the only change
is the PPO sizing, from 16384 envs / batch 512 / 4 GPUs back to the
gate's 4096 / 128 / 1 GPU. Why sizing would decide WHAT is learned
rather than merely how fast -- three mechanisms, each with a measured
track in runs 2 and 3:

1. The budget is counted in env steps, not optimizer updates. 4x more
   envs per iteration means 4x fewer gradient updates on the same
   budget (~3000 vs upstream's ~12000), at the same fixed LR 1e-4
   (brax has no adaptive-KL schedule), so the network simply moves 4x
   less per unit of data. The critic needs a fixed number of updates
   to fit a reward scale that includes -200 terminations; counted in
   env steps that warm-up now takes 4x longer. Measured: the 16384/512
   dead phase (v_loss 3.3e7 at epoch 0, policy frozen at KL ~2e-4)
   lasts to 59-134M, while 4096/128 learns from ~5M.
2. The dead phase is not neutral: it spends the exploration the sparse
   terms need. At epoch 0 the random policy's flailing pays feet_apex
   0.13/episode; nothing reinforces it while the advantages are
   garbage, and survival pressure prunes exactly those trajectories.
   By 134M apex is down to 0.02: when learning finally starts, the
   swing data is no longer in the batch. Basin selection, not delay.
3. Small-batch gradient noise doubles as exploration in weight space.
   At batch 128 one lucky high-clearance trajectory can dominate a
   minibatch and move the policy; at 512 it is averaged away against
   hundreds of leaning episodes.

Two independent evidence points: run 2's 2x2 diagnostic (identical
code learns at 4096/128 by 30M and is dead at 16384/512 at 59M, with
v1's rewards too, so it is the sizing, not the reward port) and run
3's apex trajectory (gate 1.15e-3/step at 30M vs the full run's
0.04e-3 right after the dead phase, 0.19e-3 at 1.2e9). The competing
explanation -- the gate curves are a honeymoon that would have
collapsed anyway -- survives both points, and run 4 is the
discriminating test: if the 4096/128 curve holds the gate's apex
growth past 30M (order 1e-3/step in the 100-400M window where the v3
full run sat at 0.04e-3), the hypothesis stands even if the final
policy does not fully walk; if it collapses onto the v3 full-run
trajectory, sizing is exonerated and the reward ladder resumes at run
5 (`shaping_tracking_gate`).

No new gate: run 3's gate (job-03, PASS) ran this exact config at
this exact sizing, and run 4's first 30M double as its reproduction
check (same seed 0). Walking verdict criteria unchanged from run 3:
walk_ramp vx error below v2's 0.494 and battery swings with median
apex over 2 cm. Throughput from the gate (3e7 steps in 12:12 wall
including compile, one hopper) puts the full budget at about 5h;
submitted on `<gpu-partition>` with a fixed walltime limit. The preset is renamed
`roboto_walk_v4`, content unchanged from v3: the sizing lives in the
submit line, not the yaml.

Verdict (2026-08-05, train job job-06, 3h48 on one hopper, eval
job-07, wandb z4nzo8dm,
[full report](https://claude.ai/code/artifact/383b1526-bbf1-4d21-bed3-52b28bd92fd0)):
PASS on both pre-registered criteria -- the first walking policy of
the campaign. walk_ramp vx error 0.234 (criterion: below 0.494; v3:
0.496), swings in every moving scenario with median apex 6.5-9.3 cm
against the 8 cm target (criterion: over 2 cm; v3: zero swings
anywhere), and the spin probes, dead since run 1, complete: 340 deg
on +360, -343 on -360 (v3: -4.6/-3.6). Zero falls, torque_sat 0; the
only ATTENTION flag is stand vibration 0.62. The sizing hypothesis is
confirmed on the discriminating signal with two orders of magnitude
to spare: apex per step held the gate's growth through the 100-400M
window (0.55 -> 3.1 -> 20e-3, where v3 sat at 0.04e-3) and ended at
40e-3, 200x v3's 0.19e-3 on identical rewards; final reward 48.4 vs
22.1. GPU-hours match v3 (3h48 x 1 vs 58min x 4), so 4096/128/1 GPU
becomes the default sizing for every subsequent run. Remaining
quality items for later rungs: stand vibration, medium-hard
touchdowns (softness 0.24-0.43), vx error still 0.234. Run 5
(shaping_tracking_gate) is no longer needed for its pre-registered
purpose (the policy tracks commands); it stays in reserve as a
tracking-sharpening option. Keeper/HF decision is Marcin's; the
deploy pair is exported and unpublished.

Run 4b, the budget extension (Marcin's call after watching the
videos: the gait works but limps). Warm-start from run 4's final
checkpoint via the stock `restore` key
(`restore=runs/roboto_walk_v4/checkpoints/001201766400`; brax
restores normalizer + policy + value params, optimizer state and the
step counter start fresh), same preset, sizing, and seed, another
1.209e9 steps -- 2.4e9 total, the w01-tek-terrain scale. Warm-start
over a from-scratch 2.4e9 run because the checkpoint exists, ~4h
beats a from-scratch rerun against the queue walltime limit, and the question is whether
the still-climbing curve anneals the limp, not whether a longer run
reproduces it. The epoch-0 readout doubles as the restore gate: a
restored policy opens near reward 48 / episode 975 (a silent restore
failure would open at -5.5 / 70 like a scratch run). Extension
criteria, pre-registered: walk_ramp antiphase above 0.675 (the limp
signature; turn/spin sit at 0.90+) and vx error below 0.234, with no
falls and swings retained; stand vibration below 0.5 would clear the
only ATTENTION flag. The limp itself is also a visual call on the
videos.

## Run 5: the style package

Run 4b settled that the v4 signal, not the budget, is the binding
constraint: a full second 1.2e9 bought battery inches on a flat curve.
Marcin's directive for run 5: the fastest visible move toward natural
walking without imitation (mocap/AMP is explicitly off the table). One
package of four coordinated reward-signal changes, each aimed at a
defect visible on the v4_ext videos:

- `knee_stance` -2.0 (tol 0.15 rad, new term): nothing in the v4 signal
  prices the stance leg's shape, so the permanently-crouched knee
  (home keyframe holds 0.3 rad) is reward-neutral and the optimizer
  keeps it. Charging stance-leg flexion beyond the cone -- swing leg
  free, touchdown absorption inside the cone free -- asks the policy to
  carry the body straighter, the single biggest "this is not walking"
  visual. Masked by `moving`: at zero command the stand terms anchor
  the knees on the keyframe's 0.3, and an unmasked cone would fight
  that anchor into standing dither.
- `gait_symmetry` -2.0 (new term): the limp signature priced directly.
  Per-foot EMAs of completed swing and stance durations, charged on
  their relative left-right difference, cadence-invariant, armed only
  once both feet have a completed duration on record (the first step of
  an episode is one-legged by definition). walk_ramp antiphase 0.740
  against 0.90+ on turn/spin is the number this term exists to move.
  This is a reward term, not mirrored-world augmentation -- the
  augmentation route is falsified (w01-tek v3 post-mortem: mirrored
  worlds symmetrize the data, not the policy).
- `gait.freq` 1.0-2.0 -> 0.9-1.4 Hz, `biped_air_time_threshold` 0.4 ->
  0.5 s: the clock is the cadence generator (feet_phase scores against
  it and the policy observes it), so the shuffle is partly commanded.
  A slower clock buys longer strides at the same speed; the raised
  clamp keeps the longer swings it asks for (up to ~0.55 s at duty
  0.5) paid instead of clipped at 0.4 s.
- `energy` -1e-4 -> -3e-4: a triple, not an order of magnitude --
  enough to lean the optimum toward pendulum-like motion without
  making standing the best-paid behavior.

From scratch, not warm-started from v4_ext: run 4's own mechanism says
behavior basins are selected early, and a policy already settled into a
crouched shuffle is exactly the basin the new terms are supposed to
price out of existence -- warm-starting would hand it back as the
starting point. Sizing 4096/128/1 GPU (the run-4 default), seed 0,
budget 1.2e9, DR and PPO knobs unchanged from v4.

Four confounded changes in one run is a deliberate trade against
Marcin's speed directive, de-risked by a bounded gate: 3e8 steps
(about 1h wall on one hopper), long enough to clear the dead phase and
show the new terms optimizing. Gate criteria: tracking reward rising;
`reward/knee_stance` and `reward/gait_symmetry` per-step magnitudes
FALLING over the gate window (the new signals must actually optimize,
not just subtract); battery on the gate checkpoint shows swings with
no falls; eval video sane. Pre-committed on gate FAIL: cut the package
to gait_symmetry + energy (drop knee_stance and the clock change
first), then isolate.

Full-run criteria, pre-registered: walk_ramp antiphase above 0.85
(v4_ext: 0.740); walk_ramp vx error below 0.20 (v4_ext: 0.187, so
hold, not regress); walk_ramp swings 8-13 per 300 steps (below
v4_ext's 14 = the cadence actually dropped, above zero-risk floor =
still stepping); median apex at or above 6.5 cm (no regress); zero
falls and zero torque saturation. The straighter stance leg is a
visual call on the videos, recorded in the report next to the
numbers.

Gate 1 verdict (2026-08-06, train job-11, 54:55, eval job-12,
wandb 6qbyy66n): FAIL, and cleanly diagnostic. Zero swings in every
battery scenario -- the policy stands bolt upright under every
command (walk_ramp vx err 0.501, spins at -2/+10 deg on +-360). The
full package never left the stand basin the whole ladder fought to
escape: feet_apex per episode 0.14 -> 0.05 and flat to 3e8, where
run 4 at the same point read ~2 (40x more, already walking).
Standing pays ~0.45/step of the additive tracking kernels plus
~0.65/step of feet_phase (sigma 0.01 scores a zero-clearance foot at
two-thirds of the kernel); the package's walk taxes tipped the
escape economics the wrong way. Two term-level findings from the
curves: knee_stance's raw magnitude rose monotonically (2.5 -> 13.8
per episode) -- the tol 0.15 cone taxes the loading-response flexion
walking NEEDS, so it is a tax on gait, not on crouch; and
gait_symmetry fell to ~0.1 as stepping vanished -- a policy that
never steps never arms the term, so standing is symmetry-free by
construction. The pre-committed cut applies and the curves endorse
it: knee_stance and the clock change dropped, gait_symmetry -2.0 and
energy -3e-4 stay (energy's measured cost while standing is ~0.17
per episode, too small to be the killer). Gate 2 =
roboto_walk_v5_gate2, same 3e8 bound, same criteria; 3e8 is also
exactly the window in which run 4 escaped the stand basin at this
sizing, so a second FAIL is a real signal, not impatience. A rung
that re-approaches the stance-knee idea later must price flexion
only OUTSIDE the gait's own loading window (or gate on slow knee
velocity), and the symmetry term needs a step-rate floor before it
can be trusted alone.

Gate 2 verdict (2026-08-07, train job-13, 55:55, eval job-14,
wandb ss4ithq2): FAIL the same way -- zero swings, standing under
every command, apex per episode flat at ~0.05 from 30M on (the run-3
gate, same recipe minus the style terms, read ~0.77 at 30M). That
acquits the budget and convicts the remaining term: energy's
measured standing cost is ~0.17/episode (noise), but gait_symmetry
at -2.0 charges CONTINUOUSLY once armed, and a first clumsy gait
saturates the relative-asymmetry kernel near its (2d/d)^2 = 4
per-pair maximum -- about 0.32/step of tax on exactly the fragile
first-steps window, while standing never arms the term and collects
~0.45/step of tracking free. The exploration path to walking was
priced out, not walking itself. Gate 3 applies the isolate step with
the mechanism fixed: reward.gait_symmetry_cap (new knob, default
1.0) clips the summed relative asymmetry, bounding the worst-case
fee at scale*cap*dt, and the weight drops to -1.0 -- worst case
-0.02/step during exploration, a settled 20% limp still pays ~0.5
per episode. Same 3e8 bound, same criteria. If gate 3 fails too, the
style rung stops for a design review: no more submissions on this
branch without Marcin.

Gate 3 verdict (2026-08-07, train job-15, 57:17, eval job-17 --
the first eval, job-16, ran against the WRONG run dir through a sed
slip, `v5gate2` not matching `v5_gate2`, and re-scored gate 2's
checkpoint; resubmitted correctly -- wandb i2k1q2pp): FAIL on the
pre-registered battery criterion, zero swings in every scenario,
standing under every command. The cap worked as designed (charge
~0.005/step, far under the cap) and the training curve moved where
gates 1-2 were flat: apex per episode rose 0.04 -> 0.18 through the
window. But 0.18 is micro-motion under training noise (pushes, DR,
stochastic actions); the deterministic eval policy still stands. The
mechanism, three gates deep, is now clear: ANY penalty that scales
with stepping activity taxes the fragile phase where stepping barely
pays -- the optimizer sits at the indifference point however small
the absolute charge, because what matters is the marginal advantage
of the first steps, not the episode total. THE STYLE RUNG STOPS HERE
per the gate-2 pre-commit. Design-review options for Marcin, in
recommended order: (a) warm-start v4_ext + the capped symmetry term:
the from-scratch argument was for reshaping the gait, but a limp fix
is a WITHIN-basin adjustment -- start inside the walking basin and
let the term shift the policy there (run 4b proved restore); (b)
flip symmetry to an income (pay matched left-right pairs instead of
charging mismatch), which pushes exploration toward stepping instead
of away from it; (c) abandon the style rung and take the run-5
reserve (shaping_tracking_gate as tracking sharpening). No more
cluster submissions on this rung without his call.

## Run ladder

Each step gates the next. Cluster submissions wait for explicit go-ahead.

The run config is `configs/experiment/roboto_walk_v5.yaml`. It pins
robot, preset, network, DR switches, PPO knobs, and the style package.

1. Local: `./run.sh train experiment=roboto_walk_v5 --cfg job --resolve`,
   then `./run.sh test`, then a smoke with the experiment and tiny PPO
   sizes re-pinned on the CLI.
2. GPU box: `./run.sh check-contacts` and
   `./run.sh check-friction --robot roboto_origin --preset deploy_pd
   --backend warp`.
3. Bounded run at 3e8 steps (run 5's gate; earlier rungs used 3e7),
   ALWAYS at the 4096/128/1-GPU sizing. At the old full-run sizing
   (16384/512, 4 GPU) a bounded run sits entirely inside the recipe's
   flat early phase (value loss astronomical from epoch 0, policy
   frozen) and reads as a false FAIL; the v1 full run shows escape
   happens between 59M and 134M steps. Gate criteria for run 5 are in
   its section above (new-term magnitudes falling, battery swings, no
   falls, sane video).
4. Full run: `ROBOT=roboto_origin ACTUATORS=deploy_pd
   EXPERIMENT=roboto_walk_v5 SEED=0 RUN_NAME=roboto_walk_v5
   NUM_ENVS=4096 BATCH=128 ./jobs/train.sh` on one GPU (cluster-side:
   hpc/train.job with the same NUM_ENVS/BATCH exports; extra Hydra
   overrides go in EXTRA, never RUN_ARGS -- the run-4b false start).
   wandb on.
5. After: `battery`, `report`, an eval video per battery scenario, and
   `export` of the deploy pair.

## Out of scope

AMP, BeyondMimic, Parkour, and motion retargeting (need GMR motion data);
rough terrain and the height scanner; the observation-history and
action-delay port; contact-based termination.

Changelog: 2026-08-02, prerequisites implemented on this branch; the
ladder now uses the roboto_walk_v1 experiment preset. 2026-08-03, run 2
added after run 1's zero-swing verdict: feet_air_time_biped port,
phase_sigma widening, preset renamed roboto_walk_v2. 2026-08-03, gate
PASS; gate sizing pinned to 4096/128/1 GPU after the full-sizing false
alarm. 2026-08-03, full run up as job job-01 (v1 recipe verbatim,
roboto_walk_v2 preset). 2026-08-03, full-run verdict: micro-steps, not
walking; eval chain and wandb links recorded above. 2026-08-04, run 3
added: feet_apex 5.0, preset renamed roboto_walk_v3, budget unchanged.
2026-08-04, after adversarial review: apex_target re-derived to 0.08,
verdict criterion moved to vel_err, run 4 (shaping_tracking_gate)
pre-registered. 2026-08-04, gate PASS (job-03); full run up as job
job-04. 2026-08-05, full-run verdict: still no walking, sizing
paradox recorded; run 4 recommendation = full budget at the gate
sizing, shaping_tracking_gate moves to run 5. 2026-08-05, run 4 added
on Marcin's go: sizing-hypothesis mechanism and discriminating signal
recorded above, preset renamed roboto_walk_v4, no new gate.
2026-08-05, run 4 verdict: PASS on both criteria, first walking
policy; sizing hypothesis confirmed, 4096/128 is the default sizing
from here on. 2026-08-05, run 4b added on Marcin's go: budget
extension by warm-start, criteria pre-registered on the limp
signature. 2026-08-05, run 4b false start: the first submission
(job-08) passed the restore override in RUN_ARGS, which
hpc/train.job overwrites (its contract is EXTRA), so the run
silently started from scratch -- caught by the restore gate at the
first readout (reward -5.5 and the 134M point retracing run 4's
curve), cancelled at 40 min, resubmitted as job-09 with
EXTRA=restore=... . 2026-08-05, run 4b verdict (train job-09,
3h49, restore gate PASS at 49.3, eval job-10,
wandb n0bhubrv,
[report](https://claude.ai/code/artifact/d8486e75-45f7-4b32-8da8-4fa75be77d25)):
PASS on both extension criteria -- walk_ramp vx err 0.187 (was
0.234), antiphase 0.740 (was 0.675); softer touchdowns (0.36),
apex med 7.6 cm, spins 347/-343, zero falls. But the training curve
sat flat at reward 46-49 for the whole second 1.2e9 (apex/step
38-44e-3): the battery gains are real yet bought at a full budget of
diminishing returns, so the next rung should change the signal (leg
symmetry term or tracking sharpening), not add steps. stand
vibration unchanged at 0.65 (the one ATTENTION flag);
walk_to_stop antiphase slipped 0.672 -> 0.658. Not auto-promoted;
keeper/HF decision is Marcin's. Videos in
~/Documents/robot/roboto_walk_v4_ext/. 2026-08-06, run 5 added on
Marcin's go ("dawaj pakiet"): the style package -- knee_stance and
gait_symmetry terms, slower gait clock, energy x3 -- preset renamed
roboto_walk_v5, from scratch at the run-4 sizing, gate 3e8 and
full-run criteria pre-registered above. 2026-08-06, gate 1 FAIL
(job-11): full package collapsed into stand-under-command, verdict
and term-level diagnosis above; pre-committed cut applied (preset now
carries gait_symmetry + energy only), gate 2 submitted as
roboto_walk_v5_gate2. 2026-08-07, gate 2 FAIL (job-13): still
standing; uncapped continuous symmetry charge convicted, verdict
above. Gate 3 = capped term (gait_symmetry_cap 1.0) at weight -1.0,
submitted as roboto_walk_v5_gate3; a third FAIL stops the rung for
design review. 2026-08-07, gate 3 FAIL (job-15, eval job-17 after
a wrong-RUN_NAME resubmit): still standing in eval despite a rising
training-time apex; activity-scaled penalties tax the fragile
first-steps margin however small the charge. Style rung STOPPED;
design-review options (warm-start v4_ext + capped symmetry /
symmetry-as-income / run-5 reserve) recorded above, decision is
Marcin's. Gates report:
[artifact 5393f71c](https://claude.ai/code/artifact/5393f71c-9a26-43b0-8600-90823b8bad72);
gate videos in ~/Documents/robot/roboto_walk_v5_gate{,2,3}/.
