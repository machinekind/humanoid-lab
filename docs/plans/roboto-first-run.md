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

Gate verdict (2026-08-04, job job-03): PASS. `feet_apex` income per
step rises monotonically from 10M on (0.06 to 1.15e-3, accelerating
over the last three points), reward -5.5 to ~+8 by 30M (v2 gate: +8.8
by 20M), episodes 70 to ~690 steps, KL stable. The late v_loss growth
matches the known non-blocking signature (the blocking one is
huge-at-epoch-0 plus flat reward, absent here). Evidence:
[gate report](https://claude.ai/code/artifact/e3f01b4a-3943-4ab2-8a23-b32731c2d717).
Full run up as job job-04 (16384/512, 4 GPU, v2 sizing verbatim).

Full-run verdict (2026-08-05, train job job-04, eval job-05,
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

## Run ladder

Each step gates the next. Cluster submissions wait for explicit go-ahead.

The run config is `configs/experiment/roboto_walk_v3.yaml`. It pins
robot, preset, network, DR switches, and PPO knobs.

1. Local: `./run.sh train experiment=roboto_walk_v3 --cfg job --resolve`,
   then `./run.sh test`, then a smoke with the experiment and tiny PPO
   sizes re-pinned on the CLI.
2. GPU box: `./run.sh check-contacts` and
   `./run.sh check-friction --robot roboto_origin --preset deploy_pd
   --backend warp`.
3. Bounded run, about 3e7 steps, ALWAYS at the v1-gate sizing: 4096
   envs, batch 128, one GPU. At the full-run sizing (16384/512, 4 GPU)
   a 3e7 run sits entirely inside the recipe's flat early phase (value
   loss astronomical from epoch 0, policy frozen) and reads as a false
   FAIL; the v1 full run shows escape happens between 59M and 134M
   steps, which a bounded run never reaches. Gate: tracking reward
   rises on wandb, an eval video (`--overlay-torque`) looks sane,
   battery passes, and -- the run-3 signal -- `reward/feet_apex`'s
   per-step average RISES over the gate window (v2's micro-swings
   already collect a trickle, so merely nonzero proves nothing; and the
   apex tracker under-reads 2-step flights, so early flatness is not a
   FAIL either).
4. Full run: `ROBOT=roboto_origin ACTUATORS=deploy_pd
   EXPERIMENT=roboto_walk_v3 SEED=0 RUN_NAME=roboto_walk_v3
   ./jobs/train.sh`. NUM_ENVS/BATCH from `jobs/preflight_sizing.sh` on
   the real node class. wandb on.
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
sizing, shaping_tracking_gate moves to run 5.
