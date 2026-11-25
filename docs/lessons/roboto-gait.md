# Lessons: Roboto Origin gait (Torch trainer, v1-v40)

Facts learned while training a Roboto Origin walking gait with the Torch
trainer in PR #11. The runs were v1 to v40. Each run warm-started from the
one before. The chain took about 3.1 G env steps. The PR-head Torch runner
(`wtrain/` at d8f56f7) trained them. `src/humanoid_lab/torch_warp/` replaces
that runner, and the recipe below is written for it and for the Brax trainer.

## Measurement records

The PR did not record the checkpoint, the scenario or the seed for any number
in this file. The PR author supplies them. Until then, read each number as a
single unreproduced observation.

The numbers came from constant-speed probes run by an old measurement script.
The battery runs its own scripted scenarios, so its gait shape metrics will
not reproduce these numbers one for one. See
[Gait shape](../configuration.md#gait-shape).

## Recipe

The best point of the chain is v40. The preset
`configs/experiment/roboto_gait_v40.yaml` restates it:

    ./run.sh train experiment=roboto_gait_v40

The same preset composes for the Torch trainer:

    ./run.sh train-warp experiment=roboto_gait_v40

The recipe is these config keys:

| Key | Value |
|---|---|
| `task.env.gait.freq` | `[0.8, 1.4]`. The gait clock is on and runs at 0.8 Hz at a zero command and 1.4 Hz at full speed. |
| `task.env.obs.state` | ends with `heading_err`. |
| `task.env.obs.privileged` | includes `heading_err`. |
| `task.env.command.deadband` | `0.15`. |
| `task.env.command.pure_slow_prob` | `0.25`. The PR says only "oversampled slow commands", so the value is a placeholder. |
| `ppo.sym_coef` | `1.0`. The PR names the PPO mirror-symmetry loss without a coefficient. 1.0 is the value in the PR-head example command. |
| `task.env.reward.step_length.pair_min` | `true`. |
| `task.env.reward.foot_reach.pair_min` | `true`. |
| `task.env.reward.knee_extend.pair_min` | `true`. |
| `task.env.reward.forward_lean.max_deg` | `0.0`. |

The deadband plus the slow commands gave a start-up without shuffling.

The v40 run.json is not available. The preset marks each value it could not
confirm with `(U)`, and the PR description lists every one. A heading_err
policy cannot be exported yet. See [Refusals](../deploy.md#refusals).

## Measured at v40

- Probes ran at 0.3, 0.6 and 0.9 m/s.
- Achieved speed was 100 to 107 % of the command.
- Heading drift was at most 0.3 deg.
- Torso pitch was 0.5, 0.7 and 1.3 deg at the three speeds.
- No probe showed a flight phase.

## Per-landing rewards get harvested by one leg

A reward paid per landing was collected by one leg while the other leg
shuffled. Pay the min of both feet: set `pair_min: true` on `step_length`,
`foot_reach` and `knee_extend`. Always read the per-foot left and right
table.

## A strong per-event bonus invites tapping

A large bonus per contact event taught the policy to tap a foot to collect
it. Gate event rewards on real steps.

## Narrow exp kernels give no gradient

An exp kernel narrower than the current error is flat at that error. The
policy gets no gradient from it. Use a linear kernel while the error is
large. The three landing terms pay a clipped fraction of their target, which
keeps a slope at any error below the target.

## A frozen clock phase lets the policy specialise one leg

With the gait clock frozen, the `cos_sin` phase observation is a constant
that differs between the legs. The policy used that constant to give each leg
its own role. To avoid it, set `gait.frozen_phase_obs: zeros`.

## Turning the clock off caused shuffling

Runs with the gait clock off shuffled instead of stepping. Stepping
penalties and command ramps collapsed slow walking into standing. The
command deadband fixed slow start-up instead. `command.ramp` stays off in
the preset for this reason.

## Measure achieved speed on every video

Achieved speed sat at 48 % of the command for a while before anyone noticed.
Check achieved speed on every video and every battery run.

## Open issues

- The torso sways in roll. The roll std is about 2.4 deg, and the roll rate
  is about 20 deg/s.
- The torso wobbles in yaw.
- The backward foot kick is larger than the forward reach.
