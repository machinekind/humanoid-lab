# Lessons: foot clearance is measured from the keyframe, not the floor

Found in the phase-1 code review, 2026-07-31. The numeric fix is deliberately
deferred; this file records the measurement so nobody re-derives it or reads
the thresholds as physical heights.

## What the offset is

`HumanoidEnv._foot_clearance` is `site_z - _foot_site_rest_z`, and
`_foot_site_rest_z` is measured by `mj_forward` at the **reset keyframe**.
Those keyframes deliberately float the robot: `robots/asimov_v1/robot.yaml`
says its `base_pos` z is shifted so the lowest foot bottom sits about 5 mm
above the floor. That margin is baked into the reference, so every clearance
reading is biased low by it.

Measured on the compiled models:

| | asimov_v1 | roboto_origin |
|---|---:|---:|
| keyframe gap, lowest sole to floor | 0.004920 m | 0.003105 m |
| `_foot_site_rest_z` | 0.014653 m | 0.015105 m |
| true site-to-sole distance | 0.009732 m | 0.012000 m |
| `_foot_clearance` with the sole ON the floor | **-0.004920 m** | **-0.003105 m** |

The roboto number checks out against its own robot.yaml: the site is at body
z -0.033 (the capsule centerline) and the sole plane at -0.045, so 0.012 m.

So a planted foot reads about -5 mm / -3 mm, not 0, and a swing foot at a
physical height `h` reads `h - offset`.

The docstring used to claim this reproduced w01-tek's geom-bottom semantic. It
does not: w01-tek computes `geom_xpos_z - FOOT_RADIUS`
(`training/wojtek_rl/base.py:292-299`), which is exactly 0 for a resting
sphere at any orientation.

## What it costs

- `feet_phase` (scale 1.0, **on by default**) scores clearance against a
  stance target of exactly 0, which double stance can never reach. Ceiling
  measured at `phase_sigma=0.002`: 0.976 on asimov, 0.990 on roboto — 0.010
  to 0.024 reward/step of unremovable floor. This is the same class of anchor
  bug `real_pose_ref` was added to remove, and that one was justified at
  0.032 reward/step.
- `feet_landing`'s gate is documented as "0 at `glide_height` and above". It
  actually charges `offset/glide_height` of full weight at a foot physically
  at `glide_height` — 0.164 on asimov, 0.104 on roboto — and reaches 0 only
  at 0.0349 m / 0.0331 m.
- `feet_apex` saturates only once the foot is physically at `apex_target +
  offset`: 0.0549 m / 0.0531 m against an `apex_target` of 0.05.
- The offset is silently coupled to `reset_keyframe`. Switching asimov to
  `knees_bent` changes the float margin and rescales every clearance term,
  for reasons that have nothing to do with ankle pose.

`feet_apex` and `feet_landing` are both at scale 0.0 by default, so only
`feet_phase` bites today.

## Why the fix is deferred

The fix is to reference the rest height to the floor at the same
construct-time `mj_forward`: per foot,
`rest_z_i = site_xpos_z_i - min(geom_xpos_z - radius)` over that foot's own
geoms, via `_foot_geom_foot_idx`. That yields 0.009732 (asimov) and 0.012000
(roboto), i.e. exactly the site-to-sole distances above, and makes a planted
foot read 0.

`feet_phase` is a pre-port term at scale 1.0, so the change moves stock
rewards and `tests/integration/test_golden_baseline.py` has to be
regenerated. Doing that mid-port would break the bit-exact chain the whole
port is gated on. **Do it after the port lands, as its own change, with the
golden regenerated in the same commit** — and re-derive `apex_target` and
`glide_height` at the same time, since both are already flagged for
re-derivation off w01-tek's 0.21 m leg.
