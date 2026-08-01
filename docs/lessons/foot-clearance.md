# Lessons: reference foot clearance to the floor, not the keyframe

`HumanoidEnv._foot_clearance` originally read `site_z - site_z_at_keyframe`.
The reset keyframes deliberately float the robot a few millimetres (the
keyframe measurement rule places the lowest sole 0-20 mm above the floor),
so that margin was baked into every clearance reading: a planted foot read
about -5 mm on asimov_v1 and -3 mm on roboto_origin, never 0, and every
swing apex read low by the same offset.

What that cost, measured before the fix:

- `feet_phase` (scale 1.0, on by default) scored clearance against a stance
  target of exactly 0, which a planted foot could never reach: 0.010-0.024
  reward/step of unremovable floor -- the same class of anchor bug
  `real_pose_ref` exists to remove, and that one was justified at 0.032
  reward/step.
- `feet_landing`'s "free at `glide_height`" gate actually charged
  `offset/glide_height` of full weight at a foot physically at
  `glide_height`.
- `feet_apex` saturated only at `apex_target + offset`.
- The offset was silently coupled to `reset_keyframe`: switching keyframes
  rescaled every clearance term for reasons unrelated to gait.

The fix, now in `envs/base.py`: the construct-time `mj_forward` measures,
per foot, the vertical distance from the foot site to the lowest point of
that foot's own collision geoms (capsule bottoms, via
`_foot_geom_foot_idx`), and `_foot_clearance` subtracts that site-to-sole
offset instead of the keyframe site height. A planted foot reads ~0
regardless of the keyframe's float margin, and `apex_target`,
`glide_height` and the eval `AIRBORNE_M` band are physical metres.

Kept for reference: the site-to-sole distances are 0.009732 m (asimov_v1)
and 0.012000 m (roboto_origin, matching its robot.yaml geometry: site at
body z -0.033, sole plane at -0.045). A `geom_z - radius` clearance measure
without the per-foot minimum would read exactly 0 for a resting sphere at
any orientation, which is why the offset is measured against the sole
plane, per foot, at construct time.
