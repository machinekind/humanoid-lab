# Lessons: Asimov v1

Facts learned while bootstrapping the asimov_v1 robot into humanoid-lab.
Add to this file as new lessons turn up. Keep each entry a plain statement
of what happened.

## Keyframe base height comes from measuring the compiled model

The vendored XML ships no keyframe. The first `robot.yaml` set `home`'s
`base_pos` z to v0's standing height plus a margin, 0.75 m, and `knees_bent`
to a similarly guessed 0.70 m. Neither value was ever checked against v1's
own foot geoms.

The fix is to measure the exact capsule bottom of every `foot_geoms` entry:
compile the model, set qpos to the keyframe, run `mj_forward`, and take the
lower of each capsule's two world-frame end-cap centers minus its radius.
That measurement showed both keyframes floating the robot in the air:
0.119 m clear at `home`, 0.061 m clear at `knees_bent`. The corrected,
measured `home` height is 0.636 m, with the lowest foot bottom about 5 mm
above the floor. See `robots/asimov_v1/robot.yaml`'s `keyframes` comment and
`tests/integration/test_asimov_v1.py`'s `test_keyframe_feet_touch_the_floor`.

## PD setpoints stay unclamped at the actuator

`PositionPD.inject` in `src/humanoid_lab/actuators/models.py` builds the
injected actuator with `ctrllimited=False`. A PD setpoint past the joint's
kinematic range is not clipped at the MuJoCo actuator level. This follows
the asimov-mjlab convention. Soft limits, `soft_limit_factor` (default 0.9),
apply in the RL layer as an observation and reward concern. An unclamped
setpoint lets the PD loop keep pulling toward the target even past a limit.

## `isfinite` alone does not catch a diverging simulation

MuJoCo's bad-qacc auto-reset replaces a diverging acceleration with something
large but still finite, so a simulation that has blown up can still pass an
`isfinite` check on `qpos`/`qvel`. `check_model.py`'s gate adds a `--max-qvel`
bound (default 100 rad/s) checked alongside `isfinite`, and fails if either
one trips. A stability gate needs both checks. `isfinite` by itself waves
through gross instability.

## Static mjx model fields must be set before `in_axes` is derived

`geom_priority` and other numpy-typed fields on `mjx.Model` are static aux
data in the pytree, part of the treedef rather than a leaf. `in_axes` for
`jax.vmap` is built by mapping over `model`'s pytree structure, in
`dr/randomize.py`'s `make_domain_randomize`. Mutating a static field like
`geom_priority` after `in_axes` has already been derived from `model`, or
after `model` has already been batched, changes the treedef out from under
`in_axes`. `jax.vmap(fn, in_axes=[in_axes, 0])` then fails with a
pytree-prefix mismatch. The error gives no hint that mutation order is the
real cause. The fix is to set every static field first, before deriving
`in_axes` or batching anything.

## The contact peak is a standing robot, not a fallen one

w01-tek sized its warp contact budget off a fallen quadruped, on the rule that
a robot on the ground touches more geometry than one on its feet. Measured on
asimov_v1 (`./run.sh check-contacts`, 2026-07-31) the order is the other way
round: 32 contacts standing, 30 walking, 22 fallen.

Two reasons, both structural rather than accidental. asimov_v1 carries 20 foot
collision geoms — 5 sole capsules and 5 toe capsules per side — and a capsule
against a plane makes up to two contacts, so both feet flat is up to 40
contacts before anything else touches. And a neutral action does not hold
either of this repo's robots up: under the stock preset gains both collapse
from `home` within about a second, so the "standing" regime's own opening
steps, with every sole capsule loaded, are where its peak lands.

The number that mattered: the carried-over default was `naconmax_per_env=32`,
exactly asimov's standing peak. On the warp backend that run would have been
dropping contacts from its second step, silently, with no counter or exception
anywhere. Measure the budget on the robot rather than inheriting it.

## The vendored model is mirror-symmetric to 0.1 mm, not exactly

Measured 2026-07-31 while deriving the mirror maps for port item 3.1. At the
`home` keyframe, asimov_v1's left foot site mirrored about the robot's
xz-plane misses the right foot site by 1.0e-4 m, and paired leg links differ
in mass by up to 4.6e-4 kg. The source is the vendored CAD export itself:
`right_ankle_pitch_link` sits at `pos="0 9.99999999979628E-05 ..."` where the
left one sits at exactly 0, and several other link positions carry
sub-micrometre y differences. (roboto_origin, whose MJCF is generated rather
than exported from CAD, measures 5.9e-7 m and 1.5e-3 kg.)

Two consequences. Anything that compares the two sides has to compare
DISPLACEMENTS, not positions: the sign probe in `envs/symmetry.py` originally
compared mirrored foot-site positions, and every one of its residuals came
out on that 1.0e-4 m floor, which left the two candidate signs a factor of 24
apart instead of 10^4. And this asymmetry is itself a small standing argument
for the mirror augmentation, which averages a bias like it away across the
batch.

The foot geoms mirror too, but not by name: `left_foot1_collision` is the
mirror of `right_foot5_collision`, not of `right_foot1_collision`, and the
XML's own numbering skips `left_foot5` and `right_foot6`. Nothing should pair
this robot's geometry by parsing names.
