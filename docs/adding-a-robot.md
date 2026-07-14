# Adding a robot

PLAN.md names Unitree G1 as the default candidate for robot #2, pending
confirmation with Marcin. This checklist applies to G1 or to any other
robot. `robots/asimov_v1/` is the worked example throughout. `robots/_template/`
carries a stub version of the same files to copy from.

## 1. Vendor the upstream source

Copy the upstream MJCF and meshes verbatim into `robots/<name>/source/`, at
one pinned commit. Do not edit anything under `source/`. Every robot-specific
change goes into `robot.yaml` or `actuators/*.yaml`, applied at build time
through `mujoco.MjSpec` injection.

Write `robots/<name>/PROVENANCE.md` with:

- the upstream repo URL
- the pinned commit hash
- the vendored path mapping (upstream dir to `source/`)
- the license, per upstream
- the vendoring date
- local diffs (state "none" if `source/` is untouched)

`robots/asimov_v1/PROVENANCE.md` is the worked example.

## 2. Write `robot.yaml`

`robot.yaml` is the RobotSpec every env, actuator preset, and export step
reads. Nothing downstream should hardcode a joint, site, geom, or body name.
Everything routes through this file. Required keys:

| Field | Contract |
|---|---|
| `name` | The robot identifier string. |
| `model_xml` | Path to the base MJCF, relative to the robot directory (e.g. `source/xmls/g1.xml`). |
| `actuated_joints` | The joint names to inject actuators for, **in document order from the source XML**. This order is the action/obs contract: `build_spec` injects actuators in this order, and the policy's action vector and joint-position/velocity observations follow it. Reordering this list after training starts breaks every existing checkpoint for the robot. |
| `joint_groups` | A dict mapping group name to a list of actuated joint names. Every actuated joint must belong to exactly one group. Every joint named in a group must be an actuated joint. `load_robot_spec` validates both directions and raises a named error otherwise. Actuator presets carry parameters per group, not per joint. |
| `passive_joints` | A dict mapping unactuated joint name to `{stiffness, damping}`. Applied as a spring/damper back to zero at build time. Skip a joint here if the source XML already gives it its own spring. Asimov's toe joints have one. Its neck joints carry a `passive_upper` default class instead. |
| `foot_sites` | MJCF site names, one per physical foot. Used for foot position/velocity in observations and rewards. |
| `foot_geoms` | MJCF collision geom names covering the feet. May be several per foot (asimov: 5 sole capsules plus 5 toe capsules per side). Each geom is mapped back to its owning `foot_sites` entry by body ancestry at env construction, so every geom listed here must sit somewhere under a `foot_sites` body in the kinematic tree. |

Optional keys:

| Field | Contract |
|---|---|
| `symmetry` | A dict mapping a left joint name to its right counterpart, for symmetry augmentation. |
| `keyframes` | A dict of named poses: `base_pos` (a required 3-vector), `base_quat` (a 4-vector, defaults to identity), `joints` (a sparse map of joint name to angle, defaulting anything unlisted to `0.0`). See the measurement rule below before writing `base_pos`'s z value. |
| `termination_bodies` | MJCF body names, validated to exist against the compiled model. Fall detection currently uses base height and tilt only, so the joystick and sizing envs do not read this field yet. Validate what you list here anyway. |
| `obs_layout` | Free-form dict. No code consumes this yet. Leave it `{}` unless a downstream consumer needs it. |
| `sensors` | A dict with recognized keys `gyro`, `quat`, `linvel`, `acc`, mapping each to an MJCF `<sensor>` name. Envs read the named sensor directly for any key present here, and fall back to a qpos/qvel-derived computation for any key left out. |

### Measure keyframe height against the compiled model

`robots/asimov_v1/robot.yaml`'s `home` keyframe comment records the bug this
rule exists to prevent. v1's `base_pos` z was originally copied from v0's
standing height plus a margin, and was never checked against v1's own foot
geoms. The measured contact height is 0.636 m. Both original keyframes
floated the robot in the air: 0.119 m clear at `home`, 0.061 m clear at
`knees_bent`.

Measure every keyframe's `base_pos` z the same way. Compile the model with
your actuator preset injected, set `qpos` to the keyframe, and run
`mj_forward`. Then compute the exact world-frame bottom of every
`foot_geoms` entry. For a capsule, take the lower of its two world-frame
end-cap centers and subtract its radius. Do not use `geom_rbound`: it
overestimates badly for a capsule lying flat. Shift `base_pos` z so the
lowest foot-geom bottom sits about 5 mm above the floor.
`tests/test_asimov_v1.py`'s `test_keyframe_feet_touch_the_floor` encodes
this as a `0.0 <= lowest <= 0.02` bound. Write the equivalent test for the
new robot.

## 3. Write at least one actuator preset

Add `robots/<name>/actuators/<preset>.yaml`. Two keys are required: `model`
and `groups`. `model` is `pd` or `ideal_torque`. `dc_motor_speed_saturation`
and `delayed` are registered but raise `NotImplementedError`. `groups` maps
a joint-group name to per-joint parameters. `effort_limit` is required in
every group. `kp`, `kd`, `velocity_limit`, `armature`, and `frictionloss`
are optional. Which ones apply depends on the actuator model.
`soft_limit_factor` and `action_scale_factor` default to `0.9` and `0.3` if
omitted. See `docs/configuration.md`'s preset section for how this file
relates to `configs/actuators/<name>.yaml`.

## 4. Run the build and check gates

```bash
./run.sh build --robot <name> --preset <preset>
./run.sh check --robot <name> --preset <preset>
```

`build` compiles `robot.yaml` plus the named preset and writes
`robots/<name>/mjx/<preset>.xml`. `check` steps every keyframe in plain
MuJoCo, plus a short MJX rollout unless `--skip-mjx` is set. It fails on
NaN, or if `|qvel|` exceeds `--max-qvel` (default 100 rad/s). Both gates
must pass before the robot is usable for training.

## 5. Add `tests/test_<name>.py`

Mirror `tests/test_asimov_v1.py`. At minimum:

- `load_robot_spec` parses the new `robot.yaml` and `validate_against_model`
  passes against the compiled source XML.
- `build_spec` compiles with your preset. The compiled actuator order
  matches `robot_spec.actuated_joints`.
- Every keyframe's compiled `base_pos` z matches what `robot.yaml` says.
  This guards against the value drifting from the file without the test
  being updated.
- Every keyframe's lowest foot-geom bottom sits in the same `0.0`
  to `0.02` m band as asimov's test.
- The reset (home) keyframe steps some number of times in plain MuJoCo
  without producing NaN.

## 6. Wire `configs/robot/<name>.yaml`

```yaml
name: <name>
dir: robots/<name>
```

This is a pointer only. It makes `robot=<name>` selectable on the Hydra CLI,
and it lets `train.py` find `robots/<name>/`.

## Ops rules that apply

- Never hand-edit a generated XML under `robots/<name>/mjx/`. It is build
  output. Edit `robot.yaml` or the actuator preset, then rerun `run.sh build`.
- Resolve the config before spending GPU time. Run `./run.sh train
  robot=<name> ... --cfg job --resolve` first, every time.
- Smoke before a bounded train: `./run.sh smoke robot=<name> ...` on CPU
  before any real GPU run.
