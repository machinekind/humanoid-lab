# Adding a robot

This checklist applies to any robot. `robots/asimov_v1/` is the worked
example throughout; `robots/roboto_origin/` (robot #2) is the worked example
for a source XML that needs `model_patches`. `robots/_template/` carries a
stub version of the same files to copy from.

## 1. Vendor the upstream source

Copy the upstream MJCF and meshes verbatim into `robots/<name>/source/`, at
one pinned commit. Do not edit anything under `source/`. Every robot-specific
change goes into `robot.yaml` or `actuators/*.yaml`, applied at build time
through `mujoco.MjSpec` injection.

`build_spec` always deletes every actuator already in the source XML, and
every actuatorpos/actuatorvel/actuatorfrc sensor that does not resolve to an
actuated joint, before injecting its own actuators. A vendored source XML
that ships its own `<actuator>` block and actuator sensors needs no special
handling for this; see "model_patches" below for the other build-time
patches available for a source XML that isn't otherwise MJX-ready.

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
| `model_patches` | Build-time patches for a source XML that isn't MJX-ready as vendored: `<option>` overrides, injected sites, injected collision geoms, and mesh-collision handling. Every sub-key is optional. See "model_patches" below. |

### model_patches

Reach for `model_patches` when the source XML has no sites or named
collision geoms to point `foot_sites`/`foot_geoms` at, or when its
`<option>` settings aren't MJX-compatible. Omit the whole section if the
source XML needs none of this. `build_spec` applies it right after loading
the source XML, before actuator injection: `options`, then the actuator
strip described above, then `sites`, then `geoms`, then `mesh_collisions`.

`options` overrides `<option>` values. The allowed keys are `solver`,
`iterations`, and `timestep`. `solver` is one of `pgs`, `cg`, `newton`. MJX
does not support PGS. Pick `newton` or `cg` for a source XML that ships
`solver="PGS"`.

`sites` injects a named site into a body. Each entry needs `body` and
`pos`. `quat` is optional and defaults to identity.

`geoms` injects a named collision primitive into a body. Each entry needs
`body`, `type`, and `size`. `type` is one of `box`, `capsule`, `sphere`.
`pos` and `fromto` are both optional; set at most one, matching MJCF geom
semantics. `quat` is optional and defaults to identity. An injected geom
gets no explicit `contype`/`conaffinity`; it inherits whatever default
class applies to its body in the source XML.

`mesh_collisions: visual` is the only recognized value. It zeroes
`contype` and `conaffinity` on every mesh geom in the source XML. Use it
with `geoms` when the source XML's only collision geometry is full meshes
and you are replacing them with named primitives.

`tests/integration/test_model_patches.py` has a synthetic worked example of every
sub-key.

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
`tests/integration/test_asimov_v1.py`'s `test_keyframe_feet_touch_the_floor` encodes
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
./run.sh check-contacts --robot <name> --preset <preset>
```

`build` compiles `robot.yaml` plus the named preset and writes
`robots/<name>/mjx/<preset>.xml`. `check` steps every keyframe in plain
MuJoCo, plus a short MJX rollout unless `--skip-mjx` is set. It fails on
NaN, or if `|qvel|` exceeds `--max-qvel` (default 100 rad/s). Both gates
must pass before the robot is usable for training.

`check-contacts` measures the per-world contact and constraint-row peaks a
new robot reaches and prints the warp budgets they need. A robot with more
foot collision geoms than the ones already here can outgrow
`task.env.sim.naconmax_per_env` / `njmax`, and warp drops the overflow
silently — see `docs/configuration.md`'s warp contact budgets section. Add
the new robot to `tests/integration/test_check_contacts.py`'s `ROBOTS` list
so the guard covers it.

## 5. Add `tests/integration/test_<name>.py`

Mirror `tests/integration/test_asimov_v1.py`. At minimum:

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
# @package _global_
robot:
  name: <name>
  dir: robots/<name>
```

The `@package _global_` header and the `robot: {name, dir}` block make
`robot=<name>` selectable on the Hydra CLI and let `train.py` find
`robots/<name>/`. That much is a pointer only.

The same file also carries the new robot's training-tuning overlay. The
overlay is a `task:` and/or `dr:` section that patches the task/dr base
configs. `robot` composes after `task` and `dr` in `configs/config.yaml`'s
defaults list, so the overlay wins over the base. `docs/configuration.md`'s
"Robot configs own robot-specific tuning" section covers the full merge
order. Decide per value whether the new robot pins its own number or
inherits the shared base:

- Pin a command envelope (`task.env.command.vx/vy/wz`) and obs-noise scales
  (`task.env.obs_noise.*`) if the new robot has its own published numbers.
  Otherwise leave them out of the overlay. The robot then inherits
  `configs/task/joystick.yaml`'s generic defaults.
- Pin DR ranges (`dr.dof.*`, `dr.joint_gains.*`, `dr.com_offset.*`, and so
  on) only for the sub-fields with a real robot-specific source, such as a
  hardware spec or an upstream sim-to-real training config. Leave the rest
  out. The rest inherits `configs/dr/default.yaml`'s ranges. Hydra merges
  nested `dr` dicts recursively, so the overlay only needs to name the keys
  it actually changes.
- Pin reward weights (`task.env.reward.scales.*`) only where a source
  genuinely maps onto one of this repo's reward terms. Check what each term
  actually computes in `src/humanoid_lab/rewards/terms.py`. A matching name
  does not guarantee matching math. Pin only the changed entries.
  `configs/task/joystick.yaml` has no `reward` key of its own, so the
  overlay's `reward:` section is the only reward content in the resolved
  config, and `make_env` deep-merges it onto `envs/joystick.py`'s
  `default_config()` entry by entry at env construction
  (`registry._apply_overrides`). Every unlisted entry keeps its Python
  default. `configs/robot/roboto_origin.yaml`'s `reward:` section is the
  worked example. It also records upstream values with no matching term in
  a "not ported" comment block, instead of guessing a mapping.
- `configs/robot/asimov_v1.yaml` is the worked example for the plain case:
  a command envelope and obs-noise scales, no DR or reward overlay.
  `configs/robot/roboto_origin.yaml` is the worked example for the fuller
  case.

A CLI override, such as `task.env.obs_noise.joint_vel=...` or
`dr.dof.armature=...`, still wins over anything the overlay pins. Hydra
applies command-line overrides after the whole defaults list composes.

## Ops rules that apply

- Never hand-edit a generated XML under `robots/<name>/mjx/`. It is build
  output. Edit `robot.yaml` or the actuator preset, then rerun `run.sh build`.
- Resolve the config before spending GPU time. Run `./run.sh train
  robot=<name> ... --cfg job --resolve` first, every time.
- Smoke before a bounded train: `./run.sh smoke robot=<name> ...` on CPU
  before any real GPU run.
