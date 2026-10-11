# Deploy

What a trained policy carries with it to a robot, and what refuses to ship.

## The contract

`src/humanoid_lab/deploy_contract.py` builds `policy_meta.json` from a live
env. Every field is a resolved number read off the env instance that defined
training. The deploy side re-derives nothing. It interprets the file:

```
obs  = concat(obs_layout components, in order)
act  = tanh(mlp(normalize(obs)))
ctrl = clip(anchor_ctrl + act * action_scale, ctrl_low, ctrl_high)
```

### Fields

| field | what it is |
|---|---|
| `schema_version` | contract schema, 1 or 2 (see the frozen clock below) |
| `run_name`, `task`, `checkpoint`, `robot`, `preset`, `actuator_model` | provenance |
| `obs_layout`, `obs_size` | ordered actor components with their widths, and the total |
| `action_size`, `joint_names` | the action vector's width and its column order |
| `anchor_ctrl` | the ctrl a zero action commands |
| `default_pose` | the observation anchor: `joint_pos = qpos - default_pose` |
| `action_scale` | per-joint action-to-ctrl scale |
| `ctrl_low`, `ctrl_high`, `ctrl_unit` | ctrl clip bounds and their unit |
| `ctrl_dt` | control period in seconds |
| `torque_low`, `torque_high` | per-actuator torque envelope from the compiled model |
| `gains` | the effective per-actuator kp/kd the policy trained against |
| `command_low`, `command_high` | the trained command box, ordered (vx, vy, wz) |
| `gait_clock` | the constants of the clock the actor observes as `phase`, and `phase_obs`, the phase observation mode |

`joint_names` is the canonical actuated-joint order from `robot.yaml`. Every
per-joint vector in the table uses it.

`anchor_ctrl` and `default_pose` differ by actuator model. Under a `pd`
preset both are the reset keyframe's joint pose. Under an `ideal_torque`
preset ctrl is a torque, so `anchor_ctrl` is zero while `default_pose` stays
the keyframe pose the observation subtracts. `real_pose_ref` moves neither
(see `envs/base.py`); it relocates the reward reference and the reset pose.

`torque_high` and `gains` travel in the metadata because the driver on the
robot runs those numbers. A clamp below `torque_high` truncates the torque
peaks the policy trained with.

### The key ledger

`CONSUMED_KEYS` and `TRAINING_ONLY_KEYS` classify every leaf key of
`envs/joystick.py::default_config()`. A consumed key's value reaches the
robot, directly or through a resolved field. A training-only key cannot.
Each entry carries a comment saying which and why.

`check_config_covered(env_config)` raises on a key in neither set, and
`build_contract` calls it first. A new env option therefore blocks export
until someone classifies it. `tests/unit/test_deploy_contract.py` walks
`default_config()` and fails on the same condition, so the block lands at
commit time rather than at export time.

The consumed set is small: `ctrl_dt`, `reset_keyframe`, `obs.state`, the
three command box ranges (`command.vx`, `command.vy`, `command.wz`),
`gait.freq`, `command.deadband` and `gait.frozen_phase_obs`. Everything else
shapes training only.

Two classifications carry an argument.

`gait.freq` is consumed because the actor observes the gait clock. This
clock is a fixed antiphase two-foot clock whose frequency is a lerp over the
commanded speed fraction, so it ships in `gait_clock` and the runtime
integrates it. A clock that blended a walk and a trot over speed would have
no faithful runtime copy, and a phase-observing policy would have to be
refused.

`command.deadband` is consumed for the same reason. The clock freezes when
the commanded speed is at or below the deadband, so the runtime needs the
value. It ships resolved as `gait_clock.speed_deadband`. The env compares a
float32 speed with `>` and the runtime compares a float64 speed with `<=`.
The two agree for any command farther from the deadband than float32
rounding.

`gait.frozen_phase_obs` is consumed because it picks what the actor sees as
`phase` when `gait.freq` is (0, 0). `"cos_sin"` is the legacy observation,
cos/sin of the frozen leg phases. `"zeros"` observes zeros of the same width.
The mode ships explicitly as `gait_clock.phase_obs`, and the runtime never
infers it from `freq_low` and `freq_high`. A contract with `"cos_sin"` is
schema 1, so a stock export still loads on a robot that runs an old vendored
runtime. A contract with `"zeros"` is schema 2. The current runtime reads
schema 1 (always cos/sin) and schema 2. An old vendored runtime refuses
schema 2, so a zeros policy cannot run on it with the wrong phase. An old
checkpoint trained with a frozen clock keeps cos/sin: export rebuilds the env
on `default_config()`, where the key defaults to `"cos_sin"`.

`gait.swing_height`, `gait.duty` and `gait.air_time_cap` feed the reward's
clearance targets, and the observation reads none of them.

The pure command draws are training-only on a condition the code checks. An
armed draw whose range leaves the trained box raises, because
`command_low`/`command_high` would then understate what the policy trained
under. Setting a fast range above the box is a known way to pull a policy
past a speed it deadlocks at; a run that wants that here widens `command.vx`
to match.

The command ramp (`command.ramp.lin`, `command.ramp.yaw`) is training-only.
In training it slews every new command from the old one at a bounded rate.
The runtime feeds the operator command to the policy directly and does not
ramp it. The robot side should rate-limit operator commands, so the policy
never sees a step change sharper than the ones it trained on.

### Refusals

`build_contract` raises on four conditions:

- an env config key in neither ledger set
- an armed pure command draw outside the trained box
- a task other than `joystick`
- an actor observation with no source on the robot

The last one is the list in `DEPLOYABLE_OBS`. The IMU gives `gyro` and
`gravity`, the encoders give `joint_pos` and `joint_vel`, the operator gives
`command`, and the runtime holds `last_action` and `phase` itself. A
privileged signal on the actor list (`linvel`, `height`, `contacts`,
`actuator_force`) has no deploy-side source.

`heading_err` on the actor list is refused too. The robot would need a yaw
input in the env's convention, and runtime state for `yaw_ref`, and the
runtime has neither. In the env `yaw_ref` follows the measured yaw while the
commanded turn rate exceeds `progress.TURN_WZ_THRESHOLD` (0.05 rad/s) and
holds otherwise. `heading_err` also trains without noise, while a real IMU
yaw drifts. `heading_err` on the critic list only is fine, because the
critic does not ship.

## The export

```bash
./run.sh export --run runs/<name> [--out DIR]
```

The verb writes two files, by default into `runs/<name>/deploy/`:

- `policy.npz` holds `norm_mean`, `norm_std` and the `hidden_<i>_kernel` and
  `hidden_<i>_bias` pairs of the actor MLP, as float32 numpy arrays.
- `policy_meta.json` holds the contract above.

The filenames are inherited. PLAN.md records that keeper policies publish
to a private HF repo under <hf-org> with the established flat layout and names
no artifact files, so nothing downstream pins them. The value network stays
behind: the critic reads privileged observations the robot does not have.

The export runs on CPU. It builds the run's env through `eval/battery.py`'s
loader, so the robot, preset, actuator overrides and network shape all come
from `run.json`, and it steps no physics.

The export reads `run.json` through `run_record.load_run`. A legacy Torch run
is checked against its migrated `hydra_config.task.env`. A legacy run that
sets a deleted reward term, or freezes the gait clock without
`gait.frozen_phase_obs`, is refused.

### Both validations run before either file is placed

`export/policy.py` stages the artifacts in a temp directory, validates, and
moves them into the destination afterwards. A failed export leaves the
destination as it found it, including absent, and an earlier good export
survives a later failed one.

Validation one compares the numpy forward pass in `export/runtime.py`
against the jitted brax inference function over 32 random observations.

Validation two loads `DeployPolicy` from the staged artifacts and runs it
against a reference pipeline computed from the env's own resolved fields,
over 32 steps of a random sensor stream. The observation assembly order, the
gait clock, the anchor, the scale and the clip bounds all round-trip. Each
side carries its own last action and its own clock, so a drift compounds
over the run instead of cancelling.

The run starts with scripted commands, each held for 3 steps: standing
still, and a speed and a turn rate just below and just above the deadband.
A wrong `speed_deadband` in the written meta makes the two clocks diverge
there. A scripted command outside the command box is skipped. The phase
observation on the env side comes from the env's own observation catalog.

Both bound the error at 1e-4 and raise `ExportValidationError` above it,
carrying the measured error and the bound. The check is an explicit raise
rather than an assert on purpose: `python -O` strips asserts, and a stripped
check would ship unvalidated artifacts. On a joint target in radians 1e-4 is
0.006 degrees, three orders below the 0.01 rad encoder noise the policy
trains under. The residual is float32 reassociation between JAX and numpy;
a policy with large weights has measured 3.9e-5. Measured here on a roboto
smoke run with a 512-256-128 actor: 4.9e-06 for validation one and 8.1e-07
for validation two. Every export prints its own numbers.

### The runtime

`src/humanoid_lab/export/runtime.py` imports numpy and nothing else. A
robot-side codebase copies the file next to the two artifacts.

```python
from runtime import DeployPolicy

policy = DeployPolicy.load("deploy/")
policy.reset()
ctrl = policy.step(gyro=..., gravity=..., joint_pos=..., joint_vel=..., command=...)
```

`joint_pos` is the raw encoder reading in `joint_names` order; the runtime
subtracts `default_pose` itself. `command` is (vx, vy, wz). The return value
is a ctrl vector in `ctrl_unit`, already clipped.

The phase observation is cos/sin of the leg phases, or zeros when the
contract's `gait_clock.phase_obs` is `"zeros"`. The runtime reads schema 1
and schema 2 and refuses any other. A schema 2 contract without
`gait_clock.phase_obs` is refused.

`step` advances the two pieces of observation state, the previous action and
the gait clock, after the observation is assembled. That is the order
`envs/joystick.py` step() uses. `reset` clears both.
