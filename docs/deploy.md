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
| `schema_version` | contract schema, currently 1 |
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
| `gait_clock` | the constants of the clock the actor observes as `phase` |

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

The consumed set is small: `ctrl_dt`, `action_scale`, `reset_keyframe`,
`obs.state`, `obs.include`, the three command box ranges, and `gait.freq`.
Everything else shapes training only.

Two classifications carry an argument.

`gait.freq` is consumed because the actor observes the gait clock. This
clock is a fixed antiphase two-foot clock whose frequency is a lerp over the
commanded speed fraction, so it ships in `gait_clock` and the runtime
integrates it. A clock that blended a walk and a trot over speed would have
no faithful runtime copy, and a phase-observing policy would have to be
refused.
`gait.swing_height`, `gait.duty` and `gait.air_time_cap` feed the reward's
clearance targets, and the observation reads none of them.

The pure command draws are training-only on a condition the code checks. An
armed draw whose range leaves the trained box raises, because
`command_low`/`command_high` would then understate what the policy trained
under. Setting a fast range above the box is a known way to pull a policy
past a speed it deadlocks at; a run that wants that here widens `command.vx`
to match.

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

`step` advances the two pieces of observation state, the previous action and
the gait clock, after the observation is assembled. That is the order
`envs/joystick.py` step() uses. `reset` clears both.
