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

`gait.freq` is consumed because the actor observes the gait clock. w01-tek
refuses to export a phase-observing policy: its clock blends a walk and a
trot over speed, and its runtime has no faithful copy. Ours is a fixed
antiphase two-foot clock whose frequency is a lerp over the commanded speed
fraction, so it ships in `gait_clock` and the runtime integrates it.
`gait.swing_height`, `gait.duty` and `gait.air_time_cap` feed the reward's
clearance targets, and the observation reads none of them.

The pure command draws are training-only on a condition the code checks. An
armed draw whose range leaves the trained box raises, because
`command_low`/`command_high` would then understate what the policy trained
under. w01-tek set its own fast range above its box deliberately, to pull a
policy past a speed it deadlocked at. A run that wants that here widens
`command.vx` to match.

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
