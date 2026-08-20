# humanoid-lab

**Work in progress.** Configs, reward recipes and interfaces change without notice.

humanoid-lab trains locomotion policies for open-source humanoid robots in MuJoCo MJX with Brax PPO, and exports them for deployment on the robot.

Two robots are supported. [Roboto Origin](https://github.com/Roboparty/roboto_origin) is the primary target. We plan to have the physical robot and run these policies on it. [Asimov v1](https://github.com/asimovinc/asimov-1) is the second model. Each robot lives under `robots/<name>/` as a verbatim upstream model plus a `robot.yaml` that injects actuators, keyframes and contact geometry at build time.

## What it does

- Trains a joystick velocity-tracking policy (`task=joystick`) with domain randomization and configurable reward terms.
- Measures the torque, velocity and power each joint needs (`task=sizing`), so actuators can be sized before they are bought.
- Evaluates a checkpoint with a scenario battery, a report and videos.
- Exports `policy.npz` and `policy_meta.json` for the on-robot runtime.

## Usage

Configuration is Hydra. `./run.sh <verb> [overrides]` wraps the project venv.

| verb | does |
|---|---|
| `build` / `check` | build and check a robot's MJX model |
| `train` | training run |
| `smoke` | short CPU training run, wandb off |
| `battery` / `report` / `eval` | evaluation battery, report, video |
| `sizing-collect` / `sizing-report` | actuator sizing rollout and report |
| `export` | deploy artifacts from a checkpoint |
| `test` / `test-all` | unit tests, or unit and integration tests |

Example: `./run.sh train robot=roboto_origin ppo.num_timesteps=3e8 run_name=roboto_walk_v1`

`docs/configuration.md` lists every verb and flag. `docs/adding-a-robot.md` explains the robot layout. `docs/deploy.md` states the export contract.

Remote training goes through a separate, private ops tool. `jobs/` holds the cluster-agnostic payloads it calls, and `jobs/README.md` states their contract.

## License

Apache-2.0 for this repository's code. The vendored robot models under `robots/*/source/` keep their upstream licenses. See `NOTICE`.
