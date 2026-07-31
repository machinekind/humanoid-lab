"""Export a Brax PPO checkpoint to `policy.npz` + `policy_meta.json`.

    ./run.sh export --run runs/<name> [--out DIR]

The deploy side runs the policy with numpy only, so this writes plain
arrays and a JSON contract:

    policy.npz        norm_mean/norm_std and the hidden_<i> kernel/bias pairs
    policy_meta.json  the deployment contract (see deploy_contract.py)

Two independent validations run before either artifact reaches its
destination:

1. The numpy forward pass in `export/runtime.py` against the jitted brax
   inference function, on random observations.
2. The deploy runtime, loaded from the artifacts themselves, against a
   reference pipeline computed from the env's own resolved fields. The
   observation assembly order, the gait clock, the anchor, the scale and
   the clip bounds all round-trip through the written files.

Both compare at `TOLERANCE`, 1e-4. On a joint target in radians that is
0.006 degrees, three orders below the 0.01 rad encoder noise the policy
trains under. The residual is float32 reassociation between JAX and
numpy: w01-tek measured 3.9e-5 on a policy with large weights, and the
measured numbers here are printed by every export.

w01-tek's exporter writes both artifacts first and validates afterwards,
so a failed validation leaves bad files on disk while its docstring
claims the opposite. This one validates against a temp directory and
moves the artifacts into place only after both checks pass. A failed
export leaves the destination as it found it, including absent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_lab.export import runtime

# Radians of joint target, or Nm under an ideal-torque preset.
TOLERANCE = 1e-4
# Random observations per validation. w01-tek's number.
SAMPLES = 32

# What the numpy runtime implements. A checkpoint trained with anything
# else fails validation one anyway; refusing here says why.
SUPPORTED_ACTIVATION = "swish"
SUPPORTED_DISTRIBUTION = "tanh_normal"


@dataclass
class Loaded:
    """Everything an export reads, resolved once."""

    run: dict
    env: Any
    checkpoint: Path
    inference: Any
    weights: dict
    meta: dict
    privileged_size: int


def check_network_supported(run: dict) -> None:
    """Refuse a checkpoint whose actor the numpy runtime cannot reproduce."""
    factory = (run.get("ppo_config") or {}).get("network_factory") or {}
    activation = factory.get("activation", SUPPORTED_ACTIVATION)
    if activation != SUPPORTED_ACTIVATION:
        raise NotImplementedError(
            f"this run trained a '{activation}' actor; export/runtime.py implements "
            f"{SUPPORTED_ACTIVATION} only"
        )
    distribution = factory.get("distribution_type", SUPPORTED_DISTRIBUTION)
    if distribution != SUPPORTED_DISTRIBUTION:
        raise NotImplementedError(
            f"this run trained a '{distribution}' action distribution; "
            f"export/runtime.py reads the (loc, scale) head {SUPPORTED_DISTRIBUTION} "
            "writes"
        )


def flatten_params(params, obs_key: str = "state") -> dict:
    """The actor's normalizer and MLP layers as plain float32 arrays.

    `params` is brax's (normalizer, policy, value) tuple. The value network
    stays behind: the critic reads privileged observations the robot does
    not have.
    """
    normalizer, policy_params = params[0], params[1]
    mean, std = normalizer.mean, normalizer.std
    if hasattr(mean, "keys"):
        if obs_key not in mean:
            raise KeyError(
                f"the checkpoint's normalizer has no '{obs_key}' entry (has "
                f"{sorted(mean)}) -- ppo_config's policy_obs_key and the env's "
                "observation dict disagree"
            )
        mean, std = mean[obs_key], std[obs_key]

    weights = {
        "norm_mean": np.asarray(mean, np.float32),
        "norm_std": np.asarray(std, np.float32),
    }
    layers = policy_params["params"]
    for i, name in enumerate(sorted(layers, key=lambda n: int(n.split("_")[-1]))):
        weights[f"hidden_{i}_kernel"] = np.asarray(layers[name]["kernel"], np.float32)
        weights[f"hidden_{i}_bias"] = np.asarray(layers[name]["bias"], np.float32)
    return weights


def load_for_export(run_dir) -> Loaded:
    """Env, contract, weights and inference function for one run.

    The env is the run's own, rebuilt through eval/battery.py's loader so
    the robot, preset, actuator overrides and PPO network shape all come
    from run.json. Its measurement overrides (pushes off, mirror off, the
    no-progress cut off, the command resampler idle) touch training-only
    keys, so no contract field moves.
    """
    from humanoid_lab import policy_io
    from humanoid_lab.deploy_contract import build_contract, check_config_covered
    from humanoid_lab.eval import battery

    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    # Fail on an unclassified env key before building anything.
    check_config_covered(run.get("env_config") or {})
    check_network_supported(run)

    # The export host need not have a GPU or the warp backend: sim.* is
    # training-only, so resolving the backend here changes nothing the
    # contract or the network reads.
    run, env, checkpoint, inference = battery.load_checkpoint_policy(
        run_dir, extra_env_overrides={"sim": {"backend": "auto", "num_envs": 1}}
    )
    meta = build_contract(env, run, checkpoint)

    obs_key = ((run.get("ppo_config") or {}).get("network_factory") or {}).get(
        "policy_obs_key", "state"
    )
    weights = flatten_params(policy_io.load_params(checkpoint), obs_key)
    if weights["norm_mean"].size != meta["obs_size"]:
        raise ValueError(
            f"the checkpoint's actor reads {weights['norm_mean'].size} observations "
            f"and this env config resolves to {meta['obs_size']} "
            f"({[c['name'] for c in meta['obs_layout']]}) -- run.json and the "
            "checkpoint describe different policies"
        )

    privileged = env.observation_size["privileged_state"]
    privileged_size = int(privileged[0] if hasattr(privileged, "__len__") else privileged)
    return Loaded(run, env, Path(checkpoint), inference, weights, meta, privileged_size)


def validate_numpy_vs_brax(weights, meta, inference, privileged_size, samples: int = SAMPLES):
    """Max |difference| between the numpy forward pass and brax's."""
    import jax

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    blind = np.zeros(privileged_size, np.float32)
    worst = 0.0
    for _ in range(samples):
        obs = rng.uniform(-2.0, 2.0, meta["obs_size"]).astype(np.float32)
        reference, _ = inference({"state": obs, "privileged_state": blind}, key)
        got = runtime.forward(weights, obs, meta["action_size"])
        worst = max(worst, float(np.max(np.abs(np.asarray(reference) - got))))
    return worst


def validate_runtime_vs_env(env, out_dir, inference, privileged_size, steps: int = SAMPLES):
    """Max |difference| in ctrl between the deploy runtime and the env.

    The runtime side reads the written `policy.npz` and `policy_meta.json`
    and nothing else. The reference side assembles its observation from the
    env's own catalog fields, advances the env's own gait clock, and maps
    the action through the env's own actuator model and clip bounds. Both
    see the same sensor stream, and each carries its own last action and
    clock, so any drift compounds over the run instead of cancelling.
    """
    import jax
    import jax.numpy as jp

    policy = runtime.DeployPolicy.load(out_dir)
    policy.reset()

    names = list(env.actor_obs_names)
    default_pose = np.asarray(env._default_pose)
    nu = env.action_size
    command = env._config.command
    low = np.array([command.vx[0], command.vy[0], command.wz[0]])
    high = np.array([command.vx[1], command.vy[1], command.wz[1]])

    rng = np.random.default_rng(0)
    key = jax.random.PRNGKey(0)
    blind = np.zeros(privileged_size, np.float32)
    last_action = np.zeros(nu, np.float32)
    phase = 0.0
    worst = 0.0
    for _ in range(steps):
        gyro = rng.uniform(-2.0, 2.0, 3)
        gravity = rng.uniform(-1.0, 1.0, 3)
        joint_pos = default_pose + rng.uniform(-0.4, 0.4, nu)
        joint_vel = rng.uniform(-6.0, 6.0, nu)
        cmd = rng.uniform(low, high)

        legs = np.asarray(env._leg_phases({"phase": jp.array(phase)}))
        parts = {
            "gyro": gyro,
            "gravity": gravity,
            "joint_pos": joint_pos - default_pose,
            "joint_vel": joint_vel,
            "last_action": last_action,
            "command": cmd,
            "phase": np.concatenate([np.cos(legs), np.sin(legs)]),
        }
        obs = np.concatenate([parts[name] for name in names]).astype(np.float32)
        action, _ = inference({"state": obs, "privileged_state": blind}, key)
        action = np.asarray(action, np.float32)
        reference_ctrl = np.asarray(
            jp.clip(
                env._actuator_model.ctrl_from_action(action, env._default_pose, env._action_scale),
                env._ctrl_lo,
                env._ctrl_hi,
            )
        )

        got = policy.step(
            gyro=gyro,
            gravity=gravity,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            command=cmd,
        )
        worst = max(worst, float(np.max(np.abs(reference_ctrl - got))))

        last_action = action
        phase = float(
            np.fmod(phase + float(env._phase_dt(jp.array(cmd))) + np.pi, 2 * np.pi) - np.pi
        )
    return worst


def write_validated(env, meta, weights, out_dir, inference, privileged_size) -> Path:
    """Validate, then place the artifacts. Nothing is written on a failure."""
    forward_error = validate_numpy_vs_brax(weights, meta, inference, privileged_size)
    assert forward_error < TOLERANCE, (
        f"numpy forward vs brax inference: max |diff| = {forward_error}"
    )
    print(f"validated numpy vs brax inference: max |diff| = {forward_error:.2e}")

    out_dir = Path(out_dir)
    with tempfile.TemporaryDirectory(prefix="humanoid-lab-export-") as staging:
        staging = Path(staging)
        np.savez(staging / runtime.ARTIFACT_WEIGHTS, **weights)
        (staging / runtime.ARTIFACT_META).write_text(json.dumps(meta, indent=2) + "\n")

        runtime_error = validate_runtime_vs_env(env, staging, inference, privileged_size)
        assert runtime_error < TOLERANCE, (
            f"deploy runtime vs env reference: max |diff| = {runtime_error}"
        )
        print(f"validated deploy runtime vs env reference: max |diff| = {runtime_error:.2e}")

        out_dir.mkdir(parents=True, exist_ok=True)
        # Weights first: an interrupted move leaves no meta, and the loader
        # reads both or fails.
        for name in (runtime.ARTIFACT_WEIGHTS, runtime.ARTIFACT_META):
            shutil.move(str(staging / name), str(out_dir / name))
    return out_dir


def export_run(run_dir, out_dir=None) -> Path:
    """Export one run's latest checkpoint. Returns the destination."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "deploy"
    loaded = load_for_export(run_dir)
    return write_validated(
        loaded.env,
        loaded.meta,
        loaded.weights,
        out_dir,
        loaded.inference,
        loaded.privileged_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run dir containing run.json")
    parser.add_argument("--out", default=None, help="destination dir (default <run>/deploy)")
    args = parser.parse_args()

    out_dir = export_run(args.run, args.out)
    print(f"wrote {out_dir}/{runtime.ARTIFACT_WEIGHTS} and {runtime.ARTIFACT_META}")


if __name__ == "__main__":
    main()
