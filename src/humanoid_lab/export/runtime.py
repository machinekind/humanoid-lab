"""The numpy deploy runtime: the reference a robot side vendors.

This module runs an exported policy with numpy and nothing else. No jax,
no brax, no mujoco, no humanoid_lab import. A robot-side codebase copies
this file next to `policy.npz` and `policy_meta.json` and gets the same
ctrl the training env computed.

The loop, per control step:

    obs   = concat(obs_layout components, in order)
    act   = tanh(mlp(normalize(obs))[:action_size])
    ctrl  = clip(anchor_ctrl + act * action_scale, ctrl_low, ctrl_high)

`DeployPolicy` holds the two pieces of state the observation contains:
the previous action and the gait clock. Both advance once per `act` call,
after the observation is assembled, which is the order `envs/joystick.py`
step() uses.

The network is the actor half of a brax PPO checkpoint: a SiLU MLP whose
last layer is a (loc, scale) pair, of which deterministic inference takes
tanh(loc). `export/policy.py` validates this forward pass against the
jitted brax one before any artifact is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# The contract schema this runtime interprets.
SUPPORTED_SCHEMA = 1

ARTIFACT_WEIGHTS = "policy.npz"
ARTIFACT_META = "policy_meta.json"

# Observation components with a source on the robot. The IMU gives gyro and
# gravity, the encoders give joint_pos and joint_vel, the operator gives
# command, and this runtime holds last_action and phase itself.
SENSOR_OBS = ("gyro", "gravity", "joint_pos", "joint_vel")
KNOWN_OBS = frozenset(SENSOR_OBS + ("command", "last_action", "phase"))


def _wrap(phase):
    """Fold an angle into [-pi, pi), the env's own wrap."""
    return np.fmod(phase + np.pi, 2 * np.pi) - np.pi


def forward(weights, obs, action_size: int):
    """Deterministic action from the exported actor weights.

    `weights` holds `norm_mean`/`norm_std` and the `hidden_<i>_kernel` /
    `hidden_<i>_bias` pairs in layer order. Every layer but the last is
    followed by SiLU, written in its overflow-free form.
    """
    x = (np.asarray(obs, np.float32) - weights["norm_mean"]) / weights["norm_std"]
    i = 0
    while f"hidden_{i}_kernel" in weights:
        x = x @ weights[f"hidden_{i}_kernel"] + weights[f"hidden_{i}_bias"]
        if f"hidden_{i + 1}_kernel" in weights:
            x = x * np.exp(-np.logaddexp(0.0, -x))
        i += 1
    if x.shape[-1] != 2 * action_size:
        raise ValueError(
            f"network head is {x.shape[-1]} wide for {action_size} actions; this "
            "runtime reads a (loc, scale) pair, which is what brax's tanh_normal "
            "distribution writes"
        )
    return np.tanh(x[:action_size])


class DeployPolicy:
    """An exported policy, ready to run."""

    def __init__(self, weights: dict, meta: dict):
        schema = int(meta.get("schema_version", 0))
        if schema != SUPPORTED_SCHEMA:
            raise ValueError(
                f"policy_meta.json is schema {schema}; this runtime reads schema "
                f"{SUPPORTED_SCHEMA}"
            )

        self.meta = meta
        self.weights = {k: np.asarray(v, np.float32) for k, v in weights.items()}

        self._layout = [(c["name"], int(c["size"])) for c in meta["obs_layout"]]
        unknown = [name for name, _ in self._layout if name not in KNOWN_OBS]
        if unknown:
            raise ValueError(
                f"observation component(s) {unknown} have no source on the robot; "
                f"this runtime produces {sorted(KNOWN_OBS)}"
            )
        self._obs_size = int(meta["obs_size"])
        width = sum(size for _, size in self._layout)
        if width != self._obs_size or self.weights["norm_mean"].size != self._obs_size:
            raise ValueError(
                f"obs_layout sums to {width}, obs_size says {self._obs_size} and the "
                f"normalizer holds {self.weights['norm_mean'].size} -- the artifacts "
                "do not describe one policy"
            )

        self.action_size = int(meta["action_size"])
        self._anchor = np.asarray(meta["anchor_ctrl"], np.float32)
        self._default_pose = np.asarray(meta["default_pose"], np.float32)
        self._scale = np.asarray(meta["action_scale"], np.float32)
        self._ctrl_low = np.asarray(meta["ctrl_low"], np.float32)
        self._ctrl_high = np.asarray(meta["ctrl_high"], np.float32)

        clock = meta["gait_clock"]
        # Already folded into [-pi, pi) by the env that exported them. Only
        # cos and sin of each leg phase is observed, so the fold is invisible.
        self._offsets = np.asarray(clock["offsets"], np.float32)
        self._freq_low = float(clock["freq_low"])
        self._freq_high = float(clock["freq_high"])
        self._turn_weight = float(clock["turn_weight"])
        self._speed_deadband = float(clock["speed_deadband"])
        self._cmd_speed_max = float(clock["cmd_speed_max"])
        self._ctrl_dt = float(meta["ctrl_dt"])

        self.reset()

    @classmethod
    def load(cls, out_dir) -> "DeployPolicy":
        """Load the two artifacts written by `run.sh export`."""
        out_dir = Path(out_dir)
        with np.load(out_dir / ARTIFACT_WEIGHTS) as archive:
            weights = {key: archive[key] for key in archive.files}
        meta = json.loads((out_dir / ARTIFACT_META).read_text())
        return cls(weights, meta)

    def reset(self) -> None:
        """Clear the two pieces of observation state."""
        self.last_action = np.zeros(self.action_size, np.float32)
        self.phase = 0.0

    # -- observation ---------------------------------------------------------
    def leg_phases(self):
        return _wrap(self.phase + self._offsets)

    def observe(self, *, gyro, gravity, joint_pos, joint_vel, command):
        """The actor observation, in `obs_layout` order.

        `joint_pos` is the raw encoder reading; the default pose is
        subtracted here, as the env's observation catalog does it.
        """
        legs = self.leg_phases()
        parts = {
            "gyro": np.asarray(gyro),
            "gravity": np.asarray(gravity),
            "joint_pos": np.asarray(joint_pos) - self._default_pose,
            "joint_vel": np.asarray(joint_vel),
            "last_action": self.last_action,
            "command": np.asarray(command),
            "phase": np.concatenate([np.cos(legs), np.sin(legs)]),
        }
        obs = np.concatenate([parts[name] for name, _ in self._layout]).astype(np.float32)
        if obs.size != self._obs_size:
            raise ValueError(
                f"assembled a {obs.size}-wide observation, the policy reads "
                f"{self._obs_size} -- a sensor reading has the wrong width"
            )
        return obs

    # -- one control step ----------------------------------------------------
    def command_speed(self, command) -> float:
        """The planar speed the gait clock serves; turning counts too."""
        command = np.asarray(command, np.float64)
        return float(np.linalg.norm(command[:2]) + self._turn_weight * abs(command[2]))

    def phase_increment(self, command) -> float:
        """Clock increment for one control step; frozen when told to stand."""
        speed = self.command_speed(command)
        if speed <= self._speed_deadband:
            return 0.0
        fraction = min(speed / self._cmd_speed_max, 1.0)
        freq = self._freq_low + (self._freq_high - self._freq_low) * fraction
        return 2 * np.pi * self._ctrl_dt * freq

    def act(self, *, gyro, gravity, joint_pos, joint_vel, command):
        """The policy's action, and one step of the runtime's own state."""
        obs = self.observe(
            gyro=gyro,
            gravity=gravity,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            command=command,
        )
        action = forward(self.weights, obs, self.action_size)
        self.last_action = action
        self.phase = float(_wrap(self.phase + self.phase_increment(command)))
        return action

    def ctrl_from_action(self, action):
        """The ctrl the actuators receive: anchor, scale, clip."""
        action = np.asarray(action, np.float32)
        return np.clip(self._anchor + action * self._scale, self._ctrl_low, self._ctrl_high)

    def step(self, *, gyro, gravity, joint_pos, joint_vel, command):
        """One control step: sensors and command in, ctrl out."""
        return self.ctrl_from_action(
            self.act(
                gyro=gyro,
                gravity=gravity,
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                command=command,
            )
        )
