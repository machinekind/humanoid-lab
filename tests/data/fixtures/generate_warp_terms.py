"""Record what the PR-head Torch runner computes for the promoted recipe.

The shared modules port four reward terms (step_length, foot_reach,
knee_extend, forward_lean), the land_frac record, the speed-scaled lift, the
command ramp and the straight command draw from wtrain/wenv.py. This script
runs that PR-head code itself on recorded inputs and writes the inputs and its
outputs to warp_terms.npz. tests/unit/test_warp_terms_port.py feeds the same
inputs to the shared functions and compares. The comparison holds the shared
port to the code that trained v40.

Run it from the repo root:

    JAX_PLATFORMS=cpu .venv/bin/python tests/data/fixtures/generate_warp_terms.py

It needs torch (CPU is enough) and warp-lang, because wenv.py imports both at
module level. wtrain/wenv.py must equal its PR-head version (SOURCE_SHA). The
script checks with `git diff --quiet SOURCE_SHA -- wtrain/wenv.py` and refuses
otherwise. No warp physics runs and no graph is captured: a stub env built
with object.__new__ carries only the attributes the recorded code paths read,
and the signal methods return the recorded inputs.

It is not a `test_*.py` file, so pytest never collects it.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "warp_terms.npz"
SOURCE_SHA = "d8f56f7"
WENV = "wtrain/wenv.py"

# The stub's shape. It matches tests/unit/task_toys (nu=4, qadr 7..10,
# vadr 6..9), so the unit test can reuse toy_consts.
N = 64
N_FEET = 2
NU = 4
QADR = list(range(7, 7 + NU))
VADR = list(range(6, 6 + NU))
KNEE_QIDX = [1, 3]
DT = 0.02
CMD_VMAX = 1.5
SEED = 7

# Forward commands at and around the deadbands and the 0.1 fwd floor.
EDGE_VX = [0.0, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.16, -0.3, 1.2]

TERM_CASES = {
    "clock_pair": {
        "extra": {
            "step_length": {"scale": 1.0, "pair_min": True},
            "foot_reach": {"scale": 1.0, "target_a": 0.05, "target_b": 0.15, "pair_min": True},
            "knee_extend": {"scale": 1.0, "flex_lo": 0.2, "flex_hi": 1.0, "td_mult": 0.5, "pair_min": True},
            "forward_lean": {"scale": 1.0, "max_deg": 0.0, "vx_max": 1.0, "sigma": 0.01},
        },
        "run": {},
        "scales": {},
        "deadband": 0.15,
        "freq": [0.8, 1.4],
    },
    "linear_plain": {
        "extra": {
            "step_length": {"scale": 1.0, "target_a": 0.1, "target_b": 0.2, "power": 2.0},
            "foot_reach": {"scale": 1.0, "target_a": 0.0, "target_b": 0.1, "x_min": 0.01, "power": 1.0},
            "knee_extend": {"scale": 1.0, "flex_lo": 0.1, "flex_hi": 0.8, "x_split": 0.05, "td_min_air": 0.1},
            "forward_lean": {"scale": 1.0, "max_deg": 5.0, "vx_max": 0.8, "sigma": 0.05},
        },
        "run": {},
        "scales": {},
        "deadband": 0.05,
        "freq": [1.0, 2.0],
    },
    "lift": {
        "extra": {},
        "run": {"warp_lift": {"v_full": 0.5, "min_frac": 0.25}},
        "scales": {"feet_phase": 1.0, "feet_apex": 1.0},
        "deadband": 0.05,
        "freq": [1.0, 2.0],
    },
}

STRAIGHT = {
    "seed": 11,
    "run": {"warp_cmd_straight": 0.4},
    "command": {"pure_slow_prob": 0.3, "zero_prob": 0.1},
}

RAMP = {
    "seed": 13,
    "steps": 10,
    "run": {"warp_cmd_ramp": {"lin": 0.25, "yaw": 0.6}},
    "command": {"resample_steps": 3},
}


def check_source():
    sha = subprocess.run(
        ["git", "rev-parse", f"{SOURCE_SHA}^{{commit}}"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    if subprocess.run(["git", "diff", "--quiet", sha, "--", WENV], cwd=REPO, check=False).returncode != 0:
        sys.exit(f"{WENV} differs from {SOURCE_SHA}. This script records the PR-head code only.")
    blob = subprocess.run(
        ["git", "rev-parse", f"{sha}:{WENV}"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    return sha, blob


def quat_from_euler(roll, pitch, yaw):
    """wxyz unit quaternions, ZYX order, float64."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    q = np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        -1,
    )
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rotate(q, v):
    """Rotate v (..., 3) by unit quaternion q (..., 4) wxyz, float64."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def term_inputs(rng):
    """Seeded float32 inputs for one term case, as numpy arrays."""

    def u(lo, hi, *shape):
        return rng.uniform(lo, hi, (N,) + shape)

    quat = quat_from_euler(u(-0.15, 0.15), u(-0.15, 0.15), u(-math.pi, math.pi))
    base_pos = np.stack([u(-1.0, 1.0), u(-1.0, 1.0), u(0.6, 0.8)], -1)
    side = np.array([1.0, -1.0])
    offset = np.stack([u(-0.3, 0.4, N_FEET), side * u(0.05, 0.15, N_FEET), u(-0.75, -0.6, N_FEET)], -1)
    site_xpos = base_pos[:, None, :] + rotate(quat[:, None, :], offset)

    command = np.stack([u(-0.3, 1.2), u(-0.3, 0.3), u(-0.6, 0.6)], -1)
    command[: len(EDGE_VX), 0] = EDGE_VX
    command[len(EDGE_VX) : len(EDGE_VX) + 6, 1:] = 0.0  # a few clean straight commands

    joints = u(-0.3, 0.3, NU)
    joints[:, KNEE_QIDX] = u(-0.2, 1.4, len(KNEE_QIDX))
    qpos = np.concatenate([base_pos, quat, joints], -1)

    contact = rng.uniform(size=(N, N_FEET)) < 0.5
    first_contact = contact & (rng.uniform(size=(N, N_FEET)) < 0.6)
    out = {
        "quat": quat,
        "base_pos": base_pos,
        "site_xpos": site_xpos,
        "gyro": u(-1.0, 1.0, 3),
        "linvel": u(-1.0, 1.0, 3),
        "qpos": qpos,
        "qvel": u(-1.0, 1.0, 6 + NU),
        "actuator_force": u(-40.0, 40.0, NU),
        "command": command,
        "phase": u(-math.pi, math.pi),
        "feet_air_time": u(0.0, 0.5, N_FEET),
        "land_frac": u(0.0, 1.0, N_FEET, 3),
        "swing_apex": u(0.0, 0.12, N_FEET),
        "foot_clearance": u(0.0, 0.12, N_FEET),
        "foot_linvel": u(-1.0, 1.0, N_FEET, 3),
        "action": u(-1.0, 1.0, NU),
        "last_action": u(-1.0, 1.0, NU),
        "last_last_action": u(-1.0, 1.0, NU),
        "last_torque": u(-40.0, 40.0, NU),
    }
    out = {k: v.astype(np.float32) for k, v in out.items()}
    out["contact"] = contact
    out["first_contact"] = first_contact
    return out


def base_config(freq, command=None):
    from humanoid_lab.envs import joystick

    cfg = joystick.default_config()
    cfg.gait.freq = tuple(freq)
    for k, v in (command or {}).items():
        cfg.command[k] = v
    return cfg


def make_stub(wenv, torch, cfg, run, extra, scales, deadband):
    """A WarpJoystick with only the attributes the recorded paths read."""
    w = object.__new__(wenv.WarpJoystick)
    w.num_envs = N
    w.n_feet = N_FEET
    w.nu = NU
    w.dev = torch.device("cpu")
    w.dt = DT
    w.deadband = deadband
    w.cmd_vmax = CMD_VMAX
    w.cfg = cfg
    w.run = run
    w.extra = extra
    w.scales = {k: 0.0 for k in wenv._TERMS}
    w.scales.update(scales)
    w.scales.update({k: float(v["scale"]) for k, v in extra.items()})
    w.terms = [k for k in wenv._TERMS if w.scales.get(k, 0.0)] + list(extra)
    w.qadr = torch.tensor(QADR, dtype=torch.long)
    w.vadr = torch.tensor(VADR, dtype=torch.long)
    w.base_qadr = 0
    w.base_vadr = 0
    w.foot_site_ids = torch.tensor([0, 1], dtype=torch.long)
    w.knee_qidx = torch.tensor(KNEE_QIDX, dtype=torch.long)
    w._phase_off = torch.tensor(wenv._PHASE_OFFSETS)
    w.gen = torch.Generator(device="cpu")
    return w


def record_terms(wenv, torch, name, case):
    rng = np.random.default_rng(SEED + sorted(TERM_CASES).index(name))
    x = term_inputs(rng)
    t_ = {k: torch.as_tensor(v) for k, v in x.items()}
    cfg = base_config(case["freq"])
    w = make_stub(wenv, torch, cfg, case["run"], case["extra"], case["scales"], case["deadband"])
    w.qpos = t_["qpos"]
    w.qvel = t_["qvel"]
    w.actuator_force = t_["actuator_force"]
    w.site_xpos = t_["site_xpos"]
    down = torch.tensor([0.0, 0.0, -1.0]).expand(N, 3)
    gravity = wenv.quat_rotate_inv(t_["quat"], down)
    x["gravity"] = gravity.numpy()
    w.quat = lambda: t_["quat"]
    w.gyro = lambda: t_["gyro"]
    w.local_linvel = lambda: t_["linvel"]
    w.gravity_body = lambda: gravity
    w.foot_clearance = lambda: t_["foot_clearance"]
    w.foot_linvel = lambda: t_["foot_linvel"]
    info = {
        "command": t_["command"],
        "phase": t_["phase"],
        "feet_air_time": t_["feet_air_time"],
        "land_frac": t_["land_frac"],
        "swing_apex": t_["swing_apex"],
        "last_action": t_["last_action"],
        "last_last_action": t_["last_last_action"],
        "last_torque": t_["last_torque"],
    }
    t, fall = wenv.WarpJoystick.compute_rewards(w, info, t_["action"], t_["first_contact"], t_["contact"])
    out = {f"in_{k}": v for k, v in x.items()}
    for k, v in t.items():
        out[f"t_{k}"] = v.numpy()
    out["land_frac_now"] = w._land_frac_now.numpy()
    out["fall"] = fall.numpy()
    config = {
        "extra": case["extra"],
        "run": case["run"],
        "scales": case["scales"],
        "deadband": case["deadband"],
        "freq": case["freq"],
        "cmd_vmax": CMD_VMAX,
        "knee_qidx": KNEE_QIDX,
        "dt": DT,
        "terms": sorted(t),
    }
    return out, config


def record_straight(wenv, torch):
    cfg = base_config([1.0, 2.0], STRAIGHT["command"])
    w = make_stub(wenv, torch, cfg, STRAIGHT["run"], {}, {}, 0.05)
    w.gen.manual_seed(STRAIGHT["seed"])
    got = wenv.WarpJoystick.sample_command(w, N)

    # The raw draws, from a second generator in wenv's draw order: vx, vy, wz
    # (each through _u), then straight, slow, slow vx (_u), zero.
    c = cfg.command
    g = torch.Generator(device="cpu")
    g.manual_seed(STRAIGHT["seed"])

    def rand():
        return torch.rand(N, generator=g)

    def uniform(lo, hi):
        return lo + (hi - lo) * rand()

    vx, vy, wz = uniform(*c.vx), uniform(*c.vy), uniform(*c.wz)
    straight_on = rand() < float(STRAIGHT["run"]["warp_cmd_straight"])
    slow_on = rand() < float(c.pure_slow_prob)
    slow_vx = uniform(*c.slow_vx)
    zero_on = rand() < c.zero_prob

    # Rebuild wenv's output from the draws: straight, then slow, then zero.
    vel = torch.stack([vx, vy, wz], -1)
    vel = torch.where(straight_on[:, None], vel * torch.tensor([1.0, 0.0, 0.0]), vel)
    vel = torch.where(slow_on[:, None], torch.stack([slow_vx, torch.zeros_like(slow_vx), torch.zeros_like(slow_vx)], -1), vel)
    vel = torch.where(zero_on[:, None], torch.zeros_like(vel), vel)
    assert torch.equal(vel, got) and torch.equal(torch.signbit(vel), torch.signbit(got)), "draw reconstruction drifted"
    for name, on in (("straight", straight_on), ("pure_slow", slow_on), ("zero", zero_on)):
        assert 0 < int(on.sum()) < N, f"{name} draw is all or nothing at this seed"

    out = {
        "vx": vx.numpy(),
        "vy": vy.numpy(),
        "wz": wz.numpy(),
        "straight_on": straight_on.numpy(),
        "pure_slow_on": slow_on.numpy(),
        "pure_slow_vx": slow_vx.numpy(),
        "zero_on": zero_on.numpy(),
        "command": got.numpy(),
    }
    config = {
        "seed": STRAIGHT["seed"],
        "run": STRAIGHT["run"],
        "command": {
            "straight_prob": STRAIGHT["run"]["warp_cmd_straight"],
            "pure_slow_prob": float(c.pure_slow_prob),
            "zero_prob": float(c.zero_prob),
            "slow_vx": list(c.slow_vx),
        },
    }
    return out, config


def record_ramp(wenv, torch):
    """Run wenv's own reset and step on the stub, with physics and obs stubbed."""
    cfg = base_config([1.0, 2.0], RAMP["command"])
    cfg.push.interval_steps_range = (50, 100)
    run = {**RAMP["run"], "ppo_config": {"episode_length": 1000}}
    w = make_stub(wenv, torch, cfg, run, {}, {"termination": -1.0}, 0.05)
    w.gen.manual_seed(RAMP["seed"])
    rng = np.random.default_rng(RAMP["seed"])

    def f32(a):
        return torch.as_tensor(np.asarray(a, np.float32))

    reset_qpos = np.zeros(7 + NU, np.float32)
    reset_qpos[2], reset_qpos[3] = 0.75, 1.0
    w.reset_qpos = f32(reset_qpos)
    w.qpos = torch.zeros(N, 7 + NU)
    w.qvel = torch.zeros(N, 6 + NU)
    w.ctrl = torch.zeros(N, NU)
    w.qacc_warmstart = torch.zeros(N, 6 + NU)
    w.time = torch.zeros(N)
    w.actuator_force = f32(rng.uniform(-10.0, 10.0, (N, NU)))
    w.site_xpos = torch.zeros(N, N_FEET, 3)
    w.default_pose = torch.zeros(NU)
    w.action_scale = torch.full((NU,), 0.25)
    w.ctrl_lo = torch.full((NU,), -2.0)
    w.ctrl_hi = torch.full((NU,), 2.0)
    w.neutral_ctrl = torch.zeros(NU)
    w.heading_obs = False
    w._ang_hi = torch.zeros(3)
    contact = torch.as_tensor(rng.uniform(size=(N, N_FEET)) < 0.5)
    zeros_obs = torch.zeros(N, 1)
    w._forward = lambda: None
    w._physics = lambda: None
    w.build_obs = lambda info, noise=True: (zeros_obs, zeros_obs)
    w.base_yaw = lambda: torch.zeros(N)
    w.foot_contact = lambda: contact
    w.foot_clearance = lambda: torch.zeros(N, N_FEET)
    w.quat = lambda: torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(N, 4)
    w.gravity_body = lambda: torch.tensor([0.0, 0.0, -1.0]).expand(N, 3)
    w.gyro = lambda: torch.zeros(N, 3)
    w.local_linvel = lambda: torch.zeros(N, 3)

    wenv.WarpJoystick.reset(w)
    reset_command = w.info["command"].clone()
    reset_target = w.info["cmd_target"].clone()
    before, target, after, resampled = [], [], [], []
    for _ in range(RAMP["steps"]):
        before.append(w.info["command"].clone())
        since = w.info["steps_since_cmd"].clone()
        wenv.WarpJoystick.step(w, torch.zeros(N, NU))
        resampled.append((since + 1 >= cfg.command.resample_steps).numpy())
        target.append(w.info["cmd_target"].clone())
        after.append(w.info["command"].clone())
    out = {
        "reset_command": reset_command.numpy(),
        "reset_target": reset_target.numpy(),
        "command_before": torch.stack(before).numpy(),
        "cmd_target": torch.stack(target).numpy(),
        "command_after": torch.stack(after).numpy(),
        "resampled": np.stack(resampled),
    }
    config = {
        "seed": RAMP["seed"],
        "steps": RAMP["steps"],
        "run": RAMP["run"],
        "ramp": RAMP["run"]["warp_cmd_ramp"],
        "resample_steps": RAMP["command"]["resample_steps"],
        "dt": DT,
    }
    return out, config


def main():
    sha, blob = check_source()
    import torch

    sys.path.insert(0, str(REPO / "wtrain"))
    import wenv

    torch.manual_seed(0)
    arrays, configs = {}, {}
    for name, case in TERM_CASES.items():
        out, config = record_terms(wenv, torch, name, case)
        arrays.update({f"{name}/{k}": v for k, v in out.items()})
        configs[name] = config
    for name, fn in (("straight", record_straight), ("ramp", record_ramp)):
        out, config = fn(wenv, torch)
        arrays.update({f"{name}/{k}": v for k, v in out.items()})
        configs[name] = config
    for name, config in configs.items():
        arrays[f"{name}/config"] = np.array(json.dumps(config, sort_keys=True))
    arrays["_meta"] = np.array(
        json.dumps(
            {
                "source_sha": sha,
                "wenv_blob": blob,
                "torch": torch.__version__,
                "platform": sys.platform,
                "machine": platform.machine(),
                "cases": sorted(configs),
            },
            sort_keys=True,
        )
    )
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size} bytes, {len(arrays)} arrays)")


if __name__ == "__main__":
    main()
