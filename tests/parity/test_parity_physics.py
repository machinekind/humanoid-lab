"""One control step of MJX on JAX against MJWarp, from identical states.

The JAX wrapped env runs 4 worlds for 100 steps with 50-step episodes, and
each step is recorded. Worlds 0 and 1 get full-size actions and fall. Worlds
2 and 3 get half-size actions and time out. Every 4th step from step 2 gives
one state per world: the wrapped data before the step, the ctrl the step
wrote and the post-step data before any respawn. The 100 states are stepped
once more on MJWarp, all at once as 100 worlds, and the resulting qpos and
qvel are compared with the JAX post-step data.

Pushes are off, so no kick enters between the recorded pre-step data and
the physics step. DR is off. The tolerance is tolerances.physics(device).

On CPU, MJX on JAX in float32 carries nearly all of the error. In float64 it
matches MuJoCo C to round-off, and MJWarp stays within 1e-4 of MuJoCo C on
qvel. A failure message therefore gives each backend's distance from MuJoCo
C in float64 at the worst state.

The parity figures against MJX look the same on every device at three
digits, because MJX's float32 error dominates them. So the report also gets
rows that see MJWarp itself. Every run records each backend's distance from
MuJoCo C in float64 at all 100 states (physics[mjx vs C], physics[<device>
vs C]) and a rerun from the same states with a fresh WarpPhysics
(physics[<device> rerun]). A CUDA run also steps the states on the CPU warp
device in the same process and records MJWarp on CUDA against MJWarp on CPU
(physics[cuda vs cpu]), under the CPU tolerance. The two builds are not the
same arithmetic: warp compiles CUDA with --fmad=true (module option fuse_fp,
default True) and CPU with no FP contraction, and CUDA sums with atomics and
shuffle trees. So a nonzero "differing" share is the expected result. A
share of 0 with "warp device is cuda" at 1 is the anomaly, and the
physics[<device> build] rows say which build policy was in force. In every
row "<field> max" carries the max relative error too, and "<field>
differing" is the share of elements that differ at all, so a one-ulp
disagreement is visible.
"""

from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
import parity_report
import pytest
import tolerances

torch = pytest.importorskip("torch")

import replay
import warp as wp
from mujoco.mjx.third_party.mujoco_warp._src import solver

from humanoid_lab.torch_warp.physics import WarpPhysics

N = 4
STEPS = 100
L = 50  # episode length
AMPLITUDES = [1.0, 1.0, 0.5, 0.5]
ACTION_SEED = 3
# Fresh uniform actions on every step rarely topple the robot, even at full
# size. Each world instead holds the sign of a uniform draw, times its
# amplitude, for HOLD steps, plus a per-step jitter of JITTER times the
# amplitude.
HOLD = 25
JITTER = 0.25
# The steps that give a state. Step 50 follows the step-49 timeout.
SAMPLED = range(2, STEPS, 4)


@dataclasses.dataclass
class States:
    env: object
    record: object
    pre: dict  # FRAME_FIELDS, (100, ...)
    post: dict  # FRAME_FIELDS, (100, ...)
    ctrl: np.ndarray  # (100, nu)
    rows: list  # (t, world) of each row


def _actions(action_size) -> np.ndarray:
    """float32 (STEPS, N, action_size): held bang-bang actions plus a per-step jitter."""
    draws = replay.random_actions(
        -(-STEPS // HOLD), N, action_size, AMPLITUDES, ACTION_SEED
    )
    held = np.sign(draws) * np.asarray(AMPLITUDES, dtype=np.float32)[None, :, None]
    jitter = (
        replay.random_actions(STEPS, N, action_size, AMPLITUDES, ACTION_SEED + 100)
        * JITTER
    )
    return np.clip(np.repeat(held, HOLD, axis=0)[:STEPS] + jitter, -1.0, 1.0).astype(
        np.float32
    )


def _assert_covers_events(record):
    """Fail unless the sampled states hold a fall, a respawn and a foot on the floor."""
    steps = record.steps
    done = np.stack([st.done for st in steps])  # (STEPS, N)
    truncation = np.stack([st.truncation for st in steps])
    rows = [(t, w) for t in SAMPLED for w in range(N)]
    falls = [(t, w) for t, w in rows if done[t, w] and not truncation[t, w]]
    assert falls, "no state ends in a fall"
    # A state at step t follows a respawn when its world was done at step t-1.
    respawns = [(t, w) for t, w in rows if t > 0 and done[t - 1, w]]
    assert respawns, "no state follows a respawn"
    contacts = [(t, w) for t, w in rows if steps[t].info["last_contact"][w].any()]
    assert contacts, "no state has a foot on the floor"


@pytest.fixture(scope="module")
def states() -> States:
    hydra = replay.hydra_dict(
        ["actuators=deploy_pd"], env={"push": {"enable": False}}, domain_rand=False
    )
    env = replay.jax_env(hydra)
    assert env.mj_model.na == 0, (
        f"the model has {env.mj_model.na} activation states. step_mjwarp must copy act as well"
    )
    record = replay.record_jax(env, _actions(env.action_size), episode_length=L)
    pres, posts, rows = [], [], []
    for t in SAMPLED:
        # The wrapped data after the previous step. A world that was done
        # already holds its respawned first data.
        pres.append(record.reset_data if t == 0 else record.steps[t - 1].data)
        posts.append(record.steps[t].physics)
        rows.extend((t, w) for w in range(N))
    pre = {f: np.concatenate([p[f] for p in pres]) for f in pres[0]}
    post = {f: np.concatenate([p[f] for p in posts]) for f in posts[0]}
    _assert_covers_events(record)
    return States(
        env=env, record=record, pre=pre, post=post, ctrl=post["ctrl"], rows=rows
    )


@pytest.fixture(scope="module")
def device(parity_device) -> torch.device:
    device = replay.torch_device(parity_device)
    if device.type == "cuda":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip(
                f"--parity-device={parity_device} needs CUDA, and warp.is_cuda_available() is False"
            )
    return device


def step_mjwarp(states, device):
    """Step every state once on MJWarp. Returns (qpos, qvel) as numpy."""
    env = states.env
    ctrl = states.ctrl
    n = len(states.rows)
    c = env.resolved_constants()
    physics = WarpPhysics(
        env.mj_model, n, device, c.naconmax_per_env, c.njmax, env.n_substeps
    )
    # The report's device rows rest on the warp device being the one asked
    # for. WarpPhysics raises on a mismatch; this states it where the rows
    # are made.
    assert physics.wdev.is_cuda == (device.type == "cuda"), (
        f"warp device {physics.wdev} for {device}"
    )
    assert physics.data.qpos.device.type == device.type, physics.data.qpos.device
    # The build policy behind the rows. parity_report stores floats, so the
    # flags go in as 0 or 1.
    build = f"physics[{device.type} build]"
    opts = wp.get_module(solver.__name__).options
    parity_report.record_value(
        build, "warp device is cuda", float(physics.wdev.is_cuda)
    )
    parity_report.record_value(build, "fuse_fp", float(opts.get("fuse_fp", True)))
    parity_report.record_value(build, "fast_math", float(opts.get("fast_math", False)))
    parity_report.record_value(
        build, "cpu_compiler_flags set", float(wp.config.cpu_compiler_flags is not None)
    )

    def t(a):
        return torch.as_tensor(np.array(a, dtype=np.float32), device=device)

    pre = states.pre
    physics.reset_worlds(t(pre["qpos"]), t(pre["qvel"]), t(ctrl))
    # reset_worlds zeroes the warm start, then runs forward. The forward does
    # not touch qacc_warmstart, so the JAX warm start is copied after it.
    physics.data.qacc_warmstart.copy_(t(pre["qacc_warmstart"]))
    physics.step(t(ctrl))
    return replay.to_numpy(physics.data.qpos), replay.to_numpy(physics.data.qvel)


def _mujoco_c(states, row) -> dict:
    """qpos and qvel after one control step of MuJoCo C in float64 from the state at row."""
    m = states.env.mj_model
    d = mujoco.MjData(m)
    d.qpos[:] = states.pre["qpos"][row]
    d.qvel[:] = states.pre["qvel"][row]
    d.qacc_warmstart[:] = states.pre["qacc_warmstart"][row]
    d.ctrl[:] = states.ctrl[row]
    for _ in range(states.env.n_substeps):
        mujoco.mj_step(m, d)
    return {"qpos": d.qpos.copy(), "qvel": d.qvel.copy()}


@pytest.fixture(scope="module")
def mujoco_c(states) -> dict:
    """MuJoCo C in float64 at every row, (100, nq) and (100, nv). The 100 steps take under 0.1 s."""
    rows = [_mujoco_c(states, row) for row in range(len(states.rows))]
    return {f: np.stack([r[f] for r in rows]) for f in ("qpos", "qvel")}


def _record(check, field, expected, actual, tol=None) -> np.ndarray:
    """Three report rows for |actual - expected|: '<field> max' with its max
    rel, '<field> median', and '<field> differing', the share of elements
    that differ at all. Returns the elementwise error in float64."""
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    err = np.abs(actual - expected)
    parity_report.record(check, f"{field} max", expected, actual, tol=tol)
    parity_report.record_value(check, f"{field} median", np.median(err), tol=tol)
    parity_report.record_value(
        check, f"{field} differing", np.count_nonzero(err) / err.size
    )
    return err


def _worst(states, mujoco_c, err, field, warp) -> str:
    """Name the worst row, and say how far each backend is from MuJoCo C there."""
    row = int(np.argmax(err.max(axis=1)))
    t, w = states.rows[row]
    ref = mujoco_c[field][row]
    jax_c = np.abs(states.post[field][row] - ref).max()
    warp_c = np.abs(warp[row] - ref).max()
    return (
        f"step {t}, world {w}, error {float(err[row].max()):.3g}. At that state MuJoCo C in float64 "
        f"is {jax_c:.3g} from MJX and {warp_c:.3g} from MJWarp"
    )


def test_one_control_step_matches_mjx_at_100_states(states, device, mujoco_c):
    tol = tolerances.physics(device)
    pos_tol, vel_tol = f"atol={tol['qpos_atol']:g}", f"atol={tol['qvel_atol']:g}"
    qpos, qvel = step_mjwarp(states, device)
    check = f"physics[{device.type}]"
    qpos_err = _record(check, "qpos", states.post["qpos"], qpos, tol=pos_tol)
    qvel_err = _record(check, "qvel", states.post["qvel"], qvel, tol=vel_tol)

    # Each backend's distance from MuJoCo C in float64, at every state.
    results = {
        "mjx": (states.post["qpos"], states.post["qvel"]),
        device.type: (qpos, qvel),
    }
    # Run to run, from a fresh WarpPhysics. MJWarp sums with atomics and tile
    # reductions, so a CUDA rerun may differ at round-off. The row shows it.
    rerun = step_mjwarp(states, device)
    _record(f"physics[{device.type} rerun]", "qpos", qpos, rerun[0])
    _record(f"physics[{device.type} rerun]", "qvel", qvel, rerun[1])
    cross = None
    if device.type == "cuda":
        # MJWarp against itself across devices, in this process. Two builds
        # of one float32 algorithm must agree far better than either agrees
        # with MJX, so the CPU tolerance applies.
        results["cpu"] = step_mjwarp(states, torch.device("cpu"))
        cpu_tol = tolerances.PHYSICS_CPU
        cross = (
            _record(
                "physics[cuda vs cpu]",
                "qpos",
                results["cpu"][0],
                qpos,
                tol=f"atol={cpu_tol['qpos_atol']:g}",
            ),
            _record(
                "physics[cuda vs cpu]",
                "qvel",
                results["cpu"][1],
                qvel,
                tol=f"atol={cpu_tol['qvel_atol']:g}",
            ),
        )
    for name, (q, v) in results.items():
        _record(f"physics[{name} vs C]", "qpos", mujoco_c["qpos"], q)
        _record(f"physics[{name} vs C]", "qvel", mujoco_c["qvel"], v)

    # The messages are built only on a failure.
    assert qpos_err.max() <= tol["qpos_atol"], (
        f"qpos: worst at {_worst(states, mujoco_c, qpos_err, 'qpos', qpos)}. Tolerance {pos_tol}"
    )
    assert qvel_err.max() <= tol["qvel_atol"], (
        f"qvel: worst at {_worst(states, mujoco_c, qvel_err, 'qvel', qvel)}. Tolerance {vel_tol}"
    )
    if cross is not None:
        cpu_tol = tolerances.PHYSICS_CPU
        assert cross[0].max() <= cpu_tol["qpos_atol"], (
            f"MJWarp on CUDA is {cross[0].max():.3g} from MJWarp on CPU in qpos, over atol={cpu_tol['qpos_atol']:g}"
        )
        assert cross[1].max() <= cpu_tol["qvel_atol"], (
            f"MJWarp on CUDA is {cross[1].max():.3g} from MJWarp on CPU in qvel, over atol={cpu_tol['qvel_atol']:g}"
        )
