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
    draws = replay.random_actions(-(-STEPS // HOLD), N, action_size, AMPLITUDES, ACTION_SEED)
    held = np.sign(draws) * np.asarray(AMPLITUDES, dtype=np.float32)[None, :, None]
    jitter = replay.random_actions(STEPS, N, action_size, AMPLITUDES, ACTION_SEED + 100) * JITTER
    return np.clip(np.repeat(held, HOLD, axis=0)[:STEPS] + jitter, -1.0, 1.0).astype(np.float32)


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
    hydra = replay.hydra_dict(["actuators=deploy_pd"], env={"push": {"enable": False}}, domain_rand=False)
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
    return States(env=env, record=record, pre=pre, post=post, ctrl=post["ctrl"], rows=rows)


@pytest.fixture(scope="module")
def device(parity_device) -> torch.device:
    device = replay.torch_device(parity_device)
    if device.type == "cuda":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip(f"--parity-device={parity_device} needs CUDA, and warp.is_cuda_available() is False")
    return device


def step_mjwarp(states, device):
    """Step every state once on MJWarp. Returns (qpos, qvel) as numpy."""
    env = states.env
    ctrl = states.ctrl
    n = len(states.rows)
    c = env.resolved_constants()
    physics = WarpPhysics(env.mj_model, n, device, c.naconmax_per_env, c.njmax, env.n_substeps)

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


def _worst(states, err, field, warp) -> str:
    """Name the worst row, and say how far each backend is from MuJoCo C there."""
    row = int(np.argmax(err.max(axis=1)))
    t, w = states.rows[row]
    ref = _mujoco_c(states, row)[field]
    jax_c = np.abs(states.post[field][row] - ref).max()
    warp_c = np.abs(warp[row] - ref).max()
    return (
        f"step {t}, world {w}, error {float(err[row].max()):.3g}. At that state MuJoCo C in float64 "
        f"is {jax_c:.3g} from MJX and {warp_c:.3g} from MJWarp"
    )


def test_one_control_step_matches_mjx_at_100_states(states, device):
    tol = tolerances.physics(device)
    qpos, qvel = step_mjwarp(states, device)
    qpos_err = np.abs(qpos.astype(np.float64) - states.post["qpos"])
    qvel_err = np.abs(qvel.astype(np.float64) - states.post["qvel"])
    check = f"physics[{device.type}]"
    pos_tol, vel_tol = f"atol={tol['qpos_atol']:g}", f"atol={tol['qvel_atol']:g}"
    parity_report.record_value(check, "qpos max", qpos_err.max(), tol=pos_tol)
    parity_report.record_value(check, "qpos median", np.median(qpos_err), tol=pos_tol)
    parity_report.record_value(check, "qvel max", qvel_err.max(), tol=vel_tol)
    parity_report.record_value(check, "qvel median", np.median(qvel_err), tol=vel_tol)
    # The messages are built only on a failure.
    assert qpos_err.max() <= tol["qpos_atol"], (
        f"qpos: worst at {_worst(states, qpos_err, 'qpos', qpos)}. Tolerance {pos_tol}"
    )
    assert qvel_err.max() <= tol["qvel_atol"], (
        f"qvel: worst at {_worst(states, qvel_err, 'qvel', qvel)}. Tolerance {vel_tol}"
    )
