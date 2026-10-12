"""The real WarpJoystick in a closed loop on the retained recipe.

MJWarp physics and TorchDraws run with nothing replayed from JAX. Sixteen
worlds run 500 steps of uniform random actions with 70-step episodes. The
robot falls by itself after about 60 to 100 steps, whatever the action size.
With 70-step episodes, many end on the timeout and many end on a fall.
Eight seed pairs on CPU gave 48 to 89 falls and 28 to 66 timeouts.

This is a behavior check, so it has no tolerance. After every step the
module fixture checks that every output is finite, that each respawn writes
the first physics state and the first observation, that it resets yaw_ref
and land_frac, and that no episode runs past its length. It also counts the
falls and the timeouts. The tests read the collected facts.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import replay

from humanoid_lab.torch_warp.env import WarpJoystick

N = 16
STEPS = 500
L = 70


def _float_leaves(tree, prefix=""):
    """(name, tensor) for every floating-point tensor in a nested dict."""
    for k, v in tree.items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _float_leaves(v, f"{name}.")
        elif isinstance(v, torch.Tensor) and v.is_floating_point():
            yield name, v


def _rows_equal(actual, expected, mask) -> bool:
    return bool(torch.equal(actual[mask], expected[mask]))


@pytest.fixture(scope="module")
def run(parity_device):
    device = replay.torch_device(parity_device)
    hydra = replay.hydra_dict(["experiment=roboto_gait_v40"])
    tenv = WarpJoystick(hydra, N, L, device=device, seed=0)
    assert "heading" in tenv.flags and "land" in tenv.flags, tenv.flags

    g = torch.Generator(device=device).manual_seed(0)
    nu = tenv.action_size

    facts = {
        "nonfinite": [],
        "bad_pose": [],
        "bad_obs": [],
        "bad_yaw_ref": [],
        "bad_land_frac": [],
        "too_long": [],
        "falls": 0,
        "timeouts": 0,
        "respawns": 0,
        "timeout_lengths": set(),
        "max_length": 0,
    }

    def check_finite(t, name, tensor):
        if not bool(torch.isfinite(tensor).all()):
            facts["nonfinite"].append((t, name))

    state, priv = tenv.reset()
    check_finite(-1, "reset state", state)
    check_finite(-1, "reset priv", priv)
    first = tenv.first
    count = torch.zeros(N, dtype=torch.int64, device=device)
    for t in range(STEPS):
        action = torch.rand((N, nu), generator=g, device=device) * 2 - 1
        state, priv, reward, done, truncation, terms = tenv.step(action)
        info = tenv.info

        check_finite(t, "state", state)
        check_finite(t, "priv", priv)
        check_finite(t, "reward", reward)
        for name, v in terms.items():
            check_finite(t, f"terms.{name}", v)
        for name, v in _float_leaves(info):
            check_finite(t, f"info.{name}", v)

        count += 1
        facts["max_length"] = max(facts["max_length"], int(count.max()))
        if bool((count > L).any()):
            facts["too_long"].append((t, count.tolist()))

        if bool(done.any()):
            facts["respawns"] += int(done.sum())
            falls = done & ~truncation
            facts["falls"] += int(falls.sum())
            facts["timeouts"] += int(truncation.sum())
            facts["timeout_lengths"].update(count[truncation].tolist())

            worlds = done.nonzero().flatten().tolist()
            if not _rows_equal(tenv.physics.data.qpos, first["qpos"], done):
                facts["bad_pose"].append((t, worlds))
            if not (_rows_equal(state, first["state"], done) and _rows_equal(priv, first["priv"], done)):
                facts["bad_obs"].append((t, worlds))
            if not _rows_equal(info["yaw_ref"], first["yaw"], done):
                facts["bad_yaw_ref"].append((t, worlds))
            land = info["land_frac"][done]
            if not bool((land == 0).all()):
                facts["bad_land_frac"].append((t, worlds))
            count = torch.where(done, torch.zeros_like(count), count)
    return facts


def test_outputs_stay_finite(run):
    assert not run["nonfinite"], f"non-finite outputs (step, name): {run['nonfinite'][:10]}"


def test_falls_and_timeouts_occur(run):
    assert run["falls"] > 0, "no world fell in 500 steps"
    assert run["timeouts"] > 0, "no world timed out in 500 steps"
    assert run["timeout_lengths"] == {L}, (
        f"a timeout ended an episode of length {sorted(run['timeout_lengths'])}, and only {L} times out"
    )


def test_a_respawn_restores_the_first_pose_and_obs(run):
    assert run["respawns"] > 0
    assert not run["bad_pose"], f"qpos differs from the first qpos after a respawn (step, worlds): {run['bad_pose']}"
    assert not run["bad_obs"], (
        f"state or priv differs from the first obs after a respawn (step, worlds): {run['bad_obs']}"
    )


def test_a_respawn_resets_yaw_ref_and_land_frac(run):
    assert run["respawns"] > 0
    assert not run["bad_yaw_ref"], (
        f"yaw_ref differs from the first yaw after a respawn (step, worlds): {run['bad_yaw_ref']}"
    )
    assert not run["bad_land_frac"], f"land_frac is not zero after a respawn (step, worlds): {run['bad_land_frac']}"


def test_no_episode_outlives_its_length(run):
    assert not run["too_long"], f"an episode ran past {L} steps (step, counters): {run['too_long'][:5]}"
    assert run["max_length"] == L, f"the longest episode ran {run['max_length']} steps, and timeouts end at {L}"
