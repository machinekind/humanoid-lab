"""CLI: measure a robot's warp contact and constraint budgets.

    ./run.sh check-contacts --robot roboto_origin --preset deploy_pd

Builds the joystick env for a robot/preset and rolls three regimes, reporting
the per-world peak of both budgets and what the robot.yaml `sim_budget`
block (naconmax_per_env / njmax) should record to clear that peak with
headroom.

The three regimes, each rolled from `--seeds` different reset draws (the
reset pose noise moves the contact count a long way, so one draw is not a
measurement):
  standing  neutral (all-zero) actions from the reset keyframe.
  walking   a scripted full-amplitude sinusoid on every joint, alternating
            phase between neighbours. Not a gait -- a perturbed, flailing
            motion that puts feet down hard and puts more than the soles on
            the floor.
  fallen    the base starts tilted and lifted, then crumples under zero
            actions and keeps being stepped on the floor. Each seed takes the
            next attitude from FALLEN_ATTITUDES, so the sweep lands the robot
            on its front, its side and its back rather than trusting one
            angle. Early training is mostly fallen robots, which is why this
            regime is measured at all.

Measured on our two robots, the fallen regime does NOT always dominate. A
neutral action holds neither robot's home keyframe up under the stock preset
gains -- both collapse within about a second -- so "standing" is itself a
fall, and its first few steps, with both feet flat and every sole capsule
loaded, are where the contact peak often lands. `height_end` is reported per
regime so that is visible rather than assumed.

What is measured, and what is derived. On warp, `data._impl.nacon` and
`data._impl.nefc` are live counters and both peaks are read straight off
them. On jax there are no counters -- `_impl.ncon` / `_impl.nefc` are the
buffer sizes fixed at make_data time -- so the contact peak is counted from
the contact array (`dist < 0`, sim_budget.active_contacts) and the ROW peak
is DERIVED from it, not measured:

    rows = ne + nf + nl + rows_per_contact * active_contacts

`rows_per_contact` is MuJoCo's own cost, 2*(condim-1) under a pyramidal cone.
`ne + nf + nl` come from the jax impl, where `nl` counts every limited joint
rather than the limits currently engaged, so the derived number is an upper
bound on the rows a warp run of the same state would allocate. The result
carries `nefc_derived` so a reader never reads the derivation as a
measurement. Confirm both peaks on a GPU box before trusting the row number
to the metre.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from humanoid_lab import paths, sim_budget

REGIMES = ("standing", "walking", "fallen")

# The inherited sizing rule: about 7x the measured peak. Sized against the
# fallen peak, which is the worst case early training spends most of its time
# in, so this is headroom over a bad state, not over a good one.
HEADROOM = 7.0

# Rounding steps for the two recommendations. naconmax_per_env multiplies by
# the env count into one device allocation, so it moves in small steps; njmax
# is per world and cheap, so it rounds to a coarser number.
NACON_STEP = 8
NJMAX_STEP = 32

DEFAULT_STEPS = 200  # 4 s of control at ctrl_dt=0.02; every measured peak
DEFAULT_SEEDS = 5    # landed inside the first 80 steps
WALK_HZ = 1.5

# (axis, degrees) for the fallen regime, cycled one per seed: face down, on
# its side, on its back.
FALLEN_ATTITUDES = (("y", 90.0), ("x", 90.0), ("y", 180.0))
FALLEN_DROP_M = 0.25


def _row_accounting(env, data) -> tuple[int, int]:
    """(constant rows, rows per contact) for the derived jax row count.

    The constant part is the equality, friction-loss and joint-limit rows,
    which mjx sizes from the model and not from the state. `rows_per_contact`
    uses the model's largest condim: a contact takes the larger condim of its
    two geoms, so the largest one in the model bounds every pair.
    """
    impl = getattr(data, "_impl", None)
    const = 0
    for name in ("ne", "nf", "nl"):
        value = getattr(impl, name, None)
        if value is not None:
            const += int(np.asarray(value).max())
    dim = int(np.max(env.mj_model.geom_condim))
    return const, sim_budget.rows_per_contact(int(env.mj_model.opt.cone), dim)


def _contact_dist(data):
    impl = getattr(data, "_impl", None)
    contact = getattr(impl, "contact", None)
    return None if contact is None else contact.dist


def _tilted(env, state, axis: str, tilt_deg: float, drop_m: float):
    """`state` with the base rotated `tilt_deg` about `axis` and lifted, at rest.

    The robot is dropped rather than pushed over: a push needs a direction, a
    magnitude and a duration to argue about, while a tilt past the point of no
    return lands the robot down in every case and takes two numbers.
    """
    half = np.deg2rad(tilt_deg) / 2.0
    sin = np.sin(half)
    quat = jp.array([np.cos(half), sin if axis == "x" else 0.0, sin if axis == "y" else 0.0, 0.0])
    base = env._base_qadr
    qpos = state.data.qpos.at[base + 3 : base + 7].set(quat)
    qpos = qpos.at[base + 2].add(drop_m)
    data = state.data.replace(qpos=qpos, qvel=jp.zeros_like(state.data.qvel))
    data = mjx.forward(env.mjx_model, data)
    return state.replace(data=data)


def _walk_action(n: int, i: int, dt: float):
    """Full-amplitude sinusoid, neighbouring joints in antiphase."""
    phase = (jp.arange(n) % 2) * jp.pi
    return jp.sin(2.0 * jp.pi * WALK_HZ * i * dt + phase)


def measure_env(
    env, steps: int = DEFAULT_STEPS, seeds: int = DEFAULT_SEEDS, seed: int = 0
) -> dict:
    """Roll every regime in `env` and report the peaks. See the module docstring."""
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    zeros = jp.zeros(env.action_size)

    probe = reset(jax.random.PRNGKey(seed))
    const_rows, per_contact = _row_accounting(env, probe.data)
    derived = sim_budget.live_peaks(probe.data)[1] is None

    def peaks_of(data) -> tuple[int, int]:
        nacon, nefc = sim_budget.live_peaks(data)
        if nacon is None:
            nacon = sim_budget.active_contacts(_contact_dist(data))
        if nefc is None:
            nefc = const_rows + per_contact * nacon
        return int(nacon), int(nefc)

    regimes = {}
    for name in REGIMES:
        best = {
            "nacon_max": 0, "nefc_max": 0, "peak_step": 0, "peak_seed": seed,
            "height_end": 0.0,
        }
        for k in range(seeds):
            rollout_seed = seed + k
            state = reset(jax.random.PRNGKey(rollout_seed))
            if name == "fallen":
                axis, deg = FALLEN_ATTITUDES[k % len(FALLEN_ATTITUDES)]
                state = _tilted(env, state, axis, deg, FALLEN_DROP_M)
            nacon_max, nefc_max, at = 0, 0, 0
            for i in range(steps):
                action = _walk_action(env.action_size, i, env.dt) if name == "walking" else zeros
                state = step(state, action)
                nacon, nefc = peaks_of(state.data)
                if nacon > nacon_max:
                    nacon_max, at = nacon, i
                nefc_max = max(nefc_max, nefc)
            best["nefc_max"] = max(best["nefc_max"], nefc_max)
            if nacon_max > best["nacon_max"]:
                best.update(
                    nacon_max=nacon_max,
                    peak_step=at,
                    peak_seed=rollout_seed,
                    # The reader's sanity check on the regime label: a
                    # "standing" rollout that ends near the floor was not
                    # standing.
                    height_end=round(float(state.data.qpos[env._base_qadr + 2]), 4),
                )
        regimes[name] = {**best, "steps": steps, "seeds": seeds}

    peak = {
        "nacon_max": max(r["nacon_max"] for r in regimes.values()),
        "nefc_max": max(r["nefc_max"] for r in regimes.values()),
    }
    return {
        "backend": env._backend,
        "steps": steps,
        "seeds": seeds,
        "seed": seed,
        "regimes": regimes,
        "peak": peak,
        "nefc_derived": derived,
        "const_rows": const_rows,
        "rows_per_contact": per_contact,
        "recommend": {
            "naconmax_per_env": sim_budget.recommend_budget(
                peak["nacon_max"], HEADROOM, NACON_STEP
            ),
            "njmax": sim_budget.recommend_budget(peak["nefc_max"], HEADROOM, NJMAX_STEP),
            "headroom": HEADROOM,
        },
    }


def measure_robot(
    robot_dir,
    preset: str,
    steps: int = DEFAULT_STEPS,
    seeds: int = DEFAULT_SEEDS,
    seed: int = 0,
) -> dict:
    """Build the joystick env for `robot_dir`/`preset` and measure it."""
    from humanoid_lab.registry import make_env

    robot_dir = Path(robot_dir)
    env = make_env("joystick", robot_dir, preset)
    return {
        "robot": robot_dir.name,
        "preset": preset,
        **measure_env(env, steps, seeds, seed),
    }


def format_report(result: dict) -> str:
    lines = [
        f"robot {result['robot']} / preset {result['preset']} / backend {result['backend']}",
        f"{result['steps']} control steps x {result['seeds']} seeds per regime, "
        f"from seed {result['seed']}",
    ]
    if result["nefc_derived"]:
        lines.append(
            f"rows DERIVED (no live counter on this backend): {result['const_rows']} constant "
            f"+ {result['rows_per_contact']} per contact"
        )
    lines.append(
        f"{'regime':<10} {'nacon':>6} {'nefc':>7} {'at step':>8} {'seed':>5} {'end z':>8}"
    )
    for name in REGIMES:
        r = result["regimes"][name]
        lines.append(
            f"{name:<10} {r['nacon_max']:>6} {r['nefc_max']:>7} {r['peak_step']:>8} "
            f"{r['peak_seed']:>5} {r['height_end']:>8.3f}"
        )
    peak, rec = result["peak"], result["recommend"]
    lines.append(f"{'PEAK':<10} {peak['nacon_max']:>6} {peak['nefc_max']:>7}")
    lines.append(
        f"recommend at {rec['headroom']}x headroom: "
        f"naconmax_per_env={rec['naconmax_per_env']} njmax={rec['njmax']}"
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", required=True, help="robot dir name under robots/")
    ap.add_argument("--preset", required=True, help="actuator preset name")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="control steps per rollout")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="rollouts per regime")
    ap.add_argument("--seed", type=int, default=0, help="first seed; the rest count up from it")
    ap.add_argument("--out", type=Path, default=None, help="also write the result as json here")
    args = ap.parse_args()

    result = measure_robot(
        paths.ROBOTS_DIR / args.robot, args.preset, args.steps, args.seeds, args.seed
    )
    result["date"] = date.today().isoformat()
    print(format_report(result))
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
