"""Parity tolerances between the JAX and Torch backends.

TASK holds for the shared task math on identical float32 inputs. In
test_parity_task_math.py the worst error is 3.9e-3 abs on
compose.compute_rewards, at a relative error of 2.5e-7. In
test_parity_env.py's lockstep the worst error is 1.0 abs on
terms.joint_acc, whose values are about 1e6, at a relative error of 4.5e-7.
Outside the reward terms the worst lockstep error is 1.9e-6 abs.

The physics tolerances compare one control step of MJX on JAX with MJWarp
from identical states. On CPU test_parity_physics.py measures 2.6e-5 for
qpos and 5.2e-3 for qvel at its 100 states. Some other sets of 100 states
reach a qvel error of 2.3e-2 and a qpos error of 1.2e-4 on CPU. MJX on JAX
in float32 causes that error, and MJWarp stays close to MuJoCo C in float64.
Whether to keep PHYSICS_CPU as it is stays an open question for the owner
in the PR description. The CUDA values are 10 times the CPU values. They are
provisional until the PR author's Windows GPU run sets them.

The measured values come from macOS arm64 with torch 2.14.1, CPU build.

PPO_MATH holds for the Torch GAE, log-prob and entropy against brax's on
identical float32 inputs. The measured worst case is 3.8e-6 for log_prob.

The lockstep env check uses TASK, because its physics is replayed from JAX.

Bool and int outputs always match exactly.
"""

TASK = {"rtol": 1e-5, "atol": 1e-5}

PHYSICS_CPU = {"qpos_atol": 1e-4, "qvel_atol": 1e-2}

PHYSICS_CUDA = {k: 10 * v for k, v in PHYSICS_CPU.items()}

PPO_MATH = {"rtol": 1e-5, "atol": 1e-5}


def physics(device) -> dict:
    """Return the physics tolerances for a torch device such as cpu or cuda:0."""
    return PHYSICS_CUDA if str(device).startswith("cuda") else PHYSICS_CPU
