"""Parity tolerances between the JAX and Torch backends.

TASK holds for the shared task math on identical float32 inputs. Its
measured worst case is a relative error of 2.9e-7.

The physics tolerances compare one control step of MJX on JAX with MJWarp
from identical states. On CPU the measured worst cases are 2.6e-5 for qpos
and 2.5e-3 for qvel. The CUDA values are 10 times the CPU values. They are
provisional until the PR author's Windows GPU run sets them.

PPO_MATH holds for the Torch GAE, log-prob and entropy against brax's on
identical float32 inputs. The measured worst case is 3.8e-6 for log_prob.

Bool and int outputs always match exactly.
"""

TASK = {"rtol": 1e-5, "atol": 1e-5}

PHYSICS_CPU = {"qpos_atol": 1e-4, "qvel_atol": 1e-2}

PHYSICS_CUDA = {k: 10 * v for k, v in PHYSICS_CPU.items()}

PPO_MATH = {"rtol": 1e-5, "atol": 1e-5}
