"""The Torch PPO trainer on MJWarp physics.

This is the alternative trainer for hosts without JAX-CUDA, such as Windows.
JAX/Brax stays the primary trainer.

- physics.py: MJWarp model and data, stepped from Torch tensors.
- env.py: the joystick task on Torch tensors, built on the shared task modules.
- ppo.py: the PPO loop, fed from the composed Hydra config.
- policy_io.py: networks, the normalizer and Brax checkpoint conversion.
- bench.py: a reproducible benchmark of the trainer, written to JSON.
"""

import os

# Before any jax import. On a Linux GPU host, jax would otherwise preallocate
# the VRAM that torch and warp need.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import torch  # noqa: F401
except ImportError as e:
    raise ImportError(
        "humanoid_lab.torch_warp needs torch. On Windows install CUDA torch from "
        "the PyTorch index first (`python -m pip install torch --index-url "
        "https://download.pytorch.org/whl/cu126`), then run "
        '`python -m pip install -e ".[torch_warp]"`. See '
        "src/humanoid_lab/torch_warp/README.md."
    ) from e
