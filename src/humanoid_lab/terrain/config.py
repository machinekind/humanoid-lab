"""Arena params as an env config block, and the arena cache.

A config block is `params_to_dict` output with `type_caps` expanded to every
terrain type. A fixed key set lets an override name any type's cap, and an
unknown key still raises. A cap of 1.0 means uncapped. `params_from_config`
drops such caps, so a block written from `ArenaParams()` names the default
arena and its fingerprint.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping

from humanoid_lab.terrain.arena import Arena, generate
from humanoid_lab.terrain.params import TYPES, ArenaParams, params_to_dict

# The arena jax runs on a CPU: one terrain row at difficulty 0.5 after the
# flat row, no scattered boxes and a 0.4 m border. It has 33 boxes, all
# stairs, on 221 x 821 nodes. A config block override, not a full block.
CPU_ARENA = {
    "difficulties": [0.5],
    "flat_row": True,
    "discrete_count": 0,
    "grid_fill_prob": 0.0,
    "border": 0.4,
}


def config_from_params(params: ArenaParams) -> dict:
    """The config block for `params`, with a cap for every terrain type."""
    block = params_to_dict(params)
    block["type_caps"] = {t: params.cap(t) for t in TYPES}
    return block


def params_from_config(block: Mapping) -> ArenaParams:
    """`ArenaParams` from a full or partial config block.

    Caps equal to 1.0 are dropped. An unknown key or terrain type raises."""
    data = dict(block)
    caps = dict(data.pop("type_caps", None) or {})
    data["type_caps"] = {t: c for t, c in caps.items() if float(c) != 1.0}
    return ArenaParams(**data)


@functools.lru_cache(maxsize=4)
def arena_for(params: ArenaParams) -> Arena:
    """The arena `params` builds, generated once per process.

    Its `lookup` and `hfield_data` are read-only, so no caller can change
    the arena another caller shares."""
    arena = generate(params)
    arena.lookup.flags.writeable = False
    arena.hfield_data.flags.writeable = False
    return arena
