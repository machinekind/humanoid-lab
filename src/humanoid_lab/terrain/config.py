"""Arena params as an env config block, the arena cache, and the terrain
task's warp budget check.

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


def ccd_scratch(sim: Mapping, bytes_per_slot: int, naconmax_per_env: int | None = None) -> dict:
    """The warp CCD scratch a run with sim config `sim` holds outside the
    XLA pool: `naccdmax_per_env`, else the naconmax budget, times
    `num_envs` slots of `bytes_per_slot`. `naconmax_per_env` stands in for
    an unset `sim.naconmax_per_env`. Slots and bytes are None when neither
    budget is known."""
    per_env = sim.get("naccdmax_per_env")
    source = "sim.naccdmax_per_env"
    if per_env is None:
        per_env = sim.get("naconmax_per_env")
        if per_env is None:
            per_env = naconmax_per_env
        source = "naconmax pool"
    slots = None if per_env is None else int(per_env) * int(sim.get("num_envs", 1))
    return {
        "bytes_per_slot": int(bytes_per_slot),
        "slots": slots,
        "bytes": None if slots is None else slots * int(bytes_per_slot),
        "source": source,
    }


def require_terrain_budgets(backend: str, sim: Mapping, *, ccd_slot_bytes: int) -> dict | None:
    """Refuse a terrain run on warp without explicit contact budgets.

    robot.yaml's `sim_budget` was measured on the flat floor. A terrain
    scene adds heightfield and box contacts, so it cannot stand in. On warp
    this raises unless `sim.naconmax_per_env` and `sim.njmax` are set. It
    then prints and returns the CCD scratch projection (`ccd_scratch`). The
    jax backend has no fixed buffers, and it returns None."""
    if backend != "warp":
        return None
    missing = [k for k in ("naconmax_per_env", "njmax") if sim.get(k) is None]
    if missing:
        names = " and ".join(f"task.env.sim.{k}" for k in missing)
        verb = "is" if len(missing) == 1 else "are"
        raise ValueError(
            f"a terrain run on warp needs explicit contact budgets. {names} {verb} unset. "
            "robot.yaml's sim_budget is a flat-floor measurement. Measure the arena with "
            "./run.sh check-terrain and set task.env.sim.naconmax_per_env and "
            "task.env.sim.njmax."
        )
    scratch = ccd_scratch(sim, ccd_slot_bytes)
    print(
        f"CCD scratch: {scratch['slots']} slots x {scratch['bytes_per_slot']} B = "
        f"{scratch['bytes'] / 1e9:.2f} GB outside the XLA pool "
        f"(naccdmax: {scratch['source']})"
    )
    return scratch
