"""Procedural terrain arenas for terrain training, numpy only.

`generate` builds an `Arena` from `ArenaParams`. `bilinear` is the one
sampler every height read goes through, host or device.
"""

from humanoid_lab.terrain.arena import (
    GENERATOR_VERSION,
    HFIELD_BASE_Z,
    Arena,
    Box,
    HFieldSpec,
    TerrainSpec,
    TileSpec,
    fingerprint,
    generate,
    grid_axes,
    lookup_height,
    sample_frame,
    spec_from_dict,
    spec_to_dict,
)
from humanoid_lab.terrain.params import (
    STAIR_TYPES,
    TYPES,
    ArenaParams,
    Ramp,
    params_from_dict,
    params_to_dict,
    slope_plateau_height,
    stair_flight_half,
    stair_steps,
    stair_tread_bounds,
    summit_platform_half,
)
from humanoid_lab.terrain.sampling import bilinear

__all__ = [
    "GENERATOR_VERSION",
    "HFIELD_BASE_Z",
    "STAIR_TYPES",
    "TYPES",
    "Arena",
    "ArenaParams",
    "Box",
    "HFieldSpec",
    "Ramp",
    "TerrainSpec",
    "TileSpec",
    "bilinear",
    "fingerprint",
    "generate",
    "grid_axes",
    "lookup_height",
    "params_from_dict",
    "params_to_dict",
    "sample_frame",
    "slope_plateau_height",
    "spec_from_dict",
    "spec_to_dict",
    "stair_flight_half",
    "stair_steps",
    "stair_tread_bounds",
    "summit_platform_half",
]
