"""The random draws of each backend: repeatability, independence, ranges and statistics.

TorchDraws and the JAX adapters are tested on their own. Torch samples are
never compared with JAX samples. Only the parity runs feed both sides the
same draws (replay.ReplayDraws).

The JAX adapters are jax_reset_draws, jax_step_draws, jax_command_draws and
jax_push_draws in envs/joystick.py, and jax_dr_draws in dr/randomize.py. The
file is model-free and builds no env.
"""

from __future__ import annotations

import functools

import jax
import numpy as np
import pytest

from humanoid_lab import registry
from humanoid_lab.dr import randomize, transform
from humanoid_lab.envs import joystick, progress, push

torch = pytest.importorskip("torch")

import replay

from humanoid_lab.torch_warp.env import TorchDraws

NU = 12
S = 60  # actor obs size
N = 64
N_RANGE = 4096
N_STATS = 16384
SEED = 7
DIMS = {"nbody": 20, "nu": 12, "nv": 18, "nfoot": 2}
DR_FIELDS = ("com_offset", "joint_gains", "dof", "foot_friction", "motor_strength", "base_mass_add")
ALL_ON = {k: {"enable": True} for k in DR_FIELDS}
ODD_RANGES = {
    **ALL_ON,
    "base": {
        "floor_friction": [0.3, 1.7],
        "base_mass": [0.55, 1.45],
        "link_mass": [0.85, 1.15],
        "gain_fallback": [0.75, 1.33],
    },
    "motor_strength": {"enable": True, "range": [0.37, 1.13]},
    "base_mass_add": {"enable": True, "kg": 2.7},
}
PUSH_KNOBS = ("vel_z", "ang_vel_rp", "ang_vel_yaw")
PROB_KEYS = tuple(prob_key for _, _, prob_key, _ in progress.COMMAND_DRAWS)

ALL_OVERRIDES = {
    "command": {k: 0.2 for k in PROB_KEYS},
    "push": {
        "enable": True,
        "interval_steps_range": (5, 15),
        "vel_z": 0.3,
        "ang_vel_rp": 0.5,
        "ang_vel_yaw": 0.7,
    },
    "symmetry": {"enable": True},
    "no_progress": {"enable": True},
}
RANGE_OVERRIDES = {
    **ALL_OVERRIDES,
    "command": {**ALL_OVERRIDES["command"], "vx": (-0.37, 1.13), "wz": (0.25, 0.25)},
}
# The leaves that are JAX keys.
KEY_LEAVES = ("rng", "noise_key")


def cfg(overrides):
    c = joystick.default_config()
    registry._apply_overrides(c, overrides)
    return c


def table(dr_cfg):
    return transform.DR_DRAWS(transform.resolved_cfg(dr_cfg), DIMS)


CFG_ALL = cfg(ALL_OVERRIDES)
CFG_RANGES = cfg(RANGE_OVERRIDES)
TABLE_ALL = table(ALL_ON)


@pytest.fixture(scope="module")
def device(parity_device):
    return replay.torch_device(parity_device)


# -- sequences ------------------------------------------------------------------
def flatten(tree, prefix="") -> dict:
    """{path: numpy} for every leaf. A None leaf is an off draw and is left out."""
    out = {}
    for k, v in tree.items():
        path = f"{prefix}/{k}" if prefix else k
        if v is None:
            continue
        if isinstance(v, dict):
            out.update(flatten(v, path))
        elif isinstance(v, torch.Tensor):
            out[path] = v.cpu().numpy()
        else:
            out[path] = np.asarray(v)
    return out


def seq(draws, c, t, n=N) -> dict:
    """reset, three steps and one DR draw, the order WarpJoystick uses them."""
    return flatten(
        {
            "reset": draws.reset(n, c, NU),
            "step0": draws.step(n, c, S),
            "step1": draws.step(n, c, S),
            "step2": draws.step(n, c, S),
            "dr": draws.dr(n, t),
        }
    )


@functools.cache
def _adapter(name, c_key, t_key):
    """jit(vmap(adapter)) for one config. The configs are looked up by key."""
    c, t = _CFGS[c_key], _TABLES[t_key]
    fn = {
        "reset": lambda k: joystick.jax_reset_draws(k, c, NU),
        "step": lambda k: joystick.jax_step_draws(k, c),
        "command": lambda k: joystick.jax_command_draws(k, c.command),
        "push": lambda k: joystick.jax_push_draws(k, c.push),
        "dr": lambda k: randomize.jax_dr_draws(k, t),
    }[name]
    return jax.jit(jax.vmap(fn))


_CFGS: dict = {}
_TABLES: dict = {}


def jax_seq(seed, c, t, n=N, names=("reset", "step", "command", "push", "dr")) -> dict:
    """Each JAX adapter on split(PRNGKey(seed), n), flattened."""
    c_key, t_key = repr(c.to_dict()), repr(t)
    _CFGS.setdefault(c_key, c)
    _TABLES.setdefault(t_key, t)
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    return flatten({name: _adapter(name, c_key, t_key)(keys) for name in names})


def _float_leaves(s) -> list:
    return [p for p, v in s.items() if v.dtype == np.float32 and v.size > 1]


def _assert_equal(got, want, label) -> None:
    for path in sorted(set(got) & set(want)):
        assert got[path].dtype == want[path].dtype, f"{label}: {path} dtype {got[path].dtype} != {want[path].dtype}"
        if not np.array_equal(got[path], want[path]):
            diff = np.argwhere(got[path] != want[path])[0]
            raise AssertionError(
                f"{label}: {path} differs first at {tuple(diff)}: "
                f"{got[path][tuple(diff)]} against {want[path][tuple(diff)]}"
            )


# -- repeatability ---------------------------------------------------------------
def test_torch_draws_repeat_for_a_fixed_seed(device):
    a = seq(TorchDraws(SEED, device), CFG_ALL, TABLE_ALL)
    b = seq(TorchDraws(SEED, device), CFG_ALL, TABLE_ALL)
    assert set(a) == set(b)
    _assert_equal(a, b, "same seed")

    other = seq(TorchDraws(SEED + 1, device), CFG_ALL, TABLE_ALL)
    for path in _float_leaves(a):
        assert not np.array_equal(a[path], other[path]), f"seeds {SEED} and {SEED + 1} give the same {path}"

    # A state_dict taken after the first reset carries the rest of the sequence.
    draws = TorchDraws(SEED, device)
    draws.reset(N, CFG_ALL, NU)
    saved = draws.state_dict()
    fresh = TorchDraws(0, device)
    fresh.load_state_dict(saved)
    rest = flatten(
        {
            "step0": fresh.step(N, CFG_ALL, S),
            "step1": fresh.step(N, CFG_ALL, S),
            "step2": fresh.step(N, CFG_ALL, S),
            "dr": fresh.dr(N, TABLE_ALL),
        }
    )
    assert set(rest) == {p for p in a if not p.startswith("reset/")}
    _assert_equal(rest, a, "loaded state")


# -- independence of optional draws ------------------------------------------------
def _with(overrides, group, key, value):
    out = {k: dict(v) for k, v in overrides.items()}
    out[group][key] = value
    return out


# A zero ang_vel_rp or ang_vel_yaw keeps the ang draw and zeroes some of its
# components. The other components must not move.
ANG_KEPT = {"ang_vel_rp": [2], "ang_vel_yaw": [0, 1]}


def _fold_in_toggles():
    """(label, cfg, table, ang components kept) for each draw JAX makes with fold_in."""
    out = [(f"command.{k}=0", cfg(_with(ALL_OVERRIDES, "command", k, 0.0)), TABLE_ALL, None) for k in PROB_KEYS]
    out += [(f"push.{k}=0", cfg(_with(ALL_OVERRIDES, "push", k, 0.0)), TABLE_ALL, ANG_KEPT.get(k)) for k in PUSH_KNOBS]
    out += [(f"dr.{f} off", CFG_ALL, table({**ALL_ON, f: {"enable": False}}), None) for f in DR_FIELDS]
    return out


def _torch_toggles():
    """(label, cfg, table, ang components kept) for every optional draw."""
    out = _fold_in_toggles()
    for group, key, value in (
        ("push", "interval_steps_range", None),
        ("push", "enable", False),
        ("symmetry", "enable", False),
        ("no_progress", "enable", False),
    ):
        out.append((f"{group}.{key}={value}", cfg(_with(ALL_OVERRIDES, group, key, value)), TABLE_ALL, None))
    return out


def _assert_untouched(toggled, base, label, ang_kept) -> None:
    """Every leaf in both is equal. A partly zeroed ang is compared on its kept components."""
    if ang_kept is None:
        assert set(toggled) != set(base), f"{label}: the toggle changes no leaf"
    else:
        for path in [p for p in toggled if p.endswith("/ang")]:
            zeroed = [i for i in range(3) if i not in ang_kept]
            assert np.all(toggled[path][..., zeroed] == 0.0), f"{label}: {path} components {zeroed} are not zero"
            assert not np.array_equal(toggled[path], base[path]), f"{label}: {path} did not change"
            toggled = {**toggled, path: toggled[path][..., ang_kept]}
            base = {**base, path: base[path][..., ang_kept]}
    _assert_equal(toggled, base, label)


def test_torch_optional_draws_leave_the_other_streams_alone(device):
    base = seq(TorchDraws(SEED, device), CFG_ALL, TABLE_ALL)
    must_share = [f"{s}/command/{k}" for s in ("reset", "step0") for k in ("vx", "vy", "wz", "zero_on")]
    must_share += ["reset/pose_noise", "step0/noise", "step2/noise"]
    for label, c, t, ang_kept in _torch_toggles():
        toggled = seq(TorchDraws(SEED, device), c, t)
        missing = [p for p in must_share if p not in toggled]
        assert not missing, f"{label}: {missing} missing"
        _assert_untouched(toggled, base, label, ang_kept)


def test_jax_fold_in_draws_leave_the_other_draws_alone():
    """Only the fold-in draws are independent on JAX.

    JAX's split-based optional draws (the no-progress u_term, the ranged
    push countdown at reset and the mirror flag) shift later key splits.
    That is origin/main's key scheme, and the goldens pin it, so only Torch
    promises independence for every draw.
    """
    base = jax_seq(SEED, CFG_ALL, TABLE_ALL)
    must_share = {"reset/command/vx", "step/command/zero_on", "reset/pose_noise", "step/rng", "step/noise_key"}
    for label, c, t, ang_kept in _fold_in_toggles():
        toggled = jax_seq(SEED, c, t)
        assert must_share <= set(toggled), f"{label}: {must_share - set(toggled)} missing"
        _assert_untouched(toggled, base, label, ang_kept)


# -- ranges -------------------------------------------------------------------------
def _range_key(name):
    for draw, _, _, range_key in progress.COMMAND_DRAWS:
        if name == f"{draw}_vx":
            return range_key
    return None


def _prob_key(name):
    if name == "zero_on":
        return "zero_prob"
    for draw, _, prob_key, _ in progress.COMMAND_DRAWS:
        if name == f"{draw}_on":
            return prob_key
    raise KeyError(name)


def spec(path, c, t):
    """(kind, lo, hi) of a leaf. kind is uniform, bernoulli, randint or key.

    A bernoulli's lo is its probability. lo and hi of a uniform may be lists,
    one bound per element of the trailing axis.
    """
    parts = path.split("/")
    name = parts[-1]
    if parts[0] == "dr":
        _shape, lo, hi, _slot = t[name]
        return "uniform", lo, hi
    if name in KEY_LEAVES:
        return "key", None, None
    if name in ("vx", "vy", "wz"):
        return "uniform", *c.command[name]
    if _range_key(name):
        return "uniform", *c.command[_range_key(name)]
    if name.endswith("_on"):
        return "bernoulli", c.command[_prob_key(name)], None
    if name in ("pose_noise", "noise"):
        return "uniform", -1.0, 1.0
    if name in ("mirror_u", "u_term"):
        return "uniform", 0.0, 1.0
    if name == "push_countdown":
        return "randint", *(int(v) for v in c.push.interval_steps_range)
    kind = "randint" if name == "next_countdown" else "uniform"
    return kind, *push.push_ranges(c.push)[name]


def _sources(n, device, c, t):
    """(backend, flattened draws) for both backends."""
    return [
        ("torch", seq(TorchDraws(SEED, device), c, t, n)),
        ("jax", jax_seq(SEED, c, t, n)),
    ]


RANGE_TABLES = {
    "odd_ranges": table(ODD_RANGES),
    "odd_ranges_gains_off": table({**ODD_RANGES, "joint_gains": {"enable": False}}),
}


@pytest.mark.parametrize("dr_case", sorted(RANGE_TABLES))
def test_every_draw_lies_in_its_closed_range(device, dr_case):
    t = RANGE_TABLES[dr_case]
    checked = set()
    for backend, s in _sources(N_RANGE, device, CFG_RANGES, t):
        for path, v in s.items():
            kind, lo, hi = spec(path, CFG_RANGES, t)
            where = f"{backend} {path} ({dr_case})"
            if kind == "key":
                continue
            checked.add(path.split("/")[-1])
            if kind == "bernoulli":
                assert v.dtype == np.bool_, f"{where}: dtype {v.dtype}"
                continue
            if kind == "randint":
                assert v.dtype == np.int32, f"{where}: dtype {v.dtype}"
                assert v.min() >= lo and v.max() <= hi, f"{where}: [{v.min()}, {v.max()}] outside [{lo}, {hi}]"
                continue
            assert v.dtype == np.float32, f"{where}: dtype {v.dtype}"
            # Compare in float32. float32(0.8) > 0.8, so a draw equal to hi32 is in range.
            lo32 = np.asarray(lo, np.float32)
            hi32 = np.asarray(hi, np.float32)
            bad = (v < lo32) | (v > hi32)
            if bad.any():
                i = tuple(np.argwhere(bad)[0])
                raise AssertionError(f"{where}: {v[i]!r} at {i} outside [{lo32}, {hi32}]")
            degenerate = lo32 == hi32
            if np.any(degenerate):
                assert np.all(np.where(degenerate, v == lo32, True)), f"{where}: a degenerate range gives {v}"
    assert checked >= {"vx", "wz", "zero_on", "pure_slow_vx", "next_countdown", "push_countdown", "ang", "mass_add"}


# Bounds where lo + (hi - lo) * rand rounds one float32 ulp above hi for the
# largest rand below 1. Found by a search over random bounds.
OVERSHOOT_BOUNDS = (
    (-2.9008341868288254, -2.7541588563828316),
    (-1.1807939240145284, -0.9580568557160016),
    (2.38619061070529, 2.7639893155816297),
)


def test_torch_uniform_stays_in_its_closed_range_at_the_ends(device, monkeypatch):
    """A rand at 0 or just below 1 still gives a command draw in the float32 [lo, hi].

    torch.rand is replaced, so world i of every draw gets rand value i. The
    three overshoot bounds become the vx, vy and wz ranges of a reset.
    """
    ends = torch.tensor([0.0, 1 - 2**-24, 1 - 2**-23], dtype=torch.float32, device=device)

    def rand(shape, **kwargs):
        return ends.reshape(-1, *([1] * (len(shape) - 1))).expand(shape).clone()

    c = cfg({"command": dict(zip(("vx", "vy", "wz"), OVERSHOOT_BOUNDS, strict=True))})
    monkeypatch.setattr(torch, "rand", rand)
    command = TorchDraws(SEED, device).reset(len(ends), c, NU)["command"]
    for name, (lo, hi) in zip(("vx", "vy", "wz"), OVERSHOOT_BOUNDS, strict=True):
        lo32, hi32 = np.float32(lo), np.float32(hi)
        raw = (lo + (hi - lo) * ends).cpu().numpy()
        assert raw.max() > hi32, f"[{lo}, {hi}] no longer overshoots, so this case tests nothing"
        v = command[name].cpu().numpy()
        assert v.dtype == np.float32
        assert v.min() >= lo32 and v.max() <= hi32, f"{name} in [{lo}, {hi}] gives {v!r}"
        assert v[0] == lo32, f"{name}: rand 0 gives {v[0]!r}, want {lo32!r}"


# -- statistics ---------------------------------------------------------------------
def test_draw_statistics_match_their_definitions(device):
    """Every frequency and mean lies within 5 sigma of its definition.

    This catches a flipped comparison such as rand > p, which the range test
    cannot see. The seeds are fixed, so the test is deterministic.
    """
    failures = []
    for backend, s in _sources(N_STATS, device, CFG_ALL, TABLE_ALL):
        for path, v in s.items():
            kind, lo, hi = spec(path, CFG_ALL, TABLE_ALL)
            where = f"{backend} {path}"
            if kind == "key":
                continue
            if kind == "bernoulli":
                p = float(lo)
                freq = v.astype(np.float64).mean()
                sigma = np.sqrt(p * (1 - p) / v.size)
                if abs(freq - p) > 5 * sigma:
                    failures.append(f"{where}: frequency {freq:.4f}, p {p}, sigma {sigma:.4f}")
                continue
            if kind == "randint":
                mean = v.astype(np.float64).mean()
                sigma = np.sqrt(((hi - lo + 1) ** 2 - 1) / 12 / v.size)
                if abs(mean - (lo + hi) / 2) > 5 * sigma:
                    failures.append(f"{where}: mean {mean:.4f}, want {(lo + hi) / 2}, sigma {sigma:.4f}")
                continue
            # One mean per element, over the worlds.
            lo64 = np.broadcast_to(np.asarray(lo, np.float64), v.shape[1:])
            hi64 = np.broadcast_to(np.asarray(hi, np.float64), v.shape[1:])
            mean = v.astype(np.float64).mean(axis=0)
            sigma = (hi64 - lo64) / np.sqrt(12 * v.shape[0])
            off = (hi64 > lo64) & (np.abs(mean - (lo64 + hi64) / 2) > 5 * sigma)
            if off.any():
                i = tuple(np.argwhere(off)[0])
                failures.append(
                    f"{where}: mean {mean[i]:.4f} at {i}, "
                    f"want {(lo64[i] + hi64[i]) / 2}, sigma {sigma[i]:.4f}"
                )
    assert not failures, "\n".join(failures)
