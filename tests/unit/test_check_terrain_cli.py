"""check-terrain without a model: the spawn table, the verdict and exit
rules, the report schema, the budget arithmetic and the run.sh verb.

MJWarp's messages need a CUDA device, so reports are written by hand here.
`test_main_writes_the_schema` stands a namespace in for the env and a
canned result in for the rollouts, so nothing is built.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types

import mujoco
import numpy as np
import pytest

from humanoid_lab import (
    check_contacts,
    check_terrain,
    fd_capture,
    paths,
    registry,
    sim_budget,
)
from humanoid_lab.actuators.models import PositionPD
from humanoid_lab.envs import terrain_geometry as tg
from humanoid_lab.envs.terrain_joystick import default_config as terrain_default_config
from humanoid_lab.registry import _apply_overrides
from humanoid_lab.terrain import TYPES, ArenaParams
from humanoid_lab.terrain.config import (
    CPU_ARENA,
    arena_for,
    config_from_params,
    params_from_config,
)

SCHEMA_KEYS = [
    "schema", "status", "reasons", "engine", "backend", "provenance", "robot", "preset",
    "actuator_overrides", "action_window", "arena", "model", "num_envs", "steps", "seed",
    "compile_s", "budgets", "regimes", "jax_lower_bound", "fill", "messages", "recommend", "ccd_scratch",
    "proxy", "error", "timestamp",
]
REACH = 0.15
CPU_RECIPE = ["experiment=terrain_cpu", "actuators=deploy_pd"]


def _cpu_arena():
    return arena_for(params_from_config({**config_from_params(ArenaParams()), **CPU_ARENA}))


# -- spawns -------------------------------------------------------------------------


STAIRS = ("pyramid_stairs", "inverted_pyramid_stairs")


# On the CPU arena's 4 m tiles every pad is 0.4 m. Stairs tiles reach 1.6 m
# and the rest 1.9 m. The band is [0.4 + reach, 2.0 - reach].
@pytest.mark.parametrize(
    "reach, stairs_radius",
    [
        # Both midpoints, (0.4 + 1.6) / 2 = 1.0 and (0.4 + 1.9) / 2 = 1.15,
        # lie inside [0.55, 1.85].
        (REACH, 1.0),
        # The stairs midpoint 1.0 sits inside the pad plus reach, 1.1, so
        # the band floor wins. 1.15 still lies inside [1.1, 1.3].
        (0.7, 1.1),
    ],
)
def test_spawn_table_is_deterministic_hardest_first_and_covers_every_tile(reach, stairs_radius):
    spec = _cpu_arena().spec
    n = len(spec.tiles)
    table = check_terrain.spawn_table(spec, reach, 2 * n)
    again = check_terrain.spawn_table(spec, reach, 2 * n)
    for a, b in zip(table, again):
        np.testing.assert_array_equal(a, b)

    by_tile = {(t.row, TYPES.index(t.terrain_type)): t for t in spec.tiles}
    first = list(zip(table.level[:n].tolist(), table.type[:n].tolist()))
    assert sorted(first) == sorted(by_tile)
    difficulty = [by_tile[k].difficulty for k in first]
    assert difficulty == sorted(difficulty, reverse=True)

    for i, key in enumerate(first):
        t = by_tile[key]
        offset = table.xy[i] - np.asarray(t.origin[:2])
        if t.row == 0:
            # The flat row has no feature band: its worlds take the pad.
            assert table.kind[i] == tg.SPAWN_PAD
            np.testing.assert_allclose(offset, 0.0, atol=1e-6)
            continue
        assert table.kind[i] == tg.SPAWN_FEATURE
        radius = stairs_radius if t.terrain_type in STAIRS else 1.15
        assert np.abs(offset).max() == pytest.approx(radius, abs=1e-5)
        heading = math.atan2(offset[1], offset[0]) % (2 * math.pi)
        assert heading == pytest.approx(float(table.yaw[i]), abs=1e-5)

    # World i takes heading (i + i // n_tiles) mod 8. The second lap visits
    # the same tiles, each one heading further round.
    np.testing.assert_array_equal(table.level[n:], table.level[:n])
    i = np.arange(2 * n)
    headings = check_terrain.HEADINGS
    np.testing.assert_allclose(table.yaw, 2 * math.pi * ((i + i // n) % headings) / headings, atol=1e-6)


def test_spawn_table_refuses_feet_wider_than_the_feature_band():
    with pytest.raises(ValueError, match="no feature band"):
        check_terrain.spawn_table(_cpu_arena().spec, 1.0, 16)


@pytest.mark.parametrize("arena", ["eval", "train"])
def test_three_laps_give_every_tile_every_fallen_attitude(arena):
    """The eval arena has 48 tiles, a multiple of 3, and the default 80.
    World i takes attitude (i mod n + i // n) mod 3, so each lap moves a
    tile on by one. Three laps cover all three attitudes on every tile, and
    24 laps every heading and attitude pair."""
    spec = check_terrain.effective_arena({}, check_terrain.arena_block(arena)).spec
    n = len(spec.tiles)
    assert n == {"eval": 48, "train": 80}[arena]
    n_attitudes = len(check_contacts.FALLEN_ATTITUDES)
    table = check_terrain.spawn_table(spec, REACH, n_attitudes * n)
    i = np.arange(n_attitudes * n)
    np.testing.assert_array_equal(table.attitude, (i % n + i // n) % n_attitudes)
    for k in range(n):
        assert set(table.attitude[k::n].tolist()) == set(range(n_attitudes)), k

    laps = check_terrain.HEADINGS * n_attitudes
    table = check_terrain.spawn_table(spec, REACH, laps * n)
    for k in range(n):
        pairs = set(zip(np.round(table.yaw[k::n], 4).tolist(), table.attitude[k::n].tolist()))
        assert len(pairs) == laps, k


# -- regimes ------------------------------------------------------------------------


def test_walk_is_check_contacts_antiphase_sinusoid_and_the_rest_hold_still():
    env = types.SimpleNamespace(action_size=6, dt=0.02)
    walk = np.asarray(check_terrain.regime_actions(env, "walk", 50))
    assert (walk.dtype, walk.shape) == (np.float32, (50, 6))
    assert np.abs(walk).max() > 0.9
    assert np.ptp(walk[:, 0]) > 1.0
    np.testing.assert_allclose(walk[:, 0], -walk[:, 1], atol=1e-6)
    np.testing.assert_allclose(walk, check_contacts._walk_action(6, np.arange(50)[:, None], 0.02), atol=1e-6)
    for regime in ("stand", "fallen"):
        still = np.asarray(check_terrain.regime_actions(env, regime, 50))
        assert still.dtype == np.float32
        np.testing.assert_array_equal(still, np.zeros((50, 6)))


def test_tilt_quat_turns_about_the_named_axis():
    c = math.cos(math.radians(45))
    np.testing.assert_allclose(check_terrain.tilt_quat("y", 90), [c, 0.0, c, 0.0], atol=1e-12)
    np.testing.assert_allclose(check_terrain.tilt_quat("x", 180), [0.0, 1.0, 0.0, 0.0], atol=1e-12)


# -- verdict ------------------------------------------------------------------------


def _messages(**counts) -> dict:
    return {k: counts.get(k, 0) for k in fd_capture.WARP_MESSAGES}


def _report(engine="mjx", backend="warp", messages=None, fill=(0.5, 0.5), finite=True, at_cap=None):
    regime = {"finite": finite}
    return {
        "engine": engine,
        "backend": backend,
        "regimes": {"stand": regime, "walk": {"finite": True}},
        "messages": _messages() if messages is None else messages,
        "fill": {"pool": fill[0], "rows": fill[1]},
        "proxy": {"cap": 50, "at_cap": at_cap or {}, "max_pair_count": {}},
    }


def test_a_clean_warp_run_passes():
    assert check_terrain.verdict(_report(), 0.9, True, False) == ("pass", [], 0)


@pytest.mark.parametrize("kind", fd_capture.GATING)
def test_verdict_fails_on_any_gating_message(kind):
    status, reasons, code = check_terrain.verdict(_report(messages=_messages(**{kind: 1})), 0.9, True, False)
    assert (status, code) == ("fail", 1)
    assert kind in reasons[0]


def test_epa_horizon_alone_passes():
    """EPA's horizon is fixed in MJWarp and no budget enlarges it, so it is
    reported and never gates."""
    report = _report(messages=_messages(epa_horizon=12))
    assert check_terrain.verdict(report, 0.9, True, False) == ("pass", [], 0)


@pytest.mark.parametrize("fill, name", [((0.9, 0.1), "pool"), ((0.1, 0.9), "rows"), ((1.3, 0.1), "pool")])
def test_verdict_fails_at_the_pool_and_rows_fill_limit(fill, name):
    status, reasons, code = check_terrain.verdict(_report(fill=fill), 0.9, False, False)
    assert (status, code) == ("fail", 1)
    assert reasons == [reasons[0]] and reasons[0].startswith(f"{name} fill")
    assert check_terrain.verdict(_report(fill=(0.899, 0.899)), 0.9, False, False)[0] == "pass"


def test_verdict_fails_on_non_finite_state():
    status, reasons, code = check_terrain.verdict(_report(finite=False), 0.9, True, False)
    assert (status, code) == ("fail", 1)
    assert "stand" in reasons[0]
    # A diverged jax run fails too, unless warp was required.
    assert check_terrain.verdict(_report(backend="jax", finite=False), 0.9, False, False)[::2] == ("fail", 1)


def test_jax_is_unverified_and_require_warp_exits_2():
    """jax prints no message and has no counters, so even a run with full
    buffers on paper cannot pass or fail the gate."""
    report = _report(backend="jax", messages=None, fill=(None, None))
    report["messages"] = None
    status, reasons, code = check_terrain.verdict(report, 0.9, False, False)
    assert (status, code) == ("unverified", 0)
    assert "no live counters" in reasons[0]
    assert check_terrain.verdict(report, 0.9, True, False)[::2] == ("unverified", 2)
    assert check_terrain.verdict(_report(backend="jax", finite=False), 0.9, True, False)[::2] == (
        "unverified",
        2,
    )


def test_proxy_at_risk_exits_zero_unless_strict():
    at_risk = _report(engine="mujoco", backend=None, at_cap={"torso_collision": 3})
    status, reasons, code = check_terrain.verdict(at_risk, 0.9, False, False)
    assert (status, code) == ("proxy_at_risk", 0)
    assert "torso_collision" in reasons[0]
    assert check_terrain.verdict(at_risk, 0.9, False, True)[::2] == ("proxy_at_risk", 1)
    clear = _report(engine="mujoco", backend=None)
    assert check_terrain.verdict(clear, 0.9, False, True) == ("proxy_clear", [], 0)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("at_cap", [None, {"torso_collision": 3}])
def test_a_proxy_world_c_reset_fails_strict_or_not(strict, at_cap):
    """C resets a diverging world to qpos0, so its counts stop there. The
    proxy then never clears, and a cap hit stays among the reasons."""
    diverged = _report(engine="mujoco", backend=None, finite=False, at_cap=at_cap)
    status, reasons, code = check_terrain.verdict(diverged, 0.9, False, strict)
    assert (status, code) == ("fail", 1)
    assert "C reset" in reasons[0] and "stand" in reasons[0]
    assert len(reasons) == (2 if at_cap else 1)
    if at_cap:
        assert "torso_collision" in reasons[1]


@pytest.mark.parametrize("engine, backend", [("mjx", "warp"), ("mjx", "jax"), ("mujoco", None)])
def test_a_run_that_rolled_nothing_never_passes(engine, backend):
    report = _report(engine=engine, backend=backend)
    report["regimes"] = {}
    assert check_terrain.verdict(report, 0.9, False, False) == ("error", ["no regime ran"], 2)


# -- regime results -----------------------------------------------------------------

# First lines of three MJWarp messages, as the device prints them.
CCD_LINE = "CCD overflow - please increase naccdmax to 4097"
NEFC_LINE = "nefc overflow - please increase njmax to 700"
EPA_LINE = "Warning: EPA horizon = 24 isn't large enough."


def _captured(*text) -> list[str]:
    """The lines capture_fd1 collects when `text` goes to fd 1."""
    with fd_capture.capture_fd1(True) as lines:
        for line in text:
            os.write(1, (line + "\n").encode())
    return lines


def _warp_out() -> dict:
    """A 3-step warp rollout's per-step outputs."""
    return {
        "nacon": np.array([5, 9, 7]),
        "ncollision": np.array([4, 6, 12]),
        "nefc": np.array([10, 30, 20]),
        "finite": np.array([True, True, True]),
    }


def _warp_verdict(regime) -> tuple[str, list[str], int]:
    report = _report(messages=check_terrain._sum_messages({"stand": regime}))
    report["regimes"] = {"stand": regime}
    return check_terrain.verdict(report, 0.9, True, False)


def test_a_warp_regime_counts_captured_messages_and_reads_the_pool_peaks():
    """The captured lines become the regime's message counts, and the pool
    counters its peaks. The peak step is the larger pool counter's, here
    neither nacon's nor nefc's. A captured CCD overflow then fails the gate,
    and an EPA line alone passes it."""
    r = check_terrain._regime_result(_warp_out(), True, _captured(CCD_LINE, NEFC_LINE, EPA_LINE), 0.5, 3, 4)
    assert (r["nacon_pool_max"], r["ncollision_pool_max"], r["nefc_max"]) == (9, 12, 30)
    assert (r["active_max"], r["peak_step"], r["finite"]) == (None, 2, True)
    assert r["messages"] == {
        **dict.fromkeys(fd_capture.WARP_MESSAGES, 0), "ccd_overflow": 1, "nefc_overflow": 1, "epa_horizon": 1
    }
    assert (r["steady_s"], r["env_steps_per_s"]) == (0.5, 24.0)
    status, reasons, code = _warp_verdict(r)
    assert (status, code) == ("fail", 1)
    assert "ccd_overflow" in reasons[0]

    epa = check_terrain._regime_result(_warp_out(), True, _captured(EPA_LINE), 0.5, 3, 4)
    assert _warp_verdict(epa) == ("pass", [], 0)


def test_a_jax_regime_reads_the_active_counts_and_no_messages():
    out = {"active": np.array([3, 8, 5]), "finite": np.array([True, True, True])}
    r = check_terrain._regime_result(out, False, [], 0.5, 3, 4)
    assert (r["messages"], r["active_max"], r["peak_step"], r["finite"]) == (None, 8, 1, True)
    assert (r["nacon_pool_max"], r["ncollision_pool_max"], r["nefc_max"]) == (None, None, None)
    out["finite"] = np.array([True, False, True])
    assert check_terrain._regime_result(out, False, [], 0.5, 3, 4)["finite"] is False


@pytest.mark.parametrize(
    "substeps, record",
    [
        (
            {"nacon": [1, 9, 2], "ncollision": [0, 7, 3], "nefc": [4, 11, 5], "finite": [True, False, True]},
            {"nacon": 9, "ncollision": 7, "nefc": 11, "finite": False},
        ),
        ({"active": [2, 6, 1], "finite": [True, True, True]}, {"active": 6, "finite": True}),
    ],
)
def test_a_control_step_keeps_its_worst_physics_step(substeps, record):
    """Every peak and the one non-finite step sit mid-control-step, where
    reading the first or the last physics step misses them. Training
    telemetry samples the last."""
    out = check_terrain.worst_substep({k: np.asarray(v) for k, v in substeps.items()})
    assert {k: np.asarray(v).item() for k, v in out.items()} == record


# -- configuration ------------------------------------------------------------------


def _args(**kw):
    base = {
        "naconmax_per_env": check_terrain.DEFAULT_NACONMAX_PER_ENV,
        "naccdmax_per_env": None,
        "njmax": check_terrain.DEFAULT_NJMAX,
    }
    return types.SimpleNamespace(**{**base, **kw})


def test_gate_budgets_take_the_recipe_and_fall_back_to_the_flags():
    cfg = {"task": {"env": {"sim": {"naconmax_per_env": 160, "njmax": None}}}}
    budgets = check_terrain.gate_budgets(cfg, _args(naccdmax_per_env=96))
    assert budgets["naconmax_per_env"] == 160
    assert budgets["njmax"] == check_terrain.DEFAULT_NJMAX
    assert budgets["naccdmax_per_env"] == 96
    assert budgets["source"] == {"naconmax_per_env": "recipe", "naccdmax_per_env": "flag", "njmax": "flag"}
    assert check_terrain.gate_budgets({"task": {"env": {}}}, _args())["naccdmax_per_env"] is None


def test_check_overrides_apply_to_the_terrain_default_config():
    """Every override the gate adds names a real key of the terrain config,
    with a value its type accepts: resample_steps is an int field and
    refuses a float. A swapped-in arena replaces the recipe's whole."""
    budgets = {"naconmax_per_env": 256, "naccdmax_per_env": 128, "njmax": 2048}
    recipe = {"terrain": {"arena": dict(CPU_ARENA), "spawn": {"mode": "feature"}}, "push": {"vel": 0.3}}
    overrides = check_terrain.check_overrides(recipe, backend="warp", num_envs=64, budgets=budgets)
    cfg = terrain_default_config()
    _apply_overrides(cfg, overrides)
    assert cfg.command.resample_steps == check_terrain.RESAMPLE_NEVER
    assert isinstance(cfg.command.resample_steps, int)
    assert cfg.push.enable is False and cfg.push.vel == 0.3
    assert cfg.no_progress.enable is False
    assert cfg.terrain.spawn.grace_sec == 0.0
    assert cfg.terrain.spawn.mode == "feature"
    assert cfg.terrain.arena.difficulties == [0.5]
    assert (cfg.sim.backend, cfg.sim.num_envs) == ("warp", 64)
    assert (cfg.sim.naconmax_per_env, cfg.sim.naccdmax_per_env, cfg.sim.njmax) == (256, 128, 2048)
    # The recipe's overrides are left as they were.
    assert recipe["push"] == {"vel": 0.3}

    swapped = config_from_params(ArenaParams(difficulties=(0.2, 0.4)))
    overrides = check_terrain.check_overrides(
        recipe, backend="jax", num_envs=8, budgets=budgets, arena=swapped
    )
    cfg = terrain_default_config()
    _apply_overrides(cfg, overrides)
    assert params_from_config(cfg.terrain.arena.to_dict()) == ArenaParams(difficulties=(0.2, 0.4))


def test_a_missing_eval_arena_is_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "humanoid_lab.eval.terrain_suite", None)
    with pytest.raises(check_terrain.CheckRefused, match="eval_arena_params"):
        check_terrain.arena_block("eval")
    assert check_terrain.arena_block("train") is None


@pytest.mark.parametrize("arena", ["train", "eval"])
def test_build_check_env_hands_make_env_the_recipe_and_the_gate(arena, monkeypatch):
    """make_env gets the recipe's task, robot, preset and actuator
    overrides, the gate's backend, world count and budgets, and the arena
    --arena names. Train keeps the recipe's arena block. Eval replaces it
    with the terrain scan's."""
    cfg = check_terrain.compose(
        [*CPU_RECIPE, "++task.env.sim.naconmax_per_env=48", "++actuators.overrides.groups.hip.kp=50"]
    )
    args = check_terrain.parser().parse_args(["--njmax", "2048"])
    args.backend, args.num_envs, args.arena_block = "warp", 64, check_terrain.arena_block(arena)
    seen = {}

    def record(task, robot_dir, preset, overrides, actuator_overrides):
        seen.update(task=task, robot_dir=robot_dir, preset=preset, overrides=overrides,
                    actuator_overrides=actuator_overrides)
        return "env"

    monkeypatch.setattr(registry, "make_env", record)
    assert check_terrain.build_check_env(cfg, args) == "env"

    ea = registry.env_args_from_config(cfg)
    assert (seen["task"], seen["robot_dir"], seen["preset"]) == ("terrain", ea.robot_dir, "deploy_pd")
    assert seen["actuator_overrides"] == {"groups": {"hip": {"kp": 50}}} == ea.actuator_overrides
    sim = seen["overrides"]["sim"]
    assert (sim["backend"], sim["num_envs"]) == ("warp", 64)
    # The recipe's contact budget, and the flags where the recipe sets none.
    assert (sim["naconmax_per_env"], sim["naccdmax_per_env"], sim["njmax"]) == (48, None, 2048)
    assert seen["overrides"]["push"]["enable"] is False
    assert seen["overrides"]["terrain"]["spawn"]["grace_sec"] == 0.0

    recipe_arena = ea.env_overrides["terrain"]["arena"]
    built = seen["overrides"]["terrain"]["arena"]
    tiles = len(check_terrain.effective_arena(seen["overrides"], None).spec.tiles)
    if arena == "train":
        assert built == recipe_arena
        assert tiles == len(_cpu_arena().spec.tiles)
    else:
        assert built == check_terrain.arena_block("eval")
        # Read as the env reads it, the arena is the scan's, not the recipe's.
        assert tiles != len(_cpu_arena().spec.tiles)
        assert tiles == len(check_terrain.effective_arena({}, check_terrain.arena_block("eval")).spec.tiles)


# -- recommendations ----------------------------------------------------------------


def test_recommendation_divides_the_pool_peak_by_the_worlds():
    """The pool counters count the whole batch. 8000 contacts over 100
    worlds is 80 per world, and twice that is the budget. Rows are already
    per world."""
    rec = check_terrain.recommend(8000, 6000, 300, 100, 29, False)
    assert rec["naconmax_per_env"] == sim_budget.recommend_budget(80, 2.0, 8) == 160
    assert rec["naccdmax_per_env"] == sim_budget.recommend_budget(60, 2.0, 8) == 120
    assert rec["njmax"] == sim_budget.recommend_budget(300, 2.0, 32) == 608
    assert rec["floor_4x_colliders"] == 116
    assert rec["lower_bound"] is False
    # Broadphase candidates share the pool, so the larger counter sizes it.
    assert check_terrain.recommend(6000, 8001, 300, 100, 29, False)["naconmax_per_env"] == (
        sim_budget.recommend_budget(81, 2.0, 8)
    )


def test_naccdmax_recommendation_never_exceeds_naconmax():
    rec = check_terrain.recommend(8000, 8000, 300, 100, 29, False, ccd_headroom=4.0)
    assert rec["naccdmax_per_env"] == rec["naconmax_per_env"] == 160
    rec = check_terrain.recommend(100, 8000, 300, 100, 29, False)
    assert rec["naccdmax_per_env"] <= rec["naconmax_per_env"]


def test_nothing_measured_recommends_nothing():
    rec = check_terrain.recommend(None, None, None, 16, 29, None)
    assert (rec["naconmax_per_env"], rec["naccdmax_per_env"], rec["njmax"]) == (None, None, None)
    assert rec["floor_4x_colliders"] == 116


def test_ccd_scratch_projects_the_training_batch():
    budgets = {"naconmax_per_env": 512, "naccdmax_per_env": None}
    block = check_terrain.ccd_scratch(5480, 8192, 128, budgets)
    assert block["slots_at_train"] == 128 * 8192
    assert block["bytes_at_train"] == 128 * 8192 * 5480
    assert block["source"] == "recommend"
    assert check_terrain.ccd_scratch(5480, 8192, None, budgets)["naccdmax_per_env"] == 512
    budgets["naccdmax_per_env"] = 64
    assert check_terrain.ccd_scratch(5480, 8192, None, budgets)["source"] == "budgets.naccdmax_per_env"


@pytest.mark.parametrize(
    "cfg, expected",
    [
        # The eval batch can be the larger one.
        ({"task": {"ppo": {}}, "ppo": {"num_envs": 64, "num_eval_envs": 128}}, 128),
        ({"task": {"ppo": {}}, "ppo": {"num_envs": 8192, "num_eval_envs": 128}}, 8192),
        # Every shipped config sets neither, so the default num_envs sizes it.
        ({"task": {"ppo": {}}, "ppo": {}}, 8192),
        # The top-level ppo group beats the task's, as train.py applies them.
        ({"task": {"ppo": {"num_envs": 32}}, "ppo": {"num_envs": 256}}, 256),
    ],
)
def test_train_num_envs_is_the_larger_training_batch(cfg, expected):
    assert check_terrain.train_num_envs(cfg) == expected


# -- main ---------------------------------------------------------------------------


def _fake_env(arena, backend="warp"):
    """What the report reads off an env, without a model: 11 ground geoms,
    a heightfield and 10 boxes as on a stairs arena, then 15 robot boxes
    and 14 robot capsules."""
    n_geom = 40
    box, capsule = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    hfield = int(mujoco.mjtGeom.mjGEOM_HFIELD)
    return types.SimpleNamespace(
        _backend=backend,
        _arena=arena,
        _spawn_owner=np.arange(11, n_geom),
        _feet_reach=REACH,
        _n_ground_colliders=29,
        _jax_contact_cap=116,
        _ccd_slot_bytes=5480,
        _actuator_model=PositionPD(),
        _default_pose=np.array([0.1, -0.2]),
        _action_scale=np.array([0.5, 0.5]),
        _ctrl_lo=np.array([-1.0, -0.3]),
        _ctrl_hi=np.array([1.0, 1.0]),
        robot_spec=types.SimpleNamespace(actuated_joints=("hip", "knee")),
        mj_model=types.SimpleNamespace(
            ngeom=n_geom,
            geom_bodyid=np.array([0] * 11 + [1] * 29),
            geom_type=np.array([hfield] + [box] * 10 + [box] * 15 + [capsule] * 14),
            geom_contype=np.ones(n_geom, int),
            geom_conaffinity=np.ones(n_geom, int),
            geom_condim=np.full(n_geom, 4),
            opt=types.SimpleNamespace(cone=sim_budget.CONE_PYRAMIDAL),
        ),
    )


def _fake_regime(nacon, ncollision, nefc, **messages):
    return {
        "nacon_pool_max": nacon,
        "ncollision_pool_max": ncollision,
        "nefc_max": nefc,
        "active_max": None,
        "peak_step": 3,
        "finite": True,
        "messages": _messages(**messages),
        "steady_s": 0.5,
        "env_steps_per_s": 1000.0,
    }


def test_main_writes_the_schema(tmp_path, monkeypatch):
    arena = _cpu_arena()
    seen = {}

    def build(cfg, args):
        seen["num_envs"] = args.num_envs
        seen["task"] = cfg["task"]["name"]
        return _fake_env(arena)

    def rollouts(env, spawns, regimes, steps, seed):
        seen["spawns"] = spawns
        seen["regimes"] = list(regimes)
        seen["steps"], seen["seed"] = steps, seed
        return {
            "stand": _fake_regime(2000, 3000, 400),
            "walk": _fake_regime(2400, 2800, 500, epa_horizon=2),
            "fallen": _fake_regime(1000, 1200, 300),
        }, 12.5

    monkeypatch.setattr(check_terrain, "build_check_env", build)
    monkeypatch.setattr(check_terrain, "run_mjx", rollouts)
    out = tmp_path / "report.json"
    code = check_terrain.main(
        [
            "experiment=terrain_cpu", "actuators=deploy_pd",
            "--num-envs", "64", "--steps", "7", "--seed", "3", "--out", str(out),
        ]
    )
    report = json.loads(out.read_text())
    assert list(report) == SCHEMA_KEYS
    assert (code, report["status"], report["reasons"]) == (0, "pass", [])
    assert seen["task"] == "terrain" and seen["num_envs"] == 64
    assert seen["regimes"] == list(check_terrain.REGIMES)
    assert len(seen["spawns"].xy) == 64
    assert (seen["steps"], seen["seed"]) == (7, 3)

    assert (report["engine"], report["backend"], report["robot"], report["preset"]) == (
        "mjx", "warp", "roboto_origin", "deploy_pd"
    )
    assert (report["num_envs"], report["steps"], report["seed"], report["compile_s"]) == (64, 7, 3, 12.5)
    assert report["budgets"] == {
        "naconmax_per_env": 512,
        "naccdmax_per_env": None,
        "njmax": 4096,
        "pool": 512 * 64,
        "source": {"naconmax_per_env": "flag", "naccdmax_per_env": "flag", "njmax": "flag"},
    }
    assert report["arena"]["kind"] == "train" and report["arena"]["n_boxes"] == len(arena.boxes)
    assert report["model"] == {
        "ngeom": 40, "ground_geoms": 11, "robot_colliders": 29, "rows_per_contact": 6,
        "jax_max_contact_points": 116,
    }
    assert report["action_window"] == {"hip": [-0.4, 0.6], "knee": [-0.3, 0.3]}
    assert report["messages"]["epa_horizon"] == 2 and sum(report["messages"].values()) == 2
    assert report["fill"] == {"pool": pytest.approx(3000 / (512 * 64)), "rows": pytest.approx(500 / 4096)}
    assert report["recommend"]["naconmax_per_env"] == sim_budget.recommend_budget(math.ceil(3000 / 64), 2.0, 8)
    assert report["recommend"]["njmax"] == sim_budget.recommend_budget(500, 2.0, 32)
    assert report["recommend"]["lower_bound"] is False
    assert report["jax_lower_bound"] is None and report["proxy"] is None and report["error"] is None
    # terrain_cpu trains 16 envs and evaluates 16.
    assert report["ccd_scratch"]["train_num_envs"] == 16
    assert report["ccd_scratch"]["naccdmax_per_env"] == report["recommend"]["naccdmax_per_env"]
    assert set(report["provenance"]) >= {"git_commit", "git_dirty", "versions"}


RECIPE_BUDGETS = ["++task.env.sim.naconmax_per_env=48", "++task.env.sim.njmax=1024"]


def test_main_gates_against_the_recipe_budgets(tmp_path, monkeypatch):
    """The recipe's budgets size the pool, the fill and the verdict, not
    the flags. 3000 contacts fill 0.977 of a 48 x 64 pool and fail the
    gate. Against the flags' 512 x 64 pool they would fill 0.092 and pass."""
    monkeypatch.setattr(check_terrain, "build_check_env", lambda cfg, args: _fake_env(_cpu_arena()))
    monkeypatch.setattr(check_terrain, "run_mjx", lambda *a: ({"stand": _fake_regime(2400, 3000, 500)}, 1.0))
    out = tmp_path / "r.json"
    code = check_terrain.main(
        [*CPU_RECIPE, *RECIPE_BUDGETS, "--num-envs", "64", "--regimes", "stand", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert report["error"] is None
    assert report["budgets"] == {
        "naconmax_per_env": 48,
        "naccdmax_per_env": None,
        "njmax": 1024,
        "pool": 48 * 64,
        "source": {"naconmax_per_env": "recipe", "naccdmax_per_env": "flag", "njmax": "recipe"},
    }
    assert report["fill"] == {"pool": pytest.approx(3000 / (48 * 64)), "rows": pytest.approx(500 / 1024)}
    assert (code, report["status"]) == (1, "fail")
    assert report["reasons"] == ["pool fill 0.977 reached --max-fill 0.9 of the naconmax pool"]
    assert report["ccd_scratch"]["naccdmax_per_env"] == report["recommend"]["naccdmax_per_env"]


def test_the_proxy_projects_ccd_scratch_at_the_recipe_budget(tmp_path, monkeypatch):
    """The proxy measures no pool and recommends nothing, so the CCD
    projection takes the recipe's contact budget."""
    monkeypatch.setattr(check_terrain, "build_check_env", lambda cfg, args: _fake_env(_cpu_arena()))
    monkeypatch.setattr(check_terrain, "run_proxy", lambda env, spawns, regimes, steps: _proxy_regimes(regimes))
    out = tmp_path / "r.json"
    code = check_terrain.main(
        [*CPU_RECIPE, *RECIPE_BUDGETS, "--engine", "mujoco", "--num-envs", "3", "--regimes", "stand",
         "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert (code, report["error"]) == (0, None)
    scratch = report["ccd_scratch"]
    assert (scratch["naccdmax_per_env"], scratch["source"]) == (48, "budgets.naconmax_per_env")
    # terrain_cpu trains 16 envs, and the fake env's slot is 5480 bytes.
    assert scratch["bytes_at_train"] == 48 * 16 * 5480


def _usage_commands() -> list[list[str]]:
    """The module docstring's ./run.sh lines as argument lists, with their
    continuations joined and their comments dropped."""
    block = check_terrain.__doc__.split("\n\n")[1]
    text = "\n".join(line.split("#")[0] for line in block.splitlines()).replace("\\\n", " ")
    return [line.split() for line in text.splitlines() if line.strip()]


def test_the_docstring_usage_composes():
    """No yaml file declares a `sim` block, so a budget override needs `++`.
    Hydra refuses the plain form, and the gate then exits 1."""
    commands = _usage_commands()
    assert len(commands) == 3
    sims = []
    for command in commands:
        assert command[:2] == ["./run.sh", "check-terrain"]
        _, overrides = check_terrain.parser().parse_known_args([a.replace("=N", "=64") for a in command[2:]])
        assert not [a for a in overrides if a.startswith("-")]
        cfg = check_terrain.compose(overrides)
        assert cfg["task"]["name"] == "terrain"
        sims.append((cfg["task"].get("env") or {}).get("sim"))
    assert sims[0] == {"naconmax_per_env": 64, "njmax": 64}


def _proxy_regimes(regimes) -> dict:
    return {
        r: {"finite": True, "diverged": [], "at_cap": {}, "max_pair_count": {"foot": 3}, "steady_s": 0.1}
        for r in regimes
    }


def test_main_passes_steps_to_the_proxy(tmp_path, monkeypatch):
    seen = {}

    def proxy(env, spawns, regimes, steps):
        seen["steps"] = steps
        return _proxy_regimes(regimes)

    monkeypatch.setattr(check_terrain, "build_check_env", lambda cfg, args: _fake_env(_cpu_arena()))
    monkeypatch.setattr(check_terrain, "run_proxy", proxy)
    out = tmp_path / "r.json"
    code = check_terrain.main(
        ["experiment=terrain_cpu", "--engine", "mujoco", "--num-envs", "3", "--steps", "9", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert report["error"] is None
    assert (code, report["status"]) == (0, "proxy_clear")
    assert seen["steps"] == report["steps"] == 9


@pytest.mark.parametrize("arena", ["train", "eval"])
def test_the_proxy_defaults_to_one_world_per_tile(arena, tmp_path, monkeypatch):
    """Without --num-envs the proxy runs one world per tile of the arena
    the env builds: the recipe's, or the terrain scan's with --arena eval."""
    seen = {}

    def build(cfg, args):
        seen["num_envs"] = args.num_envs
        overrides = registry.env_args_from_config(cfg).env_overrides
        return _fake_env(check_terrain.effective_arena(overrides, args.arena_block))

    def proxy(env, spawns, regimes, steps):
        seen["worlds"] = len(spawns.xy)
        return _proxy_regimes(regimes)

    monkeypatch.setattr(check_terrain, "build_check_env", build)
    monkeypatch.setattr(check_terrain, "run_proxy", proxy)
    out = tmp_path / "r.json"
    code = check_terrain.main(
        ["experiment=terrain_cpu", "--engine", "mujoco", "--arena", arena, "--regimes", "stand", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert (code, report["error"]) == (0, None)
    if arena == "train":
        tiles = len(_cpu_arena().spec.tiles)
    else:
        tiles = len(check_terrain.effective_arena({}, check_terrain.arena_block("eval")).spec.tiles)
    # The default arena's tile count, so the recipe's arena is the one read.
    assert tiles != len(check_terrain.effective_arena({}, None).spec.tiles)
    assert seen["num_envs"] == seen["worlds"] == report["num_envs"] == tiles


def test_the_default_report_path_names_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(check_terrain, "build_check_env", lambda cfg, args: _fake_env(_cpu_arena()))
    monkeypatch.setattr(check_terrain, "run_mjx", lambda *a: ({"stand": _fake_regime(100, 200, 50)}, 1.0))
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    code = check_terrain.main(
        ["experiment=terrain_cpu", "actuators=deploy_pd", "--num-envs", "8", "--regimes", "stand", "--arena", "train"]
    )
    assert code == 0
    files = list((tmp_path / "check_terrain").iterdir())
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    fp = report["arena"]["fingerprint"][:12]
    assert len(fp) == 12
    assert files[0].name == f"roboto_origin_deploy_pd_train_{fp}_mjx-warp.json"


def test_default_out_names_the_engine_and_unset_fields():
    report = {"engine": "mujoco", "backend": None, "robot": "r", "preset": "p", "arena": {"fingerprint": "f" * 40}}
    assert check_terrain.default_out(report, "eval").endswith(f"/r_p_eval_{'f' * 12}_mujoco.json")
    # A refused run has none of them yet.
    unset = {"engine": "mjx", "backend": None, "robot": None, "preset": None, "arena": None}
    assert check_terrain.default_out(unset, "eval").endswith("/none_none_eval_none_mjx-none.json")


@pytest.mark.parametrize("kind", fd_capture.GATING)
def test_main_marks_only_an_undercounting_overflow_as_a_lower_bound(kind, tmp_path, monkeypatch):
    """A dropped CCD slot, candidate pair or contact never reaches the
    later counters, so their peaks undercount. A row or nnz overflow is
    counted in full. A heightfield overflow is MJWarp's fixed per-pair cap,
    which no budget enlarges."""
    arena = _cpu_arena()
    monkeypatch.setattr(check_terrain, "build_check_env", lambda cfg, args: _fake_env(arena))
    monkeypatch.setattr(
        check_terrain,
        "run_mjx",
        lambda *a: ({"stand": _fake_regime(100, 200, 50, **{kind: 4})}, 1.0),
    )
    out = tmp_path / "r.json"
    code = check_terrain.main(["experiment=terrain_cpu", "--num-envs", "8", "--regimes", "stand", "--out", str(out)])
    report = json.loads(out.read_text())
    assert (code, report["status"]) == (1, "fail")
    undercounting = {"ccd_overflow", "broadphase_overflow", "narrowphase_overflow"}
    assert report["recommend"]["lower_bound"] is (kind in undercounting)


def test_require_warp_on_jax_exits_2(tmp_path, monkeypatch):
    """--require-warp reaches the verdict through main on the mjx engine."""
    env = _fake_env(_cpu_arena(), backend="jax")
    monkeypatch.setattr(check_terrain, "build_check_env", lambda cfg, args: env)
    regime = {**_fake_regime(None, None, None), "messages": None, "active_max": 12}
    monkeypatch.setattr(check_terrain, "run_mjx", lambda *a: ({"stand": regime}, 1.0))
    out = tmp_path / "r.json"
    code = check_terrain.main(
        ["experiment=terrain_cpu", "--num-envs", "4", "--regimes", "stand", "--require-warp", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert (code, report["status"]) == (2, "unverified")
    assert "--require-warp" in report["reasons"][0]
    assert report["jax_lower_bound"] == {"box_colliders": True, "capped_at": 116}


GEOM = mujoco.mjtGeom
# Spawn points per robot collider: a box's 8 corners, a capsule's samples
# along its axis, a sphere's centre.
SPAWN_POINTS = {GEOM.mjGEOM_BOX: 8, GEOM.mjGEOM_CAPSULE: len(tg.SPAWN_CAPSULE_POINTS), GEOM.mjGEOM_SPHERE: 1}


@pytest.mark.parametrize(
    "robot, expected",
    [
        # Capsules only, as on Asimov. The arena's stair boxes are ground
        # geoms and do not count.
        ((GEOM.mjGEOM_CAPSULE, GEOM.mjGEOM_CAPSULE, GEOM.mjGEOM_SPHERE), False),
        ((GEOM.mjGEOM_CAPSULE, GEOM.mjGEOM_BOX, GEOM.mjGEOM_CAPSULE), True),
        ((GEOM.mjGEOM_BOX,), True),
    ],
)
def test_box_colliders_are_read_off_the_robot_colliders_only(robot, expected):
    """The ground is a heightfield and 10 boxes. `_spawn_owner` names each
    robot collider once per spawn point, so it repeats."""
    ground = [GEOM.mjGEOM_HFIELD] + [GEOM.mjGEOM_BOX] * 10
    owner = np.concatenate([np.full(SPAWN_POINTS[t], len(ground) + i) for i, t in enumerate(robot)])
    env = types.SimpleNamespace(
        mj_model=types.SimpleNamespace(geom_type=np.array([int(t) for t in (*ground, *robot)])),
        _spawn_owner=owner,
    )
    assert check_terrain.has_box_colliders(env) is expected


def test_a_non_terrain_task_exits_2(tmp_path):
    out = tmp_path / "r.json"
    assert check_terrain.main(["task=joystick", "--out", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["status"] == "error"
    assert "joystick" in report["error"]
    assert report["regimes"] == {}


def test_a_box_arena_on_jax_exits_2(tmp_path, monkeypatch):
    """jax runs narrowphase on every robot-box pair, so the env refuses
    a large box arena there. The gate says so before it rolls anything."""
    env = _fake_env(_cpu_arena(), backend="jax")
    env._n_ground_boxes = 1904
    monkeypatch.setattr(check_terrain, "build_check_env", lambda cfg, args: env)
    out = tmp_path / "r.json"
    assert check_terrain.main(["--num-envs", "4", "--out", str(out)]) == 2
    assert "--engine mujoco" in json.loads(out.read_text())["error"]


def test_require_warp_with_the_proxy_exits_2(tmp_path):
    out = tmp_path / "r.json"
    assert check_terrain.main(["--engine", "mujoco", "--require-warp", "--out", str(out)]) == 2
    assert json.loads(out.read_text())["status"] == "error"


@pytest.mark.parametrize("engine", [[], ["--engine", "mujoco"]])
def test_an_unknown_flag_exits_2(engine, tmp_path):
    """A misspelt flag is refused, not handed to Hydra as an override.
    It is no prefix of a real flag, so argparse cannot expand it."""
    out = tmp_path / "r.json"
    assert check_terrain.main(["--requre-warp", *engine, "--out", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["status"] == "error"
    assert "--requre-warp" in report["error"]
    # Refused before Hydra composes anything.
    assert report["robot"] is None


@pytest.mark.parametrize("engine", [[], ["--engine", "mujoco", "--strict"]])
@pytest.mark.parametrize("regimes, error", [("stand,crawl", "crawl"), (",", "no regime"), ("", "no regime")])
def test_an_unknown_or_empty_regime_list_exits_2(regimes, error, engine, tmp_path):
    out = tmp_path / "r.json"
    assert check_terrain.main(["--regimes", regimes, *engine, "--out", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["status"] == "error"
    assert error in report["error"]


def test_regimes_parse_in_order_once_each():
    assert check_terrain.parse_regimes(" walk, stand,walk,") == ("walk", "stand")


@pytest.mark.parametrize("engine", [[], ["--engine", "mujoco"]])
def test_zero_steps_exits_2(engine, tmp_path):
    out = tmp_path / "r.json"
    assert check_terrain.main(["--steps", "0", *engine, "--out", str(out)]) == 2
    assert "--steps" in json.loads(out.read_text())["error"]


def test_an_exception_is_reported_and_exits_1(tmp_path, monkeypatch):
    def boom(cfg, args):
        raise RuntimeError("no device")

    monkeypatch.setattr(check_terrain, "build_check_env", boom)
    out = tmp_path / "r.json"
    assert check_terrain.main(["experiment=terrain_cpu", "--out", str(out)]) == 1
    report = json.loads(out.read_text())
    assert report["status"] == "error"
    assert report["reasons"] == ["RuntimeError: no device"]
    assert "Traceback" in report["error"]


# -- run.sh -------------------------------------------------------------------------


def test_run_sh_does_not_force_cpu_and_names_no_overflow_string():
    """The gate's point is the host's own backend, warp on a GPU. No
    overflow text belongs in run.sh: only a capture of fd 1 sees it."""
    text = (paths.REPO_ROOT / "run.sh").read_text()
    lines = [line for line in text.splitlines() if "humanoid_lab.check_terrain" in line]
    assert len(lines) == 1
    assert lines[0].strip().startswith("check-terrain)")
    assert "JAX_PLATFORMS" not in lines[0]
    assert "check-terrain" in next(line for line in text.splitlines() if "usage: run.sh" in line)
    for message in fd_capture.WARP_MESSAGES.values():
        assert message not in text
