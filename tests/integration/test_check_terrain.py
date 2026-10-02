"""check-terrain end to end on a CPU, and the heightfield contact facts
its report rests on.

The CLI runs Roboto Origin under deploy_pd on the CPU arena: on jax, where
the gate is unverified, and on the C proxy. The hand-built scenes put one
box on a flat 4 cm heightfield, the arena's cell size. The whole base box
is 18.6 x 27.0 cm and one of its chessboard cells 9.3 x 9.0 cm.
"""

from __future__ import annotations

import json
import math

import jax
import jax.numpy as jp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from humanoid_lab import check_contacts, check_terrain, sim_budget
from humanoid_lab.envs import terrain_geometry as tg

CPU_RECIPE = ["experiment=terrain_cpu", "actuators=deploy_pd"]
CELL = 0.04
BASE_BOX = (0.0931, 0.1352, 0.0792)
BASE_CELL = (0.0931 / 2, 0.1352 / 3, 0.0792 / 2)
CAP = mujoco.mjMAXCONPAIR


def test_jax_run_on_the_cpu_arena_reports_every_regime(tmp_path):
    """Built through build_check_env from the composed recipe. jax has no
    counters, so the pool fields stay null and the gate is unverified. Its
    active counts carry the lower-bound qualifier."""
    out = tmp_path / "jax.json"
    code = check_terrain.main(
        [*CPU_RECIPE, "--backend", "jax", "--num-envs", "8", "--steps", "5", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert report["error"] is None
    assert (code, report["status"], report["backend"]) == (0, "unverified", "jax")
    assert set(report["regimes"]) == set(check_terrain.REGIMES)
    for r in report["regimes"].values():
        assert r["finite"]
        assert r["nacon_pool_max"] is None and r["nefc_max"] is None and r["messages"] is None
        assert 0 <= r["active_max"] <= report["model"]["jax_max_contact_points"]
        assert r["steady_s"] > 0
    # Standing robots on stairs and slopes touch the ground from step one.
    assert report["regimes"]["stand"]["active_max"] > 0
    assert report["jax_lower_bound"] == {"box_colliders": True, "capped_at": 116}
    assert report["model"]["robot_colliders"] == 29
    assert report["arena"]["n_boxes"] == 33
    assert report["fill"] == {"pool": None, "rows": None}
    assert report["recommend"]["naconmax_per_env"] is None
    assert report["ccd_scratch"]["bytes_per_slot"] == 5480
    assert report["compile_s"] > 0


def test_a_non_finite_world_fails_the_jax_run(tmp_path, monkeypatch):
    """The rollout's finite flag covers every world, not only the first."""
    start = check_terrain.start_qpos

    def poisoned(env, spawns, regime):
        return start(env, spawns, regime).at[1, env._base_qadr + 2].set(np.nan)

    monkeypatch.setattr(check_terrain, "start_qpos", poisoned)
    out = tmp_path / "nan.json"
    code = check_terrain.main(
        [*CPU_RECIPE, "--backend", "jax", "--num-envs", "2", "--steps", "1", "--regimes", "stand", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert (code, report["status"]) == (1, "fail")
    assert report["regimes"]["stand"]["finite"] is False
    assert "non-finite" in report["reasons"][0]


def test_mujoco_engine_end_to_end_on_the_cpu_arena(tmp_path):
    out = tmp_path / "proxy.json"
    code = check_terrain.main(
        [*CPU_RECIPE, "--engine", "mujoco", "--num-envs", "3", "--steps", "20", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert report["error"] is None
    assert (code, report["status"], report["engine"], report["backend"]) == (0, "proxy_clear", "mujoco", None)
    assert report["num_envs"] == 3
    assert report["proxy"]["cap"] == CAP and report["proxy"]["at_cap"] == {}
    for r in report["regimes"].values():
        assert r["finite"]
        assert all(0 < n < CAP for n in r["max_pair_count"].values())
    # The standing robots hold their feet on the heightfield.
    assert any("foot" in name for name in report["regimes"]["stand"]["max_pair_count"])


@pytest.fixture(scope="module")
def cpu_env():
    cfg = check_terrain.compose(CPU_RECIPE)
    args = check_terrain.parser().parse_args([])
    args.backend, args.num_envs, args.arena_block = "jax", 16, None
    return check_terrain.build_check_env(cfg, args)


def test_spawns_start_clear_of_the_ground(cpu_env):
    """No world touches anything at its start, standing or fallen. Feature
    spawns take the table's kind, so they read the dilated spawn grid and
    sit no lower than the plain lookup puts them. A fallen start is the
    standing one tilted about the heading's own axis and dropped."""
    env = cpu_env
    spawns = check_terrain.spawn_table(env._arena.spec, env._feet_reach, 16)
    m = env.mj_model
    starts = {r: np.asarray(check_terrain.start_qpos(env, spawns, r)) for r in ("stand", "fallen")}
    for regime, qpos in starts.items():
        for w, q in enumerate(qpos):
            d = mujoco.MjData(m)
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            assert d.ncon == 0, (regime, w)

    b = env._base_qadr
    qs, qf = starts["stand"], starts["fallen"]
    np.testing.assert_allclose(qf[:, b + 2] - qs[:, b + 2], check_contacts.FALLEN_DROP_M, atol=1e-5)
    kept = np.setdiff1d(np.arange(m.nq), np.arange(b + 2, b + 7))
    np.testing.assert_array_equal(qf[:, kept], qs[:, kept])
    # Stand is yaw x reset and fallen yaw x tilt x reset, so their relative
    # rotation is yaw x tilt x yaw^-1: the tilt angle, about the tilt axis
    # turned to the world's heading.
    rel = np.asarray(tg.quat_mul(qf[:, b + 3 : b + 7], qs[:, b + 3 : b + 7] * np.array([1, -1, -1, -1])))
    for w, (q, yaw) in enumerate(zip(rel, spawns.yaw)):
        axis_name, degrees = check_contacts.FALLEN_ATTITUDES[spawns.attitude[w]]
        angle = 2 * math.acos(min(abs(float(q[0])), 1.0))
        assert math.degrees(angle) == pytest.approx(degrees, abs=0.01), w
        want = [-math.sin(yaw), math.cos(yaw), 0.0] if axis_name == "y" else [math.cos(yaw), math.sin(yaw), 0.0]
        axis = q[1:] / np.linalg.norm(q[1:])
        assert abs(float(axis @ np.array(want))) == pytest.approx(1.0, abs=1e-4), w
    # The headings vary, so a tilt about a world axis fails the check above.
    assert len(set(np.round(spawns.yaw, 4))) > 2

    z = qs[:, b + 2]
    pad = spawns._replace(kind=np.zeros_like(spawns.kind))
    z_lookup = np.asarray(check_terrain.start_qpos(env, pad, "stand"))[:, b + 2]
    feature = spawns.kind == tg.SPAWN_FEATURE
    assert np.all(z >= z_lookup - 1e-6)
    assert np.any(z[feature] > z_lookup[feature] + 1e-3)
    np.testing.assert_array_equal(z[spawns.kind == tg.SPAWN_PAD], z_lookup[spawns.kind == tg.SPAWN_PAD])


def test_each_control_step_keeps_its_worst_physics_step(cpu_env, tmp_path, monkeypatch):
    """On the second physics step of each control step, every contact slot
    reads as penetrating. The regime's peak is that step's count. Neither
    the first nor the last physics step shows it."""
    dt = float(cpu_env.mj_model.opt.timestep)
    n_substeps = cpu_env.n_substeps
    assert n_substeps >= 3
    real = sim_budget.contact_dist
    slots = []

    def spiked(data):
        dist = real(data)
        slots.append(dist.shape[-1])
        # Rollouts start at time 0, so after physics step s of a control
        # step the time is s + 1 steps past a multiple of n_substeps.
        step = jp.round(data.time / dt).astype(jp.int32) % n_substeps
        return jp.where((step == 2)[:, None], -1.0, dist)

    monkeypatch.setattr(sim_budget, "contact_dist", spiked)
    out = tmp_path / "spike.json"
    code = check_terrain.main(
        [*CPU_RECIPE, "--backend", "jax", "--num-envs", "2", "--steps", "2", "--regimes", "stand", "--out", str(out)]
    )
    report = json.loads(out.read_text())
    assert (code, report["error"]) == (0, None)
    assert report["regimes"]["stand"]["active_max"] == slots[0]


def test_the_proxy_counts_every_substep_at_the_cap(cpu_env, monkeypatch):
    """A collider whose pair count reaches the cap is counted once per
    physics step, summed over the regimes, and named in the verdict. C
    stops at the cap, so a count equal to it is a hit."""
    env = cpu_env
    m = env.mj_model
    foot = env.robot_spec.foot_geoms[0]
    foot_id = m.geom(foot).id

    def at_cap(model, data, hfield_geom):
        counts = np.zeros(model.ngeom, np.int64)
        counts[foot_id] = CAP
        return counts

    monkeypatch.setattr(check_terrain, "hfield_pair_counts", at_cap)
    spawns = check_terrain.spawn_table(env._arena.spec, env._feet_reach, 1)
    regimes = check_terrain.run_proxy(env, spawns, ["stand", "walk"], 2)
    for r in regimes.values():
        assert r["at_cap"] == {foot: 2 * env.n_substeps}
    proxy = check_terrain.proxy_summary(regimes)
    assert proxy["at_cap"] == {foot: 4 * env.n_substeps}
    assert proxy["max_pair_count"] == {foot: CAP}
    report = {"engine": "mujoco", "regimes": regimes, "proxy": proxy}
    status, reasons, code = check_terrain.verdict(report, 0.9, False, True)
    assert (status, code) == ("proxy_at_risk", 1)
    assert foot in reasons[0]


def test_the_proxy_steps_with_the_regime_ctrl_inside_the_env_clip(cpu_env, monkeypatch):
    """Every physics step of control step i holds the actuator model's
    target for the walk action i, clipped to the env's ctrl range."""
    env = cpu_env
    steps = 5
    seen = []

    def record(model, data, hfield_geom):
        seen.append(data.ctrl.copy())
        return np.zeros(model.ngeom, np.int64)

    monkeypatch.setattr(check_terrain, "hfield_pair_counts", record)
    spawns = check_terrain.spawn_table(env._arena.spec, env._feet_reach, 1)
    check_terrain.run_proxy(env, spawns, ["walk"], steps)
    ctrl = np.stack(seen).reshape(steps, env.n_substeps, -1)

    actions = check_terrain.regime_actions(env, "walk", steps)
    raw = np.asarray(env._actuator_model.ctrl_from_action(actions, env._default_pose, env._action_scale))
    want = np.clip(raw, np.asarray(env._ctrl_lo), np.asarray(env._ctrl_hi))
    np.testing.assert_allclose(ctrl, np.broadcast_to(want[:, None], ctrl.shape), atol=1e-6)
    # The clip binds inside these steps, so a proxy without it fails above.
    assert np.any(raw != want)
    assert not np.allclose(ctrl, np.asarray(env._neutral_ctrl))


def test_a_c_reset_stops_its_world_counting_and_fails_the_proxy(cpu_env, monkeypatch):
    """A NaN in one world's qvel makes C reset that world to qpos0 at its
    next physics step, counting only a bad-qvel warning. The world's counts
    stop there, the regime names it, and the proxy fails."""
    env = cpu_env
    steps, k = 3, env.n_substeps
    real = check_terrain.hfield_pair_counts
    calls, poisoned = [], []
    # World 1's sixth count: control step 1, physics step 1.
    poison_at = steps * k + k + 2

    def count(model, data, hfield_geom):
        calls.append(1)
        if len(calls) == poison_at:
            data.qvel[0] = np.nan
            poisoned.append(data)
        return real(model, data, hfield_geom)

    monkeypatch.setattr(check_terrain, "hfield_pair_counts", count)
    spawns = check_terrain.spawn_table(env._arena.spec, env._feet_reach, 3)
    regimes = check_terrain.run_proxy(env, spawns, ["stand"], steps)
    stand = regimes["stand"]
    assert stand["diverged"] == [[1, 1]] and stand["finite"] is False
    # Worlds 0 and 2 count every physics step. World 1 stops at the reset.
    assert len(calls) == 2 * steps * k + (k + 2)
    warning = poisoned[0].warning
    assert warning[int(mujoco.mjtWarning.mjWARN_BADQVEL)].number == 1
    assert warning[int(mujoco.mjtWarning.mjWARN_BADQPOS)].number == 0
    assert warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number == 0

    report = {"engine": "mujoco", "regimes": regimes, "proxy": check_terrain.proxy_summary(regimes)}
    status, reasons, code = check_terrain.verdict(report, 0.9, False, False)
    assert (status, code) == ("fail", 1)
    assert "C reset" in reasons[0]


def _box_on_heightfield(half, *, yaw_deg=0.0, depth=0.001, x=0.0, y=0.0, strip=None):
    """A box centred at (x, y), resting `depth` into a flat 4 cm
    heightfield, or into a strip raised over [lo, hi] in x. The origin is a
    grid node. Returns (model, box geom id, hfield geom id)."""
    radius = 1.0
    n = round(2 * radius / CELL) + 1
    nodes = np.linspace(-radius, radius, n)
    elevation = 0.5
    data = np.zeros((n, n), np.float32)
    top = 0.0
    if strip is not None:
        lo, hi, top = strip
        data[:, (nodes >= lo) & (nodes <= hi)] = top / elevation
    spec = mujoco.MjSpec()
    spec.add_hfield(name="hf", size=[radius, radius, elevation, 0.1], nrow=n, ncol=n, userdata=data.ravel().tolist())
    spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="hf")
    body = spec.worldbody.add_body(name="body", pos=[x, y, 0.0])
    body.add_freejoint()
    half_yaw = math.radians(yaw_deg) / 2
    body.add_geom(
        name="box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(half),
        quat=[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        pos=[0.0, 0.0, top + half[2] - depth],
    )
    m = spec.compile()
    return m, m.geom("box").id, m.geom("ground").id


def _c_contacts(m):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return d


def test_proxy_flags_a_whole_box_and_clears_a_cell():
    """A box face over many cells collects more prism hits than the cap
    lets through, so C stops at 50 wherever the box sits. The chessboard
    cell stays under it. Its count depends on where it sits on the grid:
    17 to 30 over one grid cell of offsets in x and y, and 30 centred on a
    node."""
    offsets = np.linspace(0.0, CELL, 17)
    counts = {}
    for name, half in (("box", BASE_BOX), ("cell", BASE_CELL)):
        for x in offsets:
            for y in offsets:
                m, box, hf = _box_on_heightfield(half, x=x, y=y)
                pairs = check_terrain.hfield_pair_counts(m, _c_contacts(m), hf)
                assert pairs.sum() == pairs[box]
                counts[name, x, y] = int(pairs[box])
    assert {counts["box", x, y] for x in offsets for y in offsets} == {CAP}
    cell = [counts["cell", x, y] for x in offsets for y in offsets]
    assert (min(cell), max(cell)) == (17, 30)
    assert counts["cell", 0.0, 0.0] == 30


# A cell centre 1 cm past a node line. jax starts the prism subgrid one
# largest half-size (4.66 cm) below the centre. That lands 3.66 cm low,
# inside the cell between -4 cm and 0, so the subgrid starts at the -4 cm
# node. Yawed 45 degrees, the cell's low corner reaches 6.48 cm low, to
# -5.48 cm. The strip under it ends at -4.28 cm, so the corner rests on
# prisms jax never tests.
YAWED_X = 0.01
YAWED_STRIP_OVERLAP = 0.012


def _low_corner_strip(side):
    reach = math.hypot(BASE_CELL[0], BASE_CELL[1])
    if side == "-x":
        return (-2.0, YAWED_X - reach + YAWED_STRIP_OVERLAP, 0.05)
    return (YAWED_X + reach - YAWED_STRIP_OVERLAP, 2.0, 0.05)


def test_a_yawed_cell_on_the_heightfield_gets_contacts_on_every_side():
    """C tests every prism under the yawed cell. Resting flat, it gets
    contacts past the centre on all four sides. Its low corner alone on a
    raised strip gets contacts too."""
    m, box, hf = _box_on_heightfield(BASE_CELL, yaw_deg=45, depth=0.005, x=YAWED_X)
    d = _c_contacts(m)
    rel = d.contact.pos[: d.ncon, :2] - np.array([YAWED_X, 0.0])
    for axis in (0, 1):
        assert (rel[:, axis] < -0.02).any() and (rel[:, axis] > 0.02).any()

    for side in ("-x", "+x"):
        m, box, hf = _box_on_heightfield(
            BASE_CELL, yaw_deg=45, depth=0.005, x=YAWED_X, strip=_low_corner_strip(side)
        )
        assert check_terrain.hfield_pair_counts(m, _c_contacts(m), hf)[box] > 0


def _jax_contacts(m) -> int:
    mx = mjx.put_model(m, impl="jax")
    d = jax.jit(mjx.forward)(mx, mjx.make_data(mx))
    return int((np.asarray(d._impl.contact.dist) < 0).sum())


def test_jax_misses_the_low_side_of_a_yawed_box():
    """Pins the jax miss check-terrain's report qualifies: with the yawed
    cell's -x corner on a strip, jax finds no contact, while the mirrored
    +x corner finds one. An MJX release that fixes the subgrid fails this
    test, and the jax_lower_bound wording should then be revisited."""
    low = _box_on_heightfield(BASE_CELL, yaw_deg=45, depth=0.005, x=YAWED_X, strip=_low_corner_strip("-x"))[0]
    high = _box_on_heightfield(BASE_CELL, yaw_deg=45, depth=0.005, x=YAWED_X, strip=_low_corner_strip("+x"))[0]
    assert _jax_contacts(low) == 0
    assert _jax_contacts(high) > 0
