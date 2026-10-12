"""WarpPhysics glue on the CPU warp device.

This file checks the glue only: the model copy, the budgets, the views, the
DR install and the reset and respawn writes. test_parity_physics.py checks
MJX-against-MJWarp physics parity.
"""

from __future__ import annotations

import numpy as np
import pytest
from tolerances import PHYSICS_CUDA

from humanoid_lab import paths, registry
from humanoid_lab.envs import signals

torch = pytest.importorskip("torch")

from humanoid_lab.torch_warp import physics

N = 3
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def env():
    return registry.make_env(
        "joystick", paths.ROBOTS_DIR / "roboto_origin", "deploy_pd", {"sim": {"backend": "jax"}}
    )


@pytest.fixture(scope="module")
def consts(env):
    return env.resolved_constants()


def _physics(env, consts, **kw):
    return physics.WarpPhysics(
        env.mj_model, N, CPU, consts.naconmax_per_env, consts.njmax, env.n_substeps, **kw
    )


@pytest.fixture
def phys(env, consts):
    return _physics(env, consts)


def _reset(phys, consts):
    qpos = torch.tensor(np.asarray(consts.reset_qpos, np.float32)).expand(N, -1)
    qvel = torch.zeros(N, phys.mj_model.nv)
    ctrl = torch.tensor(np.asarray(consts.neutral_ctrl, np.float32)).expand(N, -1)
    phys.reset_worlds(qpos, qvel, ctrl)
    return ctrl


def test_views_cover_the_data_fields(phys):
    m = phys.mj_model
    shapes = {
        "qpos": (N, m.nq),
        "qvel": (N, m.nv),
        "qacc": (N, m.nv),
        "sensordata": (N, m.nsensordata),
        "actuator_force": (N, m.nu),
        "xpos": (N, m.nbody, 3),
        "site_xpos": (N, m.nsite, 3),
        "geom_xpos": (N, m.ngeom, 3),
        "subtree_com": (N, m.nbody, 3),
        "cdof": (N, m.nv, 6),
    }
    assert set(shapes) == set(signals.DATA_FIELDS)
    for f in signals.DATA_FIELDS:
        v = getattr(phys.data, f)
        assert tuple(v.shape) == shapes[f], f
        assert v.dtype == torch.float32, f
    assert tuple(phys.data.ctrl.shape) == (N, m.nu)
    assert tuple(phys.data.time.shape) == (N,)


def test_none_budgets_are_refused(env, consts):
    for nacon, njmax in ((None, consts.njmax), (consts.naconmax_per_env, None)):
        with pytest.raises(ValueError, match="check-contacts"):
            physics.WarpPhysics(env.mj_model, N, CPU, nacon, njmax, env.n_substeps)


def test_foot_priority_is_set_on_a_copy(env, consts):
    before = env.mj_model.geom_priority.copy()
    phys = _physics(env, consts, foot_geom_ids=consts.foot_geom_ids, foot_priority=True)
    np.testing.assert_array_equal(env.mj_model.geom_priority, before)
    assert phys.mj_model is not env.mj_model
    assert (phys.mj_model.geom_priority[np.asarray(consts.foot_geom_ids)] == 1).all()


def test_steps_advance_time_and_stay_finite(phys, consts):
    ctrl = _reset(phys, consts)
    assert (phys.data.time == 0).all()
    dt = phys.n_substeps * float(phys.mj_model.opt.timestep)
    for i in range(1, 6):
        phys.step(ctrl)
        np.testing.assert_allclose(phys.data.time.numpy(), i * dt, rtol=1e-5)
    for f in signals.DATA_FIELDS:
        assert torch.isfinite(getattr(phys.data, f)).all(), f


def test_dr_fields_differ_per_world(phys, consts):
    nbody = phys.mj_model.nbody
    base = phys.base_fields(["body_mass"])["body_mass"]
    scale = torch.tensor([0.9, 1.0, 1.1])[:, None]
    phys.set_dr({"body_mass": base[None, :] * scale})
    got = physics.wp.to_torch(phys.m.body_mass)
    assert tuple(got.shape) == (N, nbody)
    assert not torch.equal(got[0], got[1])
    assert not torch.equal(got[1], got[2])
    torch.testing.assert_close(got[1], base)
    phys.step(_reset(phys, consts))
    assert torch.isfinite(phys.data.qpos).all()


def test_restore_writes_only_the_masked_worlds(phys, consts):
    ctrl = _reset(phys, consts)
    for _ in range(3):
        phys.step(ctrl)
    before = {f: getattr(phys, f).clone() for f in ("qpos", "qvel", "ctrl", "qacc_warmstart", "time")}
    mask = torch.tensor([False, True, False])
    qpos = torch.full_like(phys.qpos, 0.5)
    qvel = torch.full_like(phys.qvel, -0.25)
    new_ctrl = torch.full_like(phys.ctrl, 0.125)
    phys.restore_worlds(mask, qpos, qvel, new_ctrl)
    for f, want in (("qpos", qpos), ("qvel", qvel), ("ctrl", new_ctrl)):
        got = getattr(phys, f)
        torch.testing.assert_close(got[1], want[1])
        torch.testing.assert_close(got[mask.logical_not()], before[f][mask.logical_not()])
    assert (phys.qacc_warmstart[1] == 0).all()
    assert phys.time[1] == 0
    torch.testing.assert_close(phys.time[[0, 2]], before["time"][[0, 2]])
    torch.testing.assert_close(phys.qacc_warmstart[[0, 2]], before["qacc_warmstart"][[0, 2]])


def test_peaks_are_ints(phys, consts):
    ctrl = _reset(phys, consts)
    phys.step(ctrl)
    nacon, nefc = phys.peaks()
    assert type(nacon) is int and type(nefc) is int
    assert 0 <= nacon <= consts.naconmax_per_env * N
    assert 0 <= nefc <= consts.njmax


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA torch and a GPU")
def test_cuda_graph_steps_match_cpu(env, consts):
    # Steps 2 to 5 launch the captured step graph. An empty capture would
    # leave the clock and qpos where step 1 put them. The second reset
    # launches the captured forward graph. An empty capture there would
    # leave xpos and sensordata from before the reset.
    cuda = torch.device("cuda", 0)
    cpu_phys = _physics(env, consts)
    gpu_phys = physics.WarpPhysics(
        env.mj_model, N, cuda, consts.naconmax_per_env, consts.njmax, env.n_substeps
    )
    ctrl = _reset(cpu_phys, consts)
    gpu_phys.reset_worlds(cpu_phys.qpos.to(cuda), cpu_phys.qvel.to(cuda), ctrl.to(cuda))
    dt = cpu_phys.n_substeps * float(cpu_phys.mj_model.opt.timestep)
    for _ in range(5):
        cpu_phys.step(ctrl)
        gpu_phys.step(ctrl.to(cuda))
    torch.cuda.synchronize(cuda)
    np.testing.assert_allclose(gpu_phys.data.time.cpu().numpy(), 5 * dt, rtol=1e-5)
    np.testing.assert_allclose(
        gpu_phys.data.qpos.cpu().numpy(),
        cpu_phys.data.qpos.numpy(),
        atol=PHYSICS_CUDA["qpos_atol"],
    )

    # Raise the base 5 cm above the reset pose, far from the stepped state.
    qpos = torch.tensor(np.asarray(consts.reset_qpos, np.float32)).expand(N, -1).clone()
    qpos[:, 2] += 0.05
    qvel = torch.zeros(N, cpu_phys.mj_model.nv)
    cpu_phys.reset_worlds(qpos, qvel, ctrl)
    gpu_phys.reset_worlds(qpos.to(cuda), qvel.to(cuda), ctrl.to(cuda))
    torch.cuda.synchronize(cuda)
    for field in ("xpos", "sensordata"):
        np.testing.assert_allclose(
            getattr(gpu_phys.data, field).cpu().numpy(),
            getattr(cpu_phys.data, field).numpy(),
            rtol=1e-4,
            atol=PHYSICS_CUDA["qpos_atol"],
            err_msg=field,
        )
