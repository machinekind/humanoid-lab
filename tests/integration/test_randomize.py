"""Tests for dr/randomize.py: the yaml/code default pin, batching behavior
on the compiled asimov model, and the stream-isolation guards at the bottom
of the file.
"""

import functools

import jax
import jax.numpy as jp
import numpy as np
import pytest
import yaml
from mujoco import mjx
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

from humanoid_lab import paths
from humanoid_lab.dr import randomize
from humanoid_lab.envs.joystick import Joystick, default_config
from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.spec import load_robot_spec

ROBOT_DIR = paths.ROBOTS_DIR / "asimov_v1"


def test_dr_yaml_matches_code_defaults():
    cfg = yaml.safe_load((paths.CONFIGS_DIR / "dr" / "default.yaml").read_text())
    assert cfg == randomize._DEFAULT_DR


@pytest.fixture(scope="module")
def robot_spec():
    return load_robot_spec(ROBOT_DIR)


@pytest.fixture(scope="module")
def mj_model():
    spec = build_spec(ROBOT_DIR, "sizing_ideal")
    return compile_spec(spec)


@pytest.fixture(scope="module")
def mjx_model(mj_model):
    return mjx.put_model(mj_model, impl="jax")


def test_make_domain_randomize_batches_model_for_three_rngs(mj_model, mjx_model, robot_spec):
    randomize_fn = randomize.make_domain_randomize(mj_model, robot_spec)
    rng = jax.random.split(jax.random.PRNGKey(0), 3)

    model_v, in_axes = randomize_fn(mjx_model, rng)

    assert model_v.body_mass.shape[0] == 3
    assert not np.allclose(np.array(model_v.body_mass[0]), np.array(model_v.body_mass[1]))
    assert in_axes.body_mass == 0
    assert in_axes.geom_friction == 0
    assert in_axes.actuator_gainprm == 0
    assert in_axes.actuator_biasprm == 0
    assert in_axes.actuator_forcerange == 0
    assert in_axes.qpos0 is None


def test_all_disabled_reproduces_five_field_dr_shape(mj_model, mjx_model, robot_spec):
    all_disabled = {k: {"enable": False} for k in randomize._DEFAULT_DR}
    randomize_fn = randomize.make_domain_randomize(mj_model, robot_spec, all_disabled)
    rng = jax.random.split(jax.random.PRNGKey(1), 4)

    model_v, in_axes = randomize_fn(mjx_model, rng)

    # Disabled joint_gains broadcasts ONE scalar multiplier to every
    # actuator (asimov's per-joint kp values differ, so the gains
    # themselves don't collapse to one value -- the multiplier ratio does).
    orig = np.array(mjx_model.actuator_gainprm[:, 0])
    scaled = np.array(model_v.actuator_gainprm[0, :, 0])
    ratio = scaled / orig
    assert np.allclose(ratio, ratio[0])
    assert in_axes.body_ipos is None
    assert in_axes.dof_damping is None


def test_foot_friction_uses_robot_spec_foot_geoms(mj_model, mjx_model, robot_spec):
    dr_cfg = {"foot_friction": {"enable": True, "range": [0.5, 0.6]}}
    randomize_fn = randomize.make_domain_randomize(mj_model, robot_spec, dr_cfg)
    rng = jax.random.split(jax.random.PRNGKey(2), 4)

    model_v, _ = randomize_fn(mjx_model, rng)

    foot_ids = np.array([mj_model.geom(n).id for n in robot_spec.foot_geoms])
    friction = np.array(model_v.geom_friction[:, foot_ids, 0])
    assert not np.allclose(friction[0], friction[1])
    priority = np.array(model_v.geom_priority)
    assert np.all(priority[foot_ids] == 1)


@pytest.fixture(scope="module")
def joystick_env():
    cfg = default_config()
    cfg.episode_length = 10  # fast tests, not a training config
    return Joystick(ROBOT_DIR, "sizing_ideal", cfg)


def test_foot_friction_survives_vmap_over_full_model(joystick_env):
    """Regression test for the crash the fix in randomize.py guards against.

    geom_priority is a static (non-pytree-leaf) field on mjx.Model: it's
    aux_data, part of the treedef, not a batched jax.Array. The tests above
    only inspect model_v's leaves and never vmap anything, so they kept
    passing even when foot_friction.enable=true made
    `jax.vmap(reset, in_axes=[in_axes, 0])` die with a pytree-prefix
    mismatch inside brax's training loop. A minimal
    `jax.vmap(lambda m: m.geom_friction.sum(), in_axes=[in_axes])(model_v)`
    is not a faithful reproduction either -- it doesn't exercise the full
    Model pytree the way brax's own vmapped reset/step closures do. Drive
    the (model_v, in_axes) pair through mujoco_playground's actual
    BraxDomainRandomizationVmapWrapper instead, the way training does.
    """
    dr_cfg = {"foot_friction": {"enable": True, "range": [0.8, 1.2]}}
    randomize_fn = randomize.make_domain_randomize(
        joystick_env.mj_model, joystick_env.robot_spec, dr_cfg
    )
    rng = jax.random.split(jax.random.PRNGKey(3), 4)

    # brax's ppo.train partial-applies rng onto randomization_fn before
    # handing it to wrap_for_brax_training (see brax's
    # agents/ppo/train.py: `functools.partial(randomization_fn,
    # rng=randomization_rng)`); BraxDomainRandomizationVmapWrapper then
    # calls it with the model as the sole positional arg. Mirror that here
    # instead of calling randomize_fn(model, rng) directly.
    wrapped = BraxDomainRandomizationVmapWrapper(
        joystick_env, functools.partial(randomize_fn, rng=rng)
    )

    state = jax.jit(wrapped.reset)(rng)
    assert state.data.qpos.shape[0] == 4
    assert jp.all(jp.isfinite(state.obs["state"]))

    action = jp.zeros((4, joystick_env.action_size))
    next_state = jax.jit(wrapped.step)(state, action)
    assert jp.all(jp.isfinite(next_state.obs["state"]))


@pytest.mark.parametrize(
    "dr_cfg",
    [
        None,
        {k: {"enable": False} for k in randomize._DEFAULT_DR},
        {"foot_friction": {"enable": True}},
        {k: {"enable": True} for k in randomize._DEFAULT_DR},
    ],
    ids=["default", "all_disabled", "foot_friction", "all_enabled"],
)
def test_in_axes_is_usable_as_vmap_in_axes(mj_model, mjx_model, robot_spec, dr_cfg):
    """The (model, in_axes) pair make_domain_randomize returns must be
    accepted by jax.vmap under every DR config, foot_friction included.

    The wrapper that consumes
    the pair does exactly one thing -- `jax.vmap(step, in_axes=[in_axes, 0])`
    -- and vmap refuses the pair unless the two line up. Every other test in
    this file inspects individual leaves of the batched model and would keep
    passing while the pair was unusable.

    That is the geom_priority class of bug, and it is not hypothetical:
    `foot_friction.enable=true` also sets `geom_priority`, which mjx stores as
    a plain numpy array and jax therefore counts as part of the model's
    treedef rather than as data. Setting it after in_axes had been derived
    left the two describing different structures and every run with the flag
    on died the moment the env was wrapped -- fatal on the jax backend,
    absent on warp, and shipped twice before it was understood. The fix
    is in randomize.py and the lesson in docs/lessons/asimov_v1.md; this is
    the test that keeps them.
    """
    randomize_fn = randomize.make_domain_randomize(mj_model, robot_spec, dr_cfg)
    rng = jax.random.split(jax.random.PRNGKey(4), 8)
    model_v, in_axes = randomize_fn(mjx_model, rng)

    def read(model, i):
        del i
        return model.geom_friction[:, 0].sum() + model.body_mass.sum()

    per_env = jax.vmap(read, in_axes=[in_axes, 0])(model_v, jp.arange(len(rng)))

    assert per_env.shape == (len(rng),)
    # The batched model really is per-env, not one model broadcast 8 times.
    assert len(np.unique(np.asarray(per_env))) > 1


def test_find_floor_geom_id_finds_asimovs_own_floor(mj_model):
    # asimov's vendored XML ships its own floor plane geom (unlike a robot
    # with no ground plane, which build_model.py's ensure_training_scene
    # would add one for); the auto-detected id must be that geom, not the
    # override path.
    assert randomize._find_floor_geom_id(mj_model, None) == mj_model.geom("floor").id


def test_find_floor_geom_id_errors_with_no_plane_geom():
    import mujoco

    no_plane_xml = """
    <mujoco>
      <worldbody>
        <body name="b"><freejoint/><geom type="sphere" size="0.1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(no_plane_xml)
    with pytest.raises(ValueError, match="floor"):
        randomize._find_floor_geom_id(model, None)


def test_find_base_body_id_finds_the_free_joint_body(mj_model, robot_spec):
    base_id = randomize._find_base_body_id(mj_model)
    # pelvis_link (robot.yaml's first termination_bodies entry) holds
    # asimov's floating_base free joint.
    assert base_id == mj_model.body(robot_spec.termination_bodies[0]).id


# -- stream isolation -------------------------------------------------------
# `jax.random.fold_in(key, i)` is bit-identical to `jax.random.split(key, n)[i]`
# for every i < n. `rand()` splits its own rng five ways and then folds small
# indices into that same rng for the optional fields, so before the 0x100
# offset landed, `fold_in(rng, 2)` WAS r3 (the link-mass key) and
# `fold_in(rng, 4)` WAS r5 (the kd key) -- and both folded keys were used
# directly as sampling keys, not re-split. Two of DR's axes were therefore one
# axis wearing two names. The tests below pin the offset and the decorrelation
# it buys.

ALL_ENABLED = {k: {"enable": True} for k in randomize._DEFAULT_DR}

# 200 keys: enough that a sample correlation of 0 lands well inside +-0.3 and a
# correlation of 1 cannot hide, cheap enough to vmap the whole model over.
CORR_KEYS = jax.random.split(jax.random.PRNGKey(0), 200)


def corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def test_no_dr_draw_key_aliases_a_key_from_the_base_split(
    mj_model, mjx_model, robot_spec, monkeypatch
):
    """Every optional field's fold_in index must sit outside the domain of
    `rand()`'s own `split(rng, 5)`, so no field can key off a base draw's key.

    The indices are captured from the randomizer itself rather than copied out
    of the source, so a field added later on a raw index fails here. The
    non-vacuity block at the end proves the offset is doing work: the raw table
    indices it replaced really do land on the base split keys.
    """
    seen = []
    real_fold_in = jax.random.fold_in

    def spy(key, data):
        seen.append(int(data))
        return real_fold_in(key, data)

    monkeypatch.setattr(jax.random, "fold_in", spy)
    randomize_fn = randomize.make_domain_randomize(mj_model, robot_spec, ALL_ENABLED)
    randomize_fn(mjx_model, jax.random.split(jax.random.PRNGKey(5), 2))

    # One index per optional field (joint_gains, com_offset, dof,
    # foot_friction, motor_strength), all distinct.
    assert len(seen) == len(randomize._DEFAULT_DR)
    assert len(set(seen)) == len(seen)

    probe = jax.random.PRNGKey(7)
    # An 8-way split covers rand()'s own 5-way split with room for it to grow.
    base = [np.asarray(jax.random.key_data(k)) for k in jax.random.split(probe, 8)]
    for index in seen:
        folded = np.asarray(jax.random.key_data(real_fold_in(probe, index)))
        for j, key in enumerate(base):
            assert not np.array_equal(folded, key), f"dr index {index} aliases split key {j}"

    # Not vacuous: the raw indices 1..5 this offset replaces ARE base split
    # keys, which is the bug the offset fixes.
    for idx in range(1, 6):
        raw = np.asarray(jax.random.key_data(real_fold_in(probe, idx)))
        assert np.array_equal(raw, base[idx])


@pytest.fixture(scope="module")
def sampled(mj_model, mjx_model, robot_spec):
    """DR fields over CORR_KEYS with every optional field except joint_gains on.

    joint_gains stays off on purpose: the kd scale this test reads is the
    disabled branch's single scalar draw off r5, which is the key
    `fold_in(rng, 4)` used to alias.
    """
    dr_cfg = {**ALL_ENABLED, "joint_gains": {"enable": False}}
    randomize_fn = randomize.make_domain_randomize(mj_model, robot_spec, dr_cfg)
    model_v, _ = randomize_fn(mjx_model, CORR_KEYS)

    root_id = randomize._find_base_body_id(mj_model)
    foot_ids = np.array([mj_model.geom(n).id for n in robot_spec.foot_geoms])
    base_mass = np.asarray(mjx_model.body_mass)
    # Recoverable link scales only: the world body has zero mass and the root
    # body's mass carries the separate base_scale draw instead.
    bodies = [
        i for i in range(len(base_mass)) if base_mass[i] > 0 and i != root_id
    ]
    return {
        "com_offset": np.asarray(model_v.body_ipos[:, root_id, :])
        - np.asarray(mjx_model.body_ipos[root_id]),
        "link_scale": np.asarray(model_v.body_mass[:, bodies]) / base_mass[bodies],
        # kd rides actuator_biasprm[:, 2]; disabled joint_gains broadcasts one
        # scalar to every actuator, so column 0 is the whole draw.
        "kd_scale": np.asarray(model_v.actuator_biasprm[:, 0, 2])
        / float(mjx_model.actuator_biasprm[0, 2]),
        "foot_scale": np.asarray(model_v.geom_friction[:, foot_ids, 0])
        / np.asarray(mjx_model.geom_friction[foot_ids, 0]),
    }


def test_com_offset_is_uncorrelated_with_the_link_mass_scale(sampled):
    """`fold_in(rng, 2)` was r3, the link-mass key, and com_offset sampled
    straight off it -- so com_offset's three axes were an affine image of the
    first three link-mass scales, correlation 1.0. Recomputing the pre-fix
    draws below shows that; the shipped draws must not."""
    com, link = sampled["com_offset"], sampled["link_scale"]
    worst = max(
        abs(corr(com[:, axis], link[:, body]))
        for axis in range(3)
        for body in range(link.shape[1])
    )
    assert worst < 0.3, f"com_offset still tracks a link-mass scale (|r| = {worst:.3f})"

    # Not vacuous: the aliased draw really was perfectly correlated.
    def old(rng):
        r3 = jax.random.split(rng, 5)[2]
        legacy_link = jax.random.uniform(r3, (link.shape[1],), minval=0.9, maxval=1.1)
        rc = jax.random.fold_in(rng, 2)
        xy, z = randomize._DEFAULT_DR["com_offset"]["xy"], randomize._DEFAULT_DR["com_offset"]["z"]
        legacy_com = jax.random.uniform(
            rc, (3,), minval=jp.array([-xy, -xy, -z]), maxval=jp.array([xy, xy, z])
        )
        return legacy_com, legacy_link

    legacy_com, legacy_link = jax.vmap(old)(CORR_KEYS)
    assert abs(corr(np.asarray(legacy_com)[:, 1], np.asarray(legacy_link)[:, 1])) > 0.99


def test_foot_friction_is_uncorrelated_with_the_kd_scale(sampled):
    """`fold_in(rng, 4)` was r5, the kd key, and foot_friction sampled straight
    off it -- so the first foot's friction scale was an affine image of the kd
    scale, correlation 1.0."""
    foot, kd = sampled["foot_scale"], sampled["kd_scale"]
    worst = max(abs(corr(foot[:, k], kd)) for k in range(foot.shape[1]))
    assert worst < 0.3, f"foot friction still tracks the kd scale (|r| = {worst:.3f})"

    def old(rng):
        r5 = jax.random.split(rng, 5)[4]
        legacy_kd = jax.random.uniform(r5, minval=0.8, maxval=1.2)
        rf = jax.random.fold_in(rng, 4)
        lo, hi = randomize._DEFAULT_DR["foot_friction"]["range"]
        legacy_foot = jax.random.uniform(rf, (foot.shape[1],), minval=lo, maxval=hi)
        return legacy_kd, legacy_foot

    legacy_kd, legacy_foot = jax.vmap(old)(CORR_KEYS)
    assert abs(corr(np.asarray(legacy_foot)[:, 0], np.asarray(legacy_kd))) > 0.99
