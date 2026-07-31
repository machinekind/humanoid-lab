"""Tests for dr/randomize.py: the yaml/code default pin (w01-tek's
test_config_defaults.py pattern) plus batching behavior on the compiled
asimov model.
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
