"""Build-order gate for robot.yaml's model_patches section and the
unconditional source-actuator/actuator-sensor strip in build_spec.

Covers robot #2's motivating case (a vendored MJCF that ships its own
<actuator> block, actuator sensors, no named foot geoms, and a PGS solver)
without depending on any real robot: every fixture here is a tiny synthetic
MJCF written into tmp_path. CPU-only, no MJX, same shape as
tests/test_injection.py.
"""

from __future__ import annotations

import mujoco
import pytest

from humanoid_lab.robot.build import build_spec, compile_spec
from humanoid_lab.robot.spec import ModelPatches, load_robot_spec

_PRESET_NAME = "test"

_PRESET_YAML = """
model: pd
groups:
  leg: {kp: 50.0, kd: 2.0, effort_limit: 40.0}
"""

_ROBOT_YAML_BASE = """
name: patch_test
model_xml: source/xmls/robot.xml
actuated_joints: [leg_joint]
joint_groups:
  leg: [leg_joint]
passive_joints: {{}}
foot_sites: {foot_sites}
foot_geoms: {foot_geoms}
{model_patches}
"""


def _write_robot(tmp_path, *, xml, model_patches_yaml="", foot_sites="[]", foot_geoms="[]"):
    """Write a minimal robot dir (source XML, robot.yaml, one preset) into
    tmp_path and return its path. `model_patches_yaml`, if given, must be a
    complete `model_patches:` block (including the trailing newline).
    """
    (tmp_path / "source" / "xmls").mkdir(parents=True)
    (tmp_path / "source" / "xmls" / "robot.xml").write_text(xml)

    (tmp_path / "robot.yaml").write_text(
        _ROBOT_YAML_BASE.format(
            foot_sites=foot_sites, foot_geoms=foot_geoms, model_patches=model_patches_yaml
        )
    )

    (tmp_path / "actuators").mkdir()
    (tmp_path / "actuators" / f"{_PRESET_NAME}.yaml").write_text(_PRESET_YAML)
    return tmp_path


# Two-link body (free-floating torso + one hinge, "leg_joint") shared by
# most fixtures below; individual tests splice in extra bodies/actuators/
# sensors/assets/options as needed.
_BASE_BODY = """
    <body name="torso" pos="0 0 0.5">
      <freejoint/>
      <geom name="torso_geom" type="box" size="0.1 0.05 0.15" mass="5.0"/>
      {extra_torso_children}
      <body name="link1" pos="0 0 -0.2">
        <joint name="leg_joint" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        {link1_geoms}
        {link1_children}
      </body>
    </body>
"""


_DEFAULT_LINK1_GEOMS = (
    '<geom name="link1_geom" type="capsule" fromto="0 0 0 0 0 -0.25" size="0.04" mass="2.0"/>'
)


def _xml(
    *,
    options="",
    extra_torso_children="",
    link1_geoms=_DEFAULT_LINK1_GEOMS,
    link1_children="",
    assets="",
    actuator="",
    sensor="",
):
    body = _BASE_BODY.format(
        extra_torso_children=extra_torso_children,
        link1_geoms=link1_geoms,
        link1_children=link1_children,
    )
    return f"""
<mujoco model="patch_test">
  <compiler angle="radian"/>
  {options}
  {assets}
  <worldbody>
    {body}
  </worldbody>
  {actuator}
  {sensor}
</mujoco>
"""


def test_strip_replaces_source_actuator_and_keeps_its_sensor(tmp_path):
    """A source actuator on an actuated joint, plus an actuatorpos sensor on
    it, is fully replaced: build_spec's strip deletes the source actuator
    (it would collide on the "leg_joint" name with injection), injection
    recreates it with the preset's own parameters, and the sensor -- which
    references the actuator by name, not by identity -- keeps resolving.
    """
    xml = _xml(
        actuator='<actuator><motor name="leg_joint" joint="leg_joint" gear="7"/></actuator>',
        sensor='<sensor><actuatorpos name="leg_joint_p" actuator="leg_joint"/></sensor>',
    )
    robot_dir = _write_robot(tmp_path, xml=xml)

    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))

    assert model.nu == 1
    leg = model.actuator("leg_joint")
    # The preset's kp (50.0), not the source motor's gear=7: proves the
    # source actuator was deleted and re-injected, not left in place.
    assert leg.gainprm[0] == pytest.approx(50.0)

    assert model.nsensor == 1
    assert model.sensor("leg_joint_p") is not None


def test_strip_removes_actuator_and_sensor_for_joint_outside_actuated_joints(tmp_path):
    """A source actuator on a joint NOT in actuated_joints, plus an
    actuatorfrc sensor on it, is stripped entirely (injection never
    recreates an actuator for it, so leaving the sensor would dangle and
    fail spec.compile()). A non-actuator sensor elsewhere in the model
    survives untouched.
    """
    xml = _xml(
        extra_torso_children='<site name="torso_site" pos="0 0 0"/>',
        link1_children="""
        <body name="link2" pos="0 0 -0.25">
          <joint name="extra_joint" type="hinge" axis="1 0 0" range="-0.3 0.3"/>
          <geom name="link2_geom" type="sphere" size="0.03" mass="0.3"/>
        </body>
        """,
        actuator='<actuator><motor name="extra_joint" joint="extra_joint" gear="1"/></actuator>',
        sensor="""
        <sensor>
          <actuatorfrc name="extra_joint_f" actuator="extra_joint"/>
          <framequat name="torso_quat" objtype="site" objname="torso_site"/>
        </sensor>
        """,
    )
    robot_dir = _write_robot(tmp_path, xml=xml)

    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))

    assert model.nu == 1  # only leg_joint; extra_joint's source actuator is gone
    assert model.actuator(0).name == "leg_joint"

    assert model.nsensor == 1  # extra_joint_f deleted, torso_quat survives
    assert model.sensor(0).name == "torso_quat"


def test_strip_removes_actuator_named_unlike_its_joint_and_its_sensor(tmp_path):
    """A source actuator whose name differs from its joint ("drive_motor" on
    the actuated leg_joint) is stripped like any other, and its actuatorpos
    sensor goes with it: the sensor references the "drive_motor" name, which
    is not an actuated-joint name, so injection never recreates an actuator
    it could resolve against. Injection still yields exactly one actuator,
    named after the joint.
    """
    xml = _xml(
        actuator='<actuator><motor name="drive_motor" joint="leg_joint" gear="7"/></actuator>',
        sensor='<sensor><actuatorpos name="drive_motor_p" actuator="drive_motor"/></sensor>',
    )
    robot_dir = _write_robot(tmp_path, xml=xml)

    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))

    assert model.nu == 1
    assert model.actuator(0).name == "leg_joint"
    assert model.nsensor == 0


def test_injected_site_and_geom_pass_validation_and_inherit_main_default(tmp_path):
    """model_patches.sites/geoms land in the compiled model at the given
    body/pos/size, resolve as foot_sites/foot_geoms (i.e. build_spec's
    validate_against_model call passes), and an injected geom with no
    explicit contype/conaffinity inherits the source XML's main default
    class rather than mujoco's own compiler default.
    """
    xml = _xml(
        # A main (unnamed) default class with a non-default contype/conaffinity,
        # so "inherits the XML's main default" is distinguishable from
        # "inherits mujoco's compiler default" (both would be 1/1 otherwise).
        options='<default><geom contype="1" conaffinity="0" condim="4"/></default>',
        link1_children="""
        <body name="foot_link" pos="0 0 -0.25">
          <geom name="foot_link_geom" type="sphere" size="0.02" mass="0.1"/>
        </body>
        """,
    )
    model_patches_yaml = """
model_patches:
  sites:
    injected_foot_site: {body: foot_link, pos: [0.02, 0.0, -0.04]}
  geoms:
    injected_foot_box:
      {body: foot_link, type: box, size: [0.10, 0.05, 0.012], pos: [0.02, 0.0, -0.05]}
"""
    robot_dir = _write_robot(
        tmp_path,
        xml=xml,
        model_patches_yaml=model_patches_yaml,
        foot_sites="[injected_foot_site]",
        foot_geoms="[injected_foot_box]",
    )

    # build_spec calls validate_against_model internally; a missing
    # foot_sites/foot_geoms entry would raise here.
    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))

    site = model.site("injected_foot_site")
    assert list(site.pos) == pytest.approx([0.02, 0.0, -0.04])

    geom = model.geom("injected_foot_box")
    assert list(geom.size) == pytest.approx([0.10, 0.05, 0.012])
    assert list(geom.pos) == pytest.approx([0.02, 0.0, -0.05])
    assert geom.contype[0] == 1
    assert geom.conaffinity[0] == 0


def test_injected_capsule_geom_via_fromto(tmp_path):
    """A geoms entry with fromto and no pos compiles: the geom takes its
    frame from fromto (the default identity quat that build_spec always
    forwards must not conflict with it), lands as a capsule, and carries the
    given radius plus the half-length mujoco derives from the fromto pair.
    """
    xml = _xml()
    model_patches_yaml = """
model_patches:
  geoms:
    injected_capsule:
      {body: link1, type: capsule, size: [0.02], fromto: [0.0, 0.0, 0.0, 0.0, 0.0, -0.1]}
"""
    robot_dir = _write_robot(tmp_path, xml=xml, model_patches_yaml=model_patches_yaml)

    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))

    geom = model.geom("injected_capsule")
    assert geom.type[0] == mujoco.mjtGeom.mjGEOM_CAPSULE
    assert geom.size[0] == pytest.approx(0.02)  # radius from size
    assert geom.size[1] == pytest.approx(0.05)  # half-length derived from fromto


def test_options_patch_overrides_solver_iterations_timestep(tmp_path):
    xml = _xml(options='<option solver="PGS" iterations="100" timestep="0.002"/>')
    model_patches_yaml = """
model_patches:
  options: {solver: newton, iterations: 8, timestep: 0.004}
"""
    robot_dir = _write_robot(tmp_path, xml=xml, model_patches_yaml=model_patches_yaml)

    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))

    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.iterations == 8
    assert model.opt.timestep == pytest.approx(0.004)


def test_mesh_collisions_visual_zeroes_only_mesh_geoms(tmp_path):
    """mesh_collisions: visual makes every mjGEOM_MESH geom collision-inert
    (contype=conaffinity=0); a primitive geom elsewhere on the same body is
    untouched.
    """
    xml = _xml(
        assets='<asset><mesh name="tiny_mesh" vertex="0 0 0 0.05 0 0 0 0.05 0 0 0 0.05"/></asset>',
        link1_geoms=(
            '<geom name="link1_mesh_geom" type="mesh" mesh="tiny_mesh" mass="0.5"/>'
            '<geom name="link1_capsule_geom" type="capsule" '
            'fromto="0 0 0 0 0 -0.25" size="0.04" mass="2.0"/>'
        ),
    )
    model_patches_yaml = "model_patches:\n  mesh_collisions: visual\n"
    robot_dir = _write_robot(tmp_path, xml=xml, model_patches_yaml=model_patches_yaml)

    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))

    mesh_geom = model.geom("link1_mesh_geom")
    assert mesh_geom.contype[0] == 0
    assert mesh_geom.conaffinity[0] == 0

    capsule_geom = model.geom("link1_capsule_geom")
    assert capsule_geom.contype[0] == 1
    assert capsule_geom.conaffinity[0] == 1


@pytest.mark.parametrize(
    ("model_patches_yaml", "match"),
    [
        pytest.param(
            "model_patches:\n  bogus_section: {}\n",
            "bogus_section",
            id="unknown-model-patches-key",
        ),
        pytest.param(
            "model_patches:\n  options: {bogus_option: 1}\n",
            "bogus_option",
            id="unknown-options-key",
        ),
        pytest.param(
            "model_patches:\n  sites:\n    s1: {body: link1, pos: [0, 0, 0], rot: [1, 0, 0, 0]}\n",
            "rot",
            id="unknown-site-key",
        ),
        pytest.param(
            "model_patches:\n  geoms:\n    g1: {body: link1, type: box, size: [1], mass: 1.0}\n",
            "mass",
            id="unknown-geom-key",
        ),
        pytest.param(
            "model_patches:\n  options: {solver: sparse}\n",
            "sparse",
            id="bad-solver",
        ),
        pytest.param(
            "model_patches:\n  geoms:\n    g1: {body: link1, type: cylinder, size: [0.1]}\n",
            "cylinder",
            id="bad-geom-type",
        ),
        pytest.param(
            "model_patches:\n  mesh_collisions: primitive\n",
            "primitive",
            id="bad-mesh-collisions",
        ),
        pytest.param(
            "model_patches:\n  sites:\n    s1: {body: link1, pos: [0.0, 0.0]}\n",
            "pos must have 3 elements",
            id="bad-pos-length",
        ),
    ],
)
def test_model_patches_schema_violation_raises_named_error(tmp_path, model_patches_yaml, match):
    """Every schema-level validation class in _parse_model_patches raises a
    ValueError naming the offending key or value at load time, before any
    mujoco model is involved.
    """
    robot_dir = _write_robot(tmp_path, xml=_xml(), model_patches_yaml=model_patches_yaml)

    with pytest.raises(ValueError, match=match):
        load_robot_spec(robot_dir)


@pytest.mark.parametrize(
    ("model_patches_yaml", "match"),
    [
        pytest.param(
            "model_patches:\n  sites:\n    s1: {body: no_such_body, pos: [0.0, 0.0, 0.0]}\n",
            "no_such_body",
            id="site-body",
        ),
        pytest.param(
            "model_patches:\n  geoms:\n    g1: {body: no_such_body, type: box, size: [1, 1, 1]}\n",
            "no_such_body",
            id="geom-body",
        ),
    ],
)
def test_patch_referencing_missing_body_raises_named_error(tmp_path, model_patches_yaml, match):
    """A sites/geoms entry naming a body the source XML doesn't have is a
    build-time error (the schema can't know the model's bodies at load time),
    raised by build_spec with the body name in the message.
    """
    robot_dir = _write_robot(tmp_path, xml=_xml(), model_patches_yaml=model_patches_yaml)

    with pytest.raises(ValueError, match=match):
        build_spec(robot_dir, _PRESET_NAME)


def test_missing_model_patches_parses_empty_and_build_is_unaffected(tmp_path):
    """No model_patches key at all in robot.yaml parses to ModelPatches's
    all-empty defaults, and build_spec's output is unaffected: the source
    XML's own <option> values pass through unmodified.
    """
    xml = _xml(options='<option solver="PGS" iterations="100" timestep="0.002"/>')
    robot_dir = _write_robot(tmp_path, xml=xml)

    robot_spec = load_robot_spec(robot_dir)
    assert robot_spec.model_patches == ModelPatches()

    model = compile_spec(build_spec(robot_dir, _PRESET_NAME))
    assert model.nu == 1
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_PGS
    assert model.opt.iterations == 100
    assert model.opt.timestep == pytest.approx(0.002)
