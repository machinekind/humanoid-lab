"""eval/video.py's env, panel and CLI wiring: the push-free-by-default
rollout, `--push`, `--plot-torque`,
`--plot-joints`, `--joint`, the panel compositor, and the
platform-conditional GL default.

Model-free. `render_video` is stopped at its first line -- the call that
builds the env -- so what reaches `load_checkpoint_policy` is pinned
without rendering a frame, and `_composite_plots` runs against dummy panel
builders. The panels themselves are tests/unit/test_eval_plots.py.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_lab.eval import video
from humanoid_lab.eval.battery import _measurement_env_overrides


class _Stop(Exception):
    """Raised by the stub in place of loading a checkpoint."""


@pytest.fixture
def loaded_with(monkeypatch):
    """Stop render_video at load_checkpoint_policy and return what it was
    called with."""
    seen = {}

    def fake(run_dir, extra_env_overrides=None):
        seen["run_dir"] = run_dir
        seen["extra"] = extra_env_overrides
        raise _Stop

    monkeypatch.setattr(video, "load_checkpoint_policy", fake)
    return seen


# -- push-free by default ----------------------------------------------------


def test_video_rollouts_are_push_free_by_default(tmp_path, loaded_with):
    """A mid-video kick reads as a policy failure. The battery disables
    pushes for the same reason, and the video says so itself rather than
    inheriting it silently."""
    with pytest.raises(_Stop):
        video.render_video(tmp_path)

    assert loaded_with["extra"] == {"push": {"enable": False}}


def test_push_restores_the_run_s_own_pushes(tmp_path, loaded_with):
    with pytest.raises(_Stop):
        video.render_video(tmp_path, push=True)

    assert loaded_with["extra"] == {"push": {"enable": True}}


def test_the_battery_env_this_layers_onto_is_already_push_free():
    """Belt and braces: the video's own override is redundant today, and
    this is the test that says so. If the measurement env ever stops
    disabling pushes, the video keeps its own convention."""
    overrides = _measurement_env_overrides({"hydra_config": {"task": {"env": {}}}})

    assert overrides["push"]["enable"] is False


def test_push_keeps_the_run_s_other_push_settings():
    """`--push` restores the run's OWN push config -- interval and velocity
    come from the rebuilt env, not from a new default."""
    run = {"hydra_config": {"task": {"env": {"push": {"interval_steps": 50, "vel": 0.7}}}}}
    merged = video.env_overrides_for(run, push=True)

    assert merged["push"]["enable"] is True
    assert merged["push"]["interval_steps"] == 50
    assert merged["push"]["vel"] == 0.7


def test_without_push_the_run_s_other_push_settings_survive_too():
    run = {"hydra_config": {"task": {"env": {"push": {"interval_steps": 50, "vel": 0.7}}}}}
    merged = video.env_overrides_for(run, push=False)

    assert merged["push"]["enable"] is False
    assert merged["push"]["interval_steps"] == 50


# -- CLI wiring --------------------------------------------------------------


def _main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["video", *argv])
    video.main()


@pytest.fixture
def stub_render(monkeypatch):
    seen = {}

    def fake_render_video(run_dir, *args, **kwargs):
        seen["run_dir"] = run_dir
        seen["args"] = args
        seen.update(kwargs)
        return run_dir / "out.mp4"

    monkeypatch.setattr(video, "render_video", fake_render_video)
    return seen


def test_push_defaults_off_on_the_cli(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path)], monkeypatch)

    assert stub_render["push"] is False


def test_the_push_flag_reaches_render_video(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path), "--push"], monkeypatch)

    assert stub_render["push"] is True


def test_both_plot_flags_default_off(tmp_path, monkeypatch, stub_render):
    """Plain rendering must stay the pre-panel code path: the per-step device
    transfers the panels need are only paid when a flag asks for them."""
    _main(["--run", str(tmp_path)], monkeypatch)

    assert stub_render["plot_torque"] is False
    assert stub_render["plot_joints"] is False
    assert stub_render["joint"] is None


def test_the_plot_flags_reach_render_video(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path), "--plot-torque", "--plot-joints"], monkeypatch)

    assert stub_render["plot_torque"] is True
    assert stub_render["plot_joints"] is True


def test_the_joint_flag_reaches_render_video(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path), "--joint", "left_knee"], monkeypatch)

    assert stub_render["joint"] == "left_knee"


# -- --joint implies --plot-joints -------------------------------------------


def test_a_named_joint_implies_the_joint_panel():
    """`--joint NAME` alone is enough: asking for one joint's zoom panel is
    asking for a joint panel."""
    assert video.resolve_panels(False, False, "left_knee") == (False, True)


def test_the_implication_lives_below_the_cli():
    """A library caller passing joint= gets it too, which is why the
    normalization is in render_video and not in main()."""
    assert video.resolve_panels(False, False, None) == (False, False)
    assert video.resolve_panels(True, False, None) == (True, False)


def test_a_named_joint_does_not_arm_the_torque_strip():
    assert video.resolve_panels(False, True, "left_knee") == (False, True)


# -- the panel compositor ----------------------------------------------------


def _panels_env(nu=2):
    """The three attributes _composite_plots reads off the env, no model."""
    return SimpleNamespace(
        dt=0.02,
        mj_model=SimpleNamespace(actuator_forcerange=np.tile([-10.0, 10.0], (nu, 1))),
        robot_spec=SimpleNamespace(joint_groups={"leg": ["a", "b"]}),
    )


@pytest.fixture
def dummy_panels(monkeypatch):
    """Replace eval/plots.py's builders with panel factories of a known
    height, so the compositor's stacking is checkable without matplotlib."""
    def factory(height):
        def build(*args, width, **kwargs):
            return lambda t: np.full((height, width, 3), height, np.uint8)
        return build

    monkeypatch.setattr(video, "torque_strip", factory(7))
    monkeypatch.setattr(video, "joint_grid", factory(11))
    monkeypatch.setattr(video, "joint_zoom", factory(13))


def _composite(plot_torque, plot_joints, joint=None, n_samples=3, nu=2):
    frames = [np.zeros((5, 4, 3), np.uint8), np.ones((5, 4, 3), np.uint8)]
    samples = [np.zeros(nu) for _ in range(n_samples)]
    return video._composite_plots(
        frames, [0.0, 0.02], _panels_env(nu), ["a", "b"], joint,
        plot_torque, plot_joints, samples, samples, samples,
    )


def test_the_torque_strip_stacks_under_every_frame(dummy_panels):
    out = _composite(plot_torque=True, plot_joints=False)

    assert len(out) == 2
    assert [f.shape for f in out] == [(5 + 7, 4, 3)] * 2


def test_both_panels_stack_in_flag_order(dummy_panels):
    """Torque strip first, joint panel below it -- the order --plot-joints'
    help text promises ("below that")."""
    out = _composite(plot_torque=True, plot_joints=True)

    assert out[0].shape == (5 + 7 + 11, 4, 3)
    assert out[0][5, 0, 0] == 7  # the strip's row, directly under the render
    assert out[0][5 + 7, 0, 0] == 11  # the grid's, under the strip


def test_a_named_joint_swaps_the_grid_for_the_zoom(dummy_panels):
    out = _composite(plot_torque=False, plot_joints=True, joint="a")

    assert out[0].shape == (5 + 13, 4, 3)


def test_no_control_steps_composites_nothing(dummy_panels, capsys):
    """--steps 0 leaves `frames` with only the initial pose and no samples;
    the panel builders would crash on an empty times array, so the
    compositor returns the frames untouched and says so."""
    frames = [np.zeros((5, 4, 3), np.uint8)]
    out = video._composite_plots(
        frames, [0.0], _panels_env(), ["a", "b"], None, True, True, [], [], [],
    )

    assert out is frames
    assert "skipping plot panels" in capsys.readouterr().out


# -- the GL backend default --------------------------------------------------


@pytest.mark.skipif(
    "MUJOCO_GL" in os.environ,
    reason="MUJOCO_GL is exported in this shell; the module's setdefault is a no-op",
)
def test_egl_is_a_linux_only_default():
    """macOS has no EGL: forcing it there breaks offscreen rendering
    outright. Importing this module (which happened at the top of this file)
    must therefore leave MUJOCO_GL alone on darwin, and set egl on linux --
    which is what run.sh's `eval` verb comment describes.
    """
    if sys.platform == "linux":
        assert os.environ.get("MUJOCO_GL") == "egl"
    else:
        assert "MUJOCO_GL" not in os.environ
