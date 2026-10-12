"""eval/video.py's env, panel and CLI wiring: the push-free-by-default
rollout, `--push`, `--plot-torque`,
`--plot-joints`, `--joint`, the panel compositor, the outer heading
loop, the video-only `long` scenario, the side-by-side `--compare` clip
and the platform-conditional GL default.

Model-free. `render_video` is stopped at its first line -- the call that
builds the env -- so what reaches `load_checkpoint_policy` is pinned
without rendering a frame, and `_composite_plots` runs against dummy panel
builders. The panels themselves are tests/unit/test_eval_plots.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_lab.eval import video
from humanoid_lab.eval.battery import _measurement_env_overrides, battery_scenarios


class _Stop(Exception):
    """Raised by the stub in place of loading a checkpoint."""


@pytest.fixture
def loaded_with(monkeypatch):
    """Stop render_video at load_checkpoint_policy and return what it was
    called with."""
    seen = {}

    def fake(run_dir, extra_env_overrides=None, step=None):
        seen["run_dir"] = run_dir
        seen["extra"] = extra_env_overrides
        seen["step"] = step
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


# -- checkpoint selection ----------------------------------------------------
#
# The step is resolved at load time. A render without one takes whatever is
# latest then, which can differ from the checkpoint a battery measured.


def test_render_video_loads_the_requested_checkpoint(tmp_path, loaded_with):
    with pytest.raises(_Stop):
        video.render_video(tmp_path, checkpoint=200)

    assert loaded_with["step"] == 200


def test_render_video_defaults_to_the_latest(tmp_path, loaded_with):
    with pytest.raises(_Stop):
        video.render_video(tmp_path)

    assert loaded_with["step"] is None


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


def test_the_checkpoint_flag_reaches_render_video(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path), "--checkpoint", "200"], monkeypatch)
    assert stub_render["checkpoint"] == 200

    _main(["--run", str(tmp_path)], monkeypatch)
    assert stub_render["checkpoint"] is None


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


# -- the outer heading loop --------------------------------------------------


def test_heading_correction_anchors_the_target_on_the_first_step():
    cmd = np.array([0.3, 0.0, 0.2])

    out, target = video.heading_correction(cmd, 0.7, None, 1.5, 0.02)

    assert target == 0.7
    assert out[2] == pytest.approx(0.2)


def test_heading_correction_integrates_the_scenario_wz():
    cmd = np.array([0.0, 0.0, 0.5])

    out, target = video.heading_correction(cmd, 0.0, 0.0, 1.0, 0.02)

    assert target == pytest.approx(0.01)
    assert out[2] == pytest.approx(0.5 + 0.01)


def test_heading_correction_wraps_across_180_degrees():
    """Yaw 3.1 and target -3.1 are 0.083 rad apart across +-pi."""
    out, _ = video.heading_correction(np.zeros(3), 3.1, -3.1, 1.0, 0.02)

    assert out[2] == pytest.approx(2 * np.pi - 6.2, abs=1e-6)


def test_heading_correction_uses_the_unclipped_wrap():
    """An error of 1.5 rad is beyond heading_err's 1 rad clip."""
    out, _ = video.heading_correction(np.zeros(3), 0.0, 1.5, 0.3, 0.02)

    assert out[2] == pytest.approx(0.45)


def test_heading_correction_clips_at_half_a_radian_per_second():
    up, _ = video.heading_correction(np.zeros(3), 0.0, 1.0, 5.0, 0.02)
    down, _ = video.heading_correction(np.zeros(3), 0.0, -1.0, 5.0, 0.02)

    assert up[2] == pytest.approx(video.HEADING_WZ_CLIP) == pytest.approx(0.5)
    assert down[2] == pytest.approx(-0.5)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _fake_rollout(heading_gain, n_steps=3, fall_at=None):
    """Run _rollout_frames on a fake env and return the commands each step saw.

    The base starts at yaw 0.3 and turns 0.1 rad per step whatever the
    command. env._quat reads the yaw from state.data. qpos holds a decoy
    yaw of 0, so a loop that read qpos would see no turn. No jit is involved.
    """
    seen = []

    def state_at(yaw, k):
        data = SimpleNamespace(yaw=yaw, qpos=np.concatenate([[0, 0, 0.7], _yaw_quat(0.0)]))
        return SimpleNamespace(data=data, info={}, obs=None, done=k == fall_at)

    def step(state, act):
        seen.append(np.array(state.info["command"]))
        return state_at(state.data.yaw + 0.1, len(seen))

    env = SimpleNamespace(dt=0.02, _quat=lambda d: _yaw_quat(d.yaw))
    view = SimpleNamespace(frame=lambda qpos, torque=None: np.zeros((2, 2, 3), np.uint8))
    frames, *_ = video._rollout_frames(
        env, lambda rng: state_at(0.3, 0), step, lambda obs, rng: (np.zeros(1), None), view,
        lambda i: np.array([0.2, 0.0, 0.1]), n_steps,
        seed=0, heading_gain=heading_gain, capture=False, overlay_torque=False, render_every=1,
    )
    return seen, frames


def test_the_render_loop_closes_heading_on_the_imu_yaw():
    seen, frames = _fake_rollout(1.0)

    assert len(seen) == 3 and len(frames) == 4
    # First step anchors the target at the measured yaw: no correction.
    assert seen[0][2] == pytest.approx(0.1)
    # Second step: target 0.3 + 0.1 * dt, measured yaw 0.4.
    err = 0.3 + 0.1 * 0.02 - 0.4
    assert seen[1][2] == pytest.approx(0.1 + np.clip(err, -0.5, 0.5), abs=1e-6)
    target = None
    for i, cmd in enumerate(seen):
        want, target = video.heading_correction(
            np.array([0.2, 0.0, 0.1]), 0.3 + 0.1 * i, target, 1.0, 0.02
        )
        np.testing.assert_allclose(cmd, want, atol=1e-6)


def test_the_render_loop_leaves_the_command_alone_open_loop():
    seen, _ = _fake_rollout(0.0)

    for cmd in seen:
        np.testing.assert_allclose(cmd, [0.2, 0.0, 0.1])


def test_a_fall_stops_the_rollout_and_names_the_heading_gain(capsys):
    seen, frames = _fake_rollout(1.5, n_steps=5, fall_at=2)

    # The second step falls. No later step runs and the fallen state gets no frame.
    assert len(seen) == 2
    assert len(frames) == 2
    assert "1.5" in capsys.readouterr().out


# -- the video-only long scenario ----------------------------------------------


_ROBOTO_BOX = {"vx": (-0.6, 1.0), "vy": (-0.5, 0.5), "wz": (-1.57, 1.57)}
_ASIMOV_BOX = {"vx": (-0.8, 0.8), "vy": (-0.5, 0.5), "wz": (-0.6, 0.6)}


def test_long_is_a_video_only_scenario():
    battery = battery_scenarios(0.02, _ROBOTO_BOX)

    assert set(video.video_scenarios(0.02, _ROBOTO_BOX)) == set(battery) | {"long"}
    assert "long" not in battery


def test_long_lasts_forty_seconds_at_any_dt():
    assert video.long_scenario(0.02, _ROBOTO_BOX)[1] == 2000
    assert video.long_scenario(0.04, _ROBOTO_BOX)[1] == 1000


def test_long_reproduces_the_original_clip_on_roboto():
    cmd_at, _ = video.pick_scenario(0.02, _ROBOTO_BOX, "long")

    def at(t):
        return cmd_at(round(t / 0.02))

    assert at(3.5)[0] == pytest.approx(0.3)
    assert at(7.0)[0] == pytest.approx(0.6)
    assert at(15.0)[0] == pytest.approx(0.9)
    np.testing.assert_allclose(at(22.0), [0.4, 0.0, 0.5])
    np.testing.assert_allclose(at(27.0), [0.4, 0.0, -0.5])
    np.testing.assert_array_equal(at(39.5), [0.0, 0.0, 0.0])
    assert at(22.0).dtype == np.float64


def test_long_stays_inside_asimov_s_envelope():
    cmd_at, n_steps = video.long_scenario(0.02, _ASIMOV_BOX)
    cmds = np.array([cmd_at(i) for i in range(n_steps)])

    assert np.all(np.abs(cmds[:, 0]) <= 0.8)
    assert np.all(np.abs(cmds[:, 2]) <= 0.6)
    assert np.all(cmds[:, 1] == 0.0)
    assert cmds[:, 0].max() == pytest.approx(0.72)


# -- the side-by-side compare clip ---------------------------------------------


def _clip(n, value, shape=(4, 6, 3)):
    return [np.full(shape, value, np.uint8) for _ in range(n)]


def test_side_by_side_pads_the_shorter_clip_with_its_last_frame():
    left = _clip(2, 10)
    left[-1] = np.full((4, 6, 3), 11, np.uint8)
    right = _clip(4, 200)

    out = video._side_by_side(left, right)

    assert len(out) == 4
    assert all(f.shape == (4, 12, 3) for f in out)
    assert out[0][:, :6].max() == 10
    assert all(np.all(f[:, :6] == 11) for f in out[1:])
    assert all(np.all(f[:, 6:] == 200) for f in out)


class _DummyView:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_compare_loads_once_and_rolls_out_twice_at_the_same_seed(tmp_path, monkeypatch):
    loads, rollouts, written = [], [], {}
    env = SimpleNamespace(
        dt=0.02,
        mj_model=SimpleNamespace(),
        robot_spec=SimpleNamespace(
            eval_camera="cam", actuated_joints=["a", "b"], joint_groups={"leg": ["a", "b"]}
        ),
        _config=SimpleNamespace(command={"vx": (-1, 1), "vy": (-1, 1), "wz": (-1, 1)}),
        reset=lambda rng: None,
        step=lambda s, a: None,
    )

    def fake_load(run_dir, extra_env_overrides=None, step=None):
        loads.append(step)
        return {"run_name": "r"}, env, Path("000000000200"), lambda obs, rng: None

    def fake_rollout(env, reset, step, inf, view, cmd_at, n_steps, *, seed, heading_gain, **kw):
        rollouts.append((seed, heading_gain))
        n, value = (3, 10) if heading_gain == 0.0 else (5, 200)
        return _clip(n, value, shape=(120, 160, 3)), [0.0] * n, [], [], []

    def fake_write(out, frames, fps):
        written["out"] = out
        written["frames"] = frames

    monkeypatch.setattr(video, "load_checkpoint_policy", fake_load)
    monkeypatch.setattr(video, "SceneView", _DummyView)
    monkeypatch.setattr(video, "_rollout_frames", fake_rollout)
    monkeypatch.setattr(video, "write_video", fake_write)

    out = video.render_video(tmp_path, compare=True, heading_gain=1.5, seed=3, checkpoint=200)

    assert loads == [200]
    assert [g for _, g in rollouts] == [0.0, 1.5]
    assert all(s == 3 for s, _ in rollouts)
    frames = written["frames"]
    assert len(frames) == 5
    assert all(f.shape == (120, 320, 3) for f in frames)
    # Below the label box the padded left half repeats the left clip's last frame.
    for i in (3, 4):
        np.testing.assert_array_equal(frames[i][80:, :160], frames[2][80:, :160])
    assert out == written["out"] == tmp_path / "walk_ramp_compare.mp4"


def test_compare_without_a_gain_is_refused_before_loading(tmp_path, loaded_with):
    with pytest.raises(ValueError, match="heading_gain"):
        video.render_video(tmp_path, compare=True, heading_gain=0.0)

    assert loaded_with == {}


def test_compare_defaults_to_a_gain_of_one_and_a_half(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path), "--compare"], monkeypatch)

    assert stub_render["heading_gain"] == 1.5
    assert stub_render["compare"] is True


def test_compare_takes_the_heading_gain_flag(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path), "--compare", "--heading-gain", "0.8"], monkeypatch)

    assert stub_render["heading_gain"] == 0.8
    assert stub_render["compare"] is True


def test_compare_with_a_zero_gain_is_a_usage_error(tmp_path, monkeypatch, stub_render):
    with pytest.raises(SystemExit) as exc:
        _main(["--run", str(tmp_path), "--compare", "--heading-gain", "0"], monkeypatch)

    assert exc.value.code == 2
    assert "run_dir" not in stub_render


def test_without_compare_the_gain_defaults_to_open_loop(tmp_path, monkeypatch, stub_render):
    _main(["--run", str(tmp_path)], monkeypatch)

    assert stub_render["heading_gain"] == 0.0
    assert stub_render["compare"] is False


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
