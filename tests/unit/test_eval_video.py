"""eval/video.py's env and CLI wiring (port item 4.5's residual): the
push-free-by-default rollout, `--push`, and the platform-conditional GL
default.

Model-free. `render_video` is stopped at its first line -- the call that
builds the env -- so what reaches `load_checkpoint_policy` is pinned
without rendering a frame. The panels themselves are
tests/unit/test_eval_plots.py.
"""

from __future__ import annotations

import os
import sys

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
