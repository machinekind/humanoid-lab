"""check-friction end to end on jax: flat floor and the CPU terrain arena."""

from __future__ import annotations

import pytest

from humanoid_lab import check_friction

ROBOT, PRESET = "roboto_origin", "deploy_pd"


def test_probe_passes_on_flat(capsys):
    assert check_friction.probe(ROBOT, PRESET, "jax", 2, (0.8, 1.2))
    out = capsys.readouterr().out
    assert "task joystick" in out
    assert " on terrain_" not in out


def test_probe_passes_on_terrain(capsys):
    assert check_friction.probe(ROBOT, PRESET, "jax", 2, (0.8, 1.2), task="terrain")
    out = capsys.readouterr().out
    # Each world stands on its own flat-row pad, on the heightfield.
    assert out.count(" on terrain_hfield") == out.count("contact ")
    assert out.count("contact ") >= 2


def test_probe_refuses_an_unknown_task():
    with pytest.raises(ValueError, match="task"):
        check_friction.probe(ROBOT, PRESET, "jax", 2, (0.8, 1.2), task="stairs")
