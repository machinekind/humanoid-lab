"""eval/battery.py's `--set BLOCK.KEY=VALUE`: re-scoring a checkpoint under a
changed task.env block (e.g. a different observation noise).

Model-free. The parser and the override merge are pure dict logic; main() and
run_battery() run with the checkpoint loader and the scenario table stubbed
out, so no env is built and nothing is rolled out.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from humanoid_lab.eval import battery
from humanoid_lab.eval.battery import merged_env_overrides, parse_env_set

# -- parse_env_set -----------------------------------------------------------


def test_one_item_becomes_a_one_block_override():
    assert parse_env_set(["obs_noise.joint_vel=1.75"]) == {"obs_noise": {"joint_vel": 1.75}}


def test_values_go_through_yaml():
    parsed = parse_env_set(["push.enable=true", "command.vx=[-0.5, 1.0]", "obs_noise.gyro=0"])

    assert parsed["push"] == {"enable": True}
    assert parsed["command"] == {"vx": [-0.5, 1.0]}
    assert parsed["obs_noise"] == {"gyro": 0}


def test_an_exponent_without_a_decimal_point_is_still_a_float():
    """PyYAML reads `1e-4` as a string; a noise scale spelled that way is a
    number, and a string would only fail later, at env build."""
    assert parse_env_set(["obs_noise.joint_vel=1e-4"]) == {"obs_noise": {"joint_vel": 1e-4}}
    assert parse_env_set(["obs_noise.joint_vel=2E1"]) == {"obs_noise": {"joint_vel": 20.0}}


def test_items_on_the_same_block_share_it_and_the_later_one_wins():
    parsed = parse_env_set(
        ["obs_noise.joint_vel=1.75", "obs_noise.joint_pos=0.05", "obs_noise.joint_vel=0.2"]
    )

    assert parsed == {"obs_noise": {"joint_vel": 0.2, "joint_pos": 0.05}}


@pytest.mark.parametrize(
    "item",
    [
        "obs_noise.joint_vel",        # no '='
        "joint_vel=0.2",              # no '.'
        "task.obs_noise.joint_vel=0.2",  # deeper than one level
        ".joint_vel=0.2",             # empty block
        "obs_noise.=0.2",             # empty key
        "obs_noise.joint_vel=",       # empty value
    ],
)
def test_a_malformed_item_raises(item):
    with pytest.raises(ValueError):
        parse_env_set([item])


# -- merged_env_overrides under an obs_noise extra ---------------------------


def _run(env_block: dict) -> dict:
    return {"hydra_config": {"task": {"env": env_block}}}


def test_an_obs_noise_extra_keeps_the_measurement_only_changes():
    """Re-scoring changes the noise, not the measurement convention: pushes,
    the command resample and the no-progress cut stay neutralised."""
    run = _run({"push": {"enable": True, "interval_range": [5, 10]}})

    overrides = merged_env_overrides(run, parse_env_set(["obs_noise.joint_vel=1.75"]))

    assert overrides["push"]["enable"] is False
    assert overrides["push"]["interval_range"] == [5, 10]
    assert overrides["command"]["resample_steps"] == 10_000_000
    assert overrides["no_progress"]["enable"] is False


def test_an_obs_noise_extra_replaces_one_key_and_keeps_the_run_s_others():
    """One level deep: the run's own joint_pos and gyro survive a joint_vel
    re-score, so exactly one noise channel differs from training."""
    run = _run({"obs_noise": {"joint_pos": 0.03, "joint_vel": 0.2, "gyro": 0.01}})

    overrides = merged_env_overrides(run, parse_env_set(["obs_noise.joint_vel=1.75"]))

    assert overrides["obs_noise"] == {"joint_pos": 0.03, "joint_vel": 1.75, "gyro": 0.01}


# -- run_battery forwarding and recording -------------------------------------
#
# The loader is replaced by name and the scenario table emptied, so
# run_battery's own bookkeeping runs with no env, checkpoint or rollout.


@pytest.fixture
def stub_loader(monkeypatch):
    """Record what run_battery hands the checkpoint loader. Returns that
    record."""
    seen = {}

    def fake_loader(run_dir, extra_env_overrides=None):
        seen["run_dir"] = run_dir
        seen["extra_env_overrides"] = extra_env_overrides
        env = SimpleNamespace(
            reset=lambda rng: None,
            step=lambda state, act: None,
            mj_model=SimpleNamespace(actuator_forcerange=np.zeros((2, 2))),
            dt=0.02,
            _config=SimpleNamespace(command={"vx": (-1, 1), "vy": (-1, 1), "wz": (-1, 1)}),
        )
        return {"run_name": "stub"}, env, Path("100"), None

    monkeypatch.setattr(battery, "load_checkpoint_policy", fake_loader)
    monkeypatch.setattr(battery, "battery_scenarios", lambda dt, command: {})
    monkeypatch.setattr(battery.sim_budget, "budget_report_for_env", lambda env, a, b: {"stub": 1})
    return seen


def test_run_battery_without_extra_keeps_the_default_schema(tmp_path, stub_loader):
    results = battery.run_battery(tmp_path)

    assert stub_loader["extra_env_overrides"] is None
    assert list(results) == ["run", "checkpoint", "contacts"]


def test_run_battery_forwards_and_records_the_extra(tmp_path, stub_loader):
    extra = {"obs_noise": {"joint_vel": 1.75}}

    results = battery.run_battery(tmp_path, extra_env_overrides=extra)

    assert stub_loader["extra_env_overrides"] == extra
    assert results["env_overrides"] == extra


# -- the CLI -----------------------------------------------------------------


def _main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["battery", *argv])
    battery.main()


def test_set_without_out_is_an_argparse_error(tmp_path, monkeypatch, stub_loader):
    """A re-scored table never lands on the run's own battery.json."""
    with pytest.raises(SystemExit) as exc:
        _main(["--run", str(tmp_path), "--set", "obs_noise.joint_vel=1.75"], monkeypatch)

    assert exc.value.code == 2
    assert "run_dir" not in stub_loader
    assert not (tmp_path / "battery.json").exists()


def test_set_with_out_pointing_at_battery_json_is_an_argparse_error(
    tmp_path, monkeypatch, stub_loader
):
    with pytest.raises(SystemExit) as exc:
        _main(
            ["--run", str(tmp_path), "--set", "obs_noise.joint_vel=1.75",
             "--out", str(tmp_path / "battery.json")],
            monkeypatch,
        )

    assert exc.value.code == 2
    assert "run_dir" not in stub_loader


def test_a_malformed_set_is_an_argparse_error(tmp_path, monkeypatch, stub_loader):
    with pytest.raises(SystemExit) as exc:
        _main(
            ["--run", str(tmp_path), "--set", "joint_vel=1.75", "--out", str(tmp_path / "x.json")],
            monkeypatch,
        )

    assert exc.value.code == 2
    assert "run_dir" not in stub_loader


def test_set_with_out_writes_the_overrides_into_the_variant(tmp_path, monkeypatch, stub_loader):
    out = tmp_path / "battery_jv175.json"

    _main(
        ["--run", str(tmp_path), "--set", "obs_noise.joint_vel=1.75", "--out", str(out)],
        monkeypatch,
    )

    written = json.loads(out.read_text())
    assert written["env_overrides"] == {"obs_noise": {"joint_vel": 1.75}}
    assert stub_loader["extra_env_overrides"] == {"obs_noise": {"joint_vel": 1.75}}
    assert not (tmp_path / "battery.json").exists()


def test_without_set_the_written_json_has_no_env_overrides(tmp_path, monkeypatch, stub_loader):
    _main(["--run", str(tmp_path)], monkeypatch)

    written = json.loads((tmp_path / "battery.json").read_text())
    assert "env_overrides" not in written
    assert list(written) == ["run", "checkpoint", "contacts", "timestamp"]
    assert stub_loader["extra_env_overrides"] is None


def test_set_with_out_differing_only_in_case_is_refused_where_that_is_the_same_file(
    tmp_path, monkeypatch, stub_loader
):
    """On a case-insensitive filesystem Battery.json IS battery.json."""
    own = tmp_path / "battery.json"
    own.write_text("{}")
    alias = tmp_path / "Battery.json"
    if not alias.exists():
        pytest.skip("case-sensitive filesystem: Battery.json is a different file")

    with pytest.raises(SystemExit) as exc:
        _main(
            ["--run", str(tmp_path), "--set", "obs_noise.joint_vel=1.75", "--out", str(alias)],
            monkeypatch,
        )

    assert exc.value.code == 2
    assert "run_dir" not in stub_loader
    assert own.read_text() == "{}"


def test_set_refuses_a_hard_link_to_the_run_s_battery_json(tmp_path, monkeypatch, stub_loader):
    own = tmp_path / "battery.json"
    own.write_text("{}")
    link = tmp_path / "variant.json"
    link.hardlink_to(own)

    with pytest.raises(SystemExit) as exc:
        _main(
            ["--run", str(tmp_path), "--set", "obs_noise.joint_vel=1.75", "--out", str(link)],
            monkeypatch,
        )

    assert exc.value.code == 2
    assert own.read_text() == "{}"


def test_report_treats_env_overrides_as_metadata_not_a_scenario():
    from humanoid_lab.eval.report import render_markdown

    md = render_markdown(
        {
            "run": "r",
            "checkpoint": "100",
            "timestamp": "t",
            "env_overrides": {"obs_noise": {"joint_vel": 1.75}},
        }
    )

    assert "- env_overrides: {'obs_noise': {'joint_vel': 1.75}}" in md
    assert "| env_overrides |" not in md
