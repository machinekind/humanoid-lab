"""sizing/collect.py's env overrides, model-free.

Importing sizing.collect builds nothing. The override builder is plain dict
logic over run.json's recorded hydra_config.
"""

from humanoid_lab.sizing.collect import sizing_env_overrides


def _run(env_block):
    return {"hydra_config": {"task": {"env": env_block}}}


def test_sizing_forces_the_command_ramp_off():
    overrides = sizing_env_overrides(
        _run({"command": {"ramp": {"lin": 0.5, "yaw": 1.0}, "resample_steps": 300}})
    )

    assert overrides["command"]["ramp"] == {"lin": 0.0, "yaw": 0.0}
    assert overrides["command"]["resample_steps"] == 300


def test_sizing_keeps_the_run_s_other_env_keys_and_its_pushes():
    push = {"enable": True, "interval_steps": 150, "vel": 0.6}
    run = _run({"push": push, "no_progress": {"enable": True}, "real_pose_ref": True})

    overrides = sizing_env_overrides(run)

    assert overrides["push"] == push
    assert overrides["no_progress"] == {"enable": True}
    assert overrides["real_pose_ref"] is True
    assert overrides["command"] == {"ramp": {"lin": 0.0, "yaw": 0.0}}


def test_sizing_does_not_mutate_the_run():
    run = _run({"command": {"ramp": {"lin": 0.5, "yaw": 1.0}}})

    sizing_env_overrides(run)

    assert run["hydra_config"]["task"]["env"]["command"]["ramp"] == {"lin": 0.5, "yaw": 1.0}


def test_a_run_with_no_env_block_gets_the_ramp_off():
    run = {"hydra_config": {"task": {}}}

    assert sizing_env_overrides(run) == {"command": {"ramp": {"lin": 0.0, "yaw": 0.0}}}
