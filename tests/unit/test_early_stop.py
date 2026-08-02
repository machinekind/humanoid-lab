"""plateau_stop and record_eval: the early-stopping rule, model-free.

The decision is a pure function of a list of eval rewards, so the whole rule
is tested here on synthetic numbers. What feeds that list is record_eval,
which drops the progress calls that carry no eval reward -- also a pure
function of a metrics dict, also tested here.
"""

from hydra import compose, initialize_config_dir

from humanoid_lab import paths
from humanoid_lab.train import plateau_stop, record_eval

# The shipped defaults, the numbers a run gets when it only flips `enable`.
MIN_EVALS = 10
PATIENCE = 6
MIN_DELTA = 0.5


def _plateau(rewards, min_evals=MIN_EVALS, patience=PATIENCE, min_delta=MIN_DELTA):
    return plateau_stop(rewards, min_evals=min_evals, patience=patience, min_delta=min_delta)


# -- the length guard ------------------------------------------------------


def test_no_verdict_before_min_evals_however_flat_the_run():
    """A dead-flat run is the strongest possible plateau and still gets no
    verdict while the sample is short: an early plateau is normal, and one
    stop decision is worth more than a fast one."""
    flat = [10.0] * 9
    assert not _plateau(flat, min_evals=10, patience=3)
    assert _plateau(flat + [10.0], min_evals=10, patience=3)


def test_patience_plus_one_is_the_floor_when_it_exceeds_min_evals():
    """`patience` consecutive non-improvements need `patience + 1` evals to
    exist, so the guard is max(min_evals, patience + 1), not min_evals."""
    flat = [10.0] * 6
    assert not _plateau(flat, min_evals=3, patience=6)
    assert _plateau(flat + [10.0], min_evals=3, patience=6)


def test_no_verdict_on_an_empty_history():
    assert not _plateau([])


# -- improvement -----------------------------------------------------------


def test_growing_reward_never_stops():
    growing = [float(i) for i in range(30)]  # +1 per eval > min_delta
    assert not _plateau(growing)


def test_late_breakthrough_resets_patience():
    rewards = [50.0] * 12 + [55.0]  # new best on the last eval
    assert not _plateau(rewards, min_evals=5)


# -- plateau detection -----------------------------------------------------


def test_plateau_after_growth_stops():
    rewards = [float(i) for i in range(10)] + [9.1] * 6
    assert _plateau(rewards)


def test_exactly_patience_evals_without_a_new_best_stops():
    rewards = [float(i) for i in range(10)] + [9.1] * PATIENCE
    assert _plateau(rewards)


def test_one_eval_short_of_patience_does_not_stop():
    rewards = [float(i) for i in range(10)] + [9.1] * (PATIENCE - 1)
    assert not _plateau(rewards)


def test_noise_below_min_delta_does_not_reset_patience():
    """The whole reason min_delta exists: oscillating within +-0.4 of the
    best is eval noise, not progress, and must not restart the clock."""
    rewards = [50.0] + [50.0 + 0.4 * (-1) ** i for i in range(12)]
    assert _plateau(rewards, min_evals=5)


# -- the new-best boundary -------------------------------------------------


def test_a_gain_of_exactly_min_delta_is_not_a_new_best():
    """`> best + min_delta`, not `>=`. Only the last eval can save this run,
    and a gain of exactly min_delta does not."""
    rewards = [0.0] * 10 + [MIN_DELTA]
    assert _plateau(rewards)


def test_a_gain_just_over_min_delta_is_a_new_best():
    rewards = [0.0] * 10 + [MIN_DELTA + 1e-6]
    assert not _plateau(rewards)


# -- the eval filter -------------------------------------------------------


def _record(metrics, num_steps=1000):
    rewards, last_eval = [], {"steps": 0, "metrics": {}}
    was_eval = record_eval(metrics, num_steps, rewards, last_eval)
    return was_eval, rewards, last_eval


def test_a_training_metrics_call_records_nothing():
    """ppo.log_training_metrics routes EpisodeMetricsLogger calls through the
    same callback. They carry no eval reward, so a NaN would enter the
    history -- and NaN never beats the running best, so a handful would fake
    a plateau and cut the run short."""
    was_eval, rewards, last_eval = _record({"training/sps": 1.2e6})

    assert was_eval is False
    assert rewards == []
    assert last_eval == {"steps": 0, "metrics": {}}


def test_an_eval_call_appends_the_reward_and_becomes_the_last_eval():
    metrics = {"eval/episode_reward": 12.5, "eval/avg_episode_length": 400}
    was_eval, rewards, last_eval = _record(metrics, num_steps=1000)

    assert was_eval is True
    assert rewards == [12.5]
    assert last_eval == {"steps": 1000, "metrics": metrics}


def test_a_training_call_does_not_overwrite_an_earlier_eval():
    """The run summary reports from last_eval. A training call landing after
    the final eval must not replace the numbers it reports."""
    rewards, last_eval = [], {"steps": 0, "metrics": {}}
    record_eval({"eval/episode_reward": 12.5}, 1000, rewards, last_eval)
    record_eval({"training/sps": 1.2e6}, 1100, rewards, last_eval)

    assert rewards == [12.5]
    assert last_eval["steps"] == 1000
    assert last_eval["metrics"] == {"eval/episode_reward": 12.5}


# -- the shipped config ----------------------------------------------------


def test_the_shipped_block_is_off_and_carries_the_documented_defaults():
    with initialize_config_dir(version_base=None, config_dir=str(paths.CONFIGS_DIR)):
        cfg = compose(config_name="config")

    assert cfg.early_stop.enable is False
    assert cfg.early_stop.min_evals == MIN_EVALS
    assert cfg.early_stop.patience == PATIENCE
    assert cfg.early_stop.min_delta == MIN_DELTA
