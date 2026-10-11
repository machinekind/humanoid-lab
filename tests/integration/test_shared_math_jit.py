"""The shared task math under jax.jit.

tests/unit runs each batched-equals-stacked test on numpy and eager jax at
batch shapes (), (3,) and (2, 3). This file runs the same tests under jit at
batch (2, 3). jit compiles compute_rewards, gait_state.update and the
readings, which is too slow for the unit edit loop.

A batched test is any test in MODULES that takes `batch` and `mode`. Its other
parametrize values are expanded here, so a new batched test in one of those
modules runs under jit with no edit to this file. jit layouts differ from
eager ones in their last bits, so task_toys.assert_batched_matches_stacked
compares with rtol=1e-6.
"""

import inspect
import itertools
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))

import test_dr_transform
import test_gait_state
import test_progress_clock
import test_push
import test_reward_compose
import test_signals

MODULES = (test_dr_transform, test_gait_state, test_progress_clock, test_push, test_reward_compose, test_signals)
BATCH = (2, 3)


def _is_batched_test(name, fn):
    return name.startswith("test_") and inspect.isfunction(fn) and {"batch", "mode"} <= set(inspect.signature(fn).parameters)


def _batched_tests(module):
    """(name, callable, parametrize marks) for each batched test in module."""
    for name, obj in sorted(vars(module).items()):
        if _is_batched_test(name, obj):
            yield name, obj, getattr(obj, "pytestmark", [])
        elif name.startswith("Test") and inspect.isclass(obj):
            for method, fn in sorted(vars(obj).items()):
                if _is_batched_test(method, fn):
                    marks = getattr(obj, "pytestmark", []) + getattr(fn, "pytestmark", [])
                    yield f"{name}.{method}", getattr(obj(), method), marks


def _label(name, value, i):
    if isinstance(value, (str, bool, int, float, type(None))):
        return f"{name}={value}"
    return f"{name}{i}"


def _grid(marks):
    """(kwargs, id) for every combination of the parametrize values other than
    batch and mode."""
    axes = []
    for mark in marks:
        if mark.name != "parametrize":
            continue
        names = mark.args[0]
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        if names in (["batch"], ["mode"]):
            continue
        ids = mark.kwargs.get("ids")
        rows = []
        for i, value in enumerate(mark.args[1]):
            values = value if len(names) > 1 else (value,)
            if isinstance(ids, (list, tuple)):
                label = str(ids[i])
            else:
                label = "-".join(_label(n, v, i) for n, v in zip(names, values))
            rows.append((dict(zip(names, values)), label))
        axes.append(rows)
    for combo in itertools.product(*axes):
        yield {k: v for kwargs, _ in combo for k, v in kwargs.items()}, "-".join(label for _, label in combo)


def _cases():
    for module in MODULES:
        for name, test, marks in _batched_tests(module):
            for kwargs, label in _grid(marks):
                yield pytest.param(test, kwargs, id="-".join(filter(None, [module.__name__, name, label])))


CASES = list(_cases())


@pytest.mark.parametrize("test,kwargs", CASES)
def test_batched_equals_stacked_under_jit(test, kwargs):
    test(**{**kwargs, "batch": BATCH, "mode": "jit"})


def test_every_module_has_batched_tests():
    found = {case.id.split("-")[0] for case in CASES}
    assert found == {m.__name__ for m in MODULES}
