"""Guard the unit/integration split.

`./run.sh test` runs tests/unit only, and its whole point is that it finishes
in seconds so it is usable in an edit loop. That holds only while nothing in
tests/unit builds or steps a model: compiling an MjSpec, uploading a model to
the device, constructing a task env, or stepping MJX costs anywhere from
hundreds of milliseconds to tens of seconds each. A test that needs any of
those belongs in tests/integration, which `./run.sh test-slow` runs.

Ported from w01-tek's training/tests/unit/test_suite_split.py, with this
repo's names substituted for w01-tek's.
"""

import re
from pathlib import Path

# Written in pieces so this guard does not match its own source. The scan
# below also skips this file by name; the splitting keeps a future
# guard-adjacent module from tripping over the pattern list.
FORBIDDEN = {
    "env instantiation": r"(Joystick|Sizing)\(",
    "registry env": r"make" + r"_env\(",
    "spec build": r"build" + r"_spec\(",
    "spec compile": r"compile" + r"_spec",
    "model upload": r"mjx\.put" + r"_model",
    "scene parse": r"MjModel\.from" + r"_xml|from" + r"_xml_(path|string)",
    "spec parse": r"MjSpec\.from" + r"_",
    "mj data": r"mujoco\.MjData",
    "mj forward": r"mj" + r"_forward",
    "mj step": r"mj" + r"_step\(",
    "mjx forward": r"mjx\.forward",
    "mjx step": r"mjx\.step",
    # The two composite entry points. Each is a whole pipeline of the lines
    # above -- build_model's `build` compiles a spec and writes an XML;
    # eval/battery's `load_checkpoint_policy` builds an env AND restores a
    # checkpoint. Matched as CALLS, so a unit test may still name either one
    # in prose or monkeypatch it away by name (tests/unit/test_eval_video.py
    # does exactly that to stop render_video before it loads anything).
    "model build": r"build" + r"_model\.build\(|from humanoid_lab\.build" + r"_model import build\b",
    "checkpoint policy": r"load" + r"_checkpoint" + r"_policy\(",
}

UNIT_DIR = Path(__file__).parent
TESTS_DIR = UNIT_DIR.parent


def test_no_unit_test_builds_or_steps_a_model():
    offenders = {}
    for path in sorted(UNIT_DIR.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        source = path.read_text()
        hits = [label for label, pat in FORBIDDEN.items() if re.search(pat, source)]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        f"these tests/unit files touch a real model: {offenders}. Move them to "
        "tests/integration -- see this module's docstring."
    )


def test_the_split_directories_both_exist_and_are_populated():
    integration = TESTS_DIR / "integration"
    assert integration.is_dir(), "tests/integration does not exist"
    assert len(list(UNIT_DIR.glob("test_*.py"))) > 5
    assert len(list(integration.glob("test_*.py"))) > 5


def test_no_test_file_sits_outside_the_split():
    stray = sorted(p.name for p in TESTS_DIR.glob("test_*.py"))
    assert not stray, (
        f"these test files sit directly in tests/ and belong to neither suite: "
        f"{stray}. Classify each one into tests/unit or tests/integration."
    )
