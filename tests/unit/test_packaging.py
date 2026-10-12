"""Packaging and Windows checks that read files as text.

Git for Windows checks text files out with CRLF unless .gitattributes says
otherwise, and bash then fails on the carriage returns. A Windows venv puts
the interpreter at .venv/Scripts/python.exe. The torch_warp extra must not
pull PyPI torch onto Linux, where it brings CUDA 13 next to jax's CUDA 12.
"""

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN_SH = REPO / "run.sh"


def _pyproject():
    with (REPO / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)


def _shell_scripts():
    return [RUN_SH, *sorted((REPO / "jobs").glob("*.sh"))]


def test_shell_scripts_have_no_carriage_return():
    scripts = _shell_scripts()
    assert len(scripts) > 1
    offenders = [p.name for p in scripts if b"\r" in p.read_bytes()]
    assert not offenders, f"these scripts carry CRLF line endings: {offenders}"


def test_gitattributes_keeps_shell_scripts_at_lf():
    lines = (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
    rules = [
        line.split() for line in lines if line.strip() and not line.startswith("#")
    ]
    assert ["*.sh", "text", "eol=lf"] in rules


def test_the_torch_warp_extra_skips_linux_and_carries_psutil():
    extra = _pyproject()["project"]["optional-dependencies"]["torch_warp"]
    torch = [d for d in extra if re.match(r"torch\b", d)]
    assert len(torch) == 1
    name, _, marker = torch[0].partition(";")
    assert name.strip().startswith("torch")
    assert marker.strip().replace('"', "'") == "sys_platform != 'linux'"
    assert any(re.match(r"psutil\s*>=", d) for d in extra)


def test_the_pins_stay():
    deps = [d.replace('"', "'") for d in _pyproject()["project"]["dependencies"]]
    assert "warp-lang==1.13.0" in deps
    assert "jax==0.9.2; sys_platform != 'linux'" in deps
    assert "jax[cuda12]==0.9.2; sys_platform == 'linux'" in deps


def test_run_sh_falls_back_to_the_windows_venv():
    text = RUN_SH.read_text(encoding="utf-8")
    assert "PY=.venv/bin/python\n" in text
    assert (
        'if [[ ! -x "$PY" && -x .venv/Scripts/python.exe ]]; then '
        "PY=.venv/Scripts/python.exe; fi"
    ) in text


def _run_sh_verbs():
    text = RUN_SH.read_text(encoding="utf-8")
    return re.findall(r"^  ([a-z][a-z-]*)\)", text, flags=re.MULTILINE)


def _docs_verbs_table():
    text = (REPO / "docs" / "configuration.md").read_text(encoding="utf-8")
    section = text.split("## `run.sh` verbs", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([a-z][a-z-]*)` \|", section, flags=re.MULTILINE))


def test_every_run_sh_verb_is_in_the_docs_table():
    verbs = _run_sh_verbs()
    assert {"train", "train-warp", "bench-warp", "test-parity"} <= set(verbs)
    missing = sorted(set(verbs) - _docs_verbs_table())
    assert not missing, f"run.sh verbs missing from docs/configuration.md: {missing}"
