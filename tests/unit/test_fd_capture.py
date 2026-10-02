"""fd_capture: the descriptor capture and the MJWarp message counter.

MJWarp's device printf cannot run here. Its output reaches the process
through file descriptor 1, the way `os.write(1, ...)` and C's `printf` do,
so those two stand in for it.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import pathlib
import re

import pytest

from humanoid_lab import fd_capture

# One line per message as MJWarp prints it, numbers filled in.
WARP_LINES = {
    "hfield_overflow": "height field collision overflow, number of collisions >= 50 - please adjust resolution:",
    "ccd_overflow": "CCD overflow - please increase naccdmax to 4097",
    "narrowphase_overflow": "narrowphase overflow - please increase nconmax to 130 or naconmax to 1040",
    "broadphase_overflow": "broadphase overflow - please increase nconmax to 140 or naconmax to 1120",
    "nefc_overflow": "nefc overflow - please increase njmax to 700",
    "njmax_nnz_overflow": "njmax_nnz overflow - please increase njmax_nnz to 9000",
    "epa_horizon": "Warning: EPA horizon = 24 isn't large enough.",
}


# MJWarp overflow messages that are deliberately not counted. Contact match
# overflow concerns contact sensors only, and no robot here has one.
NOT_COUNTED = ("contact match overflow",)


def _mjwarp_printf_formats():
    """Every `wp.printf` format string in the installed MJWarp source, read
    as text. `\\s*` spans the newline of a call split across lines."""
    spec = importlib.util.find_spec("mujoco")
    src = pathlib.Path(spec.submodule_search_locations[0]) / "mjx/third_party/mujoco_warp/_src"
    assert src.is_dir(), src  # a moved package must fail, never skip
    return [f for p in src.glob("*.py") for f in re.findall(r'wp\.printf\(\s*"([^"]*)"', p.read_text())]


def _first_line_pattern(fmt):
    """A format's first line as a regex, its integer conversions as digits."""
    first = fmt.split("\\n")[0].rstrip()
    return re.escape(first).replace("%u", r"\d+").replace("%d", r"\d+")


def test_messages_match_the_installed_mjwarp():
    """Each counted text is in an installed MJWarp format, and each sample
    line above is a format's first line with numbers filled in. Every
    overflow format is counted or listed in NOT_COUNTED."""
    formats = _mjwarp_printf_formats()
    for name, message in fd_capture.WARP_MESSAGES.items():
        assert any(message in f for f in formats), name
        assert any(re.fullmatch(_first_line_pattern(f), WARP_LINES[name]) for f in formats), name
    for f in formats:
        if "overflow" in f:
            assert any(m in f for m in (*fd_capture.WARP_MESSAGES.values(), *NOT_COUNTED)), f


def test_counter_finds_every_message_kind():
    assert set(WARP_LINES) == set(fd_capture.WARP_MESSAGES)
    text = "\n".join(line for name, line in WARP_LINES.items() for _ in range(1 + len(name) % 3))
    counts = fd_capture.count_warp_messages(text)
    assert counts == {name: 1 + len(name) % 3 for name in WARP_LINES}
    assert fd_capture.count_warp_messages("all clear") == dict.fromkeys(WARP_LINES, 0)
    assert "epa_horizon" not in fd_capture.GATING
    assert set(fd_capture.GATING) == set(WARP_LINES) - {"epa_horizon"}


def test_capture_takes_descriptor_writes_and_puts_them_back(capfd):
    with fd_capture.capture_fd1(True) as lines:
        os.write(1, (WARP_LINES["nefc_overflow"] + "\n").encode())
        os.write(1, b"second line\n")
        assert lines == []
    assert lines == [WARP_LINES["nefc_overflow"], "second line"]
    assert fd_capture.count_warp_messages("\n".join(lines))["nefc_overflow"] == 1
    out = capfd.readouterr().out
    assert WARP_LINES["nefc_overflow"] in out and "second line" in out
    # fd 1 is back where it was.
    os.write(1, b"after\n")
    assert capfd.readouterr().out == "after\n"


def test_capture_puts_fd1_back_when_the_block_raises(capfd):
    """The block's own exception reaches the caller, and the lines written
    before it are captured and written back."""
    with pytest.raises(RuntimeError, match="boom"), fd_capture.capture_fd1(True) as lines:
        os.write(1, b"before the error\n")
        raise RuntimeError("boom")
    assert lines == ["before the error"]
    assert "before the error" in capfd.readouterr().out
    os.write(1, b"after\n")
    assert capfd.readouterr().out == "after\n"


def test_capture_takes_c_stdio_writes(capfd):
    # The line is the format string itself: it holds no conversion, and a
    # variadic call through ctypes is not portable.
    libc = ctypes.CDLL(None)
    with fd_capture.capture_fd1(True) as lines:
        libc.printf((WARP_LINES["hfield_overflow"] + "\n").encode())
    assert lines == [WARP_LINES["hfield_overflow"]]
    assert WARP_LINES["hfield_overflow"] in capfd.readouterr().out


def test_capture_disabled_leaves_stdout_alone(capfd):
    with fd_capture.capture_fd1(False) as lines:
        os.write(1, b"straight through\n")
        assert capfd.readouterr().out == "straight through\n"
    assert lines == []
    assert capfd.readouterr().out == ""


def test_capture_reemits_a_bounded_head(capfd):
    n = fd_capture.REEMIT_LINES + 5
    with fd_capture.capture_fd1(True) as lines:
        os.write(1, "".join(f"line {i}\n" for i in range(n)).encode())
    assert lines == [f"line {i}" for i in range(n)]
    out = capfd.readouterr().out.splitlines()
    assert out == [f"line {i}" for i in range(fd_capture.REEMIT_LINES)] + ["... 5 more lines"]
