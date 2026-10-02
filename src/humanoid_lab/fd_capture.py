"""Capture what MJWarp prints from the device, and count its overflow
messages.

MJWarp reports buffer overflow with `wp.printf` in device code. The text
goes to file descriptor 1. Python's `sys.stdout` never sees it, so
`contextlib.redirect_stdout` misses it, and only a `dup2` of the
descriptor catches it.

Counts are lower bounds. The device printf buffer has a fixed size, and a
storm of messages can overflow it. A gate needs only "more than zero".
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import tempfile
from collections.abc import Iterator

# Every MJWarp message that reports a dropped contact, pair or constraint
# row, keyed by a short name. The text is the message's fixed prefix.
WARP_MESSAGES = {
    "hfield_overflow": "height field collision overflow",
    "ccd_overflow": "CCD overflow",
    "narrowphase_overflow": "narrowphase overflow",
    "broadphase_overflow": "broadphase overflow",
    "nefc_overflow": "nefc overflow",
    "njmax_nnz_overflow": "njmax_nnz overflow",
    # EPA's horizon, a fixed 24 entries in MJWarp, ran out. That pair gets
    # no contact on that step. No budget enlarges the horizon, so it is
    # reported and never gates.
    "epa_horizon": "EPA horizon",
}
# The messages that mean the physics lost something.
GATING = tuple(k for k in WARP_MESSAGES if k != "epa_horizon")
# Lines of captured output written back after the capture ends.
REEMIT_LINES = 20


def count_warp_messages(text: str) -> dict[str, int]:
    """Occurrences of each WARP_MESSAGES text in `text`."""
    return {name: text.count(message) for name, message in WARP_MESSAGES.items()}


def _flush_c_stdio() -> None:
    """Flush every C stdio stream: printf from C writes through them."""
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL(None).fflush(None)


def _sync_warp() -> None:
    """Wait for the device, so its printf buffer reaches fd 1. A no-op
    without warp, and on a host with no device to wait for."""
    try:
        import warp
    except ImportError:
        return
    with contextlib.suppress(Exception):
        warp.synchronize()


@contextlib.contextmanager
def capture_fd1(enabled: bool) -> Iterator[list[str]]:
    """Send file descriptor 1 to a temp file for the block.

    Yields a list that holds the captured lines once the block ends. The
    caller blocks on its device work (`jax.block_until_ready`) inside the
    block. On exit the device is synchronized, C stdio is flushed, fd 1 is
    restored, and the first REEMIT_LINES lines are written back, plus a
    count of the rest. Disabled, it yields an empty list and leaves fd 1
    alone."""
    lines: list[str] = []
    if not enabled:
        yield lines
        return
    sys.stdout.flush()
    _flush_c_stdio()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 1)
        try:
            yield lines
        finally:
            _sync_warp()
            sys.stdout.flush()
            _flush_c_stdio()
            os.dup2(saved, 1)
            os.close(saved)
            tmp.seek(0)
            lines.extend(tmp.read().decode("utf-8", errors="replace").splitlines())
            head = lines[:REEMIT_LINES]
            if head:
                rest = len(lines) - len(head)
                tail = [f"... {rest} more lines"] if rest else []
                sys.stdout.write("\n".join(head + tail) + "\n")
                sys.stdout.flush()
