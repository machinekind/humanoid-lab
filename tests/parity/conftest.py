"""Shared setup for tests/parity, the JAX-against-Torch checks.

The suite runs in its own pytest process (`./run.sh test-parity`, and the
second process of `./run.sh test-all`), because JAX_PLATFORMS must be set
before the first jax import. This file sets it to cpu for a bare pytest run
too. It imports no torch. Each test module that needs torch passes "torch"
to `pytest.importorskip` itself, so the suite skips cleanly when the
optional Torch install is missing.

Two command-line flags work when pytest is given tests/parity or a file in
it. `--parity-report` prints the worst error of each parity check at the end
of the run. `--parity-device` names the torch device for the Torch side, for
example cuda:0. It defaults to cpu. JAX stays on CPU. This file imports no
torch, so a test module that uses the device checks a CUDA request itself.
"""

import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


def pytest_addoption(parser):
    group = parser.getgroup("parity")
    group.addoption(
        "--parity-report",
        action="store_true",
        help="print the worst error of each parity check at the end",
    )
    group.addoption(
        "--parity-device",
        default="cpu",
        help="torch device for the Torch side, for example cuda:0. JAX stays on CPU.",
    )


@pytest.fixture(scope="session")
def parity_device(request) -> str:
    return request.config.getoption("--parity-device")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--parity-report", default=False):
        return
    import parity_report

    terminalreporter.write_sep("=", "parity report")
    for line in parity_report.table().splitlines():
        terminalreporter.write_line(line)
