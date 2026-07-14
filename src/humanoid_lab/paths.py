"""Repo-relative paths shared by every module.

Robot-specific constants (XML paths, leg-name tuples, ...) do not live here:
each robot is data, resolved through a RobotSpec (robots/<name>/robot.yaml),
not a module-level constant.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_DIR = REPO_ROOT
ROBOTS_DIR = REPO_ROOT / "robots"
CONFIGS_DIR = REPO_ROOT / "configs"
RUNS_DIR = REPO_ROOT / "runs"
