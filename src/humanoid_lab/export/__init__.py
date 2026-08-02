"""Deployment artifacts for a trained checkpoint.

`policy.py` is the exporter behind `run.sh export`. `runtime.py` is the
numpy deploy runtime it validates against, and the reference a robot-side
codebase vendors. The contract they write is in
`humanoid_lab/deploy_contract.py`; docs/deploy.md describes both.
"""
