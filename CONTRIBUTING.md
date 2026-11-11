# Contributing

Use Python 3.11+. Install `python -m pip install -e ".[dev]"`, then run
`python -m pytest -q` and `python -m compileall -q prism_zero examples tests`.
Keep core tests offline with fake model outputs. Test model/template/trainer
compatibility separately using trusted local weights. Never execute candidates.
Add negative cases and input bounds to every verifier. Version dataset semantics.
Report hardware, dependency versions, exact commands, outcomes and skipped checks.
Do not claim improvement from a scripted demo or training loss. Preserve licenses
and keep private traces, model files and datasets out of public commits.
