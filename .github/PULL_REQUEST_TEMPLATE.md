## Summary

Describe the behavior change and why it is needed.

## Verification

- [ ] `python -m pytest -q -m "not ml"`
- [ ] `python examples/quickstart.py`
- [ ] `python -m compileall -q prism_zero examples tests`
- [ ] Full ML checks run when applicable, or limitations are stated.
- [ ] Documentation and examples are updated.

Test commands and results:

## Data and model safety

- [ ] No private datasets, model weights, credentials, or sensitive prompts are included.
- [ ] Generated code or untrusted outputs are not executed by verifiers.
- [ ] Training, evaluation, and curation boundaries remain explicit.
- [ ] Claims distinguish scripted fixtures from real-model evaluation.

## Compatibility

List Python, Transformers, PyTorch, PEFT, and platform considerations.
