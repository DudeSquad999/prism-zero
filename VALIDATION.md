# Validation

The regenerated GitHub bootstrap sources were checked before committing:

- Core tests: 29 passed in the Linux Python execution environment.
- compileall across generated sources: passed.
- Scripted quickstart: accepted attempt two, exit 0.
- Repeated quickstart: one accepted record remained.

Not run: real-model inference, PyTorch/PEFT training, adapter reload, GPU checks,
Windows/macOS CI, static type checking, dependency audit or held-out evaluation.
No checkpoint is bundled. PyTorch and PEFT were unavailable in the validation
environment. Token-boundary tests use a tokenizer double. No model-quality
improvement is established. GitHub Actions outcomes require separate inspection.

The earlier downloadable archive was a separate 28-test build; this GitHub
bootstrap was regenerated and checked separately. Dependency ranges are not a
fully resolved lockfile. These results do not certify production readiness.
