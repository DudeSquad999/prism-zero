# Validation

## 2026-11-12 — hardening, ML integration, evaluation, CLI revision

Checked on Linux (Debian) with Python 3.12.15, pytest 9.2.0, and CPU-only
torch 2.15.0, transformers 4.57.6, peft 0.21.2, safetensors 0.8.0:

- Core, dataset, evaluation, and CLI tests: 115 passed without ML dependencies
  installed (the ML suite skips itself); 138 passed with the ML stack installed.
- ML integration suite (23 tests): a tiny randomly-initialized GPT-2 checkpoint
  with a real WordLevel tokenizer and chat template was constructed locally.
  Local loading, generation, one-epoch LoRA training, adapter saving, adapter
  reload onto a fresh base copy, and reloaded generation all executed on CPU
  with no network access. LoRA parameters changed, base weights stayed frozen,
  and losses were finite. The random model demonstrates plumbing only; it says
  nothing about task quality.
- Scripted quickstart: accepted attempt two, exit 0; repeated run kept one
  accepted record (schema version 2 with check targets and provenance).
- Clean-install check: the wheel was built and installed into a fresh virtual
  environment; `prism-zero --help`, `prism-zero doctor --json`, `prism-zero run`,
  and `prism-zero dataset validate` worked from outside the source checkout
  without torch installed. Core imports do not require PyTorch.
- `prism-zero evaluate` was exercised end to end with the tiny checkpoint and
  a trained adapter: base and adapter conditions ran, JSON results and the
  terminal summary were produced, and overlap against the training dataset was
  reported. The random model produced no passing answers; that is expected and
  is not a benchmark.

Not run: real pretrained-model inference, CUDA or MPS execution, macOS or
Windows CI for this revision (workflows are defined but their GitHub Actions
outcomes require separate inspection), static type checking, dependency audit,
and any model-quality comparison. No model-quality improvement is claimed; a
successful training run and a successful scripted demo are not evidence of
better answers. Historical sections below describe earlier revisions and do
not certify this one.

## Bootstrap revision (historical)

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
