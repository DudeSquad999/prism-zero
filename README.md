# PRISM ZERO

### Generate locally. Verify explicitly. Train deliberately.

Offline-first neural candidate generation, symbolic verification and accepted-only
JSONL curation for local PyTorch LoRA SFT. No hosted model API or key required.

## Features

- Zero API costs at runtime using local checkpoints.
- Bounded generation, rejection feedback and retry loops.
- Python AST syntax, strict JSON Schema, timeout-bounded regex and exact rational checks.
- Accepted-only, deduplicated JSONL with task and verifier-version identifiers.
- Optional local Transformers inference and PyTorch/PEFT LoRA training.
- No eval, exec, generated-code execution or remote schema resolution.

Compute, electricity, dependencies and acquiring weights still have costs.
This does not reproduce o1 or DeepSeek-R1. Think tags serialize generated
explanations, not private internal computation. A passing answer does not prove
the explanation. Retry feedback does not update weights.

## Quick start

Python 3.11 or newer:

```sh
python -m venv .venv
# POSIX: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python examples/quickstart.py
python -m pytest -q
```

The labeled scripted model returns 41, receives rejection feedback, returns 42
on attempt two and saves one accepted record to prism_zero_dataset.jsonl. No
weights are downloaded. This tests orchestration, not learned self-correction.
Re-running the demo deduplicates the same accepted example.

```python
from pathlib import Path
from examples.quickstart import ScriptedModel
from prism_zero.engine import PrismZeroEngine, Rule
from prism_zero.verifiers import MathEqualityVerifier

engine = PrismZeroEngine(
    ScriptedModel(), [Rule(MathEqualityVerifier("6*7"))],
    Path("prism_zero_dataset.jsonl"), task_id="multiply-6-by-7",
)
result = engine.run("What is six times seven? Return an integer.")
assert result.accepted and result.attempts == 2
```

## Architecture

```text
Trusted task + independent constraints
                  |
          local generation
                  |
       think + answer envelope
                  |
          symbolic checks
             /        \
           fail       pass
            |           |
    feedback + retry  atomic JSONL
            |           |
    reject at limit   review + local SFT
                        |
                  LoRA adapter
                        |
              held-out evaluation
```

## Repository map

```text
prism_zero/
  __init__.py
  engine.py
  verifiers.py
  backends.py
  trainer.py
examples/quickstart.py
tests/test_core.py
.github/workflows/tests.yml
pyproject.toml
README.md
CONTRIBUTING.md
SECURITY.md
VALIDATION.md
LICENSE
.gitignore
```

## Real local generation

Install ML dependencies before going offline:

```sh
python -m pip install -e ".[ml,dev]"
```

Place a trusted compatible checkpoint in models/base, including safetensors,
tokenizer assets and a chat template. No model is bundled.

```python
from pathlib import Path
from prism_zero.backends import LocalHFGenerator
from prism_zero.engine import PrismZeroEngine, Rule
from prism_zero.verifiers import MathEqualityVerifier

model = LocalHFGenerator("models/base", device="cpu")
engine = PrismZeroEngine(model, [Rule(MathEqualityVerifier("42"))],
                         Path("prism_zero_dataset.jsonl"), task_id="multiply-6-by-7")
result = engine.run("Compute 6 * 7. Return an integer.")
print(result.accepted, result.attempts)
```

Loading uses local_files_only=True, trust_remote_code=False and safetensors.
Missing assets fail rather than download. For defense in depth set
HF_HUB_OFFLINE=1 and HF_HUB_DISABLE_TELEMETRY=1 before startup. Enforce strict
network isolation at the OS/container layer when required. Real models are not
guaranteed to self-correct in two attempts. Memory depends on the checkpoint.

## Verifier contracts

AST checks syntax, not runtime correctness or safety. JSON Schema checks
structure, not factual truth; format annotations are not asserted. References
are disabled, and duplicate keys, nonfinite values and numeric overflow are
rejected. Trusted schemas still require complexity review. Regex performs full
matching with an input bound and timeout; near-timeout outcomes depend on load.

Math checks an independently configured target using exact fractions. Supported:
integers, unary +/-, parentheses and +, -, *, /. Names, calls, attributes, powers,
floats and model-supplied equalities are rejected. No candidate is evaluated with eval.

Rule(verifier, target="trace") checks an explanation. Every configured rule must
pass, and at least one answer rule is required. Backend and verifier implementation
errors propagate instead of being mislabeled as content failures.

## Dataset and concurrency

Records contain schema_version, id, task_id, verifier_version, prompt, trace,
answer, attempts and checks. Change verifier_version when rule semantics change.
IDs hash task/version/prompt/accepted content; they are not signed attestations.
Rejected outputs are not exported. Keep datasets trusted and access-controlled.

Use one writer process and one store instance per path. The per-instance lock
serializes storage threads; generator thread safety remains backend-specific.
Atomic replacement prevents partial writes, not multi-process lost updates or
all power-loss failures. The entire file is rewritten: this is for small local
collections, not distributed ingestion. Invalid existing JSONL aborts an update
without replacing the existing file.

## Local SFT

Review and curate a meaningful dataset, then run:

```sh
prism-zero-train --model models/base --dataset prism_zero_dataset.jsonl --output adapters/run-001 --device cpu --epochs 1 --batch-size 1 --max-length 1024
```

Use --device cuda only with a compatible installation. LoRA reduces trainable
parameters, but this trainer loads the base model in full precision. Quantization,
multi-device training and resume are not implemented. --target-modules accepts
all-linear or architecture-specific comma-separated names.

The inference chat template is reused. Prompt and padding labels use -100;
position-based masking preserves genuine EOS targets. The causal model shifts
labels internally. Oversized examples and unstable assistant boundaries fail
rather than silently truncate answers.

The output directory must not exist. Output contains an adapter, tokenizer,
training manifest and a final COMPLETE marker, not a full base checkpoint.
An exception can leave an incomplete directory. Do not promote it. To use a
completed adapter, load the original base checkpoint then use
peft.PeftModel.from_pretrained with the local adapter and local_files_only=True.
LocalHFGenerator does not automatically load adapters.

Training loss is not quality evaluation. No automatic promotion or learned
improvement is claimed. Self-generated data can amplify errors or overfit; keep
independent held-out tasks and compare against the base checkpoint.

## Offline installation

Pip normally resolves dependencies from an index. For an airgapped system,
prepare and transfer a wheelhouse, build tooling and model assets for the same
platform, Python and accelerator stack. Install with --no-index --find-links
wheelhouse. This repository does not bundle dependency wheels or model weights.

## Contributing and license

Read CONTRIBUTING.md and SECURITY.md, add negative regressions, and report exact
commands and outcomes. Never execute generated candidate code in a verifier.
See VALIDATION.md for checks actually performed and remaining validation gaps.
MIT applies to this source; models, datasets and dependencies retain their terms.
