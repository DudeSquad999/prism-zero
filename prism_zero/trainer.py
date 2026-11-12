"""Explicit, single-device local LoRA SFT with assistant-only labels.

Training completion states are distinct: training finished, serialization
finished, artifact reload validated, quality evaluation performed. Only the
first three are established here; evaluation is always separate.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from .backends import build_messages, local_model_path
from .candidates import Candidate
from .dataset import dataset_fingerprint, read_dataset

LOGGER = logging.getLogger(__name__)
CONTENT_MODES = ('trace-and-answer', 'answer-only')
MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_DEVICES = ('cpu', 'cuda', 'mps', 'xpu')


def library_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                                text=True, timeout=5, cwd=Path(__file__).resolve().parent)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def checkpoint_fingerprint(path: Path) -> str:
    """Identity digest over configuration and tokenizer text plus weight file
    names and sizes; not a cryptographic attestation of the weights."""
    digest = hashlib.sha256()
    for name in sorted(os.listdir(path)):
        full = path / name
        if not full.is_file():
            continue
        digest.update(name.encode('utf-8'))
        if name.endswith(('.json', '.txt')):
            digest.update(full.read_bytes())
        else:
            digest.update(f':{full.stat().st_size}'.encode('utf-8'))
    return digest.hexdigest()


def require_int(name: str, value: Any, low: int = 1, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be a plain integer')
    if value < low or (high is not None and value > high):
        upper = f' and {high}' if high is not None else ''
        raise ValueError(f'{name} must be at least {low}{upper}; got {value}')
    return value


def validate_device(device: str) -> str:
    import torch
    if not isinstance(device, str) or not device.strip():
        raise ValueError('Device must be a nonempty string such as cpu, cuda, or cuda:0')
    name, _, index = device.partition(':')
    if name not in SUPPORTED_DEVICES or (index and not index.isdigit()):
        raise ValueError(f'Unsupported device {device!r}; expected one of {SUPPORTED_DEVICES} '
                         'with an optional index such as cuda:0')
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but torch.cuda.is_available() is false')
    if name == 'mps' and not getattr(torch.backends, 'mps', None):
        raise ValueError('MPS requested but this torch build lacks MPS support')
    return device


def response_text(record: dict[str, Any], content_mode: str) -> str:
    if content_mode not in CONTENT_MODES:
        raise ValueError(f'train-content must be one of {CONTENT_MODES}')
    if content_mode == 'answer-only':
        return f'<answer>{record["answer"]}</answer>'
    return Candidate(record['trace'], record['answer']).render()


def encode_record(tokenizer: Any, record: dict[str, Any], max_length: int,
                  content_mode: str = 'trace-and-answer') -> dict[str, list[int]]:
    """Use the inference template, mask prompts and refuse truncation."""
    messages = build_messages(record['prompt'])
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    response = response_text(record, content_mode)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': response}],
                                        tokenize=True, add_generation_prompt=False)
    if not prefix or full[:len(prefix)] != prefix:
        raise ValueError('Chat template has no stable assistant boundary')
    if len(full) > max_length or len(full) <= len(prefix):
        raise ValueError('Example too long or contains no assistant tokens')
    return {'input_ids': full, 'labels': [-100] * len(prefix) + full[len(prefix):]}


def collate_batch(rows: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, list[list[int]]]:
    """Right-pad with position-based label masking.

    Padding labels are always -100; genuine EOS tokens inside a response stay
    trainable even when PAD and EOS share one id, because masking is positional.
    """
    width = max(len(row['input_ids']) for row in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        padding = width - len(row['input_ids'])
        ids.append(row['input_ids'] + [pad_token_id] * padding)
        masks.append([1] * len(row['input_ids']) + [0] * padding)
        labels.append(row['labels'] + [-100] * padding)
    return {'input_ids': ids, 'attention_mask': masks, 'labels': labels}


def load_training_records(path: Path, require_review: bool) -> list[dict[str, Any]]:
    records = read_dataset(path)
    if require_review:
        unapproved = [record for record in records
                      if not (isinstance(record.get('review'), dict)
                              and record['review'].get('approved') is True)]
        if unapproved:
            raise ValueError(f'{len(unapproved)} of {len(records)} records lack recorded human '
                             'approval; review them or rerun without --require-review')
    if not records:
        raise ValueError('Dataset is empty')
    return records


def validate_reload(base_path: str, adapter_dir: Path, tokenizer: Any) -> dict[str, Any]:
    """Reload the saved adapter onto a fresh base copy and generate one token."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    started = time.perf_counter()
    base = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True,
                                                trust_remote_code=False, use_safetensors=True)
    reloaded = PeftModel.from_pretrained(base, str(adapter_dir))
    reloaded.eval()
    rendered = tokenizer.apply_chat_template(build_messages('Adapter reload validation.'),
                                            tokenize=False, add_generation_prompt=True)
    batch = tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
    with torch.inference_mode():
        reloaded.generate(**{key: value.to(reloaded.device) for key, value in batch.items()},
                          do_sample=False, max_new_tokens=1)
    return {'performed': True, 'seconds': round(time.perf_counter() - started, 3)}


def train(args: argparse.Namespace) -> dict[str, Any]:
    """Train adapters; base weights remain frozen by PEFT. No auto-promotion."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    require_int('epochs', args.epochs)
    require_int('batch-size', args.batch_size)
    require_int('max-length', args.max_length)
    require_int('rank', args.rank, low=1, high=1024)
    if args.train_content not in CONTENT_MODES:
        raise ValueError(f'--train-content must be one of {CONTENT_MODES}')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    validate_device(args.device)
    path = str(local_model_path(args.model))
    if args.output.exists():
        raise FileExistsError('Choose a new output directory; existing output is never overwritten')
    records = load_training_records(args.dataset, args.require_review)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer requires an existing PAD or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_record(tokenizer, record, args.max_length, args.train_content)
               for record in records]
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                                                 trust_remote_code=False, use_safetensors=True)
    context = getattr(model.config, 'max_position_embeddings', None)
    if isinstance(context, int) and max(len(row['input_ids']) for row in encoded) > context:
        raise ValueError('Dataset exceeds model context')
    targets: str | list[str] = args.target_modules
    if targets != 'all-linear':
        targets = [item.strip() for item in targets.split(',') if item.strip()]
        if not targets:
            raise ValueError('No LoRA target modules')
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.rank,
                           lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=targets,
                           bias='none')).to(args.device)
    model.config.use_cache = False
    model.train()
    pad_token_id = tokenizer.pad_token_id

    def collate(rows: list[dict[str, list[int]]]) -> dict[str, Any]:
        batch = collate_batch(rows, pad_token_id)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError('No trainable adapter parameters')
    trainable_count = sum(parameter.numel() for parameter in parameters)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=False)
    steps, losses = 0, []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{key: value.to(args.device) for key, value in batch.items()}).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError('Nonfinite loss; no adapter saved')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(loss.item())
            LOGGER.info('epoch=%d step=%d loss=%.6f', epoch + 1, steps, loss.item())
    duration = round(time.perf_counter() - started, 3)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    dtype = str(next(model.parameters()).dtype)
    del model, optimizer
    gc.collect()
    reload_result = validate_reload(path, args.output, tokenizer)
    manifest = build_manifest(args, records, encoded, steps, losses, duration,
                              trainable_count, path, tokenizer, reload_result, dtype)
    (args.output / 'training_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    # Written last; without this marker, output may be incomplete.
    (args.output / 'COMPLETE').write_text(
        'Training, serialization and adapter reload validation completed.\n', encoding='utf-8')
    return manifest


def build_manifest(args: argparse.Namespace, records: list[dict[str, Any]],
                   encoded: list[dict[str, list[int]]], steps: int, losses: list[float],
                   duration: float, trainable_count: int, path: str,
                   tokenizer: Any, reload_result: dict[str, Any], dtype: str) -> dict[str, Any]:
    """Record effective values and provenance; setting a seed does not guarantee
    bit-identical results across hardware or library versions."""
    return {
        'manifest_schema_version': MANIFEST_SCHEMA_VERSION,
        'prism_zero_version': library_version('prism-zero'),
        'git_commit': git_commit(),
        'python_version': platform.python_version(),
        'library_versions': {name: library_version(name) for name in
                             ('torch', 'transformers', 'peft', 'safetensors', 'tokenizers')},
        'dataset': {'name': args.dataset.name, 'record_count': len(records),
                    'fingerprint': dataset_fingerprint(args.dataset)},
        'base_checkpoint': {'name': Path(path).name, 'path': str(args.model),
                            'fingerprint': checkpoint_fingerprint(Path(path))},
        'chat_template_fingerprint': hashlib.sha256(
            (tokenizer.chat_template or '').encode('utf-8')).hexdigest(),
        'effective_arguments': {'epochs': args.epochs, 'batch_size': args.batch_size,
                                'max_length': args.max_length, 'learning_rate': args.learning_rate,
                                'seed': args.seed, 'device': args.device,
                                'train_content': args.train_content,
                                'require_review': args.require_review},
        'lora': {'r': args.rank, 'lora_alpha': 2 * args.rank, 'lora_dropout': 0.05,
                 'target_modules': args.target_modules, 'bias': 'none',
                 'trainable_parameters': trainable_count},
        'dtype': dtype,
        'steps': steps, 'epochs': args.epochs,
        'max_sequence_length': max(len(row['input_ids']) for row in encoded),
        'losses': {'final': losses[-1] if losses else None,
                   'mean': sum(losses) / len(losses) if losses else None,
                   'count': len(losses)},
        'training_duration_seconds': duration,
        'reload_validation': reload_result,
        'evaluation_performed': False,
        'artifact': 'LoRA adapter; the original base checkpoint is required to use it',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', type=Path, default=Path('prism_zero_dataset.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('prism-zero-adapter'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=2e-4)
    parser.add_argument('--target-modules', default='all-linear')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train-content', choices=CONTENT_MODES, default='trace-and-answer',
                        help='answer-only trains on the answer block alone; trace-and-answer '
                             '(default, previous behavior) also trains the explanation')
    parser.add_argument('--require-review', action='store_true',
                        help='refuse to train on records without recorded human approval')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        train(args)
        return 0
    except Exception:
        LOGGER.exception('Training failed; inspect incomplete output before reusing its path')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
