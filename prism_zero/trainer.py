"""Explicit, single-device local LoRA SFT with assistant-only labels."""
from __future__ import annotations
import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any
from .backends import build_messages, local_model_path
from .engine import Candidate
from .verifiers import strict_json

LOGGER = logging.getLogger(__name__)

def read_records(path: Path) -> list[dict[str, Any]]:
    """Check record structure; stored verification claims are not attestations."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                record = strict_json(line)
                if not isinstance(record, dict) or record.get('schema_version') != 1:
                    raise ValueError('Unsupported dataset schema')
                for key in ('id', 'prompt', 'trace', 'answer', 'task_id', 'verifier_version'):
                    if not isinstance(record.get(key), str) or not record[key].strip():
                        raise ValueError(f'Missing nonempty string: {key}')
                checks = record.get('checks')
                if not isinstance(checks, list) or not checks or not all(
                    isinstance(check, dict) and check.get('passed') is True for check in checks):
                    raise ValueError('All recorded checks must pass')
                Candidate.parse(Candidate(record['trace'], record['answer']).render())
                if record['id'] not in seen:
                    records.append(record)
                    seen.add(record['id'])
            except (ValueError, RecursionError) as error:
                raise ValueError(f'Invalid dataset line {number}: {error}') from error
    if not records:
        raise ValueError('Dataset is empty')
    return records

def encode_record(tokenizer: Any, record: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    """Use the inference template, mask prompts and refuse truncation."""
    messages = build_messages(record['prompt'])
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    response = Candidate(record['trace'], record['answer']).render()
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': response}],
                                        tokenize=True, add_generation_prompt=False)
    if not prefix or full[:len(prefix)] != prefix:
        raise ValueError('Chat template has no stable assistant boundary')
    if len(full) > max_length or len(full) <= len(prefix):
        raise ValueError('Example too long or contains no assistant tokens')
    return {'input_ids': full, 'labels': [-100] * len(prefix) + full[len(prefix):]}

def train(args: argparse.Namespace) -> None:
    """Train adapters; base weights remain frozen by PEFT. No auto-promotion."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if min(args.epochs, args.batch_size, args.max_length, args.rank) < 1:
        raise ValueError('Training counts and lengths must be positive')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be finite and positive')
    path = str(local_model_path(args.model))
    if args.output.exists():
        raise FileExistsError('Choose a new output directory')
    records = read_records(args.dataset)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer requires an existing PAD or EOS token')
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_record(tokenizer, record, args.max_length) for record in records]
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
                lora_alpha=2 * args.rank, lora_dropout=0.05, target_modules=targets, bias='none')).to(args.device)
    model.config.use_cache = False
    model.train()
    def collate(rows: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(row['input_ids']) for row in rows)
        ids, masks, labels = [], [], []
        for row in rows:
            padding = width - len(row['input_ids'])
            ids.append(row['input_ids'] + [tokenizer.pad_token_id] * padding)
            masks.append([1] * len(row['input_ids']) + [0] * padding)
            # Position-based masking preserves EOS targets when PAD equals EOS.
            labels.append(row['labels'] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in
                [('input_ids', ids), ('attention_mask', masks), ('labels', labels)]}
    loader = DataLoader(encoded, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError('No trainable adapter parameters')
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    args.output.mkdir(parents=True, exist_ok=False)
    steps = 0
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
            LOGGER.info('epoch=%d step=%d loss=%.6f', epoch + 1, steps, loss.item())
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    manifest = {'base_model': path, 'records': len(encoded), 'steps': steps, 'seed': args.seed,
                'evaluation_performed': False, 'artifact': 'LoRA adapter; base checkpoint required'}
    (args.output / 'training_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    # Written last; without this marker, output may be incomplete.
    (args.output / 'COMPLETE').write_text('Training and serialization completed.\n', encoding='utf-8')

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
