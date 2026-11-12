"""Optional ML integration suite: real tokenizer, tiny local checkpoint, LoRA.

These tests construct a tiny randomly-initialized GPT-2 checkpoint and a real
WordLevel tokenizer locally in a temporary directory. Nothing is downloaded;
a random model demonstrates plumbing only, never useful task performance.

Skipped automatically when torch, transformers, peft, or safetensors are absent.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pytest

pytest.importorskip('torch')
pytest.importorskip('transformers')
pytest.importorskip('peft')
pytest.importorskip('safetensors')

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast  # noqa: E402

from prism_zero.backends import LocalHFGenerator  # noqa: E402
from prism_zero.candidates import Candidate  # noqa: E402
from prism_zero.dataset import record_id  # noqa: E402
from prism_zero.trainer import collate_batch, encode_record, train  # noqa: E402

pytestmark = pytest.mark.ml

T_OPEN = '<' + 'think>'
T_CLOSE = '</' + 'think>'
A_OPEN = '<' + 'answer>'
A_CLOSE = '</' + 'answer>'
CHAT_TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
                 "{{ message['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
WORDS = ['system', 'user', 'assistant', 'Solve', 'the', 'task', 'Return', 'a', 'concise',
         'explanation', 'in', 'tags', 'then', 'final', 'answer', 'no', 'other', 'text',
         'What', 'is', 'six', 'times', 'seven', 'integer', 'Six', 'groups', 'of', 'total',
         'forty-two', '42', '41', 'Compute', 'product']


def build_checkpoint(directory: Path) -> Path:
    """Create a tiny random-init causal LM checkpoint with a chat template."""
    specials = ['<unk>', '<pad>', '<|im_start|>', '<|im_end|>']
    vocab = {token: index for index, token in enumerate(specials + sorted(set(WORDS)))}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>',
                                   eos_token='<|im_end|>')
    fast.chat_template = CHAT_TEMPLATE
    config = GPT2Config(vocab_size=len(vocab), n_positions=128, n_embd=32, n_layer=2,
                        n_head=4, bos_token_id=0, eos_token_id=fast.eos_token_id,
                        pad_token_id=fast.eos_token_id)
    GPT2LMHeadModel(config).save_pretrained(directory, safe_serialization=True)
    fast.save_pretrained(directory)
    return directory


def make_record(task_id: str, prompt: str, answer: str) -> dict:
    trace = 'Six groups of seven total forty-two.'
    return {'schema_version': 2, 'id': record_id(task_id, '1', prompt, trace, answer),
            'task_id': task_id, 'verifier_version': '1', 'prompt': prompt, 'trace': trace,
            'answer': answer, 'attempts': 1,
            'checks': [{'verifier': 'math_equality', 'passed': True, 'message': 'ok',
                        'target': 'answer'}],
            'created_at': '2026-01-01T00:00:00+00:00', 'generator_id': None,
            'run_id': None, 'review': None}


def write_dataset(path: Path, records: list[dict]) -> Path:
    path.write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')
    return path


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    return build_checkpoint(tmp_path / 'checkpoint')


@pytest.fixture
def training_args(tmp_path: Path, checkpoint: Path):
    import argparse
    dataset = write_dataset(tmp_path / 'data.jsonl', [
        make_record('multiply-1', 'What is six times seven?', '42'),
        make_record('multiply-2', 'Compute the product of six and seven.', '42'),
        make_record('multiply-3', 'What is seven times six?', '42')])
    return argparse.Namespace(
        model=str(checkpoint), dataset=dataset, output=tmp_path / 'adapter',
        device='cpu', epochs=1, batch_size=2, max_length=96, rank=4, learning_rate=1e-3,
        target_modules='all-linear', seed=17, train_content='trace-and-answer',
        require_review=False)


def test_full_pipeline(tmp_path: Path, training_args) -> None:
    """Load, generate, train, save, reload, generate again — all locally."""
    generator = LocalHFGenerator(str(training_args.model), max_new_tokens=8,
                                 max_input_tokens=64, device='cpu')
    assert isinstance(generator.generate('What is six times seven?', None), str)
    original = training_args.output
    train(training_args)
    expected = {'adapter_config.json', 'adapter_model.safetensors', 'training_manifest.json',
                'COMPLETE', 'tokenizer.json', 'tokenizer_config.json'}
    assert expected <= {p.name for p in original.iterdir()}
    manifest = json.loads((original / 'training_manifest.json').read_text())
    assert manifest['reload_validation']['performed'] is True
    assert manifest['evaluation_performed'] is False
    assert manifest['library_versions']['torch']
    assert manifest['steps'] >= 1 and manifest['losses']['final'] is not None
    assert manifest['base_checkpoint']['fingerprint']
    assert (original / 'COMPLETE').read_text().startswith('Training, serialization')
    reloaded = LocalHFGenerator(str(training_args.model), max_new_tokens=8,
                                max_input_tokens=64, device='cpu', adapter_path=str(original))
    assert isinstance(reloaded.generate('What is six times seven?', None), str)


def test_existing_output_refused(training_args) -> None:
    training_args.output.mkdir()
    with pytest.raises(FileExistsError):
        train(training_args)


def test_nonfinite_learning_rate_rejected(training_args) -> None:
    training_args.learning_rate = float('nan')
    with pytest.raises(ValueError, match='finite'):
        train(training_args)


def test_invalid_counts_rejected(training_args) -> None:
    import argparse
    for key in ('epochs', 'batch_size', 'max_length', 'rank'):
        for bad in (0, -1, 1.5, True):
            args = argparse.Namespace(**{**vars(training_args), key: bad})
            with pytest.raises(ValueError, match=key.replace('_', '-')):
                train(args)


def test_no_trainable_target_modules(training_args) -> None:
    training_args.target_modules = 'does-not-exist'
    with pytest.raises(Exception, match='[Tt]arget [Mm]odules'):
        train(training_args)


def test_require_review_policy(tmp_path: Path, training_args) -> None:
    records = [make_record('t1', 'prompt', '42')]
    training_args.dataset = write_dataset(tmp_path / 'reviewed.jsonl', records)
    training_args.require_review = True
    with pytest.raises(ValueError, match='approval'):
        train(training_args)
    records[0]['review'] = {'approved': True, 'source': 'human reviewer'}
    training_args.dataset = write_dataset(tmp_path / 'approved.jsonl', records)
    training_args.require_review = True
    train(training_args)


def test_labels_prompt_masked_assistant_trainable(checkpoint: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    record = make_record('t', 'What is six times seven?', '42')
    from prism_zero.backends import build_messages
    row = encode_record(tokenizer, record, 96)
    full = tokenizer.apply_chat_template(
        build_messages(record['prompt'])
        + [{'role': 'assistant', 'content': Candidate(record['trace'], record['answer']).render()}],
        tokenize=True, add_generation_prompt=False)
    assert row['input_ids'] == full
    prompt_length = next(i for i, value in enumerate(row['labels']) if value != -100)
    assert row['labels'][:prompt_length] == [-100] * prompt_length
    assert row['labels'][prompt_length:] == full[prompt_length:]
    assert tokenizer.eos_token_id in row['labels']  # genuine EOS stays trainable


def test_collate_masks_padding_but_not_eos(checkpoint: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token  # PAD and EOS share one id
    record = make_record('t', 'What is six times seven?', '42')
    long = encode_record(tokenizer, record, 96)
    short = encode_record(tokenizer, make_record('s', 'Compute the product.', '42'), 96)
    assert len(short['input_ids']) < len(long['input_ids'])
    batch = collate_batch([long, short], tokenizer.pad_token_id)
    width = len(batch['input_ids'][0])
    for position in range(len(short['input_ids']), width):
        assert batch['labels'][1][position] == -100  # padding masked
        assert batch['attention_mask'][1][position] == 0
    assert batch['labels'][1][len(short['input_ids']) - 1] != -100  # genuine EOS target kept


def test_oversized_example_refuses_truncation(checkpoint: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    record = make_record('t', 'What is six times seven?', '42')
    row = encode_record(tokenizer, record, 96)
    with pytest.raises(ValueError, match='too long'):
        encode_record(tokenizer, record, len(row['input_ids']) - 1)


def test_unstable_assistant_boundary(checkpoint: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer.chat_template = ("{% if add_generation_prompt %}<generation-prefix>{% endif %}"
                               "{% for message in messages %}{{ message['role'] }}: "
                               "{{ message['content'] }}\n{% endfor %}")
    record = make_record('t', 'What is six times seven?', '42')
    with pytest.raises(ValueError, match='stable assistant boundary'):
        encode_record(tokenizer, record, 96)


def test_answer_only_labels(checkpoint: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    record = make_record('t', 'What is six times seven?', '42')
    row = encode_record(tokenizer, record, 96, content_mode='answer-only')
    assert any(value != -100 for value in row['labels'])
    prefix = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': record['prompt']}],
        tokenize=True, add_generation_prompt=True)
    assert row['labels'][:len(prefix)] == [-100] * len(prefix)


def test_missing_checkpoint_fails_clearly() -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        LocalHFGenerator(str(Path(tempfile.gettempdir()) / 'prism-zero-missing-xyz'))


def test_file_where_directory_required(tmp_path: Path) -> None:
    file_path = tmp_path / 'weights.safetensors'
    file_path.write_bytes(b'not a directory')
    with pytest.raises(ValueError, match='checkpoint directory'):
        LocalHFGenerator(str(file_path))


def test_missing_tokenizer_assets(tmp_path: Path, checkpoint: Path) -> None:
    stripped = tmp_path / 'no-tokenizer'
    shutil.copytree(checkpoint, stripped)
    for name in ('tokenizer.json', 'tokenizer_config.json'):
        (stripped / name).unlink()
    with pytest.raises(Exception):
        LocalHFGenerator(str(stripped))


def test_missing_chat_template(tmp_path: Path, checkpoint: Path) -> None:
    stripped = tmp_path / 'no-template'
    shutil.copytree(checkpoint, stripped)
    (stripped / 'chat_template.jinja').unlink()
    config = json.loads((stripped / 'tokenizer_config.json').read_text())
    config.pop('chat_template', None)
    (stripped / 'tokenizer_config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='chat template'):
        LocalHFGenerator(str(stripped))


def test_token_budget_rejects_excess_input(checkpoint: Path) -> None:
    generator = LocalHFGenerator(str(checkpoint), max_new_tokens=8, max_input_tokens=32)
    long_prompt = 'What is six times seven? ' * 20
    with pytest.raises(ValueError, match='Token budget'):
        generator.generate(long_prompt, None)


def test_context_window_accounts_for_new_tokens(checkpoint: Path) -> None:
    generator = LocalHFGenerator(str(checkpoint), max_new_tokens=8192, max_input_tokens=2048)
    with pytest.raises(ValueError, match='Token budget'):
        generator.generate('What is six times seven?', None)


def test_unsupported_device_rejected(checkpoint: Path) -> None:
    for device in ('tpu', 'cuda:99', '', 'cpu:x'):
        with pytest.raises(ValueError, match='[Dd]evice|CUDA'):
            LocalHFGenerator(str(checkpoint), device=device)


def test_cuda_optional(checkpoint: Path) -> None:
    import torch
    if torch.cuda.is_available():
        LocalHFGenerator(str(checkpoint), device='cuda')
    else:
        with pytest.raises(ValueError, match='CUDA'):
            LocalHFGenerator(str(checkpoint), device='cuda')


def test_incomplete_adapter_directory_rejected(tmp_path: Path, checkpoint: Path,
                                               training_args) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    train(training_args)
    incomplete = tmp_path / 'incomplete'
    incomplete.mkdir()
    shutil.copy(training_args.output / 'adapter_config.json', incomplete)
    base = AutoModelForCausalLM.from_pretrained(str(checkpoint), local_files_only=True,
                                                trust_remote_code=False, use_safetensors=True)
    with pytest.raises(Exception):
        PeftModel.from_pretrained(base, str(incomplete), local_files_only=True)


def test_base_weights_frozen_and_adapter_updated(checkpoint: Path) -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    torch.manual_seed(0)
    model = GPT2LMHeadModel.from_pretrained(str(checkpoint), local_files_only=True,
                                            trust_remote_code=False, use_safetensors=True)
    base_before = model.transformer.wte.weight.clone()
    peft_model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=4,
                                lora_alpha=8, lora_dropout=0.05, target_modules='all-linear',
                                bias='none'))
    trainable = [parameter for parameter in peft_model.parameters() if parameter.requires_grad]
    assert trainable, 'LoRA should expose trainable parameters'
    frozen = [parameter for parameter in peft_model.parameters() if not parameter.requires_grad]
    assert frozen, 'base parameters should be frozen'
    lora_before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    ids = torch.randint(0, model.config.vocab_size, (1, 16))
    loss = peft_model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(loss).item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    assert torch.equal(model.transformer.wte.weight, base_before), 'base weights changed'
    assert any(not torch.equal(before, parameter.detach())
               for before, parameter in zip(lora_before, trainable)), 'adapter never updated'
    assert optimizer.param_groups[0]['params'] == trainable


def test_saved_adapter_weights_nonzero(training_args) -> None:
    from safetensors import safe_open
    train(training_args)
    with safe_open(training_args.output / 'adapter_model.safetensors', framework='pt') as f:
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    assert any('lora_A' in key for key in tensors) and any('lora_B' in key for key in tensors)
    assert any(tensor.abs().sum().item() > 0 for tensor in tensors.values())


def test_offline_flags_do_not_break_loading(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.setenv('HF_HUB_DISABLE_TELEMETRY', '1')
    generator = LocalHFGenerator(str(checkpoint), max_new_tokens=4, max_input_tokens=64)
    assert isinstance(generator.generate('What is six times seven?', None), str)

def test_cli_evaluate_end_to_end(tmp_path: Path, training_args, capsys) -> None:
    """CLI evaluate with a real tiny model: metrics, JSON, and adapter conditions."""
    from prism_zero.cli import main as cli_main
    train(training_args)
    tasks = tmp_path / 'tasks.json'
    tasks.write_text(json.dumps({
        'task_set_version': 'tiny-v1',
        'tasks': [
            {'task_id': 'arith-1', 'category': 'arithmetic', 'prompt': 'What is six times seven?',
             'rules': [{'verifier': 'math_equality', 'expected': '42'}]},
            {'task_id': 'json-1', 'category': 'json', 'prompt': 'Compute the product.',
             'rules': [{'verifier': 'json_schema',
                        'schema': {'type': 'object', 'required': ['ok']}}]},
        ]}), encoding='utf-8')
    output = tmp_path / 'results.json'
    assert cli_main(['evaluate', '--tasks', str(tasks), '--model', str(training_args.model),
                     '--adapter', str(training_args.output), '--dataset', str(training_args.dataset),
                     '--attempts', '2', '--max-new-tokens', '4', '--device', 'cpu',
                     '--output', str(output), '--json']) == 0
    captured = capsys.readouterr()
    assert captured.out.strip()
    evaluation = json.loads(captured.out)
    assert set(evaluation['conditions']) == {'base_1_attempt', 'base_budget',
                                            'adapter_1_attempt', 'adapter_budget'}
    for name in ('base_budget', 'adapter_budget'):
        assert evaluation['conditions'][name]['status'] == 'RUN'
        assert evaluation['conditions'][name]['task_count'] == 2
    assert evaluation['overlap']['checked'] is True
    assert output.exists()
    assert json.loads(output.read_text())['evaluation_schema_version'] == 1
