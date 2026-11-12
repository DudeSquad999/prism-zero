"""Prism Zero command-line interface.

Machine-readable JSON goes to stdout; logs and diagnostics go to stderr.
Exit codes: 0 success, 1 operational failure, 2 invalid configuration.
"""
from __future__ import annotations
import argparse
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any
from . import __version__ as PACKAGE_VERSION
from .dataset import DatasetError, dataset_fingerprint, read_dataset
from .engine import PrismZeroEngine, Rule
from .evaluation import (CONDITIONS, TaskSetError, load_task_set, run_evaluation,
                         format_summary)
from .scripted import ScriptedModel
from .verifiers import MathEqualityVerifier

LOGGER = logging.getLogger(__name__)
EXIT_OK, EXIT_FAILURE, EXIT_USAGE = 0, 1, 2
CONFIG_KEYS = {'mode', 'dataset', 'task_id', 'verifier_version', 'max_attempts', 'prompt',
               'rules', 'model'}
MODEL_KEYS = {'path', 'device', 'max_new_tokens', 'max_input_tokens', 'adapter_path'}


def emit_json(payload: dict[str, Any]) -> None:
    """Machine-readable output to stdout, separated from stderr logging."""
    json.dump(payload, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write('\n')


class ConfigError(ValueError):
    """A configuration file violated the documented schema."""


def load_config(path: Path) -> dict[str, Any]:
    """Validate a JSON run configuration; relative paths resolve against the
    configuration file's directory."""
    try:
        parsed = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f'Cannot read configuration: {error}') from error
    if not isinstance(parsed, dict):
        raise ConfigError('Configuration must be a JSON object')
    unknown = set(parsed) - CONFIG_KEYS
    if unknown:
        raise ConfigError(f'Unknown configuration keys: {sorted(unknown)}')
    mode = parsed.get('mode', 'demo')
    if mode not in ('demo', 'local'):
        raise ConfigError("mode must be 'demo' or 'local'")
    if mode == 'local' and not isinstance(parsed.get('model'), dict):
        raise ConfigError('local mode requires a model object')
    if mode == 'demo' and 'model' in parsed:
        raise ConfigError('demo mode must not configure a model')
    if 'model' in parsed:
        unknown = set(parsed['model']) - MODEL_KEYS
        if unknown:
            raise ConfigError(f'Unknown model keys: {sorted(unknown)}')
        if not str(parsed['model'].get('path', '')).strip():
            raise ConfigError('model.path is required')
    max_attempts = parsed.get('max_attempts', 3)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) \
            or not 1 <= max_attempts <= 20:
        raise ConfigError('max_attempts must be an integer between 1 and 20')
    if not isinstance(parsed.get('prompt'), str) or not parsed['prompt'].strip():
        raise ConfigError('prompt is required and must be nonempty')
    for key in ('dataset', 'task_id', 'verifier_version'):
        value = parsed.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ConfigError(f'{key} must be a nonempty string when present')
    rules = parsed.get('rules')
    if not isinstance(rules, list) or not rules:
        raise ConfigError('rules must be a nonempty list')
    from .evaluation import build_rule
    built: list[Rule] = []
    for index, spec in enumerate(rules, 1):
        try:
            verifier, target = build_rule(spec, f'rule {index}')
        except TaskSetError as error:
            raise ConfigError(f'Invalid rule {index}: {error}') from error
        built.append(Rule(verifier, target))
    if not any(rule.target == 'answer' for rule in built):
        raise ConfigError('At least one rule must target the answer')
    return parsed


def build_generator(config: dict[str, Any], base: Path) -> Any:
    if config.get('mode', 'demo') == 'demo':
        return ScriptedModel()
    from .backends import LocalHFGenerator
    model = config['model']
    return LocalHFGenerator(
        model['path'], device=model.get('device', 'cpu'),
        max_new_tokens=model.get('max_new_tokens', 256),
        max_input_tokens=model.get('max_input_tokens', 2048),
        adapter_path=model.get('adapter_path'))


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = args.config.parent
    dataset = base / config.get('dataset', 'prism_zero_dataset.jsonl') \
        if config.get('dataset') else Path('prism_zero_dataset.jsonl')
    generator = build_generator(config, base)
    rules = []
    from .evaluation import build_rule
    for spec in config.get('rules', []):
        verifier, target = build_rule(spec, 'rule')
        rules.append(Rule(verifier, target))
    engine = PrismZeroEngine(generator, rules, dataset,
                             max_attempts=config.get('max_attempts', 3),
                             task_id=config.get('task_id', 'default'),
                             verifier_version=config.get('verifier_version', '1'))
    result = engine.run(config['prompt'])
    emit_json({'accepted': result.accepted, 'attempts': result.attempts,
               'record_id': result.record_id})
    return EXIT_OK if result.accepted else EXIT_FAILURE


def command_doctor(args: argparse.Namespace) -> int:
    checks: list[dict[str, str]] = []

    def record(name: str, status: str, detail: str, fix: str = '') -> None:
        checks.append({'name': name, 'status': status, 'detail': detail, 'fix': fix})

    version = sys.version_info
    if version >= (3, 11):
        record('python', 'PASS', f'{platform.python_version()}')
    else:
        record('python', 'FAIL', f'{platform.python_version()}', 'Use Python 3.11 or newer')

    def optional_module(name: str) -> bool:
        try:
            __import__(name)
        except ImportError:
            return False
        return True

    for module, purpose in (('jsonschema', 'required verifier dependency'),
                           ('regex', 'required verifier dependency'),
                           ('torch', 'optional local-model backend'),
                           ('transformers', 'optional local-model backend'),
                           ('peft', 'optional adapter training and loading'),
                           ('safetensors', 'optional model serialization')):
        if optional_module(module):
            record(f'dependency {module}', 'PASS', purpose)
        else:
            record(f'dependency {module}', 'WARN', 'not installed',
                   'Install the ml optional dependencies for local-model features')

    if args.model:
        from .backends import local_model_path
        try:
            directory = local_model_path(args.model)
            record('model directory', 'PASS', str(args.model))
            required = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
            missing = [name for name in required if not (directory / name).is_file()]
            weights = any(directory.glob('*.safetensors'))
            if missing:
                record('model assets', 'FAIL', f'missing {missing}',
                       'Provide a complete local checkpoint with safetensors weights')
            elif not weights:
                record('model assets', 'FAIL', 'no safetensors weights found')
            else:
                record('model assets', 'PASS', 'config, tokenizer, safetensors present')
            if optional_module('transformers'):
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                                           trust_remote_code=False)
                try:
                    tokenizer.apply_chat_template([{'role': 'user', 'content': 'check'}],
                                                  tokenize=False, add_generation_prompt=True)
                    record('chat template', 'PASS', 'usable')
                except Exception as error:
                    record('chat template', 'FAIL', type(error).__name__,
                           'Set a chat template in the checkpoint tokenizer configuration')
        except (FileNotFoundError, ValueError) as error:
            record('model directory', 'FAIL', str(error),
                   'Provide an existing local checkpoint directory')
    else:
        record('model directory', 'WARN', 'not configured', 'Pass --model to check a checkpoint')

    if args.output and args.output.exists():
        record('output directory', 'FAIL', 'already exists',
               'Training refuses to overwrite; choose a new output directory')
    elif args.output:
        record('output directory', 'PASS', 'does not exist yet')

    if args.device:
        try:
            if args.device == 'cuda' and not optional_module('torch'):
                record('device', 'FAIL', 'cuda requested without torch installed')
            elif args.device == 'cuda':
                import torch
                if torch.cuda.is_available():
                    record('device', 'PASS', 'cuda available')
                else:
                    record('device', 'FAIL', 'torch.cuda.is_available() is false',
                           'Install a CUDA-enabled PyTorch build or use cpu')
            else:
                record('device', 'PASS', args.device)
        except Exception as error:
            record('device', 'FAIL', str(error))

    statuses = [check['status'] for check in checks]
    summary = {'pass': statuses.count('PASS'), 'warn': statuses.count('WARN'),
               'fail': statuses.count('FAIL')}
    if args.json:
        emit_json({'checks': checks, 'summary': summary,
                   'prism_zero_version': PACKAGE_VERSION})
    else:
        for check in checks:
            print(f'{check["status"]:4} {check["name"]}: {check["detail"]}', file=sys.stderr)
            if check['fix'] and check['status'] != 'PASS':
                print(f'     fix: {check["fix"]}', file=sys.stderr)
        print(f'{summary["pass"]} pass, {summary["warn"]} warn, {summary["fail"]} fail',
              file=sys.stderr)
    return EXIT_FAILURE if summary['fail'] else EXIT_OK


def command_dataset(args: argparse.Namespace) -> int:
    if args.action == 'validate':
        records = read_dataset(args.path)
        if args.json:
            emit_json({'valid': True, 'record_count': len(records),
                       'fingerprint': dataset_fingerprint(args.path)})
        else:
            print(f'valid: {len(records)} records, fingerprint '
                  f'{dataset_fingerprint(args.path)[:16]}', file=sys.stderr)
        return EXIT_OK
    records = read_dataset(args.path)
    tasks: dict[str, int] = {}
    reviews = 0
    for record in records:
        tasks[record['task_id']] = tasks.get(record['task_id'], 0) + 1
        if isinstance(record.get('review'), dict) and record['review'].get('approved'):
            reviews += 1
    payload = {'record_count': len(records), 'fingerprint': dataset_fingerprint(args.path),
               'task_count': len(tasks), 'reviewed_count': reviews,
               'schema_versions': sorted({r['schema_version'] for r in records})}
    if args.json:
        emit_json(payload)
    else:
        print(f'{len(records)} records, {len(tasks)} tasks, {reviews} reviewed',
              file=sys.stderr)
        print(f'fingerprint {payload["fingerprint"][:16]}', file=sys.stderr)
    return EXIT_OK


def command_evaluate(args: argparse.Namespace) -> int:
    try:
        task_set = load_task_set(args.tasks)
    except FileNotFoundError as error:
        print(f'Configuration error: task set not found: {error}', file=sys.stderr)
        return EXIT_USAGE
    if args.model is None:
        print('Error: --model is required for model evaluation; a scripted demo '
              'evaluation is available through the library API and test fixtures',
              file=sys.stderr)
        return EXIT_USAGE
    from .backends import LocalHFGenerator
    base = LocalHFGenerator(args.model, device=args.device,
                            max_new_tokens=args.max_new_tokens,
                            max_input_tokens=args.max_input_tokens)
    adapter = None
    if args.adapter:
        adapter = LocalHFGenerator(args.model, device=args.device,
                                   max_new_tokens=args.max_new_tokens,
                                   max_input_tokens=args.max_input_tokens,
                                   adapter_path=args.adapter)
    evaluation = run_evaluation(base, adapter, task_set, max_attempts=args.attempts,
                               dataset_path=args.dataset)
    evaluation['model'] = args.model
    evaluation['adapter'] = args.adapter
    evaluation['software'] = {'python': platform.python_version(),
                              'prism_zero': PACKAGE_VERSION}
    print(format_summary(evaluation), file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evaluation, indent=2, allow_nan=False) + '\n',
                               encoding='utf-8')
        print(f'results written to {args.output}', file=sys.stderr)
    if args.json:
        emit_json(evaluation)
    return EXIT_OK


def command_train(argv: list[str]) -> int:
    """Delegate to the training entry point, preserving prism-zero-train compatibility."""
    from . import trainer
    parser = argparse.ArgumentParser(add_help=False)
    known, remaining = parser.parse_known_args(argv)
    return trainer.main(remaining)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='prism-zero', description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    doctor = subparsers.add_parser('doctor', help='Check environment and optional dependencies')
    doctor.add_argument('--model', help='Local checkpoint directory to check')
    doctor.add_argument('--output', type=Path, help='Planned training output directory')
    doctor.add_argument('--device', help='Requested device such as cpu or cuda')
    doctor.add_argument('--json', action='store_true', help='JSON output on stdout')
    doctor.set_defaults(handler=command_doctor)

    run = subparsers.add_parser('run', help='Generate, verify, retry, and curate')
    run.add_argument('--config', type=Path, required=True, help='JSON configuration file')
    run.set_defaults(handler=command_run)

    dataset = subparsers.add_parser('dataset', help='Validate or inspect a JSONL dataset')
    dataset.add_argument('action', choices=['validate', 'inspect'])
    dataset.add_argument('path', type=Path)
    dataset.add_argument('--json', action='store_true')
    dataset.set_defaults(handler=command_dataset)

    evaluate = subparsers.add_parser('evaluate', help='Evaluate base and adapter conditions')
    evaluate.add_argument('--tasks', type=Path, required=True, help='Task set JSON file')
    evaluate.add_argument('--model', help='Local base checkpoint directory')
    evaluate.add_argument('--adapter', help='Local adapter directory to compare')
    evaluate.add_argument('--dataset', type=Path, help='Training dataset for overlap checks')
    evaluate.add_argument('--attempts', type=int, default=3,
                          help='Attempt budget for bounded conditions')
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--max-new-tokens', type=int, default=64)
    evaluate.add_argument('--max-input-tokens', type=int, default=2048)
    evaluate.add_argument('--output', type=Path, help='Write a versioned JSON result file')
    evaluate.add_argument('--json', action='store_true')
    evaluate.set_defaults(handler=command_evaluate)

    train = subparsers.add_parser('train', help='Train a local LoRA adapter',
                                  add_help=False)
    train.set_defaults(handler='__train__')
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if argv and argv[0] == 'train':
        return command_train(argv)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s',
                        stream=sys.stderr)
    try:
        return args.handler(args)
    except (ConfigError, DatasetError, TaskSetError) as error:
        print(f'Configuration error: {error}', file=sys.stderr)
        return EXIT_USAGE
    except Exception:
        LOGGER.exception('Command failed')
        return EXIT_FAILURE


if __name__ == '__main__':
    raise SystemExit(main())
