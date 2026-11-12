"""CLI regressions: commands run from outside the repository checkout.

These tests call the argument parser directly so they exercise the same
handlers without depending on a console-script installation.
"""
import json
from pathlib import Path
import pytest
from prism_zero.cli import (EXIT_FAILURE, EXIT_OK, EXIT_USAGE, build_parser, load_config,
                            command_doctor, command_dataset)
from prism_zero.scripted import ScriptedModel


def run_cli(argv: list[str]) -> int:
    from prism_zero.cli import main
    return main(argv)


def write_config(path: Path, **overrides: object) -> Path:
    config = {'mode': 'demo', 'prompt': 'What is six times seven? Return an integer.',
              'task_id': 'multiply-6-by-7', 'max_attempts': 3,
              'rules': [{'verifier': 'math_equality', 'expected': '42'}]}
    config.update(overrides)
    path.write_text(json.dumps(config), encoding='utf-8')
    return path


def test_run_demo_curates_and_deduplicates(tmp_path: Path) -> None:
    config = write_config(tmp_path / 'config.json',
                          dataset='out.jsonl')
    dataset = tmp_path / 'out.jsonl'
    assert run_cli(['run', '--config', str(config)]) == EXIT_OK
    assert run_cli(['run', '--config', str(config)]) == EXIT_OK
    assert dataset.exists()
    lines = dataset.read_text().strip().splitlines()
    assert len(lines) == 1


def test_run_rejection_exit_code(tmp_path: Path) -> None:
    config = write_config(tmp_path / 'config.json', max_attempts=1,
                          rules=[{'verifier': 'math_equality', 'expected': '99'}])
    assert run_cli(['run', '--config', str(config)]) == EXIT_FAILURE


def test_run_local_mode_requires_model(tmp_path: Path) -> None:
    config = write_config(tmp_path / 'config.json', mode='local')
    assert run_cli(['run', '--config', str(config)]) == EXIT_USAGE


def test_run_scripted_demo_writes_configured_dataset(tmp_path: Path) -> None:
    config = write_config(tmp_path / 'config.json', dataset='out.jsonl')
    assert run_cli(['run', '--config', str(config)]) == EXIT_OK
    record = json.loads((tmp_path / 'out.jsonl').read_text().splitlines()[0])
    assert record['answer'] == '42'


def test_run_unknown_keys_rejected(tmp_path: Path) -> None:
    config = write_config(tmp_path / 'config.json')
    parsed = json.loads(config.read_text())
    parsed['unexpected'] = True
    config.write_text(json.dumps(parsed))
    with pytest.raises(Exception, match='Unknown configuration keys'):
        load_config(config)


def test_run_rejects_float_and_bool_attempts(tmp_path: Path) -> None:
    for bad in (3.5, True, 0, 21):
        config = write_config(tmp_path / 'config.json', max_attempts=bad)
        with pytest.raises(Exception, match='max_attempts'):
            load_config(config)


def test_run_trace_rule_via_config(tmp_path: Path) -> None:
    config = write_config(tmp_path / 'config.json',
                          rules=[{'verifier': 'math_equality', 'expected': '42'},
                                 {'verifier': 'regex', 'pattern': 'required', 'target': 'trace'}])
    assert run_cli(['run', '--config', str(config)]) == EXIT_FAILURE  # demo trace never matches


def test_doctor_passes_core_environment(capsys: pytest.CaptureFixture) -> None:
    parser = build_parser()
    args = parser.parse_args(['doctor'])
    assert command_doctor(args) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.out == ''  # human-readable output goes to stderr
    assert 'dependency torch' in captured.err


def test_doctor_json_on_stdout(capsys: pytest.CaptureFixture) -> None:
    parser = build_parser()
    args = parser.parse_args(['doctor', '--json'])
    assert command_doctor(args) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert {'checks', 'summary'} <= set(payload)
    assert payload['summary']['pass'] >= 2


def test_doctor_missing_model_fails_clearly(capsys: pytest.CaptureFixture) -> None:
    parser = build_parser()
    args = parser.parse_args(['doctor', '--model', str(Path('/nonexistent-checkpoint'))])
    assert command_doctor(args) == EXIT_FAILURE
    assert 'model directory' in capsys.readouterr().err


def test_doctor_existing_output_directory_fails(capsys: pytest.CaptureFixture,
                                                tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(['doctor', '--output', str(tmp_path)])
    assert command_doctor(args) == EXIT_FAILURE  # training refuses to overwrite
    args = parser.parse_args(['doctor', '--output', str(tmp_path / 'new')])
    assert command_doctor(args) == EXIT_OK


def test_dataset_validate_and_inspect(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from tests.test_dataset import make_record  # reuse the shared contract fixture
    dataset = tmp_path / 'data.jsonl'
    dataset.write_text(json.dumps(make_record()) + '\n', encoding='utf-8')
    parser = build_parser()
    args = parser.parse_args(['dataset', 'validate', str(dataset)])
    assert command_dataset(args) == EXIT_OK
    args = parser.parse_args(['dataset', 'inspect', str(dataset), '--json'])
    assert command_dataset(args) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload['record_count'] == 1 and payload['reviewed_count'] == 0


def test_evaluate_requires_model(capsys: pytest.CaptureFixture) -> None:
    assert run_cli(['evaluate', '--tasks', '/tmp/none.json']) == EXIT_USAGE


def test_evaluate_missing_task_set(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert run_cli(['evaluate', '--tasks', str(tmp_path / 'missing.json')]) == EXIT_USAGE
    assert 'not found' in capsys.readouterr().err
