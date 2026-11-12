"""Shared dataset contract, storage failure, and compatibility regressions."""
import json
import os
import threading
from pathlib import Path
import pytest
from prism_zero.candidates import Candidate
from prism_zero.dataset import (DatasetError, DatasetStore, content_view,
                                dataset_fingerprint, read_dataset, record_id, validate_record)

T_OPEN = '<' + 'think>'
T_CLOSE = '</' + 'think>'
A_OPEN = '<' + 'answer>'
A_CLOSE = '</' + 'answer>'


def make_record(task_id: str = 'task', answer: str = '42', **overrides: object) -> dict:
    record = {
        'schema_version': 2, 'id': record_id(task_id, '1', 'prompt', 'explanation', answer),
        'task_id': task_id, 'verifier_version': '1', 'prompt': 'prompt',
        'trace': 'explanation', 'answer': answer, 'attempts': 1,
        'checks': [{'verifier': 'math_equality', 'passed': True,
                    'message': 'ok', 'target': 'answer'}],
        'created_at': '2026-01-01T00:00:00+00:00', 'generator_id': None,
        'run_id': None, 'review': None}
    record.update(overrides)
    return record


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    return path


@pytest.mark.parametrize('mutation', [
    {'schema_version': 3}, {'schema_version': '1'}, {'schema_version': True},
    {'id': 'not-hex'}, {'id': ''}, {'id': 123}, {'attempts': 0}, {'attempts': 21},
    {'attempts': True}, {'attempts': 1.0}, {'attempts': '2'}, {'attempts': None},
    {'prompt': ''}, {'prompt': '   '}, {'prompt': None}, {'trace': ''},
    {'checks': []}, {'checks': {}}, {'checks': None},
    {'checks': [{'verifier': 'x', 'passed': True, 'message': 'm', 'target': 'nonsense'}]},
    {'checks': [{'verifier': 'x', 'passed': False, 'message': 'm', 'target': 'answer'}]},
    {'checks': [{'verifier': '', 'passed': True, 'message': 'm', 'target': 'answer'}]},
    {'checks': [{'verifier': 'x', 'passed': True, 'message': 5, 'target': 'answer'}]},
    {'checks': [{'verifier': 'x', 'passed': True, 'message': 'm'}]},  # v2 missing target
    {'review': {'approved': False, 'source': 'human'}},
    {'review': {'approved': True, 'source': ''}},
    {'review': True},
    {'generator_id': ''}, {'run_id': 7}, {'trace': 'x' * 40000},
])
def test_invalid_records_rejected(mutation: dict) -> None:
    record = make_record()
    record.update(mutation)
    with pytest.raises(DatasetError):
        validate_record(record)


def test_v1_records_still_readable() -> None:
    record = make_record()
    v1 = {k: v for k, v in record.items() if k not in
          ('created_at', 'generator_id', 'run_id', 'review')}
    v1['schema_version'] = 1
    v1['checks'] = [{'verifier': 'math_equality', 'passed': True, 'message': 'ok'}]
    validate_record(v1)


def test_conflicting_ids_rejected(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path/'d.jsonl', [make_record(answer='42')])
    conflicting = make_record()
    conflicting['answer'] = '43'
    conflicting['trace'] = 'different'
    with pytest.raises(DatasetError, match='differ in content'):
        DatasetStore(path).add(conflicting)
    assert json.loads(path.read_text().splitlines()[0])['answer'] == '42'


def test_identical_duplicate_collapsed(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path/'d.jsonl', [make_record()])
    repeat = make_record(created_at='2027-01-01T00:00:00+00:00')  # only provenance differs
    DatasetStore(path).add(repeat)
    assert len(read_dataset(path)) == 1


def test_read_dataset_conflict_line_context(tmp_path: Path) -> None:
    first, second = make_record(task_id='a'), make_record(task_id='b')
    second['id'] = first['id']
    path = write_jsonl(tmp_path/'d.jsonl', [first, second])
    with pytest.raises(DatasetError, match='line 2'):
        read_dataset(path)


def test_invalid_existing_file_leaves_original(tmp_path: Path) -> None:
    path = tmp_path/'d.jsonl'
    path.write_text('{"schema_version": 1, "broken"\n', encoding='utf-8')
    with pytest.raises(DatasetError, match='line 1'):
        DatasetStore(path).add(make_record())
    assert path.read_text() == '{"schema_version": 1, "broken"\n'


def test_temporary_write_failure_preserves_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_jsonl(tmp_path/'d.jsonl', [make_record()])
    original = path.read_text()
    def failing_mkstemp(*args: object, **kwargs: object) -> None:
        raise OSError('no space for temporary file')
    monkeypatch.setattr('prism_zero.dataset.tempfile.mkstemp', failing_mkstemp)
    with pytest.raises(OSError):
        DatasetStore(path).add(make_record(task_id='new'))
    assert path.read_text() == original


def test_fsync_failure_preserves_original_and_cleans_temporary(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_jsonl(tmp_path/'d.jsonl', [make_record()])
    original = path.read_text()
    def failing_fsync(fd: int) -> None:
        raise OSError('disk failure during flush')
    monkeypatch.setattr('prism_zero.dataset.os.fsync', failing_fsync)
    with pytest.raises(OSError):
        DatasetStore(path).add(make_record(task_id='new'))
    assert path.read_text() == original
    assert not [p for p in tmp_path.iterdir() if p.name.startswith('.prism-zero-')]


def test_replacement_failure_reports_and_cleans(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_jsonl(tmp_path/'d.jsonl', [make_record()])
    original = path.read_text()
    def failing_replace(src: object, dst: object) -> None:
        raise OSError('replacement refused')
    monkeypatch.setattr('prism_zero.dataset.os.replace', failing_replace)
    with pytest.raises(OSError, match='replacement refused'):
        DatasetStore(path).add(make_record(task_id='new'))
    assert path.read_text() == original
    assert not [p for p in tmp_path.iterdir() if p.name.startswith('.prism-zero-')]


def test_threads_sharing_one_store_are_serialized(tmp_path: Path) -> None:
    store = DatasetStore(tmp_path/'d.jsonl')
    errors: list[Exception] = []

    def insert(index: int) -> None:
        try:
            store.add(make_record(task_id=f'task-{index}'))
        except Exception as error:  # pragma: no cover - failure path
            errors.append(error)

    threads = [threading.Thread(target=insert, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    records = read_dataset(tmp_path/'d.jsonl')
    assert len(records) == 8


def test_candidate_roundtrip_enforced() -> None:
    record = make_record(trace='nested ' + T_OPEN + 'tag' + T_CLOSE)
    with pytest.raises(DatasetError, match='envelope'):
        validate_record(record)


def test_strict_json_records_rejected(tmp_path: Path) -> None:
    path = tmp_path/'d.jsonl'
    good = make_record()
    path.write_text(json.dumps(good) + '\n' + '{"x": 1, "x": 2}\n', encoding='utf-8')
    with pytest.raises(DatasetError, match='line 2'):
        read_dataset(path)
    path.write_text(json.dumps(good) + '\n' + '{"x": NaN}\n', encoding='utf-8')
    with pytest.raises(DatasetError, match='line 2'):
        read_dataset(path)


def test_fingerprint_stable_and_sensitive(tmp_path: Path) -> None:
    first = write_jsonl(tmp_path/'a.jsonl', [make_record(), make_record(task_id='other')])
    second = write_jsonl(tmp_path/'b.jsonl', [make_record(task_id='other'), make_record()])
    assert dataset_fingerprint(first) == dataset_fingerprint(second)  # order-independent
    changed = write_jsonl(tmp_path/'c.jsonl', [make_record(), make_record(task_id='other', answer='7')])
    assert dataset_fingerprint(first) != dataset_fingerprint(changed)


def test_content_view_excludes_provenance() -> None:
    base = make_record()
    later = dict(base, created_at='2099-01-01T00:00:00+00:00', run_id='other-run')
    assert content_view(base) == content_view(later)


def test_error_messages_do_not_echo_content() -> None:
    record = make_record(prompt='SECRET-PROMPT-CONTENT', trace='SECRET-TRACE-CONTENT')
    record['prompt'] = ''
    try:
        validate_record(record)
    except DatasetError as error:
        assert 'SECRET' not in str(error)
    else:  # pragma: no cover
        pytest.fail('expected DatasetError')
