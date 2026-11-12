"""Evaluation metric, task-contract, and overlap regressions using scripted outputs."""
import json
from pathlib import Path
import pytest
from prism_zero.candidates import Candidate
from prism_zero.dataset import record_id
from prism_zero.evaluation import (TaskSetError, build_rule, check_overlap, compute_metrics,
                                   format_summary, load_task_set, run_condition,
                                   run_evaluation, task_set_fingerprint)

T_OPEN = '<' + 'think>'
T_CLOSE = '</' + 'think>'
A_OPEN = '<' + 'answer>'
A_CLOSE = '</' + 'answer>'


def envelope(answer: str) -> str:
    return Candidate('Concise explanation.', answer).render()


class ScriptedGenerator:
    """Prompt-keyed deterministic outputs, replayed identically for every
    condition; an output of None raises a scripted infrastructure failure."""
    def __init__(self, outputs: dict[str, list[str | None]]) -> None:
        self.outputs = {prompt: list(items) for prompt, items in outputs.items()}
        self.counts: dict[str, int] = {prompt: 0 for prompt in outputs}
        self.calls: list[str] = []

    def generate(self, prompt: str, feedback: str | None) -> str:
        self.calls.append(prompt)
        items = self.outputs[prompt]
        item = items[self.counts[prompt] % len(items)]
        self.counts[prompt] += 1
        if item is None:
            raise RuntimeError('scripted infrastructure failure')
        return item


def task_set() -> dict:
    return {
        'task_set_version': 'test-v1',
        'tasks': [
            {'task_id': 'arith-1', 'category': 'arithmetic', 'prompt': 'p-a1',
             'group': 'arith',
             'rules': [{'verifier': 'math_equality', 'expected': '42'}]},
            {'task_id': 'arith-2', 'category': 'arithmetic', 'prompt': 'p-a2', 'group': 'arith',
             'rules': [{'verifier': 'math_equality', 'expected': '42'}]},
            {'task_id': 'json-1', 'category': 'json', 'prompt': 'p-j1',
             'rules': [{'verifier': 'json_schema',
                        'schema': {'type': 'object', 'required': ['ok']}}]},
            {'task_id': 'json-2', 'category': 'json', 'prompt': 'p-j2',
             'rules': [{'verifier': 'json_schema',
                        'schema': {'type': 'object', 'required': ['ok']}}]},
        ]}


def write_task_set(path: Path, tasks: dict | None = None) -> Path:
    path.write_text(json.dumps(tasks or task_set()), encoding='utf-8')
    return path


def make_training_record(task_id: str, prompt: str) -> dict:
    return {'schema_version': 2, 'id': record_id(task_id, '1', prompt, 'trace', '42'),
            'task_id': task_id, 'verifier_version': '1', 'prompt': prompt, 'trace': 'trace',
            'answer': '42', 'attempts': 1,
            'checks': [{'verifier': 'math_equality', 'passed': True, 'message': 'ok',
                        'target': 'answer'}],
            'created_at': '2026-01-01T00:00:00+00:00', 'generator_id': None,
            'run_id': None, 'review': None}


def fixture_generator() -> ScriptedGenerator:
    return ScriptedGenerator({
        'p-a1': ['not an envelope', envelope('42')],
        'p-a2': [envelope('42')],
        'p-j1': [envelope('42'), envelope('41'), envelope('40')],
        'p-j2': [None],
    })


def test_condition_metrics_are_exact(tmp_path: Path) -> None:
    path = write_task_set(tmp_path / 'tasks.json')
    result = run_condition('base_budget', fixture_generator(), load_task_set(path), 3)
    metrics = compute_metrics(result)
    assert metrics['task_count'] == 4
    assert metrics['completed_task_count'] == 3
    assert metrics['accepted_task_count'] == 2
    assert metrics['first_attempt_pass_rate'] == pytest.approx(1 / 4)
    assert metrics['pass_rate_within_budget'] == pytest.approx(2 / 4)
    assert metrics['average_attempts_all_tasks'] == pytest.approx(2.0)
    assert metrics['average_attempts_successful'] == pytest.approx(1.5)
    assert metrics['invalid_envelope_rate'] == pytest.approx(1 / 6)
    assert metrics['infrastructure_error_count'] == 1
    assert metrics['per_category']['arithmetic']['pass_rate_within_budget'] == pytest.approx(1.0)
    assert metrics['per_category']['json']['accepted_task_count'] == 0
    assert metrics['per_category']['json']['infrastructure_error_count'] == 1


def test_single_attempt_condition(tmp_path: Path) -> None:
    path = write_task_set(tmp_path / 'tasks.json')
    result = run_condition('base_1_attempt', fixture_generator(), load_task_set(path), 1)
    metrics = compute_metrics(result)
    assert metrics['accepted_task_count'] == 1  # only arith-2 passes on the first attempt
    assert metrics['average_attempts_all_tasks'] == pytest.approx(1.0)


def test_four_conditions_with_not_run(tmp_path: Path) -> None:
    path = write_task_set(tmp_path / 'tasks.json')
    evaluation = run_evaluation(fixture_generator(), None, load_task_set(path),
                                max_attempts=3)
    assert set(evaluation['conditions']) == {'base_1_attempt', 'base_budget',
                                            'adapter_1_attempt', 'adapter_budget'}
    assert evaluation['conditions']['adapter_1_attempt']['status'] == 'NOT RUN'
    assert evaluation['conditions']['base_budget']['status'] == 'RUN'
    summary = format_summary(evaluation)
    assert 'NOT RUN' in summary and 'overlap: NOT CHECKED' in summary


def test_overlap_detection(tmp_path: Path) -> None:
    path = write_task_set(tmp_path / 'tasks.json')
    dataset = tmp_path / 'train.jsonl'
    dataset.write_text(json.dumps(make_training_record('arith-1', 'p-a1')) + '\n'
                       + json.dumps(make_training_record('other', 'unrelated')) + '\n')
    evaluation = run_evaluation(fixture_generator(), None, load_task_set(path),
                                dataset_path=dataset)
    overlap = evaluation['overlap']
    assert overlap['checked'] is True
    assert overlap['shared_task_ids'] == ['arith-1']
    assert overlap['shared_prompts'] == ['arith-1']
    assert 'exact-match check only' in format_summary(evaluation)


def test_task_set_validation(tmp_path: Path) -> None:
    bad_sets = [
        {'task_set_version': 1, 'tasks': []},
        {'task_set_version': 'v', 'tasks': [{'task_id': 'a', 'category': 'c', 'prompt': 'p',
                                             'rules': [{'verifier': 'evil'}]}]},
        {'task_set_version': 'v', 'tasks': [{'task_id': 'a', 'category': 'c', 'prompt': 'p',
                                             'rules': [{'verifier': 'math_equality',
                                                        'expected': '1', 'surprise': 1}]}]},
        {'task_set_version': 'v', 'tasks': [{'task_id': 'a', 'category': 'c', 'prompt': 'p',
                                             'rules': [{'verifier': 'math_equality',
                                                        'expected': '1', 'target': 'trace'}]}]},
        {'task_set_version': 'v', 'tasks': [{'task_id': 'a', 'category': 'c', 'prompt': 'p',
                                             'rules': []}]},
        {'task_set_version': 'v', 'tasks': [{'task_id': '', 'category': 'c', 'prompt': 'p',
                                             'rules': [{'verifier': 'math_equality',
                                                        'expected': '1'}]}]},
        {'task_set_version': 'v', 'tasks': [{'task_id': 'a', 'category': 'c', 'prompt': 'p',
                                             'rules': [{'verifier': 'math_equality',
                                                        'expected': '1'}]},
                                            {'task_id': 'a', 'category': 'c', 'prompt': 'q',
                                             'rules': [{'verifier': 'math_equality',
                                                        'expected': '1'}]}]},
        {'task_set_version': 'v', 'tasks': [{'task_id': 'a', 'category': 'c', 'prompt': 'p',
                                             'rules': [{'verifier': 'math_equality'}]}]},
    ]
    for index, tasks in enumerate(bad_sets):
        path = tmp_path / f'bad-{index}.json'
        path.write_text(json.dumps(tasks), encoding='utf-8')
        with pytest.raises(TaskSetError):
            load_task_set(path)
    (tmp_path / 'broken.json').write_text('{invalid', encoding='utf-8')
    with pytest.raises(TaskSetError, match='invalid JSON'):
        load_task_set(tmp_path / 'broken.json')


def test_build_rule_registry_is_allowlisted() -> None:
    verifier, target = build_rule({'verifier': 'python_ast'}, 'x')
    assert verifier.name == 'python_ast' and target == 'answer'
    with pytest.raises(TaskSetError, match='unknown verifier'):
        build_rule({'verifier': 'importlib.import_module'}, 'x')
    with pytest.raises(TaskSetError, match='unknown rule keys'):
        build_rule({'verifier': 'regex', 'pattern': 'a', 'module': 'os'}, 'x')


def test_fingerprint_tracks_content(tmp_path: Path) -> None:
    first = load_task_set(write_task_set(tmp_path / 'a.json'))
    second = load_task_set(write_task_set(tmp_path / 'b.json'))
    assert task_set_fingerprint(first) == task_set_fingerprint(second)
    changed = task_set()
    changed['tasks'][0]['prompt'] = 'different'
    third = load_task_set(write_task_set(tmp_path / 'c.json', changed))
    assert task_set_fingerprint(first) != task_set_fingerprint(third)


def test_no_evaluation_storage_side_effects(tmp_path: Path) -> None:
    """Evaluation must never curate results into the training dataset."""
    path = write_task_set(tmp_path / 'tasks.json')
    dataset = tmp_path / 'train.jsonl'
    run_evaluation(fixture_generator(), None, load_task_set(path))
    assert not dataset.exists()
    with pytest.raises(FileNotFoundError):
        run_evaluation(fixture_generator(), None, load_task_set(path), dataset_path=dataset)
    assert not dataset.exists()
