"""Independent evaluation of base and adapter conditions on held-out tasks.

Evaluation data must stay separate from training data. Expected answers and
verifiers come from the task set, never from the model being evaluated.
Simple overlap checks catch exact prompt or task-id collisions but do not
prove the absence of contamination.
"""
from __future__ import annotations
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol
from .candidates import Candidate
from .dataset import read_dataset
from .verifiers import (ASTVerifier, JSONSchemaVerifier, MathEqualityVerifier,
                        RegexVerifier, Verifier)

LOGGER = logging.getLogger(__name__)
EVALUATION_SCHEMA_VERSION = 1
CONDITIONS = ('base_1_attempt', 'base_budget', 'adapter_1_attempt', 'adapter_budget')

VERIFIER_REGISTRY: dict[str, Callable[[dict[str, Any]], Verifier]] = {
    'python_ast': lambda spec: ASTVerifier(),
    'json_schema': lambda spec: JSONSchemaVerifier(spec['schema']),
    'regex': lambda spec: RegexVerifier(spec['pattern'], spec.get('timeout', 0.05)),
    'math_equality': lambda spec: MathEqualityVerifier(spec['expected']),
}
VERIFIER_SPEC_KEYS = {
    'python_ast': set(),
    'json_schema': {'schema'},
    'regex': {'pattern', 'timeout'},
    'math_equality': {'expected'},
}


class TaskSetError(ValueError):
    """A task set violated the evaluation contract."""


class Generator(Protocol):
    def generate(self, prompt: str, feedback: str | None) -> str:
        raise NotImplementedError


def build_rule(spec: Any, location: str) -> tuple[Verifier, str]:
    if not isinstance(spec, dict):
        raise TaskSetError(f'{location}: rule must be an object')
    name = spec.get('verifier')
    if name not in VERIFIER_REGISTRY:
        raise TaskSetError(f'{location}: unknown verifier {name!r}; allowed: '
                           f'{sorted(VERIFIER_REGISTRY)}')
    allowed = VERIFIER_SPEC_KEYS[name] | {'verifier', 'target'}
    unknown = set(spec) - allowed
    if unknown:
        raise TaskSetError(f'{location}: unknown rule keys {sorted(unknown)}')
    try:
        verifier = VERIFIER_REGISTRY[name](spec)
    except (KeyError, TypeError, ValueError) as error:
        raise TaskSetError(f'{location}: invalid verifier configuration: {error}') from error
    target = spec.get('target', 'answer')
    if target not in ('answer', 'trace'):
        raise TaskSetError(f'{location}: rule target must be answer or trace')
    return verifier, target


def load_task_set(path: Path) -> dict[str, Any]:
    """Load and validate a task set; expected answers are trusted configuration."""
    raw = path.read_text(encoding='utf-8')
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise TaskSetError(f'{path}: invalid JSON: {error}') from error
    if not isinstance(parsed, dict):
        raise TaskSetError(f'{path}: task set must be an object')
    unknown = set(parsed) - {'task_set_version', 'tasks'}
    if unknown:
        raise TaskSetError(f'{path}: unknown task-set keys {sorted(unknown)}')
    version = parsed.get('task_set_version')
    if not isinstance(version, str) or not version.strip():
        raise TaskSetError(f'{path}: task_set_version must be a nonempty string')
    tasks = parsed.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise TaskSetError(f'{path}: tasks must be a nonempty list')
    seen: set[str] = set()
    for index, task in enumerate(tasks, 1):
        where = f'{path} task {index}'
        if not isinstance(task, dict):
            raise TaskSetError(f'{where}: task must be an object')
        unknown = set(task) - {'task_id', 'category', 'prompt', 'rules', 'group'}
        if unknown:
            raise TaskSetError(f'{where}: unknown task keys {sorted(unknown)}')
        task_id = task.get('task_id')
        if not isinstance(task_id, str) or not task_id.strip():
            raise TaskSetError(f'{where}: task_id must be a nonempty string')
        if task_id in seen:
            raise TaskSetError(f'{where}: duplicate task_id {task_id!r}')
        seen.add(task_id)
        if not isinstance(task.get('category'), str) or not task['category'].strip():
            raise TaskSetError(f'{where}: category must be a nonempty string')
        prompt = task.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise TaskSetError(f'{where}: prompt must be a nonempty string')
        if len(prompt) > 16384:
            raise TaskSetError(f'{where}: prompt exceeds 16384 characters')
        rules = task.get('rules')
        if not isinstance(rules, list) or not rules:
            raise TaskSetError(f'{where}: at least one rule is required')
        for position, spec in enumerate(rules, 1):
            _, target = build_rule(spec, f'{where} rule {position}')
            if position == 1 and target != 'answer':
                raise TaskSetError(f'{where}: the first rule must target the answer')
        group = task.get('group')
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise TaskSetError(f'{where}: group must be a nonempty string when present')
    return parsed


def task_set_fingerprint(task_set: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(task_set, sort_keys=True,
                                     ensure_ascii=True).encode('utf-8')).hexdigest()


@dataclass
class TaskOutcome:
    task_id: str
    category: str
    accepted: bool
    attempts: int
    first_attempt_passed: bool
    invalid_envelopes: int
    infrastructure_error: str | None = None


@dataclass
class ConditionResult:
    condition: str
    outcomes: list[TaskOutcome] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    status: str = 'RUN'
    reason: str | None = None


def run_condition(condition: str, generator: Generator, task_set: dict[str, Any],
                  max_attempts: int) -> ConditionResult:
    """Run one condition; candidate rejection and infrastructure failure stay distinct."""
    result = ConditionResult(condition)
    started = time.perf_counter()
    for task in task_set['tasks']:
        rules = [build_rule(spec, f"task {task['task_id']}") for spec in task['rules']]
        feedback: str | None = None
        outcome = TaskOutcome(task['task_id'], task['category'], False, 0, False, 0)
        for attempt in range(1, max_attempts + 1):
            outcome.attempts = attempt
            try:
                raw = generator.generate(task['prompt'], feedback)
            except Exception as error:
                LOGGER.exception('Infrastructure failure generating for task %s', task['task_id'])
                outcome.infrastructure_error = f'{type(error).__name__}: {error}'
                break
            try:
                candidate = Candidate.parse(raw)
            except ValueError:
                outcome.invalid_envelopes += 1
                feedback = 'Verification failed. Return one think block then one answer block.'
                continue
            try:
                checks = [verifier.verify(getattr(candidate, target))
                          for verifier, target in rules]
            except Exception as error:
                LOGGER.exception('Verifier failure on task %s', task['task_id'])
                outcome.infrastructure_error = f'{type(error).__name__}: {error}'
                break
            if all(check.passed for check in checks):
                outcome.accepted = True
                outcome.first_attempt_passed = attempt == 1
                break
            feedback = 'Verification failed. Return a fresh complete candidate. ' + '; '.join(
                check.message for check in checks if not check.passed)
        result.outcomes.append(outcome)
    result.elapsed_seconds = round(time.perf_counter() - started, 3)
    return result


def compute_metrics(result: ConditionResult) -> dict[str, Any]:
    """Every metric states its denominator; infrastructure failures are counted separately."""
    outcomes = result.outcomes
    total = len(outcomes)
    completed = [o for o in outcomes if o.infrastructure_error is None]
    accepted = [o for o in completed if o.accepted]
    generations = sum(o.attempts for o in completed)
    invalid = sum(o.invalid_envelopes for o in completed)
    return {
        'status': result.status,
        'reason': result.reason,
        'task_count': total,
        'completed_task_count': len(completed),
        'accepted_task_count': len(accepted),
        'first_attempt_pass_rate': (sum(1 for o in accepted if o.first_attempt_passed) / total
                                    if total else None),
        'pass_rate_within_budget': (len(accepted) / total if total else None),
        'average_attempts_all_tasks': (sum(o.attempts for o in completed) / len(completed)
                                       if completed else None),
        'average_attempts_successful': (sum(o.attempts for o in accepted) / len(accepted)
                                        if accepted else None),
        'invalid_envelope_rate': (invalid / generations if generations else None),
        'infrastructure_error_count': total - len(completed),
        'elapsed_seconds': result.elapsed_seconds,
        'average_task_seconds': (result.elapsed_seconds / len(completed) if completed else None),
        'per_category': _per_category(outcomes),
    }


def _per_category(outcomes: list[TaskOutcome]) -> dict[str, Any]:
    categories: dict[str, list[TaskOutcome]] = {}
    for outcome in outcomes:
        categories.setdefault(outcome.category, []).append(outcome)
    summary: dict[str, Any] = {}
    for category, items in sorted(categories.items()):
        accepted = [o for o in items if o.infrastructure_error is None and o.accepted]
        summary[category] = {
            'task_count': len(items),
            'accepted_task_count': len(accepted),
            'first_attempt_pass_rate': (sum(1 for o in accepted if o.first_attempt_passed)
                                        / len(items) if items else None),
            'pass_rate_within_budget': (len(accepted) / len(items) if items else None),
            'infrastructure_error_count': sum(1 for o in items if o.infrastructure_error),
        }
    return summary


def check_overlap(task_set: dict[str, Any], dataset_path: Path | None) -> dict[str, Any]:
    """Exact prompt and task-id overlap with the training dataset.

    Exact-match checks do not prove the absence of paraphrased contamination.
    """
    report: dict[str, Any] = {'dataset': str(dataset_path) if dataset_path else None,
                              'shared_task_ids': [], 'shared_prompts': [],
                              'shared_groups': []}
    if dataset_path is None:
        report['checked'] = False
        return report
    records = read_dataset(dataset_path)
    training_prompts = {record['prompt'] for record in records}
    training_task_ids = {record['task_id'] for record in records}
    training_groups = {record.get('task_id', '').rsplit('-', 1)[0] for record in records}
    for task in task_set['tasks']:
        if task['task_id'] in training_task_ids:
            report['shared_task_ids'].append(task['task_id'])
        if task['prompt'] in training_prompts:
            report['shared_prompts'].append(task['task_id'])
        group = task.get('group')
        if group and group in training_groups:
            report['shared_groups'].append(task['task_id'])
    report['checked'] = True
    return report


def run_evaluation(base_generator: Generator, adapter_generator: Generator | None,
                   task_set: dict[str, Any], max_attempts: int = 3,
                   dataset_path: Path | None = None) -> dict[str, Any]:
    """Compare four comparable conditions; missing adapter conditions are NOT RUN."""
    conditions: dict[str, Any] = {}
    generators = {'base_1_attempt': base_generator, 'base_budget': base_generator,
                  'adapter_1_attempt': adapter_generator, 'adapter_budget': adapter_generator}
    attempts = {'base_1_attempt': 1, 'base_budget': max_attempts,
                'adapter_1_attempt': 1, 'adapter_budget': max_attempts}
    for condition in CONDITIONS:
        generator = generators[condition]
        if generator is None:
            conditions[condition] = {'status': 'NOT RUN',
                                     'reason': 'No adapter generator was provided'}
            continue
        result = run_condition(condition, generator, task_set, attempts[condition])
        conditions[condition] = compute_metrics(result)
    return {
        'evaluation_schema_version': EVALUATION_SCHEMA_VERSION,
        'task_set': {'version': task_set['task_set_version'],
                     'fingerprint': task_set_fingerprint(task_set),
                     'task_count': len(task_set['tasks'])},
        'attempt_budget': max_attempts,
        'conditions': conditions,
        'overlap': check_overlap(task_set, dataset_path),
    }


def format_summary(evaluation: dict[str, Any]) -> str:
    """Concise terminal summary; machine-readable JSON goes to stdout separately."""
    lines = [f"Task set {evaluation['task_set']['version']}: "
             f"{evaluation['task_set']['task_count']} tasks, "
             f"attempt budget {evaluation['attempt_budget']}"]
    for name, metrics in evaluation['conditions'].items():
        if metrics.get('status') == 'NOT RUN':
            lines.append(f'{name:18} NOT RUN ({metrics.get("reason")})')
            continue
        first = metrics['first_attempt_pass_rate']
        budget = metrics['pass_rate_within_budget']
        lines.append(f"{name:18} first-attempt {first:.0%}  within-budget {budget:.0%}  "
                     f"invalid-envelope {metrics['invalid_envelope_rate'] or 0:.0%}  "
                     f"infra-errors {metrics['infrastructure_error_count']}")
    overlap = evaluation.get('overlap', {})
    if overlap.get('checked'):
        lines.append(f"overlap: {len(overlap['shared_prompts'])} shared prompts, "
                     f"{len(overlap['shared_task_ids'])} shared task ids "
                     "(exact-match check only)")
    else:
        lines.append('overlap: NOT CHECKED (no training dataset provided)')
    return '\n'.join(lines)
