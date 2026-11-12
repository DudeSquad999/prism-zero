"""Bounded retry orchestration and single-writer accepted-only JSONL curation."""
from __future__ import annotations
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence
from .candidates import Candidate
from .dataset import DatasetStore, SCHEMA_VERSION, record_id
from .verifiers import Check, Verifier, bounded

LOGGER = logging.getLogger(__name__)
MAX_ATTEMPTS_RANGE = (1, 20)


class Generator(Protocol):
    def generate(self, prompt: str, feedback: str | None) -> str:
        """Return one complete candidate using a local backend."""
        raise NotImplementedError


@dataclass(frozen=True)
class Rule:
    verifier: Verifier
    target: str = 'answer'

    def __post_init__(self) -> None:
        if self.target not in {'answer', 'trace'}:
            raise ValueError('Rule target must be answer or trace')


@dataclass(frozen=True)
class RunResult:
    accepted: bool
    attempts: int
    candidate: Candidate | None
    checks: tuple[Check, ...]
    record_id: str | None


class PrismZeroEngine:
    """Reject invalid candidates, supply feedback, and export only passing ones.

    Retry feedback is not weight learning. Backend and verifier implementation
    errors propagate; they are not converted into content-verification failures.
    """
    def __init__(self, generator: Generator, rules: Sequence[Rule],
                 dataset: Path = Path('prism_zero_dataset.jsonl'), max_attempts: int = 3,
                 task_id: str = 'default', verifier_version: str = '1',
                 generator_id: str | None = None, run_id: str | None = None) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError(f'max_attempts must be an integer between {MAX_ATTEMPTS_RANGE[0]} and '
                             f'{MAX_ATTEMPTS_RANGE[1]}; correct the value to a plain integer')
        if not MAX_ATTEMPTS_RANGE[0] <= max_attempts <= MAX_ATTEMPTS_RANGE[1]:
            raise ValueError(f'max_attempts must be between {MAX_ATTEMPTS_RANGE[0]} and '
                             f'{MAX_ATTEMPTS_RANGE[1]}; got {max_attempts}')
        if not rules or not any(rule.target == 'answer' for rule in rules):
            raise ValueError('At least one answer rule is required')
        if not task_id.strip() or not verifier_version.strip():
            raise ValueError('Task and verifier-version identifiers are required')
        if generator_id is not None and (not isinstance(generator_id, str) or not generator_id.strip()):
            raise ValueError('generator_id must be a nonempty string when provided')
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError('run_id must be a nonempty string when provided')
        self.generator, self.rules = generator, tuple(rules)
        self.store, self.max_attempts = DatasetStore(dataset), max_attempts
        self.task_id, self.verifier_version = task_id, verifier_version
        self.generator_id, self.run_id = generator_id, run_id

    def run(self, prompt: str) -> RunResult:
        bounded(prompt, 16384)
        if not prompt.strip():
            raise ValueError('Prompt must be nonempty')
        feedback: str | None = None
        checks: tuple[Check, ...] = ()
        for attempt in range(1, self.max_attempts + 1):
            LOGGER.info('Generation attempt %d/%d', attempt, self.max_attempts)
            raw = self.generator.generate(prompt, feedback)
            try:
                candidate = Candidate.parse(raw)
            except ValueError as error:
                checks = (Check('envelope', False, str(error)),)
            else:
                try:
                    checks = tuple(rule.verifier.verify(getattr(candidate, rule.target)) for rule in self.rules)
                except Exception:
                    LOGGER.exception('Verifier implementation failed')
                    raise
                if all(check.passed for check in checks):
                    identifier = record_id(self.task_id, self.verifier_version, prompt,
                                           candidate.trace, candidate.answer)
                    self.store.add({
                        'schema_version': SCHEMA_VERSION, 'id': identifier, 'task_id': self.task_id,
                        'verifier_version': self.verifier_version, 'prompt': prompt,
                        'trace': candidate.trace, 'answer': candidate.answer,
                        'attempts': attempt,
                        'checks': [dict(asdict(check), target=rule.target)
                                   for check, rule in zip(checks, self.rules)],
                        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        'generator_id': self.generator_id, 'run_id': self.run_id,
                        'review': None})
                    LOGGER.info('Accepted candidate on attempt %d', attempt)
                    return RunResult(True, attempt, candidate, checks, identifier)
            feedback = 'Verification failed. Return a fresh complete candidate. ' + '; '.join(
                check.message for check in checks if not check.passed)
            LOGGER.info('Candidate rejected on attempt %d', attempt)
        return RunResult(False, self.max_attempts, None, checks, None)
