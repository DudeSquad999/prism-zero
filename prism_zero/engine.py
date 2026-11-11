"""Bounded retry orchestration and single-writer accepted-only JSONL curation."""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence
from .verifiers import Check, Verifier, bounded, strict_json

LOGGER = logging.getLogger(__name__)
ENVELOPE = re.compile(r'\A<think>(?P<trace>[\s\S]*?)</think>\s*<answer>(?P<answer>[\s\S]*?)</answer>\Z')

@dataclass(frozen=True)
class Candidate:
    trace: str
    answer: str
    def render(self) -> str:
        return f'<think>{self.trace}</think>\n<answer>{self.answer}</answer>'
    @classmethod
    def parse(cls, text: str) -> Candidate:
        bounded(text)
        match = ENVELOPE.fullmatch(text.strip())
        if match is None:
            raise ValueError('Return one think block followed by one answer block')
        trace, answer = match['trace'].strip(), match['answer'].strip()
        if not trace or not answer or any(tag in trace + answer for tag in
                                         ('<think>', '</think>', '<answer>', '</answer>')):
            raise ValueError('Blocks must be nonempty and nonnested')
        return cls(trace, answer)

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

class DatasetStore:
    """Atomic replacement, one writer process/store instance per dataset path.

    The lock serializes threads using this instance, not multiple processes.
    Rewriting the complete file is intended for small local datasets.
    """
    def __init__(self, path: Path) -> None:
        self.path, self.lock = path, threading.Lock()
    def add(self, record: dict[str, object]) -> None:
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, object]] = []
            if self.path.exists():
                with self.path.open(encoding='utf-8') as stream:
                    for number, line in enumerate(stream, 1):
                        item = strict_json(line)
                        if not isinstance(item, dict) or 'id' not in item:
                            raise ValueError(f'Invalid dataset record at line {number}')
                        records.append(item)
            if any(item['id'] == record['id'] for item in records):
                LOGGER.info('Duplicate accepted record skipped')
                return
            records.append(record)
            fd, temporary = tempfile.mkstemp(prefix='.prism-zero-', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                    for item in records:
                        stream.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

class PrismZeroEngine:
    """Reject invalid candidates, supply feedback, and export only passing ones.

    Retry feedback is not weight learning. Backend and verifier implementation
    errors propagate; they are not converted into content-verification failures.
    """
    def __init__(self, generator: Generator, rules: Sequence[Rule],
                 dataset: Path = Path('prism_zero_dataset.jsonl'), max_attempts: int = 3,
                 task_id: str = 'default', verifier_version: str = '1') -> None:
        if not 1 <= max_attempts <= 20:
            raise ValueError('max_attempts must be between 1 and 20')
        if not rules or not any(rule.target == 'answer' for rule in rules):
            raise ValueError('At least one answer rule is required')
        if not task_id.strip() or not verifier_version.strip():
            raise ValueError('Task and verifier-version identifiers are required')
        self.generator, self.rules = generator, tuple(rules)
        self.store, self.max_attempts = DatasetStore(dataset), max_attempts
        self.task_id, self.verifier_version = task_id, verifier_version
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
                    identity = json.dumps([self.task_id, self.verifier_version, prompt,
                                           candidate.trace, candidate.answer], ensure_ascii=True)
                    record_id = hashlib.sha256(identity.encode()).hexdigest()
                    self.store.add({'schema_version': 1, 'id': record_id, 'task_id': self.task_id,
                                    'verifier_version': self.verifier_version, 'prompt': prompt,
                                    'trace': candidate.trace, 'answer': candidate.answer,
                                    'attempts': attempt, 'checks': [asdict(check) for check in checks]})
                    LOGGER.info('Accepted candidate on attempt %d', attempt)
                    return RunResult(True, attempt, candidate, checks, record_id)
            feedback = 'Verification failed. Return a fresh complete candidate. ' + '; '.join(
                check.message for check in checks if not check.passed)
            LOGGER.info('Candidate rejected on attempt %d', attempt)
        return RunResult(False, self.max_attempts, None, checks, None)
