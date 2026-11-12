"""Shared dataset schema, validation, and single-writer JSONL storage.

One definition of a valid record is used by dataset writing, reading,
validation commands, training, and evaluation-import checks. Schema version 2
adds the verification target to each check plus optional provenance fields;
version 1 records remain readable and are never silently reinterpreted.

Validation errors carry file and line context and never echo record content,
which may contain private prompts, traces, or answers.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any
from .candidates import Candidate
from .verifiers import strict_json

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)
ID_PATTERN = re.compile(r'\A[0-9a-f]{16,128}\Z')
REQUIRED_STRINGS = ('id', 'task_id', 'verifier_version', 'prompt', 'trace', 'answer')
TEXT_LIMIT = 32768
ATTEMPT_RANGE = (1, 20)
RULE_TARGETS = ('answer', 'trace')


class DatasetError(ValueError):
    """A dataset violated the shared contract; context is safe to display."""


def _fail(location: str, message: str) -> DatasetError:
    return DatasetError(f'{location}: {message}')


def validate_record(record: Any, location: str = 'record') -> None:
    """Validate one parsed record against the shared contract.

    Version 1 records (no check targets, no provenance) are accepted as-is;
    version 2 records must record which text each check targeted.
    """
    if not isinstance(record, dict):
        raise _fail(location, 'Record must be a JSON object')
    version = record.get('schema_version')
    if isinstance(version, bool) or not isinstance(version, int) or version not in SUPPORTED_SCHEMA_VERSIONS:
        raise _fail(location, f'Unsupported schema_version; expected one of {SUPPORTED_SCHEMA_VERSIONS}')
    for key in REQUIRED_STRINGS:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            raise _fail(location, f'Missing nonempty string: {key}')
        if len(value) > TEXT_LIMIT:
            raise _fail(location, f'Field exceeds {TEXT_LIMIT} characters: {key}')
    if not ID_PATTERN.fullmatch(record['id']):
        raise _fail(location, 'id must be a hexadecimal content digest')
    attempts = record.get('attempts')
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not ATTEMPT_RANGE[0] <= attempts <= ATTEMPT_RANGE[1]:
        raise _fail(location, f'attempts must be an integer in {ATTEMPT_RANGE}')
    checks = record.get('checks')
    if not isinstance(checks, list) or not checks:
        raise _fail(location, 'checks must be a nonempty list')
    for position, check in enumerate(checks, 1):
        if not isinstance(check, dict):
            raise _fail(location, f'check {position} must be an object')
        if not isinstance(check.get('verifier'), str) or not check['verifier'].strip():
            raise _fail(location, f'check {position} requires a nonempty verifier name')
        if check.get('passed') is not True:
            raise _fail(location, f'check {position} did not pass; rejected candidates are not exportable')
        if not isinstance(check.get('message'), str):
            raise _fail(location, f'check {position} requires a message')
        target = check.get('target')
        if version >= 2 and target not in RULE_TARGETS:
            raise _fail(location, f'check {position} must record its target as answer or trace')
        if target is not None and target not in RULE_TARGETS:
            raise _fail(location, f'check {position} has an invalid target')
    try:
        Candidate.parse(Candidate(record['trace'], record['answer']).render())
    except ValueError as error:
        raise _fail(location, f'Candidate fields do not form a valid envelope: {error}') from error
    review = record.get('review')
    if review is not None:
        if not isinstance(review, dict) or review.get('approved') is not True \
                or not isinstance(review.get('source'), str) or not review['source'].strip():
            raise _fail(location, 'review must record human approval with a nonempty source')
    for key in ('generator_id', 'run_id', 'created_at'):
        value = record.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise _fail(location, f'{key} must be a nonempty string when present')


def parse_jsonl_line(line: str, path: Path | None = None, number: int | None = None) -> Any:
    """Strictly parse one JSONL line with file and line context on failure."""
    where = ' '.join(part for part in (str(path) if path else None,
                                       f'line {number}' if number else None) if part) or 'input'
    if not line.strip():
        raise _fail(where, 'Blank line')
    try:
        return strict_json(line)
    except (ValueError, RecursionError) as error:
        raise _fail(where, f'Invalid JSON: {error}') from error


def content_view(record: dict[str, Any]) -> str:
    """Canonical view of the identity-bearing fields the record id is built from.

    Provenance fields (timestamps, run identifiers, review status) are excluded:
    repeated observations of the same accepted content keep the first record's
    provenance, while the same id over different content is a corruption.
    """
    return json.dumps([record[key] for key in
                       ('task_id', 'verifier_version', 'prompt', 'trace', 'answer')],
                      ensure_ascii=True, allow_nan=False)


def read_dataset(path: Path) -> list[dict[str, Any]]:
    """Read and validate a dataset, rejecting conflicting duplicate identifiers.

    Records that repeat the same identifier with identical content are
    collapsed; the same identifier over different content is a conflict and
    fails with line context.
    """
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            record = parse_jsonl_line(line, path, number)
            try:
                validate_record(record, f'{path} line {number}')
            except DatasetError:
                raise
            identifier = record['id']
            content = content_view(record)
            if identifier in seen:
                if seen[identifier] != content:
                    raise _fail(f'{path} line {number}',
                                'Conflicting records share one id but differ in content')
                LOGGER.info('Duplicate identical record at line %d collapsed', number)
                continue
            seen[identifier] = content
            records.append(record)
    return records


def dataset_fingerprint(path: Path) -> str:
    """Content digest over the canonical records, independent of line order."""
    digest = hashlib.sha256()
    for record in sorted(read_dataset(path), key=lambda item: item['id']):
        digest.update(json.dumps(record, sort_keys=True, ensure_ascii=True,
                                 allow_nan=False).encode('utf-8'))
        digest.update(b'\n')
    return digest.hexdigest()


def record_id(task_id: str, verifier_version: str, prompt: str,
              trace: str, answer: str) -> str:
    """Stable content identity; timestamps and provenance are excluded on
    purpose so identical accepted examples stay deduplicatable across runs."""
    identity = json.dumps([task_id, verifier_version, prompt, trace, answer],
                          ensure_ascii=True)
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()


class DatasetStore:
    """Atomic replacement, one writer process/store instance per dataset path.

    The lock serializes threads using this instance, not multiple processes.
    Rewriting the complete file is intended for small local datasets. Invalid
    existing JSONL aborts an update without replacing the existing file.
    """
    def __init__(self, path: Path) -> None:
        self.path, self.lock = Path(path), threading.Lock()

    def add(self, record: dict[str, object]) -> None:
        with self.lock:
            validate_record(record, str(self.path))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing: list[dict[str, object]] = []
            if self.path.exists():
                with self.path.open(encoding='utf-8') as stream:
                    for number, line in enumerate(stream, 1):
                        item = parse_jsonl_line(line, self.path, number)
                        try:
                            validate_record(item, f'{self.path} line {number}')
                        except DatasetError:
                            raise
                        existing.append(item)
            incoming = content_view(record)
            for item in existing:
                if item['id'] == record['id']:
                    if content_view(item) == incoming:
                        LOGGER.info('Duplicate accepted record skipped')
                        return
                    raise _fail(str(self.path),
                                'Conflicting records share one id but differ in content')
            existing.append(record)
            fd, temporary = tempfile.mkstemp(prefix='.prism-zero-', dir=self.path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
                    for item in existing:
                        stream.write(json.dumps(item, ensure_ascii=True, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
