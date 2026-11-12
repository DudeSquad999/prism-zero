"""Envelope parsing for generated candidates.

The envelope format is the contract between generation, verification, and
training: one think block followed by one answer block. Parsing never executes
or evaluates candidate content.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from .verifiers import bounded

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
