"""Labeled scripted generator for offline demos and evaluation fixtures.

This is a deterministic test double, not a pretrained model. It exists so the
demo, CLI, and evaluation tests run without model weights or ML dependencies.
"""
from __future__ import annotations

THINK_OPEN = '<' + 'think>'
THINK_CLOSE = '</' + 'think>'
ANSWER_OPEN = '<' + 'answer>'
ANSWER_CLOSE = '</' + 'answer>'


class ScriptedModel:
    """A labeled test double, not a pretrained reasoning model."""
    def generate(self, prompt: str, feedback: str | None) -> str:
        if feedback is None:
            trace = 'I estimate the product as 41.'
            answer = '41'
        else:
            trace = 'Six groups of seven total forty-two.'
            answer = '42'
        return f'{THINK_OPEN}{trace}{THINK_CLOSE}\n{ANSWER_OPEN}{answer}{ANSWER_CLOSE}'
