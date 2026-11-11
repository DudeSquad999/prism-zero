"""Reproducible two-attempt demo; no model downloads or ML dependencies."""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from prism_zero.engine import PrismZeroEngine, Rule
from prism_zero.verifiers import MathEqualityVerifier

class ScriptedModel:
    """A labeled test double, not a pretrained reasoning model."""
    def generate(self, prompt: str, feedback: str | None) -> str:
        if feedback is None:
            return '<think>I estimate the product as 41.</think><answer>41</answer>'
        return '<think>Six groups of seven total forty-two.</think><answer>42</answer>'

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('prism_zero_dataset.jsonl'))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        engine = PrismZeroEngine(ScriptedModel(), [Rule(MathEqualityVerifier('6*7'))],
                                 args.dataset, task_id='multiply-6-by-7')
        result = engine.run('What is six times seven? Return an integer.')
        if not result.accepted or result.attempts != 2 or result.candidate is None:
            raise RuntimeError('Demo failed')
        print(f'Accepted on attempt {result.attempts}; answer={result.candidate.answer}')
        print(f'Curated dataset: {args.dataset}')
        return 0
    except Exception:
        logging.exception('Quickstart failed')
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
