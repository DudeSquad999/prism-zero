"""Offline regressions with scripted generators; no model weights required."""
from pathlib import Path
import pytest
from prism_zero.candidates import Candidate
from prism_zero.dataset import DatasetError, DatasetStore, read_dataset
from prism_zero.engine import PrismZeroEngine, Rule
from prism_zero.verifiers import ASTVerifier, JSONSchemaVerifier, MathEqualityVerifier, RegexVerifier
from prism_zero.trainer import encode_record

T_OPEN = '<' + 'think>'
T_CLOSE = '</' + 'think>'
A_OPEN = '<' + 'answer>'
A_CLOSE = '</' + 'answer>'


class Script:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.feedback: list[str | None] = []
    def generate(self, prompt: str, feedback: str | None) -> str:
        self.feedback.append(feedback)
        return next(self.outputs)

def output(answer: str) -> str:
    return Candidate('Concise explanation.', answer).render()

@pytest.mark.parametrize('text,valid', [('x=1', True), ('x=', False)])
def test_ast(text: str, valid: bool) -> None:
    assert ASTVerifier().verify(text).passed is valid

@pytest.mark.parametrize('text,valid', [('42', True), ('6*7', True), ('41', False),
    ('1=1', False), ('True', False), ('__import__("os")', False),
    ('2**99999', False), ('1/0', False), ('42.0', False)])
def test_math(text: str, valid: bool) -> None:
    assert MathEqualityVerifier('42').verify(text).passed is valid

@pytest.mark.parametrize('text', ['{"x":41}', '{"x":42,"x":41}', '{"x":NaN}', '[]'])
def test_bad_json(text: str) -> None:
    assert not JSONSchemaVerifier({'type':'object','required':['x'],
                                  'properties':{'x':{'const':42}}}).verify(text).passed

def test_json_and_refs() -> None:
    assert JSONSchemaVerifier({'type':'integer'}).verify('42').passed
    assert not JSONSchemaVerifier({'type':'number'}).verify('1e9999').passed
    with pytest.raises(ValueError):
        JSONSchemaVerifier({'$ref':'https://invalid.example/schema'})

def test_regex() -> None:
    verifier = RegexVerifier(r'[0-9]{2}')
    assert verifier.verify('42').passed
    assert not verifier.verify('answer:42').passed
    assert not verifier.verify('x'*32769).passed

def test_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    class Timeout:
        def fullmatch(self, text: str, timeout: float) -> None:
            raise TimeoutError('injected')
    verifier = RegexVerifier('a+')
    monkeypatch.setattr(verifier, 'pattern', Timeout())
    assert not verifier.verify('a').passed

def test_retry_and_dedup(tmp_path: Path) -> None:
    backend = Script([output('41'), output('42'), output('42')])
    path = tmp_path/'data.jsonl'
    engine = PrismZeroEngine(backend, [Rule(MathEqualityVerifier('42'))], path)
    result = engine.run('6*7?')
    assert result.accepted and result.attempts == 2
    assert backend.feedback[0] is None and backend.feedback[1]
    engine.run('6*7?')
    records = read_dataset(path)
    assert len(records) == 1 and records[0]['answer'] == '42'
    assert records[0]['schema_version'] == 2
    assert records[0]['checks'][0]['target'] == 'answer'
    assert 'created_at' in records[0]

def test_exhaustion(tmp_path: Path) -> None:
    path = tmp_path/'data.jsonl'
    result = PrismZeroEngine(Script(['bad',output('0')]), [Rule(MathEqualityVerifier('42'))],
                             path, max_attempts=2).run('6*7?')
    assert not result.accepted and not path.exists()

def test_corruption_preserved(tmp_path: Path) -> None:
    path = tmp_path/'data.jsonl'
    path.write_text('corrupt\n')
    with pytest.raises(DatasetError):
        PrismZeroEngine(Script([output('42')]), [Rule(MathEqualityVerifier('42'))], path).run('6*7?')
    assert path.read_text() == 'corrupt\n'

def test_trace_rule(tmp_path: Path) -> None:
    rules = [Rule(MathEqualityVerifier('42')), Rule(RegexVerifier('required'), 'trace')]
    assert not PrismZeroEngine(Script([output('42')]), rules,
                              tmp_path/'data.jsonl', max_attempts=1).run('6*7?').accepted

def test_bad_envelope() -> None:
    good = T_OPEN + 'x' + T_CLOSE + '\n' + A_OPEN + '42' + A_CLOSE
    bad = ['plain',
           T_CLOSE + A_OPEN + '42' + A_CLOSE,
           T_OPEN + 'x' + T_CLOSE + A_OPEN + '42' + A_CLOSE + 'tail',
           T_OPEN + T_OPEN + 'x' + T_CLOSE + T_CLOSE + A_OPEN + '42' + A_CLOSE,
           T_OPEN + 'x' + T_CLOSE + '\n' + A_OPEN + A_OPEN + '42' + A_CLOSE + A_CLOSE,
           T_OPEN + 'x' + T_CLOSE + '\n' + A_OPEN + '42' + A_CLOSE + A_CLOSE,
           T_OPEN + '' + T_CLOSE + '\n' + A_OPEN + '42' + A_CLOSE]
    for text in bad:
        with pytest.raises(ValueError):
            Candidate.parse(text)
    assert Candidate.parse(good).answer == '42'

def test_oversized_output(tmp_path: Path) -> None:
    huge = T_OPEN + 'x'*40000 + T_CLOSE + '\n' + A_OPEN + '42' + A_CLOSE
    result = PrismZeroEngine(Script([huge]), [Rule(MathEqualityVerifier('42'))],
                             tmp_path/'d.jsonl', max_attempts=1).run('task')
    assert not result.accepted and 'envelope' == result.checks[0].verifier

def test_backend_error(tmp_path: Path) -> None:
    class Broken:
        calls = 0
        def generate(self, prompt: str, feedback: str | None) -> str:
            self.calls += 1
            raise RuntimeError('unavailable')
    backend = Broken()
    with pytest.raises(RuntimeError):
        PrismZeroEngine(backend, [Rule(MathEqualityVerifier('42'))], tmp_path/'d.jsonl').run('task')
    assert backend.calls == 1

def test_required_rule() -> None:
    with pytest.raises(ValueError):
        PrismZeroEngine(Script([]), [])

@pytest.mark.parametrize('max_attempts', [1, 20])
def test_max_attempts_bounds(max_attempts: int, tmp_path: Path) -> None:
    engine = PrismZeroEngine(Script([output('42')]), [Rule(MathEqualityVerifier('42'))],
                             tmp_path/'d.jsonl', max_attempts=max_attempts)
    assert engine.max_attempts == max_attempts

@pytest.mark.parametrize('bad', [0, -1, 21, 100, 3.5, True, False, '3', None, 3.0])
def test_max_attempts_invalid(bad: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='max_attempts'):
        PrismZeroEngine(Script([]), [Rule(MathEqualityVerifier('42'))],
                        tmp_path/'d.jsonl', max_attempts=bad)

def test_storage_failure_after_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path/'d.jsonl'
    def broken_add(record: dict) -> None:
        raise OSError('disk unavailable')
    engine = PrismZeroEngine(Script([output('42')]), [Rule(MathEqualityVerifier('42'))], path)
    monkeypatch.setattr(engine.store, 'add', broken_add)
    with pytest.raises(OSError):
        engine.run('6*7?')
    assert not path.exists()

def test_no_prompt_leak_in_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging
    secret = 'PRIVATEPROMPTTOKEN'
    with caplog.at_level(logging.INFO):
        PrismZeroEngine(Script([output('41'), output('42')]),
                        [Rule(MathEqualityVerifier('42'))], tmp_path/'d.jsonl').run(secret)
    assert secret not in caplog.text

def test_labels() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: list[dict[str,str]], tokenize: bool,
                                add_generation_prompt: bool) -> list[int]:
            return [1,2] if add_generation_prompt else [1,2,3,4]
    row = {'prompt':'task','trace':'explanation','answer':'42'}
    assert encode_record(Tokenizer(), row, 4)['labels'] == [-100,-100,3,4]
    with pytest.raises(ValueError):
        encode_record(Tokenizer(), row, 3)

def test_answer_only_content_mode() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: list[dict[str,str]], tokenize: bool,
                                add_generation_prompt: bool) -> list[int]:
            if add_generation_prompt:
                return [1,2]
            return [1,2,5,6]
    row = {'prompt':'task','trace':'explanation','answer':'42'}
    labels = encode_record(Tokenizer(), row, 8, content_mode='answer-only')['labels']
    assert labels == [-100,-100,5,6]
