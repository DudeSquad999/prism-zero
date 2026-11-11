"""Bounded checks on generated text, never generated-code execution."""
from __future__ import annotations
import ast
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Protocol
import regex
from jsonschema import Draft202012Validator

@dataclass(frozen=True)
class Check:
    verifier: str
    passed: bool
    message: str

class Verifier(Protocol):
    name: str
    def verify(self, text: str) -> Check:
        """Return a constraint result without executing candidate code."""
        raise NotImplementedError

def bounded(text: str, limit: int = 32768) -> None:
    if not isinstance(text, str) or len(text) > limit:
        raise ValueError("Text exceeds the configured character limit")

def strict_json(text: str) -> Any:
    """Reject duplicate keys, nonfinite constants and numeric overflow."""
    bounded(text, 1048576)
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid(value: str) -> Any:
        raise ValueError("Nonfinite JSON constant")
    def finite(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON numeric overflow")
        return number
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)

class ASTVerifier:
    """Check Python syntax only; parsing does not establish runtime safety."""
    name = "python_ast"
    def verify(self, text: str) -> Check:
        try:
            bounded(text)
            ast.parse(text)
            return Check(self.name, True, "Valid Python syntax; not executed")
        except (SyntaxError, ValueError, RecursionError):
            return Check(self.name, False, "Invalid Python syntax or input limit")

class JSONSchemaVerifier:
    """Trusted Draft 2020-12 schemas; remote/local references are disabled."""
    name = "json_schema"
    def __init__(self, schema: dict[str, Any]) -> None:
        copied = json.loads(json.dumps(schema, allow_nan=False))
        def inspect(value: Any) -> None:
            if isinstance(value, dict):
                if '$ref' in value or '$dynamicRef' in value:
                    raise ValueError("Schema references are disabled")
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
        inspect(copied)
        Draft202012Validator.check_schema(copied)
        self.validator = Draft202012Validator(copied)
    def verify(self, text: str) -> Check:
        try:
            bounded(text)
            error = next(self.validator.iter_errors(strict_json(text)), None)
            return Check(self.name, error is None, "Schema satisfied" if error is None
                         else f"Schema constraint failed: {error.validator}")
        except (ValueError, RecursionError):
            return Check(self.name, False, "Invalid or excessively nested JSON")

class RegexVerifier:
    """Full-match a trusted regex with a bounded input and wall-clock timeout."""
    name = "regex"
    def __init__(self, pattern: str, timeout: float = 0.05) -> None:
        if not 0 < timeout <= 1:
            raise ValueError("Timeout must be in (0,1]")
        self.pattern = regex.compile(pattern)
        self.timeout = timeout
    def verify(self, text: str) -> Check:
        try:
            bounded(text)
            passed = self.pattern.fullmatch(text, timeout=self.timeout) is not None
            return Check(self.name, passed, "Pattern satisfied" if passed else "Pattern mismatch")
        except (ValueError, TimeoutError):
            return Check(self.name, False, "Regex input limit or timeout")

def evaluate(expression: str) -> Fraction:
    """Evaluate bounded rational arithmetic through an AST allowlist, not eval."""
    bounded(expression, 512)
    tree = ast.parse(expression, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Expression too complex")
    def visit(node: ast.AST, depth: int = 0) -> Fraction:
        if depth > 32:
            raise ValueError("Expression too deep")
        if isinstance(node, ast.Constant) and type(node.value) is int:
            value = Fraction(node.value)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = visit(node.operand, depth + 1)
            value = operand if isinstance(node.op, ast.UAdd) else -operand
        elif isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            else:
                raise ValueError("Unsupported operator")
        else:
            raise ValueError("Unsupported expression")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
            raise ValueError("Arithmetic magnitude exceeded")
        return value
    return visit(tree.body)

class MathEqualityVerifier:
    """Check against a trusted independent target, not a model-supplied tautology."""
    name = 'math_equality'
    def __init__(self, expected: str) -> None:
        self.expected = evaluate(expected)
    def verify(self, text: str) -> Check:
        try:
            passed = evaluate(text.strip()) == self.expected
            return Check(self.name, passed, 'Exact equality satisfied' if passed else 'Incorrect result')
        except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
            return Check(self.name, False, 'Invalid bounded rational expression')
