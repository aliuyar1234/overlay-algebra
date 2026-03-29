from __future__ import annotations

from pathlib import Path
import sys

import pytest

from overlay_algebra.compiler import canonical_json, compile_targets
from overlay_algebra.parsers import ParseError, parse_condition


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference_impl"
sys.path.append(str(REFERENCE_DIR))

from compiler_reference import compile_targets as reference_compile_targets  # type: ignore  # noqa: E402
from compiler_reference import parse_condition as reference_parse_condition  # type: ignore  # noqa: E402


def test_compiler_matches_reference_example() -> None:
    ours = compile_targets("24 months", 2, "The warranty lasts 24 months.")
    reference = reference_compile_targets("24 months", 2, "The warranty lasts 24 months.")
    assert ours == reference


def test_canonical_json_is_compact() -> None:
    assert canonical_json({"answer": "24 months", "support": ["S2"]}) == '{"answer":"24 months","support":["S2"]}'


def test_embedded_quote_roundtrip_matches_reference() -> None:
    sentence = 'He said "go".'
    ours = compile_targets("go", 3, sentence)
    assert ours["Q"].splitlines()[1] == 'QUOTE: "He said \\"go\\"."'
    assert parse_condition(ours["Q"], "Q") == reference_parse_condition(ours["Q"], "Q")
    assert parse_condition(ours["JQ"], "JQ") == reference_parse_condition(ours["JQ"], "JQ")


def test_json_parser_rejects_extra_keys() -> None:
    with pytest.raises(ParseError):
        parse_condition('{"answer":"24 months","support":["S2"],"extra":1}', "JC")


def test_non_json_quote_requires_json_string_literal() -> None:
    with pytest.raises(ParseError):
        parse_condition('ANSWER: go\nQUOTE: He said "go".', "Q")
