import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pytest

from compiler_reference import (
    ParseError,
    canonical_json,
    compile_targets,
    parse_condition,
    quote_literal,
)


def test_canonical_json_compact_and_ordered():
    text = canonical_json({"answer": "24 months", "support": ["S2"]})
    assert text == '{"answer":"24 months","support":["S2"]}'


def test_compile_targets_exact_example():
    targets = compile_targets("24 months", 2, "The warranty lasts 24 months.")
    assert targets["JQ"] == '{"answer":"24 months","quote":"The warranty lasts 24 months."}'
    assert targets["Q"].splitlines()[1] == 'QUOTE: "The warranty lasts 24 months."'


def test_embedded_quote_round_trip():
    sentence = 'He said "go".'
    targets = compile_targets("go", 3, sentence)
    assert targets["Q"].splitlines()[1] == 'QUOTE: "He said \\"go\\"."'
    parsed_q = parse_condition(targets["Q"], "Q")
    parsed_jq = parse_condition(targets["JQ"], "JQ")
    assert parsed_q["quote"] == sentence
    assert parsed_jq["quote"] == sentence


def test_non_json_quote_must_be_json_string_literal():
    bad = 'ANSWER: go\nQUOTE: He said "go".'
    with pytest.raises(ParseError):
        parse_condition(bad, "Q")


def test_json_parser_rejects_extra_keys():
    bad = '{"answer":"24 months","support":["S2"],"extra":1}'
    with pytest.raises(ParseError):
        parse_condition(bad, "JC")


def test_json_parser_accepts_reordered_keys():
    text = '{"support":["S2"],"answer":"24 months"}'
    parsed = parse_condition(text, "JC")
    assert parsed["answer"] == "24 months"
    assert parsed["support"] == "S2"
