"""Reference implementation for the locked target compiler and strict parsers.

This file is clarity-first and intentionally small.
It is NOT production dataset code.
Its purpose is to anchor the exact compiler / parser rules so later
implementation does not drift on formatting, escaping, or schema checks.

Locked choices implemented here:
- support IDs are 1-based strings like "S2"
- JSON targets use compact canonical serialization with no spaces
- non-JSON quote lines carry a JSON string literal after `QUOTE: `
- strict JSON schema means exact keys, exact types, no extras
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Dict, Mapping, Any


SID_RE = re.compile(r"^S[1-9][0-9]*$")
SUPPORT_LINE_RE = re.compile(r"^SUPPORT: \[(S[1-9][0-9]*)\]$")


class ParseError(ValueError):
    pass


def canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def quote_literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def sid_from_index(support_idx: int) -> str:
    if support_idx <= 0:
        raise ValueError("support_idx must be 1-based and positive")
    return f"S{support_idx}"


def compile_targets(answer_text: str, support_idx: int, support_sentence: str) -> Dict[str, str]:
    sid = sid_from_index(support_idx)
    return {
        "plain": answer_text,
        "C": f"ANSWER: {answer_text}\nSUPPORT: [{sid}]",
        "Q": f"ANSWER: {answer_text}\nQUOTE: {quote_literal(support_sentence)}",
        "CQ": f"ANSWER: {answer_text}\nSUPPORT: [{sid}]\nQUOTE: {quote_literal(support_sentence)}",
        "J": canonical_json({"answer": answer_text}),
        "JC": canonical_json({"answer": answer_text, "support": [sid]}),
        "JQ": canonical_json({"answer": answer_text, "quote": support_sentence}),
        "JCQ": canonical_json({"answer": answer_text, "support": [sid], "quote": support_sentence}),
    }


def _parse_answer_line(line: str) -> str:
    prefix = "ANSWER: "
    if not line.startswith(prefix):
        raise ParseError("missing ANSWER: prefix")
    return line[len(prefix):]


def _parse_support_line(line: str) -> str:
    m = SUPPORT_LINE_RE.match(line)
    if not m:
        raise ParseError("support line must match SUPPORT: [S#] exactly")
    return m.group(1)


def _parse_quote_line(line: str) -> str:
    prefix = "QUOTE: "
    if not line.startswith(prefix):
        raise ParseError("missing QUOTE: prefix")
    payload = line[len(prefix):]
    try:
        value = json.loads(payload)
    except Exception as e:
        raise ParseError("quote payload is not a valid JSON string literal") from e
    if not isinstance(value, str):
        raise ParseError("quote payload must decode to a string")
    return value


def parse_c(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    if len(lines) != 2:
        raise ParseError("C format must have exactly 2 lines")
    answer = _parse_answer_line(lines[0])
    support = _parse_support_line(lines[1])
    return {"answer": answer, "support": support}


def parse_q(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    if len(lines) != 2:
        raise ParseError("Q format must have exactly 2 lines")
    answer = _parse_answer_line(lines[0])
    quote = _parse_quote_line(lines[1])
    return {"answer": answer, "quote": quote}


def parse_cq(text: str) -> Dict[str, str]:
    lines = text.splitlines()
    if len(lines) != 3:
        raise ParseError("CQ format must have exactly 3 lines")
    answer = _parse_answer_line(lines[0])
    support = _parse_support_line(lines[1])
    quote = _parse_quote_line(lines[2])
    return {"answer": answer, "support": support, "quote": quote}


def _require_exact_key_set(obj: Mapping[str, Any], expected_keys: tuple[str, ...]) -> None:
    if set(obj.keys()) != set(expected_keys):
        raise ParseError(f"expected exact key set {set(expected_keys)}, got {set(obj.keys())}")


def _require_sid_list(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ParseError("support must be a list of length 1")
    sid = value[0]
    if not isinstance(sid, str) or not SID_RE.match(sid):
        raise ParseError("support[0] must be a string like S2")
    return sid


def parse_json_condition(text: str, condition: str) -> Dict[str, str]:
    try:
        obj = json.loads(text)
    except Exception as e:
        raise ParseError("invalid JSON") from e
    if not isinstance(obj, dict):
        raise ParseError("JSON condition must decode to an object")

    if condition == "J":
        _require_exact_key_set(obj, ("answer",))
        if not isinstance(obj["answer"], str):
            raise ParseError("answer must be a string")
        return {"answer": obj["answer"]}

    if condition == "JC":
        _require_exact_key_set(obj, ("answer", "support"))
        if not isinstance(obj["answer"], str):
            raise ParseError("answer must be a string")
        support = _require_sid_list(obj["support"])
        return {"answer": obj["answer"], "support": support}

    if condition == "JQ":
        _require_exact_key_set(obj, ("answer", "quote"))
        if not isinstance(obj["answer"], str) or not isinstance(obj["quote"], str):
            raise ParseError("answer and quote must be strings")
        return {"answer": obj["answer"], "quote": obj["quote"]}

    if condition == "JCQ":
        _require_exact_key_set(obj, ("answer", "support", "quote"))
        if not isinstance(obj["answer"], str) or not isinstance(obj["quote"], str):
            raise ParseError("answer and quote must be strings")
        support = _require_sid_list(obj["support"])
        return {"answer": obj["answer"], "support": support, "quote": obj["quote"]}

    raise ValueError(f"unsupported JSON condition: {condition}")


def parse_condition(text: str, condition: str) -> Dict[str, str]:
    if condition == "plain":
        return {"answer": text}
    if condition == "C":
        return parse_c(text)
    if condition == "Q":
        return parse_q(text)
    if condition == "CQ":
        return parse_cq(text)
    if condition in {"J", "JC", "JQ", "JCQ"}:
        return parse_json_condition(text, condition)
    raise ValueError(f"unsupported condition: {condition}")
