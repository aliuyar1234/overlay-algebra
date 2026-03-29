from __future__ import annotations

import json
import re
from typing import Any, Mapping


SID_RE = re.compile(r"^S[1-9][0-9]*$")
SUPPORT_LINE_RE = re.compile(r"^SUPPORT: \[(S[1-9][0-9]*)\]$")


class ParseError(ValueError):
    pass


def _parse_answer_line(line: str) -> str:
    prefix = "ANSWER: "
    if not line.startswith(prefix):
        raise ParseError("missing ANSWER: prefix")
    return line[len(prefix) :]


def _parse_support_line(line: str) -> str:
    match = SUPPORT_LINE_RE.match(line)
    if not match:
        raise ParseError("support line must match SUPPORT: [S#] exactly")
    return match.group(1)


def _parse_quote_line(line: str) -> str:
    prefix = "QUOTE: "
    if not line.startswith(prefix):
        raise ParseError("missing QUOTE: prefix")
    payload = line[len(prefix) :]
    try:
        value = json.loads(payload)
    except Exception as exc:  # pragma: no cover - json error message is not stable
        raise ParseError("quote payload is not a valid JSON string literal") from exc
    if not isinstance(value, str):
        raise ParseError("quote payload must decode to a string")
    return value


def parse_c(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if len(lines) != 2:
        raise ParseError("C format must have exactly 2 lines")
    return {
        "answer": _parse_answer_line(lines[0]),
        "support": _parse_support_line(lines[1]),
    }


def parse_q(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if len(lines) != 2:
        raise ParseError("Q format must have exactly 2 lines")
    return {
        "answer": _parse_answer_line(lines[0]),
        "quote": _parse_quote_line(lines[1]),
    }


def parse_cq(text: str) -> dict[str, str]:
    lines = text.splitlines()
    if len(lines) != 3:
        raise ParseError("CQ format must have exactly 3 lines")
    return {
        "answer": _parse_answer_line(lines[0]),
        "support": _parse_support_line(lines[1]),
        "quote": _parse_quote_line(lines[2]),
    }


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


def parse_json_condition(text: str, condition: str) -> dict[str, str]:
    try:
        obj = json.loads(text)
    except Exception as exc:  # pragma: no cover - json error message is not stable
        raise ParseError("invalid JSON") from exc

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
        return {"answer": obj["answer"], "support": _require_sid_list(obj["support"])}

    if condition == "JQ":
        _require_exact_key_set(obj, ("answer", "quote"))
        if not isinstance(obj["answer"], str) or not isinstance(obj["quote"], str):
            raise ParseError("answer and quote must be strings")
        return {"answer": obj["answer"], "quote": obj["quote"]}

    if condition == "JCQ":
        _require_exact_key_set(obj, ("answer", "support", "quote"))
        if not isinstance(obj["answer"], str) or not isinstance(obj["quote"], str):
            raise ParseError("answer and quote must be strings")
        return {
            "answer": obj["answer"],
            "support": _require_sid_list(obj["support"]),
            "quote": obj["quote"],
        }

    raise ValueError(f"unsupported JSON condition: {condition}")


def parse_condition(text: str, condition: str) -> dict[str, str]:
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
