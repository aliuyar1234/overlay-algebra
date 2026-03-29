from __future__ import annotations

import json
from typing import Any, Mapping


CONDITIONS = ("plain", "J", "C", "Q", "JC", "JQ", "CQ", "JCQ")


def canonical_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def quote_literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def sid_from_index(support_idx: int) -> str:
    if support_idx <= 0:
        raise ValueError("support_idx must be 1-based and positive")
    return f"S{support_idx}"


def compile_targets(answer_text: str, support_idx: int, support_sentence: str) -> dict[str, str]:
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
