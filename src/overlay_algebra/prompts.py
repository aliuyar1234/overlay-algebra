from __future__ import annotations

import hashlib
import json


FORMAT_SPECS = {
    "plain": "plain answer text only",
    "J": 'valid JSON object with exactly one key "answer"',
    "C": "two lines: ANSWER: <text> and SUPPORT: [S#]",
    "Q": "two lines: ANSWER: <text> and QUOTE: <JSON string literal of exact support sentence text>",
    "JC": 'valid JSON object with keys "answer" and "support", where support is a one-item list like ["S2"]',
    "JQ": 'valid JSON object with keys "answer" and "quote"',
    "CQ": "three lines: ANSWER: <text>, SUPPORT: [S#], QUOTE: <JSON string literal of exact support sentence text>",
    "JCQ": 'valid JSON object with keys "answer", "support", and "quote"',
}

PROMPT_TEMPLATE = (
    "Use only the provided context. Context sentences are labeled [S1], [S2], ...\n"
    "Do not use outside knowledge.\n"
    "Required output format: {format_spec}\n\n"
    "Question:\n"
    "{question}\n\n"
    "Context:\n"
    "{labeled_context}\n\n"
    "Response:\n"
)


def prompt_protocol_fingerprint(condition: str) -> str:
    if condition not in FORMAT_SPECS:
        raise ValueError(f"Unsupported condition: {condition}")
    payload = {
        "condition": condition,
        "format_spec": FORMAT_SPECS[condition],
        "template": PROMPT_TEMPLATE,
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_prompt(question: str, labeled_context: str, condition: str) -> str:
    if condition not in FORMAT_SPECS:
        raise ValueError(f"Unsupported condition: {condition}")

    format_spec = FORMAT_SPECS[condition]
    return PROMPT_TEMPLATE.format(
        format_spec=format_spec,
        question=question,
        labeled_context=labeled_context,
    )
