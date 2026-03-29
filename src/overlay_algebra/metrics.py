from __future__ import annotations

from collections import Counter
import json
import re
import string
import unicodedata
from typing import Any

from .compiler import sid_from_index
from .parsers import ParseError, parse_condition


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_JSON_CONDITIONS = {"J", "JC", "JQ", "JCQ"}
_CITATION_CONDITIONS = {"C", "CQ", "JC", "JCQ"}
_QUOTE_CONDITIONS = {"Q", "CQ", "JQ", "JCQ"}


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    without_punct = lowered.translate(_PUNCT_TABLE)
    without_articles = _ARTICLE_RE.sub(" ", without_punct)
    return " ".join(without_articles.split())


def normalize_quote(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return _WHITESPACE_RE.sub(" ", normalized)


def _token_f1(prediction_tokens: list[str], gold_tokens: list[str]) -> float:
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    return _token_f1(pred_tokens, gold_tokens)


def quote_exact_match(prediction: str, gold: str) -> float:
    return float(normalize_quote(prediction) == normalize_quote(gold))


def quote_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_quote(prediction).split()
    gold_tokens = normalize_quote(gold).split()
    return _token_f1(pred_tokens, gold_tokens)


def _json_valid(text: str) -> float:
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return 0.0
    return 1.0


def score_prediction(
    condition: str,
    prediction_text: str,
    *,
    gold_answer: str,
    gold_support_idx: int | None = None,
    gold_support_sentence: str | None = None,
    allow_answer_salvage: bool = False,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "answer_em": 0.0,
        "answer_f1": 0.0,
        "json_valid": None,
        "json_strict_schema_valid": None,
        "citation_exact": 0.0 if condition in _CITATION_CONDITIONS else None,
        "quote_exact": 0.0 if condition in _QUOTE_CONDITIONS else None,
        "quote_f1": 0.0 if condition in _QUOTE_CONDITIONS else None,
    }

    if condition in _JSON_CONDITIONS:
        result["json_valid"] = _json_valid(prediction_text)
        result["json_strict_schema_valid"] = 0.0

    parsed: dict[str, Any] | None = None
    try:
        parsed = parse_condition(prediction_text, condition)
    except ParseError:
        if condition in _JSON_CONDITIONS and allow_answer_salvage and result["json_valid"] == 1.0:
            try:
                obj = json.loads(prediction_text)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
                parsed = {"answer": obj["answer"]}
        else:
            return result

    if condition in _JSON_CONDITIONS and parsed is not None:
        result["json_strict_schema_valid"] = 1.0

    if parsed is None:
        return result

    if "answer" in parsed:
        result["answer_em"] = answer_exact_match(parsed["answer"], gold_answer)
        result["answer_f1"] = answer_f1(parsed["answer"], gold_answer)

    if "support" in parsed:
        if gold_support_idx is None:
            raise ValueError("gold_support_idx is required for citation scoring")
        result["citation_exact"] = float(parsed["support"] == sid_from_index(gold_support_idx))

    if "quote" in parsed:
        if gold_support_sentence is None:
            raise ValueError("gold_support_sentence is required for quote scoring")
        result["quote_exact"] = quote_exact_match(parsed["quote"], gold_support_sentence)
        result["quote_f1"] = quote_f1(parsed["quote"], gold_support_sentence)

    return result
