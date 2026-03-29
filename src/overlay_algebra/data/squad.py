from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RawQaExample:
    example_id: str
    title: str
    question: str
    context: str
    answer_text: str
    answer_start: int
    source_split: str


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_source_identity(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def load_squad_records(path: Path, *, source_split: str) -> list[RawQaExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data", [])
    records: list[RawQaExample] = []

    for article in data:
        title = str(article["title"])
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph["context"])
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible"):
                    continue

                answers = qa.get("answers", [])
                if not answers:
                    continue

                answer = answers[0]
                records.append(
                    RawQaExample(
                        example_id=str(qa["id"]),
                        title=title,
                        question=str(qa["question"]),
                        context=context,
                        answer_text=str(answer["text"]),
                        answer_start=int(answer["answer_start"]),
                        source_split=source_split,
                    )
                )

    return records
