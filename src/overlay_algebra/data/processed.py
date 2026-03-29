from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessedExample:
    example_id: str
    title: str
    question: str
    context: str
    context_sentences: list[str]
    labeled_context: str
    answer_text: str
    answer_start: int
    support_sent_idx: int
    support_sentence: str
    targets: dict[str, str]

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "ProcessedExample":
        return cls(
            example_id=str(payload["example_id"]),
            title=str(payload["title"]),
            question=str(payload["question"]),
            context=str(payload["context"]),
            context_sentences=[str(item) for item in payload["context_sentences"]],
            labeled_context=str(payload["labeled_context"]),
            answer_text=str(payload["answer_text"]),
            answer_start=int(payload["answer_start"]),
            support_sent_idx=int(payload["support_sent_idx"]),
            support_sentence=str(payload["support_sentence"]),
            targets={str(key): str(value) for key, value in dict(payload["targets"]).items()},
        )


def load_processed_examples(path: str | Path, *, limit: int | None = None) -> list[ProcessedExample]:
    source_path = Path(path)
    records: list[ProcessedExample] = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(ProcessedExample.from_dict(json.loads(line)))
        if limit is not None and len(records) >= limit:
            break
    return records
