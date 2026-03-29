from __future__ import annotations

from dataclasses import dataclass
import importlib


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    index: int
    text: str
    start_char: int
    end_char: int


class SentenceSplitter:
    def __init__(self, *, expected_spacy_version: str | None = None) -> None:
        try:
            spacy = importlib.import_module("spacy")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "spaCy is required for the locked sentence splitter. Install the project environment first."
            ) from exc

        self._spacy = spacy
        self.version = str(spacy.__version__)
        if expected_spacy_version and self.version != expected_spacy_version:
            raise RuntimeError(
                "spaCy version mismatch for the locked sentence splitter: "
                f"expected {expected_spacy_version}, found {self.version}."
            )

        self._nlp = spacy.blank("en")
        if "sentencizer" not in self._nlp.pipe_names:
            self._nlp.add_pipe("sentencizer")

    @property
    def identity(self) -> str:
        return f"spacy.blank('en')+sentencizer@{self.version}"

    def split(self, text: str) -> list[SentenceSpan]:
        doc = self._nlp(text)
        sentences: list[SentenceSpan] = []
        next_index = 1
        for sent in doc.sents:
            raw_text = sent.text
            trimmed = raw_text.strip()
            if not trimmed:
                continue

            left_trim = len(raw_text) - len(raw_text.lstrip())
            right_trim = len(raw_text) - len(raw_text.rstrip())
            start_char = sent.start_char + left_trim
            end_char = sent.end_char - right_trim
            sentences.append(
                SentenceSpan(
                    index=next_index,
                    text=text[start_char:end_char],
                    start_char=start_char,
                    end_char=end_char,
                )
            )
            next_index += 1

        return sentences
