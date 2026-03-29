from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    text: str
    generated_token_count: int
    prompt_token_length: int
    stopped_by_eos: bool
    hit_max_new_tokens: bool


def _batch_indices(total: int, batch_size: int) -> list[list[int]]:
    return [list(range(start, min(start + batch_size, total))) for start in range(0, total, batch_size)]


def _sorted_indices(
    *,
    total: int,
    sort_by_length: bool,
    prompt_length_hints: Sequence[int] | None,
    generation_length_hints: Sequence[int] | None,
) -> list[int]:
    indices = list(range(total))
    if not sort_by_length:
        return indices
    if prompt_length_hints is None:
        prompt_length_hints = [0] * total
    if generation_length_hints is None:
        generation_length_hints = [0] * total
    return sorted(
        indices,
        key=lambda index: (
            int(prompt_length_hints[index]),
            int(generation_length_hints[index]),
            index,
        ),
    )


def _trim_generated_tokens(
    generated_tokens: Sequence[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_new_tokens: int,
) -> tuple[list[int], bool, bool]:
    trimmed: list[int] = []
    stopped_by_eos = False
    for token in generated_tokens:
        token_int = int(token)
        if pad_token_id is not None and token_int == int(pad_token_id) and not trimmed:
            continue
        trimmed.append(token_int)
        if eos_token_id is not None and token_int == int(eos_token_id):
            stopped_by_eos = True
            break
    generated_token_count = len(trimmed)
    hit_max = (generated_token_count >= max_new_tokens) and not stopped_by_eos
    return trimmed, stopped_by_eos, hit_max


def batched_greedy_generate_records(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    sort_by_length: bool = False,
    prompt_length_hints: Sequence[int] | None = None,
    generation_length_hints: Sequence[int] | None = None,
) -> list[GenerationRecord]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not prompts:
        return []

    ordered_indices = _sorted_indices(
        total=len(prompts),
        sort_by_length=sort_by_length,
        prompt_length_hints=prompt_length_hints,
        generation_length_hints=generation_length_hints,
    )
    results: list[GenerationRecord | None] = [None] * len(prompts)
    original_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        for batch_indices in _batch_indices(len(ordered_indices), batch_size):
            item_indices = [ordered_indices[index] for index in batch_indices]
            prompt_batch = [str(prompts[index]) for index in item_indices]
            tokenized = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            attention_rows = tokenized["attention_mask"]
            prompt_lengths = [int(sum(int(token) for token in row)) for row in attention_rows]
            prompt_width = int(tokenized["input_ids"].shape[1])
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            outputs = model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_batch = outputs[:, prompt_width:]
            for item_index, prompt_length, generated_tokens in zip(
                item_indices,
                prompt_lengths,
                generated_batch,
                strict=True,
            ):
                trimmed_tokens, stopped_by_eos, hit_max_new_tokens = _trim_generated_tokens(
                    generated_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_new_tokens,
                )
                results[item_index] = GenerationRecord(
                    text=tokenizer.decode(trimmed_tokens, skip_special_tokens=True),
                    generated_token_count=len(trimmed_tokens),
                    prompt_token_length=prompt_length,
                    stopped_by_eos=stopped_by_eos,
                    hit_max_new_tokens=hit_max_new_tokens,
                )
    finally:
        tokenizer.padding_side = original_padding_side

    return [record for record in results if record is not None]


def batched_greedy_generate_texts(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    prompts: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    sort_by_length: bool = False,
    prompt_length_hints: Sequence[int] | None = None,
    generation_length_hints: Sequence[int] | None = None,
) -> list[str]:
    records = batched_greedy_generate_records(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        sort_by_length=sort_by_length,
        prompt_length_hints=prompt_length_hints,
        generation_length_hints=generation_length_hints,
    )
    return [record.text for record in records]
