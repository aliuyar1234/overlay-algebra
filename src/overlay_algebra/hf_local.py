from __future__ import annotations

import importlib
import os
from pathlib import Path


def prepare_local_only_hf_environment(*, local_files_only: bool) -> None:
    if not local_files_only:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_pretrained_source(
    name_or_path: str | Path,
    *,
    local_files_only: bool,
) -> str:
    candidate = Path(name_or_path).expanduser()
    if candidate.exists():
        return str(candidate.resolve())

    text = str(name_or_path)
    if not local_files_only:
        return text

    prepare_local_only_hf_environment(local_files_only=True)

    try:
        huggingface_hub = importlib.import_module("huggingface_hub")
    except Exception as exc:  # pragma: no cover - transformers should already depend on this.
        raise RuntimeError(
            "huggingface_hub is required to resolve cached local model snapshots in local-files-only mode."
        ) from exc

    try:
        snapshot_path = huggingface_hub.snapshot_download(repo_id=text, local_files_only=True)
    except Exception as exc:
        raise FileNotFoundError(
            "Could not resolve a cached local snapshot for "
            f"{text!r} in local-files-only mode. Ensure the model is already downloaded."
        ) from exc
    return str(Path(snapshot_path).resolve())
