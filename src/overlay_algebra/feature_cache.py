from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence


_CACHE_FORMAT_VERSION = 1
_FEATURE_BUILDER_VERSION = 2


@dataclass(frozen=True, slots=True)
class FeatureCacheDescriptor:
    cache_format_version: int
    feature_builder_version: int
    dataset_path: str
    dataset_split: str
    dataset_version: str
    dataset_sha256: str
    manifest_sha256: str | None
    feature_kind: str
    base_model_name_or_path: str
    tokenizer_name_or_path: str
    condition: str
    prompt_protocol_sha256: str
    max_length: int
    max_examples: int

    def cache_key(self) -> str:
        payload = json.dumps(asdict(self), ensure_ascii=True, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_feature_cache_dir(dataset_path: Path) -> Path:
    return dataset_path.parent / "_feature_cache"


def build_feature_cache_descriptor(
    *,
    dataset_path: Path,
    feature_kind: str,
    base_model_name_or_path: str,
    tokenizer_name_or_path: str,
    condition: str,
    prompt_protocol_sha256: str,
    max_length: int,
    max_examples: int,
) -> FeatureCacheDescriptor:
    dataset_identity = _resolve_dataset_identity(dataset_path)
    return FeatureCacheDescriptor(
        cache_format_version=_CACHE_FORMAT_VERSION,
        feature_builder_version=_FEATURE_BUILDER_VERSION,
        dataset_path=str(dataset_path.resolve()),
        dataset_split=dataset_identity["dataset_split"],
        dataset_version=dataset_identity["dataset_version"],
        dataset_sha256=dataset_identity["dataset_sha256"],
        manifest_sha256=dataset_identity["manifest_sha256"],
        feature_kind=feature_kind,
        base_model_name_or_path=base_model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        condition=condition,
        prompt_protocol_sha256=prompt_protocol_sha256,
        max_length=max_length,
        max_examples=max_examples,
    )


def feature_cache_path(cache_dir: Path, descriptor: FeatureCacheDescriptor) -> Path:
    return cache_dir / f"{descriptor.cache_key()}.pkl"


def load_feature_cache_bundle(
    path: Path,
    *,
    descriptor: FeatureCacheDescriptor,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        payload = pickle.loads(path.read_bytes())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    cached_descriptor = payload.get("descriptor")
    if cached_descriptor != descriptor.as_dict():
        return None
    cached_features = payload.get("features")
    if not isinstance(cached_features, list):
        return None
    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        return None
    return {
        "features": [dict(feature) for feature in cached_features],
        "metadata": dict(metadata),
    }


def load_feature_cache(
    path: Path,
    *,
    descriptor: FeatureCacheDescriptor,
) -> list[dict[str, Any]] | None:
    payload = load_feature_cache_bundle(path, descriptor=descriptor)
    if payload is None:
        return None
    return payload["features"]


def save_feature_cache_bundle(
    path: Path,
    *,
    descriptor: FeatureCacheDescriptor,
    features: Sequence[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "descriptor": descriptor.as_dict(),
        "features": [dict(feature) for feature in features],
        "metadata": dict(metadata or {}),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary_path.replace(path)


def save_feature_cache(
    path: Path,
    *,
    descriptor: FeatureCacheDescriptor,
    features: Sequence[dict[str, Any]],
) -> None:
    save_feature_cache_bundle(path, descriptor=descriptor, features=features, metadata=None)


def _resolve_dataset_identity(dataset_path: Path) -> dict[str, str | None]:
    manifest_path = dataset_path.parent / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_key = _manifest_hash_key(dataset_path)
        hashes = manifest.get("hashes", {})
        dataset_hash = hashes.get(split_key)
        if isinstance(dataset_hash, str) and dataset_hash:
            return {
                "dataset_split": dataset_path.stem,
                "dataset_version": str(manifest.get("dataset_version", dataset_path.parent.name)),
                "dataset_sha256": dataset_hash,
                "manifest_sha256": str(manifest.get("manifest_sha256")) if manifest.get("manifest_sha256") else None,
            }

    return {
        "dataset_split": dataset_path.stem,
        "dataset_version": dataset_path.parent.name,
        "dataset_sha256": _sha256_file(dataset_path),
        "manifest_sha256": None,
    }


def _manifest_hash_key(dataset_path: Path) -> str:
    stem = dataset_path.stem.lower()
    if stem in {"train", "val", "test"}:
        return f"{stem}_jsonl"
    return f"{stem}_sha256"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
