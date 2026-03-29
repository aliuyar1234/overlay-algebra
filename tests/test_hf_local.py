from __future__ import annotations

from pathlib import Path

from overlay_algebra.hf_local import resolve_pretrained_source


def test_resolve_pretrained_source_returns_existing_local_path(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    resolved = resolve_pretrained_source(model_dir, local_files_only=True)

    assert resolved == str(model_dir.resolve())


def test_resolve_pretrained_source_uses_cached_snapshot_in_local_only_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    calls: list[tuple[str, bool]] = []

    class _DummyHub:
        @staticmethod
        def snapshot_download(*, repo_id: str, local_files_only: bool) -> str:
            calls.append((repo_id, local_files_only))
            return str(snapshot_dir)

    import overlay_algebra.hf_local as hf_local

    real_import_module = hf_local.importlib.import_module

    def _fake_import_module(name: str):
        if name == "huggingface_hub":
            return _DummyHub()
        return real_import_module(name)

    monkeypatch.setattr(hf_local.importlib, "import_module", _fake_import_module)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    resolved = resolve_pretrained_source("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)

    assert resolved == str(snapshot_dir.resolve())
    assert calls == [("Qwen/Qwen2.5-7B-Instruct", True)]
    assert hf_local.os.environ["HF_HUB_OFFLINE"] == "1"
    assert hf_local.os.environ["TRANSFORMERS_OFFLINE"] == "1"
