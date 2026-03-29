from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .soar import DenseDeltaFactors, LoRAFactors


@dataclass(frozen=True, slots=True)
class AdapterBundle:
    adapter_dir: Path
    base_model_name_or_path: str
    peft_config: dict[str, Any]
    lora_factors: dict[str, LoRAFactors]


def _load_safetensors_state(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path))


def _save_safetensors_state(path: Path, state_dict: Mapping[str, Any]) -> None:
    from safetensors.torch import save_file

    save_file(dict(state_dict), str(path))


def _config_value_for_module(
    module_name: str,
    patterns: Mapping[str, Any] | None,
    default_value: float,
) -> float:
    if not patterns:
        return float(default_value)
    if module_name in patterns:
        return float(patterns[module_name])
    for key, value in patterns.items():
        if module_name.endswith(str(key)):
            return float(value)
    return float(default_value)


def load_adapter_bundle(adapter_dir: str | Path) -> AdapterBundle:
    adapter_path = Path(adapter_dir).resolve()
    config_path = adapter_path / "adapter_config.json"
    model_path = adapter_path / "adapter_model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {model_path}")

    peft_config = json.loads(config_path.read_text(encoding="utf-8"))
    state_dict = _load_safetensors_state(model_path)

    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_A.weight"):
            module_name = key[: -len(".lora_A.weight")]
            grouped.setdefault(module_name, {})["A"] = tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        elif key.endswith(".lora_B.weight"):
            module_name = key[: -len(".lora_B.weight")]
            grouped.setdefault(module_name, {})["B"] = tensor.detach().cpu().numpy().astype(np.float32, copy=True)

    lora_factors: dict[str, LoRAFactors] = {}
    default_alpha = float(peft_config["lora_alpha"])
    alpha_pattern = peft_config.get("alpha_pattern")
    for module_name, parts in grouped.items():
        if set(parts.keys()) != {"A", "B"}:
            raise ValueError(f"Adapter module {module_name!r} is missing A or B factors.")
        alpha = _config_value_for_module(module_name, alpha_pattern, default_alpha)
        lora_factors[module_name] = LoRAFactors(A=parts["A"], B=parts["B"], alpha=alpha)

    return AdapterBundle(
        adapter_dir=adapter_path,
        base_model_name_or_path=str(peft_config["base_model_name_or_path"]),
        peft_config=peft_config,
        lora_factors=lora_factors,
    )


def save_compact_delta_artifact(
    output_dir: str | Path,
    factors_map: Mapping[str, DenseDeltaFactors],
    *,
    source: Mapping[str, Any],
) -> Path:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    arrays_path = artifact_dir / "factors.npz"
    manifest_path = artifact_dir / "manifest.json"

    arrays: dict[str, np.ndarray] = {}
    module_index: list[dict[str, Any]] = []
    for index, module_name in enumerate(sorted(factors_map.keys())):
        left_key = f"left_{index}"
        right_key = f"right_{index}"
        factors = factors_map[module_name]
        arrays[left_key] = factors.left.astype(np.float32, copy=False)
        arrays[right_key] = factors.right.astype(np.float32, copy=False)
        module_index.append(
            {
                "module_name": module_name,
                "left_key": left_key,
                "right_key": right_key,
                "left_shape": list(factors.left.shape),
                "right_shape": list(factors.right.shape),
            }
        )

    np.savez_compressed(arrays_path, **arrays)
    manifest = {
        "format_version": 1,
        "representation": "low_rank_dense_product",
        "arrays_path": arrays_path.name,
        "module_index": module_index,
        "source": dict(source),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def load_compact_delta_artifact(output_dir: str | Path) -> tuple[dict[str, DenseDeltaFactors], dict[str, Any]]:
    artifact_dir = Path(output_dir)
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing delta manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays_path = artifact_dir / manifest["arrays_path"]
    arrays = np.load(arrays_path)

    factors_map: dict[str, DenseDeltaFactors] = {}
    for entry in manifest["module_index"]:
        factors_map[entry["module_name"]] = DenseDeltaFactors(
            left=np.array(arrays[entry["left_key"]], copy=True),
            right=np.array(arrays[entry["right_key"]], copy=True),
        )
    return factors_map, manifest


def save_adapter_from_lora_factors(
    output_dir: str | Path,
    *,
    template_bundle: AdapterBundle,
    lora_factors: Mapping[str, LoRAFactors],
    global_rank: int,
    lora_dropout: float = 0.0,
) -> Path:
    adapter_path = Path(output_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    peft_config = dict(template_bundle.peft_config)
    peft_config["inference_mode"] = True
    peft_config["rank_pattern"] = {}
    peft_config["alpha_pattern"] = {}
    peft_config["r"] = int(global_rank)
    peft_config["lora_alpha"] = int(global_rank)
    peft_config["lora_dropout"] = float(lora_dropout)

    state_dict: dict[str, Any] = {}
    for module_name in sorted(lora_factors.keys()):
        factors = lora_factors[module_name]
        state_dict[f"{module_name}.lora_A.weight"] = _torch_tensor_from_numpy(factors.A.astype(np.float32, copy=False))
        state_dict[f"{module_name}.lora_B.weight"] = _torch_tensor_from_numpy(factors.B.astype(np.float32, copy=False))

    (adapter_path / "adapter_config.json").write_text(
        json.dumps(peft_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _save_safetensors_state(adapter_path / "adapter_model.safetensors", state_dict)
    return adapter_path


def _torch_tensor_from_numpy(array: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(array)
