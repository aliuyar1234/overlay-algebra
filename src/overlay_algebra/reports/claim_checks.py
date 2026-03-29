from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from ..config import AppConfig, load_app_config


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@dataclass(frozen=True, slots=True)
class ClaimCheckConfig:
    experiment_name: str
    milestone: str
    output_dir: Path
    single_overlay_summary_path: Path
    pair_summary_path: Path | None
    triple_summary_path: Path | None
    split_audit_path: Path | None
    dataset_manifest_path: Path | None
    factor_ablation_path: Path | None
    transfer_summary_path: Path | None
    claims: list[str]
    c1_required_overlays: int
    c1_recovery_ratio_floor: float
    c2_margin_points: float
    semantic_guardrail_points: float
    source_config: Path

    @classmethod
    def from_app_config(cls, config: AppConfig) -> "ClaimCheckConfig":
        claims = config.payload.get("claims", ["C1", "C4", "C5"])
        if not isinstance(claims, list):
            raise TypeError("payload.claims must be a list when provided")
        return cls(
            experiment_name=config.experiment_name,
            milestone=config.milestone,
            output_dir=Path(config.paths["output_dir"]),
            single_overlay_summary_path=Path(config.paths["single_overlay_summary"]),
            pair_summary_path=Path(config.paths["pair_summary"]) if "pair_summary" in config.paths else None,
            triple_summary_path=Path(config.paths["triple_summary"]) if "triple_summary" in config.paths else None,
            split_audit_path=Path(config.paths["split_audit"]) if "split_audit" in config.paths else None,
            dataset_manifest_path=Path(config.paths["dataset_manifest"]) if "dataset_manifest" in config.paths else None,
            factor_ablation_path=Path(config.paths["factor_ablation"]) if "factor_ablation" in config.paths else None,
            transfer_summary_path=Path(config.paths["transfer_summary"]) if "transfer_summary" in config.paths else None,
            claims=[str(claim) for claim in claims],
            c1_required_overlays=int(config.payload.get("c1_required_overlays", 2)),
            c1_recovery_ratio_floor=float(config.payload.get("c1_recovery_ratio_floor", 0.80)),
            c2_margin_points=float(config.payload.get("c2_margin_points", 0.03)),
            semantic_guardrail_points=float(config.payload.get("semantic_guardrail_points", 3.0)),
            source_config=config.source_path,
        )


def _c1_status(config: ClaimCheckConfig, summary: Mapping[str, Any]) -> dict[str, Any]:
    overlays = summary.get("overlays")
    if not isinstance(overlays, Mapping):
        raise TypeError("single-overlay summary is missing an overlays mapping")

    semantic_gap_floor = -(config.semantic_guardrail_points / 100.0)
    passing = []
    overlay_rows: dict[str, Any] = {}
    for overlay_name, raw_row in overlays.items():
        if not isinstance(raw_row, Mapping):
            raise TypeError("overlay summary entries must be mappings")
        row = dict(raw_row)
        overlay_rows[str(overlay_name)] = row
        recovery_ratio = row.get("recovery_ratio")
        semantic_gap = row.get("semantic_gap_vs_prompt_only")
        if recovery_ratio is None or semantic_gap is None:
            continue
        if float(recovery_ratio) >= config.c1_recovery_ratio_floor and float(semantic_gap) >= semantic_gap_floor:
            passing.append(str(overlay_name))

    if len(passing) >= config.c1_required_overlays:
        status = "supported"
    elif passing:
        status = "partially supported"
    else:
        status = "unsupported"

    return {
        "status": status,
        "reason": (
            f"{len(passing)} overlay(s) meet the default recovery threshold "
            f"(ratio >= {config.c1_recovery_ratio_floor:.2f}, semantic gap >= {semantic_gap_floor:.2f}): "
            f"{', '.join(passing) if passing else 'none'}"
        ),
        "evidence": {
            "passing_overlays": passing,
            "required_overlays": config.c1_required_overlays,
            "recovery_ratio_floor": config.c1_recovery_ratio_floor,
            "semantic_gap_floor": semantic_gap_floor,
            "overlay_rows": overlay_rows,
        },
    }


def _composition_conditions(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json(path)
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, Mapping):
        raise TypeError("composition summary conditions must be a mapping")
    return {str(key): dict(value) for key, value in conditions.items() if isinstance(value, Mapping)}


def _all_composition_conditions(config: ClaimCheckConfig) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    rows.update(_composition_conditions(config.pair_summary_path))
    rows.update(_composition_conditions(config.triple_summary_path))
    return rows


def _c2_status(config: ClaimCheckConfig) -> dict[str, Any]:
    pair_conditions = _composition_conditions(config.pair_summary_path)
    if not pair_conditions:
        return {
            "status": "unsupported",
            "reason": "pair composition summary is required to assess C2",
            "evidence": {},
        }

    winning = []
    partial = []
    for condition, row in pair_conditions.items():
        derived = row.get("derived", {})
        if not isinstance(derived, Mapping):
            raise TypeError("composition condition derived block must be a mapping")
        soar_minus_prompt = float(derived.get("soar_minus_prompt_only_mean_active", 0.0))
        soar_minus_merge = float(derived.get("soar_minus_merge_mean_active", 0.0))
        guardrail = bool(derived.get("guardrail_satisfied", False))
        if guardrail and soar_minus_prompt >= config.c2_margin_points and soar_minus_merge >= config.c2_margin_points:
            winning.append(condition)
        elif guardrail and (soar_minus_prompt > 0.0 or soar_minus_merge > 0.0):
            partial.append(condition)

    if winning:
        status = "supported"
        reason = f"SOAR beats both baselines under the guardrail for: {', '.join(sorted(winning))}"
    elif partial:
        status = "partially supported"
        reason = f"SOAR improves on at least one baseline under the guardrail for: {', '.join(sorted(partial))}"
    else:
        status = "unsupported"
        reason = "No pair beats both prompt-only and tuned merge under the semantic guardrail"
    return {
        "status": status,
        "reason": reason,
        "evidence": {
            "winning_pairs": sorted(winning),
            "partial_pairs": sorted(partial),
            "margin_points": config.c2_margin_points,
        },
    }


def _c3_status(config: ClaimCheckConfig) -> dict[str, Any]:
    summaries = []
    if config.pair_summary_path is not None:
        summaries.append(_read_json(config.pair_summary_path))
    if config.triple_summary_path is not None:
        summaries.append(_read_json(config.triple_summary_path))
    if not summaries:
        return {
            "status": "unsupported",
            "reason": "pair/triple composition summaries are required to assess C3",
            "evidence": {},
        }

    gap_rows: dict[str, list[float]] = {"J": [], "C": [], "Q": []}
    semantic_rows: dict[str, list[float]] = {"J": [], "C": [], "Q": []}
    for payload in summaries:
        overlay_summary = payload.get("overlay_summary", {})
        if not isinstance(overlay_summary, Mapping):
            raise TypeError("composition overlay_summary must be a mapping")
        for overlay in ("J", "C", "Q"):
            row = overlay_summary.get(overlay)
            if not isinstance(row, Mapping):
                continue
            if row.get("median_composition_gap_vs_direct") is not None:
                gap_rows[overlay].append(float(row["median_composition_gap_vs_direct"]))
            if row.get("median_semantic_gap_vs_prompt_only") is not None:
                semantic_rows[overlay].append(abs(float(row["median_semantic_gap_vs_prompt_only"])))

    if not gap_rows["J"]:
        return {
            "status": "unsupported",
            "reason": "no overlay-summary composition gaps were available for C3",
            "evidence": {},
        }

    gap_medians = {overlay: sum(values) / len(values) for overlay, values in gap_rows.items() if values}
    semantic_medians = {overlay: sum(values) / len(values) for overlay, values in semantic_rows.items() if values}
    json_best_gap = gap_medians.get("J", float("inf")) <= min(gap_medians.get("C", float("inf")), gap_medians.get("Q", float("inf")))
    json_best_semantic = semantic_medians.get("J", float("inf")) <= min(
        semantic_medians.get("C", float("inf")),
        semantic_medians.get("Q", float("inf")),
    )
    if json_best_gap and json_best_semantic:
        status = "supported"
        reason = "JSON has the smallest observed composition gap and collateral semantic degradation"
    elif json_best_gap or json_best_semantic:
        status = "partially supported"
        reason = "JSON leads one of the composition-gap / collateral-damage summaries but not both"
    else:
        status = "unsupported"
        reason = "composition summaries do not show a stable JSON advantage"
    return {
        "status": status,
        "reason": reason,
        "evidence": {
            "gap_medians": gap_medians,
            "semantic_medians": semantic_medians,
        },
    }


def _c4_status(config: ClaimCheckConfig, summary: Mapping[str, Any]) -> dict[str, Any]:
    factor_payload = _read_json(config.factor_ablation_path) if config.factor_ablation_path is not None else None
    if factor_payload is not None:
        explicit_status = factor_payload.get("status")
        dense_delta_preferred = factor_payload.get("dense_delta_preferred")
        if explicit_status == "supported" or dense_delta_preferred is True:
            return {
                "status": "supported",
                "reason": "factor-ablation input explicitly supports dense-delta residualization",
                "evidence": factor_payload,
            }

    c1 = _c1_status(config, summary)
    if c1["evidence"]["passing_overlays"]:
        status = "partially supported"
        reason = "single-overlay recovery remains positive, but no factor-subtraction ablation was provided"
    else:
        status = "unsupported"
        reason = "no positive recovery evidence and no factor-ablation evidence were provided"
    return {
        "status": status,
        "reason": reason,
        "evidence": {
            "factor_ablation_path": str(config.factor_ablation_path) if config.factor_ablation_path is not None else None,
            "c1_passing_overlays": c1["evidence"]["passing_overlays"],
        },
    }


def _c6_status(config: ClaimCheckConfig) -> dict[str, Any]:
    conditions = _all_composition_conditions(config)
    if not conditions:
        return {
            "status": "unsupported",
            "reason": "pair/triple composition summaries are required to assess C6",
            "evidence": {},
        }

    direct_better = []
    soar_close = []
    for condition, row in conditions.items():
        derived = row.get("derived", {})
        if not isinstance(derived, Mapping):
            raise TypeError("composition condition derived block must be a mapping")
        gap = float(derived.get("composition_gap_vs_direct", 0.0))
        if gap > 0.0:
            direct_better.append(condition)
        if gap <= 0.05:
            soar_close.append(condition)

    if direct_better:
        status = "supported"
        reason = "direct pair/triple upper bounds remain meaningfully useful ceilings for at least one condition"
    elif soar_close:
        status = "partially supported"
        reason = "SOAR stays close to the direct upper bounds, but no clear direct ceiling emerged"
    else:
        status = "unsupported"
        reason = "composition summaries do not establish a useful direct upper-bound gap"
    return {
        "status": status,
        "reason": reason,
        "evidence": {
            "direct_better_conditions": sorted(direct_better),
            "near_direct_conditions": sorted(soar_close),
        },
    }


def _c5_status(config: ClaimCheckConfig) -> dict[str, Any]:
    if config.split_audit_path is None or config.dataset_manifest_path is None:
        return {
            "status": "unsupported",
            "reason": "split_audit and dataset_manifest inputs are required to assess C5",
            "evidence": {},
        }

    split_audit = _read_json(config.split_audit_path)
    dataset_manifest = _read_json(config.dataset_manifest_path)
    split_passed = bool(split_audit.get("passed"))
    roundtrip = dataset_manifest.get("roundtrip_audit", {})
    if not isinstance(roundtrip, Mapping):
        raise TypeError("dataset manifest roundtrip_audit must be a mapping")
    roundtrip_failures = int(roundtrip.get("failed_examples", -1))
    processed_counts = dataset_manifest.get("counts", {})
    if not isinstance(processed_counts, Mapping):
        raise TypeError("dataset manifest counts must be a mapping")
    has_processed_examples = all(
        int(processed_counts.get(key, 0)) > 0
        for key in ("processed_train_examples", "processed_val_examples", "processed_test_examples")
    )

    status = "supported" if split_passed and roundtrip_failures == 0 and has_processed_examples else "unsupported"
    reason = (
        "split audit passed and dataset manifest shows zero round-trip failures on non-empty processed splits"
        if status == "supported"
        else "provided dataset artifacts do not satisfy the deterministic split/round-trip checks"
    )
    return {
        "status": status,
        "reason": reason,
        "evidence": {
            "split_audit_passed": split_passed,
            "roundtrip_failed_examples": roundtrip_failures,
            "has_processed_examples": has_processed_examples,
        },
    }


def _c8_status(config: ClaimCheckConfig) -> dict[str, Any]:
    if config.transfer_summary_path is None:
        return {
            "status": "intended",
            "reason": "transfer summary was not provided",
            "evidence": {},
        }

    payload = _read_json(config.transfer_summary_path)
    retained = payload.get("retained_pair_advantage_conditions", [])
    if not isinstance(retained, list):
        raise TypeError("transfer summary retained_pair_advantage_conditions must be a list")
    if retained:
        status = "supported"
        reason = f"transfer slice retains at least one composition advantage: {', '.join(str(item) for item in retained)}"
    else:
        status = "unsupported"
        reason = "transfer summary did not retain a pair-composition advantage over prompt-only"
    return {
        "status": status,
        "reason": reason,
        "evidence": {
            "retained_pair_advantage_conditions": [str(item) for item in retained],
        },
    }


def build_claim_checks(config: ClaimCheckConfig) -> dict[str, Any]:
    single_overlay_summary = _read_json(config.single_overlay_summary_path)
    claim_results: dict[str, Any] = {}
    for claim_id in config.claims:
        if claim_id == "C1":
            claim_results[claim_id] = _c1_status(config, single_overlay_summary)
        elif claim_id == "C2":
            claim_results[claim_id] = _c2_status(config)
        elif claim_id == "C3":
            claim_results[claim_id] = _c3_status(config)
        elif claim_id == "C4":
            claim_results[claim_id] = _c4_status(config, single_overlay_summary)
        elif claim_id == "C5":
            claim_results[claim_id] = _c5_status(config)
        elif claim_id == "C6":
            claim_results[claim_id] = _c6_status(config)
        elif claim_id == "C8":
            claim_results[claim_id] = _c8_status(config)
        else:
            claim_results[claim_id] = {
                "status": "intended",
                "reason": "no automated check is defined for this claim in the current milestone",
                "evidence": {},
            }

    report = {
        "experiment_name": config.experiment_name,
        "milestone": config.milestone,
        "claims": claim_results,
        "sources": {
            "single_overlay_summary": str(config.single_overlay_summary_path),
            "pair_summary": str(config.pair_summary_path) if config.pair_summary_path is not None else None,
            "triple_summary": str(config.triple_summary_path) if config.triple_summary_path is not None else None,
            "split_audit": str(config.split_audit_path) if config.split_audit_path is not None else None,
            "dataset_manifest": str(config.dataset_manifest_path) if config.dataset_manifest_path is not None else None,
            "factor_ablation": str(config.factor_ablation_path) if config.factor_ablation_path is not None else None,
            "transfer_summary": str(config.transfer_summary_path) if config.transfer_summary_path is not None else None,
            "config_path": str(config.source_config),
        },
    }

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        config.source_config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"
    _write_json(metrics_path, report)

    lines = ["# Claim Checks", ""]
    for claim_id in config.claims:
        claim = claim_results[claim_id]
        lines.extend(
            [
                f"## {claim_id}",
                f"- status: {claim['status']}",
                f"- reason: {claim['reason']}",
                "",
            ]
        )
    _write_markdown(report_path, "\n".join(lines))

    return {
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "claims": {claim_id: claim_results[claim_id]["status"] for claim_id in config.claims},
    }


def load_claim_check_config(path: str | Path) -> ClaimCheckConfig:
    return ClaimCheckConfig.from_app_config(load_app_config(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build automated claim-check suggestions from saved report artifacts.")
    parser.add_argument("--config", required=True, help="Path to the claim-check YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config = load_claim_check_config(Path(args.config))
    summary = build_claim_checks(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(f"Wrote claim checks to {summary['report_path']}.")
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
