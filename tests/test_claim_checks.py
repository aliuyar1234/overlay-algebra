from __future__ import annotations

import json
from pathlib import Path

from overlay_algebra.reports.claim_checks import ClaimCheckConfig, build_claim_checks


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_claim_checks_promote_c1_when_two_overlays_pass(tmp_path: Path) -> None:
    summary_path = tmp_path / "single_overlay_summary.json"
    split_audit_path = tmp_path / "split_audit.json"
    dataset_manifest_path = tmp_path / "dataset_manifest.json"
    config_path = tmp_path / "claim_checks.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    _write_json(
        summary_path,
        {
            "overlays": {
                "J": {"recovery_ratio": 1.0, "semantic_gap_vs_prompt_only": 0.18},
                "C": {"recovery_ratio": 1.0, "semantic_gap_vs_prompt_only": -0.01},
                "Q": {"recovery_ratio": 0.99, "semantic_gap_vs_prompt_only": 0.24},
            }
        },
    )
    _write_json(split_audit_path, {"passed": True})
    _write_json(
        dataset_manifest_path,
        {
            "counts": {
                "processed_train_examples": 10,
                "processed_val_examples": 5,
                "processed_test_examples": 5,
            },
            "roundtrip_audit": {"failed_examples": 0},
        },
    )

    config = ClaimCheckConfig(
        experiment_name="claim_checks_m4",
        milestone="M4",
        output_dir=tmp_path / "claim_checks_out",
        single_overlay_summary_path=summary_path,
        pair_summary_path=None,
        triple_summary_path=None,
        split_audit_path=split_audit_path,
        dataset_manifest_path=dataset_manifest_path,
        factor_ablation_path=None,
        transfer_summary_path=None,
        claims=["C1", "C4", "C5"],
        c1_required_overlays=2,
        c1_recovery_ratio_floor=0.8,
        c2_margin_points=0.03,
        semantic_guardrail_points=3.0,
        source_config=config_path,
    )

    summary = build_claim_checks(config)

    assert summary["claims"]["C1"] == "supported"
    assert summary["claims"]["C4"] == "partially supported"
    assert summary["claims"]["C5"] == "supported"
    assert (tmp_path / "claim_checks_out" / "report.md").exists()


def test_claim_checks_cover_c2_c3_and_c6_from_composition_reports(tmp_path: Path) -> None:
    summary_path = tmp_path / "single_overlay_summary.json"
    pair_summary_path = tmp_path / "pair_summary.json"
    triple_summary_path = tmp_path / "triple_summary.json"
    config_path = tmp_path / "claim_checks.yaml"
    config_path.write_text("experiment_name: test\n", encoding="utf-8")

    _write_json(
        summary_path,
        {
            "overlays": {
                "J": {"recovery_ratio": 0.99, "semantic_gap_vs_prompt_only": -0.01},
                "C": {"recovery_ratio": 0.82, "semantic_gap_vs_prompt_only": -0.02},
                "Q": {"recovery_ratio": 0.95, "semantic_gap_vs_prompt_only": -0.01},
            }
        },
    )
    _write_json(
        pair_summary_path,
        {
            "conditions": {
                "JC": {
                    "derived": {
                        "soar_minus_prompt_only_mean_active": 0.04,
                        "soar_minus_merge_mean_active": 0.05,
                        "composition_gap_vs_direct": 0.02,
                        "guardrail_satisfied": True,
                    }
                },
                "JQ": {
                    "derived": {
                        "soar_minus_prompt_only_mean_active": 0.02,
                        "soar_minus_merge_mean_active": 0.01,
                        "composition_gap_vs_direct": 0.07,
                        "guardrail_satisfied": True,
                    }
                },
            },
            "overlay_summary": {
                "J": {"median_composition_gap_vs_direct": 0.02, "median_semantic_gap_vs_prompt_only": -0.01},
                "C": {"median_composition_gap_vs_direct": 0.07, "median_semantic_gap_vs_prompt_only": -0.03},
                "Q": {"median_composition_gap_vs_direct": 0.05, "median_semantic_gap_vs_prompt_only": -0.02},
            },
        },
    )
    _write_json(
        triple_summary_path,
        {
            "conditions": {
                "JCQ": {
                    "derived": {
                        "soar_minus_prompt_only_mean_active": 0.01,
                        "soar_minus_merge_mean_active": 0.00,
                        "composition_gap_vs_direct": 0.06,
                        "guardrail_satisfied": True,
                    }
                }
            },
            "overlay_summary": {
                "J": {"median_composition_gap_vs_direct": 0.03, "median_semantic_gap_vs_prompt_only": -0.01},
                "C": {"median_composition_gap_vs_direct": 0.08, "median_semantic_gap_vs_prompt_only": -0.03},
                "Q": {"median_composition_gap_vs_direct": 0.06, "median_semantic_gap_vs_prompt_only": -0.02},
            },
        },
    )

    config = ClaimCheckConfig(
        experiment_name="claim_checks_m6",
        milestone="M6",
        output_dir=tmp_path / "claim_checks_out",
        single_overlay_summary_path=summary_path,
        pair_summary_path=pair_summary_path,
        triple_summary_path=triple_summary_path,
        split_audit_path=None,
        dataset_manifest_path=None,
        factor_ablation_path=None,
        transfer_summary_path=None,
        claims=["C2", "C3", "C6"],
        c1_required_overlays=2,
        c1_recovery_ratio_floor=0.8,
        c2_margin_points=0.03,
        semantic_guardrail_points=3.0,
        source_config=config_path,
    )

    summary = build_claim_checks(config)

    assert summary["claims"]["C2"] == "supported"
    assert summary["claims"]["C3"] == "supported"
    assert summary["claims"]["C6"] == "supported"
