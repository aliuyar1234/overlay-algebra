from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from overlay_algebra.analysis.adapter_eval import mean_metric_rows
from overlay_algebra.analysis.residualize import BetaCandidate, ResidualizeConfig, choose_best_beta, load_residualize_config
from overlay_algebra.artifacts import build_run_paths
from overlay_algebra.metrics import score_prediction
from overlay_algebra.parsers import parse_condition
from overlay_algebra.sweep_runtime import SweepRuntimeState


_PRIMARY_METRIC_BY_CONDITION = {
    "J": "json_strict_schema_valid",
    "C": "citation_exact",
    "Q": "quote_exact",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=False) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _gold_fields_from_target(condition: str, target_text: str) -> tuple[str, int | None, str | None]:
    parsed = parse_condition(target_text, condition)
    gold_answer = str(parsed.get("answer", ""))
    gold_support = parsed.get("support")
    gold_support_idx = int(str(gold_support)[1:]) if gold_support is not None else None
    gold_support_sentence = str(parsed["quote"]) if parsed.get("quote") is not None else None
    return gold_answer, gold_support_idx, gold_support_sentence


def recompute_prediction_rows(predictions_path: Path) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
    rows = _read_jsonl(predictions_path)
    if not rows:
        raise ValueError(f"No prediction rows found in {predictions_path}")

    metric_rows: list[dict[str, float | None]] = []
    repaired_rows: list[dict[str, Any]] = []
    inferred_condition: str | None = None
    for row in rows:
        repaired = dict(row)
        condition = str(repaired.get("condition", "")).strip()
        target_text = str(repaired.get("target", ""))
        prediction_text = str(repaired.get("prediction", ""))
        if not condition:
            raise ValueError(f"Prediction row in {predictions_path} is missing condition")
        if not target_text:
            raise ValueError(f"Prediction row in {predictions_path} is missing target text")
        gold_answer, gold_support_idx, gold_support_sentence = _gold_fields_from_target(condition, target_text)
        metrics = score_prediction(
            condition,
            prediction_text,
            gold_answer=gold_answer,
            gold_support_idx=gold_support_idx,
            gold_support_sentence=gold_support_sentence,
        )
        repaired["metrics"] = metrics
        repaired_rows.append(repaired)
        metric_rows.append(metrics)
        inferred_condition = condition

    assert inferred_condition is not None
    aggregate_metrics = mean_metric_rows(metric_rows)
    return inferred_condition, repaired_rows, aggregate_metrics


def repair_eval_run(run_dir: Path) -> dict[str, Any]:
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    payload = _read_json(metrics_path)
    condition, repaired_rows, aggregate_metrics = recompute_prediction_rows(predictions_path)
    payload["condition"] = condition
    payload["metrics"] = {"condition": condition, **aggregate_metrics}
    _write_jsonl(predictions_path, repaired_rows)
    _write_json(metrics_path, payload)
    return {
        "run_dir": str(run_dir),
        "condition": condition,
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path),
        "aggregate_metrics": aggregate_metrics,
    }


def _runtime_state_from_latest(run_dir: Path) -> tuple[SweepRuntimeState, Path]:
    latest_path = run_dir / "checkpoints" / "latest.json"
    latest_payload = _read_json(latest_path)
    checkpoint_dir = Path(str(latest_payload["checkpoint_dir"]))
    runtime_payload = _read_json(checkpoint_dir / "runtime_state.json")
    return SweepRuntimeState.from_mapping(runtime_payload), checkpoint_dir


def _cleanup_residualize_final_artifacts(run_dir: Path) -> None:
    for relative in [
        "metrics.json",
        "predictions.jsonl",
        "tuning_manifest.json",
        "adapter",
        "residual_adapter",
        "delta_cache\\residual",
        "delta_cache\\composed",
    ]:
        target = run_dir / relative
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()


def prepare_residualize_repair(config_path: Path) -> dict[str, Any]:
    config: ResidualizeConfig = load_residualize_config(config_path)
    run_paths = build_run_paths(config.app)
    run_dir = Path(run_paths.run_dir)
    state_path = run_dir / "state.json"
    state = _read_json(state_path)

    prompt_metrics_path = Path(str(state["prompt_only"]["metrics_path"]))
    prompt_predictions_path = Path(str(state["prompt_only"]["predictions_path"]))
    condition, repaired_prompt_rows, prompt_metrics = recompute_prediction_rows(prompt_predictions_path)
    prompt_payload = _read_json(prompt_metrics_path)
    prompt_payload["metrics"] = {"condition": condition, **prompt_metrics}
    _write_jsonl(prompt_predictions_path, repaired_prompt_rows)
    _write_json(prompt_metrics_path, prompt_payload)
    state["prompt_only"]["metrics"] = {"condition": condition, **prompt_metrics}

    baseline_answer_f1 = float(prompt_metrics.get("answer_f1", 0.0))
    primary_metric = _PRIMARY_METRIC_BY_CONDITION[config.overlay_name]
    for beta_key, candidate_state in state["candidates"].items():
        metrics_path = Path(str(candidate_state["metrics_path"]))
        predictions_path = Path(str(candidate_state["predictions_path"]))
        _, repaired_rows, candidate_metrics = recompute_prediction_rows(predictions_path)
        metrics_payload = _read_json(metrics_path)
        metrics_payload["metrics"] = {"condition": config.overlay_name, **candidate_metrics}
        _write_jsonl(predictions_path, repaired_rows)
        _write_json(metrics_path, metrics_payload)
        semantic_f1_points = float(candidate_metrics.get("answer_f1", 0.0)) * 100.0
        candidate_state["metrics"] = {"condition": config.overlay_name, **candidate_metrics}
        candidate_state["overlay_primary"] = float(candidate_metrics.get(primary_metric, 0.0))
        candidate_state["semantic_f1_points"] = semantic_f1_points
        candidate_state["guardrail_satisfied"] = semantic_f1_points >= (
            baseline_answer_f1 * 100.0 - config.semantic_guardrail_points
        )

    beta_candidates = []
    for beta in config.beta_grid:
        candidate = state["candidates"][str(beta).replace(".", "_")]
        beta_candidates.append(
            {
                "beta": beta,
                "metrics": dict(candidate["metrics"]),
            }
        )

    chosen_candidate, selection = choose_best_beta(
        config.overlay_name,
        baseline_answer_f1=baseline_answer_f1,
        guardrail_points=config.semantic_guardrail_points,
        candidates=[
            BetaCandidate(
                beta=float(candidate["beta"]),
                adapter_dir=Path(str(state["candidates"][str(candidate["beta"]).replace(".", "_")]["adapter_dir"])),
                metrics=dict(candidate["metrics"]),
                predictions=[],
            )
            for candidate in beta_candidates
        ],
    )

    state["status"] = "pending"
    state["chosen_beta"] = None
    state["selection"] = None
    state.pop("chosen_predictions_path", None)
    state.pop("residual_adapter_dir", None)
    state.pop("composed_adapter_dir", None)
    _write_json(state_path, state)

    runtime, _latest_checkpoint_dir = _runtime_state_from_latest(run_dir)
    repair_checkpoint_dir = run_dir / "checkpoints" / "metric_aggregation_repair_ready"
    repair_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime.latest_checkpoint_dir = str(repair_checkpoint_dir)
    runtime.latest_checkpoint_units_completed = runtime.units_completed
    _write_json(repair_checkpoint_dir / "runtime_state.json", runtime.as_dict())
    _write_json(repair_checkpoint_dir / "state.json", state)
    _write_json(
        repair_checkpoint_dir / "metadata.json",
        {
            "created_at": None,
            "checkpoint_dir": str(repair_checkpoint_dir),
            "units_completed": runtime.units_completed,
            "repair_note": "Prepared after metric aggregation repair; resume here to rerun selection/finalization only.",
        },
    )

    _cleanup_residualize_final_artifacts(run_dir)

    return {
        "run_dir": str(run_dir),
        "repair_checkpoint_dir": str(repair_checkpoint_dir),
        "prompt_only_metrics": prompt_metrics,
        "would_choose_beta": chosen_candidate.beta,
        "would_selection": selection,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair the saved M4 Q metric-aggregation artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("repair-eval-run", help="Recompute metrics for an eval run from saved predictions.")
    eval_parser.add_argument("--run-dir", required=True, help="Path to the eval run directory.")
    eval_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    residual_parser = subparsers.add_parser(
        "prepare-residualize-resume",
        help="Repair saved residualize candidate metrics and prepare a resume checkpoint for reselection/finalization.",
    )
    residual_parser.add_argument("--config", required=True, help="Path to the residualize config.")
    residual_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    args = parser.parse_args()

    if args.command == "repair-eval-run":
        summary = repair_eval_run(Path(args.run_dir))
    else:
        summary = prepare_residualize_repair(Path(args.config))

    if getattr(args, "json", False):
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
