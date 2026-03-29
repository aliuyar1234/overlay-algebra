from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import overlay_algebra.m4 as m4_module
from overlay_algebra.m4 import (
    M4StepSpec,
    collect_m4_status,
    evaluate_step_state,
    plan_m4_actions,
    run_m4_pipeline,
    watch_m4_pipeline,
)


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_evaluate_step_state_reads_running_train_heartbeat(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "train" / "scaffold.yaml"
    _write_yaml(
        config_path,
        "\n".join(
            [
                "experiment_name: full_scaffold",
                "milestone: M4",
                "command: overlay_algebra.train.fit",
                "seed: 17",
            ]
        ),
    )
    run_dir = tmp_path / "runs" / "oa_m4_full_scaffold_17"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    (checkpoints_dir / "latest.json").write_text(
        json.dumps({"checkpoint_dir": str(checkpoints_dir / "step_0000100_micro_0000400")}),
        encoding="utf-8",
    )
    (run_dir / "heartbeat.json").write_text(
        json.dumps(
            {
                "updated_at": _now_iso(),
                "status": "running",
                "progress": {
                    "optimizer_steps_completed": 100,
                    "optimizer_steps_total": 400,
                },
                "timing": {"eta_seconds": 1200.0},
                "message": "training",
            }
        ),
        encoding="utf-8",
    )

    state = evaluate_step_state(
        M4StepSpec(key="scaffold", label="Train scaffold", config_path=config_path),
        repo_root=tmp_path,
    )

    assert state.status == "running"
    assert state.progress_completed == 100
    assert state.progress_total == 400
    assert state.progress_unit == "optimizer_steps"
    assert state.percent_complete == 25.0
    assert state.latest_checkpoint == checkpoints_dir / "latest.json"


def test_plan_m4_actions_skips_completed_and_resumes_partial(tmp_path: Path) -> None:
    train_config = tmp_path / "configs" / "train" / "scaffold.yaml"
    residual_config = tmp_path / "configs" / "analysis" / "residualize.yaml"
    report_config = tmp_path / "configs" / "reports" / "report.yaml"
    _write_yaml(
        train_config,
        "\n".join(
            [
                "experiment_name: full_scaffold",
                "milestone: M4",
                "command: overlay_algebra.train.fit",
                "seed: 17",
            ]
        ),
    )
    _write_yaml(
        residual_config,
        "\n".join(
            [
                "experiment_name: full_residualize_j",
                "milestone: M4",
                "command: overlay_algebra.analysis.residualize",
                "seed: 17",
            ]
        ),
    )
    _write_yaml(
        report_config,
        "\n".join(
            [
                "experiment_name: full_single_overlay_recovery_j",
                "milestone: M4",
                "command: overlay_algebra.reports.single_overlay_recovery",
                "seed: 17",
                "paths:",
                "  output_dir: reports/full_single_overlay_recovery_j",
            ]
        ),
    )

    completed_run = tmp_path / "runs" / "oa_m4_full_scaffold_17"
    completed_run.mkdir(parents=True)
    (completed_run / "adapter").mkdir()
    (completed_run / "metrics.json").write_text("{}", encoding="utf-8")
    (completed_run / "run_manifest.json").write_text("{}", encoding="utf-8")

    partial_run = tmp_path / "runs" / "oa_m4_full_residualize_j_17"
    partial_run.mkdir(parents=True)
    (partial_run / "checkpoints").mkdir()
    (partial_run / "checkpoints" / "latest.json").write_text("{}", encoding="utf-8")

    steps = [
        M4StepSpec(key="scaffold", label="Train scaffold", config_path=train_config),
        M4StepSpec(key="residualize_j", label="Residualize J", config_path=residual_config),
        M4StepSpec(key="report_j", label="Build J report", config_path=report_config),
    ]

    actions = plan_m4_actions(repo_root=tmp_path, steps=steps)

    assert actions == [
        {
            "key": "scaffold",
            "label": "Train scaffold",
            "status": "completed",
            "action": "skip",
            "artifact_path": str(completed_run),
            "latest_checkpoint": None,
        },
        {
            "key": "residualize_j",
            "label": "Residualize J",
            "status": "partial",
            "action": "resume",
            "artifact_path": str(partial_run),
            "latest_checkpoint": str(partial_run / "checkpoints" / "latest.json"),
        },
        {
            "key": "report_j",
            "label": "Build J report",
            "status": "pending",
            "action": "run",
            "artifact_path": str(tmp_path / "reports" / "full_single_overlay_recovery_j"),
            "latest_checkpoint": None,
        },
    ]


def test_run_m4_pipeline_refuses_when_recent_running_run_exists(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "train" / "scaffold.yaml"
    _write_yaml(
        config_path,
        "\n".join(
            [
                "experiment_name: full_scaffold",
                "milestone: M4",
                "command: overlay_algebra.train.fit",
                "seed: 17",
            ]
        ),
    )
    other_run = tmp_path / "runs" / "oa_m3_other_17"
    other_run.mkdir(parents=True)
    (other_run / "heartbeat.json").write_text(
        json.dumps({"updated_at": _now_iso(), "status": "running"}),
        encoding="utf-8",
    )

    summary = run_m4_pipeline(
        repo_root=tmp_path,
        steps=[M4StepSpec(key="scaffold", label="Train scaffold", config_path=config_path)],
        runner=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("runner should not be called")),
    )

    assert summary["status"] == "running"
    assert "not starting another M4 step" in summary["message"]
    assert summary["active_runs"][0]["run_dir"] == str(other_run)


def test_collect_m4_status_reports_next_step_and_external_run(tmp_path: Path) -> None:
    train_config = tmp_path / "configs" / "train" / "scaffold.yaml"
    report_config = tmp_path / "configs" / "reports" / "report.yaml"
    _write_yaml(
        train_config,
        "\n".join(
            [
                "experiment_name: full_scaffold",
                "milestone: M4",
                "command: overlay_algebra.train.fit",
                "seed: 17",
            ]
        ),
    )
    _write_yaml(
        report_config,
        "\n".join(
            [
                "experiment_name: full_single_overlay_recovery_j",
                "milestone: M4",
                "command: overlay_algebra.reports.single_overlay_recovery",
                "seed: 17",
                "paths:",
                "  output_dir: reports/full_single_overlay_recovery_j",
            ]
        ),
    )

    completed_run = tmp_path / "runs" / "oa_m4_full_scaffold_17"
    completed_run.mkdir(parents=True)
    (completed_run / "adapter").mkdir()
    (completed_run / "metrics.json").write_text("{}", encoding="utf-8")
    (completed_run / "run_manifest.json").write_text("{}", encoding="utf-8")

    external_run = tmp_path / "runs" / "oa_m5_pairs_17"
    external_run.mkdir(parents=True)
    (external_run / "heartbeat.json").write_text(
        json.dumps({"updated_at": _now_iso(), "status": "running"}),
        encoding="utf-8",
    )

    summary = collect_m4_status(
        repo_root=tmp_path,
        steps=[
            M4StepSpec(key="scaffold", label="Train scaffold", config_path=train_config),
            M4StepSpec(key="report_j", label="Build J report", config_path=report_config),
        ],
    )

    assert summary["overall_status"] == "ready"
    assert summary["completed_steps"] == 1
    assert summary["total_steps"] == 2
    assert summary["next_step"] == {
        "key": "report_j",
        "label": "Build J report",
        "status": "pending",
    }
    assert len(summary["external_running_runs"]) == 1
    external = summary["external_running_runs"][0]
    assert external["run_dir"] == str(external_run)
    assert external["stale"] is False
    assert external["progress_completed"] is None
    assert external["progress_total"] is None
    assert external["latest_checkpoint"] is None


def test_collect_m4_status_uses_recent_external_run_activity_when_heartbeat_lags(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "reports" / "report.yaml"
    _write_yaml(
        config_path,
        "\n".join(
            [
                "experiment_name: full_single_overlay_recovery_j",
                "milestone: M4",
                "command: overlay_algebra.reports.single_overlay_recovery",
                "seed: 17",
                "paths:",
                "  output_dir: reports/full_single_overlay_recovery_j",
            ]
        ),
    )

    external_run = tmp_path / "runs" / "oa_m5_full_pairs_jq_17"
    checkpoints_dir = external_run / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    stale_heartbeat_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    (external_run / "heartbeat.json").write_text(
        json.dumps(
            {
                "updated_at": stale_heartbeat_time,
                "status": "running",
                "progress": {
                    "units_completed": 0,
                    "units_total": 33,
                    "unit_label": "evaluation_units",
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoints_dir / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoints_dir / "unit_0000001"),
                "state_path": str(checkpoints_dir / "unit_0000001" / "state.json"),
                "runtime_state_path": str(checkpoints_dir / "unit_0000001" / "runtime_state.json"),
                "units_completed": 1,
                "updated_at": datetime.now(timezone.utc).timestamp() - (20 * 60),
            }
        ),
        encoding="utf-8",
    )
    candidate_file = external_run / "compositions" / "JQ" / "candidates" / "soar" / "alpha__j_0_50__q_0_50" / "adapter" / "adapter_model.safetensors"
    candidate_file.parent.mkdir(parents=True)
    candidate_file.write_text("x", encoding="utf-8")

    summary = collect_m4_status(
        repo_root=tmp_path,
        steps=[M4StepSpec(key="report_j", label="Build J report", config_path=config_path)],
    )

    assert len(summary["external_running_runs"]) == 1
    external = summary["external_running_runs"][0]
    assert external["run_dir"] == str(external_run)
    assert external["stale"] is False
    assert external["progress_completed"] == 1
    assert external["progress_total"] == 33
    assert external["progress_unit"] == "evaluation_units"
    assert external["latest_checkpoint"] == str(checkpoints_dir / "latest.json")
    assert "recent filesystem activity at" in external["detail"]


def test_collect_m4_status_prefers_reduced_scope_jq_next_step(tmp_path: Path) -> None:
    def write_run_config(path: Path, experiment_name: str) -> None:
        _write_yaml(
            path,
            "\n".join(
                [
                    f"experiment_name: {experiment_name}",
                    "milestone: M4",
                    "command: overlay_algebra.analysis.residualize",
                    "seed: 17",
                ]
            ),
        )

    scaffold_config = tmp_path / "configs" / "train" / "scaffold.yaml"
    residualize_c_config = tmp_path / "configs" / "analysis" / "full_residuals_C.yaml"
    residualize_q_config = tmp_path / "configs" / "analysis" / "full_residuals_Q.yaml"
    _write_yaml(
        scaffold_config,
        "\n".join(
            [
                "experiment_name: full_scaffold",
                "milestone: M4",
                "command: overlay_algebra.train.fit",
                "seed: 17",
            ]
        ),
    )
    write_run_config(residualize_c_config, "full_residualize_c")
    write_run_config(residualize_q_config, "full_residualize_q")

    completed_run = tmp_path / "runs" / "oa_m4_full_scaffold_17"
    completed_run.mkdir(parents=True)
    (completed_run / "adapter").mkdir()
    (completed_run / "metrics.json").write_text("{}", encoding="utf-8")
    (completed_run / "run_manifest.json").write_text("{}", encoding="utf-8")

    summary = collect_m4_status(
        repo_root=tmp_path,
        steps=[
            M4StepSpec(key="scaffold", label="Train scaffold", config_path=scaffold_config),
            M4StepSpec(key="residualize_c", label="Residualize C", config_path=residualize_c_config),
            M4StepSpec(key="residualize_q", label="Residualize Q", config_path=residualize_q_config),
        ],
    )

    assert summary["status_scope"] == "jq_first_reduced_scope"
    assert summary["next_step"] == {
        "key": "residualize_q",
        "label": "Residualize Q",
        "status": "pending",
    }
    assert summary["tree_next_step"] == {
        "key": "residualize_q",
        "label": "Residualize Q",
        "status": "pending",
    }
    assert summary["deprioritized_steps"] == [
        {
            "key": "residualize_c",
            "label": "Residualize C",
            "status": "pending",
        }
    ]
    assert [item["key"] for item in summary["remaining_steps"]] == ["residualize_q"]
    assert [item["key"] for item in summary["tree_remaining_steps"]] == ["residualize_q"]


def test_evaluate_step_state_uses_checkpoint_progress_when_heartbeat_lags(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "analysis" / "full_residuals_Q.yaml"
    _write_yaml(
        config_path,
        "\n".join(
            [
                "experiment_name: full_residualize_q",
                "milestone: M4",
                "command: overlay_algebra.analysis.residualize",
                "seed: 17",
            ]
        ),
    )
    run_dir = tmp_path / "runs" / "oa_m4_full_residualize_q_17"
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    stale_heartbeat_time = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
    (run_dir / "heartbeat.json").write_text(
        json.dumps(
            {
                "updated_at": stale_heartbeat_time,
                "status": "preparing",
                "message": "Residualization state is ready; loading adapters and cached eval features.",
                "progress": {
                    "units_completed": 0,
                    "units_total": 5,
                    "unit_label": "evaluation_units",
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoints_dir / "latest.json").write_text(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoints_dir / "unit_0000001"),
                "state_path": str(checkpoints_dir / "unit_0000001" / "state.json"),
                "runtime_state_path": str(checkpoints_dir / "unit_0000001" / "runtime_state.json"),
                "units_completed": 1,
                "updated_at": datetime.now(timezone.utc).timestamp(),
            }
        ),
        encoding="utf-8",
    )

    state = evaluate_step_state(
        M4StepSpec(key="residualize_q", label="Residualize Q", config_path=config_path),
        repo_root=tmp_path,
    )

    assert state.status == "partial"
    assert state.progress_completed == 1
    assert state.progress_total == 5
    assert state.progress_unit == "evaluation_units"
    assert state.percent_complete == 20.0
    assert state.latest_checkpoint == checkpoints_dir / "latest.json"
    assert state.updated_at != stale_heartbeat_time


def test_run_m4_pipeline_defaults_to_reduced_scope_order(monkeypatch, tmp_path: Path) -> None:
    residualize_c_config = tmp_path / "configs" / "analysis" / "full_residuals_C.yaml"
    residualize_q_config = tmp_path / "configs" / "analysis" / "full_residuals_Q.yaml"

    for path, experiment_name in [
        (residualize_c_config, "full_residualize_c"),
        (residualize_q_config, "full_residualize_q"),
    ]:
        _write_yaml(
            path,
            "\n".join(
                [
                    f"experiment_name: {experiment_name}",
                    "milestone: M4",
                    "command: overlay_algebra.analysis.residualize",
                    "seed: 17",
                ]
            ),
        )

    monkeypatch.setattr(
        m4_module,
        "build_default_m4_steps",
        lambda repo_root: [
            M4StepSpec(key="residualize_c", label="Residualize C", config_path=residualize_c_config),
            M4StepSpec(key="residualize_q", label="Residualize Q", config_path=residualize_q_config),
        ],
    )

    executed_commands: list[list[str]] = []

    def _runner(command: list[str], **_: object) -> None:
        executed_commands.append(command)

    summary = run_m4_pipeline(repo_root=tmp_path, runner=_runner)

    assert summary["status_scope"] == "jq_first_reduced_scope"
    assert summary["executed_steps"] == ["residualize_q"]
    assert executed_commands[0][-2:] == ["--config", str(residualize_q_config)]


def test_watch_m4_pipeline_waits_for_partial_nonresumable_then_runs(monkeypatch, tmp_path: Path) -> None:
    summaries = iter(
        [
            {
                "overall_status": "resume_available",
                "completed_steps": 7,
                "total_steps": 22,
                "paper_completed_steps": 7,
                "paper_total_steps": 17,
                "current_step": None,
                "next_step": {
                    "key": "prompt_only_j_eval",
                    "label": "Eval prompt-only J",
                    "status": "partial",
                },
                "steps": [
                    {
                        "key": "prompt_only_j_eval",
                        "status": "partial",
                        "resumable": False,
                    }
                ],
            },
            {
                "overall_status": "ready",
                "completed_steps": 7,
                "total_steps": 22,
                "paper_completed_steps": 7,
                "paper_total_steps": 17,
                "current_step": None,
                "next_step": {
                    "key": "direct_j_eval",
                    "label": "Eval direct J",
                    "status": "pending",
                },
                "steps": [
                    {
                        "key": "direct_j_eval",
                        "status": "pending",
                        "resumable": False,
                    }
                ],
            },
        ]
    )
    slept: list[float] = []
    invoked: list[bool] = []

    monkeypatch.setattr(m4_module, "collect_m4_status", lambda **_: next(summaries))

    def _run_pipeline(**_: object) -> dict[str, object]:
        invoked.append(True)
        return {
            "status": "completed",
            "status_scope": "jq_first_reduced_scope",
            "next_step": None,
            "completed_steps": 17,
            "total_steps": 22,
            "paper_completed_steps": 17,
            "paper_total_steps": 17,
        }

    monkeypatch.setattr(m4_module, "run_m4_pipeline", _run_pipeline)

    summary = watch_m4_pipeline(repo_root=tmp_path, poll_seconds=5.0, sleep_fn=slept.append)

    assert slept == [5.0]
    assert invoked == [True]
    assert summary["status"] == "completed"


def test_watch_m4_pipeline_blocks_after_partial_timeout(monkeypatch, tmp_path: Path) -> None:
    summary_payload = {
        "overall_status": "resume_available",
        "completed_steps": 7,
        "total_steps": 22,
        "paper_completed_steps": 7,
        "paper_total_steps": 17,
        "current_step": None,
        "next_step": {
            "key": "prompt_only_j_eval",
            "label": "Eval prompt-only J",
            "status": "partial",
        },
        "steps": [
            {
                "key": "prompt_only_j_eval",
                "status": "partial",
                "resumable": False,
            }
        ],
    }
    monotonic_values = iter([0.0, 2.0])
    slept: list[float] = []

    monkeypatch.setattr(m4_module, "collect_m4_status", lambda **_: summary_payload)
    monkeypatch.setattr(m4_module.time, "monotonic", lambda: next(monotonic_values))

    summary = watch_m4_pipeline(
        repo_root=tmp_path,
        poll_seconds=1.0,
        partial_timeout_seconds=1.0,
        sleep_fn=slept.append,
    )

    assert slept == [1.0]
    assert summary["status"] == "blocked"
    assert "prompt_only_j_eval" in summary["message"]
