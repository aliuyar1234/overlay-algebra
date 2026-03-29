from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..smoke_pipeline import load_train_smoke_config, train_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the config-driven LoRA training path.")
    parser.add_argument("--config", required=True, help="Path to a train YAML config.")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume from the latest checkpoint recorded in the target run directory.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        help="Resume from a specific checkpoint directory, latest.json manifest, or trainer_state.pt file.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    if args.resume_latest and args.resume_checkpoint:
        parser.error("--resume-latest and --resume-checkpoint cannot be used together")

    config = load_train_smoke_config(
        Path(args.config),
        resume_latest=args.resume_latest,
        resume_checkpoint_path=args.resume_checkpoint,
    )
    summary = train_smoke(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"Trained smoke adapter for {config.condition} "
            f"at {summary['adapter_dir']} "
            f"using {summary['train_examples']} example(s)."
        )
        print(f"Run manifest: {summary['run_manifest_path']}")
        print(f"Metrics: {summary['metrics_path']}")
        print(f"Heartbeat: {summary['heartbeat_path']}")
        print(f"Latest checkpoint: {summary['latest_checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
