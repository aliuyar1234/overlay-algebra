from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..smoke_pipeline import evaluate_smoke, load_eval_smoke_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the M2 smoke evaluation path.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config = load_eval_smoke_config(Path(args.config))
    summary = evaluate_smoke(config)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        print(
            f"Evaluated smoke adapter for {config.condition} "
            f"with {summary['eval_examples']} example(s)."
        )
        print(f"Predictions: {summary['predictions_path']}")
        print(f"Metrics: {summary['metrics_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
