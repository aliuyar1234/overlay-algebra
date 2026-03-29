from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_builder import build_processed_dataset, load_dataset_build_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the processed SQuAD JCQ dataset.")
    parser.add_argument("--config", required=True, help="Path to the dataset YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config_path = Path(args.config)
    build_config = load_dataset_build_config(config_path)
    summary = build_processed_dataset(build_config, source_path=config_path.resolve())

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        counts = summary["counts"]
        print(
            "Built processed dataset "
            f"at {summary['output_dir']} "
            f"(train={counts['processed_train_examples']}, "
            f"val={counts['processed_val_examples']}, "
            f"test={counts['processed_test_examples']})."
        )
        print(f"Manifest: {summary['manifest_path']}")
        print(f"Split audit: {summary['audit_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
