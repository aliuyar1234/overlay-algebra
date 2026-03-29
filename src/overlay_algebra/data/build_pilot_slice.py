from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pilot_slice import build_pilot_slice, load_pilot_slice_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic pilot slice from a processed dataset.")
    parser.add_argument("--config", required=True, help="Path to the pilot-slice YAML config.")
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    config_path = Path(args.config)
    build_config = load_pilot_slice_config(config_path)
    summary = build_pilot_slice(build_config, source_path=config_path.resolve())

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    else:
        counts = summary["counts"]
        print(
            "Built pilot slice "
            f"at {summary['output_dir']} "
            f"(train={counts['pilot_train_examples']}, "
            f"val={counts['pilot_val_examples']}, "
            f"test={counts['pilot_test_examples']})."
        )
        print(f"Manifest: {summary['manifest_path']}")
        print(f"Split audit: {summary['audit_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
