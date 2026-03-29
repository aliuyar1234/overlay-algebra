from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_builder import load_dataset_build_config, run_split_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit title leakage across processed dataset splits.")
    parser.add_argument("--config", required=True, help="Path to the dataset YAML config.")
    parser.add_argument("--json", action="store_true", help="Print the audit report as JSON.")
    args = parser.parse_args()

    build_config = load_dataset_build_config(Path(args.config))
    audit = run_split_audit(build_config.output_dir.resolve())
    if not audit["passed"]:
        raise SystemExit(f"Title overlap audit failed: {audit['overlaps']}")

    if args.json:
        print(json.dumps(audit, ensure_ascii=False, sort_keys=True))
    else:
        print(
            "Title overlap audit passed "
            f"for {audit['output_dir']} "
            f"(train_titles={audit['title_counts']['train']}, "
            f"val_titles={audit['title_counts']['val']}, "
            f"test_titles={audit['title_counts']['test']})."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
