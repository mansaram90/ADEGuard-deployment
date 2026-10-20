"""Prepare versioned splits from the declared synthetic severity training CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from src.config import PROJECT_ROOT
from src.data_preparation import prepare_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/training/dataset_with_entities_and_weaklabels.csv")
    parser.add_argument("--evaluation", type=Path, default=PROJECT_ROOT / "evaluation/severity_demo.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, help="New directory; existing paths are refused")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or PROJECT_ROOT / "outputs/datasets/severity" / timestamp
    try:
        manifest = prepare_dataset(args.source, args.evaluation, output, args.seed)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print("Synthetic learning data only; no training or model changes performed.")
    for name, split in manifest["splits"].items():
        print(f"{name}: {split['examples']} examples; {split['class_counts']}")
    print(f"Source SHA-256: {manifest['source']['sha256']}")
    print(f"Package: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
