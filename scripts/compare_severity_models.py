"""Compare a fixed synthetic candidate with its parent on a prepared test split."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.config import C_MODEL_PATH, PROJECT_ROOT, model_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True, help="Exported candidate model directory")
    parser.add_argument("--parent", type=Path, default=Path(C_MODEL_PATH))
    parser.add_argument("--evaluation", type=Path, default=PROJECT_ROOT / "evaluation/severity_demo.jsonl")
    parser.add_argument("--output", type=Path, help="New report file; existing paths are refused")
    args = parser.parse_args(argv)
    from src.model_comparison import compare_models
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or PROJECT_ROOT / "outputs/evaluation" / f"comparison-{timestamp}.json"
    try:
        protected_paths = [args.dataset, args.parent, args.candidate, Path(C_MODEL_PATH), Path(model_path)]
        if output.exists() or any(output.resolve().is_relative_to(path.resolve()) for path in protected_paths):
            raise ValueError("Use a new report file outside dataset and model packages")
        report = compare_models(args.dataset, args.parent, args.candidate, args.evaluation)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Comparison failed: {error}\n")
    print("Synthetic reserved-test assessment only; no checkpoint changes or promotion.")
    print(f"Test examples: {report['dataset']['examples']}; selected epoch remains {report['selection']['best_epoch']}")
    for role, result in report["models"].items():
        metrics = result["metrics"]
        print(f"{role}: accuracy={metrics['accuracy']:.4f}; macro F1={metrics['macro_f1']:.4f}; Severe recall={metrics['per_class']['Severe']['recall']:.4f}")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
