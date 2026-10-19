"""Measure the severity serving path on a synthetic learning fixture.

Run from the project root: python -m scripts.evaluate_severity
No pass/fail quality threshold is applied; poor scores still produce a report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

from src.artifact_metadata import verify_artifact
from src.config import C_MODEL_PATH, PROJECT_ROOT
from src.evaluation import classification_metrics, load_synthetic_dataset

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "severity_demo.jsonl"


def evaluate(dataset_path: Path, model_path: Path) -> dict:
    rows, dataset_hash = load_synthetic_dataset(dataset_path)
    artifact = verify_artifact(model_path, expected_task="sequence_classification")
    # Lazy import keeps --help and metric/fixture tests independent of ML loading.
    from src.adeguard_inference import SeverityModel

    model = SeverityModel.load(str(model_path))
    predictions = []
    for row in rows:
        result = model.predict(row["text"])
        predictions.append({"id": row["id"], "expected": row["label"], "predicted": result.label,
                            "confidence": result.confidence, "correct": result.label == row["label"]})
    metrics = classification_metrics([row["label"] for row in rows], [row["predicted"] for row in predictions])
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "synthetic_pipeline_validation",
        "clinical_validation": False,
        "quality_threshold_applied": False,
        "model": {"path": str(model_path.resolve()), **artifact},
        "dataset": {"path": str(dataset_path.resolve()), "sha256": dataset_hash, "examples": len(rows)},
        "runtime": {"python": platform.python_version(), "torch": version("torch"),
                    "transformers": version("transformers"), "device": str(model.device)},
        "metrics": metrics,
        "predictions": predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=Path(C_MODEL_PATH))
    parser.add_argument("--output", type=Path, help="New report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or PROJECT_ROOT / "outputs" / "evaluation" / f"severity-{timestamp}.json"
    try:
        if output.exists():
            raise ValueError(f"Report already exists: {output}; choose a new output path")
        report = evaluate(args.dataset, args.model)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print("Synthetic evaluation only; not evidence of clinical performance.")
    print(f"Model: {report['model']['version']}; examples: {report['dataset']['examples']}")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}; macro F1: {report['metrics']['macro_f1']:.4f}")
    print(f"Report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
