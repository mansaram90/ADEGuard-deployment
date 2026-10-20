"""Record a reasoned synthetic hold decision; no activation or promotion capability."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .artifact_metadata import sha256_file, verify_artifact
from .data_preparation import load_prepared_dataset
from .evaluation import classification_metrics, SEVERITY_LABELS
from .model_comparison import validate_lineage


def build_hold_decision(comparison: Path, package: Path, parent: Path, candidate: Path,
                        evaluation_fixture: Path, reasons: list[str]) -> dict:
    """Validate evidence consistency and retain explicit reviewer rationale."""
    if not reasons or any(not isinstance(reason, str) or not reason.strip() for reason in reasons):
        raise ValueError("At least one nonempty review reason is required")
    reasons = [reason.strip() for reason in reasons]
    if len(set(reasons)) != len(reasons):
        raise ValueError("Review reasons must be distinct")
    try:
        raw = comparison.read_bytes()
        report = json.loads(raw)
        if type(report["schema_version"]) is not int or report["schema_version"] != 1 or report["scope"] != "synthetic_reserved_test_comparison":
            raise ValueError("Unsupported comparison schema/scope")
        for flag in ["clinical_validation", "checkpoint_reselected", "quality_gate_applied", "candidate_promoted"]:
            if report[flag] is not False:
                raise ValueError(f"Unexpected comparison flag: {flag}")
        splits, manifest, identity = load_prepared_dataset(package, evaluation_fixture)
        parent_info = verify_artifact(parent, expected_task="sequence_classification")
        candidate_info = verify_artifact(candidate, expected_task="sequence_classification")
        training_raw = (candidate / "training-report.json").read_bytes()
        training = json.loads(training_raw)
        validate_lineage(parent_info, candidate_info, training, manifest, identity)
        expected_dataset = {"manifest_sha256": identity, "source_sha256": manifest["source"]["sha256"],
                            "split": "test", "test_file_sha256": manifest["files"]["test.jsonl"], "examples": len(splits["test"])}
        if report["dataset"] != expected_dataset:
            raise ValueError("Comparison dataset does not match the supplied package")
        expected_selection = {"best_epoch": training["best_epoch"], "rule": training["selection"],
                              "training_report_sha256": hashlib.sha256(training_raw).hexdigest()}
        if report["selection"] != expected_selection:
            raise ValueError("Comparison selection does not match the frozen training report")
        summaries = {}
        if set(report["models"]) != {"parent", "candidate"}:
            raise ValueError("Expected exactly parent and candidate results")
        for role, info in [("parent", parent_info), ("candidate", candidate_info)]:
            result = report["models"][role]
            if result["artifact"] != info:
                raise ValueError(f"Comparison artifact identity mismatch: {role}")
            predictions = result["predictions"]
            if [row["id"] for row in predictions] != [row["id"] for row in splits["test"]]:
                raise ValueError(f"Comparison test IDs mismatch: {role}")
            for prediction, row in zip(predictions, splits["test"]):
                if prediction["expected"] != row["label"]:
                    raise ValueError(f"Comparison expected labels mismatch: {role}")
                confidence = prediction["confidence"]
                if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError(f"Invalid comparison confidence: {role}")
                if type(prediction["correct"]) is not bool or prediction["correct"] != (prediction["predicted"] == row["label"]):
                    raise ValueError(f"Comparison correctness flag mismatch: {role}")
            metrics = classification_metrics([row["label"] for row in splits["test"]], [row["predicted"] for row in predictions])
            counts = {label: sum(row["predicted"] == label for row in predictions) for label in SEVERITY_LABELS}
            mistakes = [row["id"] for row in predictions if not row["correct"]]
            if result["metrics"] != metrics or result["prediction_counts"] != counts or result["misclassified_ids"] != mistakes:
                raise ValueError(f"Comparison metrics/counts/mistakes mismatch: {role}")
            summaries[role] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                               "recall_by_class": {label: metrics["per_class"][label]["recall"] for label in SEVERITY_LABELS},
                               "prediction_counts": counts}
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as error:
        raise ValueError(f"Decision evidence is invalid: {error}") from error
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic_learning_review", "decision": "hold_candidate",
        "recommendation": "retain_existing_parent_demo_selection", "reasons": reasons,
        "review_method": "agent_assisted_manual_learning_review", "clinical_approval": False,
        "production_approval": False, "quality_gate_applied": False,
        "inference_run": False, "serving_state_changed": False, "candidate_promoted": False,
        "evidence": {"comparison_file": comparison.name, "comparison_sha256": hashlib.sha256(raw).hexdigest(),
                     "dataset": expected_dataset, "selection": expected_selection},
        "models": {"retained_parent": parent_info, "held_candidate": candidate_info},
        "metric_summaries": summaries,
        "recorder_implementation_sha256": sha256_file(Path(__file__)),
    }
