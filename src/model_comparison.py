"""Assess a fixed synthetic candidate and its recorded parent on saved test rows."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

from .artifact_metadata import sha256_file, verify_artifact
from .data_preparation import load_prepared_dataset
from .evaluation import classification_metrics, SEVERITY_LABELS


def validate_lineage(parent_info: dict, candidate_info: dict, training: dict,
                     data_manifest: dict, data_identity: str) -> None:
    try:
        if candidate_info["artifact_type"] != "synthetic_trained_learning_candidate":
            raise ValueError("Expected a synthetic trained candidate")
        if parent_info["manifest_sha256"] == candidate_info["manifest_sha256"]:
            raise ValueError("Parent and candidate must be distinct artifacts")
        if type(training["schema_version"]) is not int or training["schema_version"] != 1 or training["scope"] != "synthetic_learning_only" or training["clinical_validation"] is not False:
            raise ValueError("Unsupported training report scope/schema")
        if training["parent"] != parent_info or training["artifact_version"] != candidate_info["version"]:
            raise ValueError("Candidate does not record the supplied parent/version")
        data = training["dataset"]
        if data["manifest_sha256"] != data_identity or data["split_file_sha256"] != data_manifest["files"] or data["source_sha256"] != data_manifest["source"]["sha256"]:
            raise ValueError("Candidate was trained with a different dataset package")
        if data["counts"] != {name: split["examples"] for name, split in data_manifest["splits"].items()}:
            raise ValueError("Training dataset counts differ")
        if training["test_split_scored"] is not False or training["separate_evaluation_fixture_scored"] is not False:
            raise ValueError("Training report must reserve test/fixture examples during selection")
        if training["selection"] != "highest_validation_macro_f1_among_trained_epochs; earliest_epoch_breaks_ties":
            raise ValueError("Unsupported checkpoint selection rule")
        history = training["epochs"]
        if not history or type(training["best_epoch"]) is not int:
            raise ValueError("Missing selected epoch")
        best = max(history, key=lambda epoch: epoch["validation"]["macro_f1"])
        if training["best_epoch"] != best["epoch"] or training["selected_validation"] != best["validation"]:
            raise ValueError("Selected checkpoint metadata is inconsistent")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid candidate lineage: {error}") from error


def compare_models(package: Path, parent: Path, candidate: Path, evaluation_fixture: Path) -> dict:
    splits, data_manifest, data_identity = load_prepared_dataset(package, evaluation_fixture)
    parent_info = verify_artifact(parent, expected_task="sequence_classification")
    candidate_info = verify_artifact(candidate, expected_task="sequence_classification")
    training = json.loads((candidate / "training-report.json").read_text(encoding="utf-8"))
    validate_lineage(parent_info, candidate_info, training, data_manifest, data_identity)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .adeguard_inference import SeverityModel, SEVERITY_ID2LABEL

    rows = splits["test"]
    results = {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for role, path, info in [("parent", parent, parent_info), ("candidate", candidate, candidate_info)]:
            tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True).to("cpu")
            if model.config.id2label != SEVERITY_ID2LABEL:
                raise ValueError(f"Unsupported serving label mapping: {role}")
            model.eval()
            runner = SeverityModel(tokenizer, model, torch.device("cpu"))
            predictions = []
            for row in rows:
                prediction = runner.predict(row["text"])
                if not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1:
                    raise ValueError(f"Invalid confidence from {role} on case {row['id']}")
                predictions.append({"id": row["id"], "expected": row["label"], "predicted": prediction.label,
                                    "confidence": prediction.confidence, "correct": prediction.label == row["label"]})
            metrics = classification_metrics([row["label"] for row in rows], [row["predicted"] for row in predictions])
            results[role] = {"artifact": info, "metrics": metrics, "predictions": predictions,
                             "prediction_counts": {label: sum(row["predicted"] == label for row in predictions) for label in SEVERITY_LABELS},
                             "misclassified_ids": [row["id"] for row in predictions if not row["correct"]]}
    finally:
        torch.set_num_threads(previous_threads)
    parent_metrics, candidate_metrics = results["parent"]["metrics"], results["candidate"]["metrics"]
    paired = {"candidate_only_correct": [], "parent_only_correct": [], "both_correct": [], "both_wrong": []}
    for old, new in zip(results["parent"]["predictions"], results["candidate"]["predictions"]):
        if old["correct"] and new["correct"]:
            category = "both_correct"
        elif old["correct"]:
            category = "parent_only_correct"
        elif new["correct"]:
            category = "candidate_only_correct"
        else:
            category = "both_wrong"
        paired[category].append(old["id"])
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic_reserved_test_comparison", "clinical_validation": False,
        "checkpoint_reselected": False, "quality_gate_applied": False, "candidate_promoted": False,
        "selection": {"best_epoch": training["best_epoch"], "rule": training["selection"],
                      "training_report_sha256": sha256_file(candidate / "training-report.json")},
        "dataset": {"manifest_sha256": data_identity, "source_sha256": data_manifest["source"]["sha256"],
                    "split": "test", "test_file_sha256": data_manifest["files"]["test.jsonl"], "examples": len(rows)},
        "runtime": {"python": platform.python_version(), "torch": version("torch"), "transformers": version("transformers"), "device": "cpu", "threads": 1},
        "implementation_sha256": {name: sha256_file(Path(__file__).parent / name)
                                  for name in ["model_comparison.py", "data_preparation.py", "adeguard_inference.py", "evaluation.py"]},
        "models": results,
        "candidate_minus_parent": {"accuracy": candidate_metrics["accuracy"] - parent_metrics["accuracy"],
                                   "macro_f1": candidate_metrics["macro_f1"] - parent_metrics["macro_f1"],
                                   "recall_by_class": {label: candidate_metrics["per_class"][label]["recall"] - parent_metrics["per_class"][label]["recall"] for label in SEVERITY_LABELS}},
        "paired_outcomes": paired,
    }
