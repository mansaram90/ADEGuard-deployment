import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.evaluate_severity import DEFAULT_DATASET, evaluate, main
from src.artifact_metadata import ArtifactValidationError, verify_artifact
from src.config import PROJECT_ROOT
from src.evaluation import classification_metrics, load_synthetic_dataset


def test_metrics_against_hand_calculated_imbalanced_example():
    expected = ["Severe", "Severe", "Severe", "Moderate", "Moderate", "Mild"]
    predicted = ["Severe", "Severe", "Moderate", "Moderate", "Mild", "Mild"]
    metrics = classification_metrics(expected, predicted)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["confusion_matrix"] == {
        "labels": ["Severe", "Moderate", "Mild"], "rows": "actual", "columns": "predicted",
        "counts": [[2, 1, 0], [0, 1, 1], [0, 0, 1]],
    }
    assert metrics["per_class"]["Severe"] == pytest.approx({"precision": 1, "recall": 2/3, "f1": 0.8, "support": 3})
    assert metrics["per_class"]["Moderate"] == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2})
    assert metrics["per_class"]["Mild"] == pytest.approx({"precision": 0.5, "recall": 1, "f1": 2/3, "support": 1})
    assert metrics["macro_f1"] == pytest.approx((0.8 + 0.5 + 2/3) / 3)


def test_missing_predictions_and_absent_classes_have_defined_zero_scores():
    metrics = classification_metrics(["Severe", "Moderate", "Mild"], ["Mild"] * 3)
    assert metrics["macro_f1"] == pytest.approx(1 / 6)
    assert metrics["per_class"]["Severe"]["precision"] == 0
    absent = classification_metrics(["Mild"], ["Mild"])
    assert absent["per_class"]["Severe"]["support"] == 0
    assert absent["macro_f1"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("expected,predicted", [([], []), (["Mild"], []), (["Other"], ["Mild"]), (["Mild"], ["Other"])])
def test_metrics_reject_invalid_inputs(expected, predicted):
    with pytest.raises(ValueError):
        classification_metrics(expected, predicted)


@pytest.mark.parametrize("change", ["empty", "invalid_json", "not_object", "missing_id", "blank_text", "bad_label", "not_synthetic", "duplicate_id", "duplicate_text"])
def test_invalid_fixtures_fail_without_silent_row_drops(tmp_path, change):
    row = {"id": "test-1", "text": "A synthetic example.", "label": "Mild", "synthetic": True}
    if change == "missing_id":
        del row["id"]
    elif change == "blank_text":
        row["text"] = " \t "
    elif change == "bad_label":
        row["label"] = "Other"
    elif change == "not_synthetic":
        row["synthetic"] = False
    content = json.dumps(row) + "\n"
    if change == "empty":
        content = "\n"
    elif change == "invalid_json":
        content = "{"
    elif change == "not_object":
        content = "[]"
    elif change == "duplicate_id":
        content += json.dumps(dict(row, text="A different example."))
    elif change == "duplicate_text":
        content += json.dumps(dict(row, id="test-2", text="  A SYNTHETIC example.  "))
    path = tmp_path / "invalid.jsonl"
    path.write_text(content)
    with pytest.raises(ValueError):
        load_synthetic_dataset(path)


def test_fixture_is_balanced_hashed_and_has_no_exact_training_text_overlap():
    rows, digest = load_synthetic_dataset(DEFAULT_DATASET)
    assert digest == hashlib.sha256(DEFAULT_DATASET.read_bytes()).hexdigest()
    assert {label: sum(row["label"] == label for row in rows) for label in ("Severe", "Moderate", "Mild")} == {"Severe": 4, "Moderate": 4, "Mild": 4}
    normalize = lambda text: " ".join(text.casefold().split())
    with (PROJECT_ROOT / "data/training/dataset_with_entities_and_weaklabels.csv").open(encoding="utf-8", newline="") as stream:
        training = {normalize(row["symptom_combined"]) for row in csv.DictReader(stream)}
    assert not training.intersection(normalize(row["text"]) for row in rows)


def test_cli_writes_traceable_report_using_real_serving_model(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert main(["--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["evaluation_scope"] == "synthetic_pipeline_validation"
    assert report["clinical_validation"] is False
    assert report["quality_threshold_applied"] is False
    assert report["model"]["version"] == "demo-severity-v1"
    assert report["model"]["manifest_sha256"] == verify_artifact(PROJECT_ROOT / "severity_model", expected_task="sequence_classification")["manifest_sha256"]
    assert report["dataset"]["sha256"] == hashlib.sha256(DEFAULT_DATASET.read_bytes()).hexdigest()
    assert len(report["predictions"]) == report["dataset"]["examples"] == 12
    assert report["runtime"]["device"]
    for prediction in report["predictions"]:
        assert "text" not in prediction
        assert 0 <= prediction["confidence"] <= 1
    assert sum(sum(row) for row in report["metrics"]["confusion_matrix"]["counts"]) == 12
    assert "not evidence of clinical performance" in capsys.readouterr().out


def test_existing_report_is_preserved(tmp_path):
    output = tmp_path / "report.json"
    output.write_text("previous report")
    with pytest.raises(SystemExit) as error:
        main(["--output", str(output)])
    assert error.value.code == 1
    assert output.read_text() == "previous report"


def test_invalid_dataset_produces_no_report(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("{}")
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as error:
        main(["--dataset", str(path), "--output", str(output)])
    assert error.value.code == 1
    assert not output.exists()


def test_evaluator_rejects_tampered_model_package(tmp_path):
    target = tmp_path / "model"
    shutil.copytree(PROJECT_ROOT / "severity_model", target)
    (target / "config.json").write_text("changed")
    with pytest.raises(ArtifactValidationError):
        evaluate(DEFAULT_DATASET, target)
