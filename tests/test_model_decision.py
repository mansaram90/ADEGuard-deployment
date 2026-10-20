import copy
import json
import subprocess
import sys

import pytest

from scripts.record_model_decision import main
from src.artifact_metadata import sha256_file
from src.candidate_training import train_candidate
from src.config import PROJECT_ROOT
from src.data_preparation import prepare_dataset
from src.model_comparison import compare_models
from src.model_decision import build_hold_decision

SOURCE = PROJECT_ROOT / "data/training/dataset_with_entities_and_weaklabels.csv"
EVALUATION = PROJECT_ROOT / "evaluation/severity_demo.jsonl"
PARENT = PROJECT_ROOT / "severity_model"


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("decision")
    package = root / "data"
    prepare_dataset(SOURCE, EVALUATION, package)
    candidate = train_candidate(package, PARENT, EVALUATION, root / "run", "decision-test-v1", epochs=1)
    model_directory = candidate["model_directory"]
    report = compare_models(package, PARENT, model_directory, EVALUATION)
    comparison = root / "comparison.json"
    comparison.write_text(json.dumps(report, indent=2))
    return package, model_directory, comparison, report


@pytest.mark.parametrize("change", ["scope", "promoted", "artifact", "dataset", "epoch", "training_hash", "ids", "expected", "correct", "confidence", "metrics", "counts", "mistakes"])
def test_inconsistent_comparison_is_rejected(evidence, tmp_path, change):
    package, candidate, _, original = evidence
    report = copy.deepcopy(original)
    result = report["models"]["candidate"]
    if change == "scope":
        report["scope"] = "production"
    elif change == "promoted":
        report["candidate_promoted"] = True
    elif change == "artifact":
        result["artifact"]["manifest_sha256"] = "0" * 64
    elif change == "dataset":
        report["dataset"]["test_file_sha256"] = "0" * 64
    elif change == "epoch":
        report["selection"]["best_epoch"] = 999
    elif change == "training_hash":
        report["selection"]["training_report_sha256"] = "0" * 64
    elif change == "ids":
        result["predictions"][0]["id"] = "unknown-id"
    elif change == "expected":
        result["predictions"][0]["expected"] = "Unknown"
    elif change == "correct":
        result["predictions"][0]["correct"] = not result["predictions"][0]["correct"]
    elif change == "confidence":
        result["predictions"][0]["confidence"] = float("nan")
    elif change == "metrics":
        result["metrics"]["accuracy"] = 0.99
    elif change == "counts":
        result["prediction_counts"]["Severe"] += 1
    else:
        result["misclassified_ids"] = ["unknown-id"]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Decision evidence is invalid"):
        build_hold_decision(changed, package, PARENT, candidate, EVALUATION, ["Review reason"])


@pytest.mark.parametrize("reasons", [[], [" "], ["same", "same"]])
def test_review_requires_distinct_nonempty_reasons(evidence, reasons):
    package, candidate, comparison, _ = evidence
    with pytest.raises(ValueError):
        build_hold_decision(comparison, package, PARENT, candidate, EVALUATION, reasons)


def test_cli_records_hold_without_loading_torch_or_mutating_inputs(evidence, tmp_path):
    package, candidate, comparison, original = evidence
    paths = [comparison] + [path for directory in [package, candidate, PARENT] for path in directory.iterdir() if path.is_file()]
    before = {path: sha256_file(path) for path in paths}
    output = tmp_path / "decision.json"
    arguments = ["--comparison", str(comparison), "--dataset", str(package), "--candidate", str(candidate),
                 "--reason", "Candidate needs further review", "--output", str(output)]
    script = "import sys; from scripts.record_model_decision import main; assert main(sys.argv[1:]) == 0; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    process = subprocess.run([sys.executable, "-c", script, *arguments], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    decision = json.loads(output.read_text())
    assert decision["decision"] == "hold_candidate"
    assert decision["recommendation"] == "retain_existing_parent_demo_selection"
    assert decision["evidence"]["comparison_sha256"] == sha256_file(comparison)
    assert decision["models"]["held_candidate"] == original["models"]["candidate"]["artifact"]
    assert decision["metric_summaries"]["candidate"]["accuracy"] == original["models"]["candidate"]["metrics"]["accuracy"]
    for flag in ["clinical_approval", "production_approval", "quality_gate_applied", "inference_run", "serving_state_changed", "candidate_promoted"]:
        assert decision[flag] is False
    assert decision["review_method"] == "agent_assisted_manual_learning_review"
    assert {path: sha256_file(path) for path in paths} == before


def test_invalid_evidence_produces_no_decision(evidence, tmp_path):
    package, candidate, _, _ = evidence
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}")
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit) as error:
        main(["--comparison", str(invalid), "--dataset", str(package), "--candidate", str(candidate),
              "--reason", "Explicit reason", "--output", str(output)])
    assert error.value.code == 1
    assert not output.exists()


def test_output_guards_preserve_reports_and_artifacts(evidence, tmp_path):
    package, candidate, comparison, _ = evidence
    existing = tmp_path / "existing.json"
    existing.write_text("preserve")
    for output in [existing, comparison, candidate / "decision.json", package / "decision.json", PARENT / "decision.json"]:
        with pytest.raises(SystemExit) as error:
            main(["--comparison", str(comparison), "--dataset", str(package), "--candidate", str(candidate),
                  "--reason", "Explicit reason", "--output", str(output)])
        assert error.value.code == 1
        if output not in [existing, comparison]:
            assert not output.exists()
    assert existing.read_text() == "preserve"
