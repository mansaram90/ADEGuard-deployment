import copy
import json
import shutil

import pytest

from scripts.compare_severity_models import main
from src.artifact_metadata import sha256_file, verify_artifact, write_manifest
from src.candidate_training import train_candidate
from src.config import PROJECT_ROOT
from src.data_preparation import load_prepared_dataset, prepare_dataset
from src.model_comparison import compare_models, validate_lineage

SOURCE = PROJECT_ROOT / "data/training/dataset_with_entities_and_weaklabels.csv"
EVALUATION = PROJECT_ROOT / "evaluation/severity_demo.jsonl"
PARENT = PROJECT_ROOT / "severity_model"


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison")
    package = root / "dataset"
    prepare_dataset(SOURCE, EVALUATION, package)
    candidate = train_candidate(package, PARENT, EVALUATION, root / "run", "comparison-test-v1", epochs=2)
    return root, package, candidate


@pytest.mark.parametrize("change", ["parent", "version", "dataset", "split_hash", "counts", "test_scored", "fixture_scored", "selection", "epoch"])
def test_inconsistent_lineage_is_rejected(experiment, change):
    _, package, candidate = experiment
    _, manifest, identity = load_prepared_dataset(package, EVALUATION)
    report = copy.deepcopy(candidate["report"])
    if change == "parent":
        report["parent"]["manifest_sha256"] = "0" * 64
    elif change == "version":
        report["artifact_version"] = "other-version"
    elif change == "dataset":
        report["dataset"]["manifest_sha256"] = "0" * 64
    elif change == "split_hash":
        report["dataset"]["split_file_sha256"]["test.jsonl"] = "0" * 64
    elif change == "counts":
        report["dataset"]["counts"]["test"] += 1
    elif change == "test_scored":
        report["test_split_scored"] = True
    elif change == "fixture_scored":
        report["separate_evaluation_fixture_scored"] = True
    elif change == "selection":
        report["selection"] = "test_f1"
    else:
        report["best_epoch"] = 100
    with pytest.raises(ValueError, match="Invalid candidate lineage"):
        validate_lineage(verify_artifact(PARENT, expected_task="sequence_classification"), candidate["artifact"], report, manifest, identity)


def test_fixed_comparison_scores_only_same_test_rows_and_preserves_inputs(experiment, tmp_path, monkeypatch):
    import torch
    from src.adeguard_inference import SeverityModel

    _, package, candidate = experiment
    model_directory = candidate["model_directory"]
    snapshots = {path: sha256_file(path) for directory in [package, PARENT, model_directory] for path in directory.iterdir() if path.is_file()}
    splits, _, _ = load_prepared_dataset(package, EVALUATION)
    expected_texts = [row["text"] for row in splits["test"]]
    original = SeverityModel.predict
    calls = []

    def test_only(self, text):
        assert text in expected_texts
        calls.append(text)
        return original(self, text)

    monkeypatch.setattr(SeverityModel, "predict", test_only)
    previous_threads = torch.get_num_threads()
    output = tmp_path / "comparison.json"
    assert main(["--dataset", str(package), "--candidate", str(model_directory), "--output", str(output)]) == 0
    assert torch.get_num_threads() == previous_threads
    assert calls == expected_texts + expected_texts
    report = json.loads(output.read_text())
    assert report["dataset"]["split"] == "test"
    assert report["dataset"]["examples"] == 6
    assert report["selection"]["best_epoch"] == candidate["report"]["best_epoch"]
    assert report["checkpoint_reselected"] is report["quality_gate_applied"] is report["candidate_promoted"] is False
    for role in ["parent", "candidate"]:
        result = report["models"][role]
        assert [row["id"] for row in result["predictions"]] == [row["id"] for row in splits["test"]]
        assert all("text" not in row for row in result["predictions"])
        assert sum(result["prediction_counts"].values()) == 6
        assert sum(sum(row) for row in result["metrics"]["confusion_matrix"]["counts"]) == 6
        assert result["metrics"]["accuracy"] == sum(row["correct"] for row in result["predictions"]) / 6
    assert report["candidate_minus_parent"]["macro_f1"] == pytest.approx(report["models"]["candidate"]["metrics"]["macro_f1"] - report["models"]["parent"]["metrics"]["macro_f1"])
    paired_ids = [case_id for ids in report["paired_outcomes"].values() for case_id in ids]
    assert len(paired_ids) == len(set(paired_ids)) == 6
    assert set(paired_ids) == {row["id"] for row in splits["test"]}
    assert {path: sha256_file(path) for path in snapshots} == snapshots


def test_wrong_package_and_parent_fail_before_scoring(experiment, tmp_path, monkeypatch):
    from src.adeguard_inference import SeverityModel

    def unexpected(*args, **kwargs):
        pytest.fail("Lineage rejection must happen before scoring")

    monkeypatch.setattr(SeverityModel, "predict", unexpected)
    _, package, candidate = experiment
    different = tmp_path / "different"
    prepare_dataset(SOURCE, EVALUATION, different, seed=43)
    with pytest.raises(ValueError, match="different dataset"):
        compare_models(different, PARENT, candidate["model_directory"], EVALUATION)
    other_parent = tmp_path / "parent"
    shutil.copytree(PARENT, other_parent)
    (other_parent / "artifact-manifest.json").unlink()
    write_manifest(other_parent, version="other-parent", task="sequence_classification")
    with pytest.raises(ValueError, match="supplied parent"):
        compare_models(package, other_parent, candidate["model_directory"], EVALUATION)


def test_report_guards_preserve_existing_file_and_model_packages(experiment, tmp_path):
    _, package, candidate = experiment
    existing = tmp_path / "existing.json"
    existing.write_text("preserve")
    for output in [existing, candidate["model_directory"] / "assessment.json", PARENT / "assessment.json", package / "assessment.json"]:
        with pytest.raises(SystemExit) as error:
            main(["--dataset", str(package), "--candidate", str(candidate["model_directory"]), "--output", str(output)])
        assert error.value.code == 1
        if output != existing:
            assert not output.exists()
    assert existing.read_text() == "preserve"
