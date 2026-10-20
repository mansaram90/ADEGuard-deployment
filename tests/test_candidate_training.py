import hashlib
import json

import pytest

from src.artifact_metadata import sha256_file, verify_artifact
from src.candidate_training import train_candidate
from src.config import PROJECT_ROOT
from src.data_preparation import load_prepared_dataset, prepare_dataset
from src.evaluation import classification_metrics

SOURCE = PROJECT_ROOT / "data/training/dataset_with_entities_and_weaklabels.csv"
EVALUATION = PROJECT_ROOT / "evaluation/severity_demo.jsonl"
PARENT = PROJECT_ROOT / "severity_model"


@pytest.fixture
def package(tmp_path):
    target = tmp_path / "data"
    prepare_dataset(SOURCE, EVALUATION, target)
    return target


@pytest.mark.parametrize("change", ["file_bytes", "missing_file", "extra_file", "file_mapping", "class_counts", "seed", "source_count", "scope"])
def test_prepared_loader_rejects_corruption(package, change):
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "file_bytes":
        with (package / "train.jsonl").open("ab") as stream:
            stream.write(b"\n")
    elif change == "missing_file":
        (package / "test.jsonl").unlink()
    elif change == "extra_file":
        (package / "unexpected.txt").write_text("extra")
    elif change == "file_mapping":
        manifest["files"]["../outside"] = manifest["files"].pop("train.jsonl")
    elif change == "class_counts":
        manifest["splits"]["train"]["class_counts"]["Mild"] += 1
    elif change == "seed":
        manifest["seed"] = 43
    elif change == "source_count":
        manifest["source"]["examples"] += 1
    else:
        manifest["scope"] = "clinical"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid prepared dataset"):
        load_prepared_dataset(package, EVALUATION)


def test_loader_binds_protected_fixture(package, tmp_path):
    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(EVALUATION.read_bytes() + b'\n')
    with pytest.raises(ValueError, match="differs from preparation"):
        load_prepared_dataset(package, changed)


def test_loader_rejects_forged_case_id_even_with_updated_hash(package):
    manifest_path = package / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = package / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["id"] = "forged-id"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest["files"][path.name] = sha256_file(path)
    manifest["splits"]["train"]["ids"] = [row["id"] for row in rows]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Case ID"):
        load_prepared_dataset(package, EVALUATION)


@pytest.mark.parametrize("settings", [{"epochs":0}, {"batch_size":0}, {"learning_rate":float("nan")}, {"learning_rate":-1}, {"seed":-1}])
def test_bad_training_settings_fail_before_export(package, tmp_path, settings):
    output = tmp_path / "candidate"
    with pytest.raises(ValueError):
        train_candidate(package, PARENT, EVALUATION, output, "test-v1", **settings)
    assert not output.exists()


def test_training_refuses_existing_or_input_subdirectory(package, tmp_path):
    with pytest.raises(ValueError, match="new candidate directory"):
        train_candidate(package, PARENT, EVALUATION, package / "candidate", "test-v1")
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "marker.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError):
        train_candidate(package, PARENT, EVALUATION, existing, "test-v1")
    assert marker.read_text() == "preserve"


def test_real_training_exports_best_checkpoint_without_scoring_test(package, tmp_path, monkeypatch):
    import torch
    from src.adeguard_inference import SeverityModel

    splits, manifest, identity = load_prepared_dataset(package, EVALUATION)
    assert identity == hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
    forbidden = {row["text"] for row in splits["test"]}
    forbidden.update(json.loads(line)["text"] for line in EVALUATION.read_text().splitlines())
    original_predict = SeverityModel.predict

    def validation_only(self, text):
        assert text not in forbidden, "Reserved examples were scored during selection"
        return original_predict(self, text)

    monkeypatch.setattr(SeverityModel, "predict", validation_only)
    parent_hash = sha256_file(PARENT / "model.safetensors")
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state().clone()
    first = train_candidate(package, PARENT, EVALUATION, tmp_path / "first", "test-trained-v1", epochs=2)
    assert torch.get_num_threads() == previous_threads
    assert torch.equal(torch.get_rng_state(), previous_rng)
    second = train_candidate(package, PARENT, EVALUATION, tmp_path / "second", "test-trained-v1", epochs=2)
    assert sha256_file(PARENT / "model.safetensors") == parent_hash
    first_hash = sha256_file(first["model_directory"] / "model.safetensors")
    assert first_hash != parent_hash
    assert first_hash == sha256_file(second["model_directory"] / "model.safetensors")
    report = first["report"]
    assert report["parent"]["manifest_sha256"] == verify_artifact(PARENT, expected_task="sequence_classification")["manifest_sha256"]
    assert report["dataset"]["manifest_sha256"] == identity
    assert report["dataset"]["split_file_sha256"] == manifest["files"]
    best = max(report["epochs"], key=lambda row: row["validation"]["macro_f1"])
    assert report["best_epoch"] == best["epoch"]
    assert report["selected_validation"] == best["validation"]
    assert report["test_split_scored"] is report["separate_evaluation_fixture_scored"] is report["promoted"] is False
    assert first["artifact"]["artifact_type"] == "synthetic_trained_learning_candidate"
    assert verify_artifact(first["model_directory"], expected_task="sequence_classification") == first["artifact"]
    torch.set_num_threads(1)
    try:
        restored = SeverityModel.load(str(first["model_directory"]))
        actual = classification_metrics([row["label"] for row in splits["validation"]],
                                        [restored.predict(row["text"]).label for row in splits["validation"]])
        assert actual == report["selected_validation"]
    finally:
        torch.set_num_threads(previous_threads)
