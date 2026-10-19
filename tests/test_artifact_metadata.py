import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import src.api as api
from src.artifact_metadata import (
    ArtifactValidationError, MANIFEST_NAME, verify_artifact, write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package(tmp_path):
    target = tmp_path / "ner"
    shutil.copytree(ROOT / "ner_model", target)
    return target


@pytest.mark.parametrize("change", ["changed", "missing", "extra", "manifest_missing"])
def test_invalid_packages_fail(package, change):
    if change == "changed":
        with (package / "model.safetensors").open("ab") as stream:
            stream.write(b"modified")
    elif change == "missing":
        (package / "tokenizer.json").unlink()
    elif change == "extra":
        (package / "unexpected.bin").write_bytes(b"extra")
    else:
        (package / MANIFEST_NAME).unlink()
    with pytest.raises(ArtifactValidationError):
        verify_artifact(package, expected_task="token_classification")


@pytest.mark.parametrize("change", ["task", "schema", "version", "hash", "traversal", "json"])
def test_invalid_manifests_fail(package, change):
    path = package / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    if change == "task":
        manifest["task"] = "sequence_classification"
    elif change == "schema":
        manifest["schema_version"] = True
    elif change == "version":
        manifest["version"] = " "
    elif change == "hash":
        manifest["files"]["config.json"] = "invalid"
    elif change == "traversal":
        manifest["files"]["../outside"] = "0" * 64
    path.write_text("{" if change == "json" else json.dumps(manifest))
    with pytest.raises(ArtifactValidationError):
        verify_artifact(package, expected_task="token_classification")


def test_packaging_preserves_files_and_refuses_overwrite(package):
    (package / MANIFEST_NAME).unlink()
    before = {p.name: p.read_bytes() for p in package.iterdir()}
    write_manifest(package, version="demo-ner-v2", task="token_classification")
    info = verify_artifact(package, expected_task="token_classification")
    assert info["version"] == "demo-ner-v2"
    assert info["artifact_type"] == "synthetic_deployment_smoke_test"
    assert all((package / name).read_bytes() == data for name, data in before.items())
    with pytest.raises(ArtifactValidationError, match="already exists"):
        write_manifest(package, version="demo-ner-v3", task="token_classification")


def test_manifest_identity_survives_formatting_but_tracks_version(package):
    original = verify_artifact(package, expected_task="token_classification")
    path = package / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps(manifest, sort_keys=True))
    assert verify_artifact(package, expected_task="token_classification") == original
    manifest["version"] = "another-version"
    path.write_text(json.dumps(manifest))
    assert verify_artifact(package, expected_task="token_classification")["manifest_sha256"] != original["manifest_sha256"]


def test_bad_severity_package_prevents_either_model_loading(tmp_path, monkeypatch):
    package = tmp_path / "severity"
    shutil.copytree(ROOT / "severity_model", package)
    (package / "config.json").write_text("changed")
    monkeypatch.setattr(api, "C_MODEL_PATH", str(package))
    calls = []
    monkeypatch.setattr(api.NERModel, "load", lambda path: calls.append(path))
    with pytest.raises(ArtifactValidationError):
        with TestClient(api.app):
            pass
    assert calls == []
    assert not hasattr(api.app.state, "ner_model")


def test_model_info_is_a_startup_snapshot(package, monkeypatch):
    monkeypatch.setattr(api, "model_path", str(package))
    with TestClient(api.app) as client:
        before = client.get("/model-info").json()
        manifest_path = package / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text())
        manifest["version"] = "edited-after-startup"
        manifest_path.write_text(json.dumps(manifest))
        assert client.get("/model-info").json() == before
        assert before["ner_version"] == "demo-ner-v1"
        assert before["severity_version"] == "demo-severity-v1"
    assert not hasattr(api.app.state, "model_info")
