import copy
import json

import pytest

from scripts import check_model_preview as preview
from src.artifact_metadata import verify_artifact
from src.config import PROJECT_ROOT


@pytest.fixture
def responses(monkeypatch):
    info = {"severity_model_path": "/models/severity"}
    for task, path, expected_task in (
        ("severity", "severity_model", "sequence_classification"),
        ("ner", "ner_model", "token_classification"),
    ):
        artifact = verify_artifact(PROJECT_ROOT / path, expected_task=expected_task)
        info.update({f"{task}_{key}": value for key, value in artifact.items()})
    payloads = {
        "/ready": {"ready": True}, "/model-info": info,
        "/predict": {"text": preview.SAMPLE_TEXT, "entities": {"ADE": [], "DRUG": []},
                     "severity": {"label": "Mild", "confidence": 0.6,
                                  "probabilities": {"Severe": 0.1, "Moderate": 0.3, "Mild": 0.6}}},
    }
    calls = []

    def request(url, **kwargs):
        assert url.startswith(preview.PREVIEW_URL)
        path = url.removeprefix(preview.PREVIEW_URL)
        calls.append((path, kwargs))
        return json.dumps(payloads[path]).encode(), {}

    monkeypatch.setattr(preview, "request", request)
    return payloads, calls


def test_preview_checks_identity_and_prediction(responses):
    _, calls = responses
    report = preview.check_preview(PROJECT_ROOT / "severity_model", "rollback")
    assert [path for path, _ in calls] == ["/ready", "/model-info", "/predict"]
    assert calls[-1][1]["method"] == "POST"
    assert json.loads(calls[-1][1]["body"]) == {"text": preview.SAMPLE_TEXT}
    assert report["stage"] == "rollback"
    assert report["candidate_promoted"] is False
    assert report["production_approval"] is False


@pytest.mark.parametrize("change", ["ready", "severity_hash", "ner_hash", "version", "type", "mount", "prediction"])
def test_ready_alone_does_not_pass_a_wrong_model_or_response(responses, change):
    payloads, _ = responses
    if change == "ready":
        payloads["/ready"]["ready"] = False
    elif change == "prediction":
        payloads["/predict"]["severity"]["probabilities"]["Mild"] = 0.2
    else:
        key, value = {
            "severity_hash": ("severity_manifest_sha256", "0" * 64),
            "ner_hash": ("ner_manifest_sha256", "0" * 64),
            "version": ("severity_version", "wrong-version"),
            "type": ("severity_artifact_type", "wrong-type"),
            "mount": ("severity_model_path", "/app/severity_model"),
        }[change]
        payloads["/model-info"][key] = value
    with pytest.raises(ValueError):
        preview.check_preview(PROJECT_ROOT / "severity_model", "candidate")


def test_mutation_during_check_is_rejected(monkeypatch, responses):
    original = preview.verify_artifact
    reads = 0

    def changed(path, expected_task):
        nonlocal reads
        reads += 1
        info = copy.deepcopy(original(path, expected_task=expected_task))
        if reads == 3:
            info["manifest_sha256"] = "0" * 64
        return info

    monkeypatch.setattr(preview, "verify_artifact", changed)
    with pytest.raises(ValueError, match="changed during"):
        preview.check_preview(PROJECT_ROOT / "severity_model", "parent")


def test_failed_cli_writes_no_report(monkeypatch, tmp_path):
    monkeypatch.setattr(preview, "PROJECT_ROOT", tmp_path)

    def failed(*args):
        raise OSError("Cannot reach preview")

    monkeypatch.setattr(preview, "check_preview", failed)
    with pytest.raises(SystemExit) as error:
        preview.main(["--model", "severity_model", "--stage", "parent"])
    assert error.value.code == 1
    assert not (tmp_path / "outputs").exists()


def test_cli_saves_separate_local_reports(monkeypatch, tmp_path):
    monkeypatch.setattr(preview, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(preview, "check_preview", lambda *args: {"model_info": {"severity_version": "demo"}})
    for _ in range(2):
        assert preview.main(["--model", "severity_model", "--stage", "parent"]) == 0
    reports = list((tmp_path / "outputs/model-preview").glob("parent-*.json"))
    assert len(reports) == 2
