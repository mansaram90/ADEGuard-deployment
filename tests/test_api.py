from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_after_startup():
    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"ready": True}


def test_model_info_endpoint_marks_demo_artifacts():
    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["ner_artifact_type"] == "synthetic_deployment_smoke_test"
    assert body["severity_artifact_type"] == "synthetic_deployment_smoke_test"


def test_predict_endpoint_returns_entities_and_severity():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Patient reported fever after Pfizer booster."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Patient reported fever after Pfizer booster."
    assert set(body["entities"]) == {"ADE", "DRUG"}
    assert set(body["severity"]) == {"label", "confidence", "probabilities"}
    assert body["severity"]["label"] in {"Severe", "Moderate", "Mild"}
