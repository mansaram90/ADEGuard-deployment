from fastapi.testclient import TestClient
import pytest

from src.api import app


@pytest.mark.parametrize("text", [" \n\t ", " ".join(["patient"] * 127), "x" * 5001])
def test_predict_rejects_invalid_input_with_cors(text):
    origin = "http://127.0.0.1:8080"
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": text}, headers={"Origin": origin})

    assert response.status_code == 422
    assert response.headers["access-control-allow-origin"] == origin
    assert response.json()["detail"]


def test_predict_accepts_exact_demo_token_capacity():
    text = " ".join(["patient"] * 126)
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": text})

    assert response.status_code == 200
    assert response.json()["text"] == text


def test_predict_returns_actionable_token_error():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": " ".join(["patient"] * 127)})

    assert response.status_code == 422
    assert "129 tokens" in response.json()["detail"]
    assert "maximum 128" in response.json()["detail"]
    assert "Shorten the text" in response.json()["detail"]


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8080", "http://localhost:8080"])
def test_local_frontend_cors_preflight(origin):
    with TestClient(app) as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        prediction = client.post(
            "/predict",
            headers={"Origin": origin},
            json={"text": "Patient reported fever after Pfizer booster."},
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]
    assert prediction.status_code == 200
    assert prediction.headers["access-control-allow-origin"] == origin


def test_cors_rejects_unapproved_origin():
    with TestClient(app) as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


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
