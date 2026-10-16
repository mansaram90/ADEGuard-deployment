"""Exercise request logging without loading the ML artifacts."""

from datetime import datetime
import json
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
import pytest

from src.request_logging import RequestLoggingMiddleware, logger


@pytest.fixture
def logged_app(caplog):
    app = FastAPI()
    app.add_middleware(RequestLoggingMiddleware)

    @app.post("/predict")
    async def predict(request: Request):
        body = await request.json()
        if body.get("reject"):
            raise HTTPException(status_code=422, detail="Invalid input")
        return {"prediction": body["text"], "request_id": request.state.request_id}

    @app.get("/reports/{report_id}")
    def report(report_id: str):
        return {"ok": True}

    @app.get("/fail")
    def fail():
        raise RuntimeError("Sensitive exception text must not enter the request log")

    # This logger deliberately does not propagate into other configured handlers.
    logger.addHandler(caplog.handler)
    try:
        yield app
    finally:
        logger.removeHandler(caplog.handler)


def request_records(caplog):
    return [json.loads(record.getMessage()) for record in caplog.records if record.name == logger.name]


@pytest.mark.parametrize("reject,status,level", [(False, 200, "INFO"), (True, 422, "WARNING")])
def test_request_metadata_and_sensitive_data_exclusion(logged_app, caplog, reject, status, level):
    secret = "Fictional private report marker"
    with TestClient(logged_app) as client:
        response = client.post(
            "/predict?report=private-query-marker",
            json={"text": secret, "reject": reject},
            headers={"Authorization": "Bearer private-token-marker", "X-Request-ID": "untrusted-id"},
        )

    records = request_records(caplog)
    assert len(records) == 1
    record = records[0]
    assert record["path"] == "/predict"
    assert record["method"] == "POST"
    assert record["status_code"] == response.status_code == status
    assert record["level"] == level
    assert record["event"] == "request_completed"
    assert record["duration_ms"] >= 0
    assert datetime.fromisoformat(record["timestamp"]).utcoffset().total_seconds() == 0
    assert UUID(record["request_id"]).version == 4
    assert record["request_id"] == response.headers["X-Request-ID"]
    if not reject:
        assert response.json()["request_id"] == record["request_id"]
    serialized = json.dumps(record)
    for value in (secret, "private-query-marker", "private-token-marker", "untrusted-id", "prediction"):
        assert value not in serialized


def test_server_failure_is_logged_and_not_swallowed(logged_app, caplog):
    with TestClient(logged_app) as client:
        with pytest.raises(RuntimeError):
            client.get("/fail")

    record = request_records(caplog)[0]
    assert record["status_code"] == 500
    assert record["level"] == "ERROR"
    assert record["event"] == "request_failed"
    assert record["error_type"] == "RuntimeError"
    assert "Sensitive exception text" not in json.dumps(record)


def test_unhandled_failure_still_returns_server_error(logged_app, caplog):
    with TestClient(logged_app, raise_server_exceptions=False) as client:
        response = client.get("/fail")
    assert response.status_code == 500
    assert len(request_records(caplog)) == 1


def test_unique_ids_and_safe_route_paths(logged_app, caplog):
    with TestClient(logged_app) as client:
        first = client.get("/reports/private-report-marker")
        second = client.get("/unknown-private-path")

    records = request_records(caplog)
    assert len(records) == 2
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    assert records[0]["path"] == "/reports/{report_id}"
    assert records[1]["path"] == "<unmatched>"
    assert records[1]["status_code"] == 404
    assert "private-report-marker" not in json.dumps(records)
    assert "unknown-private-path" not in json.dumps(records)
