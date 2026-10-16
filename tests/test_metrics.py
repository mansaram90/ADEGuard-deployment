"""Metrics integration checks without ML artifacts or a Prometheus server."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
import pytest

from src.metrics import RequestMetrics, RequestMetricsMiddleware


@pytest.fixture
def monitored_app():
    app = FastAPI()
    metrics = RequestMetrics()
    app.add_middleware(RequestMetricsMiddleware, metrics=metrics)

    @app.get("/metrics")
    def scrape():
        return metrics.response()

    @app.post("/predict")
    async def predict(request: Request):
        body = await request.json()
        if body.get("reject"):
            raise HTTPException(status_code=422, detail="Invalid input")
        return {"prediction": body["text"]}

    @app.get("/fail")
    def fail():
        raise RuntimeError("fictional error marker")

    @app.get("/reports/{report_id}")
    def report(report_id: str):
        return {"ok": True}

    return app, metrics


def samples(response):
    return [sample for family in text_string_to_metric_families(response.text) for sample in family.samples]


def test_counts_durations_and_scrape_exclusion(monitored_app):
    app, _ = monitored_app
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": "private report marker"}).status_code == 200
        assert client.post("/predict", json={"text": "private report marker", "reject": True}).status_code == 422
        first = client.get("/metrics")
        second = client.get("/metrics")

    assert first.status_code == 200
    assert first.headers["content-type"].startswith("text/plain")
    assert first.text == second.text
    parsed = samples(first)
    counters = [s for s in parsed if s.name == "adeguard_http_requests_total"]
    assert {(s.labels["status_code"], s.value) for s in counters} == {("200", 1), ("422", 1)}
    assert all(s.labels["method"] == "POST" and s.labels["path"] == "/predict" for s in counters)
    counts = [s for s in parsed if s.name == "adeguard_http_request_duration_seconds_count"]
    assert len(counts) == 1 and counts[0].value == 2
    sums = [s for s in parsed if s.name == "adeguard_http_request_duration_seconds_sum"]
    assert len(sums) == 1 and sums[0].value > 0
    buckets = [s for s in parsed if s.name == "adeguard_http_request_duration_seconds_bucket"]
    assert buckets and next(s.value for s in buckets if s.labels["le"] == "+Inf") == 2
    assert "private report marker" not in first.text
    assert 'path="/metrics"' not in first.text


def test_unhandled_error_counted_once_without_swallowing(monitored_app):
    app, metrics = monitored_app
    with TestClient(app) as client:
        with pytest.raises(RuntimeError):
            client.get("/fail")
    assert metrics.registry.get_sample_value(
        "adeguard_http_requests_total", {"method": "GET", "path": "/fail", "status_code": "500"}
    ) == 1
    assert metrics.registry.get_sample_value(
        "adeguard_http_request_duration_seconds_count", {"method": "GET", "path": "/fail"}
    ) == 1


def test_labels_are_bounded_and_exclude_sensitive_values(monitored_app):
    app, _ = monitored_app
    with TestClient(app) as client:
        for report_id in ("private-id-one", "private-id-two"):
            client.get(f"/reports/{report_id}?query=private-query", headers={"X-Request-ID": "private-request-id"})
        client.get("/private-unknown-one")
        client.get("/private-unknown-two")
        client.request("PRIVATE_METHOD", "/private-unknown-three")
        response = client.get("/metrics")

    counters = [s for s in samples(response) if s.name == "adeguard_http_requests_total"]
    assert len(counters) == 3
    assert {s.labels["path"] for s in counters} == {"/reports/{report_id}", "<unmatched>"}
    assert {s.labels["method"] for s in counters} == {"GET", "OTHER"}
    assert all(set(s.labels) == {"method", "path", "status_code"} for s in counters)
    for value in ("private-id-one", "private-id-two", "private-query", "private-request-id", "private-unknown", "PRIVATE_METHOD"):
        assert value not in response.text


def test_registries_are_independent(monitored_app):
    app, _ = monitored_app
    other = RequestMetrics()
    with TestClient(app) as client:
        client.get("/reports/123")
    assert other.registry.get_sample_value(
        "adeguard_http_requests_total", {"method": "GET", "path": "/reports/{report_id}", "status_code": "200"}
    ) is None
