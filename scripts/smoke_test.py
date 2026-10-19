"""Check a running ADEGuard stack over HTTP, without ML dependencies.

Run after Compose reports both services healthy. This verifies service wiring,
not browser interactions or the clinical accuracy of the synthetic models.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from urllib.error import URLError
from urllib.request import Request, urlopen


SAMPLE_TEXT = "Patient reported fever after Pfizer booster."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def request(url: str, *, method: str = "GET", headers=None, body=None):
    with urlopen(Request(url, data=body, headers=headers or {}, method=method), timeout=30) as response:
        require(response.status == 200, f"{method} {url}: expected HTTP 200")
        return response.read(), response.headers


def validate_prediction(prediction: dict) -> None:
    require(isinstance(prediction, dict), "Prediction must be a JSON object")
    require(prediction.get("text") == SAMPLE_TEXT, "Prediction did not echo submitted text")
    entities = prediction.get("entities")
    require(isinstance(entities, dict), "Missing entity object")
    for label in ("ADE", "DRUG"):
        values = entities.get(label)
        require(isinstance(values, list), f"{label} entities must be a list")
        require(all(isinstance(value, str) for value in values), f"{label} entities must be strings")

    severity = prediction.get("severity")
    require(isinstance(severity, dict), "Missing severity object")
    probabilities = severity.get("probabilities")
    labels = {"Mild", "Moderate", "Severe"}
    require(isinstance(probabilities, dict) and set(probabilities) == labels, "Unexpected severity labels")
    for label, value in probabilities.items():
        require(
            type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
            f"Invalid probability for {label}",
        )
    require(math.isclose(sum(probabilities.values()), 1, abs_tol=0.001), "Probabilities must sum to 1")
    label = severity.get("label")
    require(label in labels, "Invalid predicted severity label")
    confidence = severity.get("confidence")
    require(
        type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1,
        "Invalid confidence value",
    )
    require(math.isclose(confidence, probabilities[label], abs_tol=0.001), "Confidence differs from label probability")
    require(math.isclose(confidence, max(probabilities.values()), abs_tol=0.001), "Predicted label is not the most probable")


def run_checks(api_url: str, frontend_url: str) -> None:
    for path, expected in (("/", b'id="prediction-form"'), ("/styles.css", b""), ("/app.js", b"")):
        content, _ = request(frontend_url + path)
        require(bool(content) and expected in content, f"Missing or unexpected frontend content: {path}")
    print("PASS: frontend page and assets", flush=True)

    content, _ = request(api_url + "/ready")
    require(json.loads(content).get("ready") is True, "API is not ready")
    print("PASS: API readiness", flush=True)

    content, _ = request(api_url + "/model-info")
    info = json.loads(content)
    for task in ("ner", "severity"):
        require(bool(info.get(task + "_version")), f"Missing {task} artifact version")
        fingerprint = info.get(task + "_manifest_sha256", "")
        require(len(fingerprint) == 64 and all(c in "0123456789abcdef" for c in fingerprint),
                f"Invalid {task} manifest identity")
    print("PASS: loaded model artifact versions and identities", flush=True)

    _, headers = request(
        api_url + "/predict",
        method="OPTIONS",
        headers={
            "Origin": frontend_url,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    require(headers.get("Access-Control-Allow-Origin") == frontend_url, "Preflight origin not allowed")
    require("POST" in headers.get("Access-Control-Allow-Methods", ""), "Preflight must allow POST")
    require("content-type" in headers.get("Access-Control-Allow-Headers", "").lower(), "Preflight must allow JSON content type")
    print("PASS: browser CORS preflight", flush=True)

    content, headers = request(
        api_url + "/predict",
        method="POST",
        headers={"Origin": frontend_url, "Content-Type": "application/json"},
        body=json.dumps({"text": SAMPLE_TEXT}).encode("utf-8"),
    )
    require(headers.get("Access-Control-Allow-Origin") == frontend_url, "Prediction origin not allowed")
    validate_prediction(json.loads(content))
    print("PASS: prediction structure, probabilities, and CORS response", flush=True)

    content, headers = request(api_url + "/metrics")
    require(headers.get("Content-Type", "").startswith("text/plain"), "Metrics must use Prometheus text format")
    lines = content.decode("utf-8").splitlines()
    for metric_name, required_labels in (
        ("adeguard_http_requests_total", ('method="POST"', 'path="/predict"', 'status_code="200"')),
        ("adeguard_http_request_duration_seconds_count", ('method="POST"', 'path="/predict"')),
    ):
        matching = [line for line in lines if line.startswith(metric_name + "{") and all(label in line for label in required_labels)]
        require(bool(matching) and float(matching[0].rsplit(" ", 1)[1]) >= 1, f"Missing prediction measurement: {metric_name}")
    print("PASS: metrics expose successful prediction count and duration observation", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    try:
        run_checks(args.api_url.rstrip("/"), args.frontend_url.rstrip("/"))
    except (URLError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("Smoke test passed. This does not validate model quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
