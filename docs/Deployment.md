# ADEGuard Deployment Guide

This document describes the current local deployment path for the ADEGuard
FastAPI backend.

The current model artifacts are synthetic demo artifacts created for deployment
smoke testing. They are valid for checking that the API, Docker image, model
loading code, and endpoint contracts work. They are not the original ADEGuard
research models and are not clinically meaningful.

## Current Architecture

```text
Client or frontend
  -> FastAPI backend: src/api.py
  -> shared inference layer: src/adeguard_inference.py
  -> model artifact folders:
       ner_model/
       severity_model/
```

The Streamlit app remains useful as a research/demo interface, but the
production-style serving path is the FastAPI backend.

## Model Artifact Caveat

The folders `ner_model/` and `severity_model/` currently contain tiny
Hugging Face-compatible artifacts generated for smoke testing. They include
metadata files named `ADEGUARD_DEMO_ARTIFACT.json` with the artifact type:

```text
synthetic_deployment_smoke_test
```

These artifacts help validate deployment mechanics:

```text
model files exist -> models load -> API starts -> endpoints return JSON
```

They do not validate model quality, clinical correctness, or the original
ADEGuard research claims. Before real use, replace them with controlled,
validated model artifacts from a reproducible training pipeline or model
registry.

## Local API Setup

From the project root:

```powershell
cd C:\Users\Tanu\Downloads\ADEGuard-main\ADEGuard-main
```

Install dependencies for local development:

```powershell
python -m pip install -r requirements.txt
```

For API-only environments, use:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-api.txt
```

The API uses these default model paths:

```text
ner_model/
severity_model/
```

You can override them with environment variables:

```powershell
$env:ADEGUARD_NER_MODEL_PATH = "C:\path\to\ner_model"
$env:ADEGUARD_SEVERITY_MODEL_PATH = "C:\path\to\severity_model"
```

This is useful because production systems should not depend on one developer's
local filesystem layout. The same code can run locally, in Docker, in CI/CD, or
in cloud with different environment-specific configuration.

## Preflight Check

Run the environment check before starting the app:

```powershell
python src\check_environment.py
```

This verifies:

- required Python packages
- NER model folder exists
- severity model folder exists
- each model folder has Hugging Face-style files such as `config.json`,
  tokenizer files, and model weights

A preflight check is a deployment guardrail. It fails early with clearer
messages instead of waiting for a vague runtime error during app startup.

## Run Tests

Run the test suite:

```powershell
pytest
```

The current tests validate service wiring and response shapes:

- inference classes load and return structured predictions
- `/health` responds
- `/ready` reports the app is ready after startup
- `/model-info` identifies the demo artifact type
- `/predict` returns NER and severity JSON

These tests do not prove model quality. They prove that the serving path is
connected correctly.

## Run the API Locally

Start the FastAPI service:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open the generated API docs:

```text
http://127.0.0.1:8000/docs
```

## Standalone Frontend (Local Demo)

The browser UI lives in `frontend/` and uses plain HTML, CSS, and JavaScript.
It sends JSON to FastAPI's `/predict` endpoint; models remain in the backend.
No Node.js packages or frontend build step are required.

From the project root, start the API in one terminal:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In a second terminal, from the same project root, serve only the frontend folder:

```powershell
python -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Open http://127.0.0.1:8080 in your browser. Select **Use sample**, then
**Analyze text**. Entity lists, a severity label, and probability bars should
appear. These values come from random synthetic models and have no clinical
meaning. Use fictional reports for this demo.

The JavaScript API address is `http://127.0.0.1:8000`. FastAPI permits browser
requests from `http://127.0.0.1:8080` and `http://localhost:8080` using CORS.
An origin includes scheme, hostname, and port. Opening `index.html` directly
or choosing a different frontend port will not match this policy. CORS is not
authentication and does not prevent requests from non-browser clients.

If the UI cannot reach the API, verify `/ready` at port 8000, the two terminal
processes, and the browser URL. Restart the API after code changes. A Docker
backend also needs rebuilding to include the new CORS configuration.

Manual checks: submit sample text, check that empty/whitespace input is
rejected, resize the browser, navigate with the keyboard, and stop the API
before submitting again to verify the connection error message. Editing text
hides previous results so they cannot be mistaken for results for the new text.
Use `Ctrl+C` in each terminal to stop the servers.

The frontend is local-only in this phase. See Docker Compose below to run both
services in containers. Hosting and expanded CI are separate future steps.

## Run Both Services with Docker Compose

Docker Compose describes multiple services in one YAML file. `compose.yaml`
builds the API using the existing root Dockerfile and the frontend using
`frontend/Dockerfile`. Nginx serves the static HTML, CSS, and JavaScript.
Docker Desktop must be running with Linux containers enabled.

Stop the manually started API and frontend servers with `Ctrl+C` in their
respective terminals first. Ports 8000 and 8080 must be free. From the project
root, validate the configuration and start both services:

```powershell
docker compose config --quiet
docker compose up --build
```

`--build` builds the images before starting containers, reusing unchanged
cached layers. Leave this terminal open to see logs. The first build needs
network access to download base images and Python dependencies.

Compose checks the API's `/ready` endpoint and waits for `ready: true` before
starting the frontend. This gives model loading time to finish. The frontend
also has an HTTP health check. In a second terminal, inspect service status:

```powershell
docker compose ps
```

Open http://127.0.0.1:8080 and select **Use sample**, then **Analyze text**.
The API remains available at http://127.0.0.1:8000/docs. Both published ports
are bound to loopback for local use only. Predictions still use the synthetic
demo models and are not clinically meaningful.

Although Compose creates a container network, the JavaScript executes in your
host browser. Its API address therefore remains `http://127.0.0.1:8000`, the
published host port. The name `api` is resolvable between Compose containers,
but is not the API hostname for your browser. The existing local CORS policy
continues to allow the frontend on port 8080.

Useful commands from the project root:

```powershell
docker compose logs --tail 100 api
docker compose logs --tail 100 frontend
docker compose down
```

`down` stops and removes this Compose project's containers and network; it
retains built images and repository files. For background startup instead,
use `docker compose up --build -d --wait`; use `down` when finished.

If a port is already allocated, stop the earlier local server or container
using it. If the API becomes unhealthy, inspect its logs and model paths.
The startup dependency does not continuously restart services or disable the
frontend when the API later fails; the UI still handles connection errors.

## API Checks

### Input validation

The API accepts at most 5,000 characters and rejects whitespace-only text.
There is also a separate token limit: each inference path checks the full
tokenized input, including special tokens, against the smallest of the model's
position capacity, tokenizer limit, and the existing 512-token serving ceiling.
The current tiny BERT models support 128 positions. Word count and character
count are not reliable token counts because tokenizers can split words.

Oversized text returns HTTP 422 with a message giving the token count and
limit. Input is not silently truncated. The frontend displays the validation
message so the user can shorten the report. Other consumers of the shared
inference classes receive `InputValidationError` for oversized or blank text.
This changes the earlier truncation behavior for long input; short valid
prediction behavior is preserved.

In another terminal, check process health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected shape:

```json
{
  "status": "ok"
}
```

Check readiness:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
```

Expected shape:

```json
{
  "ready": true
}
```

Check model metadata:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/model-info
```

Expected fields:

```json
{
  "ner_model_path": "...",
  "severity_model_path": "...",
  "ner_artifact_type": "synthetic_deployment_smoke_test",
  "severity_artifact_type": "synthetic_deployment_smoke_test"
}
```

Send a prediction request:

```powershell
$body = @{
  text = "Patient reported fever after Pfizer booster."
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Expected response shape:

```json
{
  "text": "Patient reported fever after Pfizer booster.",
  "entities": {
    "ADE": ["..."],
    "DRUG": ["..."]
  },
  "severity": {
    "label": "Mild",
    "confidence": 0.337,
    "probabilities": {
      "Severe": 0.332,
      "Moderate": 0.331,
      "Mild": 0.337
    }
  }
}
```

Because the current artifacts are synthetic and randomly initialized, the
predicted labels, confidence values, and extracted entities are not meaningful.
For this phase, the important result is that the endpoint returns valid JSON.

## Docker Build

Build the API image from the project root:

```powershell
docker build -t adeguard-api:demo .
```

The Docker image is API-only. It copies:

- `src/`
- `ner_model/`
- `severity_model/`
- `requirements-api.txt`

It intentionally does not copy notebooks, docs, training data, or the Streamlit
app dependencies. This keeps serving separate from research and training
workflows.

## Docker Run

Run the container:

```powershell
docker run --rm -p 8000:8000 --name adeguard-api adeguard-api:demo
```

Then check the API from another terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/ready
Invoke-RestMethod http://127.0.0.1:8000/model-info
```

Send a prediction:

```powershell
$body = @{
  text = "Patient reported fever after Pfizer booster."
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Stop the container with `Ctrl+C` in the terminal running Docker.

## Docker Health Check

The `Dockerfile` includes a container health check against:

```text
http://127.0.0.1:8000/health
```

The health check answers the question: "Is the web process alive?"

Readiness answers a slightly different question: "Has the application loaded
the models and become ready to serve predictions?"

For production-style systems, both concepts matter. A process can be alive but
not ready if model loading failed or has not completed.

## Troubleshooting

If imports fail, install the relevant requirements file:

```powershell
python -m pip install -r requirements.txt
```

For API-only environments:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-api.txt
```

If model loading fails, run:

```powershell
python src\check_environment.py
```

If Docker build fails during dependency installation, confirm Docker has
network access and retry the build.

If the API starts locally but not in Docker, confirm that the `Dockerfile`
copies both model folders:

```text
COPY ner_model ./ner_model
COPY severity_model ./severity_model
```

If `/predict` returns odd labels or entity spans, remember that this is expected
with the synthetic demo artifacts. The current purpose is deployment validation,
not clinical prediction quality.

## CI Integration Smoke Test

On pushes and pull requests, GitHub Actions first runs backend tests. If they
pass, the Docker job validates Compose, builds both images, and starts the
stack with a 240-second readiness timeout. `scripts/smoke_test.py` uses only
Python's standard library to check the frontend page and assets, API readiness,
browser CORS preflight, and a fictional prediction's response structure.
It checks severity probabilities are finite, between 0 and 1, sum to 1, and
agree with the returned label and confidence. It does not require any specific
predicted entities or severity label from these synthetic models.

To reproduce locally, ensure ports 8000 and 8080 are free, then run each command
from the project root. Stop if validation or building fails:

```powershell
docker compose config --quiet
docker compose build
docker compose up -d --wait --wait-timeout 240
python scripts/smoke_test.py
```

A nonzero exit code means a check failed. Inspect logs before cleanup:

```powershell
docker compose ps --all
docker compose logs --no-color --tail 100
docker compose down --remove-orphans
```

Always run cleanup after testing, including when startup or HTTP checks fail.
CI prints container status and logs on failure and schedules cleanup with
`if: always()`. This cleanup is best effort if the runner is forcibly stopped.
The smoke test does not execute browser JavaScript, check visual layout, or
evaluate model quality. A green local test does not establish that the remote
Actions run passed; check that run after pushing.

## Structured Request Logging

`src/request_logging.py` adds one JSON record per HTTP request to standard
output using the `adeguard.requests` logger. View it with:

```powershell
docker compose logs --tail 100 api
```

Records include a UTC timestamp, level, event, generated request ID, method,
route template, status code, and duration in milliseconds. Successful responses
use INFO, 4xx responses WARNING, and 5xx or unhandled exceptions ERROR.
Unhandled exceptions also record the exception class, never its message or
traceback in this custom log. The middleware re-raises them so FastAPI/Uvicorn
retain their original error-handling behavior.

Request IDs are generated by the server, not copied from client headers.
Handled responses include `X-Request-ID`; CORS exposes it for browser clients.
An unhandled 500 generated by the outer server-error middleware may lack this
header. Its request ID remains in the JSON log. Early CORS preflight rejection
and unknown routes use `<unmatched>` when no route template is available.

The custom records omit report text, predictions, query strings, raw URL paths,
path-parameter values, authorization headers, and client IP addresses. A path
such as `/reports/123` would be recorded as `/reports/{report_id}` if that route
existed. This exclusion applies to this logger; default Uvicorn access logs,
server exception logs, and dependency logs remain separately configured and
have not been audited for sensitive content. The complete log stream is not
guaranteed to be JSON or free of sensitive data.

Duration uses a monotonic clock from middleware entry through completion of
the downstream ASGI call (including streaming/background work if present).
It is not isolated model inference time or the browser's end-to-end latency.
Startup failures before HTTP serving and requests that never reach the API
cannot produce these request records. Logging does not add metrics, dashboards,
alerts, persistent log storage, or distributed tracing.

## Request Metrics

The API exposes Prometheus text-format metrics at
http://127.0.0.1:8000/metrics using `prometheus-client`. Install updated serving
requirements for local runs and rebuild the API image for Docker runs.

```powershell
python -m pip install -r requirements-api.txt
docker compose up --build
```

From a second terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
(Invoke-WebRequest http://127.0.0.1:8000/metrics -UseBasicParsing).Content
```

`adeguard_http_requests_total` counts requests by `method`, route-template
`path`, and `status_code`. For example, a successful prediction increases the
POST `/predict` series with status 200; rejected input increases status 422.
Error counts can be derived from 4xx and 5xx status labels without duplicating
the same information in separate error counters.

`adeguard_http_request_duration_seconds` is a histogram labelled by method and
route. Its `_bucket` series count observations at or below each time threshold;
`_count` gives the number of observations and `_sum` their total seconds.
The histogram includes a `+Inf` bucket. `_sum / _count` yields a cumulative
average for that series when count is nonzero; histogram buckets can support
approximate percentiles once a collector and queries are configured.

Metrics middleware measures elapsed downstream ASGI time, including the inner
logging middleware. This is not isolated model inference time or browser
latency. It records failures in `finally`, preserving exceptions. Failures
before any response default to status 500; if a response already started, its
sent status remains the label even if a later exception occurs.

Labels exclude report text, predictions, query strings, raw URL parameter
values, and request IDs. Unknown paths use `<unmatched>` and nonstandard methods
use `OTHER`. Normal health/readiness probes and preflight requests are counted;
filter by path/method when looking specifically at prediction traffic. Requests
to `/metrics` (including its trailing-slash form) do not increment these
measurements. Existing request logging still records scrapes.

An explicit registry contains only the application metrics, not the client's
default Python/process collectors. The current implementation is in-memory
and process-local: counters reset on API restart and are not aggregated across
multiple Uvicorn workers or replicas. Keep the current single-worker setup;
scaling requires a deliberate metrics aggregation design. Labelled series
appear after their first observation, not necessarily on an untouched startup.

No Prometheus server, historical storage, dashboards, alerts, or authentication
has been added. The endpoint is available to callers who can reach this local
API; review access before future public hosting. The HTTP smoke test now checks
that `/metrics` contains a successful prediction counter and duration observation,
so CI exercises the packaged endpoint as well as the in-process tests.
Exposing telemetry does not
measure model accuracy, calibration, drift, or clinical correctness.

## Next Steps

Frontend browser interaction tests and model evaluation are separate future
checks. Cloud deployment remains out of scope until explicitly approved.

Later production-readiness work:

- replace demo artifacts with real validated model artifacts
- add model versioning and artifact provenance
- collect existing request logs and metrics in a monitoring system
- define operational dashboards and alerts
- add security review for public deployment
- add image publishing to a registry
- add cloud deployment only after explicit approval

