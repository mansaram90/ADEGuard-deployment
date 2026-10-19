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

## Optional Local Prometheus Collector

The API's `/metrics` endpoint exposes current process measurements. Prometheus
periodically scrapes that endpoint and stores samples, enabling queries across
time. The optional `monitoring` profile adds a collector without changing the
default two-service application or existing CI startup command.

From the project root, with Docker Desktop running and ports 8000, 8080, and
9090 free:

```powershell
docker compose --profile monitoring config --quiet
docker compose --profile monitoring run --rm --no-deps --entrypoint promtool prometheus check config /etc/prometheus/prometheus.yml
docker compose --profile monitoring up --build -d --wait --wait-timeout 240
docker compose --profile monitoring ps
```

`promtool` validates the mounted configuration, including its schema. `up`
starts the API/frontend and waits for API readiness before starting Prometheus.
An ordinary `docker compose up --build` does not enable monitoring unless the
profile is enabled separately (for example through `COMPOSE_PROFILES`).

Open http://127.0.0.1:9090. Check the Targets page (typically
http://127.0.0.1:9090/targets) for job `adeguard-api` to be UP. Collector readiness
at http://127.0.0.1:9090/-/ready and a healthy container do not by themselves
prove successful API scraping: the target must also be UP.

`monitoring/prometheus.yml` scrapes `api:8000/metrics` every 15 seconds, with a
five-second scrape timeout. `api` is Compose's internal DNS name. A collector
target of `localhost:8000` would point at the Prometheus container, not FastAPI.
Host-published port 8000 is for host/browser access and is not needed for this
container-to-container connection.

Generate prediction traffic using the frontend or:

```powershell
python scripts/smoke_test.py
```

Wait at least 15 seconds for the next scrape. In the Prometheus query interface,
try these PromQL expressions:

```promql
up{job="adeguard-api"}
```

Value 1 means the latest scrape succeeded; 0 means it failed.

```promql
adeguard_http_requests_total{job="adeguard-api",path="/predict"}
```

This shows sampled request counters split by method/status. Missing prediction
series can simply mean no predictions have been made and scraped yet.

```promql
sum(rate(adeguard_http_requests_total{job="adeguard-api",method="POST",path="/predict"}[5m]))
sum(rate(adeguard_http_requests_total{job="adeguard-api",method="POST",path="/predict",status_code=~"5.."}[5m]))
```

These are prediction requests per second and server-error requests per second.
Use `status_code=~"4.."` separately for client/input rejections. An absent
server-error series is not a collector failure and may produce no query result
rather than an explicit zero.

```promql
sum(rate(adeguard_http_request_duration_seconds_sum{job="adeguard-api",method="POST",path="/predict"}[5m]))
/
sum(rate(adeguard_http_request_duration_seconds_count{job="adeguard-api",method="POST",path="/predict"}[5m]))
```

This estimates average prediction request duration in seconds over the window.
Rate-based queries need at least two samples and traffic in the selected
window to produce useful values. Zero observations can yield NaN for the
average. Low demo traffic is not a reliable latency benchmark.

### Persistence and cleanup

The fixed image tag `prom/prometheus:v3.14.0` identifies the collector version;
it is not an immutable digest pin. A named volume, `prometheus-data`, stores
samples at `/prometheus`, scoped to the Compose project name. Configuration is
mounted read-only. Retention is configured for 24 hours or 512 MB of retained
blocks, whichever limit is reached first. Retention is not a strict total-disk
quota: active data and the write-ahead log also consume storage.

```powershell
docker compose --profile monitoring restart prometheus
docker compose --profile monitoring logs --tail 100 prometheus
docker compose --profile monitoring down --remove-orphans
```

Restart or normal `down` retains the named volume. Reusing the same Compose
project reconnects it, so previously stored samples remain within retention.
Changing the project name uses a different volume/history. API counters still
reset when the API process restarts; collector persistence does not make the
application counters persistent. Prometheus rate functions account for
observed counter resets.

For disposable validation only, use a distinct project name on every command:

```powershell
docker compose -p adeguard-prometheus-check --profile monitoring up --build -d --wait --wait-timeout 240
docker compose -p adeguard-prometheus-check --profile monitoring down --remove-orphans --volumes
```

The final command deletes that test project's collector history as well as its
containers/network. Use `--volumes` only when intentionally discarding those
samples; omit it to preserve normal learning history. A separate project name
does not avoid host-port conflicts. Existing application smoke tests do not
verify collector targets, historical queries, or persistence; these are
separate monitoring validation checks in this phase.

Prometheus has no authentication in this local setup. Port 9090 is bound to
loopback. The following Grafana service adds visualization; alert rules,
Alertmanager, and cloud resources are not included.

## Provisioned Local Grafana Dashboard

The `monitoring` profile also includes Grafana on http://127.0.0.1:3000.
Start or update your existing stack from the project root:

```powershell
docker compose --profile monitoring up --build -d --wait --wait-timeout 240
```

This command adds Grafana to a stack already running under the same Compose
project name. API/frontend/Prometheus retain their existing ports; Grafana
publishes port 3000 on loopback. Default application startup still includes
only API and frontend unless the monitoring profile is enabled.

On a fresh Grafana data volume, sign in with username `admin` and password
`admin`, then follow Grafana's password-change prompt. Keep your chosen password
out of tracked files. The named `grafana-data` volume retains the account and
password across recreation; initial defaults do not reset an existing account.
Anonymous access and sign-up are disabled. This local setup is not a complete
public-hosting access/security design.

Open **Dashboards -> ADEGuard -> ADEGuard API Overview**, or:

```text
http://127.0.0.1:3000/d/adeguard-api-overview
```

### Provisioning: dashboards and data sources as files

The files below are read-only mounts and the source of truth:

| File | Role |
| --- | --- |
| `monitoring/grafana/provisioning/datasources/prometheus.yml` | Register the built-in Prometheus data source with stable UID `adeguard-prometheus` |
| `monitoring/grafana/provisioning/dashboards/adeguard.yml` | Load dashboard files into the ADEGuard folder |
| `monitoring/grafana/dashboards/adeguard-api.json` | Define seven panels, PromQL queries, units, layout, default time range, and refresh |

Grafana queries `http://prometheus:9090` through the internal Compose network.
Its built-in Prometheus plugin is used; no extra API/JSON plugin is required.
Prometheus collects timestamped FastAPI samples; Grafana queries that stored
history. A direct fetch of FastAPI's `/metrics` snapshot would not provide
history needed for rate and histogram-window calculations.

The provisioned dashboard refreshes every 15 seconds and initially shows the
last 30 minutes. UI edits cannot be saved over it. Change the JSON definition
and let file provisioning reload it, or recreate Grafana after changing its
provisioned data source. Stable UIDs link panels to the source reproducibly.

### Panels and interpretation

| Panel | Meaning |
| --- | --- |
| API scrape status | UP/1 or DOWN/0 from Prometheus's latest API scrape; not model accuracy |
| Successful predictions | Current sampled cumulative POST `/predict` status-200 counter; resets on API restart |
| Rejected predictions | Current sampled cumulative prediction 4xx counter, including invalid input |
| Prediction request rate | Requests per second across prediction statuses over a five-minute window |
| Prediction server-error rate | Prediction 5xx responses per second over five minutes |
| Average prediction request duration | Windowed duration sum divided by observation count; all prediction statuses |
| Approximate p95 prediction duration | Bucket-based estimate of the 95th percentile across prediction statuses |

The two counter cards are instant/current values, not totals restricted to the
selected dashboard time range. Rate/duration charts use five-minute windows
at each plotted time. Sparse traffic, fewer than two scrapes, absent error
series, or no recent observations can give empty panels/NaN. Missing data is
not automatically replaced by zero. A UP status does not guarantee every
other series exists. Low-volume demo timing is not a performance benchmark.

Generate fictional requests in the browser at port 8080. To see a rejection,
submit text exceeding the model's token capacity; the UI rejects blank text
before sending it, so use the API if testing whitespace rejection. Wait for
the 15-second scrape/refresh cycle and a few samples. A server-error panel can
remain empty when no 5xx response has been observed; do not break the model
service just to fill that panel.

### Validation and operations

```powershell
docker compose --profile monitoring ps
docker compose --profile monitoring logs --tail 100 grafana
Invoke-RestMethod http://127.0.0.1:3000/api/health
```

Health checks establish Grafana's own readiness. Connection health and actual
dashboard queries must also succeed. The Grafana service waits for Prometheus
readiness, but successful target scraping remains a separate condition.

Temporary validation uses a distinct project and a temporary Compose override
for API port 18000, Prometheus port 19090, and Grafana port 13000. Only those
three services are started, without a frontend container. This tests Grafana
without conflicting with an existing learning stack on the normal ports.
Compose's `!override` port replacement requires a compatible recent Compose
version; the installed Compose supports it. The override is a test artifact,
not a tracked application configuration.

Normal shutdown preserves both monitoring volumes:

```powershell
docker compose --profile monitoring down --remove-orphans
```

`down --volumes` intentionally deletes both Prometheus history and Grafana
account/database state for that project. Avoid it for learning data you want
to retain. Dashboard/data-source definitions remain in repository files even
if volumes are discarded. The fixed Grafana image tag is `13.2.4`; it is not
a digest pin. Grafana does not replace Prometheus storage or the separate
JSON application logs. This step adds no alerts, model evaluation, or cloud
deployment. Existing CI does not start the optional monitoring profile.

## First Alert: Sustained API Scrape Failure

`monitoring/rules/adeguard-alerts.yml` defines `ADEGuardAPIScrapeFailed`:

```yaml
expr: up{job="adeguard-api"} == 0
for: 2m
```

The comparison selects configured API targets whose latest scrape failed.
It does not prove every API operation is unavailable: a metrics-endpoint or
network failure can also cause it. The `for` clause requires the condition to
remain true across rule evaluations for two minutes before firing. Two minutes
is a learning-environment choice, not a universal operational standard.

Prometheus independently scrapes every 15 seconds and evaluates rules every
15 seconds. An outage first needs a failed scrape and then an evaluation.
The pending timer begins at that qualifying evaluation, not when you stop the
container. Actual detection includes scrape, timeout, and evaluation timing.

| Rule state | Meaning |
| --- | --- |
| Inactive | The expression currently selects no failing API target |
| Pending | A failure is selected, but the two-minute continuous wait has not elapsed |
| Firing | Failure remains selected at an evaluation at least two minutes after becoming pending |

Recovery clears the alert at a subsequent successful scrape/evaluation; no
extra `keep_firing_for` interval is configured. A brief recovery also resets
the pending timer if another failure occurs later.

`severity: warning` is a label for classification, not notification delivery.
Annotations provide a summary and instance-specific troubleshooting message.
The rule file is mounted read-only and loaded by `rule_files` in the collector
configuration. No Alertmanager or notification receiver is configured, and no
email, Slack message, or other external notification will be sent.

### Validate before loading

From the project root, run each command separately:

```powershell
docker compose --profile monitoring config --quiet
docker compose --profile monitoring run --rm --no-deps --entrypoint promtool prometheus check config /etc/prometheus/prometheus.yml
docker compose --profile monitoring run --rm --no-deps --volume ./monitoring/tests:/etc/prometheus/tests:ro --entrypoint promtool prometheus test rules /etc/prometheus/tests/alerts.test.yml
```

The first command checks Compose structure. The second validates Prometheus
configuration and the referenced rule. The third uses simulated samples and
time to test six scenarios without waiting minutes or stopping a live service:
healthy, brief outage, sustained outage/recovery, reset of pending timer,
missing series, and an unrelated job failing. These checks are not currently
included in the existing application CI job.

Start/update the normal monitoring stack:

```powershell
docker compose --profile monitoring up --build -d --wait --wait-timeout 240
```

The new rule mount causes collector recreation when applying this Compose
change. For later content-only rule/config edits, restart Prometheus after
validation because a mounted file change alone does not reload its rules:

```powershell
docker compose --profile monitoring restart prometheus
```

Open http://127.0.0.1:9090/alerts and locate `ADEGuardAPIScrapeFailed`.
This is a Prometheus-managed rule, not a newly created Grafana-managed alert.

### Controlled manual exercise

Only do this when intentionally interrupting your local demo. It stops the
API for that Compose project. Our automated live validation used a separate
test project and alternate host ports to preserve your normal stack.

```powershell
docker compose stop api
```

Keep Prometheus running. Watch `up{job="adeguard-api"}` become 0 and the rule
become pending, then firing after its two-minute wait. Restore the API:

```powershell
docker compose up -d --wait --wait-timeout 240 api
```

Wait for model startup, a successful scrape, and another evaluation. Confirm
`up` is 1 and the alert is inactive. For pending/firing series, query:

```promql
ALERTS{alertname="ADEGuardAPIScrapeFailed"}
```

Those active series disappear when the alert is inactive. The rule remains
listed in the rules interface/API; an empty active-alert list after recovery
is expected. Restore the API even if the exercise fails or is interrupted.

**Limits:** A missing/stale `up` series is not equal to 0. Removing the target
from configuration does not trigger this particular rule. Stopping Prometheus
itself also prevents it from evaluating or reporting alerts. This rule does
not test `/predict` correctness, model quality, notification delivery, or
monitoring-system availability. Additional rules and external monitoring are
separate future work.

## Next Steps

Frontend browser interaction tests and model evaluation are separate future
checks. Cloud deployment remains out of scope until explicitly approved.

Later production-readiness work:

- replace demo artifacts with real validated model artifacts
- add model versioning and artifact provenance
- centralize application logs separately from collected Prometheus metrics
- define operational alerts and validate failure notification behavior
- add security review for public deployment
- add image publishing to a registry
- add cloud deployment only after explicit approval

