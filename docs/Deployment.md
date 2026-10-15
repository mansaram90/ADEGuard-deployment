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

The frontend is local-only in this phase. Frontend containers, hosting, and
expanded CI are separate future steps.

## API Checks

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

## Next Steps

Recommended next phase: CI/CD with GitHub Actions.

Initial CI/CD goals:

- run `pytest` on push and pull request
- optionally build the Docker image
- keep cloud deployment out of scope until explicitly approved

Later production-readiness work:

- replace demo artifacts with real validated model artifacts
- add model versioning and artifact provenance
- add request logging and monitoring
- add structured error handling and observability
- add security review for public deployment
- add image publishing to a registry
- add cloud deployment only after explicit approval

