"""Verify an isolated synthetic preview transition; never activate or approve models."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.artifact_metadata import verify_artifact
from src.config import PROJECT_ROOT
from scripts.smoke_test import SAMPLE_TEXT, request, require, validate_prediction

PREVIEW_URL = "http://127.0.0.1:18000"


def check_preview(model: Path, stage: str) -> dict:
    """Check readiness, exact severity/NER identity and one structural prediction."""
    severity = verify_artifact(model, expected_task="sequence_classification")
    ner = verify_artifact(PROJECT_ROOT / "ner_model", expected_task="token_classification")
    content, _ = request(PREVIEW_URL + "/ready")
    require(json.loads(content).get("ready") is True, "Preview is not ready")
    content, _ = request(PREVIEW_URL + "/model-info")
    info = json.loads(content)
    for task, expected in (("severity", severity), ("ner", ner)):
        for field in ("version", "manifest_sha256", "artifact_type"):
            require(info.get(f"{task}_{field}") == expected[field],
                    f"Preview {task} {field} differs from the expected package")
    require(info.get("severity_model_path") == "/models/severity", "Unexpected preview model mount")
    content, _ = request(PREVIEW_URL + "/predict", method="POST",
                         headers={"Content-Type": "application/json"},
                         body=json.dumps({"text": SAMPLE_TEXT}).encode("utf-8"))
    prediction = json.loads(content)
    validate_prediction(prediction)
    require(verify_artifact(model, expected_task="sequence_classification") == severity,
            "Severity package changed during the check")
    require(verify_artifact(PROJECT_ROOT / "ner_model", expected_task="token_classification") == ner,
            "NER package changed during the check")
    return {"schema_version": 1, "scope": "synthetic_preview_operational_check",
            "checked_at": datetime.now(timezone.utc).isoformat(), "stage": stage,
            "api_url": PREVIEW_URL, "ready": True, "model_info": info,
            "prediction_structure_valid": True, "predicted_label": prediction["severity"]["label"],
            "clinical_validation": False, "production_approval": False,
            "candidate_promoted": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Expected severity package")
    parser.add_argument("--stage", choices=("parent", "candidate", "rollback"), required=True)
    args = parser.parse_args(argv)
    # Keep reports in the ignored operational-output directory; no custom write targets.
    try:
        report = check_preview(args.model, args.stage)
        directory = PROJECT_ROOT / "outputs/model-preview"
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = directory / f"{args.stage}-{stamp}.json"
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Preview verification failed: {error}\n")
    print(f"PASS {args.stage}: {report['model_info']['severity_version']}; report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
