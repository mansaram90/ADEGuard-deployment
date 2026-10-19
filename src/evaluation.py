"""Small, dependency-free classification metrics and synthetic fixture loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SEVERITY_LABELS = ("Severe", "Moderate", "Mild")


def classification_metrics(expected: list[str], predicted: list[str]) -> dict:
    """Rows are actual classes, columns predictions; undefined scores are zero."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected nonempty, equally sized label lists")
    if any(label not in SEVERITY_LABELS for label in expected + predicted):
        raise ValueError("Labels must be Severe, Moderate, or Mild")
    indices = {label: index for index, label in enumerate(SEVERITY_LABELS)}
    matrix = [[0] * len(indices) for _ in indices]
    for actual, prediction in zip(expected, predicted):
        matrix[indices[actual]][indices[prediction]] += 1
    per_class = {}
    for label, index in indices.items():
        true_positive = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "accuracy": sum(matrix[i][i] for i in range(len(indices))) / len(expected),
        "macro_f1": sum(scores["f1"] for scores in per_class.values()) / len(indices),
        "per_class": per_class,
        "confusion_matrix": {"labels": list(SEVERITY_LABELS), "rows": "actual", "columns": "predicted", "counts": matrix},
        "zero_division": 0,
    }


def load_synthetic_dataset(path: Path) -> tuple[list[dict], str]:
    """Hash the exact bytes parsed; fail rather than skip malformed examples."""
    raw = path.read_bytes()
    rows = []
    ids = set()
    texts = set()
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on dataset line {number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Dataset line {number} must be an object")
        case_id, text, label = row.get("id"), row.get("text"), row.get("label")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError(f"Dataset line {number} requires a unique nonempty id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Dataset line {number} requires nonempty text")
        normalized = " ".join(text.casefold().split())
        if normalized in texts:
            raise ValueError(f"Duplicate evaluation text on line {number}")
        if not isinstance(label, str) or label not in SEVERITY_LABELS:
            raise ValueError(f"Invalid severity label on dataset line {number}")
        if row.get("synthetic") is not True:
            raise ValueError(f"Dataset line {number} must explicitly declare synthetic: true")
        ids.add(case_id)
        texts.add(normalized)
        rows.append({"id": case_id, "text": text, "label": label})
    if not rows:
        raise ValueError("Evaluation dataset is empty")
    return rows, hashlib.sha256(raw).hexdigest()
