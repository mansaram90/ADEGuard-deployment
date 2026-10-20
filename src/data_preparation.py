"""Validate and reproducibly split the synthetic severity training CSV."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re

from .evaluation import SEVERITY_LABELS, load_synthetic_dataset

SPLIT_WEIGHTS = {"train": 70, "validation": 15, "test": 15}
LABEL_IDS = {"0": "Severe", "1": "Moderate", "2": "Mild"}


def normalize_case_text(text: str) -> str:
    return " ".join(text.casefold().split())


def read_training_csv(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    required = {"symptom_combined", "weak_label_id", "weak_label"}
    fields = reader.fieldnames or []
    if not required.issubset(fields) or len(fields) != len(set(fields)):
        raise ValueError("CSV requires unique headers including symptom_combined, weak_label_id, weak_label")
    rows, texts = [], set()
    for number, record in enumerate(reader, 2):
        if None in record or any(record.get(key) is None for key in required):
            raise ValueError(f"Malformed CSV row {number}")
        text = record["symptom_combined"]
        normalized = normalize_case_text(text)
        if not normalized:
            raise ValueError(f"Blank text at CSV row {number}")
        if normalized in texts:
            raise ValueError(f"Duplicate or conflicting normalized text at CSV row {number}")
        label = record["weak_label"].strip()
        if LABEL_IDS.get(record["weak_label_id"].strip()) != label:
            raise ValueError(f"Unknown or inconsistent label mapping at CSV row {number}")
        texts.add(normalized)
        case_id = "syn-sev-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        rows.append({"id": case_id, "text": text, "label": label, "synthetic": True})
    if not rows:
        raise ValueError("Training CSV contains no examples")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Case ID collision")
    return rows, hashlib.sha256(raw).hexdigest()


def stratified_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    splits = {name: [] for name in SPLIT_WEIGHTS}
    weights = list(SPLIT_WEIGHTS.values())
    for label in SEVERITY_LABELS:
        group = [row for row in rows if row["label"] == label]
        if len(group) < 3:
            raise ValueError(f"Need at least 3 examples of {label} for coverage in all splits")
        # Hash ranking avoids dependence on input order or random-library versions.
        group.sort(key=lambda row: (hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), row["id"]))
        counts = [len(group) * weight // 100 for weight in weights]
        remaining = len(group) - sum(counts)
        priorities = sorted(range(3), key=lambda i: (-(len(group) * weights[i] % 100), i))
        for index in priorities[:remaining]:
            counts[index] += 1
        # With very small classes, prioritize one example per split over ratios.
        for index in range(3):
            if counts[index] == 0:
                donor = max((i for i in range(3) if counts[i] > 1),
                            key=lambda i: (counts[i] * 100 - len(group) * weights[i], -i))
                counts[donor] -= 1
                counts[index] += 1
        start = 0
        for name, count in zip(splits, counts):
            splits[name].extend(group[start:start + count])
            start += count
    return {name: sorted(records, key=lambda row: row["id"]) for name, records in splits.items()}


def validate_splits(splits: dict[str, list[dict]], expected_ids: set[str], protected_texts: set[str]) -> None:
    if set(splits) != set(SPLIT_WEIGHTS):
        raise ValueError("Expected train, validation and test splits")
    seen_ids, seen_texts = set(), set()
    for name, rows in splits.items():
        if {row["label"] for row in rows} != set(SEVERITY_LABELS):
            raise ValueError(f"Every class must appear in {name}")
        for row in rows:
            normalized = normalize_case_text(row["text"])
            if row["id"] in seen_ids or normalized in seen_texts:
                raise ValueError("Duplicate ID or text across prepared split rows")
            if normalized in protected_texts:
                raise ValueError("Training source overlaps the separate evaluation fixture")
            seen_ids.add(row["id"])
            seen_texts.add(normalized)
    if seen_ids != expected_ids:
        raise ValueError("Split coverage does not match source IDs")


def prepare_dataset(source: Path, evaluation_fixture: Path, output: Path, seed: int = 42) -> dict:
    """Write a NEW package; manifest is written last, after all validation."""
    if output.exists():
        raise ValueError(f"Output directory already exists: {output}; choose a new directory")
    rows, source_hash = read_training_csv(source)
    protected, protected_hash = load_synthetic_dataset(evaluation_fixture)
    splits = stratified_split(rows, seed)
    validate_splits(splits, {row["id"] for row in rows}, {normalize_case_text(row["text"]) for row in protected})
    contents = {
        f"{name}.jsonl": "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
        for name, records in splits.items()
    }
    manifest = {
        "schema_version": 1,
        "scope": "synthetic_learning_only",
        "clinical_validation": False,
        "algorithm": "stratified_hash_rank_v1",
        "case_id_algorithm": "sha256_casefold_whitespace_v1",
        "seed": seed,
        "requested_percentages": SPLIT_WEIGHTS,
        "rounding": "largest_remainder_then_minimum_one_per_class; ties use train, validation, test order",
        "source": {"file": source.name, "sha256": source_hash, "examples": len(rows)},
        "protected_evaluation": {"file": evaluation_fixture.name, "sha256": protected_hash, "examples": len(protected)},
        "splits": {name: {"examples": len(records), "ids": [row["id"] for row in records],
                          "class_counts": {label: sum(row["label"] == label for row in records) for label in SEVERITY_LABELS}}
                   for name, records in splits.items()},
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()},
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, content in contents.items():
        with (output / name).open("xb") as stream:
            stream.write(content)
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_prepared_dataset(package: Path, evaluation_fixture: Path) -> tuple[dict, dict, str]:
    """Verify saved package contents before use; hashes do not prove authorship."""
    try:
        names = {"manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"}
        if package.is_symlink() or not package.is_dir() or {p.name for p in package.iterdir()} != names:
            raise ValueError("Expected exactly the manifest and three split files")
        if any(not path.is_file() or path.is_symlink() for path in package.iterdir()):
            raise ValueError("Dataset package must contain regular files without symlinks")
        raw = (package / "manifest.json").read_bytes()
        manifest = json.loads(raw)
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise ValueError("Unsupported dataset manifest version")
        if manifest["scope"] != "synthetic_learning_only" or manifest["clinical_validation"] is not False:
            raise ValueError("Only declared synthetic learning packages are supported")
        if manifest["algorithm"] != "stratified_hash_rank_v1" or manifest["case_id_algorithm"] != "sha256_casefold_whitespace_v1":
            raise ValueError("Unsupported dataset algorithms")
        if type(manifest["seed"]) is not int or manifest["requested_percentages"] != SPLIT_WEIGHTS:
            raise ValueError("Invalid split settings")
        if set(manifest["files"]) != {f"{name}.jsonl" for name in SPLIT_WEIGHTS} or set(manifest["splits"]) != set(SPLIT_WEIGHTS):
            raise ValueError("Invalid split/file mapping")
        for value in [manifest["source"]["sha256"], manifest["protected_evaluation"]["sha256"], *manifest["files"].values()]:
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Invalid dataset SHA-256")
        protected, protected_hash = load_synthetic_dataset(evaluation_fixture)
        if manifest["protected_evaluation"]["sha256"] != protected_hash or manifest["protected_evaluation"]["examples"] != len(protected):
            raise ValueError("Protected evaluation fixture differs from preparation")
        splits = {}
        for name in SPLIT_WEIGHTS:
            rows, digest = load_synthetic_dataset(package / f"{name}.jsonl")
            if digest != manifest["files"][f"{name}.jsonl"]:
                raise ValueError(f"Dataset hash mismatch: {name}")
            details = manifest["splits"][name]
            counts = {label: sum(row["label"] == label for row in rows) for label in SEVERITY_LABELS}
            if details["examples"] != len(rows) or details["ids"] != [row["id"] for row in rows] or details["class_counts"] != counts:
                raise ValueError(f"Split metadata mismatch: {name}")
            for row in rows:
                expected = "syn-sev-" + hashlib.sha256(normalize_case_text(row["text"]).encode()).hexdigest()
                if row["id"] != expected:
                    raise ValueError(f"Case ID does not match text: {name}")
            splits[name] = rows
        ids = {case_id for split in manifest["splits"].values() for case_id in split["ids"]}
        validate_splits(splits, ids, {normalize_case_text(row["text"]) for row in protected})
        if manifest["source"]["examples"] != sum(len(rows) for rows in splits.values()):
            raise ValueError("Source example count mismatch")
        if stratified_split([row for rows in splits.values() for row in rows], manifest["seed"]) != splits:
            raise ValueError("Saved membership does not match the recorded splitting algorithm")
        return splits, manifest, hashlib.sha256(raw).hexdigest()
    except (KeyError, TypeError, AttributeError, ValueError, OSError) as error:
        raise ValueError(f"Invalid prepared dataset {package}: {error}") from error
