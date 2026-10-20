import copy
import csv
import hashlib
import json

import pytest

from scripts.prepare_severity_data import main
from src.config import PROJECT_ROOT
from src.data_preparation import (
    normalize_case_text, prepare_dataset, read_training_csv, stratified_split, validate_splits,
)
from src.evaluation import SEVERITY_LABELS, load_synthetic_dataset

SOURCE = PROJECT_ROOT / "data/training/dataset_with_entities_and_weaklabels.csv"
EVALUATION = PROJECT_ROOT / "evaluation/severity_demo.jsonl"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["symptom_combined", "weak_label_id", "weak_label"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def source_records():
    with SOURCE.open(encoding="utf-8", newline="") as stream:
        return [{key: row[key] for key in ["symptom_combined", "weak_label_id", "weak_label"]} for row in csv.DictReader(stream)]


def test_preparation_is_byte_repeatable_and_preserves_all_rows(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = prepare_dataset(SOURCE, EVALUATION, first)
    assert prepare_dataset(SOURCE, EVALUATION, second) == manifest
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {path.name: path.read_bytes() for path in second.iterdir()}
    source_rows, source_hash = read_training_csv(SOURCE)
    assert manifest["source"]["sha256"] == source_hash
    assert manifest["protected_evaluation"]["sha256"] == hashlib.sha256(EVALUATION.read_bytes()).hexdigest()
    all_ids = []
    all_texts = []
    protected, _ = load_synthetic_dataset(EVALUATION)
    for name, details in manifest["splits"].items():
        content = (first / f"{name}.jsonl").read_bytes()
        rows = [json.loads(line) for line in content.splitlines()]
        assert details["ids"] == [row["id"] for row in rows]
        assert details["examples"] == len(rows)
        assert all(row["synthetic"] is True for row in rows)
        assert {row["label"] for row in rows} == set(SEVERITY_LABELS)
        assert hashlib.sha256(content).hexdigest() == manifest["files"][f"{name}.jsonl"]
        all_ids.extend(row["id"] for row in rows)
        all_texts.extend(normalize_case_text(row["text"]) for row in rows)
    assert len(all_ids) == len(set(all_ids)) == len(source_rows) == 45
    assert set(all_ids) == {row["id"] for row in source_rows}
    assert not set(all_texts).intersection(normalize_case_text(row["text"]) for row in protected)
    assert {name: value["examples"] for name, value in manifest["splits"].items()} == {"train": 32, "validation": 7, "test": 6}


def test_input_reordering_preserves_ids_and_assignments(tmp_path):
    records = source_records()
    first, _ = read_training_csv(write_csv(tmp_path / "forward.csv", records))
    second, _ = read_training_csv(write_csv(tmp_path / "reverse.csv", list(reversed(records))))
    assert stratified_split(first, 42) == stratified_split(second, 42)
    assert stratified_split(first, 42) != stratified_split(first, 43)
    assert {row["id"] for row in first} == {row["id"] for row in second}


def test_case_id_survives_case_whitespace_and_label_correction(tmp_path):
    row = {"symptom_combined": "Synthetic CASE", "weak_label_id": "2", "weak_label": "Mild"}
    first, _ = read_training_csv(write_csv(tmp_path / "first.csv", [row]))
    changed = dict(row, symptom_combined="  synthetic   case  ", weak_label_id="0", weak_label="Severe")
    second, _ = read_training_csv(write_csv(tmp_path / "second.csv", [changed]))
    assert first[0]["id"] == second[0]["id"]


def test_small_class_rounding_keeps_all_classes_in_each_split():
    rows = [{"id": f"{label}-{i}", "text": f"{label} case {i}", "label": label} for label in SEVERITY_LABELS for i in range(3)]
    splits = stratified_split(rows, 42)
    assert all(len(records) == 3 for records in splits.values())
    validate_splits(splits, {row["id"] for row in rows}, set())
    with pytest.raises(ValueError, match="at least 3"):
        stratified_split(rows[1:], 42)


@pytest.mark.parametrize("change", ["blank", "unknown_id", "mismatch", "unknown_label", "duplicate", "conflicting"])
def test_invalid_source_fails_before_creating_output(tmp_path, change):
    records = source_records()
    if change == "blank":
        records[0]["symptom_combined"] = " \t "
    elif change == "unknown_id":
        records[0]["weak_label_id"] = "5"
    elif change == "mismatch":
        records[0]["weak_label_id"] = "2"
    elif change == "unknown_label":
        records[0]["weak_label"] = "Unknown"
    else:
        duplicate = dict(records[0], symptom_combined="  " + records[0]["symptom_combined"].upper() + "  ")
        if change == "conflicting":
            duplicate.update(weak_label="Severe", weak_label_id="0")
        records.append(duplicate)
    source = write_csv(tmp_path / "invalid.csv", records)
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        prepare_dataset(source, EVALUATION, output)
    assert not output.exists()


@pytest.mark.parametrize("content", ["", "text,label\nhello,Mild\n", "symptom_combined,weak_label_id,weak_label\ntext,2,Mild,extra\n", 'symptom_combined,weak_label_id,weak_label\n"unfinished,2,Mild'])
def test_bad_csv_structure_is_rejected(tmp_path, content):
    source = tmp_path / "bad.csv"
    source.write_text(content)
    with pytest.raises((ValueError, csv.Error)):
        read_training_csv(source)


def test_protected_evaluation_overlap_is_rejected(tmp_path):
    protected, _ = load_synthetic_dataset(EVALUATION)
    records = source_records()
    records[0]["symptom_combined"] = protected[0]["text"]
    source = write_csv(tmp_path / "overlap.csv", records)
    with pytest.raises(ValueError, match="evaluation fixture"):
        prepare_dataset(source, EVALUATION, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["duplicate", "missing", "label_coverage"])
def test_integrity_check_detects_leakage_or_lost_coverage(change):
    rows, _ = read_training_csv(SOURCE)
    splits = copy.deepcopy(stratified_split(rows, 42))
    if change == "duplicate":
        splits["test"].append(splits["train"][0])
    elif change == "missing":
        splits["train"].pop()
    else:
        splits["test"] = [row for row in splits["test"] if row["label"] != "Mild"]
    with pytest.raises(ValueError):
        validate_splits(splits, {row["id"] for row in rows}, set())


def test_cli_preserves_existing_output_directory(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("preserve")
    with pytest.raises(SystemExit) as error:
        main(["--output-dir", str(output)])
    assert error.value.code == 1
    assert marker.read_text() == "preserve"
    assert list(output.iterdir()) == [marker]
