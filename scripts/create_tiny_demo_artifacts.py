"""Create tiny ADEGuard-compatible model artifacts for deployment smoke tests.

These artifacts are random, tiny BERT-compatible models. They are only intended
to validate model loading, Streamlit startup, and deployment mechanics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from transformers import BertConfig, BertForSequenceClassification, BertForTokenClassification, BertTokenizerFast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NER_OUTPUT = PROJECT_ROOT / "ner_model"
SEVERITY_OUTPUT = PROJECT_ROOT / "severity_model"
TOKENIZER_DIR = PROJECT_ROOT / "outputs" / "tiny_tokenizer"

NER_LABELS = ["B-ADE", "B-DRUG", "I-ADE", "I-DRUG", "O"]
SEVERITY_LABELS = ["Severe", "Moderate", "Mild"]

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    ".",
    ",",
    "-",
    "Patient",
    "patient",
    "reported",
    "experienced",
    "developed",
    "after",
    "following",
    "vaccine",
    "dose",
    "booster",
    "shot",
    "Pfizer",
    "BioNTech",
    "Moderna",
    "AstraZeneca",
    "Covaxin",
    "Covishield",
    "Janssen",
    "Johnson",
    "fever",
    "chills",
    "rash",
    "itching",
    "fatigue",
    "dizziness",
    "nausea",
    "vomiting",
    "pain",
    "swelling",
    "redness",
    "headache",
    "muscle",
    "joint",
    "arm",
    "chest",
    "shortness",
    "breath",
    "hospitalized",
    "death",
    "dead",
    "mild",
    "moderate",
    "severe",
    "critical",
    "injection",
    "site",
    "soreness",
    "discomfort",
    "tightness",
    "weakness",
    "body",
    "ache",
    "and",
    "with",
    "was",
    "the",
    "a",
    "of",
    "to",
    "for",
    "two",
    "days",
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_tokenizer() -> BertTokenizerFast:
    reset_dir(TOKENIZER_DIR)
    vocab_path = TOKENIZER_DIR / "vocab.txt"
    vocab_path.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    return tokenizer


def tiny_config(num_labels: int, id2label: dict[int, str], label2id: dict[str, int]) -> BertConfig:
    return BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
        type_vocab_size=2,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def write_metadata(path: Path, task: str) -> None:
    metadata = {
        "artifact_type": "synthetic_deployment_smoke_test",
        "task": task,
        "clinical_use": False,
        "description": "Tiny random model for validating ADEGuard loading and deployment only.",
    }
    (path / "ADEGUARD_DEMO_ARTIFACT.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def save_ner(tokenizer: BertTokenizerFast) -> None:
    reset_dir(NER_OUTPUT)
    id2label = {idx: label for idx, label in enumerate(NER_LABELS)}
    label2id = {label: idx for idx, label in id2label.items()}
    model = BertForTokenClassification(tiny_config(len(NER_LABELS), id2label, label2id))
    model.save_pretrained(NER_OUTPUT)
    tokenizer.save_pretrained(NER_OUTPUT)
    (NER_OUTPUT / "label.txt").write_text(
        "\n".join(f"{idx}:'{label}'" for idx, label in id2label.items()) + "\n",
        encoding="utf-8",
    )
    write_metadata(NER_OUTPUT, "token_classification")


def save_severity(tokenizer: BertTokenizerFast) -> None:
    reset_dir(SEVERITY_OUTPUT)
    id2label = {idx: label for idx, label in enumerate(SEVERITY_LABELS)}
    label2id = {label: idx for idx, label in id2label.items()}
    model = BertForSequenceClassification(tiny_config(len(SEVERITY_LABELS), id2label, label2id))
    model.save_pretrained(SEVERITY_OUTPUT)
    tokenizer.save_pretrained(SEVERITY_OUTPUT)
    (SEVERITY_OUTPUT / "label.txt").write_text(
        "\n".join(f"{idx} : '{label}'" for idx, label in id2label.items()) + "\n",
        encoding="utf-8",
    )
    write_metadata(SEVERITY_OUTPUT, "sequence_classification")


def main() -> int:
    tokenizer = build_tokenizer()
    save_ner(tokenizer)
    save_severity(tokenizer)
    print(f"Saved tiny NER artifact to {NER_OUTPUT}")
    print(f"Saved tiny severity artifact to {SEVERITY_OUTPUT}")
    print("These artifacts are random smoke-test models, not clinical/research models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
