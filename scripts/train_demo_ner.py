"""Train a small ADEGuard NER model for deployment smoke testing.

This uses synthetic data from data/training/project1.conll. The exported model
is only for validating the app/deployment path, not clinical performance.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForTokenClassification,
    BertTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "training" / "project1.conll"
OUTPUT_PATH = PROJECT_ROOT / "ner_model"
BASE_MODEL = "bert-base-cased"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_conll(path: Path) -> list[dict[str, list[str]]]:
    examples = []
    tokens: list[str] = []
    tags: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "tags": tags})
                    tokens, tags = [], []
                continue

            token, tag = line.rsplit(maxsplit=1)
            tokens.append(token)
            tags.append(tag)

    if tokens:
        examples.append({"tokens": tokens, "tags": tags})

    return examples


def build_dataset(examples: list[dict[str, list[str]]], label2id: dict[str, int]) -> DatasetDict:
    encoded_examples = [
        {"tokens": row["tokens"], "ner_tags": [label2id[tag] for tag in row["tags"]]}
        for row in examples
    ]

    train_rows, temp_rows = train_test_split(encoded_examples, test_size=0.3, random_state=42)
    val_rows, test_rows = train_test_split(temp_rows, test_size=0.5, random_state=42)

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "validation": Dataset.from_list(val_rows),
            "test": Dataset.from_list(test_rows),
        }
    )


def tokenize_and_align_labels(batch, tokenizer, label2id):
    tokenized = tokenizer(
        batch["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True,
    )

    aligned_labels = []
    for index, labels in enumerate(batch["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=index)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(labels[word_id])
            else:
                label_ids.append(labels[word_id] if labels[word_id] != label2id["O"] else label2id["O"])
            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)

    true_predictions = []
    true_labels = []
    for pred_row, label_row in zip(predictions, labels):
        for pred, label in zip(pred_row, label_row):
            if label == -100:
                continue
            true_predictions.append(int(pred))
            true_labels.append(int(label))

    if not true_labels:
        return {"token_accuracy": 0.0}

    accuracy = np.mean(np.array(true_predictions) == np.array(true_labels))
    return {"token_accuracy": float(accuracy)}


def main() -> int:
    set_seed()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training data: {DATA_PATH}")

    examples = read_conll(DATA_PATH)
    labels = sorted({tag for row in examples for tag in row["tags"]})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    print(f"Loaded {len(examples)} NER examples")
    print(f"Labels: {labels}")

    dataset = build_dataset(examples, label2id)
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    tokenized_dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer, label2id),
        batched=True,
    )

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "outputs" / "demo_ner_checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test metrics: {metrics}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Saved demo NER model to {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
