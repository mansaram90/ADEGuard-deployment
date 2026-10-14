"""Train a small ADEGuard severity model for deployment smoke testing.

This uses synthetic data from data/training/dataset_with_entities_and_weaklabels.csv.
The exported model is only for validating the app/deployment path, not clinical
performance.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "training" / "dataset_with_entities_and_weaklabels.csv"
OUTPUT_PATH = PROJECT_ROOT / "severity_model"
BASE_MODEL = "bert-base-cased"

ID2LABEL = {0: "Severe", 1: "Moderate", 2: "Mild"}
LABEL2ID = {value: key for key, value in ID2LABEL.items()}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing training data: {path}")

    df = pd.read_csv(path)
    required_columns = {"symptom_combined", "weak_label_id"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = df[["symptom_combined", "weak_label_id"]].copy()
    df["symptom_combined"] = df["symptom_combined"].fillna("").astype(str)
    df["weak_label_id"] = df["weak_label_id"].astype(int)
    df = df[df["weak_label_id"].isin(ID2LABEL)]

    if df.empty:
        raise ValueError("No usable severity rows found")

    return df


def build_dataset(df: pd.DataFrame) -> DatasetDict:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["weak_label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df["weak_label_id"],
    )

    def to_dataset(frame: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "text": frame["symptom_combined"].tolist(),
                "label": frame["weak_label_id"].tolist(),
            }
        )

    return DatasetDict(
        {
            "train": to_dataset(train_df),
            "validation": to_dataset(val_df),
            "test": to_dataset(test_df),
        }
    )


def compute_metrics(eval_prediction):
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


def main() -> int:
    set_seed()

    df = load_rows(DATA_PATH)
    print(f"Loaded {len(df)} severity examples")
    print(f"Label counts: {df['weak_label_id'].value_counts().sort_index().to_dict()}")

    dataset = build_dataset(df)
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(PROJECT_ROOT / "outputs" / "demo_severity_checkpoints"),
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=2e-5,
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
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test metrics: {metrics}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Saved demo severity model to {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
