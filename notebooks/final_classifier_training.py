"""
Train a BioBERT-based classifier to predict symptom severity (Mild / Moderate / Severe)
using Snorkel weak labels and a CSV dataset.
"""

# -----------------------------
# 0️⃣ Imports
# -----------------------------
import os
import json
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 1️⃣ Load CSV + Snorkel weak labels
# ============================================================
DATA_PATH = "/content/sample_data/dataset_with_entities_and_weaklabels.csv"
assert os.path.exists(DATA_PATH), f"❌ Dataset not found: {DATA_PATH}"

df = pd.read_csv(DATA_PATH)

# Replace invalid weak labels (-1) with Moderate (1)
df["severity_id"] = df["weak_label_id"].apply(lambda x: 1 if x == -1 else int(x))

print("✅ Severity distribution after replacement:")
print(df["severity_id"].value_counts())


# ============================================================
# 2️⃣ Save JSONL for classifier
# ============================================================
def save_jsonl(filename, df):
    """Save tokens + labels to JSONL format."""
    records = df.to_dict(orient="records")
    with open(filename, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({
                "tokens": rec["symptom_combined"].split(),
                "severity_id": rec["severity_id"]
            }) + "\n")
    print(f"Saved → {filename} ({len(df)} samples)")


# Train/val/test split
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["severity_id"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["severity_id"])

save_jsonl("train.jsonl", train_df)
save_jsonl("val.jsonl", val_df)
save_jsonl("test.jsonl", test_df)


# ============================================================
# 3️⃣ Prepare Hugging Face Dataset
# ============================================================
def prepare_severity_dataset(jsonl_file):
    """Convert JSONL into HuggingFace Dataset."""
    texts, labels = [], []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = " ".join(item["tokens"])
            severity = item["severity_id"]
            if severity in [0, 1, 2]:
                texts.append(text)
                labels.append(severity)
    return Dataset.from_dict({"text": texts, "label": labels})


train_ds = prepare_severity_dataset("train.jsonl")
val_ds   = prepare_severity_dataset("val.jsonl")
test_ds  = prepare_severity_dataset("test.jsonl")

print("Label distribution (train):", Counter(train_ds["label"]))
print("Label distribution (val):", Counter(val_ds["label"]))


# ============================================================
# 4️⃣ Tokenization
# ============================================================
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
sev_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return sev_tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize_fn, batched=True)
val_ds   = val_ds.map(tokenize_fn, batched=True)
test_ds  = test_ds.map(tokenize_fn, batched=True)

train_ds = train_ds.remove_columns(["text"])
val_ds   = val_ds.remove_columns(["text"])
test_ds  = test_ds.remove_columns(["text"])


# ============================================================
# 5️⃣ Define model
# ============================================================
num_labels = 3  # MILD, MODERATE, SEVERE
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)


# ============================================================
# 6️⃣ Metrics
# ============================================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# ============================================================
# 7️⃣ TrainingArguments
# ============================================================
training_args = TrainingArguments(
    output_dir="./bioBERT_severity_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    report_to=[],  # disable wandb/tensorboard
    push_to_hub=False
)


# ============================================================
# 8️⃣ Trainer
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=sev_tokenizer,
    compute_metrics=compute_metrics
)


# ============================================================
# 9️⃣ Train
# ============================================================
trainer.train()


# ============================================================
# 🔟 Evaluate on test set
# ============================================================
metrics = trainer.evaluate(test_ds)
print("✅ Test set metrics:", metrics)


# ============================================================
# 1️⃣1️⃣ Save final model
# ============================================================
save_path = "bioBERT_severity_model_final"
model.save_pretrained(save_path)
sev_tokenizer.save_pretrained(save_path)

print(f"✅ Model and tokenizer saved to '{save_path}'")
