"""
Train a BioBERT model for ADE/Drug NER from CoNLL-style data with class weighting.
"""

import os
import random
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import torch
import torch.nn as nn
from seqeval.metrics import classification_report


# ============================================================
# STEP 1: Detect labels from CoNLL file(s)
# ============================================================
def collect_labels(files):
    labels = set()
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                labels.add(parts[-1])  # last col = label
    return sorted(list(labels))


# ============================================================
# STEP 2: Read CoNLL-style data
# ============================================================
def read_conll(filepath, label2id):
    examples = []
    tokens, tags = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append({"tokens": tokens, "ner_tags": [label2id[tag] for tag in tags]})
                    tokens, tags = [], []
                continue
            parts = line.split()
            token, tag = parts[0], parts[-1]
            tokens.append(token)
            tags.append(tag)
        if tokens:
            examples.append({"tokens": tokens, "ner_tags": [label2id[tag] for tag in tags]})
    return examples


# ============================================================
# STEP 3: Dataset creation + splitting
# ============================================================
def create_datasets(conll_path, label2id):
    silver_data = read_conll(conll_path, label2id)
    random.shuffle(silver_data)

    train_data, temp_data = train_test_split(silver_data, test_size=0.15, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data),
    })


# ============================================================
# STEP 4: Compute class weights
# ============================================================
def compute_class_weights(dataset, num_labels, label_column="ner_tags"):
    counts = Counter()
    for seq in dataset["train"][label_column]:
        counts.update(seq)
    total = sum(counts.values())
    weights = []
    for i in range(num_labels):
        weights.append(total / (num_labels * counts[i]) if counts[i] > 0 else 1.0)
    return torch.tensor(weights, dtype=torch.float)


# ============================================================
# STEP 5: Tokenization + alignment
# ============================================================
def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=512,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # same word → same label unless O
                label_ids.append(label[word_idx] if label[word_idx] != label2id["O"] else label2id["O"])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# ============================================================
# STEP 6: Weighted Trainer
# ============================================================
class WeightedTrainer(Trainer):
    def __init__(self, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ============================================================
# STEP 7: Metrics
# ============================================================
def compute_metrics(p, id2label):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1": report["micro avg"]["f1-score"]
    }


# ============================================================
# STEP 8: Main
# ============================================================
def main():
    # Paths
    conll_path = "/content/sample_data/project1.conll"  # <-- update this path as needed
    output_dir = "./ner_biobert"

    # Detect labels
    label_list = collect_labels([conll_path])
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    print("Detected labels:", label_list)

    # Build dataset
    dataset = create_datasets(conll_path, label2id)

    # Model & tokenizer
    model_name = "dmis-lab/biobert-base-cased-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        return_dict=True
    )

    # Freeze layers (except last 4)
    for param in model.bert.parameters():
        param.requires_grad = False
    for layer in model.bert.encoder.layer[-4:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Compute class weights
    class_weights = compute_class_weights(dataset, num_labels)
    class_weights = class_weights / class_weights.max()
    print("Normalized class weights:", class_weights)

    # Tokenize
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label2id),
        batched=True
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=1000,
        logging_steps=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        learning_rate=5e-5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=[],
    )

    # Trainer
    trainer = WeightedTrainer(
        weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    # Train & evaluate
    trainer.train()
    metrics = trainer.evaluate(tokenized_datasets["test"])
    print("Test Metrics:", metrics)

    # Save model
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
    print(f"✅ Model and tokenizer saved to {output_dir}/final_model")

    # Detailed report
    preds_output = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(preds_output.predictions, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in preds_output.label_ids]
    true_preds = [
        [id2label[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(preds, preds_output.label_ids)
    ]
    print("\nValidation Classification Report:")
    print(classification_report(true_labels, true_preds, digits=3))


if __name__ == "__main__":
    main()
