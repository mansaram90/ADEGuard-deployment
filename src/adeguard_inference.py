"""Reusable ADEGuard inference components.

This module keeps model loading and prediction logic outside the Streamlit UI so
the same code can later be reused by FastAPI, batch jobs, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

try:
    from .config import C_MODEL_PATH, model_path
except ImportError:
    from config import C_MODEL_PATH, model_path


NER_ID2LABEL = {0: "B-ADE", 1: "B-DRUG", 2: "I-ADE", 3: "I-DRUG", 4: "O"}
SEVERITY_ID2LABEL = {0: "Severe", 1: "Moderate", 2: "Mild"}


class InputValidationError(ValueError):
    """Text cannot be analyzed in full by the loaded model."""


def validate_encoded_length(encoded: Any, model: Any, tokenizer: Any, task: str) -> None:
    # Count special tokens too. Keep the existing 512-token serving ceiling,
    # but respect smaller model/tokenizer limits such as our tiny demo's 128.
    limit = min(512, model.config.max_position_embeddings, tokenizer.model_max_length)
    token_count = encoded["input_ids"].shape[-1]
    if token_count > limit:
        raise InputValidationError(
            f"Text is too long for the {task} model: {token_count} tokens; "
            f"maximum {limit}, including special tokens. Shorten the text and try again."
        )

POSTPROCESS_DICT = {
    "DRUG": {
        "pfizer",
        "moderna",
        "astrazeneca",
        "covaxin",
        "janssen",
        "johnson",
        "johnson and johnson",
        "biontech",
        "covishield",
    },
    "ADE": {
        "fever",
        "headache",
        "dizziness",
        "nausea",
        "rash",
        "fatigue",
        "chills",
        "itching",
        "sweating",
        "chest pain",
        "pain",
        "body ache",
        "swelling",
        "redness",
        "muscle pain",
        "joint pain",
        "vomiting",
        "discomfort",
        "chest discomfort",
        "bodyache",
        "injection site pain",
        "injection site swelling",
        "injection site redness",
        "arm pain",
        "arm soreness",
        "muscle soreness",
        "weakness",
        "tingling",
        "numbness",
        "fainting",
        "shortness of breath",
        "palpitations",
        "blurred vision",
        "abdominal pain",
        "stomach ache",
        "loss of appetite",
        "pain at injection site",
        "passed away",
        "died",
        "burning sensation",
        "injection site tenderness",
        "dead",
        "death",
        "fatal",
        "fatality",
        "deceased",
    },
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_pipeline_scores(preds: list[Any]) -> list[dict[str, float | str]]:
    """Return a flat list of label-score dictionaries from HF pipeline output."""
    if preds and isinstance(preds[0], dict):
        return preds
    return preds[0]


def clean_entities(entities: dict[str, list[str]]) -> dict[str, list[str]]:
    cleaned = {"DRUG": [], "ADE": []}

    for ade in entities.get("ADE", []):
        ade = ade.strip("., ")
        if ade and ade.lower() not in ["and", "reported", "later", "severe"]:
            cleaned["ADE"].append(ade)

    for drug in entities.get("DRUG", []):
        drug = re.sub(r"\band\b.*", "", drug)
        drug = drug.strip("., ")
        if re.search(r"[A-Z]", drug) or drug.lower() in POSTPROCESS_DICT["DRUG"]:
            cleaned["DRUG"].append(drug)

    return cleaned


def postprocess_entities(text: str, entities: dict[str, list[str]]) -> dict[str, list[str]]:
    new_entities = {"DRUG": list(entities["DRUG"]), "ADE": list(entities["ADE"])}
    text_norm = normalize_text(text)

    for ent_type, vocab in POSTPROCESS_DICT.items():
        for word in vocab:
            word_norm = normalize_text(word)
            already_found = any(word_norm in normalize_text(entity) for entity in new_entities[ent_type])
            if word_norm in text_norm and not already_found:
                new_entities[ent_type].append(word)

    return new_entities


@dataclass(frozen=True)
class NERPrediction:
    ade: list[str]
    drug: list[str]
    highlights: list[tuple[str, str, float]]


@dataclass(frozen=True)
class SeverityPrediction:
    label: str
    confidence: float
    scores: list[float]
    probabilities: dict[str, float]


class NERModel:
    def __init__(self, tokenizer: Any, model: Any, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def load(cls, path: str = model_path) -> "NERModel":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForTokenClassification.from_pretrained(path)
        model.to(device)
        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device)

    def predict_one(self, text: str) -> NERPrediction:
        if not text.strip():
            raise InputValidationError("Enter non-whitespace text to analyze.")
        tokens = re.findall(r"\w+|[^\w\s]", text)
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=False,
            return_tensors="pt",
        )
        validate_encoded_length(encoded, self.model, self.tokenizer, "NER")
        encoded = encoded.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            predictions = preds[0].cpu().numpy()
            confidences = confs[0].cpu().numpy()

        pred_labels = []
        previous_word_idx = None
        for idx, word_idx in enumerate(encoded.word_ids(batch_index=0)):
            if word_idx is None or word_idx == previous_word_idx:
                continue
            label = NER_ID2LABEL[int(predictions[idx])]
            confidence = float(confidences[idx])
            pred_labels.append((tokens[word_idx], label, confidence))
            previous_word_idx = word_idx

        raw_entities = self._merge_entities(pred_labels)
        entities_clean = clean_entities(raw_entities)
        entities_final = postprocess_entities(text, entities_clean)

        return NERPrediction(
            ade=entities_final.get("ADE", []),
            drug=entities_final.get("DRUG", []),
            highlights=pred_labels,
        )

    def predict_many(self, texts: list[str]) -> list[NERPrediction]:
        return [self.predict_one(text) for text in texts]

    @staticmethod
    def _merge_entities(pred_labels: list[tuple[str, str, float]]) -> dict[str, list[str]]:
        entities = {"DRUG": [], "ADE": []}
        current_entity = None
        current_words = []

        for word, label, _confidence in pred_labels:
            if label.startswith("B-") or label.startswith("I-"):
                entity_type = label.split("-")[1]
                if current_entity == entity_type:
                    current_words.append(word)
                else:
                    if current_entity and current_words:
                        entities[current_entity].append(" ".join(current_words))
                    current_entity = entity_type
                    current_words = [word]
            else:
                if current_entity and current_words:
                    entities[current_entity].append(" ".join(current_words))
                current_entity, current_words = None, []

        if current_entity and current_words:
            entities[current_entity].append(" ".join(current_words))

        return entities


class SeverityModel:
    def __init__(self, tokenizer: Any, model: Any, device: torch.device) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @classmethod
    def load(cls, path: str = C_MODEL_PATH) -> "SeverityModel":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        model.to(device)
        model.eval()
        return cls(tokenizer=tokenizer, model=model, device=device)

    def predict(self, text: str) -> SeverityPrediction:
        if not text.strip():
            raise InputValidationError("Enter non-whitespace text to analyze.")
        encoded = self.tokenizer(
            text,
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        validate_encoded_length(encoded, self.model, self.tokenizer, "severity")
        encoded = encoded.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoded)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        scores = [float(score) for score in probs]
        pred_id = int(np.argmax(scores))
        label = SEVERITY_ID2LABEL.get(pred_id, "Unknown")
        confidence = float(scores[pred_id])
        probabilities = {
            SEVERITY_ID2LABEL[index]: float(score)
            for index, score in enumerate(scores)
            if index in SEVERITY_ID2LABEL
        }

        return SeverityPrediction(
            label=label,
            confidence=confidence,
            scores=scores,
            probabilities=probabilities,
        )
