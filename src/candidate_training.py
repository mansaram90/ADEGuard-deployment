"""CPU-only synthetic severity candidate training; never promotes serving models."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

from .artifact_metadata import sha256_file, verify_artifact, write_manifest
from .config import C_MODEL_PATH, model_path
from .data_preparation import load_prepared_dataset
from .evaluation import classification_metrics


def train_candidate(package: Path, parent: Path, evaluation_fixture: Path, output: Path,
                    artifact_version: str, *, epochs: int = 5, batch_size: int = 8,
                    learning_rate: float = 0.0005, seed: int = 42) -> dict:
    if not isinstance(artifact_version, str) or not artifact_version.strip() or type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError("A version and positive integer epochs/batch size are required")
    if type(seed) is not int or not 0 <= seed < 2**63 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Use a seed in [0, 2**63) and a finite positive learning rate")
    output = output.resolve()
    if output.exists() or any(output.is_relative_to(Path(path).resolve()) for path in [parent, package, C_MODEL_PATH, model_path]):
        raise ValueError("Use a new candidate directory outside input and active model directories")
    splits, data_manifest, data_identity = load_prepared_dataset(package, evaluation_fixture)
    parent_info = verify_artifact(parent, expected_task="sequence_classification")
    if parent_info["artifact_type"] not in {"synthetic_deployment_smoke_test", "synthetic_trained_learning_candidate"}:
        raise ValueError("This learning trainer accepts only synthetic model artifacts")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .adeguard_inference import SeverityModel, SEVERITY_ID2LABEL, validate_encoded_length

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            tokenizer = AutoTokenizer.from_pretrained(str(parent), local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(str(parent), local_files_only=True).to("cpu")
            if model.config.id2label != SEVERITY_ID2LABEL:
                raise ValueError("Parent label mapping differs from the serving severity mapping")
            runner = SeverityModel(tokenizer, model, torch.device("cpu"))
            # Test rows are integrity-checked by the loader but never tokenized/scored.
            for row in splits["train"] + splits["validation"]:
                encoded = tokenizer(row["text"], truncation=False, return_tensors="pt")
                validate_encoded_length(encoded, model, tokenizer, "severity")

            def validation_metrics():
                model.eval()
                return classification_metrics([row["label"] for row in splits["validation"]],
                                              [runner.predict(row["text"]).label for row in splits["validation"]])

            baseline = validation_metrics()
            label_ids = {label: index for index, label in SEVERITY_ID2LABEL.items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            generator = torch.Generator().manual_seed(seed)
            history, best_state, best_epoch, best_score = [], None, None, -1.0
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(splits["train"]), generator=generator).tolist()
                total_loss = 0.0
                for start in range(0, len(order), batch_size):
                    rows = [splits["train"][i] for i in order[start:start + batch_size]]
                    encoded = tokenizer([row["text"] for row in rows], truncation=False, padding=True, return_tensors="pt")
                    targets = torch.tensor([label_ids[row["label"]] for row in rows], dtype=torch.long)
                    optimizer.zero_grad(set_to_none=True)
                    loss = model(**encoded, labels=targets).loss
                    if not math.isfinite(loss.item()):
                        raise ValueError("Training produced nonfinite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item() * len(rows)
                metrics = validation_metrics()
                history.append({"epoch": epoch, "train_loss": total_loss / len(order), "validation": metrics})
                if metrics["macro_f1"] > best_score:
                    best_score, best_epoch = metrics["macro_f1"], epoch
                    best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            model.load_state_dict(best_state)
            model.eval()
            report = {
                "schema_version": 1, "scope": "synthetic_learning_only", "clinical_validation": False,
                "artifact_version": artifact_version, "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "parent": parent_info,
                "dataset": {"manifest_sha256": data_identity, "source_sha256": data_manifest["source"]["sha256"],
                            "split_file_sha256": data_manifest["files"],
                            "counts": {name: len(rows) for name, rows in splits.items()}},
                "settings": {"seed": seed, "epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                             "optimizer": "AdamW", "weight_decay": 0.01, "max_gradient_norm": 1.0,
                             "device": "cpu", "threads": 1, "deterministic_algorithms": True},
                "runtime": {"python": platform.python_version(), "torch": version("torch"), "transformers": version("transformers")},
                "implementation_sha256": {name: sha256_file(Path(__file__).parent / name)
                                          for name in ["candidate_training.py", "data_preparation.py", "adeguard_inference.py", "evaluation.py"]},
                "baseline_validation": baseline, "epochs": history, "best_epoch": best_epoch,
                "selected_validation": history[best_epoch - 1]["validation"],
                "selection": "highest_validation_macro_f1_among_trained_epochs; earliest_epoch_breaks_ties",
                "test_split_scored": False, "separate_evaluation_fixture_scored": False, "promoted": False,
            }
            output.mkdir(parents=True, exist_ok=False)
            model_directory = output / "model"
            model.save_pretrained(model_directory, safe_serialization=True)
            tokenizer.save_pretrained(model_directory)
            (model_directory / "label.txt").write_text("\n".join(f"{index} : '{label}'" for index, label in SEVERITY_ID2LABEL.items()) + "\n", encoding="utf-8")
            (model_directory / "training-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            metadata = {"artifact_type": "synthetic_trained_learning_candidate", "task": "sequence_classification",
                        "clinical_use": False, "description": "Tiny model trained on synthetic cases for MLOps learning only."}
            (model_directory / "ADEGUARD_DEMO_ARTIFACT.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            write_manifest(model_directory, version=artifact_version, task="sequence_classification")
            info = verify_artifact(model_directory, expected_task="sequence_classification")
            return {"model_directory": model_directory, "artifact": info, "report": report}
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
