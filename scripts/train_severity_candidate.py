"""Train a separate CPU candidate from verified synthetic severity splits."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.config import C_MODEL_PATH, PROJECT_ROOT


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared package directory")
    parser.add_argument("--version", required=True)
    parser.add_argument("--parent", type=Path, default=Path(C_MODEL_PATH))
    parser.add_argument("--evaluation", type=Path, default=PROJECT_ROOT / "evaluation/severity_demo.jsonl")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    from src.candidate_training import train_candidate
    output = args.output_dir or PROJECT_ROOT / "outputs/candidates" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        result = train_candidate(args.dataset, args.parent, args.evaluation, output, args.version,
                                 epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Candidate training failed: {error}\n")
    report = result["report"]
    print("Synthetic CPU training only; candidate was not promoted.")
    print(f"Parent validation macro F1: {report['baseline_validation']['macro_f1']:.4f}")
    for epoch in report["epochs"]:
        print(f"Epoch {epoch['epoch']}: loss={epoch['train_loss']:.4f}; validation macro F1={epoch['validation']['macro_f1']:.4f}")
    print(f"Selected epoch: {report['best_epoch']}; test split was not scored.")
    print(f"Candidate: {result['model_directory']}")
    print(f"Artifact manifest SHA-256: {result['artifact']['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
