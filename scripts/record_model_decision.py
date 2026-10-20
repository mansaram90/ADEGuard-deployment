"""Record an explicit HOLD review of fixed synthetic model comparison evidence."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.config import C_MODEL_PATH, PROJECT_ROOT, model_path
from src.model_decision import build_hold_decision


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=Path(C_MODEL_PATH))
    parser.add_argument("--evaluation", type=Path, default=PROJECT_ROOT / "evaluation/severity_demo.jsonl")
    parser.add_argument("--reason", action="append", required=True, help="Explicit review reason; repeat for multiple reasons")
    parser.add_argument("--output", type=Path, help="New decision file; existing paths are refused")
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or PROJECT_ROOT / "outputs/decisions" / f"hold-{timestamp}.json"
    try:
        protected = [args.dataset, args.parent, args.candidate, Path(C_MODEL_PATH), Path(model_path)]
        if output.exists() or any(output.resolve().is_relative_to(path.resolve()) for path in protected):
            raise ValueError("Use a new decision file outside dataset and model packages")
        decision = build_hold_decision(args.comparison, args.dataset, args.parent, args.candidate, args.evaluation, args.reason)
        payload = json.dumps(decision, indent=2, allow_nan=False) + "\n"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Decision recording failed: {error}\n")
    print("Decision: HOLD candidate; retain the existing demo selection.")
    print("Assisted learning review only; no inference, production approval or serving change.")
    print(f"Evidence SHA-256: {decision['evidence']['comparison_sha256']}")
    print(f"Decision record: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
