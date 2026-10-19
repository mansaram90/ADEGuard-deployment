"""Package existing model files without regenerating model weights.

Run from the project root: python -m scripts.create_artifact_manifests ...
"""

import argparse
from pathlib import Path

from src.artifact_metadata import ArtifactValidationError, TASKS, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    args = parser.parse_args()
    try:
        path = write_manifest(args.directory, version=args.version, task=args.task)
    except (ArtifactValidationError, OSError) as error:
        parser.exit(1, f"Packaging failed: {error}\n")
    print(f"Created {path}; existing model files were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
