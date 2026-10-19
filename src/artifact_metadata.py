"""Version and verify an immutable directory of model artifacts.

Hashes detect changed bytes, not trusted provenance or model quality.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

MANIFEST_NAME = "artifact-manifest.json"
TASKS = {"token_classification", "sequence_classification"}


class ArtifactValidationError(ValueError):
    """An artifact package cannot be safely identified and loaded."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir() or directory.is_symlink():
        raise ArtifactValidationError(f"Invalid artifact directory: {directory}")
    files = {}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ArtifactValidationError(f"Artifact symlinks are unsupported: {path}")
        name = path.relative_to(directory).as_posix()
        if path.is_file() and name != MANIFEST_NAME:
            files[name] = path
    if not files:
        raise ArtifactValidationError(f"Empty artifact directory: {directory}")
    return files


def write_manifest(directory: Path, *, version: str, task: str) -> Path:
    """Explicit packaging operation; never called during serving/startup."""
    if not isinstance(version, str) or not version.strip() or task not in TASKS:
        raise ArtifactValidationError("A nonempty version and supported task are required")
    manifest = {
        "schema_version": 1,
        "version": version,
        "task": task,
        "files": {name: sha256_file(path) for name, path in sorted(package_files(directory).items())},
    }
    target = directory / MANIFEST_NAME
    if target.exists():
        raise ArtifactValidationError(f"Manifest already exists: {target}; package a new directory/version")
    # Exclusive creation avoids silently replacing a previously approved manifest.
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, indent=2) + "\n")
    return target


def verify_artifact(directory: str | Path, *, expected_task: str) -> dict:
    directory = Path(directory)
    try:
        files = package_files(directory)
        manifest_path = directory / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ArtifactValidationError("Manifest must be an object")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
            raise ArtifactValidationError("Unsupported manifest schema_version")
        version = manifest.get("version")
        if not isinstance(version, str) or not version.strip():
            raise ArtifactValidationError("Manifest requires a nonempty version")
        if expected_task not in TASKS or manifest.get("task") != expected_task:
            raise ArtifactValidationError(f"Manifest task must be {expected_task}")
        fingerprints = manifest.get("files")
        if not isinstance(fingerprints, dict) or set(fingerprints) != set(files):
            raise ArtifactValidationError("Artifact file set differs from manifest (missing or extra files)")
        # Only paths enumerated inside the package are opened, never untrusted manifest paths.
        for name, path in files.items():
            expected_hash = fingerprints[name]
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
                raise ArtifactValidationError(f"Invalid SHA-256 for {name}")
            if sha256_file(path) != expected_hash:
                raise ArtifactValidationError(f"SHA-256 mismatch: {name}")
        metadata_path = directory / "ADEGUARD_DEMO_ARTIFACT.json"
        artifact_type = None
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ArtifactValidationError("Artifact metadata must be an object")
            artifact_type = metadata.get("artifact_type")
            if artifact_type is not None and not isinstance(artifact_type, str):
                raise ArtifactValidationError("artifact_type must be a string")
        # Canonical manifest identity includes version, task and every file hash.
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {"version": version, "manifest_sha256": identity, "artifact_type": artifact_type}
    except (OSError, ValueError) as error:
        raise ArtifactValidationError(f"Artifact verification failed for {directory}: {error}") from error
