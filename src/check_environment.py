"""Read-only ADEGuard environment and model artifact checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path

from config import C_MODEL_PATH, model_path


@dataclass(frozen=True)
class PackageCheck:
    name: str
    import_name: str
    required: bool = True


PACKAGE_CHECKS = [
    PackageCheck("streamlit", "streamlit"),
    PackageCheck("pandas", "pandas"),
    PackageCheck("numpy", "numpy"),
    PackageCheck("torch", "torch"),
    PackageCheck("transformers", "transformers"),
    PackageCheck("sentence-transformers", "sentence_transformers"),
    PackageCheck("scikit-learn", "sklearn"),
    PackageCheck("shap", "shap"),
    PackageCheck("plotly", "plotly"),
]

REQUIRED_HF_FILES = ["config.json"]
TOKENIZER_FILES = ["tokenizer.json", "vocab.txt"]
WEIGHT_FILES = [
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "flax_model.msgpack",
]


def package_status(package: PackageCheck) -> tuple[bool, str]:
    if util.find_spec(package.import_name) is None:
        return False, "not installed"

    try:
        version = metadata.version(package.name)
    except metadata.PackageNotFoundError:
        version = "installed; version unavailable"

    return True, version


def has_any_file(path: Path, candidates: list[str]) -> bool:
    return any((path / candidate).is_file() for candidate in candidates)


def check_model_dir(name: str, raw_path: str) -> bool:
    path = Path(raw_path).expanduser()
    print(f"\n{name} model path:")
    print(f"  {path}")

    if not path.exists():
        print("  FAIL: path does not exist")
        return False

    if not path.is_dir():
        print("  FAIL: path is not a directory")
        return False

    passed = True
    for filename in REQUIRED_HF_FILES:
        exists = (path / filename).is_file()
        print(f"  {'OK' if exists else 'FAIL'}: {filename}")
        passed = passed and exists

    has_tokenizer = has_any_file(path, TOKENIZER_FILES)
    print(f"  {'OK' if has_tokenizer else 'FAIL'}: tokenizer file ({' or '.join(TOKENIZER_FILES)})")
    passed = passed and has_tokenizer

    has_weights = has_any_file(path, WEIGHT_FILES)
    print(f"  {'OK' if has_weights else 'FAIL'}: model weights ({' or '.join(WEIGHT_FILES)})")
    passed = passed and has_weights

    return passed


def check_packages() -> bool:
    print("Python package checks:")
    passed = True

    for package in PACKAGE_CHECKS:
        installed, detail = package_status(package)
        status = "OK" if installed else "FAIL"
        print(f"  {status}: {package.name} - {detail}")
        if package.required:
            passed = passed and installed

    return passed


def main() -> int:
    packages_ok = check_packages()
    ner_ok = check_model_dir("NER", model_path)
    severity_ok = check_model_dir("Severity", C_MODEL_PATH)

    print("\nSummary:")
    if packages_ok and ner_ok and severity_ok:
        print("  OK: environment appears ready for local ADEGuard execution")
        return 0

    print("  FAIL: one or more required checks did not pass")
    print("  Next: install missing packages and provide complete Hugging Face model folders")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
