
"""Configuration values for ADEGuard.

The app can run from different machines, so model locations should not depend
on one developer's local Windows path. Environment variables are used when set;
otherwise we fall back to the repo-local model folders.
"""

from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_NER_MODEL_PATH = PROJECT_ROOT / "ner_model"
DEFAULT_SEVERITY_MODEL_PATH = PROJECT_ROOT / "severity_model"


model_path = os.getenv("ADEGUARD_NER_MODEL_PATH", str(DEFAULT_NER_MODEL_PATH))
C_MODEL_PATH = os.getenv(
    "ADEGUARD_SEVERITY_MODEL_PATH",
    str(DEFAULT_SEVERITY_MODEL_PATH),
)

