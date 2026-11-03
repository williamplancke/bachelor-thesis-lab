"""Settings shared by the real and synthetic training experiments."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "CLINC150"
SYNTHETIC_PATH = DATA_DIR / "synthetic_train_length_enforced.parquet"
SYNTHETIC_WITH_DEFINITION_PATH = DATA_DIR / "synthetic_train_with_definition" / "data.parquet"
# Use the same settings for both conditions. Repeat with other seeds later.
PARAMS = {
    "seed": 42,
    "sequence_length": 500,
    "embedding_dim": 64,
    "learning_rate": 0.01,
    "epochs": 20,
}


def load_training_data():
    """Do not load the test split during training/model selection."""
    train = pd.read_parquet(DATA_DIR / "train-00000-of-00001.parquet")
    validation = pd.read_parquet(DATA_DIR / "validation-00000-of-00001.parquet")
    return train, validation


def validate_synthetic_data(synthetic, real_train):
    """Require usable, unique text and exactly matching per-class counts."""
    if not {"text", "intent"}.issubset(synthetic.columns):
        raise ValueError("Synthetic data needs text and intent columns (numeric IDs).")
    if synthetic[["text", "intent"]].isna().any().any():
        raise ValueError("Synthetic data contains missing text or labels.")
    if not synthetic["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError("Synthetic text must contain non-empty strings.")
    normalized = synthetic["text"].str.lower().str.split().str.join(" ")
    if normalized.duplicated().any():
        raise ValueError("Synthetic data contains duplicate sentences; regenerate them.")
    expected = real_train["intent"].value_counts().sort_index()
    actual = synthetic["intent"].value_counts().sort_index()
    if not actual.equals(expected):
        raise ValueError("Synthetic class IDs/counts do not match the real training set.")
