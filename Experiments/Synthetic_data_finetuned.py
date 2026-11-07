"""Definition-data training followed by full-model replay + supplement fine-tuning.

Run: python -m Experiments.Synthetic_data_finetuned
Both stages save separate runs; no test data is loaded. The generated supplement
has mechanical filtering, not guaranteed semantic label correctness.
"""
import warnings

import pandas as pd

from Experiments.common import (
    DATA_DIR, PARAMS, SYNTHETIC_WITH_DEFINITION_PATH,
    load_training_data, validate_synthetic_data,
)
from Training.train import main as train

SUPPLEMENT_PATH = DATA_DIR / "finetuning_candidates_v6" / "data.parquet"
FINETUNING_EPOCHS = 3
LEARNING_RATE_FACTOR = 0.1
SUPPLEMENT_FRACTION = 0.2


def replay_mixture(original, supplement, fraction=SUPPLEMENT_FRACTION, seed=42):
    """Keep every original row and sample supplement rows to approximate fraction.

    Sampling with replacement explicitly oversamples the small supplement. The
    trainer reshuffles this fixed mixture each epoch; every original row receives
    one exposure per epoch. Metadata records actual rather than requested counts.
    """
    if not 0 < fraction < 1:
        raise ValueError("Supplement fraction must be strictly between zero and one.")
    if original.empty or supplement.empty:
        raise ValueError("Replay and supplement must both be nonempty.")
    if not {"text", "intent"}.issubset(supplement.columns):
        raise ValueError("Supplement needs text and intent columns.")
    if supplement[["text", "intent"]].isna().any().any():
        raise ValueError("Supplement contains missing text or labels.")
    if not supplement["text"].map(lambda text: isinstance(text, str) and bool(text.strip())).all():
        raise ValueError("Supplement text must contain nonempty strings.")
    if not supplement["intent"].isin(original["intent"].unique()).all():
        raise ValueError("Supplement contains labels absent from the original dataset.")
    normalize = lambda frame: frame["text"].str.lower().str.split().str.join(" ")
    normalized = normalize(supplement)
    if normalized.duplicated().any():
        raise ValueError("Supplement contains duplicate sentences.")
    if normalized.isin(normalize(original)).any():
        raise ValueError("Supplement overlaps the replay dataset.")
    count = max(1, round(len(original) * fraction / (1 - fraction)))
    sampled = supplement[["text", "intent"]].sample(n=count, replace=True, random_state=seed)
    mixture = pd.concat([original[["text", "intent"]], sampled], ignore_index=True)
    metadata = {
        "replay_examples_per_epoch": len(original),
        "supplement_examples_per_epoch": count,
        "supplement_source_examples": len(supplement),
        "supplement_unique_sampled_examples": len(sampled.drop_duplicates()),
        "requested_supplement_fraction": fraction,
        "actual_supplement_fraction": count / len(mixture),
        "supplement_sampling": "fixed mixture, with replacement",
        "supplement_sampled_counts_per_class": {
            str(label): int(value) for label, value in sampled["intent"].value_counts().items()
        },
    }
    return mixture, metadata


def main():
    real_train, validation = load_training_data()
    original = pd.read_parquet(SYNTHETIC_WITH_DEFINITION_PATH)
    validate_synthetic_data(original, real_train)
    supplement = pd.read_parquet(SUPPLEMENT_PATH)
    # Validate before spending time on initial training. Never alter source files.
    mixture, mixture_info = replay_mixture(original, supplement, seed=PARAMS["seed"])
    warnings.warn("Supplement labels are generator-assigned; mechanical filtering does not ensure semantic correctness.")
    initial_params = {
        **PARAMS,
        "experiment": "synthetic_data_finetuned_initial",
        "training_stage": "initial",
        "training_data": str(SYNTHETIC_WITH_DEFINITION_PATH),
    }
    initial_run = train(original, validation, initial_params, label_col_name="intent")
    finetuning_params = {
        **PARAMS,
        "experiment": "synthetic_data_finetuned",
        "training_stage": "full_finetuning",
        "training_data": str(SYNTHETIC_WITH_DEFINITION_PATH),
        "supplement_data": str(SUPPLEMENT_PATH),
        "learning_rate": PARAMS["learning_rate"] * LEARNING_RATE_FACTOR,
        "epochs": FINETUNING_EPOCHS,
        "optimizer_reset": True,
        **mixture_info,
    }
    return train(mixture, validation, finetuning_params,
                 label_col_name="intent", checkpoint=initial_run)


if __name__ == "__main__":
    main()
