"""Train on synthetic_train.parquet; validate on the same real validation split."""
import pandas as pd

from Experiments.common import PARAMS, SYNTHETIC_PATH, load_training_data, validate_synthetic_data
from Training.train import main as train


def main():
    real_train, df_val = load_training_data()
    if not SYNTHETIC_PATH.exists():
        raise FileNotFoundError(
            f"Create {SYNTHETIC_PATH} with text/intent columns, using audit/generation_targets.csv."
        )
    synthetic = pd.read_parquet(SYNTHETIC_PATH)
    validate_synthetic_data(synthetic, real_train)
    params = {**PARAMS, "experiment": "synthetic_data_raw", "training_data": str(SYNTHETIC_PATH), "tokenizer_instance": "subword-bpe", "tokenizer_params": {"vocabulary_limit": 500}, "sequence_length": 1200}
    return train(synthetic, df_val, params, label_col_name="intent")


if __name__ == "__main__":
    main()
