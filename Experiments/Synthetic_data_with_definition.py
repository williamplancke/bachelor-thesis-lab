"""Train on synthetic_train.parquet; validate on the same real validation split."""
import pandas as pd

from Experiments.common import PARAMS, SYNTHETIC_WITH_DEFINITION_PATH, load_training_data, validate_synthetic_data
from Training.train import main as train


def main():
    real_train, df_val = load_training_data()
    if not SYNTHETIC_WITH_DEFINITION_PATH.exists():
        raise FileNotFoundError(
            f"Create {SYNTHETIC_WITH_DEFINITION_PATH} with text/intent columns, using audit/generation_targets.csv."
        )
    synthetic = pd.read_parquet(SYNTHETIC_WITH_DEFINITION_PATH)
    validate_synthetic_data(synthetic, real_train)
    params = {**PARAMS, "experiment": "synthetic_data_with_definition", "training_data": str(SYNTHETIC_WITH_DEFINITION_PATH)}
    return train(synthetic, df_val, params, label_col_name="intent")


if __name__ == "__main__":
    main()