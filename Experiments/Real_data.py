"""Run from the project root: python -m Experiments.Real_data"""
from Experiments.common import PARAMS, load_training_data
from Training.train import main as train


def main():
    df_train, df_val = load_training_data()
    params = {**PARAMS, "experiment": "real_data"}
    return train(df_train, df_val, params, label_col_name="intent")


if __name__ == "__main__":
    main()
