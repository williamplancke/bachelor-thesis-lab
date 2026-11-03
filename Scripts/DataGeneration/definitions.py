import ollama
import json
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "CLINC150"
MAPPING_PATH = DATA_DIR / "audit" / "intent_names.json"
def main():
    labels = json.load(Path(MAPPING_PATH))
    df = pd.read_parquet(Path(DATA_DIR) / "train-00000-of-00001.parquet")
    for label in labels:
        label_idx = labels.index(label)
        relevant_samples = df.loc[df["intent"] == label]
        prompt = """
        """
    return
def get_response(prompt):
    return
if __name__ == '__main__':
    main()