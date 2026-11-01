"""Run with python -m Experiments.audit_data. Does not modify source data."""
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from Experiments.common import DATA_DIR


def load_intent_names(path):
    """Hugging Face embeds the ID-to-name mapping in Parquet metadata."""
    metadata = pq.read_schema(path).metadata
    info = json.loads(metadata[b"huggingface"])
    return info["info"]["features"]["intent"]["names"]


def normalized_texts(df):
    # Catch case/whitespace differences, not semantic paraphrases.
    return df["text"].dropna().str.lower().str.split().str.join(" ")


def audit_data(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    splits = {}
    names = load_intent_names(data_dir / "train-00000-of-00001.parquet")
    report = {"splits": {}, "cross_split_overlap": {}}

    for split in ("train", "validation", "test"):
        path = data_dir / f"{split}-00000-of-00001.parquet"
        if load_intent_names(path) != names:
            raise ValueError(f"Intent-name mapping differs in {split}.")
        df = pd.read_parquet(path)
        splits[split] = df
        normalized = normalized_texts(df)
        report["splits"][split] = {
            "examples": len(df),
            "missing_text": int(df["text"].isna().sum()),
            "missing_labels": int(df["intent"].isna().sum()),
            "blank_text": int(normalized.eq("").sum()),
            "duplicate_text_rows": int(normalized.duplicated().sum()),
            "missing_class_ids": sorted(set(range(len(names))) - set(df["intent"].dropna().tolist())),
            "unknown_class_ids": sorted(set(df["intent"].dropna().tolist()) - set(range(len(names)))),
        }

    # Only counts are exported: no validation/test sentences enter generation files.
    for first, second in combinations(splits, 2):
        overlap = set(normalized_texts(splits[first])) & set(normalized_texts(splits[second]))
        report["cross_split_overlap"][f"{first}/{second}"] = len(overlap)

    counts = splits["train"]["intent"].value_counts()
    targets = pd.DataFrame({
        "intent": range(len(names)),
        "intent_name": names,
        "real_training_count": [int(counts.get(i, 0)) for i in range(len(names))],
    })
    targets["required_synthetic_count"] = targets["real_training_count"]
    targets["generation_strategy"] = [
        "diverse requests outside all 150 supported intents" if name == "oos"
        else "requests expressing this intent"
        for name in names
    ]
    report["oos"] = {
        "included": "oos" in names,
        "intent_id": names.index("oos") if "oos" in names else None,
        "note": "Treat as one supervised class for this baseline; generate separately from ordinary intents.",
    }

    output_dir = data_dir / "audit"
    output_dir.mkdir(exist_ok=True)
    targets.to_csv(output_dir / "generation_targets.csv", index=False)
    with (output_dir / "intent_names.json").open("w", encoding="utf-8") as file:
        json.dump(names, file, indent=2)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Generation targets: {output_dir / 'generation_targets.csv'}")
    return report, targets


if __name__ == "__main__":
    audit_data()
