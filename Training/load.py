"""Restore saved experiments for inference without their original datasets."""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from Architecture.Classifier import architecture_config, build_model
from Architecture.Tokenizers.BytePairTokenizer import SubwordBPETokenizer
from Architecture.Tokenizers.WordTokenizer import Tokenizer as WordTokenizer
from Training.data import prepare_input


@dataclass
class LoadedRun:
    model: torch.nn.Module
    tokenizer: WordTokenizer | SubwordBPETokenizer
    config: dict
    labels: list

    def logits(self, text):
        """Return CPU logits [1, number_of_classes] for one input string."""
        ids, mask = prepare_input(
            self.tokenizer, text, self.config["sequence_length"], self.config["padding_id"],
        )
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.no_grad():
            return self.model(ids.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)).cpu()

    def predict(self, text):
        """Return the original class label, preserving its saved numeric/string type."""
        return self.labels[self.logits(text).argmax(dim=1).item()]


def load_run(run_dir, device="cpu"):
    """Load model.pt, tokenizer.json, and config.json from an explicit run path.

    Legacy runs without architecture metadata use the unchanged mean-pooling
    classifier. Their metrics.json supplies original label types lost in JSON
    dictionary keys. Only tensor weights are deserialized from the checkpoint.
    This restores inference, not optimizer state for resumed training.
    """
    run_dir = Path(run_dir)
    with (run_dir / "config.json").open("r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("run_format_version", 1) != 1:
        raise ValueError("Unsupported run format version.")
    architecture, model_params = architecture_config(config)

    with (run_dir / "tokenizer.json").open("r", encoding="utf-8") as file:
        tokenizer_config = json.load(file)
    tokenizer_type = tokenizer_config.get("type", "word")
    expected_type = config.get("tokenizer_instance", "word")
    if expected_type == "word" and tokenizer_type == "word":
        tokenizer = WordTokenizer.load(run_dir / "tokenizer.json")
    elif expected_type == "subword-bpe" and tokenizer_type == "subword_bpe":
        tokenizer = SubwordBPETokenizer()
        tokenizer.load(run_dir / "tokenizer.json")
    else:
        raise ValueError("Unsupported tokenizer or tokenizer/config type mismatch.")
    if len(tokenizer.vocabulary) != config["vocabulary_size"]:
        raise ValueError("Saved vocabulary size does not match the model configuration.")
    if tokenizer.obtain_token("<PAD>") != config["padding_id"]:
        raise ValueError("Saved padding ID does not match the tokenizer.")

    labels = config.get("labels")
    if labels is None:
        # Old runs already store an ordered, type-preserving label list here.
        with (run_dir / "metrics.json").open("r", encoding="utf-8") as file:
            labels = json.load(file)["labels"]
    if len(labels) != config["number_of_classes"]:
        raise ValueError("Saved labels do not match the number of classes.")
    if any(config["label_to_id"].get(str(label)) != index for index, label in enumerate(labels)):
        raise ValueError("Saved label order does not match label_to_id.")

    model = build_model(
        architecture,
        vocabulary_size=config["vocabulary_size"],
        number_of_classes=config["number_of_classes"],
        padding_id=config["padding_id"],
        architecture_params=model_params,
    )
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()
    return LoadedRun(model=model, tokenizer=tokenizer, config=config, labels=labels)
