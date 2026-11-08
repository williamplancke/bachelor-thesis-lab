"""Optionally infer definitions from real TRAIN examples only."""
from .settings import SETTINGS
from Scripts.DataGeneration.text_classification import definitions_cli

if __name__ == "__main__":
    definitions_cli(SETTINGS)
