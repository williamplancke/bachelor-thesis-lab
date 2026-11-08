"""Prepare training-only generation targets; retain imported validation split."""
from .settings import SETTINGS
from Scripts.DataGeneration.text_classification import audit_cli

if __name__ == "__main__":
    audit_cli(SETTINGS)
