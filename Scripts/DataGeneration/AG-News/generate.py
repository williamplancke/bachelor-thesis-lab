"""Run with python -m Scripts.DataGeneration.AG-News.generate (plan-only)."""
from .settings import SETTINGS
from Scripts.DataGeneration.text_classification import generation_cli

if __name__ == "__main__":
    generation_cli(SETTINGS)
