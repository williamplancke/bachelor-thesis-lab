import json
from datetime import datetime
from pathlib import Path

import ollama
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "CLINC150"


def save_batch(content, intent_id, example_num, output_path):
    """Validate the response and save rows with the requested intent's ID."""
    generated = json.loads(content)
    if not isinstance(generated, dict) or not isinstance(generated.get("examples"), list):
        raise ValueError("Generator response must contain an 'examples' list.")
    texts = generated["examples"]
    if len(texts) != example_num:
        raise ValueError(f"Expected {example_num} examples, received {len(texts)}.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every example must be a non-empty string.")

    df = pd.DataFrame({
        "text": [text.strip() for text in texts],
        "intent": intent_id,
    })
    normalized = df["text"].str.lower().str.split().str.join(" ")
    if normalized.duplicated().any():
        raise ValueError("Generator returned duplicate examples.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally overwriting an existing batch.
    with output_path.open("xb") as file:
        df.to_parquet(file, index=False)
    return output_path


def main(params: dict, output_dir=None):
    if "example_num" not in params.keys():
        raise KeyError("'example_num' must be passed in input params")
    if "intent_label" not in params.keys():
        raise KeyError("'intent_label' must be passed in input params")
    if "temperature" not in params.keys():
        raise KeyError("'temperature' must be passed in input params")
    if "seed" not in params.keys():
        raise KeyError("'seed' must be passed in input params")
    if not isinstance(params["example_num"], int) or params["example_num"] < 1:
        raise ValueError("example_num must be a positive integer.")

    # List position is the original CLINC150 class ID, not a generated label.
    mapping_path = DATA_DIR / "audit" / "intent_names.json"
    intent_names = json.loads(mapping_path.read_text(encoding="utf-8"))
    intent_id = intent_names.index(params["intent_label"])

    prompt = f"""Generate {params["example_num"]} synthetic training examples for an English intent classifier.

    Intent: {params["intent_label"].replace("_", " ")}

    Scope:
    - Every message must clearly express the specified intent.
    - Exclude messages whose primary purpose belongs to a different intent.
    - Avoid combining multiple intents in one message.
    - A related topic alone is not enough; the user's underlying goal must
    match the specified intent.

    Requirements:
    - Write realistic user messages, not assistant responses.
    - Vary sentence structure, vocabulary, and length.
    - Include both direct requests and indirect statements.
    - Avoid near-duplicates.
    - Do not include class annotations or explanations alongside the messages.

    Return only a JSON object in this format:
    {{
        "examples": [
        "first user message",
        "second user message"
        ]
    }}"""
    response = ollama.chat(
        model = "qwen2.5:3b",
        messages = [{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": params["temperature"], "seed": params["seed"]},
    )
    if output_dir is None:
        output_dir = DATA_DIR / "synthetic_batches"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = Path(output_dir) / f"intent_{intent_id}_{timestamp}.parquet"
    save_batch(response.message.content, intent_id, params["example_num"], output_path)
    print(f"Saved {params['example_num']} examples to {output_path}")
    return output_path


if __name__ == "__main__":
    params = {
        "example_num": 10,
        "intent_label": "accept_reservations",
        "temperature": 0.8,
        "seed": 42,
    }
    batch_path = main(params)
