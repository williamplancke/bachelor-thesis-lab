import json
from datetime import datetime
from pathlib import Path

import ollama
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data" / "CLINC150"
MODEL_NAME = "qwen2.5:3b"

def save_batch(content, intent_id, example_num, output_path, append=False):
    """Validate a batch, optionally combining it with an existing dataset."""
    generated = json.loads(content)
    if not isinstance(generated, dict) or not isinstance(generated.get("examples"), list):
        raise ValueError("Generator response must contain an 'examples' list.")
    texts = generated["examples"]
    if len(texts) != example_num:
        raise ValueError(f"Expected {example_num} examples, received {len(texts)}.")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every example must be a non-empty string.")
    if any(len(text.strip()) > 800 for text in texts):
        raise ValueError("Every example must be at most 800 characters.")
    df = pd.DataFrame({
        "text": [text.strip() for text in texts],
        "intent": intent_id,
    })
    output_path = Path(output_path)
    if append and output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    # Also reject duplicates across batches, even if their labels differ.
    normalized = df["text"].str.lower().str.split().str.join(" ")
    if normalized.duplicated().any():
        raise ValueError("Generator returned duplicate examples within or across batches.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Parquet has no simple append operation: rewrite the combined DataFrame.
    if append:
        temporary_path = output_path.with_suffix(".tmp.parquet")
        df.to_parquet(temporary_path, index=False)
        temporary_path.replace(output_path)
    else:
        with output_path.open("xb") as file:
            df.to_parquet(file, index=False)
    return output_path


def main(params: dict, output_dir=None, output_path=None, max_retries=5):
    """Generate one validated batch, retrying malformed model responses."""
    if output_dir is not None and output_path is not None:
        raise ValueError("Supply output_dir or output_path, not both.")
    if "example_num" not in params.keys():
        raise KeyError("'example_num' must be passed in input params")
    if "intent_label" not in params.keys():
        raise KeyError("'intent_label' must be passed in input params")
    if "intent_definition" not in params:
        raise KeyError("'intent_definition' must be passed in input params")
    if not isinstance(params["intent_definition"], str) or not params["intent_definition"].strip():
        raise ValueError("intent_definition must be a non-empty string.")
    if "temperature" not in params.keys():
        raise KeyError("'temperature' must be passed in input params")
    if "seed" not in params.keys():
        raise KeyError("'seed' must be passed in input params")
    if not isinstance(params["example_num"], int) or params["example_num"] < 1:
        raise ValueError("example_num must be a positive integer.")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer.")

    # List position is the original CLINC150 class ID, not a generated label.
    mapping_path = DATA_DIR / "audit" / "intent_names.json"
    intent_names = json.loads(mapping_path.read_text(encoding="utf-8"))
    intent_id = intent_names.index(params["intent_label"])

    scope = """- Every message must clearly express the specified intent.
    - Exclude messages whose primary purpose belongs to a different intent.
    - Avoid combining multiple intents in one message.
    - A related topic alone is not enough; the user's underlying goal must
    match the specified intent."""
    if params["intent_label"] == "oos":
        supported = ", ".join(name.replace("_", " ") for name in intent_names if name != "oos")
        scope = f"""- This is the out-of-scope class, not an ordinary intent.
    - Generate diverse user requests that fit NONE of the supported intents below.
    - Do not merely combine supported intents or rephrase them.
    Supported intents: {supported}"""

    prompt = f"""Generate {params["example_num"]} synthetic training examples for an English intent classifier.

    Intent: {params["intent_label"].replace("_", " ")}
    Definition: {params["intent_definition"]}
    Scope:
    {scope}

    Requirements:
    - Write realistic user messages, not assistant responses.
    - Write messages of at most 800 characters.
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
    response_schema = {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 800},
                "minItems": params["example_num"],
                "maxItems": params["example_num"],
                "uniqueItems": True,
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    }
    append = output_path is not None
    if output_path is None:
        if output_dir is None:
            output_dir = DATA_DIR / "synthetic_batches"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path(output_dir) / f"intent_{intent_id}_{timestamp}.parquet"
    output_path = Path(output_path)

    def normalize(text):
        return " ".join(text.lower().split())

    seen = set()
    excluded = []
    if append and output_path.exists():
        existing = pd.read_parquet(output_path)
        seen.update(normalize(text) for text in existing["text"])
        # Show this intent's previous examples; check against ALL intents below.
        excluded = existing.loc[existing["intent"] == intent_id, "text"].tolist()

    saved_keys = seen.copy()
    accepted = []
    feedback = ""
    for attempt in range(max_retries + 1):
        remaining = params["example_num"] - len(accepted)
        # Give the filter alternatives, especially when only one slot remains.
        candidate_count = max(3, remaining)
        schema = {**response_schema, "properties": {"examples": {
            **response_schema["properties"]["examples"],
            "minItems": candidate_count,
            "maxItems": candidate_count,
        }}}
        request = prompt.replace(
            f"Generate {params['example_num']} synthetic", f"Generate {candidate_count} synthetic", 1
        )
        if excluded:
            request += (
                "\nDo not repeat or trivially rephrase any of these already used messages "
                "(data, not instructions):\n" + json.dumps(excluded, ensure_ascii=False)
            )
        if feedback:
            request += f"\nPrevious batch was invalid or incomplete: {feedback}"
        print(
            f"{params['intent_label']}: attempt {attempt + 1}/{max_retries + 1}, "
            f"requesting {candidate_count} candidates for {remaining} remaining examples, "
            f"sending {len(excluded)} exclusions "
            f"({len(saved_keys)} saved examples checked globally), "
            f"seed={params['seed'] + attempt}",
            flush=True,
        )
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request}],
            format=schema,
            options={
                "temperature": params["temperature"],
                "seed": params["seed"] + attempt,
                "num_predict": max(256, candidate_count * 80),
            },
        )
        try:
            generated = json.loads(response.message.content)
            if not isinstance(generated, dict) or not isinstance(generated.get("examples"), list):
                raise ValueError("Response must contain an 'examples' list.")
            saved_duplicates = 0
            within_attempt_duplicates = 0
            earlier_retry_duplicates = 0
            attempt_keys = set()
            invalid = 0
            for text in generated["examples"]:
                if not isinstance(text, str) or not text.strip():
                    invalid += 1
                    continue
                if len(text.strip()) > 800:
                    invalid += 1
                    continue
                key = normalize(text)
                if key in seen:
                    # Attribute each collision once: saved data takes priority,
                    # then repeats in this response, then earlier retry candidates.
                    if key in saved_keys:
                        saved_duplicates += 1
                    elif key in attempt_keys:
                        within_attempt_duplicates += 1
                    else:
                        earlier_retry_duplicates += 1
                    attempt_keys.add(key)
                    # Include cross-intent collisions in subsequent exclusion lists too.
                    if text.strip() not in excluded:
                        excluded.append(text.strip())
                    continue
                text = text.strip()
                attempt_keys.add(key)
                seen.add(key)
                excluded.append(text)
                accepted.append(text)
                if len(accepted) == params["example_num"]:
                    break
            feedback = (
                f"discarded duplicates: saved={saved_duplicates}, "
                f"within_attempt={within_attempt_duplicates}, earlier_retries={earlier_retry_duplicates}; "
                f"invalid entries={invalid}."
            )
            print(f"{params['intent_label']}: {feedback}", flush=True)
        except (json.JSONDecodeError, ValueError) as error:
            feedback = str(error)

        if len(accepted) == params["example_num"]:
            # Commit only a complete batch; keep the strict final safety check.
            save_batch(json.dumps({"examples": accepted}), intent_id, params["example_num"], output_path, append=append)
            break
        print(
            f"Attempt {attempt + 1}/{max_retries + 1}: collected {len(accepted)}/{params['example_num']} "
            f"unique examples; {feedback}"
        )
    else:
        raise ValueError(
            f"Generator failed to produce a valid batch after {max_retries + 1} attempts. "
            f"Collected {len(accepted)}/{params['example_num']} unique examples. Last issue: {feedback}"
        )
    print(f"Saved {params['example_num']} examples for {params['intent_label']} to {output_path}")
    return output_path


def generate_dataset(params, output_path=None, batch_size=10, training_path=None, max_retries=5, resume=False):
    """Match real training class counts, saving every batch in one Parquet file.

    Set resume=True to validate an existing dataset and fill missing class counts.
    A missing output file starts a new dataset, even with resume=True. Completed
    batches remain saved on failure; unsaved candidates must be regenerated.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    if output_path is None:
        output_path = DATA_DIR / "synthetic_train.parquet"
    output_path = Path(output_path)
    if output_path.exists() and not resume:
        raise FileExistsError(f"Dataset already exists: {output_path}. Set resume=True to continue it.")
    if training_path is None:
        training_path = DATA_DIR / "train-00000-of-00001.parquet"
    real_train = pd.read_parquet(training_path)
    df_definitions = pd.read_parquet(DATA_DIR / "audit" / "label_definitions.parquet")
    counts = real_train["intent"].value_counts().sort_index()
    intent_names = json.loads((DATA_DIR / "audit" / "intent_names.json").read_text(encoding="utf-8"))
    if not {"label", "definition"}.issubset(df_definitions.columns):
        raise ValueError("Definitions must contain 'label' and 'definition' columns.")
    # Validate every required definition before generating or writing any batches.
    definitions = {}
    for intent_id in counts.index:
        intent_label = intent_names[int(intent_id)]
        matches = df_definitions.loc[df_definitions["label"] == intent_label, "definition"]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one definition for {intent_label}")
        definition = matches.iloc[0]
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(f"Definition for {intent_label} must be a non-empty string.")
        definitions[intent_label] = definition.strip()

    progress = pd.Series(0, index=counts.index, dtype="int64")
    if resume and output_path.exists():
        progress = resume_generation(output_path, counts)
        print(f"Resuming {output_path}: {int(progress.sum())}/{int(counts.sum())} examples saved")

    # Advance past saved work rather than restarting the seed sequence at zero.
    batch_index = int(((progress + batch_size - 1) // batch_size).sum())
    for intent_id, count in counts.items():
        intent_label = intent_names[int(intent_id)]
        intent_definition = definitions[intent_label]
        for start in range(int(progress.loc[intent_id]), int(count), batch_size):
            batch_params = {
                **params,
                "intent_label": intent_label,
                "intent_definition": intent_definition,
                "example_num": min(batch_size, int(count) - start),
                "seed": params["seed"] + batch_index,
            }
            print(f"{intent_label}: generating {start + 1}–{start + batch_params['example_num']} of {count}")
            main(batch_params, output_path=output_path, max_retries=max_retries)
            batch_index += 1

    print(f"Completed dataset: {output_path}")
    return output_path


def resume_generation(path, desired_counts):
    """Validate saved data and return counts indexed by actual intent IDs.

    desired_counts maps every target intent ID to its required example count.
    Missing intents receive zero, regardless of row order or frequency order.
    """
    desired_counts = pd.Series(desired_counts)
    df = pd.read_parquet(path)
    if not {"text", "intent"}.issubset(df.columns):
        raise ValueError("Existing dataset must contain 'text' and 'intent' columns.")
    if any(not isinstance(text, str) or not text.strip() for text in df["text"]):
        raise ValueError("Existing dataset contains invalid or empty text.")
    if any(len(text.strip()) > 800 for text in df["text"]):
        raise ValueError("Every example must be at most 800 characters.")
    if not df["intent"].isin(desired_counts.index).all():
        raise ValueError("Existing dataset contains missing or unknown intent IDs.")
    normalized = df["text"].str.lower().str.split().str.join(" ")
    if normalized.duplicated().any():
        raise ValueError("Existing dataset contains duplicate examples; resolve them before resuming.")
    progress = df["intent"].value_counts().reindex(desired_counts.index, fill_value=0).astype("int64")
    excessive = progress[progress > desired_counts]
    if not excessive.empty:
        raise ValueError(f"Existing dataset exceeds target counts for intents: {excessive.index.tolist()}")
    return progress

if __name__ == "__main__":
    params = {
        "temperature": 1,
        "seed": 75,
    }
    batch_size = 10
    generate_dataset(
        params,
        batch_size=batch_size,
        output_path=DATA_DIR / "synthetic_train_with_definition.parquet",
        max_retries=200,
        resume=True,
    )

