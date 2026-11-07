"""Generate review-only candidates from the Markdown pilot; never writes training data."""
import json
from datetime import datetime, timezone
from pathlib import Path

import ollama
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "Experiments" / "finetuning_pilot.md"
MODEL = "qwen2.5:3b"


def main():
    document = PILOT.read_text(encoding="utf-8")
    spec_section = document.split("## Proposed intent specifications", 1)[1].split("## Generator instruction draft", 1)[0]
    specs = {}
    for line in spec_section.splitlines():
        if line.startswith("| ") and not line.startswith("| Intent |"):
            label, definition, boundary = [part.strip() for part in line.strip("|").split("|")]
            specs[label] = {"definition": definition, "boundary": boundary}
    if len(specs) != 11:
        raise ValueError("Expected eleven reviewed intent specifications.")
    instructions = document.split("## Generator instruction draft", 1)[1].split("## Pilot examples", 1)[0].strip()
    names = json.loads((ROOT / "Data/CLINC150/audit/intent_names.json").read_text())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "Data/CLINC150/finetuning_trials" / stamp
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "status": "in_progress", "model": MODEL, "temperature": 0.8,
        "base_seed": 142, "examples_per_intent": 4, "purpose": "review_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specifications": specs, "instructions": instructions,
        "note": "No automatic retries or semantic filtering; retain the first response for review.",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Output: {output}", flush=True)
    item = {"type": "object", "properties": {
        "text": {"type": "string"}, "justification": {"type": "string"}},
        "required": ["text", "justification"], "additionalProperties": False}
    schema = {"type": "object", "properties": {"examples": {
        "type": "array", "items": item, "minItems": 4, "maxItems": 4}},
        "required": ["examples"], "additionalProperties": False}
    candidates = []
    for index, (label, spec) in enumerate(specs.items()):
        prompt = (
            f"{instructions}\n\nGenerate exactly four examples for target intent: {label}.\n"
            f"Definition: {spec['definition']}\nBoundary: {spec['boundary']}\n"
            f"Neighboring intent specifications (reference, not requested labels):\n"
            f"{json.dumps({k:v for k,v in specs.items() if k != label})}\n"
            "For this small batch, use two messages with at most eight words and two with nine to sixteen words. "
            "Use distinct constructions and scenarios. Return only the JSON object with examples, "
            "each containing text and justification. The label is supplied externally."
        )
        (output / f"{index:02d}_{label}_prompt.txt").write_text(prompt, encoding="utf-8")
        response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                               format=schema, options={"temperature": 0.8, "seed": 142 + index, "num_predict": 1500})
        (output / f"{index:02d}_{label}_response.json").write_text(response.model_dump_json(indent=2), encoding="utf-8")
        examples = json.loads(response.message.content)["examples"]
        if len(examples) != 4:
            raise ValueError(f"Expected four candidates for {label}")
        for example in examples:
            if not all(isinstance(example.get(k), str) and example[k].strip() for k in ("text", "justification")):
                raise ValueError(f"Malformed candidate for {label}")
            candidates.append({"candidate_id": len(candidates) + 1, "label": label,
                               "intent": names.index(label), "text": example["text"].strip(),
                               "justification": example["justification"].strip()})
        # Save each completed class; candidates are NOT approved training rows.
        pd.DataFrame(candidates).to_json(output / "candidates.json", orient="records", indent=2, force_ascii=False)
        print(f"{label}: {len(examples)} candidates ({len(candidates)}/44)", flush=True)
    config.update(status="awaiting_review", completed_at=datetime.now(timezone.utc).isoformat())
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    main()
