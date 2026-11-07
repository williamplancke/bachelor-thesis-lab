"""Review-first targeted data generation. Default is a no-inference plan.

Trials and full generation always write NEW directories. Full mode also exports
mechanically filtered data.parquet; labels are generator-assigned, not semantically
verified. Raw candidates remain available for later inspection.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import ollama
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = ROOT / "Experiments/finetuning_generation_spec.json"


def normalized(text):
    return " ".join(text.casefold().split())


def canonical(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def load_spec(path):
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    names = json.loads((ROOT / "Data/CLINC150/audit/intent_names.json").read_text())
    labels = [s["label"] for s in spec["intents"]]
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("Intent specifications must be nonempty and unique")
    for intent in spec["intents"]:
        if intent["label"] not in names or not intent["definition"].strip():
            raise ValueError("Invalid label or definition")
        if any(label not in labels for label in intent["neighbors"]):
            raise ValueError("Unknown neighbor")
        if not intent["demonstrations"]:
            raise ValueError("Few-shot demonstrations required")
    return spec, names


def build_prompt(spec, target, count, exclusions=(), offset=0):
    by_label = {s["label"]: s for s in spec["intents"]}
    references = [target] + [by_label[label] for label in target["neighbors"]]
    contrasts = [{"category": "TARGET" if i == 0 else f"OTHER_{i}", "definition": s["definition"],
                  "examples_for_reference_only": s["demonstrations"]} for i, s in enumerate(references)]
    coverage = target.get("coverage", [])
    styles = spec.get("batch_styles", [])
    style = styles[(offset // 4) % len(styles)] if styles else None
    slots = [{"position": i + 1, "scenario_and_form": coverage[(offset + i) % len(coverage)]}
             for i in range(count)] if coverage else []
    return (
        spec["instructions"] + "\n\n"
        f"Generate exactly {count} NEW utterances for the TARGET category only.\n"
        f"Target definition: {target['definition']}\n"
        f"Mandatory target distinction: {target['required_cue']}\n"
        f"Contrastive references, NOT text to copy:\n{json.dumps(contrasts, ensure_ascii=False)}\n"
        "Return only JSON following the supplied schema. Write actual messages, never placeholders, "
        "numbered template names, or bracketed fill-in slots. "
        "Review fields are not classifier input. Do not justify a label by imagining context. "
        "Use varied commands, questions, and fragments where appropriate. "
        "Prefer two to eight words for about half the messages, nine to sixteen for the rest. "
        "Do not add filler to reach a length. Correct meaning takes priority.\n"
        f"Do not copy or trivially rephrase these previous outputs: {json.dumps(list(exclusions), ensure_ascii=False)}\n"
        f"POSITIVE COVERAGE TASKS: produce one message per position, in order: {json.dumps(slots)}\n"
        "Follow each position's scenario and form, rather than generating the same default messages. "
        "For repeated coverage cycles, choose different concrete details and constructions.\n"
        f"FINAL TARGET REMINDER: {target['definition']}\n"
        f"Required meaning in EVERY utterance: {target['required_cue']}\n"
        f"BATCH STYLE (overrides form suggestions in coverage tasks): {json.dumps(style)}\n"
        "Check the actual words before returning them: would a reader without the metadata infer this goal? "
        "The action and referent fields must describe the words, not add absent information."

    )


def response_schema(count, style=None):
    item = {"type": "object", "properties": {k: {"type": "string", "minLength": 1}
            for k in ("text", "action", "referent")},
            "required": ["text", "action", "referent"], "additionalProperties": False}
    if style:
        low, high = style["words"]
        item["properties"]["text"]["pattern"] = rf"^(?:\S+\s+){{{low - 1},{high - 1}}}\S+$"
    return {"type": "object", "properties": {"examples": {"type": "array", "items": item,
            "minItems": count, "maxItems": count}}, "required": ["examples"], "additionalProperties": False}


def validate_response(content, count):
    data = json.loads(content)
    examples = data.get("examples") if isinstance(data, dict) else None
    if not isinstance(examples, list) or len(examples) != count:
        raise ValueError(f"Expected exactly {count} examples")
    for example in examples:
        if not isinstance(example, dict) or set(example) != {"text", "action", "referent"} or any(not isinstance(example.get(k), str) or not example[k].strip()
                for k in ("text", "action", "referent")):
            raise ValueError("Missing nonempty review fields")
        if len(example["text"]) > 300:
            raise ValueError("Message exceeds 300 characters")
    return examples


def overlap_sources(spec):
    sources = {}
    for relative in spec["overlap_training_paths"]:
        frame = pd.read_parquet(ROOT / relative)
        sources[relative] = {canonical(t) for t in frame["text"]}
    sources["demonstrations"] = {canonical(t) for s in spec["intents"] for t in s["demonstrations"]}
    return sources


def flag_candidates(rows, sources):
    counts = Counter(canonical(r["text"]) for r in rows)
    for row in rows:
        key = canonical(row["text"])
        row["mechanical_flags"] = ["overlap:" + name for name, values in sources.items() if key in values]
        if counts[key] > 1:
            row["mechanical_flags"].append("duplicate_in_run")
        row["word_count"] = len(row["text"].split())
    return rows


def export_parquet(rows, output):
    """Export unflagged candidates only; this is not semantic label approval.

    All members of a duplicate group are excluded, including conflicting labels.
    Existing files are never overwritten. The raw candidates are kept separately.
    """
    selected = [row for row in rows if not row["mechanical_flags"] and canonical(row["text"])]
    if not selected:
        raise ValueError("No candidates remain after mechanical filtering; no Parquet written")
    if any(not isinstance(row["intent"], int) or isinstance(row["intent"], bool)
           or row["intent"] < 0 for row in selected):
        raise ValueError("Parquet intent IDs must be nonnegative integers")
    frame = pd.DataFrame([{"text": row["text"], "intent": row["intent"]} for row in selected])
    frame = frame.astype({"intent": "int64"})
    destination = Path(output) / "data.parquet"
    with destination.open("xb") as file:
        frame.to_parquet(file, index=False)
    return {"output_data": "data.parquet", "exported_examples": len(frame),
            "excluded_examples": len(rows) - len(frame),
            "exported_per_class": {str(key): int(count) for key, count in
                                   frame["intent"].value_counts().sort_index().items()},
            "semantic_review_performed": False,
            "export_policy": "exclude all mechanically flagged or canonically empty candidates"}


def run(spec_path, mode="plan", seed=242, examples=4, output=None, labels=None, batch_size=4):
    if examples < 1 or batch_size < 1:
        raise ValueError("Counts must be positive")
    if mode == "trial" and examples > 12:
        raise ValueError("Trials are capped at twelve candidates per class; use explicit full mode for larger runs")
    spec, names = load_spec(spec_path)
    targets = spec["intents"]
    if labels:
        requested = set(labels)
        if not requested.issubset({s["label"] for s in targets}):
            raise ValueError("Unknown requested label")
        targets = [s for s in targets if s["label"] in requested]
    plan = {"mode": mode, "model": spec["model"], "prompt_version": spec["version"],
            "classes": len(targets), "examples_per_class": examples,
            "total_candidates": len(targets) * examples, "base_seed": seed,
            "scope": "targeted supplement, not an all-class foundation",
            "labels": [s["label"] for s in targets], "batch_size": batch_size,
            "full_mode_output": "data.parquet (mechanically filtered; labels unverified)"}
    print(json.dumps(plan, indent=2), flush=True)
    if mode == "plan":
        return plan
    if mode not in ("trial", "full"):
        raise ValueError("Unknown mode")
    if mode == "full" and output is None:
        raise ValueError("Full generation requires an explicit NEW --output directory")
    # Refuse existing paths, even empty ones. Never resume into or overwrite a dataset.
    output = Path(output) if output else ROOT / "Data/CLINC150/finetuning_trials" / (
        spec["version"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    sources = overlap_sources(spec)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Output: {output}", flush=True)
    config = {**plan, "status": "in_progress", "created_at": datetime.now(timezone.utc).isoformat(),
              "specification": spec, "spec_sha256": hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "training_approved": False, "automatic_retries": 0}
    def save_config():
        (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    save_config()
    rows = []
    call_index = 0
    try:
        for target in targets:
            for start in range(0, examples, batch_size):
                count = min(batch_size, examples - start)
                exclusions = [r["text"] for r in rows if r["label"] == target["label"]]
                prompt = build_prompt(spec, target, count, exclusions, offset=start)
                prefix = f"{call_index:03d}_{target['label']}"
                (output / f"{prefix}_prompt.txt").write_text(prompt, encoding="utf-8")
                styles = spec.get("batch_styles", [])
                style = styles[(start // 4) % len(styles)] if styles else None
                response = ollama.chat(model=spec["model"], messages=[{"role": "user", "content": prompt}],
                    format=response_schema(count, style), **({"think": spec["think"]} if "think" in spec else {}),
                    options={"temperature": spec["temperature"],
                    "seed": seed + call_index, "num_predict": 1600})
                (output / f"{prefix}_response.json").write_text(response.model_dump_json(indent=2), encoding="utf-8")
                for item in validate_response(response.message.content, count):
                    rows.append({"candidate_id": len(rows) + 1, "label": target["label"],
                                 "intent": names.index(target["label"]), "seed": seed + call_index,
                                 **{k: v.strip() for k, v in item.items()}})
                flag_candidates(rows, sources)
                (output / "candidates.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
                call_index += 1
                print(f"{target['label']}: {start + count}/{examples}; total {len(rows)}", flush=True)
        if mode == "full":
            config.update(export_parquet(rows, output))
        config.update(status="completed_unreviewed" if mode == "full" else "awaiting_semantic_review", candidates=len(rows),
                      flagged_candidates=sum(bool(r["mechanical_flags"]) for r in rows),
                      completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        config.update(status="failed", error=str(error), candidates=len(rows))
        raise
    finally:
        save_config()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--mode", choices=["plan", "trial", "full"], default="plan")
    parser.add_argument("--seed", type=int, default=242)
    parser.add_argument("--examples", type=int, default=4, help="Candidates per class, not automatically accepted rows")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", nargs="+")
    args = parser.parse_args()
    run(args.spec, args.mode, args.seed, args.examples, args.output, args.labels, args.batch_size)


if __name__ == "__main__":
    main()
