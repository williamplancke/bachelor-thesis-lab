# AG News and SST-2 synthetic generation

## Layout

The setup uses the existing dataset folders:

- `Scripts/DataGeneration/AG-News/`
- `Scripts/DataGeneration/SST-2/`

Each contains `settings.py`, `audit_data.py`, `definitions.py`, and `generate.py`. Shared helpers live in `Scripts/DataGeneration/text_classification.py`. Existing CLINC150 scripts and imported Parquet files are unchanged.

## Splits and targets

| Dataset | ID | Class | Default synthetic training target |
|---|---:|---|---:|
| AG-News | 0 | World | 27,000 |
| AG-News | 1 | Sports | 27,002 |
| AG-News | 2 | Business | 26,998 |
| AG-News | 3 | Sci/Tech | 26,999 |
| SST-2 | 0 | negative | 29,780 |
| SST-2 | 1 | positive | 37,569 |

IDs are checked against the imported training Parquet's ClassLabel metadata.

AG News has no imported validation split. A fixed 10% holdout is reserved from its original training data before calculating targets: **107,999 training / 12,001 validation rows**. Normalized identical texts stay together, including conflicting-label duplicates. Groups are stratified using their first label with seed 42, so row counts are not exactly balanced. Membership is recorded in `Data/AG-News/audit/split_membership.parquet`. This is a project-defined validation split, not an official benchmark split.

SST-2 preserves its imported validation split and original class imbalance. Its `sentence` column is mapped to `text` by the common loader, without rewriting imported data.

Generation, audits, and definition inference use only TRAIN data. The AG News holdout is excluded from generation targets and definition examples. External validation/test text is not supplied to the generator. Do not assume imported SST-2 test labels are available for local evaluation.

## Conditions and definitions

- **raw:** dataset style instructions and class name; no definitions or real examples in the generation prompt.
- **with-definition:** the same instructions plus a class definition.

Default definitions are handwritten in `settings.py` and exported in `Data/<dataset>/audit/label_definitions.parquet`.

Optional model-inferred definitions use up to 32 seeded real TRAIN examples per class. They are a separate experimental condition: explicitly pass `--definitions` to use them. The real examples appear only in definition inference, not in text-generation prompts. Definition inference saves prompts/responses in a JSON sidecar.

Current default model: `qwen3.5:9b`. Qwen3 and Qwen3.5 tags (including namespaced tags) are called with `think=False` for both example generation and definition inference. Other model families keep their default thinking policy. The original CLINC150 initial-data generator used `qwen2.5:3b`; specify that model explicitly when reproducing that configuration. Keep model, seed, temperature, batch size, and definition method consistent across comparison conditions.

## Commands

Run from the repository root. Python's `-m` entry points support these existing hyphenated folder names; internal imports are relative.

### Safe previews: no inference or output writes

```bash
python -m Scripts.DataGeneration.AG-News.generate
python -m Scripts.DataGeneration.SST-2.generate
python -m Scripts.DataGeneration.AG-News.generate --condition raw
python -m Scripts.DataGeneration.SST-2.generate --condition raw
```

Audit files are already prepared in `Data/<dataset>/audit/`. For a new snapshot without replacing those files:

```bash
python -m Scripts.DataGeneration.AG-News.audit_data --output Data/AG-News/audit_v2
python -m Scripts.DataGeneration.SST-2.audit_data --output Data/SST-2/audit_v2
```

### Optional inferred definitions

These commands call Ollama and have not been executed. Omit `--generate` for a preview.

```bash
python -m Scripts.DataGeneration.AG-News.definitions --generate
python -m Scripts.DataGeneration.SST-2.definitions --generate
```

Outputs: `Data/<dataset>/audit/inferred_label_definitions.parquet` plus a JSON provenance sidecar. Existing files are refused; use `--output` for another experiment.

### Small smoke runs

```bash
python -m Scripts.DataGeneration.AG-News.generate --generate --examples-per-class 4 --output Data/AG-News/synthetic_smoke
python -m Scripts.DataGeneration.SST-2.generate --generate --examples-per-class 4 --output Data/SST-2/synthetic_smoke
```

`--examples-per-class` is a per-class override for trials or size ablations. Omitting it matches the real training partition's class counts. These datasets are much larger than CLINC150; do not run full generation just to check connectivity.

### Full raw and inferred-definition conditions

These commands actually generate data and have not been executed. Ollama must be running. Run definition inference first for the inferred-definition conditions.

```bash
python -m Scripts.DataGeneration.AG-News.generate --generate --condition raw
python -m Scripts.DataGeneration.AG-News.generate --generate --condition with-definition --definitions Data/AG-News/audit/inferred_label_definitions.parquet
python -m Scripts.DataGeneration.SST-2.generate --generate --condition raw
python -m Scripts.DataGeneration.SST-2.generate --generate --condition with-definition --definitions Data/SST-2/audit/inferred_label_definitions.parquet
```

Default output directories are `Data/<dataset>/synthetic_train_raw/` and `Data/<dataset>/synthetic_train_with_definition/`. Use explicit `--output` directories if generating both handwritten- and inferred-definition variants.

Each output contains:

- `data.parquet`: exactly **`text` and numeric `label`**; no prompt or review metadata in classifier inputs.
- `config.json`: source hash, split settings, original class mapping, definitions, model parameters, counts, status, and next request seed.
- `generation_log.jsonl`: each request's class, seed, prompt, and raw response.

Blank/oversized text, normalized duplicates across generated classes, and normalized exact real-training overlaps are rejected. Only missing examples are retried, with bounded retries. These checks do not establish semantic label correctness or detect every paraphrase. Heldout text is not used as an adaptive generation filter.

Each attempt now reports separate counts for empty responses, invalid JSON, incorrect structure, invalid examples, training overlaps, and generated duplicates. JSONL logs also include the effective thinking option, thinking-character count, Ollama output/prompt token counts, completion reason, and durations (when supplied by Ollama). Output-token counts include all model output; they are not a separate thinking-token measurement. An empty response is no longer described simply as a failure to find unique examples. If empty content coincides with a length stop and substantial thinking output, inspect the thinking policy/token budget before increasing duplicate retries.

Existing failed runs can still be resumed with the same original options. Their old log entries are preserved; newly appended requests record the patched policy and diagnostics. A four-example live Qwen3.5 smoke test succeeded in `Data/SST-2/thinking_fix_smoke/` with thinking disabled. The full failed SST-2 run was not restarted.

### Resume

Repeat the same command and options with `--resume`, for example:

```bash
python -m Scripts.DataGeneration.SST-2.generate --generate --condition raw --resume
```

Source hash, settings, targets, parameters, and definition contents/path must match. Saved rows are checked before continuing. Completed batches survive interruption; unsaved partial batches must be regenerated. A complete matching run makes no model calls. Existing outputs are refused without `--resume`. Status becomes `completed` only when all target class counts have been reached.

## Reuse the same splits for experiments

The existing `Experiments.common` still selects CLINC150; no existing experiments are silently redirected. Load a new dataset's settings via `importlib` because normal `from ... import ...` syntax cannot contain hyphens:

```python
from importlib import import_module
from Scripts.DataGeneration.text_classification import load_training_data

settings = import_module("Scripts.DataGeneration.AG-News.settings").SETTINGS
real_train, validation = load_training_data(settings)
```

Use `Scripts.DataGeneration.SST-2.settings` for SST-2. Both frames use `text` / `label`; synthetic Parquet has matching IDs and columns. Pass `label_col_name="label"` to the existing training function. For AG News, compare synthetic counts against this returned training partition, not all 120,000 imported rows.

Keep the same real validation partition across real/raw/definition-guided experiments. Choose tokenizer sequence length from TRAIN data; news snippets are substantially longer than sentiment fragments. Later targeted fine-tuning needs dataset-specific confusion analysis, not CLINC150's eleven-intent specification.

## Validation

- Both real training schemas/mappings inspected; source files preserved.
- Audits and plan-only entry points checked; a separate four-example Qwen3.5 smoke run also succeeded. No full-generation run was restarted and no classifier training was started.
- Nineteen focused unit tests cover grouping, leakage boundaries, class mappings, output schema/counts, conditions, bounded retries, overlap rejection, definition inference, resume, Qwen3.5 thinking control, and distinct empty/malformed response diagnostics.

```bash
python -m unittest Tests.test_text_classification_generation -q
```
