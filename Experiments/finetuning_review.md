# Fine-tuning generation: review handoff

## Decision

**Ready for operator review of a modest targeted candidate-generation run. Full generation has NOT been executed.**

Recommended configuration: `Experiments/finetuning_generation_spec.json`, version **contrastive-v6**, local **qwen3:8b**, thinking disabled, temperature 0.6.

Scope is an **eleven-intent supplement**, not a replacement dataset for all 151 classes. Retain all-class coverage during fine-tuning through a separately tracked replay mixture; this work does not establish that a targeted-only fine-tune is safe. No experiment training, existing-dataset edits, test-split inspection, or full-generation run was performed. Unit tests did exercise tiny temporary training fixtures.

## What was done

- Inspected synthetic and real training data and corrected the working intent specifications in `finetuning_pilot.md`.
- Preserved 16 generator trials, totaling **960 raw candidates**, including the original 44-example trial. Generated 916 additional candidates during autonomous development. These are experimental outputs, not a pooled training dataset.
- Reviewed every candidate. Original trial judgments are in its `review.md`; later trials have per-ID `assistant_review.json` annotations.
- Used multiple seeds, two-batch stress tests, and a three-batch stress test that revisits earlier coverage tasks.
- Preserved prompts, responses, seeds, model options, and specification snapshots. Later runs also contain the runner source hash.
- Prepared a plan-first generator and unit tests. No existing source files were changed.

## Why the final approach changed

1. **Original 3B, instruction-only:** name-role reversals, minimum-payment ambiguity, and music recommendations mislabeled as playback. Only 28/44 provisionally usable.
2. **v2/v3, 3B with contrasts:** improvements were inconsistent between seeds; placeholders and name-role errors persisted.
3. **v4, 8B:** better labels, but an eight-per-class trial repeated entire batches despite earlier-output exclusions. Only 56/88 provisionally usable in that stress run.
4. **v5, explicit coverage tasks:** the eight-per-class test looked promising (79/88 usable), but a twelve-per-class test exposed repetition when coverage tasks cycled: only 70/132 usable. This failed the scale-up gate.
5. **v6, coverage plus batch-specific forms and length guidance:** alternating concise messages and longer direct/indirect requests prevented the repeated-batch failure in the tested three-batch run. This is the selected configuration.

These changes were not a controlled ablation. Do not attribute all improvement to model size, temperature, or any single prompt change.

## Final-version results

Single-assistant judgments, not independent human annotation or measured classifier performance:

| Trial | Generated | Defensible target label | Provisionally usable |
|---|---:|---:|---:|
| Focused smoke test, seed 8544, three classes | 36 | 36/36 | 35/36 |
| All eleven classes, seed 9544 | 44 | 43/44 (97.7%) | 38/44 (86.4%) |
| All eleven classes, seed 10544 | 44 | 44/44 (100%) | 39/44 (88.6%) |
| Three batches/class, seed 11544 | 132 | 130/132 (98.5%) | 122/132 (92.4%) |

The two seed checks and the stress run passed the stated gates: at least 90% defensible labels, at least 85% provisionally usable, and zero name-role reversals or placeholders. The original gate called for an eight-per-class stress test; v6 was tested more aggressively at twelve per class.

### Final stress run details

Directory: `Data/CLINC150/finetuning_trials/contrastive-v6_20261107_113057_997640/`

- 132 unique canonical utterances; **zero within-run duplicates**.
- Seven candidates overlap real training or original synthetic data under punctuation-insensitive canonical screening. They are not eligible for use in the supplement.
- Two semantic exclusions: a vague low-payment query without a decisive minimum cue, and a music-genre query instead of song identification.
- One additional quality exclusion: “Stop using Morgan and call me Morgan instead.”
- No speaker-role reversals or placeholders.
- Average length: **8.73 words**. **51.5%** have at most eight words, versus 57.4% in real training data and 0.7% in the original synthetic data. This is descriptive, not proof of distribution matching.
- All eleven classes retain at least eight provisionally usable examples. Counts differ after screening; generated counts are not accepted counts.
- Generation took approximately 135 seconds locally. A 220-candidate run extrapolates to roughly four minutes under similar conditions, excluding review and any system contention.

### Important remaining limitations

- Independent seeds are not independently novel datasets: the two final 44-example seed runs share **27 canonical utterances**. Do not pool all trials as if they were fresh data.
- Semantic diversity still requires review. Different wording does not guarantee different scenarios, and exact/canonical deduplication does not detect all near-duplicates.
- The batch word-count pattern is guidance, not a verified hard guarantee. Some returned examples violate their requested word range. Check actual word counts rather than trusting the schema or the generator's metadata.
- Action/referent metadata can be inaccurate or generic even when the utterance is usable. It is review assistance, not an automatic semantic validator and never classifier input.
- Twenty candidates per class would extend beyond the tested twelve. The later batch styles are configured but not all were exercised in the final all-class stress run. A full run remains a candidate collection requiring review, not guaranteed clean output.
- Only the eleven pilot intents are covered. Other poor classes and out-of-scope detection remain unaddressed.
- Validation aggregate confusions guided development; validation is consequently development feedback. Keep final test evaluation untouched.
- No claim of improved classifier accuracy is made. That requires a controlled fine-tuning experiment after data approval.

## What to review first

1. `Experiments/finetuning_generation_spec.json`: definitions, exclusions, demonstrations, eight coverage tasks per intent, and batch styles.
2. Final stress `candidates.json` alongside `assistant_review.json`: inspect both accepted and rejected cases, especially naming, minimum payments, and song identification.
3. `Experiments/finetuning_trial_results.json`: machine-readable summaries for all later trials and cross-seed overlap.
4. Earlier failures under `Data/CLINC150/finetuning_trials/`: failed trials are deliberately retained.

Specification snapshots v2–v5 are preserved separately. v1 instructions are in the original trial configuration. Every trial embeds its own specification/instructions, so later edits do not rewrite its evidence.

## Commands — full generation is deliberately pending

Run from the repository root. Ollama must be running and `qwen3:8b` installed; both were already available locally. The loaded model was released after the trials to free GPU memory.

### Safe preview: no inference and no output directory

```bash
python -m Scripts.DataGeneration.finetuning_pipeline --examples 20 --seed 20544
```

This previews **220 raw candidates: 20 per class across eleven classes**. It does not promise 220 accepted rows.

### Only after approval: generate a NEW candidate collection

**Not executed by the assistant:**

```bash
python -m Scripts.DataGeneration.finetuning_pipeline --mode full --examples 20 --seed 20544 --output Data/CLINC150/finetuning_candidates_v6
```

This writes `data.parquet`, `candidates.json`, `config.json`, prompts, and responses to a new directory. Existing paths are refused, including empty directories. There are no automatic semantic retries or silent relabeling. The Parquet includes only unflagged, canonically nonempty candidates, with exactly `text` and numeric `intent` columns. All members of a duplicate group are excluded, including conflicting labels. Existing-training overlaps and copied demonstrations are excluded. All raw candidates remain in JSON for later inspection.

At the operator's request, full mode now automatically writes the mechanically filtered training Parquet without requiring per-example semantic review. Generator-assigned labels can still be wrong: this is not automatic semantic approval. `config.json` records exported and excluded counts, exported counts per class, and `semantic_review_performed: false`; completion status is `completed_unreviewed`. Filtering may reduce or unbalance class counts, and no missing rows are automatically regenerated. If every row is excluded, generation fails without creating a Parquet. Trial mode continues to save JSON only. Review the actual exported class counts and replay mixture before fine-tuning. If generation fails, partial candidates and failure metadata are saved; choose a new output directory for another attempt rather than treating the partial run as complete.

## Proposed experiment after data approval

- Start both conditions from the same original checkpoint.
- Compare replay-only continued training against replay plus the new targeted supplement.
- Keep optimizer, batch size, update budget, evaluation procedure, and seeds fixed; document sample exposure and the replay/supplement ratio.
- Report aggregate and per-class metrics, including classes outside these eleven to detect forgetting.
- Do not assume the pilot's 90% usability predicts the ultimate fine-tuning effect. Consider a matched-budget real-training-data condition if it fits the thesis question, explicitly labeling it as real-data fine-tuning.

## Software validation

- New pipeline tests cover default no-inference planning, explicit full-mode output, refusal to overwrite, trial caps, response validation, overlap/duplicate flags, coverage advancement, review-field separation, numeric label IDs, incremental saving, and failure status.
- The existing full test suite was run before the last two new safety tests were added: 88 tests, with 2 failure reports and 8 error reports, all in existing CPU/CUDA device handling. New pipeline tests passed in that run. Existing training and architecture code was not changed to resolve unrelated failures.
- After adding automatic Parquet export, the focused suite has 14 tests, including export schema, integer label IDs, filtering of overlaps and conflicting duplicates, empty-output failure, preservation of raw candidates, and overwrite protection. Run with `python -m unittest Tests.test_finetuning_pipeline -q`.

Local final model metadata recorded at handoff: `qwen3:8b`, Q4_K_M, 8.2B parameters, digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41`. Model tags can change; retain this digest and the saved prompts when comparing later runs.
