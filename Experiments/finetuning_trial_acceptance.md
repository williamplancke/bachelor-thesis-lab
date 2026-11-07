# Final trial acceptance gates

Set before executing contrastive-v4 and retained for subsequent versions. These are small engineering pilot gates, not statistical guarantees or evidence of classifier improvement. Final v6 used a twelve-per-class three-batch stress test (stronger than the original eight-per-class requirement); it also exercised reuse of earlier coverage tasks. See finetuning_review.md for results and remaining limits.

- Run two fresh 44-candidate seeds across all eleven targeted intents.
- Require at least 90% of each run to have defensible target labels on first-pass assistant review. Exact overlaps count separately from semantic correctness.
- Require zero name-role reversals or placeholders in either run. If this fails, revise again or explicitly stop as not ready rather than silently filtering away the failure.
- Record awkward language, copied demonstrations, within-run duplicates, and existing-training overlap separately. Require at least 85% provisionally usable after these checks.
- Then run an eight-per-class, two-batch-per-class stress trial to test earlier-output exclusion and scenario repetition. Apply the same gates.
- Screen cross-seed duplicates and report them; independent trial seeds need not have mutually exclusive output. Full generation must flag duplicates across its entire output.
- Keep every raw response and all review decisions. Do not relabel failures to inflate scores.
- Passing means ready for user review of a full-generation plan, not approved for automatic training. Manual semantic review is still required for the eventual generated candidates.
- Scope: an eleven-intent targeted supplement. This does not validate generation for all 151 classes or justify replacing all-class training coverage. Fine-tuning should retain an all-class replay/control condition.
