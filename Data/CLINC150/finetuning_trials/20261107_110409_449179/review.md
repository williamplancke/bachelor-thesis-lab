# Generator trial review

## Outcome

Model: qwen2.5:3b (local Ollama); temperature 0.8; seeds 142–152.
44 first-response candidates, four per intent. No retries, cherry-picked generations, or relabeling. Prompts and full responses are preserved alongside candidates.json and config.json.

First-pass assistant review: **28 provisionally usable; 16 rejected** for this clean pilot. This is not independent human annotation. Usable means the utterance has a defensible target label, not that the resulting collection is balanced or approved for training. No accepted training Parquet was created.

## Mechanical checks

- No normalized exact overlaps with the existing synthetic-with-definition dataset or real training split (lowercase + whitespace normalization only).
- Candidate IDs 36 and 41 are identical text with different assigned labels. Both labels are wrong for that text; both candidates are rejected rather than relabeled.
- 32/44 messages (72.7%) have at most eight whitespace-separated words. The requested two-short/two-long allocation per class was frequently ignored.
- No validation or test utterances were inspected or used for overlap screening.
- Semantic near-duplicate checking remains qualitative, not an exhaustive similarity search.

## Per-candidate review

Read utterance text and generator justification in candidates.json using the IDs below. The judgments below prioritize the utterance over the generator's self-justification.

| ID | Decision | Reason |
|---|---|---|
| 1 | Provisionally usable | Explicit minimum-payment query. |
| 2 | Reject | Amount due is not necessarily the minimum; no decisive minimum-payment cue. |
| 3 | Provisionally usable | Explicit minimum amount owed on a credit card. |
| 4 | Reject | Asks for the monthly amount, not specifically the minimum required payment. |
| 5 | Provisionally usable | Requests payment of rent bill. |
| 6 | Provisionally usable | Requests help making utility payment. |
| 7 | Provisionally usable | Requests guidance making electric-bill payment. |
| 8 | Reject | Groceries bill is awkward and lacks a clear billing context; exclude from a clean pilot rather than declare universally invalid. |
| 9 | Provisionally usable | Credit-score retrieval. |
| 10 | Provisionally usable | Credit-rating verification. |
| 11 | Provisionally usable | FICO terminology supplies lexical variation for credit-score retrieval. |
| 12 | Provisionally usable | Current credit-rating retrieval. |
| 13 | Provisionally usable | Explicit score improvement. |
| 14 | Provisionally usable | Asks how debt repayment affects credit score. |
| 15 | Provisionally usable | Asks for a way to increase credit score. |
| 16 | Provisionally usable | Asks about an action's effect on credit score. |
| 17 | Provisionally usable | Current credit-limit lookup. |
| 18 | Provisionally usable | Credit-limit lookup command. |
| 19 | Provisionally usable | Credit-limit lookup; little novelty relative to 17/20. |
| 20 | Provisionally usable | Credit-limit lookup; little novelty relative to 17/19. |
| 21 | Provisionally usable | Explicit request for higher limit. |
| 22 | Reject | Queries existing maximum charge, not a limit change; neighboring credit_limit intent is more defensible. |
| 23 | Provisionally usable | Asks how to increase limit. |
| 24 | Provisionally usable | Explicit limit-increase command and amount. |
| 25 | Reject | Playlist recommendation, not a request to start playback. |
| 26 | Reject | Song recommendation, not a request to start playback. |
| 27 | Provisionally usable | Explicit music-playback command. |
| 28 | Reject | Music availability/recommendation query does not clearly request playback. |
| 29 | Reject | Main intent is plausible, but “can't hear the title” is unnatural filler; exclude from this clean pilot. |
| 30 | Provisionally usable | Identifies a currently playing song. |
| 31 | Reject | “Title” has no music referent; requires missing context and could refer to another medium. |
| 32 | Provisionally usable | Current playback plus artist cue supplies a defensible music-identification context. |
| 33 | Reject | “Your current username” refers to the assistant, not the user's stored name. |
| 34 | Reject | Requests use of a gamer tag; does not retrieve the known user name. |
| 35 | Reject | Asks permission to use a nickname; does not retrieve the known user name. |
| 36 | Reject | Asks for assistant's name, not user's name; identical to 41. |
| 37 | Provisionally usable | Direct assistant-name query. |
| 38 | Provisionally usable | Assistant-name query, although phrasing is less natural; review if enforcing a strict fluency gate. |
| 39 | Provisionally usable | Assistant-name query. Generator explanation misstates roles, but the utterance is correctly labeled. |
| 40 | Reject | “Name yourself” can request choosing a new assistant name rather than reporting its current name. |
| 41 | Reject | Assistant-name query, not change_user_name; identical to 36. |
| 42 | Provisionally usable | Requests using a different form of address for user; benchmark supports name changes without specifying the new name. |
| 43 | Provisionally usable | Explicit new user name. |
| 44 | Reject | “Can you go by my name?” asks assistant to adopt user's name; not changing how user is addressed. |

## Class coverage after first-pass review

| Intent | Usable / generated |
|---|---:|
| min_payment | 2/4 |
| pay_bill | 3/4 |
| credit_score | 4/4 |
| improve_credit_score | 4/4 |
| credit_limit | 4/4 |
| credit_limit_change | 3/4 |
| play_music | 1/4 |
| what_song | 2/4 |
| user_name | 0/4 |
| what_is_your_name | 3/4 |
| change_user_name | 2/4 |
| Total | 28/44 |

## Interpretation and next intervention

The revised instructions alone did not solve label fidelity. Most strikingly, all four user_name candidates failed the intended distinction. Several generator justifications confidently assert an interpretation absent from the text. Do not use those explanations as automatic correctness checks.

The prompt contained every pilot intent as a purported neighbor, including unrelated intents. This may add distraction, but this trial cannot establish it as the cause. The old hand-drafted demonstrations were intentionally not included, so this is a zero-shot instruction-following trial, not a few-shot test.

Before scale-up:
1. Run a second, separately versioned trial on failed groups, supplying only immediate competing intents.
2. For user_name, explicitly state: user = speaker (“me/my”); assistant = addressee (“you/your”). Provide a few contrasting demonstrations, held out from the output and checked for copying.
3. For play_music, require an affirmative playback cue; explicitly exclude recommendations and availability questions.
4. For min_payment, require a minimum/least/lowest-required cue, not merely an amount due.
5. Ask for the requested action and referent as review fields, but retain human/assistant review: a generator can rationalize incorrect output.
6. Check semantic diversity even in usable classes: the four credit_limit examples have very little scenario or structure variation.
7. Do not fine-tune on this unbalanced filtered collection. Repair coverage first, then construct a balanced naturalistic foundation and separately tracked targeted supplement.

No model training or changes to existing datasets were performed.
