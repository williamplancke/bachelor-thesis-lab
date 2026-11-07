# Fine-tuning data pilot — review draft

Status: first-pass assistant review completed; 44 hand-drafted examples retained for a small generator trial, not an approved full training dataset. Existing data and definitions are unchanged. These examples have not been produced by the local generator or evaluated by training a model. This is not an independent human annotation review.

## Purpose and evidence

Target five confusion groups identified during development. Inspection used synthetic data and real training data; validation aggregate errors guided group selection. No test data was inspected. Consequently, this is validation-informed development, not an independent evaluation.

Observed issues motivating this draft:
- Synthetic messages average 18.6 words versus 8.3 in real training data.
- Only 0.7% of synthetic messages have at most eight words, versus 57.4% of real training messages.
- All 100 synthetic meeting_schedule examples mention Thursday.
- Some definitions confuse speaker roles or unnecessarily restrict brands.
- Some synthetic min_payment examples ask about affordability rather than the minimum required payment.

The goal is concise, realistic coverage and correct intent boundaries, not ambiguity for its own sake. This deliberately targeted pilot is not a proposed full-dataset class distribution.

## Proposed intent specifications

These are working definitions for this pilot, not an exhaustive re-audit of the taxonomy. Confirm boundaries against the training split before scaling. Do not use test examples to refine them.

| Intent | Proposed definition | Boundary / exclusion |
|---|---|---|
| min_payment | Ask for the minimum required payment on a bill or credit obligation. | For this clean pilot, exclude payment execution, total-balance queries, and affordability questions. Training data contains at least one minimum-payment execution request; that boundary is not absolute in the benchmark. |
| pay_bill | Request a bill payment or assistance arranging or making that payment. | Not merely asking for the minimum due or outstanding balance. |
| credit_score | Ask to retrieve, locate, or verify the user's credit score or rating. | Not advice about improving or protecting the score. |
| improve_credit_score | Ask how to improve or protect a credit score, or whether an action could affect it. | Not simply retrieving the current score. |
| credit_limit | Ask for the current credit limit or spending limit on a credit account. | Not requesting a change. Remaining-credit wording also occurs under this label in training data; avoid it in this pilot until its boundary with balance-related intents is reviewed. |
| credit_limit_change | Ask to change a credit account's limit, including requesting an increase or asking how to obtain one. Do not restrict account brands. | Not retrieving the existing limit. This pilot covers increases; confirm decrease coverage against training data before adding it. |
| play_music | Ask the assistant to start music, a song, an artist, a genre, or a playlist. | Not identifying music that is already playing, changing volume, or modifying a playlist. |
| what_song | Ask to identify a song currently playing, including its title or performer. | Not asking to start a known track. Performer coverage is supported by inspected real training examples. |
| user_name | Ask what name the assistant knows, uses, or would use for the user, including recalling or guessing the user's name. | Not asking for the assistant's name or establishing a new preferred name. Do not generate unrelated online-handle advice. |
| what_is_your_name | Ask what the assistant is named or should be called. | The referent is the assistant, not the user. |
| change_user_name | Establish or change the name or nickname the assistant should use for the user, including introducing oneself or asking how to change that name. | Not querying an existing name. Training includes generic username-change wording, so do not categorically exclude it from the benchmark label; for this pilot avoid unrelated account-login scenarios. |

## Generator instruction draft

Generate English user utterances for the target intent using its reviewed definition and the supplied neighboring-intent exclusions.

- Write user messages, never assistant replies or explanations.
- Each message must have one defensible target label without inventing missing conversational context.
- Favor concise language. Across the larger dataset, aim approximately for 55–60% of messages to contain at most eight whitespace-separated words. Include some longer natural requests; this is a distribution target, not a requirement for every tiny batch.
- Include questions, commands, fragments, informal speech, and indirect requests where they remain labelable. Do not force every form into every intent.
- Vary scenarios, entities, dates, grammatical structures, and the way the requested action is expressed. Do not recycle one scenario with cosmetic substitutions.
- Avoid requiring the intent name, a specific brand, or a repeated topic keyword. Shared topic vocabulary across neighboring intents is allowed and desirable.
- Explicitly preserve the target's distinguishing action and speaker roles. Short messages must not become unlabeled fragments such as “my credit.”
- Do not deliberately add ambiguity, multiple goals, gratuitous backstories, or heavy typo corruption. Occasional natural imperfections are sufficient.
- Do not copy or lightly paraphrase supplied examples. Treat them as coverage demonstrations, not templates to fill.
- For review only, return the utterance, target label, and a short explanation of the decisive cue. Explanations and labels must never be concatenated into classifier input text.

Generate small batches and review semantic diversity between batches. Exact deduplication alone does not detect repetitive scenarios or incorrect labels.

## Pilot examples

Each class has four examples. Each group-level review cue applies to its examples; the cue is not model input. The real-data snippets from the inspection are intentionally not reused here.

### Group A: minimum due versus payment action

Review cue: required minimum amount versus making or arranging a payment. Neither label should be assigned merely because a message mentions a bill.

| Intent | Utterance |
|---|---|
| min_payment | How small can this month's card payment be? |
| min_payment | Minimum due on the broadband account? |
| min_payment | What's the required minimum for my store card? |
| min_payment | Tell me the lowest payment required on my card statement, not the full balance. |
| pay_bill | Send the broadband payment today. |
| pay_bill | Help me settle the water invoice. |
| pay_bill | Put fifty toward my store card bill. |
| pay_bill | Walk me through making a payment on my electricity account. |

### Group B: score retrieval versus score improvement

Review cue: learning the current score versus learning what affects or improves it.

| Intent | Utterance |
|---|---|
| credit_score | Pull up my latest credit rating. |
| credit_score | Where's my current credit score listed? |
| credit_score | Show the score on my credit file. |
| credit_score | Help me find out what my credit rating is right now. |
| improve_credit_score | How do I rebuild damaged credit? |
| improve_credit_score | Would closing a card hurt my score? |
| improve_credit_score | Ways to keep my credit rating healthy? |
| improve_credit_score | My credit rating has dropped; what changes could help it recover? |

### Group C: current limit versus changing the limit

Review cue: retrieving an existing account limit versus requesting or learning how to obtain an increase. Avoid confusing a limit with remaining available credit.

| Intent | Utterance |
|---|---|
| credit_limit | What's my card's approved spending ceiling? |
| credit_limit | Check the limit on my Mastercard. |
| credit_limit | Show my current card limit. |
| credit_limit | I forgot the credit limit on my bank card; can you look it up? |
| credit_limit_change | Raise my Mastercard limit to three thousand. |
| credit_limit_change | Can my card's limit be increased? |
| credit_limit_change | Help me request a higher spending limit. |
| credit_limit_change | What do I need to do to get a larger limit on my card? |

### Group D: start music versus identify playing music

Review cue: playback is requested versus a currently heard song is being identified. Both classes deliberately share music vocabulary.

| Intent | Utterance |
|---|---|
| play_music | Put on some quiet piano music. |
| play_music | Start a Nina Simone track. |
| play_music | I'd like some jazz playing now. |
| play_music | Could you play Here Comes the Sun by the Beatles? |
| what_song | Title of the track playing now? |
| what_song | Who sings the song I'm hearing? |
| what_song | Identify this tune that's on. |
| what_song | The song playing right now sounds familiar; what's it called? |

### Group E: whose name, and retrieve versus change?

Review cue: identify the referent (user or assistant), then distinguish retrieving a known name from establishing a preferred name. Bare introductions are included because real training data assigns that form to change_user_name.

| Intent | Utterance |
|---|---|
| user_name | Which name have you saved for me? |
| user_name | Remind me what you call me. |
| user_name | What nickname do you know me by? |
| user_name | When you address me, what name do you currently use? |
| what_is_your_name | What should I call you, assistant? |
| what_is_your_name | Tell me the name you go by. |
| what_is_your_name | Do you have a name I can use? |
| what_is_your_name | I know you're my assistant, but what are you called? |
| change_user_name | From now on, address me as Noor. |
| change_user_name | Use Jules when talking to me. |
| change_user_name | I'm Priya, by the way. |
| change_user_name | Please stop calling me Robert and use Robbie instead. |

## Pilot checks performed

- 44 examples across 11 classes; all 44 are unique after lowercasing and whitespace normalization.
- Zero normalized exact matches against synthetic_train_with_definition.parquet or the real training split. Validation and test utterances were not used for this check.
- 33/44 examples (75%) contain at most eight whitespace-separated words. This small pilot intentionally emphasizes concise forms; it does not yet match the proposed full-dataset length mix.
- Semantic novelty and label correctness still require review; these checks are not an independent annotation audit.

## First-pass semantic review

Review scope: all 100 real training examples in each of min_payment, credit_limit_change, what_song, user_name, and change_user_name; 15 fixed-seed examples in each of the other six pilot classes. This is 590 inspected training rows, not a complete audit of all eleven classes. No validation or test utterances were inspected. Each of the 44 draft utterances was reviewed against its target action and speaker role.

Outcome: retain all 44 as plausible clean pilot examples. They cover the intended boundaries, but this is a single-assistant judgment, not an independent reliability measurement. Do not infer from this review that data changes alone will fix model performance.

Confirmed specification corrections:
- what_is_your_name refers to the assistant, not the user.
- credit_limit_change is not restricted to Visa/Discover. Training supports many issuers and generic change requests; explicit decreases are not established by this inspection.
- what_song includes identifying the currently playing performer, not only the title.
- change_user_name includes bare self-introductions, so the pilot's introduction example is consistent with the observed taxonomy.

Important benchmark caveats discovered during review:
- min_payment training includes “pay the minimum payment for m and t bill.” Therefore, a universal rule that payment execution always means pay_bill would contradict at least one observed label. Keep clean minimum queries and payment requests in this pilot; defer mixed cases for adjudication.
- credit_limit training includes “how much to i have left on my visa card limit.” Do not silently redefine the benchmark by declaring every remaining-credit query out of class. Review neighboring balance classes before generating those cases.
- change_user_name training includes “i need to change my username.” The problem with the old synthetic examples is their heavy emphasis on elaborate online identities, not proof that the word username is always invalid.
- user_name includes future-address and name-guessing requests, not only lookup of a stored name. The revised definition preserves that breadth.

Readiness: suitable for a small model-produced trial after separately approving its execution. Generate four candidates per intent (44 total), retaining label justifications for review separately from classifier text. Use a separate output location and preserve the existing datasets. Review all candidates for meaning, repeated scenarios, and overlap before committing any training data. The current generator does not accept these revised definitions/exclusions as a ready-to-run pilot workflow; it must be adapted or invoked through a separate trial first.

## Review gates before scale-up

1. Verify the proposed definitions, especially name roles, card-brand scope, and artist identification.
2. Review every pilot example for label correctness and a decisive cue; remove uncertain examples rather than forcing labels.
3. Check normalized exact overlap against existing synthetic and real training data. Separately inspect semantic near-duplicates; passing an exact check is not a novelty guarantee.
4. Generate a small model-produced batch with the revised instructions and review it using the same gates. Do not assume the local generator will follow the instructions faithfully.
5. Expand the naturalistic foundation across all classes, with a separate contrastive supplement. Keep source/type metadata out of model inputs. Do not fine-tune only on these eleven classes without considering forgetting.
6. For the experiment, compare continued training on the original synthetic data against fine-tuning on the new data from the same checkpoint and for the same update budget. Record batch size, learning rate, seeds, and data mixture; equal update counts do not imply equal example exposure when batch sizes differ.
7. Retain untouched test evaluation. Since validation errors informed this design, report validation as development feedback rather than independent evidence.
