# Task 2 Japanese Nemotron refusal confirmation preregistration

Date frozen: 6 August 2026, before any FinEval judgment from this checkpoint.
Team: ADML.

## Motivation and scope

The frozen two-seed Japanese-BERT ordinal incumbent is correct on 175/253
source-OOF rows (0.691700), but it predicts only `0`, `+1`, or `+2` and is
correct on **0/13** gold refusal rows (`-2: 0/2`, `-1: 0/11`). The already
frozen 22-row Japanese Nemotron pilot contains all 13 refusals and may advance
only if it correctly classifies at least seven of them.

If and only if that pilot passes every preregistered gate, run the model on the
complete 253-row public JF-ICR set under the identical prompt, checkpoint,
schema, and inference settings. This confirmation tests one narrow hypothesis:
a Japanese-specialized reasoning model can repair the incumbent's absent
refusal boundary without becoming a noisy global replacement.

The 253 public labels are allowed development labels. The supervised incumbent
predictions are genuine reconstructed-source OOF predictions; Nemotron is
zero-shot and receives no labels, demonstrations, source metadata, incumbent
prediction, or test information. Current or hidden test data are excluded.

## Frozen model and generations

- Model, revision, BF16 serving configuration, prompt, strict JSON schema,
  thinking mode, temperature 0.6, top-p 0.95, seed 2026, and 2048-token output
  limit are byte-for-byte identical to
  `docs/icr_nemotron_japanese_pilot_preregistration.md`.
- Reuse the exact 22 successful pilot judgments rather than drawing a second
  stochastic sample. Generate one judgment for each of the other 231 rows.
- A byte-identical retry is allowed only after a transport failure. An invalid
  completed generation is invalid and incorrect; no repair prompt is allowed.
- Require one unique judgment for every ID 0 through 252, with the stored gold
  used only after all judgments are complete.

## Frozen incumbent

Load aligned logits from:

- `artifacts/task2_japanese_cv/seed7/oof_predictions.csv`;
- `artifacts/task2_japanese_cv/seed19/oof_predictions.csv`.

Softmax each seed, average equally, decode by rounded expected ordinal score,
then apply the immutable `safe_rules` override exactly as in
`scripts/evaluate_icr_ordinal_decode.py`. The expected reproduction is 175/253.
Any drift stops evaluation.

## Single primary candidate: grounded refusal override

Start from the incumbent label. Replace it with Nemotron's `-2` or `-1` label
only when **all** of the following input/output-only conditions hold:

1. the strict JSON judgment is valid;
2. `label` is `-2` or `-1`;
3. `direction` is exactly `reject`;
4. numeric `confidence >= 0.80`;
5. after Unicode NFKC normalization and removal of whitespace, the
   `decisive_span` contains at least two characters and is an exact substring
   of the similarly normalized company response.

Otherwise retain the incumbent. No rule may change a row to `0`, `+1`, or
`+2`. The 0.80 threshold, factor requirement, grounding check, normalization,
and minimum span length are frozen. There is no alternative confidence grid,
label-specific exception, or post-hoc prompt.

For analysis only, report Nemotron standalone labels and a general two-encoder
plus Nemotron-one-hot equal-expert blend. Neither diagnostic can be promoted by
this protocol, even if its point estimate is higher.

## Frozen promotion gate

The refusal override becomes a hidden-test submission candidate only if every
condition holds:

1. the prior 22-row pilot passed and all 253 full judgments are schema-valid;
2. the incumbent reproduces exactly 175/253 correct;
3. the candidate is correct on at least 179/253 rows, a gain of at least four;
4. it rescues at least four incumbent errors and harms at most two incumbent
   correct rows;
5. at least 75% of changed rows are beneficial;
6. at least 7/13 gold refusal rows are correct, versus 0/13 for the incumbent;
7. at least three of five source folds improve and at most one worsens;
8. a 100,000-draw source-group bootstrap with seed 20260806 has probability of
   positive gain at least 0.90 and 5th-percentile delta at least zero; and
9. no current/hidden test input, label, upload score, or leaderboard feedback
   is used.

Failure retains the 0.691700 ordinal incumbent and stops confidence/prompt/rule
tuning. Passing promotes only this frozen refusal override; it does not license
the standalone judge, the diagnostic blend, positive-extreme overrides, silver
label training, or a larger model without a new preregistration.
