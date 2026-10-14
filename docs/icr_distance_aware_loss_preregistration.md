# Task 2 distance-aware five-class loss preregistration

Date frozen: 5 August 2026, before training or scoring this loss.

## Motivation and single hypothesis

The incumbent is a conventional five-class Japanese-BERT classifier whose
probabilities are decoded by expected ordinal score. The rejected cumulative
head changed the output architecture, and the rejected external multi-task
model used pairwise ranking plus unrelated promise objectives. Neither tested
the narrower hypothesis here: retain the successful five-class head and add a
small proper distributional penalty that distinguishes adjacent from extreme
errors during training.

For ordered labels `[-2, -1, 0, +1, +2]`, let `p = softmax(logits)` and let
`F_p(k) = sum_{j<=k} p(j)` for the first four thresholds. For gold class `y`,
let `F_y(k) = 1[y<=k]`. The only candidate loss is:

`cross_entropy(logits, y) + 0.5 * mean_k (F_p(k) - F_y(k))^2`.

The coefficient is fixed at **0.5** before results. It will not be swept,
adapted by class, or selected by a fold. Cross-entropy remains dominant; the
normalized Cramér/ordered-CDF term only teaches the geometry already assumed
by the deployed expected-score decoder.

## Frozen training configuration

- Base: `cl-tohoku/bert-base-japanese-v3`, Hugging Face revision
  `65243d6e5629b969c77309f217bd7b1a79d43c7e`.
- Architecture: unchanged five-logit sequence-classification head.
- Data: official 253-row JF-ICR public set only.
- Splits: the existing five `source_group`-disjoint folds, all run in the pilot.
- Seed: 7; four epochs; batch size 8; maximum length 512; longest-first pair
  truncation; AdamW at `2e-5`, weight decay `0.01`; FP32 training.
- No class weights, resampling, external data, silver labels, rules in the loss,
  early stopping, scheduler, or fold-dependent behavior.
- The evaluator uses the frozen expected-score decoder and the existing
  input-only `safe_rules`, exactly as the incumbent does.

Every fold records cross-entropy, CDF loss, total loss, raw argmax accuracy,
and logits. Non-finite components fail closed. The new script must reproduce
ordinary cross-entropy when the distance coefficient is zero in unit tests,
although coefficient zero is not a scored experiment.

## Frozen seed-7 profiles

The baseline probabilities are the already-frozen general seed-7 and seed-19
OOF files. Evaluate exactly these profiles:

1. `distance_seed7` alone;
2. `primary_replace_general_seed7`: equal probability mean of distance seed 7
   and general seed 19;
3. `secondary_general2_plus_distance_seed7`: equal probability mean of both
   general seeds and distance seed 7.

No alternative weight, logit blend, class-specific route, temperature,
threshold, or post-hoc subset is eligible.

## Advance and promotion gates

Train distance-aware seed 19 only if the seed-7 primary replacement satisfies
all of the following on the full source-disjoint OOF set:

- at least one more correct row than the incumbent;
- rescues strictly exceed harms;
- at least two source folds improve and at most one worsens;
- the secondary diversity profile does not score below the incumbent.

If any check fails, stop this branch. Do not try another coefficient, seed,
fold order, model, loss normalization, or profile.

If seed 19 is authorized, final promotion still requires the established
strict gate: overall accuracy strictly above the incumbent, at least three of
five source folds improved, at most one worsened, source-bootstrap probability
of positive gain at least 0.90, and 5th-percentile gain at least zero. A final
two-seed distance ensemble and the equal four-model diversity ensemble may be
evaluated; their definitions must be frozen before seed-19 training.

Post-hoc error labels and changed rows may explain a result but cannot select a
new configuration. Current or hidden test answers are not used.
