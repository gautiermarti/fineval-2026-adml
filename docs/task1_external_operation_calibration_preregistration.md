# Task 1 external operation-calibration preregistration

Frozen before fitting the combined operation selector or scoring its validation
predictions. Date: 5 August 2026. Team: ADML.

## Motivation and evidential status

The promoted textual system scores 0.933157 on the 80-row validation split.
An already-inspected residual audit shows that existing raw proposers contain
the exact missing entity for 10 of the 11 imperfect rows; the remaining row has
a gold location absent from its input. Candidate recall is therefore not the
main bottleneck. The previous dual-raw operation selector was trained only on
320 OOF rows. It transferred one no-harm repair to the independently released
100-row Plutus test set but abstained on validation because OOF and deployment
proposal scores are differently calibrated.

This experiment tests one fixed remedy: treat those 100 genuine public labels
as external operation-calibration data. The validation residuals motivated the
hypothesis, so validation is not claimed as pristine discovery data. All
choices below are frozen before the combined model is run.

## Frozen data and proposals

- Primary training pool: 320 Plutus GRFinNER train rows, genuine entity-linked
  OOF seed-7/seed-19 raw proposals, and the frozen OOF nested-BERT incumbent.
- External calibration pool: the independently published 100-row Plutus test
  split, the seed-7/seed-19 vote-three proposal files, and grouped incumbent
  predictions generated without reading those 100 answers.
- Final evaluation: the separate 80-row Plutus validation split, its frozen
  seed-7/seed-19 vote-three proposals, and the promoted 0.933157 incumbent.
- Current/hidden FinEval inputs and answers are excluded.

For every pool, a proposal is eligible only when both seeds agree on exact
character start, end, and entity type. Its score is the minimum seed score.
The unchanged add/replace/nesting operation construction and the unchanged
character-context plus numeric logistic feature representation from the
dual-raw experiment are used.

## Frozen calibration procedure

1. Build operations on the 320-row OOF pool with their original five
   entity-linked folds.
2. Build operations on all 100 external rows and assign them to a sixth,
   indivisible calibration group. Offset their row indices only for combined
   evaluation; no text, entity, or label is changed.
3. Cross-fit operation probabilities over the six groups. When group 5 is
   held out, no external test label is in its classifier; when any OOF fold is
   held out, the other four folds plus the external group may train it.
4. Evaluate the fixed threshold grid 0.05, 0.06, ..., 0.99 on the combined
   cross-fitted predictions. A threshold is admissible only if it selects at
   least 25 operations, has positive row-bootstrap p05, has strictly positive
   mean delta in each of all six groups, has no negative aggregate entity-type
   delta, and has harmful-operation fraction at most 0.15.
5. Select the admissible threshold maximizing
   `(combined mean delta, bootstrap p05, threshold)`. If none exists, abstain.
6. Fit the unchanged classifier once on all combined operations. Apply it once
   to the 80 validation rows. No validation-derived threshold, signature,
   lexical exception, model choice, or feature is permitted.

The classifier uses `max_features=20000`; no alternate regularization, data
weight, seed, score transform, or proposal view is screened.

## Promotion gate

The candidate replaces the textual primary only if every condition holds:

1. validation macro Entity F1 improves by at least 0.005 over 0.933157;
2. at least two rows improve and zero rows worsen;
3. all three entity-type F1 deltas are nonnegative;
4. validation row-bootstrap probability of gain is at least 0.90;
5. the six-group training gate exists and every implementation/replay test
   passes.

Failure stops this calibration branch; no post-hoc exception or second
threshold is allowed. Passing permits a separately named full external-label
deployment refit, provided the competition's external-data policy remains
compatible. Exact lookup is not part of this experiment.
