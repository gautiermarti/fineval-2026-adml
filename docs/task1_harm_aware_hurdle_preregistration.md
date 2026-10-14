# Task 1 harm-aware operation-hurdle preregistration

Date frozen: 5 August 2026, before fitting or scoring the hurdle model.
Team: ADML.

## Motivation and evidential status

The six-group external operation-calibration experiment stopped because its
strongest threshold accepted 17 harmful operations among 74 edits, a 0.229730
harmful fraction above the frozen 0.15 limit.  Its binary classifier predicts
`beneficial` versus a negative class that combines 262 harmful operations with
52 score-neutral operations.  This experiment tests one structural remedy:
model whether an edit has a nonzero effect separately from the direction of a
nonzero effect.

The combined calibration labels and aggregate failure are already inspected.
The 80-row validation residuals motivated the broader operation program and
are not claimed as pristine discovery data.  No validation-derived lexical
exception, operation signature, threshold, or feature is permitted here.

## Frozen inputs and candidate construction

- Use exactly the 320 genuine entity-linked OOF rows and five folds from the
  prior experiment plus the independently published 100-row Plutus test split
  as one indivisible sixth group.
- Use the same incumbent predictions, exact typed intersection of seed-7 and
  seed-19 raw GLiNER proposals, vote-three requirement on the external and
  deployment splits, operation construction, character-context TF-IDF
  features, numeric features, and `max_features=20000`.
- Candidate operations and their row-level application order are unchanged.
- The 80-row validation split is loaded only after the complete six-group gate
  and final fitted components have been fixed.
- Current or hidden FinEval inputs and answers are excluded.

## Frozen hurdle model

For operation delta `d` define:

- consequence target `z = 1[abs(d) > 1e-12]`, trained on every operation;
- direction target `b = 1[d > 1e-12]`, trained only on consequential
  operations (`z = 1`).

Both heads use the unchanged sparse feature matrix and independent unweighted
`LogisticRegression(C=0.5, solver="liblinear", max_iter=1500,
random_state=7)` classifiers.  There is no class weighting, resampling,
feature selection, calibration transform, or hyperparameter sweep.

Let `p_z` be the consequence probability and `p_b` the probability of a
beneficial direction conditional on consequence.  The only operation score is

`s = p_z * (2 * p_b - 1)`.

This is the signed expected consequence under equal benefit/harm utility.  A
negative score predicts harm; a positive score predicts benefit.  Operations
are ranked and conflict-resolved exactly as before.

## Frozen cross-fitting and threshold gate

1. Cross-fit both heads over groups 0 through 5.  For each held group, neither
   head sees any operation label from that group.
2. Evaluate thresholds `0.00, 0.01, ..., 0.95` on the signed cross-fitted
   scores.  No second score, blend, type-specific threshold, or operation
   exception is evaluated.
3. A threshold is admissible only if it selects at least 25 operations, has
   strictly positive row-bootstrap p05, has a strictly positive mean delta in
   every one of the six groups, has no negative aggregate entity-type F1
   delta, and has harmful-operation fraction at most 0.15.
4. Select the admissible threshold maximizing `(combined mean delta,
   bootstrap p05, threshold)`.  If none exists, the policy abstains and no
   validation edit may be applied.
5. If a gate exists, fit both heads once on all 420 calibration rows and apply
   the frozen score and threshold to validation.  If no gate exists, still
   replay the incumbent as an implementation check, but do not inspect an
   alternative validation threshold.

## Promotion gate

Replace the textual Task 1 primary only if every condition holds:

1. validation macro Entity F1 improves by at least 0.005 over 0.9331570513;
2. at least two validation rows improve and zero worsen;
3. all three entity-type F1 deltas are nonnegative;
4. validation row-bootstrap probability of gain is at least 0.90;
5. the six-group calibration gate exists; and
6. all targeted and repository tests pass.

Any failure stops this hurdle-model branch.  No post-hoc threshold, alternate
head, coefficient, class weight, operation family, or validation exception is
allowed.  A passing result permits a separately named external-label full
deployment refit only if the competition's external-data policy remains
compatible.
