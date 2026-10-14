# Task 1 rank-invariant harm-aware hurdle preregistration

Date frozen: 5 August 2026, before computing any percentile-gated validation
prediction. Team: ADML.

## Motivation and evidential status

The preregistered harm-aware hurdle passed its complete six-group calibration
gate but applied zero validation edits.  Its cross-fitted operation scores
ranged from -0.952 to +0.900, while the independently fitted deployment scores
ranged from -0.855 to only +0.170.  The exact same raw-proposal domain shift was
seen in the earlier binary selector.  The model's relative ordering may
transfer even though its absolute probability scale does not.

This experiment changes exactly one component: replace the absolute hurdle
score by its empirical within-split percentile before applying a global gate.
The prior validation attempt accepted zero operations, so no validation
operation outcome from the hurdle was observed.  Broader validation residuals
are already known and the split is not claimed as pristine discovery data.

## Frozen data, model, and operation construction

- Use exactly the same 320 OOF rows in five entity-linked folds, 100 public
  Plutus test rows as an indivisible sixth group, 80-row validation split,
  incumbent files, exact dual-seed proposals, external/deployment vote-three
  requirement, operation construction, sparse feature representation, and
  `max_features=20000` as the harm-aware hurdle experiment.
- Refit the exact same two unweighted logistic heads with fixed `C=0.5`,
  `solver="liblinear"`, `max_iter=1500`, and `random_state=7`.
- The raw operation score remains
  `s = p(nonzero) * (2 * p(beneficial | nonzero) - 1)`.
- No type, operation-family, row, lexical, confidence, or model exception is
  permitted.

## Frozen rank transform

For every held calibration group independently, rank its raw scores in
ascending order with average ranks for exact ties.  For a group of `n > 1`
operations, define

`r = (rank(s) - 1) / (n - 1)`.

For the degenerate `n = 1` case, set `r = 1`.  This transform is label-free.
At deployment, the entire candidate-operation batch is one group and receives
the same transform.  Thus `r = 1` is the highest-scoring operation within its
own proposal regime, irrespective of absolute calibration scale.

## Frozen threshold and calibration gate

1. Cross-fit the unchanged hurdle over groups 0 through 5.
2. Compute the empirical percentile separately inside every held group.
3. Evaluate only percentile thresholds `0.50, 0.51, ..., 0.99`.
4. A threshold is admissible only if it selects at least 25 combined
   operations, has strictly positive row-bootstrap p05, has strictly positive
   mean delta in all six groups, has no negative aggregate entity-type F1
   delta, and has harmful-operation fraction at most 0.15.
5. Select the admissible threshold maximizing `(combined mean delta,
   bootstrap p05, threshold)`.  If none exists, abstain.
6. Fit the unchanged hurdle on all 420 calibration rows.  Compute validation
   operation percentiles without labels and apply the frozen threshold exactly
   once.  No absolute-score fallback or second percentile definition is tried.

## Promotion gate

Replace the textual primary only if all conditions hold:

1. validation macro Entity F1 improves by at least 0.005 over 0.9331570513;
2. at least two rows improve and zero rows worsen;
3. all three entity-type F1 deltas are nonnegative;
4. row-bootstrap probability of validation gain is at least 0.90;
5. the six-group calibration gate exists; and
6. all targeted and full repository tests pass.

Any failure stops this rank-invariant branch.  No threshold, quota, rank tie
method, grouping, exception, or deployment rule is changed after inspection.
Current or hidden FinEval answers are never used.
