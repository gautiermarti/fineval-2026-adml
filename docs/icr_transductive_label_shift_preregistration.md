# Task 2 transductive label-shift preregistration

Frozen before evaluating any label-shift-adjusted held-fold prediction on
5 August 2026. Team: ADML.

## Hypothesis

The hidden JF-ICR packet may differ from the public sources mainly in its class
mixture rather than its class-conditional language. The unlabeled target
packet's aggregate frozen-model probabilities can estimate that mixture. A
regularized correction may therefore improve the already-frozen two-seed
ordinal ensemble without using target labels or fitting new neural parameters.

## Primary method

For each of the five reconstructed-source folds:

1. Use the frozen seed-7/seed-19 mean OOF probabilities.
2. Estimate a soft confusion matrix on the other four folds only.
3. Collapse labels into three ordered bins: negative (`-2/-1`), neutral (`0`),
   and positive (`+1/+2`). This avoids estimating two extremely rare classes
   independently.
4. Estimate the held-fold prior by constrained least squares on its **unlabeled**
   mean probability vector, with squared shrinkage of 0.1 toward the four-fold
   source prior.
5. Reweight each held-fold posterior by target/source bin-prior ratio, preserving
   the model's within-bin class odds. Decode by the incumbent's parameter-free
   expected ordinal score and apply the unchanged safe rules last.

The primary is named `soft_bbse_3bin_ridge_0.1`. No held-fold label may enter
its prior estimate or posterior adjustment.

## Fixed sensitivity controls

- `soft_bbse_3bin_ridge_1.0` tests stronger shrinkage.
- `soft_bbse_5way_ridge_0.1` tests whether collapsing rare labels is necessary.
- `em_3bin_shrink_0.5` tests an iterative EM-style prior estimate.
- `identity` must reproduce the 0.691700 ordinal incumbent exactly.

These controls diagnose the mechanism. A favorable control does not replace
the primary post hoc; it can motivate a separately preregistered replication.

## Promotion gate

Promotion requires all of the following against the unchanged ordinal system:

- accuracy strictly above 0.691700;
- at least three source folds improve and no more than one regresses;
- source-cluster bootstrap P(gain) at least 0.90 and 5th-percentile delta
  non-negative;
- no loss of recall for the combined negative labels or `+2`;
- the adjustment remains finite, normalized, and label-free on every fold.

Failure retains the ordinal incumbent and records the result as a negative
control. The current or hidden competition test packet is not used in this
experiment.
