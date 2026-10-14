# Task 2 rationale-diversity confirmation preregistration

Frozen after the source-fold-2 discovery result and before any rationale-model
result on source folds 0, 1, 3, or 4. Date: 5 August 2026. Team: ADML.

## Discovery result and why this is a new branch

The original human-rationale preregistration required the rationale model to be
non-inferior alone and limited harms from its two-model replacement. It failed:
on source fold 2 the rationale model scores 33/56 versus 34/56 for general seed
7, while the replacement blend makes four harms. The original branch is stopped
and its gate will not be changed.

A secondary profile fixed before that result nevertheless exposed a different
hypothesis: equal averaging of general seeds 7 and 19 with rationale seed 7
scores 38/56 versus the incumbent's 34/56, with five rescues and one harm.
An existing ordinary third seed also improves this fold, but only to 36/56
(four rescues, two harms), so some of the signal may be generic seed diversity.
This document defines an independent confirmation; fold 2 is discovery-only
and is excluded from every confirmatory promotion statistic.

Discovery artifact hashes:

- rationale fold-2 OOF:
  `72816ef51c5be84babce63d722a18c27bff5e8ab2f320e18775b3722be1b7eba`
- rationale fold-2 metrics:
  `2c80acdf5e3795e03d88c97339a7a35fd841b409acb8efd41c9bcac6dfb8deb5`
- frozen original evaluation:
  `25a91a38bb620eae55d05e6eed6918a82e0075e0d150c5f77a3db15f6796cf00`
- original preregistration:
  `870257cb14573e19f9bb109912ef18aae8e6d565c53435a4260eea9190cf3c88`

## Fixed candidate and controls

- Candidate model/training is unchanged from the original preregistration:
  Japanese BERT v3, seed 7, four epochs, batch 8, AdamW 2e-5, 512 tokens,
  rationale loss weight 0.10, positive-token weight 3.0, and fold-local seed
  `7 + 1000 * source_fold`.
- Primary candidate is the equal probability average of general seed 7,
  general seed 19, and rationale seed 7, followed by expected-score rounding
  and the frozen safe rules.
- Generic-diversity control is the already completed equal blend of ordinary
  general seeds 7, 19, and 31.
- Attribution control is the same rationale training code, fold-local seed, and
  optimization with rationale weight exactly zero. Its deployment profile is
  the equal blend of general seeds 7 and 19 plus that zero-weight model.
- No ensemble weight, threshold, rule, rationale span, loss weight, seed, or
  epoch count may be changed from a confirmation result.

## Sequential confirmation

Source fold 0 is the first confirmation because it is the lowest-numbered fold
not used for discovery. Train the rationale candidate there first. Run its
same-code zero-weight control only if the candidate blend:

1. is at least the incumbent's 30/51 = 0.588235;
2. harms at most two incumbent-correct rows; and
3. is at least as accurate as the fixed ordinary three-seed control.

After the control is available, advance only if the rationale blend is at least
as accurate as the zero-weight blend. If any condition fails, stop; source
folds 1, 3, and 4 are not trained for this branch.

If fold 0 passes, train candidate and zero-weight seed 7 on source folds 1, 3,
and 4. The confirmatory population is exactly the 197 rows in folds 0, 1, 3,
and 4. Fold 2 remains excluded from all conditions below.

## Confirmation and promotion gates

On the 197 confirmatory rows, the rationale three-model blend must:

- strictly exceed the two-seed incumbent and the zero-weight three-model blend;
- exceed the fixed ordinary seed-7/19/31 blend;
- improve at least two of four source folds versus the incumbent and worsen at
  most one;
- have source-cluster bootstrap P(gain) at least 0.90 and non-negative p05
  delta versus the incumbent; and
- have source-cluster bootstrap P(gain) at least 0.80 versus the same-code
  zero-weight blend, establishing that the gain is not merely a third seed.

As a final sanity check, its full 253-row score after restoring discovery fold
2 must strictly exceed 0.691700. The model remains ineligible for submission
unless organizers separately confirm that the EBISU annotation export and
external public resources are permitted. Failure of either statistical or
policy gate retains the existing policy-independent two-seed ordinal primary.
