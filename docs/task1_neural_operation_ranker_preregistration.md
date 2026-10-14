# Task 1 neural operation-ranker preregistration

Frozen on 5 August 2026 before this architecture is evaluated. Team: ADML.

## Hypothesis and fixed design

The promoted linear annotation program proves that beneficial GLiNER versus
incumbent disagreements are learnable. A Greek legal cross-encoder may improve
that bounded operation decision if it first learns true span boundaries from
the 7,945 entity-linked span-surgery pairs, then fine-tunes on genuine OOF
add/replace/nesting operations. It may approve an enumerated operation or
abstain; it may never generate a new span.

- Encoder: `spyrosbriakos/greek_legal_bert_v2`, revision
  `e06fba0eb7a107f401055bb174f9e1ba1a29ba17`.
- Input: complete row/context with candidate boundaries marked by the encoder's
  existing `[MASK]` token, plus entity type, candidate surface, operation and
  removed incumbent entities as the paired sequence.
- Stage 1: one epoch of weighted pairwise softplus ranking on synthetic surgery
  pairs, batch 16, maximum length 256, AdamW 2e-5, weight decay 0.01.
- Stage 2: six epochs of unweighted binary beneficial-operation fine-tuning on
  genuine OOF disagreements, batch 16, AdamW 1e-5, weight decay 0.01.
- Decoder: fixed sigmoid threshold 0.5 followed by the existing deterministic
  conflict-resolving operation applicator. No threshold is learned from the
  held fold.
- Seed: 7. All surgery pairs and genuine operations from the held entity-linked
  fold are excluded from both stages.

## Futility gate

Run fold 0 only. Advance only if the row-macro Entity-F1 gain is at least the
already observed linear policy's fold-0 gain, +0.0778831845, and no more than
two incumbent-correct rows are harmed. This deliberately requires practical
improvement over the existing method, not merely a positive neural result.

If it advances, run all five entity-linked folds. Full OOF promotion eligibility
requires at least four improving folds, at most one regression, no entity-type
delta below -0.005, bootstrap P(gain) at least 0.95 and p05 at least zero. Only
then run the no-surgery-pretraining ablation and touch the 80-row validation
once. Deployment additionally requires agreement across two proposer seeds or
conformal abstention. The public test and hidden shared-task test are excluded
from all architecture, threshold and gate decisions.

