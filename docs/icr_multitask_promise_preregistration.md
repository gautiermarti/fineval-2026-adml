# Task 2 ordinal-promise multi-task preregistration

Frozen before any source-disjoint result from this architecture. Date:
5 August 2026. Team: ADML.

## Hypothesis

The benchmark-disjoint ML-Promise signal is ordinally associated with JF-ICR,
but adding its frozen probabilities after the incumbent is redundant and
slightly harmful. The remaining hypothesis is representation transfer: jointly
learning the exact promise event and ordinal pair ordering may teach Japanese
BERT a decisive-event representation that cannot be recovered by a shallow
router over frozen probabilities.

## Fixed pilot

- Encoder: `cl-tohoku/bert-base-japanese-v3`, the incumbent architecture,
  pinned at Hugging Face revision
  `65243d6e5629b969c77309f217bd7b1a79d43c7e`.
- JF objective: ordinary five-way cross entropy, weight 1.0.
- Pairwise objective: hinge ordering of softmax expected scores, margin 0.25
  per label level, weight 0.20.
- External objectives: ML-Promise paragraph status, weight 0.25; exact promise
  token span, weight 0.10 with fixed positive-token weight 3.0. One of 315
  positive spans tokenizes to 527 tokens and cannot fit BERT's 510-token content
  window; that row retains paragraph supervision but its token loss is masked.
  The other 314 positive spans are retained completely rather than truncated;
  annotation-only whitespace differences are mapped back to exact source-text
  character bounds before tokenization.
- Data: 357 curated Japanese ML-Promise training rows from six non-overlapping
  issuers. Its official test partition remains sealed.
- Optimization: seed 7, four epochs, batch 8 for each task, AdamW at 2e-5 and
  weight decay 0.01, maximum length 512.
- First result: reconstructed-source fold 0 only. No architecture or loss
  weight changes may be made from that fold result.

The CPU smoke cap is not an experiment and cannot be reported as performance.

## Futility and promotion gates

Apply the same safe-rule and expected-score ordinal decoder as the incumbent.
Stop after fold 0 if accuracy is below the incumbent's 0.588235 on that fold,
or if rare-boundary recall fails to improve while more than two previously
correct rows are harmed. If it survives, run all five folds for seeds 7 and 19.

Promotion requires full source-disjoint accuracy at least 0.72, at least three
folds improved or tied, no more than one fold regression, and source-cluster
bootstrap P(gain) at least 0.90. The candidate is used only as an abstaining
residual or a predeclared equal-weight diversity view unless that complete gate
passes. Mandatory ablations are ranking weight zero and both external-objective
weights zero; they are run only after the full model survives fold 0.

No hidden-test input, label, or leaderboard result may change this design.
