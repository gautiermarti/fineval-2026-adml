# Task 2 adjudication-disagreement weighting preregistration

Frozen: 5 August 2026, before training or evaluating any disagreement-weighted
model. Team: ADML.

## Motivation and disclosure

The official EBISU Label Studio export contains 185 judgments aligned to 94 of
the 253 released JF-ICR rows. Eighty rows are unanimous and 14 contain label
disagreement. The frozen two-seed ordinal incumbent is correct on 81.25% of the
unanimous rows but only 50.0% of disagreement rows, so disagreement is a useful
hard-example marker.

It is not safe to train on the raw annotation distribution. The adjudicated
gold is present in every row but is not the plurality label in IDs 38 and 86;
more generally, disagreement often places the adjudicated label at a stronger
ordinal extreme than the other judgments (`+2` versus `+1`, `-2` versus `0`,
or `-1` versus a milder label). Soft targets would erase those rare distinctions.

This experiment therefore preserves the released official gold exactly and
uses only a binary disagreement flag to upweight the hard example. It remains
policy-gated: the EBISU GitHub repository has no explicit license even though
the linked Hugging Face dataset declares Apache-2.0. No resulting model is
submission-eligible without organizer clarification.

## Frozen inputs

- JF-ICR parquet SHA-256:
  `c6575169894835dd7264f0e90768a924c4a084474c9b8fa30f1739dcdab43e95`
- source folds SHA-256:
  `d1189522a05fc1a1a0d8899d35d9303c4df9033716c86c0fbeb65ad3dfad1f79`
- curated annotation rows SHA-256:
  `b0023a2e3fa7b6ec505ca19667ec831105a138107fa1b1a8948a2f153caac452`
- raw export SHA-256:
  `970c6ca44d1319b962ce15855c0df345752b45ff89b02d3db76cce9dafb2e375`
- disagreement audit summary SHA-256:
  `530ae813771b8e21d57e93ec619d5ec52a9298cabf50de9a08327597d5bbbf1d`
- general seed-7 OOF SHA-256:
  `4724c9d3bee05185bcd99305cec984cd387c9c921ae98a25c703eddf748cb0b0`
- general seed-19 OOF SHA-256:
  `526504e29d1d97894997861587df5118ed15acb458846c9890fb5c5d27f4a0e2`
- base encoder: `cl-tohoku/bert-base-japanese-v3`, pinned Hugging Face
  revision `65243d6e5629b969c77309f217bd7b1a79d43c7e`

## Frozen model and loss

The architecture and optimizer exactly match the general seed-7 incumbent:

- pair input: parsed question plus response;
- longest-first truncation, maximum 512 tokens;
- five-way sequence-classification head ordered `-2,-1,0,+1,+2`;
- seed 7, four epochs, batch size 8, AdamW learning rate `2e-5`, weight
  decay `0.01`;
- no rationale head, no soft target, no new token, and no inference-time use of
  annotations.

For an official training row whose valid annotation labels contain at least two
distinct values, multiply its ordinary hard-label cross-entropy by exactly
`2.0`. Every other row has weight `1.0`. Normalize each batch by the sum of
weights. The official gold remains the only class target. No entropy-dependent
weight, label-specific weight, oversampling, calibration, or threshold search
is allowed.

Held-fold rows and their disagreement flags must not enter the training
dataset. Validation is built only from the ordinary released query and gold;
no annotations or rationales are loaded into it.

## Pilot and fixed profiles

Source fold 2 is the sole pilot. It has 56 rows and zero rows in the annotation
export, making it the cleanest generalization check for this supervision. The
existing two-general-seed ordinal incumbent and general seed 7 each score
34/56 on this fold.

Apply the existing expected-score rounding and safe-rule override unchanged to:

1. weighted seed 7 alone;
2. **primary replacement:** equal probability mean of weighted seed 7 and
   general seed 19;
3. **secondary diversity:** equal probability mean of general seeds 7 and 19
   plus weighted seed 7.

No ensemble weight or decoder may be tuned. The pilot advances only if all are
true:

1. weighted seed 7 alone is at least 34/56;
2. the primary replacement is at least 35/56;
3. the primary replacement rescues more incumbent errors than it harms;
4. primary harms are at most two; and
5. the secondary diversity profile does not underperform the incumbent.

If any condition fails, stop without training other folds or a second seed.

## Full confirmation gate

If the pilot passes, train the same seed-7 method on folds 0, 1, 3, and 4 in
that order. For every fold, only disagreement flags belonging to its training
partition may affect loss. Evaluate the fixed primary replacement over all 253
OOF rows against the frozen two-general-seed ordinal incumbent.

Promotion requires all of:

- accuracy strictly above 0.6916996;
- at least three folds improve and at most one worsens;
- total rescues exceed total harms;
- source-cluster bootstrap probability of positive gain at least 0.90;
- bootstrap 5th-percentile delta non-negative.

Even if the statistical gate passes, competition use remains blocked until the
organizers confirm the annotation export's license and external-resource
policy. The current or hidden competition test is never accessed.

