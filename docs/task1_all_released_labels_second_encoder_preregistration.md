# Task 1 public-label transfer: second-encoder replication

Date: 5 August 2026. Team: ADML.

## Purpose

Replicate the positive public-label transfer result with the independently
initialized `amichailidis/greek_legal_bert_v2-finetuned-ner-V3` encoder before
retaining its all-released-label refit in the optional Task 1 candidate. This
tests whether the gain is supervision-driven rather than specific to one base
checkpoint.

## Frozen protocol

- Training: Apache-2.0 Plutus GRFinNER `train` plus published `test` labels.
- Evaluation: all 80 Plutus `validation` rows, excluded from training.
- Encoder: `amichailidis/greek_legal_bert_v2-finetuned-ner-V3`; its incompatible
  three-label head is reinitialized exactly as in the incumbent experiment.
- Flat seven-label BIO architecture; current corrected aligner.
- Seed 7; AdamW; learning rate 3e-5; weight decay 0.01; batch size 8.
- Eight epochs; report the best validation checkpoint in the same fixed window.
- Frozen historical train-only baseline: macro Entity F1 0.858097527473;
  person 0.818181818182, organization 0.881057268722, location
  0.776470588235.
- Current/hidden FinEval inputs and answers are excluded.

## Replication gate

Retain the second all-label deployment encoder only if every condition holds:

1. macro Entity F1 is strictly above 0.858097527473;
2. at least two of the three type F1 values are nonnegative relative to the
   frozen baseline;
3. no type regresses by more than 0.01;
4. at least two validation rows improve;
5. row-bootstrap probability of positive mean delta is at least 0.80;
6. training metadata explicitly records the public-test hash and held-out
   validation provenance.

If the gate fails, the already trained all-label copy of this encoder is
removed from the deployment ensemble. The optional lane will instead combine
the demonstrated all-label legal-base encoder with the existing 395-row
full-refit legal-NER encoder and the original finance-adapted diversity model.
No threshold, epoch, seed, or row filter will be changed after evaluation.
