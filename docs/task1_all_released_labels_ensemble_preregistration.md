# Task 1 public-label transfer: ensemble replacement gate

Date: 5 August 2026. Team: ADML.

## Question

Does the strongly verified train+published-test legal-base checkpoint improve
the actual frozen three-model flat ensemble when it replaces only the original
train-only legal-base member? This is the configuration-level test required
before treating the all-release deployment lane as more than model-level
diversity.

## Frozen baseline

The baseline is equal-logit averaging of the original train-only legal-base,
legal-NER-initialized, and finance-adapted checkpoints, followed by the frozen
release encoder and high-support nested-location rules. On the 80 validation
rows it scores **0.916783424908**. Type F1 is person 0.818181818182,
organization 0.929203539823, and location 0.857142857143. The reproduced
prediction SHA-256 is
`4966d63b504d1c10cb6d3801024d1a468483b83d3a7dd4c1040e7a0a2e4e960c`.

## Candidate

Replace only the original legal-base member with
`artifacts/task1_textual_train_plus_public_test_seed7_v1`, which trained on
Plutus train+published-test and kept validation held out. Retain the original
train-only legal-NER and finance-adapted models, equal logit weights, decoder,
release encoder, and postprocessors. Do not test weights, thresholds, or rules.

## Promotion gate

The ensemble-level transfer is confirmed only if all conditions hold:

1. macro Entity F1 is strictly above 0.916783424908;
2. at least two entity types are nonnegative relative to baseline;
3. no entity type regresses by more than 0.01;
4. at least two validation rows improve;
5. row-bootstrap P(gain) is at least 0.80;
6. the candidate contains exactly 80 aligned rows and records the public-label
   model provenance.

If the gate passes, the analogous deployment ensemble uses the 494-row
all-release legal-base refit plus the existing 395-row legal-NER refit and the
original finance diversity member. If it fails, the all-release model remains
standalone scientific evidence but the external-data submission lane is
demoted rather than tuned post hoc. Current/hidden FinEval data is excluded.
