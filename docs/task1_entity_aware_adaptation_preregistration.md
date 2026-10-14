# Task 1 entity-aware adaptation preregistration

Frozen on 8 August 2026 before building the silver corpus, adapting either
encoder, or scoring the held fold. Team: ADML.

## Question

Does whole-entity masked-language adaptation on high-precision Greek financial
spans improve GRFinNER more than ordinary random-token MLM on the identical
documents, chunks, initialization, seed, and update budget?

This is a bounded causal ablation, not another generic corpus expansion. The
existing grouped-consensus submission remains the incumbent unless the complete
multi-fold and cross-seed gate below eventually passes.

## Frozen data construction

- Base corpus: `data/processed/finance_corpus_url_recovered_v2`, exactly 273
  quality-controlled documents and 29,708,220 characters.
- Evaluation: entity-linked fold 0 from the existing five-fold seed-7 split of
  the 320 released GRFinNER training rows.
- All released fold-0 organization and person surface keys are excluded from
  the silver inventory. Held labels are used only for this exclusion and final
  evaluation, never as positive silver supervision.
- Training-label inventory: exact person, organization, and location surfaces
  from the other four folds. A corpus occurrence is admitted only when the
  surface occurs in at least two distinct documents and the persisted fold-0
  GLiNER teacher—trained without fold 0—predicts the exact same span and type.
- Official inventory: legal issuer names parsed from the immutable ATHEX URL
  ledger. An occurrence is admitted only inside a document carrying the same
  issuer ID, after the same held-surface exclusion, and only when that surface
  occurs in at least two documents for the issuer. This source-metadata match
  is sufficient without GLiNER agreement.
- Exact normalized containment with any released GRFinNER train or validation
  text rejects a corpus chunk. Validation answers are never loaded.
- Documents are split deterministically into non-overlapping 180-word chunks.
  Chunks are ranked by SHA-256 of their stable document ID and text; retain at
  most 4,096 admitted chunks. Fail closed below 512 chunks or 1,000 spans.
- Every persisted span must be an exact substring, have a supported entity
  type, fit inside its chunk, and survive longest-first same-type overlap
  resolution.

The GLiNER checkpoint is
`artifacts/task1_gliner_deploy_seed7_width16/fold_0/final`, derived only from
the four training folds, with a frozen proposal threshold of 0.50.

## Frozen adaptation ablation

Both arms start from `spyrosbriakos/greek_legal_bert_v2` revision
`e06fba0eb7a107f401055bb174f9e1ba1a29ba17` and use the identical ordered
silver chunks:

1. `matched_random_mlm`: ordinary 15% token MLM.
2. `whole_entity_mlm`: select one admitted entity per example, mask its entire
   token span, then add random non-special tokens until at least 15% of eligible
   tokens are selected. The selected entity remains whole even if it exceeds
   that budget.

Both use seed 7, maximum length 256, batch size 16, AdamW at 5e-5 with weight
decay 0.01, FP32, exactly 256 optimizer updates, no scheduler, no warmup, and
no gradient accumulation. Standard 80/10/10 mask/random/unchanged replacement
is applied to every selected token. Non-finite loss fails closed.

## Frozen downstream test

Fine-tune three encoders independently with a fresh seven-label BIO head:

1. the unadapted pinned base;
2. `matched_random_mlm`;
3. `whole_entity_mlm`.

Use only the 256 non-fold-0 rows, seed 7, seven fixed epochs, batch size 8,
maximum length 512, AdamW at 3e-5 with weight decay 0.01, and FP32. There is no
early stopping or held-fold model selection. Evaluate fold 0 exactly once after
epoch 7 with the existing alignment, decoder, row-macro Entity F1, and per-type
micro Entity F1.

## Fold-0 futility gate

Let the higher-macro arm among the unadapted base and matched random MLM be the
control; an exact macro tie selects matched random MLM. Advance only if
whole-entity MLM satisfies every condition:

- macro Entity F1 is at least 0.005 above each control;
- rescues strictly exceed harms relative to the stronger control;
- at least two entity types are nonnegative relative to the stronger control;
- no entity type regresses by more than 0.01; and
- corpus construction and every provenance/hash check passed.

Failure is a stop for this recipe: do not change the masking rate, update
budget, epoch count, threshold, corpus cap, or fold after seeing the score.

## Conditional full experiment

Only a fold-0 pass authorizes fold-specific silver construction and matched
adaptation for folds 1–4. Full seed-7 OOF promotion requires macro gain at least
0.005 over matched random MLM, at least four improving folds, at most one
regression, no type delta below -0.01, row-bootstrap P(gain) at least 0.95, and
p05 at least zero. Only that pass authorizes an unchanged seed-19 replication.
The current validation, published test, and hidden competition packet remain
untouched throughout this experiment.
