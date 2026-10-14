# Task 1 dual-raw-proposer preregistration

Date frozen: 2026-08-05 (before running the new policy on either the 80-row
FinEval validation labels or the 100-row historical Plutus test labels)

## Motivation and evidential status

An audit of the already-inspected FinEval validation residuals showed that
several missing exact entities are present in the raw GLiNER outputs from both
independent proposer seeds even when the existing annotation-operation views
reject them.  That observation motivates this experiment, so the 80-row
validation split is **not** represented as pristine hypothesis-generation
data.  It is used only for a frozen incremental no-harm check.  Transfer to the
separately released 100-row Plutus test set is evaluated in the same run as an
additional replication; the future hidden FinEval test remains decisive.

## Frozen inputs

| Role | Path | SHA-256 |
|---|---|---|
| train labels | `data/external/plutus_finner_text/train-00000-of-00001.parquet` | `e02165bae337c4d4f465d297bea4dc74a3f99fbc5b0a955358299ff657819c8c` |
| seed-7 genuine OOF proposals | `artifacts/task1_gliner_deploy_seed7_width16/oof_raw_predictions.jsonl` | `2a504b6e162704e0aa94b1e21eed34aaed83df75c8b38674f7f01f0317acd474` |
| seed-19 genuine OOF proposals | `artifacts/task1_gliner_deploy_seed19_width16/oof_raw_predictions.jsonl` | `cb0a16215a57b8fbf60fff2e88bbfde53ea01d3bf2e9cc9f04c53926fdb984e9` |
| OOF nested-BERT incumbent | `artifacts/task1_multilabel_oof_legal_seed7/oof_predictions.jsonl` | `87534061311ebe96c44abb03b172720c6fb22e85dc1c1d18ab0c975200d723ff` |
| validation seed-7 vote view | `artifacts/task1_gliner_deploy_seed7_width16/validation_ensemble_vote3_raw.jsonl` | `1f2c726f67e3755b7928bf78895c294c9b725a434314d2071e07580ea9b32244` |
| validation seed-19 vote view | `artifacts/task1_gliner_deploy_seed19_width16/validation_ensemble_vote3_raw.jsonl` | `299ec622db777eb71b927a2ff6fa7d4bb4bfe74f9b506b5cc536ea42f85b36fd` |
| validation incumbent | `artifacts/task1_conservative_composition_grouped_seed_v1/predictions.jsonl` | `837a7a9dcda8cffba4e28658c6f582642053d44e6c838db00bd188d88d000f69` |
| historical seed-7 vote view | `artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_enhanced_work/seed7/ensemble_vote3_raw.jsonl` | `410a618a4ab0269fe09cd79d98e15af496ba0751fb39b6528120e9aa68ea650e` |
| historical seed-19 vote view | `artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_enhanced_work/seed19/ensemble_vote3_raw.jsonl` | `067de32aa2d7d0a1c80b7c90db4d82c08b0ec6dff96f5fab0ce737406ae16d82` |
| historical incumbent | `artifacts/submission_portfolio_task1_enhanced_public_test_v1/task1_grouped_consensus_primary.jsonl` | `8a084a3d9e537bbb9ca2a9d973c5d4f45a29b1759882481bd4eabe6486e93799` |
| historical labels | `data/external/plutus_finner_text/test-00000-of-00001.parquet` | `2fcf02878fd8fbfbec8e0a72ad5454fbb7828b6b43fe82d4f2647199929677bb` |

Both OOF proposer sets cover 320 rows exactly once.  Seed 7 uses the same
entity-linked fold assignment as the OOF incumbent; seed 19 uses an independent
entity-linked assignment.  Both split audits report zero linked
organization/person keys crossing a model's train/held boundary.

## Frozen policy

1. A proposal is eligible only when the two raw OOF views agree exactly on
   `(row, character start, character end, entity type)`.  Its score is the
   minimum of the two proposer scores.
2. Eligible spans are converted into the unchanged add/replace/nesting
   operations and feature representation from
   `evaluate_task1_oof_annotation_program.py`.
3. The unchanged character-context plus numeric logistic classifier is trained
   in nested OOF fashion on the seed-7 entity-linked folds.
4. Candidate probability thresholds are the fixed grid 0.05, 0.06, ..., 0.99.
   A threshold is admissible only if it:
   - selects at least 20 operations;
   - has a positive row-bootstrap 5th-percentile mean delta;
   - produces a strictly positive mean delta in every one of the five folds;
   - has no negative aggregate entity-type F1 delta; and
   - has harmful-operation fraction at most 0.15, where an operation is
     harmful only when its individual row F1 delta is below zero.
5. Among admissible thresholds choose the largest tuple
   `(mean OOF delta, bootstrap p05, threshold)`.  If none is admissible, the
   policy abstains and validation/historical predictions must remain unchanged.
6. Fit the classifier once on all eligible OOF operations.
7. At deployment, intersect the seed-7 and seed-19 **vote-3** views by exact
   typed character span.  Each seed must therefore supply at least three fold
   votes.  Use the minimum of the two mean-present scores.  No mean-only or
   one-seed proposal is eligible.
8. Generate operations relative to the frozen grouped incumbent and apply the
   classifier at the frozen OOF threshold.  No validation-derived lexical
   exception or threshold adjustment is permitted.

## Frozen promotion rule

The candidate replaces the Task 1 textual primary only if all of the following
hold:

- FinEval validation: positive incremental macro Entity F1, at least two rows
  improved, zero rows harmed, every entity-type delta nonnegative, and
  row-bootstrap probability of gain at least 0.80;
- historical Plutus test replication: nonnegative incremental macro Entity F1,
  more improved than harmed rows, bootstrap probability of gain at least 0.60,
  and no entity-type delta below -0.005; and
- the implementation/replay tests pass.

If the gate fails, retain the current `0.933157` grouped validation incumbent.
The first results are final for this exact hypothesis; no post-hoc repair is
allowed.
