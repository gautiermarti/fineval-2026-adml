# Task 1 all-released-labels transfer preregistration

Date: 5 August 2026. Team: ADML.

## Question

Does adding the 100 labeled examples in the independently published Apache-2.0
Plutus GRFinNER test split improve generalization to the original 80-row
validation split under the already selected Greek legal-BERT architecture?
This is an external/public-data experiment, not hidden-test adaptation.

## Data boundary and provenance

- Training: Plutus GRFinNER `train` plus published `test` only.
- Evaluation: Plutus GRFinNER `validation` only.
- The validation rows and labels are excluded from training.
- Current/hidden FinEval inputs and answers are excluded.
- The three input files and their hashes are recorded in
  `data/external/plutus_finner_text/README.md`.
- Any eventual model trained on train+validation+test is a separately named
  policy-gated candidate until the organizers explicitly permit public/external
  labeled data.

## Frozen pilot

- Encoder: `spyrosbriakos/greek_legal_bert_v2`.
- Seed: 7.
- Flat seven-label BIO head and the existing corrected one-to-one aligner.
- AdamW, learning rate 3e-5, weight decay 0.01, batch size 8.
- Eight epochs are run to reproduce the original train-only selection window;
  the best validation epoch is reported.
- Baseline: the already persisted train-only seed-7 legal-base checkpoint,
  0.8573489011 validation macro row-level Entity F1 before downstream ensemble
  and release-encoding corrections.

## Advancement gate

The extra-label route advances to an all-500-row refit only if all of these
hold:

1. pilot validation macro Entity F1 is strictly above 0.8573489011;
2. at least two of person, organization, and location type F1 do not regress;
3. no type loses more than 0.01 absolute F1;
4. the pilot improves at least one validation row and does not obtain its
   entire gain from a single row;
5. every training record and the resulting checkpoint record the public-test
   provenance explicitly.

If the pilot passes, the deployment refit uses all three released splits and a
fixed five epochs, approximately preserving the update budget of the current
six-epoch 395-row refit. It remains a diversity/policy-gated lane; neither this
pilot nor the hidden packet may be used to tune architecture, seed, or epoch
count further.

## Interpretation

A positive result supports the ordinary sample-size hypothesis: more genuine,
independently released Greek financial NER annotations help under split shift.
It does not license exact answer lookup and does not prove that the competition
permits public-test labels. A negative result stops this branch without
post-hoc filtering or threshold changes.
