# Preregistration: historical-to-future sparse-edit selection

Date locked: 2026-08-05

## Motivation and data split

The frozen sparse-edit evaluator previously used only the 192 newer reference-
overlap rows.  Rebuilding the identical lattice over the full July historical
corpora yields 343 valid rows: 151 exact packet fingerprints absent from the
newer discovery lattice (`early_confirmation`) and all 192 frozen discovery
rows (`late_discovery`).  One early Easy cash row is excluded because its v15
baseline fails the existing validator.  Every late candidate, answer, and score
is copied from and therefore byte-aligned to the frozen lattice.

This is not a blind benchmark: the late references have been used in prior
experiments.  It is, however, a new **locked temporal/source generalization
test**.  Late labels may not train a selector, choose a threshold, or break a
tie.  The current FinNLP blind test is unavailable and unused.

## Frozen candidates and features

The complete v15 answer remains immutable except for the existing validated
operations: full, core-only, evidence-only, first-sentence, one-sentence
deletion, or baseline.  The learned selector reuses without modification the
earlier ridge feature function:

- character TF-IDF over tier, family, question, baseline, candidate, and edit;
- categorical tier/family/operation/variant;
- answer/core/evidence lengths and numeric counts;
- candidate-versus-baseline token similarity;
- input-source token coverage.

Reference text and late candidate scores are excluded from feature
construction.  Training targets are early candidate-minus-v15 ROUGE deltas.

## Frozen selectors

### 1. Exact signature gate

For `(tier, family, exact operation)`, admit an edit only when every available
training issuer has at least three examples, overall mean gain is at least
0.005, and the worst issuer mean is at least -0.002.  If several admitted edits
exist, choose the one with greatest training mean gain; ties favor baseline and
then the shorter answer.  These are the same thresholds used by the original
sparse family gate.

### 2. Ridge risk model

Use the already frozen grid:

- ridge alpha: `{1, 10, 100, 1000}`;
- minimum predicted gain: `{0, 0.0025, 0.005, 0.01, 0.02}`.

Within any training issuer set, choose alpha and threshold by inner
leave-one-issuer-out complete-answer ROUGE.  Exact score ties choose the largest
threshold and then the largest alpha.  Train once on all permitted early rows
and apply to late rows.

## Evaluation variants

Two variants are reported for each selector:

1. **Time-only:** train on all early issuers; evaluate all late issuers.
2. **Source-and-time disjoint (primary):** for each late held issuer, train only
   early rows from the other three issuers.  Hyperparameters/gates are also
   chosen without the held issuer.  Concatenate the four untouched late folds.

The second is the promotion test.  The first is diagnostic for future packets
from known issuers and cannot independently promote an edit.

## Frozen promotion gate

The primary source-and-time-disjoint variant must satisfy all conditions:

1. Mean complete-answer ROUGE is strictly above v15.
2. At least four rows change and improved rows outnumber harmed rows.
3. At least three of four late issuer folds are nonnegative.
4. At most one issuer fold regresses by more than 0.002.
5. Exact issuer-cluster bootstrap has P(gain) at least 0.90 and p05 at least 0.

No family, operation, threshold, issuer exception, or hybrid of the two
selectors may be introduced after late scores are observed.  Failure retains
v15.  A passing result may promote only the preregistered primary variant.

## Artifacts fixed before evaluation

- Historical lattice:
  `artifacts/polyfiqa_historical_sparse_lattice_v1/lattice.jsonl`
- Lattice summary SHA-256:
  `8906a689c4e4f571827bc6fe3eabbdf287b22f50a147672d52ad7dadaef6adf9`
- Lattice SHA-256:
  `5993531c1a627237be36a753c5364f061243d99bae5a71a318a99ce67e57483c`

