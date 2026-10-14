# Task 2 human-rationale multi-task preregistration

Frozen before any source-disjoint result from this architecture. Date:
5 August 2026. Team: ADML.

## Hypothesis and policy boundary

The official EBISU Label Studio export contains human-written decisive cues for
94 of the 253 released JF-ICR examples. Seventy-five rows yield at least one
unique exact response span from an annotation whose choice agrees with the
released label. A token-level cue-localization objective may teach the existing
Japanese BERT encoder what part of a full response controls pragmatic intent,
without requiring a rationale at inference time.

The pinned EBISU GitHub repository has no explicit license file, although its
linked Hugging Face JF-ICR dataset declares Apache-2.0. This experiment may be
reported as research, but it is **not submission-eligible** until the organizers
confirm both that external public data are allowed and that the annotation
export is covered. No hidden/current test input or label may enter training,
selection, or curation.

## Frozen inputs

- JF-ICR public parquet SHA-256:
  `c6575169894835dd7264f0e90768a924c4a084474c9b8fa30f1739dcdab43e95`
- Reconstructed source folds SHA-256:
  `d1189522a05fc1a1a0d8899d35d9303c4df9033716c86c0fbeb65ad3dfad1f79`
- Curated rationale JSONL SHA-256:
  `b0023a2e3fa7b6ec505ca19667ec831105a138107fa1b1a8948a2f153caac452`
- Raw Label Studio export SHA-256:
  `970c6ca44d1319b962ce15855c0df345752b45ff89b02d3db76cce9dafb2e375`
- Encoder: `cl-tohoku/bert-base-japanese-v3`, revision
  `65243d6e5629b969c77309f217bd7b1a79d43c7e`.
- Frozen incumbent OOF hashes: seed 7
  `4724c9d3bee05185bcd99305cec984cd387c9c921ae98a25c703eddf748cb0b0`;
  seed 19
  `526504e29d1d97894997861587df5118ed15acb458846c9890fb5c5d27f4a0e2`.

## Fixed model and training procedure

- Architecture: the incumbent five-class Japanese BERT sequence classifier,
  plus a two-class linear token head used only during training.
- Input: complete question/response pair under the incumbent `longest_first`
  512-token encoding.
- Primary loss: ordinary five-way cross entropy, weight 1.0.
- Auxiliary loss: response-token rationale/non-rationale cross entropy, weight
  0.10 and positive-token class weight 3.0. Question, special, padding, and
  every row without a recoverable cue are ignored by this loss.
- A cue is positive only when its non-unknown token sequence has exactly one
  match wholly inside the encoded response segment. The positive target is the union of all
  recoverable spans. If no positive span survives truncation/tokenization, the
  row has no token supervision; it is never turned into an all-negative row.
- A validation dataset is constructed with an empty rationale map. Held-out
  rationales and held-out labels are never accessed by the training loss.
- Optimization: seed 7, four epochs, batch 8, AdamW 2e-5, weight decay 0.01,
  maximum length 512. Each fold resets all random generators to
  `seed + 1000 * source_fold`, so an isolated pilot fold is bitwise comparable
  to the same fold in a later complete run. Evaluation uses the incumbent safe rules and
  expected-score rounding.

An exact same-code control uses rationale weight zero. It is required for
attribution if the rationale candidate survives the first pilot.

## Source-fold-2 futility pilot

Source fold 2 is the first and only pilot because the official rationale export
contains no row from that fold. It is therefore a strict test of transfer to a
completely unannotated held-out source family. The frozen incumbent two-seed
ensemble and the old general seed-7 model both score 34/56 = 0.607143 there.

The primary deployment-shaped pilot replaces incumbent general seed 7 with
rationale seed 7, keeps general seed 19 unchanged, averages probabilities,
then applies expected-score rounding and safe rules. Advance to all five folds
only if:

1. the replacement ensemble is at least 34/56;
2. rationale seed 7 alone is at least 34/56;
3. the replacement ensemble harms no more than two incumbent-correct rows; and
4. rationale seed 7 is not worse than the same-code zero-weight control.

The zero-weight control is run only if conditions 1--3 pass. No loss weight,
span rule, decoder, epoch count, or alternative pilot fold may be selected from
the result.

## Full evaluation and promotion

If the pilot advances, run all five source folds for rationale seed 7. The
primary profile is the predeclared replacement pair (rationale seed 7 plus
general seed 19). An equal three-model blend with both incumbent models is a
secondary diversity diagnostic and cannot rescue failure of the primary gate.

Promotion requires the primary profile to:

- strictly exceed the 0.691700 incumbent accuracy over all 253 rows;
- improve at least three source folds and worsen at most one;
- have source-cluster bootstrap probability of positive gain at least 0.90;
- have non-negative bootstrap 5th-percentile delta; and
- remain submission-policy eligible after organizer clarification.

Only after that gate passes is seed 19 trained as an independent replication.
The two-rationale-seed ensemble must pass the same gate and not underperform the
seed-7 replacement before it may become primary. Otherwise the current
two-general-seed ordinal system remains unchanged.
