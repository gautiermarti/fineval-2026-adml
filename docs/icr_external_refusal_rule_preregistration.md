# Task 2 external-corpus refusal rule preregistration

Date frozen: 6 August 2026, before running the rule on JF-ICR labels.
Team: ADML.

## Hypothesis and separation from Nemotron

The source-OOF and full-refit Task 2 incumbents never emit `-2` or `-1` and are
both correct on 0/13 released refusal rows. Independently collected Japanese IR
Q&A contains 69 high-recall refusal-cue candidates. These external pairs have
no JF-ICR labels and no benchmark issuer/domain or near-duplicate, so they can
define literal, input-only refusal forms without fitting the 13 benchmark
refusals.

This is a deterministic CPU control for the separately frozen Japanese
Nemotron hypothesis. It is frozen and evaluated before any Nemotron output. No
rule result may change the Nemotron prompt, pilot, or confirmation gates, and
the two candidates cannot be combined without a new preregistration.

External discovery file:
`data/external/japanese_ir_qa/forward_candidates.jsonl`, SHA-256
`e84a94e2088cf66ba5d4723b3835d894b9fe45aab4b90ec09865e8041e897f02`.
Its refusal queue has cue counts `現時点では:35`, `差し控え:14`,
`申し上げられ:10`, `開示してい:8`, `未定:3`, `コメントでき:1`, and
`具体的には:1`. The cues themselves are not labels; only the explicit
predicate patterns below can fire.

## Frozen incumbent

Reconstruct the exact two-seed Japanese-BERT source-OOF incumbent from aligned
seed-7 and seed-19 logits, equal probability averaging, rounded expected-score
decoding, and the immutable `safe_rules`. It must reproduce 175/253 correct.

## Frozen rule

Normalize the company response with Unicode NFKC. Search the entire response;
do not select a position or sentence after seeing outcomes.

The **withholding family** is:

```text
(?:回答|言及|コメント|開示|説明|公表|詳しく|詳細|具体的)
.{0,20}
(?:差し控え|お答えでき(?:ない|ません)|回答でき(?:ない|ません)|
   コメントでき(?:ない|ません)|申し上げられ(?:ない|ません)|非開示|開示しない)
```

The **action-rejection family** is either:

```text
(?:予定|計画|方針|考え|意向).{0,12}(?:ない|ありません|ございません)
```

or:

```text
(?:考えて|想定して|検討して|実施して|行って)(?:いない|おりません)
```

If withholding matches anywhere, predict `-1`. Otherwise, if action rejection
matches, predict `-1` when the same response contains `現時点`, `現在`, `当面`,
`今のところ`, or `現状`; predict `-2` otherwise. If neither family matches,
retain the incumbent. Withholding has precedence because non-disclosure is a
weak refusal even when other negative language is present.

There is no generic `未定`, `現時点では`, `開示していない`, negation, or
sentiment rule. No question keyword, label prior, source identity, match
position, confidence threshold, or pattern grid may be added after evaluation.

## Frozen promotion gate

Promote the deterministic rule only if every condition holds:

1. the incumbent reproduces exactly 175/253 correct;
2. the candidate reaches at least 179/253 correct;
3. it rescues at least four incumbent errors and harms at most one incumbent
   correct row;
4. at least 80% of changed rows are beneficial;
5. at least four of 13 gold refusal rows become correct;
6. at least three of five source folds improve and at most one worsens;
7. a 100,000-draw source-group bootstrap with seed 20260806 has probability of
   positive gain at least 0.90 and 5th-percentile delta at least zero; and
8. no current/hidden test input, answer, upload score, or leaderboard feedback
   is used.

Failure stops this exact rule with no cue/pattern/precedence/label adjustment.
Passing makes it a submission candidate but does not automatically combine it
with Nemotron; the later independent model result determines whether a
separately frozen union is justified.
