# Task 3 large-model content-planner preregistration

Date frozen: 2026-08-05. Team: ADML.

## Hypothesis

The rejected 27B/35B realizer experiment showed that larger models do not
improve Task 3 by rewriting already grounded prose. The distinct hypothesis is
that scale may help at the unresolved decision boundary: choosing which typed
facts a short answer should contain. The model therefore sees a question and a
finite list of input-derived fact IDs and may return only an ordered subset of
those IDs. It never writes numbers or answer prose.

## Immutable screen

- Corpus: strict historical originals in
  `artifacts/polyfiqa_graph_to_reference_v5/loco/*/evaluation.jsonl`.
- Eligibility: fact-only records; oracle-copy records are excluded.
- Split: leave one company out. No demonstration may use the held company's
  original or counterfactual record.
- Targets: one record from every company × family cell (4 × 8 = 32). Within a
  cell, select the lexicographically smallest SHA-256 of
  `large-planner-v1|held_company|family|task_id|record_index`.
- Demonstrations: exactly one historical original from each of the other three
  companies in the same family. Within each donor company, use the smallest
  SHA-256 of `large-planner-demo-v1|held|family|donor|task_id|source_index`.
- Demonstrations expose the reference-derived ordered fact IDs; target prompts
  do not expose reference text, requested slots, or labels.
- Model input contains the target question and candidate `(metric:index, unit,
  derived, value)` records. Output is one JSON object with an ordered `selected`
  list. Every returned ID must be unique and belong to the offered candidates.
- Decode once at temperature zero with no score-aware retry. Invalid output is
  reported and receives zero item score rather than an empty-set reward.
- Models: cached `qwen3.6:35b-a3b` and
  `llama3.3:70b-instruct-q4_K_M`. Quantization changes storage, not total model
  parameters.

The marketed 70B model is a research comparison. It is not declared submission
eligible until the organizers confirm whether the limit means the named 70B
class or a literal decimal parameter count.

## Metrics and fixed decisions

Primary metric is macro set F1 over the selected fact IDs. Secondary metrics
are exact set accuracy, nonempty-target macro set F1, micro F1, and ordered-LCS
F1. Existing v5 fixed-threshold, two-stage, top-k, and select-all predictions
are scored on the same 32 rows.

A model advances from the 32-row screen to all 154 held-company fact-only rows
only if it:

1. beats the strongest existing planner on screen macro set F1 by at least
   0.01;
2. is non-negative against that planner in at least three companies; and
3. produces valid outputs for at least 30/32 rows.

If neither model passes, this branch stops. We will not alter demonstrations,
prompt wording, output parsing, sampling, or model order after observing the
screen. Passing the planner screen is not an end-to-end answer claim; any later
compiler must be separately preregistered and beat complete v15 answers under
company/time-held-out ROUGE with zero grounding failures.
