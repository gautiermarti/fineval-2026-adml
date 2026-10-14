# Task 3 Nemotron 30B fact-filter confirmation preregistration

Date frozen: 2026-08-06. Team: ADML.

## Hypothesis

Previous constrained large planners recovered relevant numeric facts but tended
to mention too many of them. A genuinely different model family may still add
value as a conservative verifier of the existing input-only planner. The
primary candidate therefore starts from the incumbent ordered fact IDs and
keeps only IDs independently selected by Nemotron. It may delete a fact, but
it cannot add or reorder a fact, write prose, calculate a value, or introduce
an unsupported number.

This is independent of the failed Japanese Nemotron 9B Task 2 hypothesis. No
Task 2 output, label, threshold, or prompt is used here.

## Frozen model and inference

- Checkpoint: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
- Revision: `2d59de1cbd51c0adf384eb906b766d1aee0e0517`.
- Audited tensor elements: 31,577,940,288 (31,577,934,400 BF16 and 5,888
  F32), below the organizer's 70B total-parameter ceiling.
- Runtime: local vLLM 0.26.0, one DGX Spark GPU, BF16 checkpoint.
- Decode: thinking disabled, temperature 0, top-p 1, seed 2026, maximum 192
  completion tokens, one generation per target.
- Output: strict JSON with exactly one field, `selected`, containing a unique
  ordered list of offered string fact IDs. Unknown or duplicate IDs, extra
  fields, malformed JSON, transport failures, and provenance drift are invalid.
- No score-aware retry, alternate prompt, sampling sweep, or token-budget
  change is allowed after the first scored output.

## Frozen evaluation population

- Corpus: strict historical originals in
  `artifacts/polyfiqa_graph_to_reference_v5/loco/*/evaluation.jsonl`.
- Include every fact-only held-company record and exclude every oracle-copy or
  non-fact requested-slot record. The expected population is 154 rows:
  HON 38, JNJ 30, MSFT 44, and UVV 42.
- The target company is absent from model demonstrations. Each prompt uses the
  existing deterministic rule from the earlier large-planner protocol: one
  historical original from each of the other three companies in the same
  family, selected by the frozen stable hash
  `large-planner-demo-v1|held|family|donor|task_id|source_index`.
- Target input contains only the question, family, and input-derived candidate
  facts `(id, unit, derived, value)`. It does not contain the target reference,
  requested slots, target IDs, or a target label.
- Historical reference-derived IDs are used only for held-company scoring and
  other-company demonstrations. No current/public-test answer, hidden-test
  input, or hidden-test answer is used.

This historical population has supported earlier planner research, so the
experiment is a rigorous family-diversity confirmation rather than an untouched
test-set claim. The model, prompt, policy, and gate are frozen before any 30B
FinEval output, and a passing fact-selector result still requires a separate
end-to-end answer confirmation before deployment.

## Frozen candidate and diagnostics

The incumbent is `two_stage` from
`artifacts/polyfiqa_numeric_planner_v5/predictions.jsonl`.

For a valid Nemotron output, the sole promotable candidate is the incumbent
ordered list filtered to IDs also selected by Nemotron. For an invalid output,
the candidate fails closed to the unchanged incumbent. This is called
`nemotron_intersection`. Direct Nemotron selection is reported as a diagnostic
only and cannot be promoted from this experiment.

The runner must verify that all 154 identities are unique and aligned, every
incumbent and model ID is offered, and every candidate is an order-preserving
subset of both the incumbent and offered facts.

## Metrics and uncertainty

Report for the incumbent, direct model, and intersection:

- macro set F1 (primary), exact-set accuracy, nonempty-target macro set F1;
- ordered-LCS F1, micro precision/recall/F1, and valid output count;
- paired row changes, rescues, harms, and ties;
- paired macro-set-F1 deltas by held company and by family.

Run 10,000 deterministic hierarchical bootstrap replicates with seed 20260806.
For each replicate, sample the four held companies with replacement and then
sample that company's rows with replacement at its original fold size. Report
the mean delta, 5th and 95th percentiles, and P(intersection gain > 0).

## Frozen promotion gate

The intersection advances to a separately preregistered end-to-end confirmation
only if every condition holds:

1. at least 152/154 Nemotron outputs are valid;
2. intersection macro set F1 is at least incumbent macro set F1 + 0.010;
3. intersection exact-set accuracy is not below the incumbent;
4. all four company deltas are at least -0.010 and at least three are
   non-negative;
5. at least six of eight family deltas are non-negative and none is below
   -0.025;
6. hierarchical-bootstrap P(gain > 0) is at least 0.95 and its 5th-percentile
   delta is non-negative;
7. every candidate passes the order-preserving grounded-subset invariants; and
8. no current/public-test or hidden-test answer was used.

Failure of any condition is a stop. A failure cannot be repaired with a
company exception, family exception, threshold, union policy, prompt rewrite,
thinking-mode change, or second decode on these labels. A pass authorizes
another experiment; it is not itself an end-to-end Task 3 win claim.
