# Task 3 large-planner intersection confirmation preregistration

Frozen before selecting, inspecting, or scoring any confirmation target.
Date: 5 August 2026. Team: ADML.

## Discovery result and hypothesis

On the already completed 32-row large-content-planner screen, the 35B planner
had higher recall on nonempty reference plans but over-selected facts when the
historical reference was qualitative. A post-hoc diagnostic found that keeping
only incumbent-selected facts that the 35B planner also selected would have
improved macro set F1 from 0.836742 to 0.876028 (four changed rows, four wins,
zero harms). That number is discovery-only and cannot promote a system.

The independently testable hypothesis is narrower: the incumbent two-stage
planner is the content-policy gate, while the large planner is a conservative
fact verifier. The hybrid may delete an incumbent fact but may never add a fact
or change its order. This should preserve the incumbent's strong qualitative
policy while reducing unsupported or reference-irrelevant numeric details.

## Immutable confirmation screen

- Source: the existing `polyfiqa_graph_to_reference_v5/loco` evaluations.
- Eligible rows must have only fact-sourced requested slots.
- Exclude every exact target identity in the v1 32-row screen.
- Use the seven families with at least one unused eligible row for all four
  companies: `balance`, `capex`, `cash`, `margin`, `rd`, `revenue`, and
  `revenue_focus`. `allocation` is excluded because HON has no second eligible
  row; this exclusion was established from input/eligibility counts, not target
  values.
- Choose exactly one row per company x family cell by the smallest SHA-256 of
  `large-planner-intersection-confirm-v1|company|family|task_id|record_index`.
  This yields 28 rows. No replacement or alternate screen is allowed.
- Demonstrations follow the v1 rule: one stable-hash fact-only example from
  each of the other three companies in the same family. The target reference
  is never placed in the prompt.

The manifest may contain targets for later deterministic scoring, but it must
be generated only after this file is hashed. No confirmation target or
aggregate label statistic may be inspected before inference finishes.

## Fixed model and hybrid

- Model: local `qwen3.6:35b-a3b`, temperature 0, seed 7, thinking disabled,
  the same system prompt, JSON schema, and three-example prompt constructor as
  the v1 screen.
- The model can output only an ordered list of offered fact IDs. Invalid output
  falls back to the incumbent and counts as invalid.
- Let `I` be the incumbent two-stage ordered fact list and `Q` the model set.
  The only candidate is `[fact for fact in I if fact in Q]`, preserving the
  exact incumbent order. It can never introduce a value, metric, or fact.
- No 57B model, alternate prompt, union, threshold, family-specific exception,
  or model confidence is tested on this confirmation screen.

## Metrics and advance gate

Primary metric: macro set F1 against the typed reference plan. Also report
exact-set accuracy, ordered-LCS F1, micro precision/recall/F1, per-company
deltas, changed rows, wins, harms, and invalid outputs.

Advance to the full historical fact-only evaluation only if all conditions
hold:

1. hybrid macro set F1 is at least incumbent macro set F1 + 0.01;
2. at least three of four company deltas are nonnegative and no company delta
   is below -0.01;
3. exact-set accuracy does not decline;
4. at least two rows change, wins exceed harms, and harms are at most one;
5. at least 27 of 28 model outputs are valid.

Failure stops this branch. Passing permits one fixed full-history content-plan
evaluation, not automatic end-to-end promotion. Any submission change still
requires a separately frozen compiler integration and answer-level grounding,
word-limit, and reference-metric checks. Hidden/current test answers and
leaderboard feedback may not alter this experiment.
