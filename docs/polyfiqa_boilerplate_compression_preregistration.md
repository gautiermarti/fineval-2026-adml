# Task 3 transferable-boilerplate compression preregistration

Frozen before inspecting any deletion score on 5 August 2026. Team: ADML.

## Motivation and label-free feasibility

The v15 sparse-edit lattice contains eight delexicalized answer-core sentence
signatures that recur across at least two issuers. Six recur in all four
historical issuers with 24 rows each. This makes sentence identity a stronger
transfer unit than the previously rejected family/position rule: the new method
asks whether one exact kind of boilerplate is consistently worth retaining,
not whether “sentence 1” is usually useful.

The feasibility audit used only candidate text, tier/family metadata, and
issuer counts. It did not read references, candidate ROUGE scores, or oracle
choices.

## Preregistered primary

For each held issuer independently:

1. Start from v15 byte-for-byte.
2. Consider only an already-generated `delete_sentence:*` candidate that has
   passed complete-answer structure, word-budget, and numeric-grounding checks.
3. Normalize the removed sentence by lowercasing, replacing the four historical
   issuer names with `<company>`, replacing numeric values with `<num>`, and
   collapsing whitespace.
4. On the other three issuers only, admit a signature when every training
   issuer supplies at least three examples, its source-balanced mean deletion
   gain is at least 0.005, and its worst training-issuer mean gain is
   non-negative.
5. If multiple signatures qualify on one held row, choose the one with the
   largest source-balanced mean gain; break exact ties by larger support and
   then lexical signature. Otherwise return v15 unchanged.

This method is named `loco_exact_boilerplate_deletion_v1`. It never composes
multiple deletions and never learns from the held issuer.

## Promotion gate

- Score complete answers with the official item-level ROUGE-1 implementation.
- Overall delta must be positive.
- At least three issuer folds must be non-negative.
- At most one fold may regress, and its mean regression must be smaller than
  0.002 absolute.
- Exact four-issuer bootstrap P(gain) must be at least 0.90 and p05 non-negative.
- Every selected answer must remain structure-, budget-, and grounding-valid.

Failure retains v15. No looser threshold or post-hoc signature exception will
be promoted from this evaluation; such a hypothesis would require a separately
frozen packet- or time-disjoint replication.
