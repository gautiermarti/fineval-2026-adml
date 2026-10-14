# Preregistration: packet-disjoint confirmation of cash-flow evidence expansion

Date locked: 2026-08-05

## Status and motivation

This is a **post-hoc discovery hypothesis**, not a blind result.  In the frozen
192-row sparse-edit lattice, the exact `easy:cash` / `evidence:recall` edit is
eligible on 10 rows.  It improves eight, harms two, has positive mean gain in
each of HON, JNJ, MSFT, and UVV, and increases complete-answer ROUGE by 0.03770
on the eligible rows.  No existing experiment selected this exact route.

The operation is also semantically interpretable: preserve the complete v15
answer core byte-for-byte and replace only its news-evidence section with the
validated recall realization.  Because sparse-edit generation deduplicates
candidates in `precision`, `extractive`, `recall` order, the exact
`evidence:recall` name occurs only when the recall evidence is different from
the one-evidence extractive candidate.  It therefore tests a real second-source
evidence expansion, not a naming alias.

## Input-only feasibility audit

Before reading confirmation references, the older 172-row Easy packet was
partitioned by exact normalized `(query, question)` fingerprint against the
192-row discovery lattice.  Nineteen cash-flow packets are outside discovery;
one has an invalid historical v15 baseline and is excluded by the pre-existing
validator.  Exactly two of the remaining packet-disjoint rows expose a valid,
distinct `evidence:recall` edit.  Both are later MSFT packets.  Their reference
answers and edit scores were not inspected during feasibility.

## Frozen policy

For every historical Easy cash-flow row:

1. Rebuild the typed fact graph and all `precision`, `extractive`, and `recall`
   realizations from the input packet.
2. Enumerate sparse edits in the existing deterministic order.
3. Apply the existing schema, 100-word, and numeric-grounding validator.
4. Select `evidence:recall` if and only if that exact valid operation exists;
   otherwise return the v15 baseline byte-for-byte.

No threshold, lexical cue, issuer exception, or score-dependent fallback may
be added after confirmation scores are observed.

## Confirmation partition and gates

The confirmation partition contains only exact packet fingerprints absent from
the discovery lattice.  References are used only after predictions are frozen.
The route is eligible for promotion only if all of the following hold:

1. At least the two feasibility-audited confirmation edits are scored.
2. Neither confirmation edit harms v15.
3. At least one confirmation edit strictly improves v15.
4. Mean gain over eligible confirmation edits is strictly positive.
5. In the reconstructed discovery partition, the policy's mean gain is
   strictly positive for each of HON, JNJ, MSFT, and UVV.
6. Mean complete-answer gain over all reconstructed rows is positive.

If any gate fails, the route is rejected and v15 remains unchanged.  Passing
the gate would justify only this exact evidence-only operation; it would not
validate broader recall candidates, source-MBR reranking, or answer-core edits.

## Leakage statement

The current FinNLP blind test set is not available and is not used.  The
discovery references motivated the hypothesis and are disclosed as such.  The
packet-disjoint references are evaluation-only and may be read exactly once
after this document and the deterministic evaluator are frozen.
