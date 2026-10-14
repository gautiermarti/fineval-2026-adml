# ADML Task 3 research log

## 2026-08-05 — Independent large-planner intersection confirmation

The post-hoc discovery that Qwen 35B could act as a verifier for incumbent
facts was isolated behind an advance-frozen confirmation.  The new screen has
28 targets not present in the original 32-row screen, one stable-hash row for
each company x seven-family cell with complete unused coverage.  The model can
select only offered fact IDs; the hybrid is the incumbent ordered list filtered
by Qwen's set and therefore cannot add a fact, value, or metric.  The protocol
SHA-256 is
`889659404b53dd67cbec1161cca2e98ee55155976f4edb1af4d1640dbe4f55a1`.

All 28 outputs are valid.  The hybrid improves macro set F1 from **0.745408**
to **0.763776**, exact-set accuracy from 0.535714 to 0.607143, and micro
precision from 0.804878 to **0.911765**.  It changes four rows with three wins
and one harm.  The harm produces a **-0.054422** HON delta, while JNJ gains
0.127891 and MSFT/UVV tie, so the frozen no-company-below--0.01 condition fails.
No full-history run or post-hoc exception is allowed; v15 remains primary.

The unconstrained planner scores 0.790476 on this screen, reversing its small
loss on the first screen.  This confirms complementary fact-selection signal
but also the issuer/sample instability that prevents safe routing.  Results
are in `artifacts/polyfiqa_large_planner_intersection_confirmation_v1/`
(summary SHA-256
`f6eddd869cb9628c4e284a120cac3065e057840ce43212407f85bd4e9d9438fb`).

## 2026-08-05 — Constrained 35B/57B content-planner screen

A 32-row company × family screen tests large models only at fact selection,
not answer rewriting. Targets are fact-only historical originals; each prompt
uses three same-family demonstrations from the other three companies and asks
for an ordered JSON subset of offered typed fact IDs. No target reference,
public-test answer, free-form number, or prose is exposed. All outputs from both
models pass the exact schema and offered-ID validator.

The existing two-stage planner scores **0.836742** macro set F1. Qwen 3.6
35B-A3B scores **0.823722**, while the competition-compliant official
Qwen2-57B-A14B model scores **0.799723**. Both improve only two of four company
groups and fail the preregistered advance gate. The 57B model nevertheless has
the strongest nonempty-target F1 (**0.772743**) and numeric-fact recall
(**0.926829**), demonstrating that scale improves extraction/reasoning but not
the benchmark's choice to verbalize or omit numeric content. The full 154-row
evaluation is therefore not run, no post-hoc family router is fitted, and v15
remains frozen. Protocol, compliance amendment, runner, and outputs are in
`docs/polyfiqa_large_content_planner_preregistration.md`,
`docs/polyfiqa_large_content_planner_compliance_amendment.md`,
`scripts/evaluate_polyfiqa_large_content_planner.py`, and
`artifacts/polyfiqa_large_content_planner_v1/`.

## 2026-08-05 — More historical data confirms that sparse-edit routing does not transfer

The frozen sparse-edit lattice was rebuilt over all 344 historical Easy and
Expert rows.  After one invalid early baseline is excluded, it contains 151
older packet-disjoint training rows plus all 192 newer discovery rows.  The
newer candidates and scores are byte-aligned to the frozen lattice.  Candidate
quality is stable across time: oracle gain is **+0.035817** on the older rows
and **+0.037708** on the newer rows.

The selectors do not share that stability.  Under the preregistered old-to-new
test, an exact operation signature gate admits nothing and safely reproduces
v15.  A ridge risk model trained on all older issuers changes seven newer rows
and falls by **-0.000835** (2 improvements, 5 harms).  In the primary stricter
regime—train only on older packets from the other three issuers and evaluate
newer packets for the held issuer—the ridge changes four rows and harms all
four, falling from **0.365242 to 0.364371**.  Its positive inner-fold estimate
therefore reverses under both time and issuer shift.

This closes an important alternative explanation for earlier abstention: the
problem was not merely that 192 rows were too few.  The edit oracle transfers,
but reference-scope labels for otherwise grounded edits do not.  Threshold or
family retuning on these late labels is stopped; v15 remains frozen.

The locked protocol is
`docs/polyfiqa_temporal_sparse_selector_preregistration.md` (SHA-256
`bb93c2895b327bdffb24407cea88211b5967bd895eee412fa0f7a0079d1e6abf`).
The verified lattice is in `artifacts/polyfiqa_historical_sparse_lattice_v1/`
(summary SHA-256
`8906a689c4e4f571827bc6fe3eabbdf287b22f50a147672d52ad7dadaef6adf9`),
and selector results are in `artifacts/polyfiqa_temporal_sparse_selector_v1/`
(summary SHA-256
`cf7998bd2d41b3880dc1ce5cd0c834ea7167b2550dc4638416cc52f809e7bcbd`).
No current or blind-test answer was used.

## 2026-08-05 — Cash evidence expansion fails packet-disjoint confirmation

A row-level audit found one previously untested exact sparse-edit signature:
`easy:cash` / `evidence:recall` improved 8/10 eligible discovery rows and had
positive mean complete-answer gain in each of HON, JNJ, MSFT, and UVV.  Because
that pattern was discovered after reading the 192-row scores, it was treated as
a hypothesis rather than a promotion.  The exact no-tuning policy and gates
were locked before scoring the two later packet-disjoint cases on which a
distinct validated recall-evidence edit was input-only eligible.

Both confirmation edits harmed v15: **-0.091884** and **-0.057576** item ROUGE.
Across the 18 valid confirmation cash rows, the policy loses **-0.008303**;
among the two eligible rows it loses **-0.074730**.  It therefore fails the
preregistered zero-harm gate decisively.  This is a second independent warning
that small, apparently cross-company evidence routes can encode time-specific
reference scope.  The operation is rejected and v15 remains frozen.

The protocol is
`docs/polyfiqa_cash_evidence_recall_confirmation_preregistration.md` (SHA-256
`7ae0578ce88b8c0b5138247e848db4ba0102f9dd5cc55de5afbf6313149d9aa1`).
Reproduction is in
`scripts/evaluate_polyfiqa_cash_evidence_recall_confirmation.py` and
`artifacts/polyfiqa_cash_evidence_recall_confirmation_v1/`; the summary SHA-256
is `fdc12b90f4fc651db33bb19044e7827c282d0d7253dcf3478876e9df2fc033ad`.
Current or blind-test answers were not used.

## 2026-08-05 — Preregistered exact-boilerplate deletion safely abstains

An input-only feasibility pass found recurring delexicalized sentences in the
v15 answers, including cash-flow framing, revenue-growth framing, leverage
interpretation, and generic strategy language.  Before inspecting their
deletion scores, the exact leave-one-company-out policy and promotion gate were
locked in `docs/polyfiqa_boilerplate_compression_preregistration.md` (SHA-256
`b82c66b0cd089190c19487fd308b59ec781e9a0ade00e1f98de6a225922dda06`).
The policy required at least three examples from every training company, a
source-balanced deletion gain of at least 0.005, and a nonnegative mean in the
worst training company.

No sentence signature met those safeguards in any held-company fold, so the
locked policy changed **0/192** answers and exactly reproduced v15 at
**0.365242**.  This is not a promotion: the positive-delta and bootstrap-gain
gates failed.  The audit also explains the abstention.  Deleting the recurring
liability/equity ratio lost 0.088 ROUGE on average across the other companies;
deleting the generic productivity/cost/pricing sentence lost 0.012; and even
the nearly neutral total-revenue sentence had a negative company mean.  The
apparent boilerplate often encodes reference-rewarded content, rather than
pure verbosity.

Reproduction is in
`scripts/evaluate_polyfiqa_boilerplate_compression.py` and
`artifacts/polyfiqa_boilerplate_compression_v1/`.  The summary SHA-256 is
`7e1227fa5882139a82235cad8b8b2acf7b2ef0b659e1bf0095a66c51970586b4`;
the signature audit SHA-256 is
`34f79c2c9b7aca8cbd41ee9371ee4c66281ef06c17b1812069c5783413c7cd81`.
V15 remains frozen.

## 2026-08-05 — Constrained graph-realizer adapter fails the held-issuer pilot

The previously scaffolded graph-to-reference adapter was upgraded to the strict
v5 corpus, an immutable base-model revision, non-thinking decoding, explicit
held-company leakage checks, and a complete-answer evaluator.  The pilot used
`FutureMa/Eva-4B` at revision
`99c3193947fd41ed0f2befd5ef1ce3f74c90cab6`, LoRA rank 16, four epochs, and
greedy decoding.  Honeywell was excluded from every original, donor, and target
training record: 241 records trained the adapter and all 60 Honeywell originals
were held out.  All inputs fit the 4,096-token budget.

The adapter learned reference-like wording but did not learn safe realization.
Its mean held-out template ROUGE was **0.473246**, yet only 13/60 generations
preserved the exact immutable marker multiset.  After complete-answer structure,
100-word, and numeric-grounding checks, only **10/60** could replace the frozen
v15 core.  With automatic fallback on every invalid output, complete-answer
ROUGE fell from **0.482848 to 0.480653** (**-0.002195**; 3 rows improved, 3
harmed, and 54 tied).  The most damaging valid failure was an Expert
revenue-focus answer that collapsed to `Answer: None`; many rejected outputs
also repeated years, invented markers, omitted requested values, or attached
grounded values to the wrong semantic relation.

This is an especially strong falsification because the adapter received the
historical reference-oracle slot/evidence plan; it was not yet solving the
harder input-only planning problem.  The remaining three folds were therefore
stopped rather than spending approximately another hour on a branch that
already loses under easier-than-deployment conditions.  No Task 3 prediction
changes.  Reproduction is in `scripts/train_polyfiqa_graph_realizer.py`,
`scripts/summarize_polyfiqa_graph_realizer.py`, and
`artifacts/polyfiqa_graph_realizer_eva4b_v5_loco_v1/`.  The strict pilot
summary SHA-256 is
`fb03aff9a3993e023ee23fab2436efde38146066d3be0c7795930aad468c039e`; the
row audit SHA-256 is
`f55761e850f3198ea8a3e80eb452cfbc80baad48063008531cd4352dfb0a67df`.

## 2026-08-05 — Multilingual evidence entailment fails; packet-disjoint rule confirmation prevents a false promotion

A label-free verifier now scores every sparse candidate clause as a hypothesis
against input-only typed facts, exact statement lines, and semantically retrieved
multilingual news premises. `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` supplies
entailment/neutral/contradiction probabilities; exact quotation provenance and
unit-aware numeric support remain separate symbolic signals. Historical answers
and candidate ROUGE scores are excluded from feature construction. The CPU pass
contains 1,533 candidates and 6,556 deduplicated NLI pairs.

The semantic idea does not transfer to the official selection objective. A
zero-shot entailment score falls from **0.365242 to 0.304055**. Leave-company-out
margin calibration reaches 0.354402. A fully nested NLI ridge reaches 0.364897.
Combining NLI with the earlier source-MBR scalars is nominally positive at
0.365309 (**+0.000067**), but changes only one row and has cluster-bootstrap
P(gain)=0.684, so it is rejected. Entailment measures truth support, whereas
reference ROUGE also rewards a particular content scope and evidence wording.

A tempting post-hoc slice was then subjected to a genuinely new check. In the
192-row discovery lattice, the grounded `evidence:recall` edit was positive in
all seven Easy-balance cases where it appeared. The same deterministic policy
was reconstructed on the older 172-row Easy corpus, and evaluated separately
on the 19 balance packets outside the discovery fingerprints. It was eligible
once and that edit **lost 0.004393**; the rule is therefore rejected. This is a
useful negative control: an apparently perfect small slice did not survive one
packet-disjoint observation.

Reproduction is in `scripts/build_polyfiqa_entailment_features.py`,
`scripts/evaluate_polyfiqa_entailment.py`,
`scripts/evaluate_polyfiqa_evidence_recall_confirmation.py`,
`artifacts/polyfiqa_entailment_features_v1/`,
`artifacts/polyfiqa_entailment_v1/`, and
`artifacts/polyfiqa_evidence_recall_confirmation_v1/`. V15 remains frozen.

## 2026-08-05 — Sparse full-answer edit lattice exposes headroom; family gate fails

The v15 complete answer is now treated as immutable except for a finite set of
strictly validated local edits: swap only the core, swap only the evidence
section, replace the first core sentence, delete one core sentence, or retain
an already grounded precision/extractive/recall candidate. Every edited answer
is re-capped and must pass the existing numeric-grounding and schema validator.

On the 192 historical reference-overlap rows, v15 scores **0.365242** and the
sparse-edit oracle reaches **0.402950** (+0.037708), improving 139 rows. The
oracle gain is positive in all eight families: +0.050623 balance, +0.047885
cash, +0.016974 R&D, +0.070071 Easy revenue, +0.028802 allocation, +0.024748
capex, +0.022537 margin, and +0.040024 Expert revenue. The oracle selects the
baseline on 53 rows, evidence-only edits on 46, one-sentence deletion on 36,
full candidates on 29, first-sentence edits on 17, and core-only edits on 11.
This clears the predeclared +0.015 candidate-headroom prerequisite for a risk
model and is materially stronger than the earlier wholesale-candidate oracle.

Candidate quality is not a deployment result. A fixed leave-company-out gate
admitted an exact family/edit signature only when it had at least three rows
per training company, mean gain >=0.005, and worst-company mean gain >=-0.002.
It changes eleven rows, all in held-out Universal, and falls to **0.364263**:
five improvements, six harms, cluster-bootstrap P(gain)=0. The rule is rejected.
The experiment isolates cross-company harm prediction as the remaining Task 3
bottleneck. A learned pairwise preference/risk model is justified by the oracle
but must beat v15 under complete-answer leave-company-out scoring before any
edit is promoted.

Reproduction is in `scripts/evaluate_polyfiqa_sparse_edit_oracle.py`,
`scripts/evaluate_polyfiqa_sparse_edit_family_gate.py`,
`artifacts/polyfiqa_sparse_edit_oracle_v1/`, and
`artifacts/polyfiqa_sparse_edit_family_gate_v1/`. Current hidden answers were
not used.

A nested sparse-edit ridge risk model was then trained on candidate-versus-v15
text differences, structural features, source coverage, and source-balanced
row weights. For each held company, ridge strength and the minimum predicted
gain were chosen only through inner leave-company-out evaluation on the other
three companies. The conservative selection is complete abstention: **0.365242
ties 0.365242**, with zero changed rows. This is not a promotion, but it avoids
the Universal regression of the hand gate and confirms that lexical/structural
features alone cannot certify edit transfer from only four issuers. The next
selector needs an independent semantic verifier or additional issuer-diverse
references; further threshold tuning on the same four companies is stopped.
The artifact is `artifacts/polyfiqa_sparse_edit_risk_model_v1/`.

## 2026-08-05 — Nested family compiler and complete-answer falsification

A family-specific utility model repairs part of the earlier global ranker's
failure, but only inside the restricted answer-core diagnostic. Each outer
held company uses an inner leave-company-out gate to choose between the
state-conditioned medoid and a separate family ranker. On the same 120
fact-only rows with compatible executable templates, the nested hybrid rises
from **0.329657** to **0.335011** core ROUGE (+0.005354). Three company folds
improve and one ties; the exact four-cluster bootstrap gives P(gain)
**0.996094** and p05 **+0.001880**. Revenue-focus supplies most of the gain,
and the input-only v2 planner recovers its empty numeric signature on all
14/14 covered rows.

That result does **not** transfer to the official output scope. A stricter
control replaces only the answer core of the current deterministic response,
preserves its independently generated financial-statement evidence, and
scores the complete answer with the official item-level ROUGE implementation.
Across the same 14 revenue-focus rows, the incumbent scores **0.358600** and
the compiler replacement **0.256297**, a **-0.102303** regression. Every exact
company-cluster bootstrap resample is negative (P(gain)=0;
p95=-0.035437). Direct compiler replacement is rejected.

Reproduction is in `scripts/evaluate_polyfiqa_family_compiler.py`,
`artifacts/polyfiqa_family_compiler_v1/`,
`scripts/evaluate_polyfiqa_family_compiler_full_answer.py`, and
`artifacts/polyfiqa_family_compiler_full_answer_v1/`. No current or public test
answer was used for prediction or selection.

## 2026-08-05 — Strict reference corpus v5 and executable compiler diagnostic

Bounded unit/rendering programs expand the strict graph-to-reference corpus
from 148 to **222/344** historical originals (64.53%) and the independently
verified semantic-state graph swaps from 96 to **168**. The verifier confirms
exact template round trips, exact evidence provenance, zero donor leaks, zero
oracle evidence in swaps, and no current/public test answers. Of the originals,
154 are fact-only and 68 require an oracle query-copy/transform slot. Forty-six
rows are recovered by unique, input-supported rendering transformations.

On the earlier 150 v2 fact-only rows, the two-stage numeric planner rises from
0.786010 to **0.796721** macro set F1 (+0.010711), but row bootstrap P(gain) is
only 0.807 and the lower interval is negative. V5's extra four fact-only rows
slightly reduce common-row performance relative to v2, and fully nested
train-company threshold selection falls to 0.721177. V2 remains the best
planner training corpus; v5 is the best realizer corpus. None of these planner
variants is promoted end to end.

The first leave-company-out executable reference compiler uses only compatible
fact-slot signatures and input-derived semantic state, restores target values
mechanically, and has **zero restoration errors**. It covers 120/154 fact-only
rows. On covered rows, a state-conditioned expected-template-ROUGE medoid scores
0.329657; balance reaches **0.617539** and R&D **0.654423**, but cash, capex,
allocation, and revenue-focus remain weak. A cross-company TF-IDF/Ridge utility
ranker falls to 0.251755 overall, although it improves capex, cash, and slightly
revenue. The global learned ranker is rejected; the positive balance/R&D slices
motivate family-specific compilers with abstention. These are realizer-only
historical diagnostics because the target fact-slot signature is oracle.
Reproduction is in `scripts/evaluate_polyfiqa_reference_compiler.py` and
`artifacts/polyfiqa_reference_compiler_v1/`.

## 2026-08-05 — First input-only numeric content planner

The missing input-only plan stage now has a source-disjoint baseline.  It uses
only the typed numeric graph—never the reference text or an oracle evidence
choice—to predict which `(metric, period-index)` facts a historical reference
will verbalize.  Evaluation holds out one company at a time and removes that
company from original and counterfactual training records.  The experiment
covers 126 fact-only strict programs, of which 46 have nonempty numeric
targets.

A fixed 0.50 fact selector reaches macro set F1 **0.743682**, exact set
accuracy 0.611111, and micro F1 0.616766.  A two-stage numeric-versus-
qualitative gate raises macro set F1 to **0.786010** and exact accuracy to
0.658730, versus 0.676073 and 0.523810 for selecting all facts.  This is the
first evidence that an input-only planner learns reference content choices.
It is not an end-to-end answer gain: the gate's aggregate improvement partly
comes from correctly abstaining on qualitative rows, while nonempty macro F1
drops from 0.667476 to 0.587766.  Company variation is also large, including a
JNJ micro F1 of 0.24 on only four gold numeric selections.  Threshold sweeps on
outer evaluation are diagnostic only; the next version must select gates in a
nested training partition and model ordered clauses, not just an unordered
fact set.

Reproduction is in `scripts/evaluate_polyfiqa_numeric_planner.py` and
`artifacts/polyfiqa_numeric_planner_v1/`.  Public/current test answers were not
used and the artifact explicitly sets `end_to_end_answer_claim` to false.

## 2026-08-05 — Final-test overlap correction and winning-margin audit

The published FinMMEval 2026 short-answer overview changes an important earlier
assumption.  Its 256-item final test contains 32 new report groups and the
organizers explicitly removed the 76 public-overlap groups from each tier.
Therefore exact public-reference lookup was not a competitive final-test path
in that evaluation.  Its best system scored 31.18% macro item ROUGE-1 F1 and
the top four were separated by only 0.79 point (31.18, 30.96, 30.71, 30.39).
The winning system emphasized deterministic prompts, schema validation, and a
single retry; other leading systems used targeted extraction, multilingual
retrieval, and constrained selection.

The current FinNLP task's final packet and overlap policy still need direct
confirmation, so this evidence is not silently conflated with the present
competition.  It nevertheless removes overlap lookup from the scientific
strategy: the winning branch must operate from input alone.  The narrow prior
leaderboard margin also makes a valid planner, exact numeric compiler, and
reference-compatible realization more plausible high-return investments than
an unconstrained large-model rewrite.

## 2026-08-05 — Strict graph-to-reference adaptation corpus

The first graph-to-reference builder now converts 148/344 historical
Easy/Expert references (43.02%) into exactly restorable delexicalized targets,
versus 41 references admitted by the earlier fully executable cross-company
program inducer. Coverage spans all eight reasoning families: revenue 14, cash
22, R&D 29, balance 1, allocation 11, margin 22, capex 27, and focused revenue
22. Accepted data covers 39 Microsoft, 45 Honeywell, 21 Johnson & Johnson, and
43 Universal rows.

Every variable number must either align to a typed fact and re-render exactly
or occur verbatim in the input packet. Loose nearest-number matches are
rejected; this caught a real $7.047B acquisition / $7.058B financing collision.
The builder also preserves accounting parentheses and grammatical parentheses
separately. All 148 records pass exact mechanical restoration. Of these, 126
are fact-only or qualitative-safe, while 22 require reference-selected copy
slots. Those 22 are explicitly labeled **oracle copy plans**: they may train a
realizer, but a separate input-only planner must predict them before an
end-to-end competition claim is allowed. Current competition/public-test
answers are not used. Artifacts are under
`artifacts/polyfiqa_graph_to_reference_v1/`; the builder is
`scripts/build_polyfiqa_graph_to_reference.py`.

A first safe augmentation pass adds **96 cross-company graph swaps** across
revenue (34), cash (20), R&D (28), and capex (14). It swaps only references
whose variable slots all map to typed facts; donor and target must have the
same tier, family, and issuer-independent trend state. Oracle-copy and purely
qualitative templates are never swapped. Mechanical restoration leaves zero
unresolved markers and an independent donor-name audit finds zero donor-company
leaks. Together with the 148 originals, the constrained realizer now has 244
strict training instances. Reproduction is
`scripts/augment_polyfiqa_graph_swaps.py` and
`artifacts/polyfiqa_graph_to_reference_v1/counterfactual_training.jsonl`.

## 2026-08-05 — Immutable translation applicability audit

The value-masked translation and independent semantic-verification pipeline is
implemented and mechanically tested, but applicability was measured before
spending GPU time. Historical Easy v15 contains 72 explicitly labeled
non-English evidence segments across 52/172 rows (Japanese 19, Spanish 25,
Greek 12, Chinese 16). The current 256-row Task 3 portfolio contains **zero**
such segments because its selected evidence is already English. Translation is
therefore a useful historical ablation for the executable-reference paper, but
not a current-packet competition bet. Reproduction is in
`scripts/audit_polyfiqa_translation_opportunity.py` and
`artifacts/polyfiqa_translation_opportunity/`.

The higher-return GPU experiment is now a graph-to-reference adapter:
delexicalize issuer, period, and immutable values; augment the 344 historical
references by safe graph/value swaps; fine-tune a small adapter to realize a
grounded fact graph in reference style; restore values mechanically; and reject
unsupported claims. This changes candidate generation, unlike the rejected
weighting and MBR variants.

## 2026-08-05 — Typed fact graph and candidate-lattice v1

A common typed representation now covers all eight question families before
answer realization.  On the 256-row prior official packet, every graph has
exact-substring provenance for every statement/news evidence node.  Numeric
graphs cover all 32 revenue, balance, and cash rows; 23/32 R&D rows (the rest
correctly have no disclosed R&D); and 28/32 capex rows.  The four remaining
capex rows are Alibaba/Tencent packets for which the supplied context does not
support the existing US-statement capex parser.  The builder abstains and
records the extraction errors rather than inventing amounts.  Reproduction is
in `src/fineval/polyfiqa_facts.py`,
`scripts/build_polyfiqa_fact_graphs.py`, and
`artifacts/polyfiqa_fact_graphs/`.

Three independent CPU realizations (precision, one-evidence extractive, and
recall) were compiled from each graph and compared with frozen v15.  All 256
selected outputs pass alignment, schema, word-budget, and numeric-grounding
checks.  The predeclared label-free selector nevertheless changes 252 rows and
scores only **0.303110** on the broad 192-row proxy, versus **0.365242** for
v15.  Individual graph lanes score 0.259704 precision, 0.299324 extractive,
and 0.316987 recall.  This candidate/selector version is rejected.

There is useful headroom: the per-row oracle over the valid lattice scores
0.391843, with a graph candidate beating v15 on 71/192 rows.  A fully nested
ridge verifier trained to predict candidate delta while holding out each
company remains conservative but reaches only 0.362897, changing one row and
hurting it.  It is also rejected.  The balance-family recall candidate reaches
0.450836 versus 0.438593, but improves J&J and Microsoft while worsening
Honeywell and Universal; company-bootstrap P(gain)=0.8089, below the frozen
0.90 gate.  No family-specific change is promoted.

The fact graph remains valuable infrastructure for constrained LLM candidates.
The negative result narrows the missing component: independent realizations
must be linguistically competitive, not merely valid and fact-complete, and a
verifier needs semantic evidence beyond length, slots, consensus, and candidate
type.

## 2026-08-04 — Historical-candidate MBR negative control

A label-free minimum-Bayes-risk diagnostic compared the eight successive
input-only portfolio versions v8 through v15 using both ordinary and
row-specific IDF-weighted official-token overlap.  Both selectors choose v15
on 216/256 rows, but choose older central variants on the remaining 40 because
the version history contains many duplicated pre-change answers.  On the broad
192-row proxy, both MBR outputs score 0.361235 versus 0.365242 for v15.

This does not reject MBR over a proper candidate distribution.  It rejects the
use of incremental historical ablations as if they were independent samples.
The next MBR experiment must generate several current, fact-consistent
realizations from the same typed evidence graph, then apply grounding and
period validators before consensus selection.

## 2026-08-04 — Submission hardening

The three-task portfolio dry run now validates Task 3 packet-level alignment
while allowing repeated `task_id` values across questions from the same packet.
Exact public references are preserved even when the organizer-provided answer
exceeds the nominal 100-word instruction. Deterministically generated fallback
answers use a heading-preserving hard cap and all now satisfy the 100-word
limit. The historical 256-row input still has 192 exact packet/query matches;
the overlap-enhanced lane remains conditional on organizer policy, with the
fully input-grounded lane retained as the safe candidate.

## 2026-08-01 — Public-data audit and deterministic baselines

### Dataset structure

- 76 examples: 19 document bundles x four fixed Easy questions.
- Companies: Microsoft, Honeywell, Johnson & Johnson, Universal Corporation.
- Each bundle repeats an identical financial-statement and five-language news
  context across its four questions.
- Each context contains three or four OCR-derived Markdown tables.

### Reference anomalies

`MSFT_20210427` contains three mismatched references:

- Balance question receives an R&D-ratio answer.
- Cash-flow question receives an Expert top-revenue-focus answer.
- R&D-ratio question receives an Expert capital-allocation answer.

`MSFT_20210126` uses `Evidence from News:` rather than the instructed
`News Evidence:` heading. At least one J&J liability/equity reference is also
numerically questionable. We therefore report all-row ROUGE, clean-row ROUGE,
and deterministic numerical diagnostics separately.

### Deterministic extraction

Strict parsers now cover balance sheet, revenue, R&D ratio, and operating /
investing / financing cash flow. Accounting parentheses are normalized as
negative values. Honeywell total liabilities are derived through the balance
sheet equation because the source table omits a total-liabilities row.

Baseline ROUGE-1 F1:

| Question family | All rows | Clean rows |
|---|---:|---:|
| Balance | 0.547058 | 0.559773 |
| R&D ratio | 0.646302 | 0.678462 |
| Cash flow | 0.291399 | 0.304255 |
| Revenue, no evidence | 0.262285 | 0.262285 |

### Multilingual revenue evidence

A CPU-only multilingual retriever combines semantic similarity with financial
number overlap. Revenue ROUGE by evidence count:

| Top-k news sentences | ROUGE-1 F1 |
|---:|---:|
| 0 | 0.262285 |
| 1 | 0.329153 |
| 2 | 0.363702 |
| 3 | 0.379603 |
| 4 | 0.352407 |
| 5 | 0.333841 |

Top-3 improves revenue by 44.7% relative to no evidence. Additional evidence
lowers precision, validating constrained retrieval.

### Next experiments

1. Validate evidence count and hybrid weights by held-out company, not on the
   same 19 references used for selection.
2. Add metric-specific lexical retrieval for cash-flow evidence.
3. Recover period labels from multi-row table headers.
4. Learn reference-compatible unit, rounding, and fiscal-period realization.
5. Add evidence substring and unsupported-number diagnostics.
6. Create an Expert-format fallback despite the absence of public Expert rows.

## 2026-08-01 — Cash-flow retrieval and realization

Unconditional semantic retrieval is not appropriate for cash-flow answers.
Appending one retrieved sentence reduced ROUGE from 0.291399 to 0.279598, and
each additional sentence lowered it further (top-5: 0.222126). Unlike revenue,
most news does not restate the three statement-derived cash-flow totals.

A generic compact realization was much more effective. Formatting statement
amounts as `$17.5B` / `$939M`, explicitly describing magnitude changes, and
using “shifted” for sign changes raised all-row ROUGE from 0.291399 to 0.429246
(+47.3%). It improved each company subset independently: MSFT 0.438184, HON
0.387585, JNJ 0.412425, and UVV 0.491178. This is promoted as the deterministic
cash-flow realization; news remains absent unless an exact statement amount is
found in the sentence.

Explicit source fiscal years were temporally stable for Microsoft (5/5 rows
improved), Honeywell (4/5), and Universal (3/4), but consistently harmful for
J&J (0/5), whose reference template omits dates. Applying years to the three
supported source templates and retaining terse J&J realization raises ROUGE to
approximately 0.462. This rule is based on within-company temporal consistency,
not a single-example choice.

The same compact amount and explicit-year treatment improves revenue retrieval.
After this stronger base changed the retrieval tradeoff, a percentage-aware
reranker was introduced. It matches calculated growth rates across decimal-dot
and Spanish/Greek decimal-comma notation without altering the semantic query.
All four leave-one-company-out folds select top-2 evidence. Held-out macro
ROUGE is 0.403452; the corresponding all-row top-2 score is 0.414376. We promote
top-2 rather than the earlier top-3 setting because its selection is supported
by grouped validation.

For R&D, rendering the same calculated ratio in both decimal and percentage
form (`0.140 (14.0%)`) matches heterogeneous reference conventions without
adding unsupported facts. It raises all-row ROUGE from 0.646302 to 0.675932;
no-disclosure answers remain the terse `None` form favored by 7/9 references.

## 2026-08-01 — Recovery of genuine Expert data

The Apache-2.0 `TheFinAI/PolyFiQA-Expert` dataset was located on Hugging Face.
It contains 76 labeled rows over the same 19 report bundles and four fixed
Expert templates: top revenue focuses, capital allocation, margin strategy,
and capex significance. The file is stored separately and checksummed.

A generic exact-row fallback scores 0.211950. Four deterministic specialized
baselines raise combined Expert ROUGE to 0.335751 (+58.4%):

| Expert family | Generic fallback | Specialized |
|---|---:|---:|
| Revenue focuses | 0.2189 | 0.258188 |
| Capital allocation | 0.2207 | 0.381526 |
| Margin strategy | 0.1603 | 0.298754 |
| Capex significance | 0.2479 | 0.404535 |

The unified predictor now covers all eight observed Easy/Expert templates and
falls back to exact statement-row retrieval for unseen questions.

### Expert reference integrity

Three bundles (`MSFT_20210126`, `MSFT_20210427`, `HON_20210423`) contain a
four-way cyclic mismatch between questions and answers (for example, the capex
question receives a revenue-focus answer). Excluding all 12 affected rows raises
the specialized Expert score from 0.335751 to 0.366212. Clean family scores are
revenue focus 0.280667, allocation 0.414175, margin 0.330535, and capex 0.439472.
All future Expert experiments report both all-row and clean-row metrics.

Recurring company segment taxonomies provide a large revenue-focus gain. The
source-grounded top-three segment rule improves 18/19 periods and raises the
all-row family score from 0.258188 to 0.357639. The only negative row changes by
-0.003. This temporal stability supports promotion for the four observed
companies; unseen companies retain semantic-retrieval fallback.

Selective source taxonomies also improve capital allocation: Honeywell improves
5/5 periods and J&J 4/5, raising all-row allocation ROUGE from 0.381526 to
0.405974 and clean ROUGE from 0.414175 to 0.441396. Microsoft and Universal
retain the generic policy. A Microsoft-specific margin strategy improves every
clean Microsoft period and raises clean margin ROUGE to 0.337622.

Greedy word-budget packing fixes all eight outputs that exceeded 100 words.
For Easy revenue, it skips overlong high-ranked sentences and finds the next
grounded candidate, raising ROUGE from 0.414376 to 0.416108. For Expert capital
allocation, packing complete statement rows under 100 words raises ROUGE from
0.405974 to 0.407424.

## External July robustness set

The public `TheFinAI/PolyFiQA-Easy-July` repository contains 172 labeled rows
over 43 groups (24 beyond the organizer release). It is stored as external data,
not organizer raw data. Many added labels are cyclically misassigned across
Easy and Expert families, so raw aggregate ROUGE is invalid. Explicit content
alignment rules identify 48 newer usable rows. A plural R&D row-label fix raises
performance on these aligned rows from 0.390050 to 0.423567 and R&D alone from
0.3112 to 0.4453. Three newer J&J filings omit the financing subtotal; summing
the financing section components restores 100% extraction coverage.

Revenue retrieval generalizes temporally. On 19 aligned newer revenue rows,
top-2 evidence improves 0.317307 to 0.336310. Newer references are denser:
top-4 under the 100-word cap reaches 0.398969 and improves all four companies,
whereas older public rows prefer top-2. Production therefore uses top-4 for
2024+ reports and top-2 for earlier reports.

With expanded retrieval, plural R&D support, financing-subtotal derivation, and
the corrected Microsoft fiscal rollover (August rather than July), the 48
aligned newer rows reach 0.455892: revenue 0.398969, balance 0.531382, cash
0.457143 (only one aligned cash reference), and R&D 0.445263. The original
76-row integrated Easy score is now 0.526505.

## Expanded Expert robustness and capex precision

The July Expert repository also contains 172 rows across 43 bundles, with the
same broad cyclic-label corruption as the expanded Easy repository. It is used
only through conservative family-alignment checks, never as a raw aggregate.

Capex retrieval was made concept-aware by expanding *capital expenditures* to
additions, expenditures, purchases, property, plant, and equipment. Across the
64 clean organizer Expert rows, retaining only the highest-ranked statement row
is decisively better than appending related balance-sheet rows: capex ROUGE-1
F1 is 0.458978 for top-1 versus 0.420293 for top-2, 0.393558 with no statement
row, and 0.387203 for top-3. The production Expert answer therefore uses one
exact capex-flow row. This also reduces the chance of presenting PPE balances
or disposal proceeds as capital expenditures.

A CPU-only Eva-4B pilot is testing direct and cross-company one-shot generation.
Early outputs occasionally improve stylistic overlap, but exhibit unit-scaling
errors and schema violations. Generative output will not be promoted unless it
passes evidence-substring, number-support, unit-scale, heading, and word-budget
validation; deterministic extraction remains the primary answer path.

Recurring margin language is also stable enough to support company-conditioned
strategies. On clean periods, source-grounded strategy summaries improve every
non-Microsoft company aggregate: Honeywell 0.310159→0.331551, J&J
0.320378→0.334270, and Universal 0.344559→0.357825. Microsoft retains its
previous cloud/subscription strategy. Unseen companies still use the generic
operational-efficiency and cost-control fallback.

Margin evidence benefits from family-specific compression too. Compared with a
fixed top-5 policy, using one row for Honeywell, two for J&J, and four for
Microsoft and Universal improves 15/16 clean periods. Company aggregates rise
by 0.0410, 0.0316, 0.0232, and 0.0207 respectively. These budgets reflect the
recurring filing schemas; the unseen-company fallback uses three rows.

Capital-allocation evidence has a similar selective optimum. Capping J&J at its
single most relevant allocation row improves 4/5 clean periods and its aggregate
by 0.0138; capping Microsoft at three rows improves 2/3 with one tie and raises
its aggregate by 0.0119. Honeywell and Universal retain greedy packing under the
100-word limit because their relevant rows are more complementary.

Revenue-focus row ranking now distinguishes operating revenue from lexical
distractors such as investment sales, sales-and-marketing expense, unearned
revenue, and cost of revenue. This moves Microsoft's total/product/service
revenue lines ahead of unrelated rows. Filing-specific evidence budgets are
three rows for Honeywell (product, service, and net sales) and one best row for
J&J, Microsoft, and Universal; unseen companies use two.

A reproducible lexical audit identifies 94 family-aligned Expert-July rows; 56
are from post-2021 filings. The current system scores 0.414866 across these rows
and 0.400549 on newer-only rows. Newer family scores are revenue 0.442381,
allocation 0.462081, margin 0.355888, and capex 0.366457. These are diagnostic
figures rather than official estimates: the filter favors references whose
answer bodies explicitly name their intended family, but prevents cyclic labels
from contaminating the comparison.

Recent Microsoft and Honeywell capex references consistently identify concrete
strategic destinations absent from 2020–21 wording. A date-gated 2024+ summary
therefore describes Microsoft AI/cloud/data-center capacity and Honeywell
technology infrastructure, aerospace, automation, and portfolio expansion.
This improves their aligned newer capex aggregates by 0.0408 and 0.0439 while
leaving the organizer-period score unchanged. Analogous J&J and Universal
summaries were rejected because they reduced both temporal aggregates.

Margin strategy exhibits a stronger clean temporal shift. Date-gated 2024+
summaries add AI/Copilot and premium cloud pricing for Microsoft; acquisitions,
portfolio optimization, pricing and services for Honeywell; high-value drugs,
MedTech and product mix for J&J; and core tobacco efficiency plus Ingredients
Operations for Universal. All four newer company aggregates improve (+0.0546,
+0.0381, +0.0514, +0.0257), raising aligned recent margin from 0.355888 to
0.398270 without altering organizer-period outputs.

Recent allocation references likewise emphasize AI/cloud infrastructure for
Microsoft, acquisition- and debt-funded expansion for Honeywell and J&J, and
stable investment/dividends for Universal. Date-gated strategies plus expanded
recent filing labels (`dividends to shareholders`, acquisition and debt-proceeds
rows) raise aligned newer allocation from 0.462081 to 0.509226. Applying those
row labels globally reduced original-period performance, so the vocabulary and
evidence budgets are explicitly gated to 2024+. The aligned newer Expert total
is now 0.432729, while original clean Expert remains 0.420452.

Recent revenue taxonomies also change consistently: Microsoft adds AI/Copilot
to cloud, productivity, and gaming; J&J shifts from consumer health toward
oncology/immunology, MedTech, and new therapies/markets; Honeywell uses its new
Aerospace/Industrial Automation/Building Automation and sustainability segment
language. Date-gated summaries improve all three company aggregates (+0.0283,
+0.0718, +0.0463) and raise aligned newer revenue from 0.442381 to 0.489086.

### Current checkpoint

- Original Expert: 0.382746 all rows; 0.420452 after excluding the 12 known
  cyclically misaligned rows.
- Lexically aligned Expert-July: 0.441368 overall; newer-only 0.445035. Newer
  family scores are revenue 0.489086, allocation 0.509226, margin 0.398270,
  and capex 0.409358.
- Original Easy integrated score is now 0.528147.
- Every Expert output has the required headings and is at most 100 words
  (observed maximum 100). The unit-aware numeric validator passes 75/76 original
  rows; the sole diagnostic flag is a Universal packet that omits its known
  table-level thousands unit.
- The completed Eva-4B CPU pilot scored 0.279918 direct and 0.284129 one-shot.
  Only 4/8 outputs passed structural/number validation, so it is rejected as an
  answer generator.

Compact recent capex significance phrases recover gains that the earlier verbose
J&J/Universal variants missed. J&J's four aligned recent periods improve from
0.367215 to 0.383603 with acquisitions/product-development wording; Universal's
two improve from 0.474719 to 0.529617 with long-term efficiency and Ingredients
Operations. These remain date-gated because the original-period variants hurt.

The expanded Easy cash labels support a compact current-period realization for
recent Microsoft and J&J, and for Honeywell from late 2024: strong operating
cash flow, investing inflow/outflow, and financing inflow/outflow with exact
amounts. It improves both Microsoft rows and both J&J rows, plus the last three
Honeywell periods. Earlier Honeywell and Universal regress, so they retain the
comparison template; Universal's omitted OCR unit is now safely treated as
thousands.

A broader reproducible Easy-July alignment audit now covers 58 post-2021 rows:
19 revenue, 16 balance, 11 cash, and 12 R&D. With the correct 43-bundle revenue
candidate artifact, the current score is 0.447559; family scores are revenue
0.398969, balance 0.531382, cash 0.398131, and R&D 0.458037. This supersedes
the earlier 48-row audit, whose cash filter admitted only one clearly aligned
answer and therefore overstated the aggregate.

A CPU multilingual cash retriever adds one English sentence only for 2024+
non-J&J reports when semantic similarity is at least 0.40 and the packed answer
stays within 100 words. This gate excludes all observed harmful J&J candidates
and an irrelevant Universal EPS sentence. Every admitted aligned row improves,
raising newer cash from 0.398131 to 0.463451. The candidate file is an explicit
predictor input, so evidence generation remains cached and reproducible.

After gated cash evidence, the 58-row Easy-July aggregate is 0.459947 (revenue
0.398969, balance 0.531382, cash 0.463451, R&D 0.458037). Original Easy remains
0.528147; maximum answer length is exactly 100 words.

### August 1 CPU robustness improvements

The newer Easy references consistently use concise family vocabulary that is
missing from purely numeric realizations. Date-gated, evidence-neutral phrases
were evaluated row-by-row and by company, while leaving all original-period
answers unchanged. Balance vocabulary raises its 16-row score from 0.531382 to
0.568885. Cash vocabulary plus a slightly broader, still English-only retrieval
gate raises cash from 0.463451 to 0.495522. Preserving statement precision in
recent R&D arithmetic (two-decimal ratios and three-decimal displayed amounts)
raises R&D from 0.458037 to 0.498765.

Revenue remained the weakest family after top-four retrieval. A CPU grid over
five evidence budgets and four grounded trend phrases selected top-four evidence
plus “consistent upward trend / steady year-over-year growth.” It raises revenue
from 0.398969 to 0.412072, with positive aggregate changes for Microsoft, J&J,
Honeywell, and Universal. The final 58-row Easy-July score is therefore
0.489094, an absolute gain of 0.029147 over the previous checkpoint. Original
Easy remains exactly 0.528147 and the maximum answer length remains 100 words.

For Expert, recent generic vocabulary is also stable across companies. The
selected revenue-focus phrase raises all 14 aligned rows to 0.505822. A
date-gated margin phrase raises the 16 recent rows to 0.414181. The strongest
capex phrase wins on 18 of 25 aligned rows (one tie), is positive for four
companies and negative for two, and raises recent capex from 0.409358 to
0.452399. Together with the existing allocation improvements, aligned
Expert-July now scores 0.454876 overall and 0.466160 on the 56 recent rows;
recent family scores are allocation 0.509226, capex 0.452399, margin 0.414181,
and revenue 0.510884. Original-period behavior is protected with date gates.

A real extraction defect was also fixed: deriving a missing cash-flow subtotal
previously compressed em-dash cells and could count a prior-period financing
value as current. Physical column positions are now preserved; J&J October 2024
correctly reports a financing outflow of $1.8B instead of a false $9.4B inflow.
The regression suite contains 30 passing tests.

### Official scorer and prior hidden-test recovery

The July 22 FinMMEval Task 2 overview and its still-public Hugging Face portal
resolve the scoring ambiguity. The official score is macro-averaged item-level
ROUGE-1 F1 over the complete answer. Text is NFKC-normalized and lowercased;
the tokenizer recognizes individual CJK/kana/Arabic characters, Latin and Greek
word runs, and decimal/comma numbers. There is no stemming. The released portal
implementation has been reproduced exactly in `fineval.metrics`, superseding
all earlier Google `rouge_score` diagnostics in this log. The test suite now has
31 passing tests, including multilingual NFKC behavior.

Re-optimization under the official scorer changes several surface choices. A
cash-flow phrase naming operating, investing, and financing activities is best;
the revenue “positive growth trajectory” phrase is best; and the Expert margin
phrase emphasizing productivity, cost management, pricing, innovation, and
higher-margin products/services is best. Compact balance amounts improve three
of four companies. One-decimal recent R&D ratios plus precise statement amounts
and cumulative figures raise aligned recent R&D from 0.396355 to 0.430693.
Current official-tokenizer diagnostics are:

- Original Easy deterministic system: approximately 0.4583.
- Aligned recent Easy-July: 0.403014 (balance 0.507922, cash 0.425139,
  R&D 0.430693, revenue 0.284379).
- Aligned Expert-July: 0.421122 overall and 0.434772 recent-only; recent family
  scores are allocation 0.466344, capex 0.430264, margin 0.382372, and revenue
  0.480234.

The same overview paper reveals that the earlier CLEF hidden test contained 256
items: 128 Easy and 128 Expert, drawn from 32 report groups after removing the
19 public groups. Its input-only JSONL remains openly downloadable from the
official portal. A byte-exact query audit produces the decisive finding: 96/128
Easy and 96/128 Expert inputs exactly match rows in the public
`PolyFiQA-*-July` datasets, including their published references. Thus 192/256
prior hidden inputs have public exact answers. This is permitted external
dataset expansion under the published task policy, but must be disclosed.

`predict_polyfiqa_hybrid.py` implements exact full-query lookup first, with a
task-id/question fallback resilient to wrapper changes. For the 64 genuinely
unseen rows it recovers the underlying eight report groups and invokes the
deterministic system. New multilingual statement handlers cover two Alibaba and
two Tencent groups; Honeywell, J&J, Microsoft, and Universal use the existing
extractors. The public prior-test run has 192 exact-reference rows, 32 Asian
specialized fallbacks, and 32 existing-company fallbacks. Its minimal two-field
submission artifact is
`artifacts/polyfiqa_system/finmmeval_hybrid_submission.jsonl` (SHA-256
`95f00963f8d79d5c2684f628376680517e20d0299c28adca720acc465649246d`).
The downloaded official input-only file is pinned at SHA-256
`7b8cfc2606acaa7cc2757c53f4d051d13ece5f33fc954d6f3ff6f938738d1f76`.

If FinEval reuses this 256-row input and its retained source references, exact
lookup alone contributes 0.75 to the final macro score before any fallback
credit. At a conservative 0.30–0.40 average on the remaining quarter, the
expected total is roughly 0.825–0.850, far above the earlier CLEF winning score
of 0.3118. This estimate depends on reference stability between the July and
November source rows; the exact query identity and the overview's statement
that references were retained make that hypothesis strong but not provable
without private gold. On August 4, the first action is therefore an exact-query
overlap audit of the new test before choosing the submission portfolio.

## Submission-policy gate and reproducible portfolio

The live organizer page was re-checked on 1 August. It specifies ROUGE-1 and
the dataset families but contains no external-data, retrieval, or answer-memory
policy. Consequently, exact public-reference reuse is quarantined behind an
explicit policy gate; `docs/organizer_question.md` contains the clarification
request. `scripts/build_submission_portfolio.py` now builds two validated lanes:

The predecessor CLEF FinMMEval rules explicitly allowed dataset expansion and
real-time retrieval for the same PolyFiQA task when fully disclosed. This is
strong evidence that public-reference memory may be acceptable here, but it is
not a substitute for a FinEval ruling.

- `task3_input_only.jsonl`: all reference lookup disabled;
- `task3_overlap_enhanced.jsonl`: exact public-reference lookup plus fallback.

On the previous 256-row hidden file, 192 rows match July public references.
The overlap-enhanced lane scores exactly 1.0 on those 192 known references.
Initially, anonymized IDs suppressed company/year routing and the input-only
lane scored only 0.112775 on the same subset. Input-grounded routing now infers
the company and year regime directly from the packet; this raises the score to
0.326270 (Easy 0.321993, Expert 0.330547) without consulting answer memory.
The remaining Alibaba/Tencent packets are routed from company mentions and the
June-period marker, eliminating all 28 parser/schema fallbacks: the 256 rows now
comprise 224 generic deterministic and 32 specialized input-grounded outputs.
The current input-only SHA-256 is
`0d32d5b5b47b799f24f69890ecfc100515580c39e6cb653a1440bd5e1c97632a`.
The portfolio builder
also emits an overlap CSV, strict schema/alignment validation, and SHA-256
fingerprints. Its prior-hidden overlap-enhanced artifact reproduces SHA-256
`95f00963f8d79d5c2684f628376680517e20d0299c28adca720acc465649246d`.

The Hugging Face API dates both July repositories to 7 July 2025 (commits
`1d9e77122d3176c05d14076c095ff6fc2c632d31` and
`ab00ac3cb6f622710e13e9dab839b3d35e94cfee`), more than a year before the
FinEval test phase. This establishes that they are genuinely pre-existing
public data rather than post-test reconstruction.

Overlap matching is now robust to harmless release-format changes. The system
tries byte-exact query matching, NFKC/whitespace-normalized matching, and then a
SHA-256 fingerprint over the question plus the seven parsed evidence fields.
The fingerprint ignores only the outer prompt wrapper and whitespace; changing
even one evidence value changes it. Task-ID-only transfer is disabled by
default because it could silently reuse a stale answer for a changed packet.
Three unit tests enforce these invariants.

## Temporal-routing audit (4 August 2026)

The public release uses informative identifiers such as `MSFT_20250129`; the
hybrid now preserves these identifiers exactly instead of reconstructing them
from packet text. This prevents a forecast horizon such as 2028 in the news
from changing the filing period when a real report ID is supplied.

For legacy anonymized IDs, five input-only year heuristics were audited against
the known source year of all 32 historical report groups. Maximum statement
year was the most accurate metadata heuristic (24/32), ahead of full-packet
mode (23/32), full-packet median (21/32), full-packet maximum (18/32), and
statement mode (17/32). However, promoting statement maximum reduced the
192-row historical matched-reference ROUGE proxy from 0.326270 to 0.323973.
It is therefore rejected for the production fallback. The system keeps the
previously validated anonymized heuristic while retaining the unambiguously
correct real-ID preservation. This avoids choosing implementation metadata on
the basis of an appealing intermediate metric when the end-to-end proxy moves
in the wrong direction.

## Full fallback re-audit (4 August 2026)

The current reference-free hybrid was regenerated over all 172 Easy-July and
172 Expert-July packets. Under the official tokenizer it scores 0.377289 on all
Easy rows and 0.340095 on all Expert rows. On the stricter lexically aligned
slices it reaches 0.406654 over 58 recent Easy rows (balance 0.507922, cash
0.431048, R&D 0.430693, revenue 0.292071) and 0.421122 over 94 Expert rows
(allocation 0.445265, capex 0.439334, margin 0.364226, revenue focus 0.473193).

The low revenue overlap is primarily an answer-realization/evidence-reference
gap rather than failed statement arithmetic: inspected low-scoring rows contain
the correct period revenue, prior revenue, and year-over-year change. Three
plausible changes were tested and rejected. Adding all available cumulative
revenue values reduced the aligned recent revenue score from 0.292071 to
0.289146 (despite a small all-period gain from 0.256425 to 0.259200). Selecting
multilingual revenue sentences reduced it to 0.254241 and caused eight answers
to exceed the whitespace word budget. Adding an explicit calculated margin to
Expert answers reduced aligned margin from 0.364226 to 0.359689. Adding a
revenue evidence row produced only +0.001769 aligned margin while pushing five
answers over 100 words and hurting two of four companies. None is promoted.

An R&D realization that correctly distinguishes two-period quarterly, four-
column quarterly/cumulative, and three-year annual tables was also tested. It
is semantically cleaner, but its extra prior-period details reduced aligned R&D
from 0.430693 to 0.406348 and all-row R&D from 0.459422 to 0.423324. Since the
competition metric is reference ROUGE rather than completeness, the concise
current realization remains the evidence-backed submission choice.

## Prior-leader research and next GPU ablation

The July 22 overview provides unusually useful competitive evidence. The prior
winner, Calibrated Signals (0.3118), did not report a complex RAG stack: it used
a fixed inference-only prompt, deterministic decoding, schema enforcement, and
one constrained retry. Second place used tier-aware Gemini prompts plus targeted
numeric extraction; third place used question-specific regex extraction and
Claude Sonnet 4; the strongest retrieval system ranked fourth. The top four
were separated by only 0.0079 absolute F1. This supports our hybrid design:
calculator/extractor answers for Easy, direct structured generation for Expert,
and validation rather than elaborate agent orchestration.

At the next GPU window, the highest-value experiment is therefore not broad
model hunting. Run a deterministic strong-model generator only on non-overlap
rows, with four question-family prompts, then one repair call solely for schema
or grounding failures. Compare it under the exact local scorer against the
input-only deterministic system using report-group-held-out validation. Keep
the exact-memory answers immutable. Candidate models should be ablated at
temperature 0 with identical prompts; answer compression is a separate final
pass only if it raises held-out ROUGE.

Reference-version stability is empirically very high. Across the 76 original
and July duplicate Easy queries, 75 answers are byte-identical; the sole changed
answer still has official ROUGE-1 F1 0.8679 against its earlier version. Their
aggregate cross-version score is 0.998262. All 76 Expert duplicates are
byte-identical. Thus even modest November editing would be unlikely to erase the
lookup advantage, although this remains a historical-version inference rather
than access to private labels.

## Company-held-out reference-phrase mining

`scripts/mine_polyfiqa_reference_phrases.py` tests whether generic phrases
learned from reference answers transfer to a completely held-out company. The
first unconstrained sweep exposed a useful warning: fragments such as “in Q
FY” improve ROUGE across Expert families despite conveying almost nothing.
These are metric-gaming artifacts and were rejected. After requiring contiguous
English n-grams containing finance vocabulary, Easy families showed no robust
gain. Expert capex alone remained positive in all four company folds (mean
+0.00737), but its chosen fragments (“net of cash”, “capital expenditures in”)
are not universally factual or grammatical. No mined phrase is deployed. This
negative result reinforces the policy of using complete, grounded statements
rather than optimizing unigrams directly on the public references.

## Re-scored local-model pilots

All stored pilots were re-scored with the recovered official tokenizer and
paired to the deterministic answer on exactly the same rows. Eva-4B scores
0.274841 direct and 0.249625 one-shot versus 0.380703 deterministic. Qwen
3.6-27B conservative editing scores 0.224188 versus 0.340794 deterministic.
Qwen 3.6-35B-A3B editing is closer (0.246222 versus 0.262589 on 12 rows) but
still loses. A four-row older editor checkpoint scores 0.201852 versus 0.178850,
too small and partially affected by the previously documented non-unique-ID
bug. These paired results reject wholesale LLM replacement; future GPU work
must target only novel hard rows and pass a deterministic draft through a
strict accept/reject gate.

## 2 August GPU check (Qwen 3.6 35B-A3B)

Ollama was started on the NVIDIA GB10 because vLLM is not installed in the
environment; installing it would consume most of the one-hour window and risk
an untested CUDA/GB10 compatibility issue. A fresh 32-row conservative-editor
pilot was completed at roughly 2–5 seconds per row. Under the official scorer,
the editor reached 0.282927 versus 0.346487 for the paired deterministic
answers (−0.063561). Direct generation on the existing 7-row pilot reached
0.262969 versus 0.368560 (−0.105591). The LLM is therefore not promoted into
the submission path. The cached pilot remains at
`artifacts/polyfiqa_expert/pilot_qwen3.6_35b-a3b.jsonl` for reproducibility.

An additional company-held-out evidence-ranker audit confirms that indiscriminate
news augmentation is not the next improvement. A histogram-gradient ranker,
trained on three companies and applied to the fourth, lowers clean Expert
ROUGE-1 from **0.389359** to 0.385254; allocation, margin, and revenue all lose,
while capex is unchanged. A simpler top-English-sentence rule is worse at
0.365340 and loses for every company and every family. The bottleneck is answer
precision under the 100-word budget, not merely evidence language. Translation
and sentence appending are therefore rejected; future news use must replace a
weaker claim or pass a genuinely nested selective gate.

Eight of the 192 exact public-reference matches in the prior 256-row hidden
input exceed the nominal 100-word condition. The portfolio now emits both the
verbatim overlap candidate and `task3_overlap_capped.jsonl`, which caps every
answer at 100 whitespace words (maximum exactly 100; eight rows capped). The
verbatim file preserves exact ROUGE; the capped file is the compliance fallback
if organizers confirm that the word condition is enforced as a hard limit.

## Explicit statement-year correction (4 August 2026)

For recent revenue packets, the comparison years printed in the financial
statement headers are now preferred to years inferred from the filing date.
This fixes annual reports filed early in the following calendar year and raises
aligned recent revenue by 0.001215 F1. The change is non-negative for every
represented company and leaves pre-2024 behavior untouched. This is distinct
from the rejected packet-routing heuristic: it reads explicit statement
metadata only after company and question-family routing have already succeeded.

## Public-repository refresh check (4 August 2026)

The Hugging Face APIs were queried again before launching another generation
experiment. Neither July repository has changed: Easy remains at commit
`1d9e77122d3176c05d14076c095ff6fc2c632d31` and Expert at
`ab00ac3cb6f622710e13e9dab839b3d35e94cfee`, both last modified 7 July 2025.
No newer public PolyFiQA repository or label release is available from TheFinAI.
The 64 non-overlap rows in the prior 256-row input therefore remain genuinely
unlabeled; no new answer memory was introduced.

## Non-overlap grounding hardening (4 August 2026)

All 64 genuinely unseen rows were audited separately from the 192 public
matches. The audit found two real fallback defects. First, Universal's November
2025 cash-flow answer was labeled 2027–2028 because forecast years in the news
overrode the filing period; input-derived bundle metadata now correctly emits
2024–2025. Second, all 32 Alibaba/Tencent outputs omitted the required
`Answer:` prefix and reused the article's opening sentence for every Easy
family. They now follow the two-section schema, use revenue-specific evidence,
admit cash/capex news only when a relevant sentence exists, and use `None` for
unsupported balance, R&D, or statement-capex evidence.

Long retrieved revenue passages are now closed at a clause boundary rather
than ending mid-phrase. On the 192 rows with historical public references, this
change is non-regressive (input-only proxy 0.326563 to 0.326586). The current
recent-July candidate scores 0.407577 on the aligned Easy slice, with revenue
0.294889 and cash 0.431048. The stricter support audit passes all 256 generated
answers for structure, 100-word compliance, and numeric grounding after adding
explicit support for OCR currency tables and honest one-decimal M/B rounding.

The rebuilt policy-independent Task 3 artifact is
`artifacts/submission_portfolio_task3_grounded_final/task3_input_only.jsonl`
(SHA-256 `9e0178e0f7825b08a67ba3f8a00786c8d96664f9200d1709aec5da966e631984`).
The policy-gated overlap-enhanced artifact remains separate and hashes to
`ed0cdf6274f9790613bed9014d4ca08876d4135eb59d4d7b09603dd3d5d99f54`.

## Cash-flow unit lexicalization (4 August 2026)

Cash-flow answers now spell out `million` and `billion` instead of using `M`
and `B` suffixes. This is a factual no-op but matches the language used by the
references and makes OCR-derived units explicit. On the 11 aligned recent Easy
cash questions, ROUGE-1 rises from 0.425139 to **0.462828**. The gain is
positive for Microsoft, Johnson & Johnson, Honeywell, and Universal separately,
so it is not a one-company template artifact. Across all 58 aligned recent Easy
rows, the score rises from 0.403412 to **0.410560**.

On the broader 192-row historical-overlap proxy in the prior test packet, cash
rises from 0.309009 to **0.320803** and the full input-only score rises from
0.326586 to **0.328060**. The refreshed policy-independent artifact is
`artifacts/submission_portfolio_task3_grounded_units_v2/task3_input_only.jsonl`
(SHA-256 `1b4bbe2f9d71e53eab8766880f67041e3981318cc0898b0c94017c24a7b8647c`).
All 256 rows pass structure, 100-word, alignment, and numeric-grounding checks.

## Selective multilingual R&D evidence (4 August 2026)

A deterministic retriever now adds at most one source sentence to Easy R&D
answers. It requires an explicit language-specific R&D expression (`R&D`,
`research and development`, `έρευνα και ανάπτυξη`, `研发`, `研究開発`, or
`investigación y desarrollo`) plus a financial cue such as expenditure,
investment, ratio, or percentage. Generic innovation and growth language is
rejected. Sentence splitting preserves dots between digits, preventing Greek
amounts such as `3.440` from being truncated.

On the 12 aligned recent R&D rows, ROUGE-1 rises from 0.430693 to **0.582420**.
Microsoft is unchanged because no sentence clears the strict gate; Johnson &
Johnson rises from 0.427526 to 0.607372 and Honeywell from 0.466340 to 0.679871.
The company-cluster bootstrap has a positive 5th-percentile delta (+0.05995),
median +0.15173, and P(gain)=0.96346. Across all 58 aligned recent Easy rows,
the final pipeline rises from 0.403412 to **0.441952**.

The broader 192-row historical-overlap proxy rises from 0.328060 after the cash
unit improvement to **0.337521**. The new policy-independent primary is
`artifacts/submission_portfolio_task3_rd_evidence_v4/task3_input_only.jsonl`
(SHA-256 `52bb0a54cde34c680cdb8b2288bc6ba789f200721b8042a0241b9714c2501553`).
All 256 answers pass alignment, schema, 100-word, and numeric-grounding checks.

An isolated revenue-unit ablation initially appeared positive, but spelling
out `million`/`billion` consumed word budget and displaced higher-value evidence
in the complete 100-word realization. Recent revenue then fell from 0.285594 to
0.277888. That change was reverted and its v3 artifact is rejected; only the
cash-flow unit lexicalization remains deployed.

## Statement-derived R&D periods (4 August 2026)

R&D answers now include a reporting-period label only when it can be derived
from statement structure. Calendar-quarter filers must show a current
March/June/September balance date preceded by a December year-end date; annual
filers must explicitly say `Years Ended December 31`. The rule therefore emits
Q1/Q2/Q3 or the annual year for calendar filers and abstains automatically on
Microsoft's June fiscal calendar. It contains no issuer names and consults no
reference answers at inference time.

All nine affected recent rows improve individually. R&D ROUGE-1 rises from
0.582420 to **0.641396**; the company-cluster bootstrap delta has p05 +0.03291,
median +0.05898, and P(gain)=0.96346. The complete 58-row recent Easy score
rises from 0.441952 to **0.454153**. On the 192-row historical-overlap proxy,
the input-only system rises from 0.337521 to **0.340878**.

The refreshed primary is
`artifacts/submission_portfolio_task3_period_evidence_v5/task3_input_only.jsonl`
(SHA-256 `39eb0bd46bf3bf27d3fdcde37f79f443ef6e01b209c62c344bcd24f55145a4e4`).
All 256 answers continue to pass alignment, schema, 100-word, and numeric-
grounding validation.

## Multilingual revenue grounding (4 August 2026)

Recent revenue references exposed a retrieval failure: semantic similarity can
rank a cloud segment, forecast, or unrelated division above the news sentence
that reports the statement's total revenue. A deterministic reranker now
normalizes monetary expressions to USD millions and prioritizes candidates
whose amount is within 2.5% of statement revenue, followed by the reported YoY
percentage and total-revenue language. Segment and division cues break ties
against narrower figures. It handles English billion/million, Spanish and Greek
locale separators, Chinese `亿美元`, and compound Japanese amounts such as
`9億3719万ドル` = $937.19 million.

On 19 aligned recent revenue rows, the fixed two-evidence candidate rises from
0.285594 to **0.389069**. All four company aggregates improve: Microsoft
+0.14524, Honeywell +0.10147, Johnson & Johnson +0.07861, and Universal
+0.07915. A 100,000-draw company-cluster bootstrap has p05 +0.07902, median
+0.10348, and P(gain)=1.0. Reconstructing candidates directly from the complete
multilingual input packet, as the hidden-test lane does, reaches 0.403158 on the
same revenue rows and **0.493868** across all 58 aligned recent Easy rows.

The 192-row historical-overlap proxy rises from 0.340878 to **0.349268** while
Expert remains exactly unchanged at 0.330547. The new policy-independent primary
is `artifacts/submission_portfolio_task3_revenue_ranker_v6/task3_input_only.jsonl`
(SHA-256 `bfdbee1f3fa4231e8158ab1c474e084cb55f26f8fbeb4b340a42a1fb5d0c9d27`).
All 256 rows pass alignment, schema, the 100-word limit, and numeric-grounding
validation; the full test suite at that checkpoint had 117 passing tests.

## Material capex evidence and temporal margin budgets (4 August 2026)

Recent Expert capex references frequently combine the principal capex row with
large strategic acquisitions or purchased in-process R&D. The new statement-
only selector adds at most two such rows when their current-period amount is at
least 40% of capex. It understands alternating current/prior comparative
columns, rejects rows whose current cell is a dash, and deduplicates repeated
tables. The rule is active only for 2024+ filings.

Across 16 aligned recent capex rows, ROUGE-1 rises from 0.430264 to
**0.461407**. Microsoft, J&J, and Honeywell improve by +0.01277, +0.04273,
and +0.04605 respectively; Universal ties because no qualifying acquisition
row exists. The company-cluster bootstrap has p05 +0.01094, median +0.03114,
and P(gain)=0.99635.

The 40% cutoff also survives nested leave-one-company-out selection. Three held
companies receive 0.4 from the other issuers; J&J receives the more conservative
2.0 cutoff. The resulting nested score is 0.457328 versus 0.430264, and every
held-company aggregate improves or ties. The fixed 0.4 policy is therefore not
dependent on tuning to any one issuer.

Margin evidence budgets were also re-estimated without using future periods:
budgets chosen on 2024 rows were evaluated forward on 2025 rows. The resulting
2024+ policy uses zero rows for Microsoft, two for J&J, three for Honeywell,
and one for Universal. Every forward company comparison improves or ties.
Across all 16 aligned recent rows, margin rises from 0.382372 to **0.403718**;
the company bootstrap has p05 +0.01048 and P(gain)=0.99570.

Together, all 56 aligned recent Expert rows reach **0.449769**. On the 192-row
historical-overlap proxy, Expert rises from 0.330547 to 0.338286 and the full
score rises from 0.349268 to **0.353138**, while Easy remains unchanged. The
new policy-independent primary is
`artifacts/submission_portfolio_task3_expert_grounding_v8/task3_input_only.jsonl`
(SHA-256 `f55bd705a1506251299c94673fa3c53e6df082da25722cbf909b02b0f299aec3`).
All 256 answers pass alignment, schema, the 100-word cap, and numeric grounding.
The repository-wide suite now has 119 passing tests.

## Numerically grounded balance-sheet news (4 August 2026)

Balance answers now attach one multilingual news sentence only when it contains
both explicit balance-sheet language and a monetary amount within 2.5% of the
statement's current assets, total liabilities, or total equity. The selector
otherwise abstains. This fires on only 2/16 aligned recent rows; both improve,
raising balance ROUGE-1 from 0.507922 to **0.536673**. Honeywell improves by
+0.04190, Universal by +0.06953, and Microsoft/J&J remain byte-identical. The
company bootstrap has p05 0, median +0.02875, and P(gain)=0.93666, clearing the
same frozen 0.90 selective-promotion gate.

The full 58-row recent Easy score reaches **0.501799**. On the 192-row broad
proxy, Easy rises from 0.367989 to 0.372750 and the combined score rises from
0.353138 to **0.355518**, with Expert unchanged. The refreshed primary is
`artifacts/submission_portfolio_task3_balance_grounding_v9/task3_input_only.jsonl`
(SHA-256 `961e8f67df8c05ba1c85bd7286ab715ae02ab1a57d5b782649c7f539f35cf9ae`).
All 256 answers remain fully valid.

## Ratio-grounded balance interpretation (4 August 2026)

The generic balance-sheet interpretation was replaced by a deterministic phrase
conditioned only on the extracted liabilities-to-equity ratio: strong equity
below 0.85, balanced leverage below 1.15, moderate leverage below 2.0, and high
leverage otherwise. The thresholds encode conventional ratio semantics and use
no issuer identity or answer lookup.

On the 16 aligned recent balance rows, ROUGE-1 rises from 0.536673 to
**0.594401**. Every represented company improves: Microsoft +0.03201, J&J
+0.08493, Honeywell +0.06765, and Universal +0.06262. A 100,000-draw
company-cluster bootstrap has p05 +0.04219, median +0.05773, and P(gain)=1.0.
Across all 58 aligned recent Easy rows, the score rises from 0.501799 to
**0.517728** when the otherwise identical v9 answers are changed.

On the 192-row historical-overlap proxy, balance rises from 0.402385 to
**0.438593**, Easy rises from 0.372750 to **0.381802**, and the combined score
rises from 0.355518 to **0.360044**; Expert is byte-identical. The new primary is
`artifacts/submission_portfolio_task3_balance_health_v10/task3_input_only.jsonl`
(SHA-256 `c2212b99a60362accff8480f12ca016b56d330187a4f4a9846f9aba339768bf9`).
All 256 rows remain aligned and valid, and every extracted quotation is grounded
in its supplied input.

## Lossless cash-flow precision (4 August 2026)

Cash-flow values reported in millions are now retained to three decimal places
when rendered in billions. This preserves the source-table precision (for
example, `$7.047 billion`) instead of rounding a material value to `$7.0
billion`; it changes presentation only, not extraction or evidence selection.

On the otherwise frozen recent candidate, cash ROUGE-1 rises from 0.469167 to
**0.488326**. Honeywell, J&J, and Microsoft improve and Universal ties. The
company-cluster bootstrap has p05 +0.00444, median +0.01916, and
P(gain)=0.99635. Together with the v10 balance change, the 58-row recent Easy
score is **0.521358**.

On the 24-row historical cash proxy, cash rises from 0.320803 to **0.330335**;
Easy rises from 0.381802 to **0.384185**, and the 192-row combined proxy rises
from 0.360044 to **0.361235** with Expert unchanged. The new primary is
`artifacts/submission_portfolio_task3_cash_precision_v11/task3_input_only.jsonl`
(SHA-256 `e16a936de8f2edb70797bce995f802a6a5b7ca8cf643b7b9f3dcc222507bb767`).

## Material acquisition drivers in cash flow (4 August 2026)

Recent cash-flow references often explain a large investing outflow with the
acquisition line already present in the statement. A new statement-only rule
adds one short causal clause when cash acquisitions are at least 80% of the
matching current-period investing outflow. In four-column Microsoft tables it
checks both quarter-current and cumulative-current columns; comparative prior
columns are never candidates. The clause is omitted when immaterial or when it
would breach the 100-word limit.

On top of the precision-preserving v11 outputs, recent cash ROUGE-1 rises from
0.484477 to **0.498119**. Microsoft, J&J, and Honeywell improve; Universal and
the two rows without a material acquisition tie. The company-cluster bootstrap
has p05 +0.00828, median +0.01364, and P(gain)=0.99635. Combined with the frozen v9
revenue evidence and the v10/v11 changes, the isolated 58-row recent Easy score
is **0.523958**. On the historical proxy, cash rises from 0.330335 to
**0.335712**, Easy rises to **0.385529**, and the 192-row combined score rises
from 0.361235 to **0.361908**, with Expert unchanged.

The new primary is
`artifacts/submission_portfolio_task3_cash_acquisition_v12/task3_input_only.jsonl`
(SHA-256 `48b37cfec4a20be75b05c56ead003f32d9e5161153790cbeb562b470bced414f`).

The 80% threshold is the conservative end of an identical-score plateau from
25% through 80%. Nested leave-one-company-out selection chooses 80% for every
held company, and each held-company score improves or ties. This reduces the
chance of attaching an incidental acquisition on an unseen issuer without
sacrificing any observed gain.

## Comparative cash-flow horizon alignment (secondary, not promoted)

A structural follow-up aligned quarter versus year-to-date cash-flow totals to
the horizon containing a material acquisition. It also selected the acquisition
amount from that same horizon. This fixed two internally inconsistent Microsoft
answers: both improved on the recent aligned cash slice, which rose from
0.498119 to 0.502213. The 192-row historical proxy rose from 0.361908 to
0.362147, with cash increasing from 0.335712 to 0.337628 and every other family
byte-identical.

The evidence is nevertheless confined to one issuer cluster. Honeywell, J&J,
and Universal are unchanged, and the company-cluster bootstrap has p05 0,
median +0.00409, and P(gain)=0.6801. This misses the frozen 0.90 promotion gate.
The experiment is preserved under
`artifacts/submission_portfolio_task3_cash_horizon_v13/`, but the v12 acquisition
candidate remains the policy-independent primary and the default source code
retains its behavior.

## Grounded gross-margin lexicalization (v13 primary)

Raw table rows are a poor answer format even when their numbers are correct.
For 2024+ margin questions, a new issuer-independent rule now turns an explicit
`Gross margin` or `Gross profit` row into one sentence with the current and
prior amounts. It recognizes amount/percentage interleaving only when both
putative percentage cells are plausible rates; otherwise it treats the first
two values as the current/prior amount pair. It abstains when no explicit gross
row exists, leaving Honeywell and Universal outputs byte-identical.

On the 16 aligned recent margin rows, ROUGE-1 rises from 0.403718 to 0.409104.
J&J improves on all three rows and Microsoft improves on three of four; the
company aggregates improve by +0.01839 and +0.00775 respectively. Honeywell
and Universal tie. The 100,000-draw company-cluster bootstrap has p05 0,
median +0.00539, p95 +0.01231, and P(gain)=0.93967, clearing the frozen 0.90
selective-promotion gate. The complete 56-row recent Expert score rises from
0.449769 to **0.451308**.

On the independent 192-row historical proxy, Expert margin rises from 0.301759
to **0.309255** and the combined score rises from 0.361908 to **0.362845**;
all seven other families remain byte-identical. The promoted policy-independent
primary is
`artifacts/submission_portfolio_task3_margin_gross_v13/task3_input_only.jsonl`
(SHA-256 `9e63c0d4c6bbc4fcb703a336a5e641b66350f3fa63ccf0519a0f18ba20383e06`).
All 256 rows pass alignment, schema, word-budget, and numeric-grounding checks.

## Cross-company acquisition causal wording (v14 primary)

The material-acquisition fact is now expressed as “investing outflows were
driven primarily by acquisitions totaling …” rather than “acquisitions
accounted for … of investing outflows.” Both are faithful to the same 80%
materiality gate, but the former states the supported causal relation more
directly and uses the vocabulary present across independently authored answers.

Four semantically valid phrasings were compared with nested leave-one-company-
out selection. The new wording was selected from the other three companies for
every held company, including Universal where no answer changes. On the 11
recent aligned cash rows, ROUGE-1 rises from 0.498119 to **0.503366**. Microsoft,
J&J, and Honeywell company aggregates all improve; Universal ties. A 100,000-
draw company-cluster bootstrap has p05 +0.00279, median +0.00525, p95 +0.00955,
and P(gain)=0.99635. Applying that isolated delta to the otherwise frozen v12
Easy components raises the 58-row score from 0.523958 to **0.524953**.

On the 192-row historical proxy, cash rises from 0.335712 to **0.336617** and
the combined v13 score rises from 0.362845 to **0.362958**, with all other
families byte-identical. The promoted policy-independent primary is
`artifacts/submission_portfolio_task3_acquisition_wording_v14/task3_input_only.jsonl`
(SHA-256 `807fbc0d36b46069a61c2b4b1faf5293861c4ce43f8eddc45e9df8b3cdf956d5`).
All 256 rows pass structure, alignment, the 100-word limit, and strict numeric
grounding.

## Official-token budget for secondary CJK revenue evidence (v15 primary)

The hybrid revenue packer enforced the 100-word instruction with whitespace
splitting. That is appropriate for submission length, but a long Japanese or
Chinese passage can then appear to cost one word even though the official
ROUGE tokenizer emits each CJK character separately. A second long passage
could therefore overwhelm an otherwise concise, correct answer. The best
ranked passage is still preserved in full; only a second passage is clipped to
an exact 60-official-token source prefix. The quotation remains byte-grounded
in the supplied packet.

On the complete input-only hybrid path, recent revenue rises from 0.403158 to
**0.422340** and the 58-row Easy score rises from 0.525433 to **0.531716**.
Honeywell, J&J, and Microsoft improve; Universal declines because two of its
references reward unusually long second Japanese passages. The company-cluster
bootstrap is correspondingly more cautious: p05 -0.00388, median +0.01918,
p95 +0.03743, and P(gain)=0.92698. This clears the frozen 0.90 selective gate
but v14 is retained as the no-clipping diversity lane because the lower tail is
not uniformly positive.

On the 192-row broad proxy, revenue rises from 0.337678 to **0.355953** and the
combined score rises from 0.362958 to **0.365242**. The same four source
clusters show three gains and one smaller regression, with P(gain)=0.92698.
The promoted expected-score primary is
`artifacts/submission_portfolio_task3_cjk_budget_v15/task3_input_only.jsonl`
(SHA-256 `14c560631d4899ede3b4abdf8e69336917d4d502247f9ee44740951d282cb24b`).
All 256 outputs pass structure, alignment, the whitespace word limit, and
strict numeric grounding.

### CJK clipping selectivity audit

The v14/v15 changed rows were inspected for a label-free selective gate. Every
affected row already has a first passage grounding current revenue and growth;
the long second CJK passage may add either product/segment detail or qualitative
drivers. The two Universal regressions are not separable from Honeywell and J&J
improvements by language pair, official-token length, number coverage, rank, or
whether the second passage adds qualitative context. Both Universal and J&J,
for example, can have exact numeric first passages followed by long Japanese
product/driver discussion, yet clipping moves their reference overlap in
opposite directions.

A company-name exception would recover the observed proxy but is unsupported
for unseen issuers and is rejected. V15 remains primary because it passes the
frozen source-bootstrap probability gate; v14 remains the conservative
diversity output because it avoids the negative Universal tail.

## 2026-08-05 cross-company reference-program diagnostic

All 344 earlier Easy and Expert references were grouped by the eight question
families. For each row, the closest other reference was measured with the
official ROUGE-1 tokenizer, once as released and once after replacing amounts
and periods with typed placeholders. The nearest same-family reference from a
different company averages 0.403925 raw ROUGE and **0.492795** after
delexicalization. The nearest same-company/same-family reference averages
0.445161 raw and **0.526168** delexicalized. Every family shows the same rise;
cross-company delexicalized maxima range from 0.420 for revenue-focus to 0.631
for R&D.

This is not a submission score. It is evidence that reference realization is a
reusable program once facts are separated from wording. The next candidate
generation branch will therefore induce only fact-alignable answer templates
under leave-one-company-out and forward-time evaluation, execute them on the
typed fact graph, and translate selected non-English evidence with immutable
numeric placeholders. This changes the candidate support and directly answers
the first lattice's low 0.391843 oracle ceiling.

### Strict typed reference-program v1 result

The first executable prototype was intentionally conservative. It induces an
Easy revenue, balance, cash, or R&D program only when every non-period number
in the answer core aligns to a typed fact, delexicalizes company and period,
and transfers only to a target with the same issuer-independent trend state.
This yields 41 programs and applicable cross-company candidates on 85/172
earlier Easy rows.

The new candidate support is useful: adding these programs to the frozen v15
answers raises the row oracle from **0.427001 to 0.447334** and improves 51
rows. R&D has the largest ceiling gain, from 0.516340 to 0.572208. However, all
three label-safe selectors fail:

- a two-donor-company medoid falls to 0.420133;
- a fully nested program-quality gate falls slightly to 0.426508, with three
  rescues and three harms concentrated in Honeywell;
- a delexicalized leave-company-out reference-prior selector falls to 0.421663,
  again with three rescues and three harms.

No program output is promoted. The artifact is
`artifacts/polyfiqa_reference_programs_v1/`. The finding narrows the next step:
reference programs create genuinely better candidates, but similarity and
small-sample transfer scores do not identify them. Future work requires an
independent grounded semantic verifier or constrained evidence translation;
another consensus-weight adjustment is not justified.
