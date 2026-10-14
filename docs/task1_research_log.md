# Task 1 research log

## 2026-08-05 — Within-batch percentile calibration fails external transfer

The hurdle score was converted to an empirical average-rank percentile within
each held group, a transform frozen before any validation operation outcome at
SHA-256
`010c7b953b879c0f27f5db8999d622309968678382512cc7b626a1524f9cb1e9`.
This tests whether relative ordering transfers when absolute probabilities do
not.

No percentile threshold passes.  Threshold 0.94 is the closest: 27 selected
operations, +0.018184 combined macro Entity F1, bootstrap p05 +0.008805, and
harmful fraction **0.148148**.  Nevertheless the independent external group
regresses by **-0.010606** while the five OOF groups improve.  Threshold 0.99
has positive deltas in all groups and zero harmful edits, but selects only six,
failing the immutable minimum of 25.  The policy therefore abstains without
using validation outcomes and exactly preserves **0.933157**.  No quota or
group-specific threshold is introduced after inspection.  The summary is
`artifacts/task1_rank_invariant_hurdle_v1/summary.json` (SHA-256
`5aafc482c0b118db2e5438f9f7bd29f443fa1d6bd06748e4e4dcda2240b2f740`).

## 2026-08-05 — Harm modeling succeeds; absolute deployment calibration does not

A preregistered hurdle model separates the probability that an operation has a
nonzero scoring effect from the direction of that effect.  Its frozen signed
score is `p(nonzero) * (2 p(beneficial | nonzero) - 1)`; the preregistration
SHA-256 is
`55f879bdab0f2b0af141252edc3ab9cb26b24a1dc778b8accf532ff2bb339672`.

The six-group calibration gate now passes.  At threshold 0.39, 36 of 43 edits
are beneficial, five harmful, and two neutral, for harmful fraction
**0.116279**.  Combined macro Entity F1 improves from **0.837413 to 0.871185**
(+0.033772); every group and entity type improves, bootstrap p05 is +0.021474,
and P(gain) is 1.0.  This confirms that consequence/direction decomposition is
substantially safer than the previous binary target.

The score scale shifts at deployment: calibration spans -0.952 to +0.900, but
validation spans -0.855 to only +0.170.  The immutable 0.39 gate therefore
accepts nothing and exactly preserves **0.933157**.  The branch is not promoted
and the threshold is not changed after inspection.  The remaining hypothesis
is rank- or quantile-invariant calibration, not a looser absolute threshold.
The summary is `artifacts/task1_harm_aware_hurdle_v1/summary.json` (SHA-256
`b7f5568b357c55be0b88e4e85aad99828f20fefae762413c85506e526f97d831`).

## 2026-08-05 — External-label calibration remains too harmful to deploy

The 100 independently published Plutus test labels were used as a sixth,
indivisible calibration group alongside the 320 genuine OOF rows.  All 80
FinEval validation examples remained held out.  The preregistration was frozen
at SHA-256
`a1666ac4e13025d2dbd8590f6578d6db0069033b62b4408ac99ae8d8c6753263`
before running the six-group cross-fit experiment.

Across 420 calibration rows, the exact dual-seed proposer intersections induce
414 incumbent-disagreement operations, 100 of them beneficial.  Thresholds
0.43 and 0.44 are the strongest diagnostics: they improve macro Entity F1 from
**0.837413 to 0.874939** (+0.037526), improve every group and entity type, and
have bootstrap p05 +0.021373.  But 17 of 74 accepted operations are harmful
(**0.229730**), failing the frozen maximum of 0.15.  Consequently the gate
abstains, applies zero validation operations, and reproduces the **0.933157**
incumbent exactly.  No threshold is retuned after inspection.  The branch is
stopped and the primary remains unchanged.  The summary is
`artifacts/task1_external_operation_calibration_v1/summary.json` (SHA-256
`3def36ce80b084320f1e74b21f2cdb007ae803c0cd1ea55c1e20489df19c4d83`).

## 2026-08-05 — Exact dual-raw-proposer policy transfers by abstaining

A residual-driven hypothesis asked whether exact span/type agreement between
the two independent raw GLiNER proposer seeds could recover candidates rejected
by the existing operation views. Because the validation residuals motivated
the idea, the evidential limitation and a second 100-row historical replication
were frozen in advance. The preregistration SHA-256 is
`62693bebebd4df982bf3b9c1246f6dc2e06a3763db3162d629bd0051c67a4083`.

On 320 genuine OOF rows, the two seeds share 981 exact typed character spans.
Among the 330 induced incumbent-disagreement operations, 86 are beneficial.
The frozen harm-controlled gate selects probability 0.56: 52 operations, of
which 40 are beneficial, seven harmful, and five neutral. It raises OOF macro
Entity F1 from **0.819655 to 0.865548** (+0.045893), improves all five
entity-linked folds and all three entity types, and has bootstrap p05
+0.029125.

The same fitted classifier is much more conservative under dual vote-3
deployment. It selects zero of 65 validation operations, exactly preserving
the **0.933157** primary. On the independently released 100-row Plutus test it
selects one operation, correctly joining `Χρηματιστήριο Αξιών EURONEXT` and
raising the historical score from **0.894236 to 0.897570** with no harm.

The historical replication gate passes, but the preregistered validation gate
requires at least two improvements and therefore fails. The branch is rejected
rather than recalibrated after seeing these results. This is useful evidence
that exact proposal agreement transfers, while the operation probability scale
does not transfer from single OOF models to stricter fold ensembles. The
current grouped primary remains unchanged. The frozen summary is
`artifacts/task1_dual_raw_proposer_v1/summary.json` (SHA-256
`8dc34b1a73e8a50adfbb23c853186f70fa8f80a75612869161b92ce23f9150c2`).

## 2026-08-05 — Public-label expansion passes held-out and same-code controls

The Apache-2.0 Plutus GRFinNER release contains 100 labeled test examples that
the current FinEval deployment models had deliberately quarantined.  A new
external-data lane tests whether those examples provide ordinary transferable
supervision rather than exact lookup.  The experiment was preregistered before
training (SHA-256
`b0860cf10cc28631499bf5d419111e0b525fe40c8301fef1039585f7c97e8026`):
the legal-BERT encoder trained on Plutus train+test while all 80 validation rows
remained held out.

The best fixed-window checkpoint reaches **0.879109** raw flat-model validation
macro Entity F1 versus the persisted train-only model's **0.857349**
(**+0.021761**).  Eight rows improve, three are harmed, and 69 tie;
row-bootstrap P(gain) is 0.9242.  All entity types improve: person
0.818182→1.000000, organization 0.876106→0.911504, and location
0.785714→0.790698.  The gains are recognizable boundary/generalization fixes,
including a full organization rather than two fragments, a complete
parenthesized person alias, and removal of stray closing punctuation.

A fairness audit noticed that the historical checkpoint used an older aligner
that dropped one additional partially alignable train row.  Before the pilot
finished, a stricter amendment was frozen (SHA-256
`cf9b6404d0a2c57963157cba96569eab9e1f07d97295428457b51a7dd393ac2c`)
requiring an otherwise identical current-code train-only run.  That control
reaches **0.829645**; the public-label pilot is +0.049464 and improves all three
types relative to it.  Both the original promotion gate and the attribution
gate therefore pass.  Current/hidden FinEval inputs or answers were never used.

An independently preregistered replication then tested the second legal-NER
initialization. Its point estimate is also positive, 0.858098→**0.863633**:
five rows improve, two are harmed, person and organization improve, and
location ties. However, bootstrap P(gain) is only **0.6517**, below the frozen
0.80 replication gate. Its all-label deployment copy is therefore removed from
the candidate despite the favorable mean; the replication summary SHA-256 is
`e50d90ee66460bde49b5421d476c8e3d2043df284e31e2916fe9dda2e03ca6a1`.

A final preregistered test asked whether the strongly verified model helps the
actual three-model ensemble when it replaces only the original legal-base
member. It does not. The frozen ensemble falls from **0.916783 to 0.912617**:
one row improves, one is harmed more severely, and 78 tie; bootstrap P(gain) is
0.3404. The external model correctly joins `ΕΠΙΧΕΙΡΗΣΕΙΣ ΠΙΚΕΡΜΙ Α.Ε.` but
also expands `CNL Capital` incorrectly through `στο Χ.`. The ensemble gate
therefore fails, with summary SHA-256
`9044f315b01a66aa3e881cf0855f16168a8a9c6e0e0762d7dcda13ef118973cf`.

Consequently, the public-label result is retained as a useful standalone
sample-size finding but **the submission lane is demoted even if external data
is permitted**. No post-hoc weight, boundary rule, or row filter is introduced.
The diagnostic manifest is under
`artifacts/submission_portfolio_task1_external_labels_v1/` (SHA-256
`9c7bec1b8470d44fb177bda0bd0c58002a5c859f2f0b5579c8ef6e6be4876b07`)
and records `eligible_for_submission=false`. The policy-independent grouped
primary remains unchanged. The first held-out standalone summary SHA-256 is
`5dda186d6f44bcc6d5a70d19d815f9e2974648b6d9e2a5dbf46185ae4a0072fd`.

## 2026-08-05 — Seed-19 replication and grouped cross-seed promotion gate

The independent GLiNER seed-19 run now covers all 320 rows exactly once across
five entity-linked folds. Standalone OOF proposal F1 peaks at **0.866360**.
More importantly, the unchanged annotation-operation learner sees 424 genuine
disagreements (91 beneficial), selects threshold 0.44 through nested OOF, and
raises the incumbent from **0.819655 to 0.867830** (+0.048175). All five folds
improve, every entity type improves, bootstrap p05 is +0.026569, and
P(gain)=0.9999.

At deployment, the fold-mean view reaches **0.936164** versus 0.920831 with
five improved and zero harmed rows. The stricter fold-vote view reaches
**0.930875**, with four improved and zero harmed rows. Their exact two-view
intersection reaches 0.930395 with three improvements and no harms.

A strict majority across all five seed/deployment views was then required,
with at least three supporting views and at least one view from each proposer
seed. This label-free grouped consensus retains four exact semantic operations:
all are supported across seeds even when one seed's mean and vote views choose
different competing person boundaries. The operation policy reaches
**0.928707** alone; composition with the independently train-gated structural
and union policies reaches **0.933157** (+0.012326), with four improved rows,
zero harms, positive deltas for all three entity types, bootstrap p05
+0.002747, and P(gain)=0.9842.

The proposer-seed stability requirement is therefore satisfied. This is not a
second labeled validation set—the same 80-row validation score has been
inspected repeatedly—so the hidden test remains the decisive generalization
check. The promoted conservative artifact is
`artifacts/task1_conservative_composition_grouped_seed_v1/`; proposer and
consensus evidence are under
`artifacts/task1_gliner_oof_seed19_width16_ensemble/` and
`artifacts/task1_operation_grouped_seed_consensus_v1/`.

## 2026-08-05 — Genuine OOF operation learning and cross-view consensus

A five-fold entity-linked OOF nested-BERT incumbent is now complete over all
320 textual training rows, with zero normalized organization/person leakage.
Its macro entity F1 is **0.819655**. A logistic annotation-operation learner is
trained only on the 418 genuine OOF GLiNER/incumbent disagreements (87
beneficial operations). Its gate is selected through nested OOF prediction at
0.52. OOF F1 rises to **0.859409** (+0.039753), with 38 improved and 11 harmed
rows, four improving folds and one small regression, all three type deltas
positive, bootstrap p05 +0.019416, and P(gain)=0.9991.

With the gate frozen, the full-data GLiNER deployment view raises untouched
80-row validation from **0.920831 to 0.927264** (+0.006433), with three
improvements and no harms. A five-fold majority-vote deployment view
independently reaches **0.928157**, again with three improvements and no harms.
The all-fold mean view remains positive at 0.924540 but creates one false
`SKOPJE` location and fails the promotion gate.

A label-free two-of-three quorum over the semantic operations accepted by the
full-data, fold-mean, and fold-majority views retains four operations and
rejects that mean-only false positive. The consensus policy alone reaches
**0.928707**, with three improved rows, zero harmed rows, positive deltas for
person/organization/location, bootstrap p05 +0.001786, and P(gain)=0.9525.
Composing it with the independently train-gated structural repair and
high-confidence union reaches **0.933157** versus 0.920831 (+0.012326): four
rows improve, none is harmed, every type improves, bootstrap p05 +0.002747,
and P(gain)=0.9842. Public test answers were not used. This first-seed result
was subsequently confirmed through the grouped seed-19 audit above.

Artifacts are `artifacts/task1_multilabel_oof_legal_seed7/`,
`artifacts/task1_oof_annotation_program_seed7/`,
`artifacts/task1_operation_view_consensus_v1/`, and
`artifacts/task1_conservative_composition_consensus_v1/`.

## 2026-08-05 — Boundary-surgery corpus and real-proposer transfer control

A counterfactual boundary corpus now contains **7,945** pairwise choices around
696/706 aligned gold entity occurrences (98.58% alignment) in the 320 training
rows. Alternatives include one-to-three-token expansions/contractions, legal
suffix deletion, leading-article changes, parenthetical aliases, and type
confusions. Every preferred and rejected surface is an exact source substring,
row-level alternatives that are already gold are removed, the existing
entity-linked folds are reused, and validation/public test rows are absent.

A character/context plus intrinsic-geometry classifier trained on four entity
folds ranks gold above synthetic alternatives with **0.973820** pairwise
accuracy, **0.742816** top-1 accuracy among roughly twelve candidates, and
0.865278 MRR. Shortest-span top-1 is only 0.354 and longest-span is zero, so the
model learns more than a trivial length rule.

The necessary real-proposer transfer control is negative. GLiNER exactly
recalls 660/696 aligned gold spans; 172 of those have an overlapping non-gold
GLiNER rival. On these held-fold ambiguity groups, the surgery-trained policy
falls to **0.638009** pairwise and **0.662791** top-1, below GLiNER confidence at
**0.761628** top-1. Synthetic surgery is therefore not a standalone selector.
It remains an auxiliary boundary-ranking loss inside the genuine
incumbent/GLiNER disagreement model. That model still requires the five-fold
OOF nested-BERT incumbent and a second proposer seed before promotion.

Reproduction is in `scripts/build_task1_span_surgery_corpus.py`,
`scripts/evaluate_task1_span_choice_cpu.py`,
`artifacts/task1_span_surgery_v1/`, and
`artifacts/task1_span_choice_cpu_v1/`.

## 2026-08-05 — Five-fold GLiNER OOF result and corrected operation gate

The width-16 GLiNER experiment has completed all five entity-linked folds.  The
320 training rows appear exactly once out of fold, each fold contains 64 rows,
and all 366 normalized organization/person linkage keys remain on one side of
each split.  Validation labels and the public test were excluded while making
the proposals and selecting the threshold.  At the best common OOF threshold
of 0.50, standalone macro Entity F1 is **0.869687** with 641 exact typed hits,
774 predictions, and 706 gold entities.  Fold F1 ranges from 0.811963 to
0.915502.  This is lower and much more variable than the direct-validation
0.906571 result, confirming that entity/source-disjoint evaluation is
essential and that GLiNER should remain a proposer rather than replace the
incumbent.

The conservative union gate was recalculated at the official answer-operation
level.  The first implementation incorrectly counted repeated surface
occurrences as independent calibration events even though the released answer
is a row-level set.  After deduplication, the train-only gate chooses confidence
0.94: 566/608 unique row/entity/type operations are correct (precision
0.930921; Wilson lower bound 0.907940).  Validation then receives six new
entities, four correct and two incorrect.  Four rows improve and none is
harmed; the incumbent rises from **0.920831 to 0.926355** (+0.005524), with
row-bootstrap P(gain)=0.9842 and a positive 5th-percentile delta of 0.001591.
The two incorrect additions are shortened person variants in a row that also
gains the correct full person form, explaining why no row-level score falls.

The structural replace operation is higher in point estimate but has less
support.  Its OOF gate retains 186/197 operations (precision 0.944162; Wilson
lower bound 0.902795) and exactly repairs two fragmented organizations on
validation.  It reaches **0.929998** (+0.009167), with two improved and zero
harmed rows, but P(gain)=0.8716 because only two rows change.  It therefore
remains a diversity candidate rather than the primary.

The corrected simple union is the strongest statistically supported new
candidate, but it is still marked secondary until the incumbent itself has
genuine OOF predictions and a second GLiNER seed tests proposal stability.
Reproduction is in `scripts/summarize_task1_gliner_oof.py`,
`scripts/audit_task1_gliner_oof_calibration.py`, and
`scripts/evaluate_task1_gliner_boundary_repair_oof.py`; summaries are under
`artifacts/task1_gliner_oof_seed7_width16_ensemble/`,
`artifacts/task1_gliner_oof_calibration_width16_mean/`, and
`artifacts/task1_gliner_boundary_repair_oof_mean/`.

## 2026-08-05 — Filing-context retrieval applicability audit

A labels-free Aho–Corasick audit tested whether each short NER input can be
localized inside the collected 29.0-million-character Greek financial corpus,
which would permit issuer metadata and neighboring filing text to resolve
truncated legal names.  After NFKC, case-folding, and removal of punctuation and
whitespace, full-query matches occur on 2/320 train, 0/80 validation, and 5/100
historical test rows; three independent 72-character anchors raise coverage to
3/320, 1/80, and 9/100 respectively.

The route is too sparse to be a primary architecture.  Its exact historical
matches nevertheless show that document-context RAG can be an abstaining
fallback when a unique filing is found.  It should never guess a document for
the other rows.  Reproduction is in
`scripts/audit_task1_corpus_retrieval.py` and
`artifacts/task1_corpus_retrieval_audit_v1/`.

## 2026-08-05 — GLiNER fine-tuning and entity-disjoint calibration

The first real nested-GLiNER fine-tune exposed two issues that the forward-only
smoke could not detect.  GLiNER 0.2.24 used the removed Transformers 5
`Trainer(tokenizer=...)` argument, then silently skipped every optimizer step
because it also assumed the legacy `use_apex` attribute.  The local trainer now
bridges both APIs and a two-step profile verifies nonzero loss and parameter
gradients.  The released `SpanMarkerV0` also stores its enumeration width in a
nested reshape attribute.  Updating processor width alone is invalid; updating
the processor, config, and width-agnostic marker together passes both a
long-span forward and inference test at width 16.

A predeclared two-epoch CPU run at the checkpoint's original width 12 improves
zero-shot standalone validation macro Entity F1 from **0.775551 to 0.906571**.
At threshold 0.10 it retrieves 162/169 exact typed entities.  Its low-threshold
union with the old candidate pool covers 163/169, one fewer than the zero-shot
union, so adaptation greatly improves precision but does not replace the
zero-shot model as the broadest proposer.

A diagnostic confidence-0.90 union with the 0.920831 incumbent reaches
**0.927458**, improving four rows and harming none; row-bootstrap P(gain) is
0.9842.  This is not promoted because the threshold was observed on validation.
Five 64-row folds are now being generated with normalized organization/person
connected components: every fold has 64 rows, all 366 linkage keys stay inside
one fold, and the training budget is exactly two epochs per held fold.  That
OOF run will freeze confidence calibration before validation and will support a
stronger disagreement-conditioned annotation program.  Width 16 is treated as
a controlled ablation, not silently mixed with the width-12 result.  The public
test remains sealed.

## 2026-08-05 — Nested GLiNER proposal and annotation-policy audit

`urchade/gliner_multi-v2.1` was tested on CPU as a genuinely different span
architecture. The model was not allowed to replace the incumbent merely by
union: at its best standalone threshold it reaches only **0.775551** validation
macro Entity F1. At a deliberately low 0.05 threshold, however, its union with
the existing flat/nested candidate files covers **164/169 typed entities
(97.04%)**, versus 154/169 for those files alone. It contributes ten exact
typed gold proposals the old pool lacks. This is the first new model family in
the second research wave to materially raise the Task 1 span oracle.

A character/context annotation-policy logistic model was trained with five
row-held-out folds on 2,268 GLiNER training proposals. Its confidence gate was
frozen with a 0.80 Wilson precision lower bound before validation. It failed on
the relevant disagreement subset: only 1/10 additions was correct and the
incumbent fell from 0.920831 to **0.902056**. This policy is rejected. Global
proposal precision is not conditional precision when the incumbent disagrees.

A narrower structural repair is promising. It considers only organization
proposals with a legal/corporate designator and learns its score threshold from
training only. Threshold 0.74 retains 178/188 correct train proposals
(precision 0.9468; Wilson 95% lower bound 0.9049). It also enforces the train
annotation convention that none of 706 unique entities starts with a
standalone definite article. On validation it repairs two BIO boundary
fractures—`ΕΠΙΧΕΙΡΗΣΕΙΣ ΠΙΚΕΡΜΙ Α.Ε.` and `South Gas Company of Iraq`—and
harms no row, raising **0.920831 to 0.929998**. The row-bootstrap probability
of gain is only 0.8716 because just two rows change, so this remains secondary
rather than promoted. The public test was not consulted.

Reproduction is in `scripts/evaluate_task1_gliner_proposals.py`, the rejected
`artifacts/task1_annotation_policy_v1/`, and the structural candidate
`artifacts/task1_gliner_boundary_repair_v1/`. GPU-ready nested fine-tuning is
in `scripts/train_task1_gliner.py`; its CPU alignment/forward smoke passes.

Task 1 is Greek Financial NER (GRFinNUM/GRFinNER), evaluated with entity F1 over spans and categories. The organizer page currently exposes a GRFinNUM train/validation folder; the downloaded parquet files contain 320/80 examples and labels in the answer column.

## CPU baseline

`scripts/task1_numeric_baseline.py` emits the requested `entity, type` answer format. The first version scored 0.560 validation macro entity F1. After aligning spans to the annotations (numeric core only, percentage symbols retained, date components split, and Greek quantity/money context rules), the score is **0.749102** on the 80-row validation set.

This is a strong diagnostic baseline, not yet a final submission: the hidden test schema and the separate GRFinNER data folder still need confirmation. The main remaining errors are ambiguous plain numbers (identifiers vs quantities), long-range context, and unseen textual entities.

## GPU model experiment

Installed PyTorch 2.13/CUDA 13 and fine-tuned `nlpaueb/bert-base-greek-uncased-v1` as a BIO token classifier over the five GRFinNUM categories. Training used the 320 training rows, a maximum sequence length of 512, AdamW at 3e-5, batch size 8, and early-checkpoint selection on the 80-row validation set. The best of five completed epochs reached **0.941036 validation macro Entity F1**, a +0.174 improvement over the rules.

Checkpoint: `artifacts/task1_transformer_seed19/` (best validation F1 **0.945159**; seed 19). The original seed-7 checkpoint reached 0.941036. XLM-R base was also tested but reached only 0.8101 in the same short run, so it is not preferred. The Greek BERT checkpoint should be the primary Task1 candidate once the organizers confirm the hidden-test format. The rule model remains useful as a fallback/ensemble for out-of-domain numeric patterns.

### Legal-domain transfer

The Hugging Face inventory also contains `amichailidis/greek_legal_bert_v2-finetuned-ner-V3`, a Greek legal-domain BERT encoder with an incompatible 3-class NER head. Reinitializing only its head and fine-tuning on GRFinNUM produced **0.958866** (seed 19) and **0.961023** (seed 7), the current best. The winning checkpoint is `artifacts/task1_legal_bert_seed7/`.

Additional model search found the underlying legal encoder `spyrosbriakos/greek_legal_bert_v2`. Fine-tuning it directly reached **0.961943** (seed 7), the current record, while seed 19 reached 0.9541. `joelniklaus/legal-greek-roberta-base` was tested but remained below 0.04 after eight short epochs, so it is unsuitable here. No dedicated Greek-finance encoder with usable public weights was found in the Hugging Face search; legal-domain pretraining is currently the best domain match.

### Ensemble

A logit-average ensemble of the two strongest legal-BERT checkpoints (`task1_legal_base` and `task1_legal_bert_seed7`) reached **0.963571** validation macro Entity F1, improving over the best single model by 0.0016. Predictions are saved under `artifacts/task1_ensemble/`. This is currently the preferred candidate; synthetic augmentation is lower priority until the hidden-test schema is available.

### High-precision identifier recovery

The equal-weight legal/finance ensemble systematically missed explicit Greek registry identifiers following `Α.Μ. Σ.Ο.Ε.Λ.`, `ΓΕΜΗ`, and `Α.Φ.Μ.`. A conservative postprocessor that adds only these numeric spans as `ΑΛΛΑ` initially raised validation macro Entity F1 from 0.964464 to 0.976548 with no new validation false positives. A subsequent structural fix recognizes dotted/spaced `Γ.Ε.Μ.Η.` variants and removes model-produced numeric fragments only when every occurrence lies inside the recovered full identifier. It changes exactly two validation rows, makes both perfect, and raises the frozen system to **0.983214**.

This is supported by external corpus evidence rather than only the validation set: the final regex matches 850 occurrences across 241 of 298 real ATHEX filing documents, covering 316 unique plausible auditor, tax, and GEMI identifiers. All four validation matches are correct, and the 320-row training set contains none, so the rule was not learned by memorizing training examples. The implementation is `scripts/task1_postprocess.py`.

An independent train-derived calendar rule recovers a day number immediately
before a Greek month name as `ΧΡΟΝΙΚΑ`. It is correct on all 58 training
occurrences and all 19 validation occurrences, with 26,581 external matches
across 282 ATHEX filings. It fixes one missed validation date and raises the
frozen system to **0.985000**. A tempting `Ζώνη 2` rule was rejected because it
has validation-only support and would be overfit.

The frozen submission generator is `scripts/predict_task1_ensemble.py`. It uses equal weights for the three retained checkpoints, applies only the validated identifier and calendar recovery, and emits JSONL records containing the organizer-style multiline `entity, type` answer. A full round trip over the 80 validation rows reproduces **0.985000** macro Entity F1.

An explicitly quarantined diversity profile adds a number following the
standalone designator `Ζώνη` as `ΑΛΛΑ`. It repairs the sole non-corrupt residual
validation error and reaches **0.997500** macro Entity F1; the remaining loss is
an annotation anomaly that labels an organization name as `ΠΟΣΟΣΤΑ`. The rule
is semantically consistent with training annotations for article, paragraph,
and note numbers, and the external filing corpus contains genuine `ζώνη 1` and
`ζώνη 5` usages. Nevertheless, the exact rule family was selected after seeing
the validation miss and has no direct labeled training support. It is therefore
available only as `--zone-designators` and as the portfolio's
`task1_zone_designator_ensemble`, not promoted over the full-data refit.

The safe postprocessor now also guarantees legal designator numbers following
`άρθρο`/`άρ.`, `παράγραφος`/`παρ.`, `Σημείωση`, and an explicitly bounded Greek
law cue (`Ν.`/`Νόμος`/`Κ.Ν.`). This rule was derived from the training ontology,
not from a residual validation error: 38 matched training occurrences and eight
validation occurrences are all annotated `ΑΛΛΑ` after deduplication. The law
regex deliberately requires a standalone cue, preventing the earlier tempting
false match of `των 100%`. It does not change the current validation score
because the ensemble already recognizes those examples, but protects recall
under hidden-domain/model variation with measured zero-error labeled support.

### Financial-domain adaptation

The collected corpus was used for two epochs of masked-language pretraining (8,969 chunks, 256-token windows) starting from `spyrosbriakos/greek_legal_bert_v2`. The adapted encoder alone reached 0.9580 after GRFinNUM fine-tuning, below the legal-BERT ensemble, but its logits were complementary. A three-model average including the adapted checkpoint reached **0.964464**, the current best validation score. The adapted model is in `artifacts/task1_domain_adapted/`, and the fine-tuned version is in `artifacts/task1_domain_finetuned/`.

The expanded 35 MB cleaned corpus produced a new adapted checkpoint (`artifacts/task1_domain_adapted_clean/`) and fine-tuned model (`artifacts/task1_domain_finetuned_clean/`). The fine-tuned model reached 0.9604 alone; adding it to the three-model ensemble did not change the 0.964464 score, so the earlier three-model ensemble remains preferred.

## Expanded Hugging Face search

The search also identified `AI-team-UoA/GreekLegalRoBERTa_v2/v3/v4`, `alexaapo/greek_legal_bert_v1/v2`, and several already-fine-tuned Greek legal NER heads. `AI-team-UoA/GreekLegalRoBERTa_v3` was fine-tuned under the same protocol and reached 0.9130, well below the legal-BERT ensemble. `joelniklaus/legal-greek-roberta-base` was even weaker. No publicly downloadable Greek corporate/finance encoder with stronger evidence than the legal-BERT family was found. We therefore retain the legal-BERT ensemble rather than diluting it with weaker models.

## Next high-return experiments

1. Verify the organizer's exact hidden-test/output schema when it is released.
2. Avoid broad validation-derived rules for the three remaining anomalous/error rows.
3. Keep the three-model ensemble and audited identifier recovery frozen as the primary candidate.

## 2026-08-04 — Corrected annotation alignment and full-data refit

The original trainer incorrectly assumed that answer entities were listed in
text order. They are often grouped by label type, causing 117 of 1,074 training
entities and 7 of 255 validation entities to be silently skipped. The new
one-to-one boundary-aware aligner recovers all 255 validation entities and
1,055 training entities. The remaining 19 annotations belong to four corrupted
rows whose answer entities do not occur anywhere in their text; those rows are
excluded from the full-data refit.

Under the unchanged 320/80 protocol, corrected alignment improves every
selected encoder: legal base 0.961943→0.964622, legal-NER initialization
0.961023→0.964396, and finance-adapted 0.958023→0.965357. All three peak at the
predeclared fifth epoch. Their corrected development ensemble does not beat the
older ensemble because the latter retains useful error diversity, so the old
0.985000 ensemble remains the validated fallback.

For hidden-test inference, the three selected encoders are refitted on all 396
usable released rows for four epochs. This preserves approximately the same
number of optimizer updates as five epochs on 320 rows while incorporating the
80 validation rows. This system is the primary hidden-test candidate, but its
in-sample output is explicitly not reported as a generalization score. Models
and provenance are in `artifacts/task1_full_refit`; invoke the predictor with
`--final-refit`.

After all numeric architecture and refit decisions were frozen, the public
100-row Plutus numeric test was opened once. The development-validated ensemble
scores **0.987596** with 92/100 exact rows, while the predeclared full-data refit
scores **0.992691** with 95/100 exact rows. This independently validates the
full-data refit as the policy-independent numeric primary; public test was not
used to select its architecture, rules, or update budget.

## 2026-08-04 — Recovered GRFinNER textual track

The Plutus benchmark paper and its Apache-2.0 Hugging Face repositories reveal
that Task 1 has two parallel 320/80/100 tracks: numeric GRFinNUM and textual
GRFinNER. The organizer's numeric train and validation files are byte-identical
to `TheFinAI/plutus-finner-numeric`, establishing dataset identity. The omitted
textual train and validation splits were therefore downloaded from
`TheFinAI/plutus-finner-text`. Public test answers exist in both repositories;
they are sealed from architecture and model selection and available only to a
quarantined exact-lookup submission lane pending organizer policy confirmation.

GRFinNER labels people, organizations, and locations. Its flat answer format
contains some nested spans that a single BIO sequence cannot represent; the
trainer deterministically retains the longer outer span. Four training rows
have no alignable gold entity at all. A fifth apparent failure exposed a release
corruption in which the Latin substring `PER` inside company names was replaced
by the Greek label `ΠΡΟΣΩΠΟ` (for example, `PAPERLESS` became
`PAΠΡΟΣΩΠΟLESS`). Alignment now repairs that corruption without changing the
text or the emitted gold spelling.

Fine-tuning `spyrosbriakos/greek_legal_bert_v2` on textual train only reaches
**0.857349 validation macro Entity F1** at epoch 7: person 0.8182,
organization 0.8761, and location 0.7857 micro F1. This is substantially above
the Plutus paper's reported 0.57 textual F1 for Plutus-8B. Fifty-five of 80
validation rows are exact. The remaining errors are dominated by inconsistent
nested-location annotation, corrupted answer strings, punctuation boundaries,
and a few adjacent or alias spans rather than broad entity blindness.

Initializing from `amichailidis/greek_legal_bert_v2-finetuned-ner-V3` reaches
0.858098 on the same split after repairing the embedded-`PER` corruption. Its
errors are complementary: equal logit averaging with the legal-base model
reaches **0.861843**, with 58/80 exact rows versus 55/80 for the first model.
This established a stronger intermediate flat candidate.

The cleaned finance-domain-adapted encoder reaches 0.860027 alone. It adds
useful diversity despite not winning standalone: the equal three-model flat
ensemble reaches **0.867676**. Applying the same independently predicted,
organization-anchored nested-location gate raises the final model-only result
to **0.886010** macro Entity F1, again with 63/80 exact rows.

A validation postprocessor that removes unmatched punctuation, joins person
aliases, and joins adjacent same-type spans reaches 0.873782. It is not a safe
promotion: the training gold contains unmatched punctuation entities and an
adjacent pair that must remain separate. It is retained only as a transparent
diversity hypothesis. A train-derived nested-location gazetteer similarly
raises validation to 0.87098 but has only 0.60 leave-one-row-out precision on
training, confirming inconsistent annotation; it is quarantined as well.

A more principled overlapping-span model replaces the single seven-way BIO
head with three independent BIO heads over one shared legal-BERT encoder. It
can represent an organization and a location on the same tokens, while retaining
only the longer span for impossible same-type nesting. Standalone validation is
**0.866195** (person 0.9524, organization 0.8631, location 0.8602), with 59/80
exact rows. Using the flat ensemble for people/organizations/non-nested
locations and adding only multi-head locations that are strict substrings of a
predicted organization reaches 0.882676 with the two-model flat base and
**0.886010** with the finance-diversified three-model base, with 63/80 exact
rows. The gate
adds eight nested locations; seven are correct. Unlike a surface gazetteer, the
location decision comes from an independently trained label head and is
anchored to an already predicted organization. This is the strongest validation
candidate at this stage, although the post-validation choice is retained
alongside the ungated 0.866195 model and 0.867676 flat ensemble for diversity.

The architecture was then frozen and evaluated once on the previously sealed
100-row public test. This reverses the validation ranking: the three-model flat
ensemble scores **0.843172** with 72/100 exact rows, whereas the nested gate
scores 0.831447 with 68/100 exact rows. The flat held-out type F1 values are not
used to redesign the system; the decisive result is simply that the apparently
strong nested validation gain does not generalize. The nested profile is
therefore demoted to diversity, and the plain finance-diversified flat ensemble
is the honest policy-independent primary. Public test remained excluded from
all training and architecture/rule selection before this single evaluation.

The two predeclared flat architectures were then refitted for six epochs on all
395 alignable textual train/validation rows and combined with the frozen
finance-adapted diversity model. On the same held-out test this all-data flat
ensemble reaches **0.868736** with 76/100 exact rows, a clear improvement over
the 0.843172 train-only ensemble. Because refitting strategy and update budget
were fixed before the test was opened, this is the policy-independent primary
rather than a test-selected architecture. The four fully corrupt training rows
and one fully corrupt validation row remain excluded; partially alignable rows
retain only their observable annotations.

## External Greek legal NER transfer audit

The public `joelniklaus/greek_legal_ner` corpus supplies 17,699 Greek Gazette
training sentences under CC-BY-NC-SA-4.0. A predeclared auxiliary pass used all
6,530 rows containing compatible entities plus 1,500 negative rows. `ORG`,
`PERSON`, and `GPE`/location labels were mapped into the GRFinNER ontology;
facilities, legislation references, and public documents were deliberately
mapped to outside. External validation/test and all Plutus labels were excluded
from auxiliary training. Provenance and hashes are in
`data/external/greek_legal_ner/README.md`.

After Plutus-train fine-tuning, the transferred model reaches only **0.843661**
validation macro Entity F1, below the finance-adapted model (0.860027) and the
three-model flat ensemble (0.867676). Adding its logits to that ensemble yields
only 0.868510 (+0.000833) and brings a restrictive share-alike/non-commercial
license plus an unresolved external-data policy dependency. The branch is not
promoted. It is retained as a reproducible, explicitly policy-gated diversity
artifact and is not evaluated again on the already opened public test.

Exact prompt-aware lookup reproduces all 100 public numeric test answers and all
100 public textual test answers. This demonstrates full public overlap, not
hidden generalization, and `scripts/predict_task1_lookup_safe.py` keeps the lane
separate with a per-row overlap audit. The production requirement is now a
mixed router that sends numeric and textual packets to their respective robust
models while retaining original row order.

That production requirement was tested on a deterministic 200-row fixture that
alternates the 100 public numeric prompts and 100 public textual prompts. The
full-refit mixed router preserves every row and exactly reproduces the separate
track results: **0.992691** numeric, **0.868736** textual, and **0.930713**
combined macro Entity F1, with 171/200 exact rows. The separate lookup lane
matches 200/200, but remains policy-gated because this measures public-answer
reuse rather than generalization.

## Released-answer encoding recovery (4 August 2026)

The earlier alignment audit identified a dataset construction bug but repaired
it only on the training side. Published textual answers replace uppercase Latin
`PER` inside organization names with the Greek PERSON label word `ΠΡΟΣΩΠΟ`:
examples include `SUPERVISOR`, `EXPERT`, `PAPERLESS`, and `PROPERTY`. Thus a
correctly extracted surface form was still scored as wrong at inference time.

A train-derived output encoder now applies the release convention only to
uppercase `PER` inside predicted organizations. The gate is narrow: all three
affected training rows follow the corruption, while mixed/lowercase names such
as `PricewaterhouseCoopers` and `Grivalia Properties` remain unchanged. It
transfers to four validation rows containing two different organization names
and raises the flat three-model validation score from 0.867676 to **0.902498**
(+0.034821). The nested diversity profile rises from 0.886010 to 0.920831 but
remains secondary because its underlying nested-span rule previously failed
held-out evaluation.

The encoder was frozen before one confirmation against the existing public-test
predictions. It changes one row, correctly converting two mentions of `NOVAL
PROPERTY` to the published `NOVAL PROΠΡΟΣΩΠΟTY`, and raises the full-refit flat
score from 0.868736 to **0.875402** without changing model weights or reducing
exact rows (76/100). This independently supports using the encoded flat
full-refit as the policy-independent primary.

The same train-only audit found that released entity strings collapse repeated
source whitespace. No gold entity in any released split retains doubled
whitespace, while a training organization contains multiple source spaces but
a single-space gold form. Collapsing whitespace inside predicted entities is
neutral on validation and independently fixes one public-test organization.
Together, whitespace normalization and the uppercase-`PER` encoder raise the
full-refit flat test score to **0.885402**, with 77/100 exact rows. The
interleaved numeric/textual fixture consequently reaches **0.939047** combined
macro Entity F1 and 172/200 exact rows.

A separate standards-correct BIO decoder that closed a span on every `O` token
was also tested. It left the three-model validation score unchanged at
0.902498 but reduced the frozen full-refit public-test result to 0.868428.
Because it supplied no development gain and failed the independent check, that
decoder change was rejected; the established decoder remains in production.

## Exact high-support nested location (4 August 2026)

The generic nested-location model remains rejected, but a much narrower
train-derived convention is stable: in five independent training rows,
`Αθηνών` inside the organization `Χρηματιστήριο Αθηνών` is also annotated as a
location, with no counterexample. The postprocessor now recovers only this
exact organization phrase (including the visually equivalent Greek mu forms),
and does not generalize to arbitrary organizations containing Athens.

On validation it changes four rows, all four correctly, raising the flat
three-model textual score from 0.902498 to **0.912498** and exact rows from
60/80 to 64/80. The already-opened public test was deliberately not consulted
to select or score this addition. On the frozen mixed fixture the rule changes
only one substantive textual answer; all 200 rows remain aligned and valid.
The rebuilt policy-independent artifact is
`artifacts/submission_portfolio_task1_athens_nested_v2/task1_mixed_full_refit.jsonl`
(SHA-256 `f3782efe5fab1fbfc88e9e06950f9dd3137eab1cd53bbe07a4480500ea94931e`).

Two additional nested country genitives pass the same train-only convention
audit. Every occurrence of `Ελλάδας` (six annotations across five training
rows) and `Ρουμανίας` (two across two rows) is labeled as a location, including
when it is nested in an organization name. The exact postprocessor adds either
location only when the corresponding surface occurs inside a predicted
organization; it does not use a general gazetteer or add country adjectives.
On validation it changes exactly two rows, both correctly, raising flat
three-model macro Entity F1 from 0.912498 to **0.916783** and exact rows from
64/80 to 66/80. The promotion gate is based on train consistency and validation;
the public test was not re-scored for this change. The rebuilt mixed fixture is
`artifacts/submission_portfolio_task1_nested_countries_v3/task1_mixed_full_refit.jsonl`.
It remains byte-identical on that fixture (SHA-256
`f3782efe5fab1fbfc88e9e06950f9dd3137eab1cd53bbe07a4480500ea94931e`), as
neither exact organization form occurs in a model prediction there.

## ATHEX provenance recovery and quality audit (4 August 2026)

The legacy collector used issuer plus page position as its filename. Because
new filings reorder issuer pages, repeated crawls could overwrite earlier
reports even though the append-only provenance ledger retained 2,987 unique
successful URLs. Collection now uses the first 16 hexadecimal characters of
the URL SHA-256 in each filename and records both URL and content hashes.

All 283 Euronext-domain ledger URLs were replayed with strict Greek-financial
content checks and retry handling. This recovered 269 stable reports. A
50-document sample from the retired ATHEX domain recovered none (48 explicit
quality rejects and two request errors), so stale legacy links are not treated
as available data. The audit also moved 33 large 404 pages and 15 English,
scanned, or non-financial extractions into recoverable quarantine directories.

After conservative line cleaning and global paragraph deduplication, the new
quality-controlled corpus contains 273 documents, 142 ATHEX issuers, and
29,708,220 characters. It is smaller than the previous noisy build because
invalid payloads and duplicates were removed. Given that larger prior MLM
builds did not improve downstream GRFinNER, this audit improves provenance and
quality but does not by itself justify another GPU adaptation run; a retrain
requires a meaningful increase in unique Greek financial text or a different
adaptation objective.

## Residual textual-error convention audit (4 August 2026)

The 14 non-exact validation rows remaining after the v3 nested-country rule
were re-audited against training annotations before proposing further rules.
Several tempting fixes are contradicted by training data. `Χ.Α.` occurs
unlabeled in training but is labeled as an organization in validation. Generic
`Όμιλος/Όμιλο` prefixes are inconsistently included in organization spans.
`Ελλάδος` is labeled as a nested location in one `Τράπεζας της Ελλάδος`
training row but omitted inside three other bank-name rows. Splitting or joining
multi-token locations and organizations is likewise inconsistent across the
remaining examples.

The adjectival forms `ελληνικού` and `Ελληνικό` each have only one positive
training occurrence. Although matching rules would repair one validation row
apiece, that support is too weak for the repeated-example convention gate used
by v3. Other errors include absent/corrupt gold text, foreign-name tokenization,
and mutually incompatible span boundaries. No new Task 1 rule is promoted from
this audit; the 0.916783 validation primary remains unchanged.

## Neural operation-ranker fold-0 result (5 August 2026)

The bounded neural follow-up was preregistered before evaluation in
`docs/task1_neural_operation_ranker_preregistration.md`. A pinned Greek legal
BERT cross-encoder first received one entity-fold-pure epoch over 7,945
weighted span-surgery comparisons, then six epochs over the 418 genuine OOF
GLiNER/incumbent operations (87 beneficial). Both stages exclude every example
from the held entity-linked fold. Candidate boundaries use the existing
`[MASK]` token; synthetic preferred and rejected candidates have the same
descriptor, preventing a role-word shortcut. The decoder is the unchanged
conflict resolver at a fixed 0.5 probability threshold.

The real seed-7 fold-0 run completed 400 surgery and 132 operation updates with
finite losses and gradients. Operation discrimination is above chance
(average precision 0.540156, ROC-AUC 0.638298), but that does not translate to
a safe annotation policy. The 64-row incumbent scores 0.735844 row-macro
Entity F1 and the neural policy scores **0.729777** (delta -0.006067): five rows
improve, six are harmed, and 13 operations are accepted.

The advance-frozen gate required at least the existing linear operation
policy's +0.077883 fold-0 gain and no more than two harms. The neural model
fails decisively, so folds 1--4, validation, the public test, seed 19, and the
no-surgery ablation are not run. No threshold is tuned on the held fold. The
artifact is `artifacts/task1_neural_operation_ranker_fold0_seed7_v1/` and the
promoted grouped-consensus linear policy remains unchanged. The negative result
supports the paper's central finding: a semantically richer scorer is not
automatically safer than a sparse, calibrated, abstaining operation model.

## Reproducible deployment lanes and independent public-test confirmation (5 August 2026)

The promoted policy is now exposed through one fail-closed orchestrator,
`scripts/build_task1_enhanced_submission.py`. It reads the packet schema first
and, when a released answer column is present, constructs an answer-stripped
effective input before invoking any predictor. The default mode uses the
full-data refits intended for a genuinely hidden deployment packet. A separate
`--validation-safe-replay` mode uses only the train-only incumbents and is
restricted to the pinned 80-row validation input; the two modes must not be
confused when reporting scores.

The validation-safe replay exactly reproduced the promoted single-seed grouped
primary (SHA-256
`837a7a9dcda8cffba4e28658c6f582642053d44e6c838db00bd188d88d000f69`)
at **0.933157** macro Entity F1 and the cross-seed diversity output (SHA-256
`ec3d41a97d760f602791a720d4072b3c5837a6379cd67b23f8c50bb013007122`)
at **0.932264**. The primary retains the preregistered four improvements and
zero harms. Its manifest, candidate files, and input fingerprint are under
`artifacts/submission_portfolio_task1_enhanced_replay_v2/` and pass the
standalone manifest verifier.

After the policy and primary/diversity roles were frozen, the default full-refit
orchestrator was run once on the separate 100-row released textual test. The
answer column was stripped before every prediction stage. The current mixed
full-refit baseline scores **0.882069**. The single-seed grouped primary scores
**0.894236** (+0.012167), with eight improved and six harmed rows; its paired
row-bootstrap delta has p05 -0.007190, median +0.011280, p95 +0.034830, and
P(gain)=0.82923. It improves organization F1 from 0.904412 to 0.927536 and
location F1 from 0.706897 to 0.741935 while person F1 remains 1.0. The exact
full-refit output hash is
`8a084a3d9e537bbb9ca2a9d973c5d4f45a29b1759882481bd4eabe6486e93799`.

The two-seed candidate reaches **0.898331** (+0.016263), with nine improved and
six harmed rows; bootstrap p05 is -0.005540, median +0.015420, p95 +0.041060,
and P(gain)=0.88102. Its output hash is
`164dfca10253ae49849249655eb8489a5d4c70439f094e450a29d2edfc66660d`.
Although its point estimate is higher on this one already-opened set, changing
the primary after observing that result would be post-hoc selection. It remains
the explicitly named diversity lane. The preregistered single-seed grouped
policy remains the expected-score primary, and the untouched mixed refit
remains a conservative upload lane if multiple submissions are permitted.

The current baseline is lower than the previously logged 0.885402 flat result
because it also contains the later train/validation-gated exact nested
`Αθηνών` convention. That convention was frozen without consulting public-test
answers but harms one released-test row; the whitespace repair still improves
another. This is evidence that the public-test gain is positive but not
uniform, not a reason to reverse a development-gated rule after the fact. The
audited files are under
`artifacts/submission_portfolio_task1_enhanced_public_test_v1/`. Public-answer
lookup remains quarantined and requires an explicit organizer policy decision.

## Inflected calendar-day normalization audit (5 August 2026)

One residual released-public numeric error contains both the model's inflected
surface `8ης, ΧΡΟΝΙΚΑ` and the postprocessor's recovered numeric core
`8, ΧΡΟΝΙΚΑ`; the released annotation retains only the core. Because this
hypothesis arose after inspecting that public error, the audit is explicitly
retrospective and its public score is not treated as blind hidden-test evidence.
The frozen protocol is
`docs/task1_calendar_suffix_dedup_audit_protocol.md` (SHA-256
`3a1ace945f37a3a1433ea9c83cc7f59cd0558fa4d43a599bb19cf29eccad4317`).

Independent annotation evidence is unanimous. The exact inflected day plus
Greek-month pattern occurs 17 times across 16 training rows and six times
across six validation rows. All 23 occurrences label the numeric core as
`ΧΡΟΝΙΚΑ`; none labels the inflected surface. The implementation removes a
same-typed suffix only in exact month syntax, only when every occurrence of the
surface in the input is such a calendar expression, and deterministically
retains or adds the numeric core. It abstains on other labels, suffix-like text
outside month syntax, and ambiguous repeated surfaces.

Applying only this normalization to the frozen 80-row validation output changes
zero rows and leaves macro Entity F1 exactly **0.985000**. The targeted suite
passes 16/16 tests, including an explicit idempotence assertion. Only after
those gates passed was the already-opened public
diagnostic run: it changes one of 100 rows, improves that row, harms none, and
moves macro Entity F1 from **0.9926910867 to 0.9941196581**. The rule is promoted
as an input-only deployment invariant; no postal-code, accounting-standard, or
other tempting public-error rule is added. Reproduction and exact changed-row
evidence are under `artifacts/task1_calendar_suffix_dedup_v1/`.

## Entity-aware domain adaptation fold-0 gate (8 August 2026)

The advance-frozen experiment in
`docs/task1_entity_aware_adaptation_preregistration.md` tested whether continued
pretraining on leakage-safe Greek financial text improves textual entity
recognition. The 273-document corpus builder excluded every normalized entity
surface appearing in held fold 0 and rejected chunks containing an exact
train/validation-text substring. It retained 885 chunks and 1,438 silver spans
from 198 documents and 118 issuers. Coverage was 779 organization, 658
location, and one person span; the treatment is therefore primarily an
organization/location experiment. The final corpus SHA-256 is
`fea498736196338a4b3ea223a4244f57ef884e42d9a0429ec5da6051603fc108`.

Matched random-token MLM and whole-entity MLM used the same pinned Greek Legal
BERT revision, corpus, seed 7, batch size 16, and exactly 256 FP32 updates. Their
mean adaptation losses were 1.652338 and 1.664129. Each adapted encoder and the
untouched base encoder was then trained with the same fixed seven-epoch fold-0
token-classification recipe; held labels were never used for checkpoint or
hyperparameter selection.

The untouched base scores **0.806392** macro Entity F1, matched random MLM
scores **0.755717**, and whole-entity MLM scores **0.758021**. Thus the entity
treatment gains only +0.002305 over random MLM and loses -0.048371 to the
stronger untouched control. Against that control, its per-type deltas are
-0.157895 person, -0.019377 organization, and -0.049751 location; it makes
eight rescues and ten harms. Every frozen efficacy condition fails, while all
corpus and provenance checks pass. The formal decision is **stop**: folds 1--4,
additional seeds, post-hoc masking changes, and validation/public/hidden-test
evaluation are not run. The result is in
`artifacts/task1_entity_aware_fold0_v1/evaluation.json`; the full repository
suite passes 348 tests.
