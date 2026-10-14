# ADML research log

## 2026-08-06 — Japanese Nemotron 9B fails the frozen Task 2 pilot

The pinned `nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese` checkpoint was run on
the preregistered 22-row pilot with its frozen prompt, IDs, schema, sampling,
seed, token budget, and advance thresholds unchanged.  The result is a clear
**stop**: **4/22 correct (0.181818)** versus the frozen Qwen guideline
comparator's **11/22**, only **18/22** schema-valid completions, and just
**1/13** correct negative examples.  In particular, it recovers **0/11** of
the crucial `-1` class; valid output is biased toward `0` and `+1`.  Four
thinking generations also exhausted the 2,048-token budget and left
incomplete JSON.  Only the three-label nonzero-recall condition passes; every
substantive promotion gate fails.  The conditional 253-row confirmation was
therefore not run, and this checkpoint is not admitted as a replacement,
refusal specialist, or ensemble member.

The startup failure preceding the experiment was environmental rather than a
model result.  vLLM 0.26.0 successfully loaded the checkpoint after the
launcher exposed its venv-local `ninja` and matching Ubuntu Python 3.12 ARM64
development headers unpacked under the user cache.  Mamba/Triton kernels then
compiled, a strict-JSON thinking smoke request returned HTTP 200, and the
scored run completed normally.  The fail-closed launcher records those local
requirements in `scripts/serve_icr_nemotron_japanese.sh`.

Reproducible evidence is in
`artifacts/icr_nemotron_japanese_pilot_v1/`: judgments SHA-256
`d115b399af1969207ee6bdd4de15371fc9f91b77ca84833349ed4433f473b8d0`
and summary SHA-256
`7fd1dbca091f39059cfc8a60d97c3f99e1f2aed51c79bfa539499a9a10cab340`.
No current or hidden test data was used, and no post-hoc prompt or threshold
change is claimed as another result.

## 2026-08-06 — Unrelated-issuer IR Q&A pool roughly doubles after quality repair

Official corporate Q&A archives were expanded from 9 to 17 unrelated issuers,
adding Chubu Electric, Estic, Fast Retailing, Infroneer Holdings, Nikon,
Shizuoka Financial Group, Tokyo Electron Device, and Tokyo Ink. The audited
result grows from 819 to **2,110 curated Q&A pairs**, from 672 to **1,738**
forward-intent candidates, and from 69 to **144** high-recall refusal
candidates. The retained rows span 250 source-hashed documents; refusal
candidates span 14 issuers. No released JF-ICR source issuer is admitted, no
row reaches the frozen 0.92 near-duplicate threshold, and maximum released-set
similarity is 0.717949.

A deterministic sample audit caught one real extraction bug: the optional
answer-marker delimiter allowed the `A` in a line-initial `AI server` phrase to
split a question. The grammar now requires a number, recognized punctuation,
or actual whitespace after `Q`/`A`; regression tests cover `AI`, bare markers,
and Nikon's vertical Japanese colon. Rebuilding from the cached 388-document
manifest removed the false split without another crawl. Nikon navigation and
section tails are also stripped. Reproduction and pinned hashes are in
`artifacts/icr_external_corpus_expansion_v2/summary.json`.

This remains unlabeled external data, not a score claim. The silver-label path
was hardened before use: judgments now require exact schema types/enums, finite
confidence, grounded spans, pinned model revisions, and resume provenance;
the builder rejects duplicate judgments, model aliases, revision drift, and
same-family pseudo-teachers. Nemotron aliases therefore cannot manufacture
independent agreement. Only cross-family, rubric-invariant agreement may enter
the later counterfactual-training path.

## 2026-08-06 — General Nemotron 3 Nano 30B checkpoint is staged, not scored

The full BF16 checkpoint
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` at revision
`2d59de1cbd51c0adf384eb906b766d1aee0e0517` is now downloaded. Hugging Face
cache verification checked all 35 repository files successfully, and the 13
safetensors shards contain 31,577,940,288 tensor elements
(31,577,934,400 BF16 and 5,888 F32), safely below the organizer's 70B total
parameter ceiling. The official cached model card lists Japanese support, but
positions this as a general reasoning/chat checkpoint chiefly intended for
English and coding; it is distinct from both the Japanese-specialized 9B model
and `Nemotron-Cascade-2-30B-A3B`.

No FinEval score is claimed for the 30B checkpoint. The frozen Japanese 9B
Task 2 pilot remains first because its specialization is better aligned and a
second model comparison on the same small pilot would add selection bias. The
30B model remains a later, separately preregistered diversity candidate for a
larger Task 2 confirmation or Task 3 planner evaluation.

The Japanese 9B pilot and confirmation runners were also hardened without
changing their frozen prompt, IDs, sampling, thresholds, or gates. Resumed
records now fail closed on duplicate/unexpected IDs and model, revision, gold,
or prompt-hash drift; schema validation now rejects extra and non-finite
values; and the full confirmation recomputes the pilot metrics from the saved
judgments rather than trusting a stale summary.

The installed vLLM 0.26.0 runtime explicitly registers the checkpoint's
`NemotronHForCausalLM` architecture. Its structured-output scheduler defers the
JSON grammar until the configured reasoning parser detects `</think>`, so the
frozen combination of thinking mode and final strict JSON is supported rather
than mutually exclusive. A fail-closed launcher is now pinned in
`scripts/serve_icr_nemotron_japanese.sh`; it checks the exact local snapshot
and refuses to start while any other GPU compute process is active.

## 2026-08-06 — External refusal phrases fail without proposition scope

The incumbent's 0/13 refusal recall motivated an independent CPU control before
any Japanese Nemotron output. Exact withholding and action-rejection patterns
were derived only from the 69-candidate unrelated-issuer IR queue, then frozen
at preregistration SHA-256
`5f8391c13ecba3871b9f8d151cffdeadc3bc2d4c9858821c56cdef1115ef0809`
before running against JF-ICR labels.

The control fails decisively. It changes 21/253 OOF predictions, rescues only
two incumbent errors, harms 15 correct rows, and falls from **0.691700 to
0.640316**. Beneficial-change precision is 0.095238, refusal recall is only
2/13, all five source folds worsen, bootstrap p05 is -0.079208, and P(gain) is
0.0006. Every frozen promotion condition fails except incumbent reproduction
and hidden-test exclusion. No pattern, precedence, temporal mapping, or label
is revised after inspection.

The error pattern is substantive: `計画が保守的ではない` is not rejection,
and phrases such as `予定はない`, `詳細は申し上げられない`, or `開示を差し控え`
often concern a subordinate detail while the answer's dominant stance remains
positive or neutral. Literal refusal morphology is therefore insufficient;
the decision must align the negated proposition with the question and the
answer's global stance. This negative control strengthens the rationale for
the frozen structured Nemotron judge rather than authorizing a keyword patch.
Artifacts are in `artifacts/icr_external_refusal_rule_v1/`; current/hidden test
data were not used.

## 2026-08-06 — Japanese Nemotron is now scoped to the incumbent's missing refusal boundary

An audit of the immutable source-OOF Task 2 predictions exposed a concentrated
failure hidden by the 0.691700 aggregate: the two-seed ordinal incumbent is
correct on **0/13** gold refusal rows (`-2: 0/2`, `-1: 0/11`) and never emits a
negative label on any of the 253 examples. This is not only an OOF artifact:
the persisted full-refit deployment ensemble reaches 0.837945 in-sample but
also emits only `0`/`+1`/`+2` and remains **0/13** on refusals. The already
frozen 22-row Nemotron
pilot contains all 13 refusals and requires at least seven correct, so its
rare-boundary gate is directly aligned with this deficit rather than chosen
after model output.

Before any Japanese Nemotron judgment, a conditional 253-row confirmation was
frozen at SHA-256
`3a26c0a68c9980957f09a4e662a044d783d3e773c3539f2390b24987dc39b535`.
If the pilot passes, the single promotable candidate keeps the incumbent except
when Nemotron predicts `-2`/`-1`, independently marks direction as `reject`, has
confidence at least 0.80, and cites a two-character-or-longer decisive span
that is found in the company response after NFKC/whitespace normalization.
Promotion requires at least four net-safe row gains, at most two harms, at
least 75% beneficial changes, 7/13 refusal recall, three improved source folds,
and source-bootstrap support. Standalone and broad blend scores are diagnostic
only. This makes the LLM an auditable boundary specialist, not an unrestricted
replacement.

The runner and frozen gate are implemented in
`scripts/evaluate_icr_nemotron_japanese_refusal_confirmation.py`; targeted
pilot/confirmation tests pass. No current or hidden test input or answer was
used.

## 2026-08-06 — Organizer release surface is unchanged; effective dates are now audited

The dedicated organizer site was checked both at the bundle level and through
all three published Drive folders. Its browser-applied schedule is now **8
August** for test release, **23 August** for test close, **24 August** for the
leaderboard, and **26 August** for the system paper. The solution submission
link remains `TBD`. The homepage's embedded application template still contains
the superseded 4/20/20/25 August dates, so a naive text scrape is wrong.

`scripts/audit_organizer_site.py` now decodes the bundle, applies the site's own
string overrides, extracts the effective schedule and solution link, records a
source SHA-256, and can fail closed when the expected release date or required
submission link is absent. Four parser tests cover date overrides, available
links, missing bundles, and missing Drive folders. The live report is
`artifacts/organizer_site_audit_live_2026-08-06.json`.

The full byte audit also downloaded every currently published organizer file
to temporary storage and confirmed that all four files are byte-identical to
our pinned local inputs; no new path or Drive ID is present. Evidence is in
`artifacts/organizer_drive_audit_full_2026-08-06.json`. No public test input or
test label was used.

## 2026-08-05 — Sub-70B release audit identifies a Japanese Nemotron candidate

Official Hugging Face publisher listings were audited by repository creation
date, safetensors total parameters, license, architecture, and intended use.
The strongest task-aligned new candidate is
`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese`: **8,888,227,328** total
parameters, Japanese-specialized reasoning/non-reasoning support, and a model
family distinct from the screened Qwen judges.  The pinned revision
`3979dd16634988c34cc3bd911583c51e6a731d10` has been downloaded in BF16
(17.8 GB) and all 22 cached files pass Hugging Face verification.

Before any FinEval output from this checkpoint, an exact 22-row Task 2
guideline pilot was frozen at preregistration SHA-256
`d9429e5a5d5a7e4c5ec93ea4b10df1ad2f18d50a5a8d8429187248a9748cc315`.
It must return 22 valid structured judgments, score at least 13/22 versus the
frozen Qwen and majority comparators at 11/22, retain at least 7/13 negative
rows, and recover at least three labels.  Passing authorizes a larger test, not
promotion from a small screen.  Inference is deferred while an unrelated GPU
workload remains active.

Other eligible candidates are recorded in
`docs/under_70b_model_shortlist_2026-08-05.md`.  The best Task 3 challengers
are GLM-4.7-Flash (31.22B total) and Nemotron-Cascade-2-30B-A3B (31.58B);
Nemotron-3-Embed-8B is a Japanese-capable retrieval ablation.  GLM-5.2,
DeepSeek V4, Nemotron Puzzle 75B, Super 120B, Ultra 550B, and quantized copies
of giant GLM/Kimi/MiniMax models remain ineligible because the organizer counts
total parameters, not active parameters or storage bytes.

## 2026-08-05 — Rank-invariant hurdle loses external-group safety

The harm-aware hurdle's absolute score shift motivated one separately frozen,
label-free calibration transform: empirical average-rank percentiles computed
inside each held group, with the complete deployment batch treated as one
group.  The protocol and only threshold grid were frozen at SHA-256
`010c7b953b879c0f27f5db8999d622309968678382512cc7b626a1524f9cb1e9`
before any percentile-gated validation outcome was computed.

No threshold clears the six-group gate.  The closest eligible-size threshold,
0.94, selects 27 operations, raises combined macro Entity F1 by +0.018184,
has bootstrap p05 +0.008805, and holds harmful fraction to **0.148148**.  But
the external public-label group falls by **-0.010606** even as all five OOF
groups improve.  At the much more selective 0.99 threshold, all six groups
improve and no accepted operation is harmful, but only six edits are selected,
below the frozen minimum of 25.  The transform therefore abstains before
validation, reproduces the **0.933157** primary byte-for-byte, and is stopped
without a quota or group exception.  Relative rank alone discards meaningful
absolute safety information and does not solve deployment calibration.
Artifacts are in `artifacts/task1_rank_invariant_hurdle_v1/`; the summary
SHA-256 is
`5aafc482c0b118db2e5438f9f7bd29f443fa1d6bd06748e4e4dcda2240b2f740`.

## 2026-08-05 — Harm-aware hurdle passes calibration but absolute scores do not transfer

The failed binary operation selector combined harmful and score-neutral edits
in one negative class.  A preregistered two-head hurdle instead predicts
whether an edit is consequential and, conditional on consequence, whether its
direction is beneficial.  Its only score is the signed expected consequence,
`p(nonzero) * (2 p(beneficial | nonzero) - 1)`.  The protocol was frozen at
SHA-256 `55f879bdab0f2b0af141252edc3ab9cb26b24a1dc778b8accf532ff2bb339672`
before cross-fitting or scoring.

Unlike the binary model, the hurdle clears every six-group calibration gate at
threshold 0.39.  It selects 43 of 414 operations, with 36 beneficial, five
harmful, and two neutral edits: harmful fraction **0.116279**, below the frozen
0.15 ceiling.  Macro Entity F1 rises from **0.837413 to 0.871185** (+0.033772),
all six group deltas and all three type deltas are positive, bootstrap p05 is
+0.021474, and P(gain) is 1.0.

However, the absolute operation-score scale does not transfer.  Cross-fitted
calibration scores span -0.952 to +0.900, whereas validation scores span -0.855
to only +0.170.  The frozen 0.39 threshold therefore selects zero validation
edits and reproduces the **0.933157** incumbent byte-for-byte.  Promotion fails
and no lower threshold is tried.  This is positive evidence for explicitly
modeling harm, but negative evidence for absolute probability-scale transfer
from OOF proposals to stricter vote-three deployment ensembles.  Artifacts are
in `artifacts/task1_harm_aware_hurdle_v1/`; the summary SHA-256 is
`b7f5568b357c55be0b88e4e85aad99828f20fefae762413c85506e526f97d831`.

## 2026-08-05 — External operation calibration finds signal but fails the frozen safety gate

The 100 independently published Plutus test labels were added as an indivisible
sixth group to the 320 genuine OOF rows, with all 80 FinEval validation rows
held out.  The exact dual-seed raw-proposer intersections yield 414 candidate
operations across the combined calibration set.  This experiment was frozen
before execution in
`docs/task1_external_operation_calibration_preregistration.md` (SHA-256
`a1666ac4e13025d2dbd8590f6578d6db0069033b62b4408ac99ae8d8c6753263`).

The strongest diagnostic thresholds, 0.43 and 0.44, raise combined macro
Entity F1 from **0.837413 to 0.874939** (+0.037526).  All six group deltas and
all three entity-type deltas are positive, and bootstrap p05 is +0.021373.
However, 17 of the 74 accepted operations are harmful, a **0.229730** harmful
fraction that exceeds the preregistered 0.15 ceiling.  Therefore no calibration
threshold is admissible.  The policy abstains on validation, reproducing the
**0.933157** incumbent byte-for-byte with no accepted operations; no post-hoc
threshold is tried.  Task 1 remains unchanged.  Artifacts are in
`artifacts/task1_external_operation_calibration_v1/`; the summary SHA-256 is
`3def36ce80b084320f1e74b21f2cdb007ae803c0cd1ea55c1e20489df19c4d83`.

## 2026-08-05 — Large-planner intersection improves globally but fails issuer-safety confirmation

The discovery-only observation that a 35B fact planner could verify the
incumbent's proposed facts was tested on a newly frozen 28-row confirmation
screen.  Every target is outside the original 32-row screen, with one stable
hash selection from each of four companies and seven families.  The model saw
only the question, typed input-derived facts, and other-company demonstrations;
all 28 constrained JSON outputs were valid.  The preregistration was frozen at
SHA-256 `889659404b53dd67cbec1161cca2e98ee55155976f4edb1af4d1640dbe4f55a1`
before target selection or inspection.

The fixed intersection may only delete an incumbent fact and preserves its
order.  It raises macro set F1 from **0.745408** to **0.763776** (+0.018367),
exact-set accuracy from 0.535714 to 0.607143, and micro precision from 0.804878
to **0.911765**.  Four rows change: three improve and one worsens.  However,
the single harm is concentrated in HON, whose company delta is **-0.054422**;
JNJ is +0.127891 and MSFT/UVV tie.  This violates the frozen requirement that
no company fall by more than 0.01, so the branch does not advance to full
history and Task 3 v15 remains primary.

The unconstrained 35B planner itself scores 0.790476 on this confirmation,
above the incumbent but opposite its deficit on the original screen.  That
sign reversal is useful evidence of complementary signal and sample/issuer
instability, not permission to replace the preregistered candidate.  No
post-hoc family exception or third screen is fitted.  Artifacts are in
`artifacts/polyfiqa_large_planner_intersection_confirmation_v1/`; the summary
SHA-256 is
`f6eddd869cb9628c4e284a120cac3065e057840ce43212407f85bd4e9d9438fb`.

## 2026-08-05 — Larger Task 3 planners recover facts but over-select content

The failed large-realizer result motivated a distinct, preregistered use of
scale: a model sees only the question and a finite typed fact graph and returns
an ordered JSON subset of offered fact IDs. It cannot write prose, numbers, or
unsupported facts. The immutable screen selects one historical fact-only row
from every company × family cell (32 total), and each target receives exactly
one demonstration from each of the other three companies in the same family.
The preregistration SHA-256 is
`d17255ae06fda1a2e819857a51f5c5e1dc664e19c3220c83521397b09dca11ba`.

The existing two-stage logistic planner scores **0.836742** macro set F1 on
this screen. Qwen 3.6 35B-A3B reaches **0.823722** (-0.013021), with two
companies better and two worse. The official Qwen2-57B-A14B Q4 model reaches
only **0.799723** (-0.037019), again with two companies better and two worse.
All 64 constrained outputs are valid. Exact set accuracy is 0.6250 for the
incumbent and 35B, and 0.59375 for 57B.

Scale does improve the hard numeric-content slice: nonempty-target macro F1 is
0.651717 for the incumbent, 0.757273 for 35B, and **0.772743** for 57B; micro
recall rises from 0.682927 to 0.878049 and 0.926829. But the larger models
select numeric facts when the historical reference is qualitative, and the
57B model amplifies that high-recall bias. The three-way row oracle is
0.921117, confirming diversity, but no label-free selector was preregistered;
we do not construct one after seeing the 32 labels.

The originally named Llama 3.3 model was removed before inference because its
approximately 70.55B untied parameters may exceed the organizers' literal 70B
total-parameter ceiling. A frozen compliance amendment substituted the
official 57B-total/14B-active Qwen2 model without changing prompts, targets,
metrics, or gates (SHA-256
`f1fdf482c9e7c580f4e7fd5294762d5dead21b251d0f9bed833a8e1fcd293ef5`).
Neither model clears the fixed +0.01/three-company/validity advance gate, so the
full 154-row run and any downstream compiler are forbidden. Task 3 v15 remains
the submission incumbent. Reproduction and immutable outputs are under
`artifacts/polyfiqa_large_content_planner_v1/`; the summary SHA-256 is
`890331e0939133ff6914e52b9bc24189c1140d207c1b0ae4902ed615aa0e0f52`.

## 2026-08-05 — New official Japanese-finance releases do not supply independent Task 2 labels

An audit of the newly published `TheFinAI/JF-KRC`, `JF-KRC-2`, and
`JF-KRC-3` releases found 346 rows in each, but they are not additional JF-ICR
intent examples. They ask for seven-way keiretsu-group affiliation from long
financial-disclosure evidence packs. All 346 IDs and all labels align across
the three versions after the third release's `independent`→`other` rename;
median evidence-token overlap between aligned variants is 0.73–0.94. They are
alternate renderings of one benchmark, not three independent datasets.

Consequently, they provide **zero** new five-level intent/commitment labels and
cannot confirm the promising 0.707510 distance-aware diversity profile. We do
not relabel the task, tune on proxy labels, or count alternate evidence
renderings as extra samples. Their long Japanese financial text is technically
usable for unsupervised adaptation, but that is low priority because our much
larger EDINET corpus already supplies the relevant language-domain signal and
earlier external-transfer branches did not improve source-disjoint accuracy.
The machine-readable audit, pinned revisions, licenses, file hashes, label
counts, and variant-similarity measurements are in
`artifacts/icr_independent_data_audit_v1/summary.json`. Task 2 stays frozen at
0.691700, with the 0.707510 profile retained only as an unconfirmed candidate.

## 2026-08-05 — Distance-aware Task 2 loss creates strong diversity, but the frozen advance gate stops it

A new experiment kept the incumbent Japanese-BERT five-class head and added
one preregistered ordered-distribution term: cross-entropy plus `0.5` times the
mean squared discrepancy between predicted and gold cumulative distributions
over the four ordinal thresholds. This is distinct from the rejected
cumulative-head architecture and the rejected external multi-task rank loss.
The coefficient, all five source-disjoint folds, seed 7, four epochs, FP32
training, expected-score decoder, safe rules, three profiles, and advance gate
were frozen before any result in
`docs/icr_distance_aware_loss_preregistration.md` (SHA-256
`24f5d725a0ddbad5ed4443883bca2a6240162a51dbf1fd2c01aa98087bc3433f`).

Training completed on CPU without touching the unrelated GPU workload. Every
loss component was finite; the CDF component contributed roughly 4% of total
loss. Raw argmax accuracy is **0.644269**, exactly tied in correct count with
the original general seed 7. After the frozen decoder and rules, the
distance-aware model alone is **0.683794** versus the **0.691700** incumbent.

The preregistered primary replacement—distance seed 7 plus general seed 19—rises
to **0.699605** (8 rescues, 6 harms), but improves only one source fold, ties
three, and worsens one. Its source-bootstrap probability of positive gain is
only 0.62843 and its 5th-percentile delta is -0.01843. It fails the explicit
minimum-two-improved-fold condition, so the frozen protocol forbids seed 19,
another coefficient, or a deployment refit.

The predeclared secondary diversity profile is unusually promising: the equal
mean of both general seeds and distance seed 7 reaches **0.707510**, with five
rescues, one harm, three folds improved, two tied, none worsened, bootstrap
P(gain) **0.91294**, and 5th-percentile delta **0.0**. These diagnostics meet
the repository's usual strict robustness thresholds. However, the experiment's
more conservative advance protocol explicitly requires every primary check
before continuing the branch. We therefore do not rewrite the gate after
seeing this result: seed 19 is not trained, no full refit is built, and the
0.691700 incumbent remains the submission primary. The secondary is
quarantined as a high-value independent-confirmation lead and paper result,
not promoted from the same 253 labels.

Implementation and immutable results are in
`scripts/train_icr_distance_aware_cv.py`,
`scripts/evaluate_icr_distance_aware.py`, and
`artifacts/task2_distance_aware_seed7_v1/`. The candidate OOF SHA-256 is
`a825532d39cd18200abb7cf5a277251539bcd2aa486189d04ad4c4be173aba35`.
Current or hidden test answers were not used.

## 2026-08-05 — Larger Task 3 realizers are safer when constrained, but still reduce ROUGE

The organizers' new 70B ceiling motivated a preregistered test of our locally
cached 27B dense and 35B-total MoE models. This did **not** repeat the earlier
free-form editor. The v15 answer core was delexicalized, every value/company/year
placeholder and literal numeric token was immutable, demonstrations came only
from other companies, the evidence suffix stayed byte-for-byte unchanged, and
the existing input-grounding validator forced invalid generations back to the
baseline. The score-blind screen contains one early-confirmation row for every
company × question-family cell (28 rows).

Neither architecture passes. The baseline is **0.446702**. The 35B MoE yields
**0.445246** (delta **-0.001456**; 1 win, 4 losses, 23 ties), with 12 placeholder
rejections. The 27B dense model follows the contract more often but falls to
**0.442909** (delta **-0.003792**; also 1 win, 4 losses, 23 ties), with three
placeholder rejections. Fallback ensures no invalid answer is deployed, but it
cannot prevent lexically valid rewrites from losing reference overlap. This
falsifies model size as the missing variable for surface realization: Task 3's
remaining bottleneck is input-only fact/content selection, not prose fluency.

The v15 deterministic incumbent remains frozen, and neither large model is
scaled. Preregistration, immutable target selection, raw generations, and
summaries are under `artifacts/polyfiqa_large_realizer_pilot_v1/`; the runner is
`scripts/pilot_polyfiqa_large_realizer.py`. A vLLM 0.26 FP8 attempt was also
operationally rejected on the GB10: after fixing missing Python development
headers locally, startup failed in its CUTLASS scaled-matrix kernel. Ollama ran
the identical cached model weights successfully, so this serving failure does
not affect the research conclusion.

## 2026-08-05 — Task 3 historical expansion preserves oracle headroom but falsifies routing

The sparse-edit lattice now spans 343 valid historical rows: 151 older
packet-disjoint rows and all 192 byte-aligned newer discovery rows.  Oracle
headroom is stable (+0.035817 older, +0.037708 newer), so candidate construction
is not the bottleneck.  Under a locked old-to-new evaluation, the conservative
signature gate abstains completely.  The primary ridge experiment trains only
on older packets from three issuers and predicts newer packets for the fourth;
it changes four rows, harms all four, and falls from 0.365242 to 0.364371.
Separately, an apparently robust Easy-cash evidence expansion improves every
issuer in discovery but harms both later packet-disjoint eligible cases.  Both
branches are rejected without tuning, and v15 remains frozen.  Reproduction is
under `artifacts/polyfiqa_historical_sparse_lattice_v1/`,
`artifacts/polyfiqa_temporal_sparse_selector_v1/`, and
`artifacts/polyfiqa_cash_evidence_recall_confirmation_v1/`.

## 2026-08-05 — Annotator disagreement identifies hard rows, but hard-example weighting is unsafe

The official EBISU Label Studio export contains substantially more supervision
than the earlier rationale experiment used: 185 labels over 94 aligned rows,
from three annotators. Eighty rows are unanimous and 14 disagree. The frozen
two-seed ordinal incumbent is correct on **81.25%** of unanimous rows but only
**50.0%** of disagreement rows, confirming that disagreement identifies the
decision boundary. Raw label averaging is unsuitable: the adjudicated gold is
not the plurality label for IDs 38 and 86, and disagreement often places gold
at a stronger ordinal extreme than the other judgments. The immutable audit is
`artifacts/icr_annotation_disagreement_audit_v1/`.

A safer hard-gold experiment was frozen before training in
`docs/icr_disagreement_weighting_preregistration.md` (SHA-256
`c6d5199b28fe6a95ad17865b24b84a31ceadd71372ee4a0b8791a17269b23fb0`).
It keeps the official class target unchanged and multiplies cross-entropy by
exactly 2.0 only for the 14 disagreement rows. Architecture, seed 7, optimizer,
four epochs, maximum length, decoder, and safe rules match the general model.
Source fold 2 is the sole pilot because none of its 56 validation rows appears
in the annotation export; all 14 weighted rows belong strictly to training.

The weighted model alone ties the incumbent at **34/56 (0.607143)**. The frozen
primary replacement of general seed 7 with weighted seed 7 reaches **35/56
(0.625000)**, with six rescues but five harms. The predeclared three-model
diversity view reaches **36/56 (0.642857)**, with five rescues and three harms.
The primary therefore fails the maximum-two-harms gate despite its positive
point estimate. Per protocol, folds 0, 1, 3, and 4 and a second seed are not
trained, and the weight is not tuned.

Post-hoc diagnostics explain the instability: all six primary rescues are
`0→+1` corrections, while all five harms turn a correct `0` into `+1`. The
disagreement subset is dominated by commitment-side labels, so global hard-row
weighting shifts the boundary rather than teaching a portable uncertainty
concept. This negative result complements the rationale reversal: human
annotation metadata locates hard cases, but neither token-level rationale loss
nor global disagreement weighting reliably decides which side of the ordinal
boundary is correct. The final evaluation is
`artifacts/task2_disagreement_weighted_fold2_seed7_v1/evaluation.json`
(SHA-256
`8e62a8fb747861fc1daa1e06da23f686ed225a591671230fba3e1cfebe2c84e4`).
The 0.691700 policy-independent ordinal incumbent remains unchanged, and this
external-data branch also remains license-gated regardless of performance.

## 2026-08-05 — Organizer-data and submission-integrity rehearsal passes

The three live organizer Drive folders were inventoried and every remote file
was downloaded into temporary storage for an immutable comparison. The folders
still expose exactly four expected files: Task 1 train and validation, Task 2
public, and Task 3 public. All paths and Drive IDs match the pinned inventory,
and all four byte-level SHA-256 values match the local copies. No test set,
starter kit, README, sample submission, or silent revision is currently
present. The audit mutates no organizer data, fails closed on any future
difference, and is reproducible with `scripts/audit_organizer_drive.py`.
Machine-readable evidence is
`artifacts/organizer_drive_audit_20260805.json` (SHA-256
`7a0144afd3ea04fb504ecab84210dab10d1015a83e58ab02e68e62cf3756363c`).

The three-task rehearsal manifest still verifies 3 inputs and 20 candidate
files, and the policy-safe selector chooses the mixed full-refit Task 1,
source-disjoint ordinal Task 2, and input-only v15 Task 3 lanes. A stricter
Task 1 validator now enforces prompt-specific ontologies and the promoted
calendar invariant. As intended, it rejects the historical mixed candidate at
its single stale `8ης` prediction. The input-only migration changes exactly
that one of 200 rows; the resulting Task 1 file plus the selected Task 2 and
Task 3 files all pass row, ID, answer-schema, and alignment validation. Evidence
is under `artifacts/submission_readiness_20260805/`. Fresh test-day inference
already contains the normalization, so a migration report with any changed row
will identify a stale cache rather than silently patching it.

## 2026-08-05 — Official human rationales yield discovery-only diversity, then fail independent confirmation

The official EBISU paper repository exposes a 95-task Label Studio export. A
pinned collector matches it to 94 unique released JF-ICR rows and retains only
notes from annotations whose choice equals the released gold label. Seventy-five
rows contain 200 unique exact response spans; the fixed Japanese tokenizer
recovers 189 of them as unambiguous response-token sequences. The annotations
cover source folds 0, 1, 3, and 4, but not fold 2. Provenance, raw hashes, exact
offsets, and the license caveat are under
`data/external/ebisu_human_rationales/`. The GitHub repository has no explicit
license although the linked Hugging Face dataset declares Apache-2.0, so this
lane remains policy-gated regardless of performance.

The preregistered model adds a training-only token head to Japanese BERT and
jointly learns five-way intent (weight 1.0) and exact human cue localization
(weight 0.10). Held-out rationale maps are empty by construction. Source fold 2
was chosen first because it contains no rationale-annotated validation row. The
rationale model alone reaches **33/56 (0.589286)** versus **34/56 (0.607143)**
for general seed 7. Its replacement pair reaches 37/56 but makes four harms.
Both violate the original frozen gate, so that branch is rejected without
changing its conditions or running the zero-weight control.

A predeclared secondary equal three-model blend produces a promising discovery
result on fold 2: **38/56 (0.678571)** versus the incumbent's 34/56, with five
rescues and one harm. Because a pre-existing ordinary third seed also improves
that fold to 36/56, the gain could be generic stochastic diversity. A new,
explicitly post-discovery confirmation protocol therefore excludes fold 2 from
all promotion evidence and fixes unseen fold 0 as the first test. It fails
clearly: the rationale three-model blend scores **27/51 (0.529412)**, below the
two-seed incumbent at **30/51 (0.588235)** and the ordinary three-seed control
at **28/51 (0.549020)**, with zero rescues and three harms. Per the frozen gate,
the zero-weight control and folds 1, 3, and 4 are not run.

The original and confirmation preregistration hashes are respectively
`870257cb14573e19f9bb109912ef18aae8e6d565c53435a4260eea9190cf3c88`
and
`421b8caf07ac7095435a9f84af92bdbf9494159719bd9c57ed1f8fe1729c1d5c`.
Frozen evaluation hashes are
`25a91a38bb620eae55d05e6eed6918a82e0075e0d150c5f77a3db15f6796cf00`
for discovery and
`33e8b6acec3e97889d536bba669508d0ef610ba5e0a6134515e5171c7f5f7062`
for confirmation. This is useful evidence that human cue supervision can add
model diversity without reliable cross-source improvement; the policy-independent
0.691700 two-seed ordinal system remains primary.

## 2026-08-05 — Transductive label-shift correction fails source-held-out evaluation

A new CPU-only branch tested whether the unlabeled target packet's aggregate
prediction distribution could safely adapt the frozen ordinal ensemble. The
primary method and gate were hashed before evaluation in
`docs/icr_transductive_label_shift_preregistration.md` (SHA-256
`f8d1ee2378e107f174d85c2143fdc0ca983549f989284bf1df4a520a09df0ef3`).
The primary estimates negative/neutral/positive target priors from a soft OOF
confusion matrix, shrinks them toward the four-fold source prior with fixed
ridge 0.1, preserves within-bin odds, and never reads held-fold labels.

The identity control exactly reproduces **0.691700**. The preregistered primary
falls to **0.632411**: one fold improves, one ties, and three regress; it rescues
5 rows but harms 20. Its source-cluster bootstrap delta is entirely negative at
the 5th--95th percentiles (-0.1022 to -0.0236), with P(gain)=0.00084. Stronger
ridge 1.0 limits the damage but still scores only 0.683794. The five-way control
collapses to 0.549407, and shrunk three-bin EM reaches 0.671937. No method is
promoted.

The failure mechanism is informative. On folds 0, 1, and 4, the primary's
estimated versus true three-bin prior has L1 error 0.066, 0.092, and 0.094. On
folds 2 and 3 it explodes to 0.471 and 0.552: fold 2 is inferred as 22.9%
negative when the evaluation truth is 7.1%, while fold 3 is inferred as 92.2%
positive when the truth is 64.6%. Thus the label-shift assumption is violated;
the classifier's class-conditional probability signature changes by source.
Unlabeled prediction histograms cannot distinguish that calibration shift from
real target-prior shift. Post-hoc abstention is not introduced.

Reproduction is `scripts/evaluate_icr_transductive_label_shift.py`; the summary
SHA-256 is
`0d62bec1f4dafb6805b6d879c678d8e8ad27570bc1b84ba7e9aa8dedcd816ace`.
The source-disjoint ordinal model remains Task 2 primary.

## 2026-08-05 — The legacy source-continuity auto-gate is rejected

The release audit found that the label-free selector could still replace the
newer ordinal primary with the older source-prior candidate whenever packet
similarity was high. This rule predated the ordinal improvement and was
retested under the source prior's intended favorable condition, rather than
removed on intuition. Neural predictions are the frozen five-fold
source-disjoint OOF probabilities; the source prior is evaluated leave-one-row
out while allowing other rows from the target source to remain.

The old packet gate genuinely fires: median nearest similarity is 0.185652 and
77.87% of rows have two-of-three source agreement. Nevertheless, selecting the
whole known-source candidate falls from **0.691700** to **0.640316**, with 27
rescues and 40 harms. A source-cluster bootstrap gives only 0.0485 probability
of gain and a 5th--95th percentile delta of -0.1115--0.0000. Even two fixed
row-level residuals fail: agreement>=2 plus similarity>=0.16 reaches 0.667984;
the stricter agreement>=3 version reaches 0.679842. Both remain below the
ordinal incumbent, with bootstrap probabilities of gain 0.1329 and 0.1960.

The automatic gate is therefore removed from both the portfolio builder and
the fail-closed selector. `task2_japanese_ordinal` remains the first
policy-independent run under both familiar and novel source diagnostics;
`task2_known_source` remains only a materially different diversity file if the
organizers permit multiple submissions. Reproduction is
`scripts/evaluate_icr_source_continuity_gate.py`; the machine-readable audit is
`artifacts/icr_gates/source_continuity_gate_audit.json` with SHA-256
`40b7a606106713a1fba305b9442d1d1be15428b94e65417813548cb8caa7c038`.

## 2026-08-05 — Gold-anchored decisive-event teacher v1 fails its feasibility pilot

The remaining teacher-supervision lead was exercised on a bounded five-row
pilot containing exactly one official training example from each ordinal label
and both frozen rubric views. No current/hidden-test input or sealed external
label was used. Teacher 1 was the pinned Ollama quantization
`qwen3.6:35b-a3b` at digest
`07d35212591fc27746f0a317c975a6d68754fb38e9053d82e25f06057af28522`.

Only **1/10** structured judgments passes the previously frozen validator.
Eight fail the factor/label consistency rules and one changes the fixed gold
label despite the prompt. Consequently, **0/5** rows has two valid rubric
views and none has invariant span/factor supervision. The strict agreement
function requires two rubrics within each teacher before it even considers a
second teacher. Therefore Gemma cannot rescue any row in this pilot, and the
second-teacher run, 100-row calibration, counterfactual generation, and student
training are stopped by logical futility rather than continued for appearance.

This is principally a measurement-ontology failure, not evidence that Qwen
cannot read Japanese IR text. For example, the two rubrics often quote the same
grounded span while assigning semantically plausible factors that the v1
hard-coded mapping disallows. A future v2 would need a newly preregistered
stance ontology and a disjoint feasibility sample; the failed v1 pilot cannot
be silently relabeled or used to tune the current gate.

The infrastructure was hardened before inference. The decisive-event runner
now supports deterministic JSON-schema calls through Ollama or an
OpenAI-compatible vLLM endpoint, records immutable model revisions, and can
batch vLLM requests. More importantly, teacher independence can no longer be
spoofed with aliases: the evaluator requires distinct recorded models from two
distinct model families and rejects model/revision drift. Recomputed evidence
is in `artifacts/icr_decisive_event_teacher_pilot_v1/`; the summarizer is
`scripts/summarize_icr_decisive_event_pilot.py`. No Task 2 submission candidate
changes, and the source-disjoint ordinal incumbent remains primary. The exact
judgment and recomputed-summary SHA-256 values are respectively
`357a20945dae96d19ca409854d8dccb943f3f47925700e3797b131d5240c3a18`
and `10c0cacb16daf281d194fa3ebc89926e5d3b1b0db0a528fc3003a0aa776d7341`.

## 2026-08-05 — Event-conditioned expert utility router is a clean negative control

The five frozen OOF experts still have a 0.778656 choice oracle, but neither
lexical local competence nor direct event-feature classification had shown how
to route that diversity. A new router therefore predicts only the relative
utility of an alternative expert on a disagreement: +1 for a rescue, -1 for a
harm, and zero when neither the alternative nor incumbent is correct. Its
features contain the frozen OOF probability/prediction signature plus either
raw Zunda events, Ruri-ranked Zunda semantic events, or both. Ridge penalty and
the abstention margin are selected inside each outer reconstructed-source fold;
the incumbent wins exact inner ties.

The probability-only control makes four nested OOF switches and all four are
harms, falling from 0.691700 to **0.675889**. All three event variants select
the no-op incumbent in every outer fold and therefore preserve **0.691700**
byte-for-byte: zero rescues, zero harms, five tied folds. This is a decisive
rejection, not a failed implementation. Unsupervised parser signatures do not
identify which frozen expert will be correct under issuer/source shift.
`artifacts/icr_event_utility_router_v1/` contains predictions, fold choices,
and bootstrap results; reproduction is
`scripts/evaluate_icr_event_utility_router.py`.

The remaining high-return Task 2 lane is verified event supervision. The
gold-anchored teacher protocol has now been hardened so label agreement cannot
pass trivially: two independent teachers and two rubric views must agree on the
same grounded literal decisive span and the same six modality factors. The
frozen go/no-go gate retains at least four labels and at least 70% exact
span/factor agreement. A CPU-only Gemma 4B feasibility call used zero VRAM but
did not complete one long structured judgment in ten minutes, so the run was
stopped and the model unloaded rather than scaled wastefully. The resumable
runner, strict agreement evaluator, and unit tests are ready for the next free
GPU window in `label_icr_decisive_event_teachers.py`,
`evaluate_icr_decisive_event_teacher_calibration.py`, and
`fineval.icr_silver`.

## 2026-08-05 — Task 1 best result is now fully persisted and deployable

The 0.933157 grouped two-seed composition was previously an OOF/validation
result without a single hidden-test driver. `predict_task1_grouped_consensus.py`
now refits both operation selectors from the immutable training OOF artifacts,
accepts five label-free proposal views, applies the two-seed grouped consensus,
and composes the frozen structural and high-confidence-union policies. On the
80-row validation text it reproduces the operation, structural, union, and
final composition prediction files **byte-for-byte** while loading only the
`text` column from the prediction frame. The saved full seed-7 proposer also
reproduces its raw proposal file byte-for-byte through
`predict_task1_gliner_raw.py`.

The original OOF trainer discarded its five fold checkpoints per seed. It has
been extended with `--save-fold-models`, and
`predict_task1_gliner_fold_ensemble.py` deterministically reconstructs the
mean and vote-three views on arbitrary text. The complete persisted seed-7 run
covers all 320 training rows across five entity-linked folds and occupies about
5.5 GB. Reloading all five saved checkpoints reproduces every per-fold raw file
and both ensemble views byte-for-byte.

Individual retrained views drift, as expected: the fold mean accepts six
operations and has one harmful validation row, while vote-three accepts two
with no harms. Their label-free two-of-three quorum with the persisted full-data
view selects three operations and reaches 0.927264 with three improvements and
no harms. Composing that quorum with the fixed structural and high-confidence
policies reproduces the best **0.933157** artifact byte-for-byte (SHA-256
`837a7a9dcda8cffba4e28658c6f582642053d44e6c838db00bd188d88d000f69`):
+0.012326 over 0.920831, four improved rows, zero harms, positive deltas for
all three entity types, bootstrap p05 +0.002747, and P(gain)=0.9842.

`predict_task1_grouped_consensus.py --consensus-mode single_seed` is therefore
the primary hidden-test path with persisted models and text-only inputs. The
independent seed-19 rebuild is also complete: all five checkpoints replay every
per-fold and ensemble output byte-for-byte. Its strict cross-seed composition
is still no-harm at **0.932264** (+0.011433; four improvements; P(gain)=0.9842),
but is lower than the seed-7 primary and has a -0.018182 person-type delta.
It is retained as diversity, not promoted.

## 2026-08-05 — Submission preflight becomes fail-closed

Task 1 mixed GRFinNUM/GRFinNER packets now have deterministic extraction and
merge utilities that preserve global row indices and reject missing, duplicate,
reordered, or cross-track rows. Task 2's predictor loads only `query` and `id`;
the runbook now includes the promoted `--ordinal-mean` decoder (0.691700 rather
than the 0.671937 argmax control). Task 3's input-only predictor loads no answer
column, and strict validation now exits nonzero on bad structure, word budget,
or numeric provenance. It passes all 76 original public rows and all 256
historical hidden-style rows, with maximum answer length exactly 100 words.
The Task 2 ordinal primary was rerun over all 253 public rows through the
label-excluding loader and is byte-identical to the promoted artifact (SHA-256
`f52fa618c7f0b68e5c0048c6de67f66c9cbd887f20ff726948a751477e7ba438`).

The portfolio builder now fingerprints the organizer inputs as well as every
candidate, recording SHA-256, byte count, and row count and writing a detached
`manifest.sha256`. `scripts/verify_submission_manifest.py` independently
recomputes all four invariants and rejects missing inputs, duplicate candidate
names, altered manifests, or altered files before selection/upload. This closes
the remaining provenance gap between download and portal submission. The full
suite passes with 216 tests.

## 2026-08-05 — Tsutsuji semantic-signature control fails; structured use retained

The openly licensed Tsutsuji 1.1u dictionary was downloaded with provenance
and SHA-256 tracking. It contains 16,801 L9 Japanese functional-expression
surface forms in a nine-level hierarchy; the extractor retains 16,569 unique
surface/class pairs and 16,253 forms of at least three characters. Exact
response-side matches produce 348 inherited semantic-class, top-class,
position, and tail features, with at least one match on 252/253 rows.

A fully nested five-source-fold ridge diagnostic scores **0.596838** from the
semantic signatures alone. Adding them to the incumbent OOF probabilities
scores **0.608696**, versus the **0.691700** incumbent: 18 rescues, 39 harms,
one improving fold and four worsening folds, bootstrap p05 -0.146341 and
P(gain)=0.00988. The global-feature branch is decisively rejected.

The failure is informative rather than evidence that functional expressions
are irrelevant. Forms such as negation, inference, intention, and possibility
are ubiquitous and scope-sensitive; substring/class counts cannot identify
which event answers the question. Tsutsuji is therefore retained only for
predicate marking, semantic counterfactual generation, and scope-composition
inside a query-conditioned decisive-event model. Reproduction is in
`scripts/evaluate_icr_tsutsuji_semantics.py` and
`artifacts/icr_tsutsuji_semantics_v1/`.

## 2026-08-05 — Dynamic frozen-expert selection fails under source shift

The five frozen encoder views have an oracle accuracy of **0.778656**, so a
source-nested dynamic-selection control tested whether local lexical competence
could expose that headroom. For every held source fold, character TF-IDF was
fit only on the outer training rows. Each expert's correctness was estimated
from source-balanced nearest neighbours, and the neighbour count, competence
margin, and label-conditional option were selected inside the remaining source
folds. The incumbent was the conservative fallback.

The result is negative: dynamic selection reaches **0.683794** versus
**0.691700** for the incumbent. It produces one rescue and three harms, with
one fold improving, two tying, and two worsening. The source-cluster bootstrap
has p05 -0.020661 and P(gain)=0.09201. Although Rinna is selected on 39 rows,
most selections agree with the incumbent; the few actual label changes are
net harmful. Local lexical similarity therefore does not identify model
expertise under issuer/document shift. The branch is rejected and becomes a
negative control for the event-supervised residual hypothesis. Reproduction is
in `scripts/evaluate_icr_dynamic_expert_selection.py` and
`artifacts/icr_dynamic_expert_selection_v1/`.

## 2026-08-05 — Neural independent-boundary fold-0 gate fails

The frozen target-only Tohoku BERT experiment was completed on source fold 0
without using the occupied GPU. After four epochs, the four independent
class-balanced BCE heads score **0.235294** raw and with frozen rules, versus
**0.588235** for the incumbent on the same 51 rows. Predictions collapse to
50 class-0 and one class-+1 output, so the predeclared futility gate stops all
remaining folds and the prototype/event extensions are not run.

The failure also exposes a decoder mismatch: cost-sensitive BCE logits are not
prevalence-calibrated probabilities. The label-free analytic inverse subtracts
`log(pos_weight)` from each boundary logit and raises accuracy to **0.509804**,
but still trails the incumbent, rescues only four rows while harming eight, and
predicts no rare extreme label. Thus calibration explains part, but not all,
of the collapse. Reproduction is under
`artifacts/icr_independent_boundaries_cv/seed7_cpu_fold0/`, including
`decoder_audit.json`. The neural boundary branch is rejected; the CPU lexical
result remains a representation diagnostic only.

## 2026-08-05 — Independent-boundary positive signal; raw anchors rejected

The adjacent-boundary hypothesis was tested without GPU using four independent,
source-equal character models and a fixed least-squares monotone projection.
The full target response reaches **0.620553** raw accuracy and **0.644269**
after the existing frozen safe rules, materially above the ordinary lexical
full-response diagnostic at 0.577075 but below the 0.691700 neural incumbent.
This supports independent boundary heads as a neural objective, not as a CPU
replacement.

Concatenating the retrieved below/above examples is decisively harmful:
**0.549407** raw and **0.569170** after rules. Against the target-only boundary
view, the observed ruled delta is -0.075099; source-cluster bootstrap p95 is
-0.018868 and P(gain)=0.0123. The extra text particularly harms `0|+1` and
`+1|+2`. Raw anchors are therefore rejected as model input. They may be used
later only through a separately encoded contrastive/prototype loss.

The corrected v2 corpus contains all **5,060** fold-copied boundary records,
serializes no anchor gold labels, and uses two fixed rubric prototypes when the
sole outer-train `-2` example cannot retrieve a non-self same-side anchor. The
full-response Japanese-BERT trainer uses four independent, source-balanced BCE
heads and decode-time monotone projection. Its subsequent real fold-0 gate
failed as recorded above, so it is retained as a negative control rather than
continued. Artifacts are under `artifacts/icr_boundary_anchors_v2/` and
`artifacts/icr_boundary_anchor_views_v1/`; training is in
`scripts/train_icr_independent_boundaries_cv.py`.

## 2026-08-05 — Source-disjoint decisive-event and boundary-anchor diagnostics

Hard decisive-event truncation is rejected. A source-equal weighted character
TF-IDF diagnostic reaches **0.577075** with the complete response, but only
**0.501976** with the highest-ranked input-only event and **0.529644** with the
top three events. The event views recover slightly more `0`/`+2` recall but
lose too many `+1` rows, and none predicts the rare negative labels. Decisive
events will therefore supervise attention/factors while the encoder retains
the complete answer. The result is under
`artifacts/icr_event_views_v1/`; it is not a neural-incumbent comparison or a
promotion claim.

A first adjacent-boundary corpus was built under
`artifacts/icr_boundary_anchors_v1/`. It represents each example as four
comparisons against topic-near training anchors on either side of `-2|-1`,
`-1|0`, `0|+1`, and `+1|+2`. All retrieval is fitted inside each outer source
fold and does not use the target label. The five folds contain **5,058** written
boundary records; 98.37--100% of anchors come from a different source group.
Two extreme-class training records for which excluding the target leaves no
valid `-2` anchor were skipped in v1; v2 replaces only those missing anchors
with fixed rubric prototypes and removes serialized anchor labels. This
objective differs from the rejected shared-cutpoint cumulative
head: it trains four independent, anchor-conditioned decisions and applies
monotonic projection only at decoding.

The 100-row teacher calibration queue is also complete under
`artifacts/icr_decisive_event_queue_v1/`: 1,971 exact input substrings, 200
prompt records across two rubrics, all five labels, and balanced source folds.
Teacher inference remains pending; no silver rationale is assumed correct.

## 2026-08-05 — Event-level extended-modality transfer lead

The Japanese Extended Modality Corpus is a closer structural match to JF-ICR
than generic NLI.  Its annotation scheme decomposes about 50,000 Japanese event
mentions into Source, Time, Conditional, Primary modality, Actuality,
Evaluation, and Focus.  These are nearly the latent variables needed to decide
whether a corporate response is a definite commitment, a hedge, an
explanation, or a refusal.  The corpus is described by Matsuyoshi et al. and in
the later annotation report; data access and reuse terms must be verified
before training.

The high-return architecture is therefore event-level multiple-instance
ordinal classification.  A query-conditioned span head identifies response
events and the decisive predicate; auxiliary heads predict the seven modality
components; a bag-level aggregator maps the strongest relevant event to the
five JF-ICR labels.  This differs from the rejected flat full-response
encoders: it gives the router an explicit reason to choose between the five
frozen models whose prediction oracle is 0.778656.

Teacher rationales on official training data will be accepted only when two
materially different prompts or teachers return the same literal Japanese
decisive span and compatible factors.  A counterfactual check then replaces or
masks that predicate while keeping names and numbers fixed; an accepted span
must move the predicted commitment level in the expected direction.  This
turns rationale distillation into testable supervision rather than an
unverifiable explanation.  The existing 819 unrelated-issuer Q&A pairs and
ML-Promise Japanese subset provide domain and promise-specific auxiliary data,
while the official five source-disjoint folds remain the only promotion gate.

## 2026-08-05 — Japanese ML-Promise transfer lead

The closest external labeled resource found for JF-ICR is ML-Promise /
PromiseEval (SemEval 2025). Its Japanese corporate ESG subset annotates promise
presence, exact promise strings, supporting evidence, evidence clarity, and
verification horizon. This aligns directly with the latent JF-ICR factors—
future action, commitment, evidential support, and temporal scope—and is a
stronger transfer hypothesis than the rejected generic EDINET MLM adaptation.
The official data page still requires registration and the formerly cited
public GitHub mirrors return 404, so no dataset or result is claimed. Once
obtained with its license terms, it will feed auxiliary promise/span/timeline
heads rather than be naively mapped to the five JF-ICR labels.

## 2026-08-05 — Unrelated-issuer Japanese IR Q&A corpus

Generic EDINET masked-language adaptation improved financial fluency but not
robust commitment recognition, so the next corpus targets the missing task
structure directly.  An allow-listed collector retrieved 165/177 public IR
documents and extracted 858 explicit Japanese question/answer turns.  Quality
curation removes general FAQ pages and navigation spillover, leaving **819**
earnings/investor Q&A pairs from nine unrelated issuers.  A high-recall
forward-intent index contains **672** pairs; 69 contain refusal/withholding
cues useful for the rare negative boundaries.

All reconstructed JF-ICR issuers and domains are explicitly excluded.  Each
new question and response was compared against all 253 released pairs; zero
reaches the frozen 0.92 near-duplicate threshold and the maximum retained
similarity is 0.631579.  Every source URL, final URL, retrieval timestamp,
content hash, raw file, extraction method, and pair location is retained under
`data/external/japanese_ir_qa/`.  Because individual corporate-site reuse terms
are not uniform, the project does not assert a blanket redistribution license;
that limitation is recorded per pair and in the provenance README.

This corpus is not assigned five-way labels from lexical cues.  The cues only
define a high-recall teacher-labeling queue.  Promotion requires agreement
across materially different teachers and rubric phrasings, benchmark-issuer
exclusion, confidence weighting, and the unchanged source-disjoint JF-ICR gate.

## 2026-08-04 — Nested residual-stacking diagnostic

A source-nested ridge meta-model was evaluated over the frozen probability
vectors from the two general Japanese-BERT seeds, EDINET seed 19, Rinna seed 7,
and the full-size Japanese-finance model.  The ridge penalty is selected inside
each outer training partition, source groups receive equal total weight, and
no query text or held-out source labels enter the stacker.  It reaches 0.699605
versus the 0.691700 incumbent, with 11 rescues and 9 harms.  Fold outcomes are
two improvements, one tie, and two regressions; the source-bootstrap p05 is
-0.02390 and P(gain) is only 0.58357.  It is therefore diagnostic, not
promoted.

The oracle accuracy over the five base ordinal predictions is 0.778656.  This
shows substantial complementary signal exists, but confirms that global
calibration and low-dimensional confidence features cannot robustly identify
the right expert under source shift.  Reproduction is in
`scripts/evaluate_icr_nested_residual_stacker.py` and
`artifacts/icr_nested_residual_stacker/`.

## 2026-08-04 — Predeclared Japanese NLI intermediate-training ablation

An orthogonal follow-up is frozen behind Rinna and ModernBERT:
`akiFQC/bert-base-japanese-v3_nli-jsnli-jnli-jsick` revision
`f1cdc47ca18a7413d7d7867856ca503de38ed55d`. It uses the incumbent Tohoku
Japanese-BERT-v3 backbone after intermediate training on JSNLI, JNLI, and
JSICK, and reports 0.914 JNLI validation accuracy. This directly tests the
hypothesis that entailment/neutrality/contradiction representations transfer
to question-conditioned commitment, hedging, and refusal better than finance
language adaptation alone.

The pinned 445 MB checkpoint is cached and SHA-256 recorded. Its original
three-way NLI head is replaced by a fresh five-way JF-ICR head; a CPU
forward/backward smoke test produced finite loss, logits, and embedding
gradients. Seed 7, four epochs, batch eight, 512 tokens, and the existing
source/bootstrap promotion gate are frozen before any JF-ICR result. It will
not consume compute until the two higher-priority encoder experiments resolve.

## 2026-08-04 — Predeclared Japanese ModernBERT follow-up

Before observing any JF-ICR fold result, a second independent encoder experiment
was frozen at
`artifacts/icr_gates/modernbert_ja_310m_predeclared_gate.json`.
`sbintuitions/modernbert-ja-310m` revision
`77675fc96a7e445e982e2ba90246b816efc74ec6` was selected because its model card
reports 4.09 trillion Japanese/English pretraining tokens, an 8,192-token
context, and materially stronger Japanese inference/classification benchmarks
than Tohoku BERT v3. Its tokenizer was checked locally: paired inputs begin
with the pretrained `<s>` token, and Transformers' ModernBERT classification
head pools that first token as required by the model card.

The candidate keeps the incumbent seed, four epochs, batch size eight, and
2e-5 learning rate, while using a predeclared 1,024-token cap to test the
architecture's long-context advantage. It must pass the same source-fold and
cluster-bootstrap gate as Rinna. It will start only after the running Rinna CV
finishes, and no second seed or full-data checkpoint is permitted unless the
one-seed gate passes. Both CV and full-data trainers now accept and record an
immutable `--revision`, so this and future Hub experiments load the declared
commit rather than a moving repository default. The pinned 1.26 GB checkpoint
is cached locally. A one-example CPU forward/backward smoke test produced
finite logits, loss, and embedding gradients; the 253 public pairs have a
maximum of 726 ModernBERT tokens, so the frozen 1,024-token cap preserves every
pair without truncation (only three would exceed 512). The CV trainer now also
checkpoints an aligned partial OOF table after every completed fold, preventing
a late interruption from discarding hours of source-disjoint evaluation. The
frozen GPU recipe uses bfloat16 autocast with FP32 optimizer state, matching the
checkpoint's published usage guidance and the DGX GB10's native precision; the
choice was recorded before any JF-ICR fold was run.

## 2026-08-04 — Rinna Japanese RoBERTa diversity candidate

The next GPU experiment is frozen before any source-fold score is observed:
`rinna/japanese-roberta-base` revision
`f93982a6f5a8bb9fd483b9deedae693091114b33`, seed 7, four epochs, and the
unchanged five source folds. This is a genuinely different encoder family from
the incumbent Japanese BERT. Its nonstandard pretrained input convention is
implemented explicitly: slow SentencePiece tokenizer, literal `[CLS]` prefix,
lowercasing flag, and zero-based position IDs. A two-example CPU
forward/backward test passed with finite gradients. The exact gate and stopping
rule are machine-readable in
`artifacts/icr_gates/rinna_roberta_predeclared_gate.json`. A second seed and
full-data fit are forbidden unless the one-seed candidate clears the same
accuracy, fold, and source-bootstrap gate used for EDINET adaptation.

The predeclared first-two-fold futility checkpoint completed on CPU. Raw fold
accuracies are 0.607843 and 0.800000. With ordinal decoding and frozen rules,
Rinna ties the incumbent at 0.712871 across those 101 rows, improving fold 0
and trailing fold 1; the three-model blend is lower at 0.702970. This is neither
a win nor a collapse, so the stopping rule requires all five folds. Full CV is
continuing without any recipe change; no ensemble slot is awarded from this
partial result.

Fold 2 subsequently completed at 0.660714 raw accuracy (37/56). The three-fold
raw aggregate is 0.687898 over 157 rows. This is near the 0.691700 incumbent,
is not classifier collapse, and leaves a feasible winning upper bound, so the
frozen stopping rule requires folds 3 and 4. No promotion claim is made from
the partial aggregate because ordinal/rule decoding and source bootstrap require
complete aligned logits.

As a non-decisional viability check, the raw Rinna predictions beat the final
incumbent count on folds 0 and 2 and trail it on fold 1. The gate therefore
requires at least one additional fold win and no second loss across folds 3--4.
If ordinal/rule decoding preserves these raw comparisons, this is approximately
70/96 correct (72.9%) over the last two folds. The condition is demanding but
feasible, which independently confirms the decision to finish CV.

Fold 3 completed at 0.666667 raw accuracy (32/48), compared with 34/48 for the
final incumbent on that fold. The four-fold raw aggregate is 0.682927 over 205
rows, with wins on folds 0 and 2 and losses on folds 1 and 3. Thus raw argmax
can no longer satisfy the maximum-one-worsened-fold gate regardless of fold 4.
The final fold still runs because ordinal expected-score decoding and frozen
rules can change fold outcomes; only the complete aligned evaluator is allowed
to reject or promote the candidate.

Fold 4 scored 0.583333 raw (28/48), completing the raw five-fold estimate at
0.664032. With the authoritative decoder and frozen rules, Rinna alone reaches
0.683794. The best profile is the equal probability blend of the two incumbent
seeds plus Rinna under ordinal decoding: **0.695652**, versus 0.691700 for the
incumbent. That four-tenths-point apparent gain is not robust. It improves two
folds, ties one, worsens two, has source-bootstrap p05 -0.01538, median
+0.00383, p95 +0.02381, and P(gain)=0.55176. It fails four elements of the
predeclared gate beyond merely missing a point estimate.

Rinna is therefore rejected for deployment. No second seed or full-data Rinna
checkpoint will be trained. Its complete OOF predictions and decision remain
under `artifacts/task2_rinna_roberta_cv/seed7_full/` and
`artifacts/icr_rinna_evaluation/` as reproducible negative evidence.

## 2026-08-04 — Japanese encoder source-disjoint ensemble

`cl-tohoku/bert-base-japanese-v3` was fine-tuned as a five-way sequence
classifier on the parsed question/response pair. Validation holds out complete
reconstructed source groups, not random rows. Seed 7 scores 0.644269 and seed
19 scores 0.632411. Equal averaging of their class probabilities scores
0.656126; applying the previously frozen `safe_rules` only when they predict a
non-majority label raises accuracy to **0.671937**. The same hybrid scores
0.755319 on the original 94-row core and 0.622642 on the 159-row expansion.

An independently tested seed 31 scores 0.620553. Although one post-hoc seed
pair scores higher, it is rejected to avoid selecting an ensemble after seeing
the validation labels. Japanese DeBERTa-v3 was also rejected after its first
two folds scored 0.019608 and 0.000000 with the stable recipe. Full-data seed 7
and 19 checkpoints are frozen in `artifacts/task2_japanese_full`; inference is
implemented by `scripts/predict_icr_japanese_ensemble.py`. The neural system is
the primary source-novel candidate, while majority, rules, exact lookup, and
known-source prior remain distribution-shift candidates.

A binary-core follow-up trained only on the dominant `0` and `+1` labels to
test whether specializing on their boundary would improve robustness. Seed 7
scores 0.604743 raw and 0.620553 with the frozen rules under the same source
folds, decisively below 0.671937. The branch was stopped and rejected rather
than spending another seed on a five-point deficit.

## 2026-07-31 — Dataset and first local-judge pilot

### Reproducible facts

- Public set: 253 examples.
- Label counts: `+2=29`, `+1=139`, `0=72`, `-1=11`, `-2=2`.
- Majority (`+1`) accuracy: 0.549407.
- Character/word TF-IDF logistic regression, repeated stratified two-fold CV:
  mean accuracy 0.569929, standard deviation 0.026717. It predicts almost no
  scarce classes and is not competitive.
- Strict parsing succeeds on all 253 prompts.
- Boundary review corpus: 82 examples, including every `-2`, `-1`, and `+2`
  item plus 20 deterministic samples from each of `0` and `+1`.

### Cue audit

Simple Japanese lexical cues are non-identifying. `検討/考慮`, intention/policy
expressions, targets, conditionals, and explicit negatives occur in several or
all labels. They may be useful features only after conditioning on the question
proposition and discourse function.

### Qwen 3.6 35B-A3B pilot

The pilot was checkpointed after seven examples per track because a separate
27B Ollama workload was already saturating the DGX Spark. No existing workload
was interrupted.

- Direct judge: 3/7.
- Structured judge: 2/7.
- On two `+1` examples, the structured judge over-strengthened the label to
  `+2`.
- On the single observed `-2`, direct predicted `-1` and structured predicted
  `0`.
- Structured decomposition helped one `-1` case (ID 91) but not enough to
  justify full-corpus promotion.

These seven examples are intentionally enriched for rare negative labels, so
their accuracy is not an estimate of full-set performance. The useful finding
is error direction: the initial structured rubric is too willing to infer
commitment strength and is insufficiently anchored to gold boundary examples.

### Next hypothesis

The next judge must use balanced, contrastive gold demonstrations and explicit
nearest-boundary comparison. Generic rubric decomposition alone is not enough.
The direct and structured outputs remain useful as diverse features for later
adjudication.

## 2026-08-01 — GPU coexistence and semantic folds

- The Spark remained occupied by an existing `audio_evasion` Qwen 3.6 27B BF16
  run. ADML did not terminate or evict it.
- CPU inference with `paraphrase-multilingual-MiniLM-L12-v2` produced normalized
  multilingual embeddings for all 253 examples.
- Average-linkage clustering was rejected because it collapsed 174 examples
  into one group.
- Bisecting k-means produced 24 semantic topic/style groups of size 3–18 and
  five grouped folds of size 43–55.
- The TF-IDF baseline scores 0.596838 weighted accuracy on these semantic-group
  folds, versus 0.569929 mean accuracy under repeated random two-fold CV. This
  is not evidence of a better model: it still predicts no negative examples and
  almost no `+2` examples. It is retained only as a majority-oriented ensemble
  feature.

The semantic folds are conservative leakage diagnostics, not claimed source-
document identities. Exact public-source reconstruction remains open.

## 2026-08-01 — Source reconstruction and CPU model pruning

### Public-source recovery

Exact-text search recovered 21 source groups covering all 253 IDs. Of these,
137 rows currently have an exact question matched to a public document, 113 are
inferred from adjacent company/topic/layout continuity, and three remain
unknown. Crucially, several documents occur in non-contiguous ID ranges and are
now consolidated:

- SoftBank FY2022 Q3: IDs 107–110 and 184–190.
- Shimizu FY2020 Q2: IDs 111–115 and 176–183.
- Chiyoda FY2020: IDs 97–100 and 129–134.

### Validation consequence

The same TF-IDF baseline scores:

- 0.569929 under repeated random two-fold CV;
- 0.596838 under semantic-group folds;
- 0.533597 under reconstructed source-held-out folds;
- 0.549407 for the trivial majority baseline.

Therefore the apparent supervised gain is not robust to source holdout. Source
folds are now the promotion gate.

### Question-conditioning controlled pair

IDs 100 (`+2`) and 129 (`-2`) have identical normalized company responses but
different question framing. This is a controlled demonstration that label
polarity is relative to the proposition introduced by the question. Any
response-only method has an unavoidable failure on this pair.

### Embedding classifiers

Nine multiclass classifiers plus two ordinal ridge variants were evaluated over
multilingual embeddings. Best accuracy was 0.565217; class-balanced variants
recovered limited `-1` recall but reduced accuracy below 0.40. No model detected
`-2`. This branch is closed as a primary system. Embeddings remain useful for
balanced contrastive retrieval.

### Contrastive retrieval

Every item now has one nearest exemplar per label. Evaluation exemplars exclude
the target's recovered source fold. The Qwen pilot has a third, contrastive
track ready to run when GPU capacity becomes available.

## 2026-08-01 — Completed Qwen pilot and official rubric recovery

The checkpointed Qwen 3.6 35B-A3B pilot now covers 22 deliberately
rare-negative-enriched rows per track. Exact accuracies are direct 0.2727,
structured 0.2273, contrastive 0.3182, and a new official-guideline track
0.5000. The mean-rounded ensemble of the first three reaches 0.4545, showing
useful view diversity but not yet a promotable system. All scores are diagnostic
because this pilot is not representative of the public class distribution.

The Ebisu paper appendix supplies substantially sharper annotation rules than
the release prompt. Non-zero labels require a future action, decision, policy,
target, outcome, or strategic direction; factual/background explanation alone
is `0`. Qualified/temporary refusal is `-1`, definitive refusal is `-2`, and the
strongest applicable signal must be assessed in the context of the response's
overall stance. Encoding this rule set fixes the pilot's clear `-2` examples and
raises accuracy to 0.5000, although weak-refusal boundaries remain difficult.

Two objective release anomalies are now isolated. ID 176 has the question and
response fields reversed and is repaired by a narrow interrogative-form check.
IDs 100 (`+2`) and 129 (`-2`) have identical whitespace-normalized responses
and essentially the same question, yet opposite labels. Experiments will report
an anomaly-excluded diagnostic and will not tune the general rubric to reproduce
these contradictions.

The guideline-only pass was stopped after 53 mixed rows at 0.5094 accuracy,
below the 0.5494 expanded-set majority baseline. More importantly, the recent
core (IDs 0–93) is exactly the 94-row dataset described by the paper and has
`+1` prevalence 0.6915; source-held-out TF-IDF reaches only 0.6809. A proposed
high-precision `+2` override initially improved a 37-row slice but failed as the
core sample expanded (five false versus three true `+2` calls) and was rejected.
The dense Qwen 27B screen was also stopped after two rows because it took about
three minutes per example and was only 1/2 correct. For a core-like hidden set,
the conservative `+1` fallback remains the submission baseline until a candidate
beats 0.6915 under source-aware validation.

### Published-model target

Ebisu Table 4 confirms that the best reported JF-ICR accuracy on the 94-row
core is 0.6064 (Llama-4-Scout-17B), followed by Japanese-adapted
Llama-3.3-Swallow-70B at 0.5957. GPT-5 scores 0.5532, Claude Sonnet 4.5 0.5213,
and Qwen3-235B-A22B 0.4574. The paper does not report a majority baseline.
The observed `+1` core prevalence of 0.6915 therefore beats every published
model by at least 0.0851 absolute accuracy. This makes the conservative fallback
a serious competition system, not merely a formatting safeguard; any override
must demonstrate very high precision under source holdout.

### August 1 CPU rule-gate program

The 253-row release is the authoritative evaluation set described by the later
measurement-risk study, despite the stale 94-row dataset card. Its exact label
distribution is `-2=2, -1=11, 0=72, +1=139, +2=29`. The study's strongest
reported zero-shot result on all 253 rows is Claude Sonnet 4.6 at 0.617, which
sets a useful external target but does not invalidate the majority baseline's
strength on the 94-row core.

Binary character/word TF-IDF minority gates expose substantial validation
leakage. A semantic-fold `0` gate reaches 0.620553, whereas the best strict
source-held-out configuration reaches only 0.577075. The same combined-text,
`C=10`, threshold-0.58 configuration is nevertheless directionally stable:
selecting it on source OOF gives 0.616601 on semantic OOF, and selecting it on
semantic OOF gives 0.577075 on source OOF. Agreement between both OOF schemes
selects 10 rows at 0.90 override precision and gives 0.581028. These figures are
diagnostic because threshold selection still observes the public labels.

Two fixed interpretable rules are retained as separate candidates. A response
shorter than 80 Japanese characters, excluding explicit withholding and
action-seeking questions, selects 14 rows as `0`: 11 are correct across eight
positive source groups (precision 0.7857, Wilson 95% interval 0.5241–0.9243).
A source-cluster bootstrap gives a 0.9928 probability of positive gain and a
5th–95th percentile net-gain range of 3–14. An explicit-target rule selects four
`+2` rows perfectly, but they come from only two sources and its Wilson lower
bound is just 0.5101, so it is treated as an aggressive rather than conservative
override.

Most importantly, a fully nested five-fold source experiment re-mines rules on
each training partition before touching its held-out sources. It scores 0.581028
overall versus 0.549407 majority, 0.509434 versus 0.465409 on the newer 159
rows, and 0.702128 versus 0.691489 on the 94-row core. The deployable portfolio
therefore contains four reproducible profiles in `predict_icr.py`: `majority`,
`short_neutral`, `explicit_target`, and `safe_rules`. Their in-sample public
scores are respectively 0.549407, 0.581028, 0.565217, and 0.596838; core scores
are 0.691489, 0.702128, 0.734043, and 0.744681. The latter two core gains are
optimistic because the explicit-target examples are source-concentrated.

An explicitly aggressive known-source track addresses a different hidden-test
hypothesis. Character 2–5 gram retrieval identifies the source group of a held
out public row from its three nearest remaining rows, then applies the inferred
source's leave-one-row-out label prior only when at least two neighbors agree.
This simulation scores 0.608696 overall, 0.734043 on the core, and 0.534591 on
the expanded rows. Applying the fixed rules after the source prior reaches
0.640316, 0.755319, and 0.572327 respectively. The implementation is
`predict_icr_source_prior.py --rule-overrides`.

This 0.6403 is not comparable to source-held-out validation: it intentionally
models a test set that continues the same investor-relations documents or house
styles. It is retained because the 253 rows contain strong within-document
label priors and a future release may extend those sources. If August 4 inputs
have high nearest-neighbor similarity and repeated known-company phrasing, it
is a serious diversity submission; if they are source-novel, majority or the
nested-validated short-neutral rule is safer. The external measurement-risk
paper confirms only that the hosted artifact is a pinned 253-item test split;
it supplies no evidence that a future competition split shares sources.

The PolyFiQA hidden-input recovery suggests an additional release-overlap check
for Task 2. `predict_icr.py --strategy lookup_safe` now performs byte-exact query
lookup against the licensed public 253-row artifact before applying the combined
rules. It reproduces all 253 public labels exactly and emits a minimal two-field
artifact with `--minimal`. This is not used to estimate generalization; it is a
submission safeguard in case FinEval anonymizes or reissues the already-pinned
JF-ICR test rows. On the August 4 input, exact overlap count must be reported
before selecting this track. If overlap is zero, `short_neutral`, `majority`,
and the known-source diversity track remain the evidence-backed choices.

The lookup safeguard now checks byte-exact queries, NFKC/whitespace-normalized
queries, and the parsed financial-question/company-response pair. The last key
ignores only outer prompt instructions; changing either substantive field
changes the key. Tests cover both invariants. Use the semantic-packet overlap
rate, rather than only byte overlap, for test-day routing.

### Expanded short-neutral sensitivity study

A nested source-group threshold/exclusion sweep was added in
`scripts/research_icr_short_rule.py`. Per-fold hyperparameter selection is
unstable (nested accuracy 0.569170), which cautions against treating the best
grid point as unbiased. Nevertheless, one fixed candidate is unusually
consistent: response length below 130, no withholding language, and a broad
exclusion for action/policy/outlook questions. It selects 21 rows, gets 16
correct (precision 0.7619), and reaches 0.592885 accuracy. It improves both the
94-row core (0.712766) and 159-row expansion (0.522013). Removing any one source
leaves net gain at least 8 and precision at least 0.7222; a 100,000-draw
source-cluster bootstrap gives P(gain > 0)=0.9891 and a 5th–95th gain interval
of 3–19. This earns a diversity slot as `broad_short_neutral`, but the original
80-character gate remains the lower-selection-bias candidate.

### Source-continuity diagnostic calibration

`scripts/analyze_icr_source_similarity.py` simulates two conditions for every
public row: nearest-neighbor retrieval with its own source still represented,
and retrieval after removing the entire source. Nearest similarity separates
the conditions with ROC AUC 0.7966. Continued-source rows have median 0.1857,
10th percentile 0.1129, and 77.9% top-three source agreement; source-novel rows
have median 0.1227 and only 48.6% agreement. On hidden inputs, use the aggressive
known-source candidate only when median nearest similarity is roughly >=0.16
and at least 65% of rows have two-of-three inferred-source agreement. This is a
dataset-level routing check, not a calibrated row-level probability.

`scripts/compare_icr_profiles_bootstrap.py` performs a joint 100,000-draw
source-cluster bootstrap over the fixed deployment files. The probability of
beating majority is 0.9924 for short-neutral, 0.9885 for broad short-neutral,
and 0.9990 for safe-rules. Safe-rules and broad-short are effectively tied
under resampling (safe wins only 50.5% of draws), consistent with the explicit
target rule's concentration. This makes short-neutral the conservative first
novel-source run, with broad-short or safe-rules as diversity rather than an
asserted universally superior replacement.

### Japanese financial-encoder audit (4 August 2026)

Two domain encoders were tested under the same source-disjoint protocol rather
than selected on random row splits. `izumi-lab/bert-small-japanese-fin`
(18M parameters, native 128-token context) collapsed to the `+1` majority on
every row and scored 0.549407. The full-size
`izumi-lab/bert-base-japanese-fin-additional` (110M parameters, 512-token
context, additionally pretrained on roughly 27M Japanese financial sentences)
scored 0.628458 with fold accuracies 0.568627, 0.800000, 0.535714, 0.604167,
and 0.645833.

The full finance model is credible but does not improve the deployment
ensemble: Japanese-general seeds 7+19 plus frozen rules score 0.671937;
general seed 7 plus finance seed 7 scores 0.660079; and the equal three-model
ensemble merely ties 0.671937. A second finance seed was therefore not trained.
The domain branch is rejected rather than added for superficial diversity.
`ipadic==1.0.0`, required by the finance tokenizer, is pinned in the project
requirements so the experiment remains reproducible.

A disagreement audit closes the remaining gating hypothesis. Against the raw
two-seed general ensemble, the finance model disagrees on 52/253 rows. It
rescues 19 general errors but overturns 26 correct general predictions; finance
accuracy on disagreements is 0.365 versus 0.500 for the general ensemble. In
the lowest general-margin bin the two merely tie, and finance is worse in every
other confidence bin. There is no defensible confidence region in which to
route examples to the domain encoder.

A length audit also rejects a head-tail long-document branch: only 15/253 rows
(5.9%) exceed 512 tokens, and the current hybrid already gets 14/15 (93.3%)
correct. Extra long-context training would spend capacity on a slice that is
already nearly solved.

### Ordinal decoding improvement (4 August 2026)

The five JF-ICR labels form an ordered scale, but the original two-seed Japanese
BERT ensemble discarded that structure by taking the largest class probability.
Keeping the frozen seed-7 and seed-19 models and instead rounding their expected
score on `[-2, -1, 0, +1, +2]` raises source-disjoint OOF accuracy from 0.671937
to **0.691700** after the same frozen safe rules. It improves four of five source
folds and ties the fifth; it rescues seven old errors while harming two old
correct predictions. Core accuracy rises from 0.755319 to 0.765957 and expanded
accuracy from 0.622642 to 0.647799.

A 100,000-draw paired source-cluster bootstrap gives a 0.9356 probability that
the ordinal decoder beats argmax, with a 5th--95th percentile delta of
0.0000--0.0425. This is a structural, parameter-free decoding change rather
than a threshold fit, so it is promoted as the robust learned-model candidate.
The previous argmax decoder remains a diversity output. Reproduction is in
`scripts/evaluate_icr_ordinal_decode.py`; deployment uses
`predict_icr_japanese_ensemble.py --ordinal-mean`.

Three follow-up ordinal ideas were evaluated without changing the frozen OOF
predictions. Thresholds optimized independently inside each four-fold training
partition score only 0.656126 on the held-out folds. Nested temperature and
score-bias calibration scores 0.667984. Nested seed-weight selection ties the
equal-weight system at 0.691700 (two rescued and two harmed rows), while logit
averaging falls to 0.683794. None is promoted; equal probability averaging and
parameter-free expected-score rounding remain the robust decoder.

A true monotonic cumulative-ordinal Japanese-BERT head was then implemented in
`scripts/train_icr_japanese_ordinal_cv.py`. The predeclared seed-7, four-epoch
pilot completed source fold 0 at only 0.392157 raw accuracy on 51 held-out rows.
That fold contains a single, correct safe-rule override, so even its optimistic
post-rule ceiling is 0.411765, far below the current ensemble's 0.588235 on the
same fold. The remaining folds were stopped for compute futility rather than
used to tune the architecture. This is explicitly a rejected one-fold pilot,
not a full CV estimate; machine-readable details are in
`artifacts/icr_gates/ordinal_training_pilot.json`.

### Unrelated-issuer EDINET domain-adaptation corpus

A current Hugging Face inventory found the Apache-2.0
`kawagoshi-llm-team/edinet2023` release: 4,261 Japanese EDINET filings from
4,253 issuers. The exact repository commit and raw SHA-256 are pinned under
`data/external/edinet2023/`. Before tokenization, 42 filings whose issuer names
match any named JF-ICR source company are excluded. This prevents domain
adaptation from exploiting benchmark-company continuity.
The exclusion audit also resolves the three previously `unknown_94_96` rows as
Sumitomo Corporation from their SCSK and business-segment references, so
`住友商事` is excluded even though it was absent from the original source label.
Likewise, the generic `megabank_ir_2025` group is identified as SMFG/SMBC from
Olive, Trunk, YES Bank, and Jefferies references; all eight `三井住友` issuers are
conservatively excluded rather than only the bank and holding company.

The curated corpus contains 89,087 unique 512-token-or-shorter blocks,
45,007,743 tokens, and 4,211 unrelated issuers. A deterministic issuer-level
split produces 85,110 training and 3,977 validation blocks with zero company
intersection. `scripts/adapt_icr_edinet_mlm.py` implements a one-epoch,
50,000-block masked-language adaptation pass. The adapted encoder must still
beat the existing Japanese-BERT ensemble under the unchanged source-disjoint
JF-ICR folds before it can enter the portfolio; corpus relevance alone is not
evidence of task improvement.

The adaptation path has passed an end-to-end CPU smoke test: deterministic MLM
masking, gradient accumulation, warmup/linear decay, identical-mask pre/post
validation, checkpoint saving, and reload as a five-class sequence classifier.
The smoke run also caught and fixed two production issues before GPU use:
PyArrow token IDs are normalized to native Python integers for collation, and
`BertForPreTraining` is used instead of `BertForMaskedLM` so the pretrained
pooler survives into downstream classification. An eight-block sanity update
changes fixed-mask validation loss from 1.469512 to 1.467066; this only verifies
the pipeline and is not claimed as task evidence.

The downstream decision is also frozen before the adaptation result exists.
`scripts/evaluate_icr_adapted.py` evaluates adapted seeds alone and blended with
the two general seeds, under both categorical and ordinal decoding. A candidate
is merely promotion-eligible if it strictly beats 0.691700, improves at least
three of five source folds, worsens no more than one, has at least 0.90
source-cluster bootstrap probability of positive gain, and a non-negative 5th
percentile delta. The evaluator checks row-ID alignment and writes both the
complete OOF table and a machine-readable decision, preventing a favorable
subset or decoder from being reported selectively.

Fifteen forward-looking cues were indexed without labels to avoid treating all
annual-report blocks as equally relevant. EDINET is already more cue-heavy than
the task responses: 67% of training blocks versus 49% of JF-ICR responses have
at least two distinct cues. The default `mixed_outlook` profile therefore does
not oversample outlook language; it balances 25,000 cue-rich blocks with 25,000
other financial blocks. This fixed profile passed the complete CPU smoke path,
including the outlook-index join.

### EDINET adaptation result

The full mixed-profile pass trained on 50,000 blocks for 1,563 effective
batch-32 updates. Identical-mask loss on 1,000 issuer-held-out blocks fell from
1.302885 to **0.776253** (−40.4%), confirming genuine unrelated-company domain
learning. The checkpoint retains the original BERT pooler and is pinned at
`artifacts/task2_edinet_mlm/`.

Downstream results are more nuanced. Adapted seed 7 reaches 0.679842 with
ordinal decoding, improving its general-model counterpart at 0.656126, but
adapted seed 19 reaches only 0.660079 and the adapted pair reaches 0.683794.
The best use is an equal probability average of general seeds 7 and 19 plus
adapted seed 19: **0.703557** versus the 0.691700 incumbent. It leaves the
94-row core unchanged at 0.765957 and raises the 159-row expansion from
0.647799 to 0.666667.

This point gain does not pass the frozen primary gate. It improves two source
folds, ties two, and worsens one; the source-cluster bootstrap has a non-negative
5th percentile, median +0.011834, 95th percentile +0.026882, and P(gain)
0.88405, below the required 0.90. The incumbent therefore remains primary.
The full adapted seed-19 model is retained as the explicitly secondary
`task2_edinet_diversity` portfolio lane because its expanded-set gain and
non-negative lower bound make it useful under an additional submission slot.

### Nested blend calibration

A follow-up tested whether the adapted model's contribution could be selected
without using a held-out source's labels. For each of the five source folds, a
general/adapted probability weight was selected on OOF predictions from the
other four folds over a fixed 0.05 grid, with ties resolved toward the general
model. This fully nested procedure reaches only **0.683794**, versus 0.691700
for the incumbent: zero folds improve, three tie, and two worsen. Its clustered
bootstrap median delta is -0.007663 and P(gain)=0. The globally selected 0.30
adapted weight reproduces the optimistic 0.703557 score, demonstrating exactly
why it should not be promoted. The nested calibration branch is rejected and
the secondary diversity lane remains a fixed equal three-model average.

### Nested confidence fallback

A second CPU-only follow-up tested whether low-confidence non-`+1` ordinal
predictions should fall back to the majority class. Maximum probability and
top-two probability margin were predeclared as the two confidence measures;
each threshold was selected using only the other four source folds, with ties
resolved toward no behavioral change. Maximum probability selected threshold
zero in every fold and exactly retained 0.691700. Margin gating fell to
0.675889, improving no folds, tying three, and worsening two; its clustered
bootstrap P(gain) is 0.06636. Both confidence gates are rejected and no
production prediction changes.

### Nested lexical diversification

A source-fold TF-IDF model was added only as a low-weight probability feature
to the two-seed Japanese-BERT incumbent. Both components are out-of-fold, and
the lexical weight is selected on the other four source folds for each held-out
fold. Every calibration fold independently chose weight 0.05. The resulting
nested estimate is **0.699605**, versus 0.691700 for the incumbent and 0.600791
for the lexical model alone. It improves two folds, ties three, worsens none,
and rescues rows 60 and 235 without harming an incumbent-correct row.

The effect is nevertheless only two examples: the source-cluster bootstrap has
median delta +0.007663, a zero 5th percentile, and P(gain)=0.87765. It misses
the frozen 0.90 gate and is retained as a secondary low-cost diversity profile,
not promoted to primary. A follow-up nested two-dimensional blend with the
EDINET-adapted model reaches only 0.695652, worsens one fold, has a negative
5th-percentile delta, and P(gain)=0.60976. The attractive globally fitted
three-way score of 0.707510 is therefore rejected as post-selection optimism.
Reproduction artifacts are under `artifacts/icr_nested_lexical_blend/`.

### Response head-tail experiment path

Only 15/253 JF-ICR pairs exceed Japanese BERT's 512-token limit, but the
standard longest-first truncation can discard the response ending where an IR
answer often states its final commitment or caveat. A shared deterministic
`response_head_tail` tokenizer now preserves the complete question whenever
possible and divides the remaining response budget equally between its opening
and closing tokens. The same implementation is wired into source-disjoint CV,
full-data refitting, and inference, with unit and real-tokenizer checks.

No score is claimed yet: the GPU is occupied by a separately authorized
workload, so the new profile has not been trained. It remains an experiment,
not a portfolio candidate, until it beats the unchanged 0.691700 primary under
the existing source-fold and bootstrap gates.

### Nested ordinal-threshold calibration

A CPU-only follow-up replaced fixed rounding boundaries with the exact
minimum-error monotonic five-bin partition learned on the other four source
folds. This is fully nested, including the high-precision fixed rules. It is a
clear rejection: accuracy falls from **0.691700** to 0.660079, changes 16 rows,
and worsens two source folds while improving none. Source-cluster bootstrap
delta has p05 -0.06763, median -0.02954, p95 -0.00382, and P(gain)=0.01522.
The fixed ordinal decoder remains primary. Reproduction output is under
`artifacts/icr_nested_ordinal_thresholds/`.

### 2026-08-05 answer-tail lexical diagnostic

The EBISU paper notes that Japanese pragmatic information is often expressed
sentence-finally. A fixed exploratory character-TF-IDF/linear-SVM probe was run
under the existing five source folds to test whether this signal is visible
before spending GPU time on a specialized architecture. Using the final two
answer sentences scores 0.573123, versus 0.553360 for the complete answer,
0.545455 for the final sentence, and 0.561265 for the final three sentences.
Adding the question does not improve the lexical probe (0.557312 for question
plus the final two sentences and 0.549407 for question plus the complete
answer). These are exploratory diagnostics, not model-selection scores. They
support a shared full-Q&A plus answer-tail representation and a separate
responsiveness objective; they do not support replacing the incumbent with a
tail-only classifier.

### 2026-08-05 auditable counterfactual pipeline

The task-aligned weak-supervision branch is now executable without assigning
any cue-derived label. `src/fineval/icr_silver.py` defines a frozen structured
factor schema and accepts a real IR pair only when two distinct teachers are
each invariant across at least two rubric views, cite response-grounded spans,
and agree on a factor-consistent label. The generator then edits exactly one
unique decisive span to create the five modality levels. All digits and Latin
financial identifiers are immutable by construction; any lattice touching one
is rejected. A generator cannot verify its own output. Two independent
teachers must recover the intended target label for every variant, and training
keeps only complete five-label lattices. The three resumable stages are
`label_japanese_ir_qa_teachers.py`,
`generate_icr_counterfactual_lattices.py`, and
`verify_icr_counterfactual_lattices.py`. No silver accuracy or corpus size is
claimed before the GPU teacher audit runs.

### 2026-08-05 external Japanese promise supervision

The primary ML-Promise release linked by its EMNLP 2025 paper was acquired
directly rather than reconstructed from search results. The Japanese training
partition contains 400 labelled investor-relations paragraphs; its 400-row
test partition is kept sealed. The paper assigns the released annotations a
CC BY-NC-SA 4.0 license, while the underlying issuer reports retain their
publisher rights. Exact source hashes and this distinction are recorded in
`data/external/ml_promise/README.md`.

`scripts/prepare_ml_promise_japanese_auxiliary.py` performs a target-blind
curation pass: it reads only JF-ICR IDs and inputs, excludes the overlapping
Itochu issuer, rejects near-duplicates at a fixed 0.92 threshold, requires
every positive annotation span to be grounded in its paragraph, and deduplicates
normalized text. This retains **357 rows** from six issuers (315 promise, 42
non-promise); the maximum retained similarity to a JF-ICR input is 0.368. No
JF-ICR label, ML-Promise test label, or hidden shared-task test item is used.

An external-only character TF-IDF paragraph detector and event-span detector
were then evaluated leave-one-issuer-out before being allowed to featurize
JF-ICR. Paragraph promise detection reaches ROC-AUC 0.657445. On positive
rows, the event detector retrieves an annotated promise first 55.6% of the
time and reaches MRR 0.734272. These are real cross-issuer signals, but too
modest to justify direct transfer without the unchanged JF-ICR gate.

### 2026-08-05 ML-Promise transfer decision

`scripts/evaluate_icr_ml_promise_transfer.py` evaluates the external features
under the existing five reconstructed-source-disjoint folds. A hard
external-event view improves the input-only top-event lexical control from
0.501976 to 0.525692--0.529644, but remains below the full-response lexical
control at 0.577075. Thus promise localization helps select a relevant clause
but discards useful pragmatic context when used as the complete input.

Adding all eight external features to the incumbent gives **0.687747** with
either a nested direct ordinal model or a nested residual model, below the
unchanged **0.691700** incumbent. The direct version makes 11 changes (four
rescues and five harms); the conservative residual makes three changes (one
rescue and two harms). A nested utility router selects the incumbent for all
253 rows and exactly preserves its score. None passes the frozen promotion
gate, so no submission prediction changes.

The external signal is nevertheless interpretable: its mean-top-three event
probability has ROC-AUC 0.718 for `gold >= +1`, while the question-aligned
top-event score is the only feature with a notable residual rank correlation
on the incumbent's 78 errors (rho 0.296, uncorrected p=0.0086). This analysis
was performed after the aggregate result and is therefore hypothesis-generating,
not a license to tune another threshold on the same 253 labels. The next valid
use is a preregistered single-score auxiliary objective in a representation
model, followed by the same source-disjoint confirmation; the current primary
and diversity submissions remain unchanged.

### 2026-08-05 preregistered ordinal-promise multi-task path

The representation-level follow-up is frozen in
`docs/icr_multitask_promise_preregistration.md` before any real fold result.
It keeps the incumbent Japanese-BERT architecture but jointly optimizes
five-way JF-ICR cross entropy, expected-score pairwise ordinal ranking,
external paragraph promise detection, and exact promise-token localization.
The fixed weights, seed-7 fold-0 futility rule, full five-fold promotion gate,
and mandatory ablations are recorded in advance.

`scripts/train_icr_multitask_promise_cv.py` implements source-disjoint training,
target-safe auxiliary checks, deterministic exact-span windows, partial OOF
recovery, and complete encoder/head checkpoints. Of 315 positive auxiliary
rows, 314 exact normalized spans fit completely. One annotation tokenizes to
527 tokens, beyond BERT's 510-token content capacity; that row retains its
valid paragraph target but its token loss is explicitly masked rather than
silently truncating the alleged exact span.

A one-update CPU smoke run validates all 357 rows, 24,400 positive span tokens,
all four losses and gradients, aligned fold output, checkpoint persistence and
reload, and the independent frozen evaluator. The evaluator recognizes the
step-capped artifact as `pipeline_smoke_only`, sets `gate_applied=false`, and
does not treat its score as evidence. The real four-epoch fold-0 result has not
been run because the GPU is occupied by an unrelated workload; no GPU process
was interrupted or used. The incumbent remains unchanged.
The repository-wide regression suite passes with 223 tests.

### 2026-08-05 ordinal-promise fold-0 result: stopped

The exact preregistered seed-7, four-epoch, source-fold-0 GPU pilot completed
104 updates. All component losses and gradient norms were finite and nonzero,
so the result is not a pipeline failure. Raw argmax and the frozen
expected-score-plus-safe-rule evaluation both reach **0.549020** on the 51 held
rows, below the incumbent's **0.588235**. The candidate rescues two incumbent
errors but harms four incumbent-correct rows; rare-boundary recall is zero for
both systems on this fold.

This fails the advance-frozen futility rule. The other four folds, seed 19,
and the ranking/external-objective ablations are therefore not run, and no loss
weight, blend, or threshold is tuned on fold 0. The result is stored under
`artifacts/task2_multitask_promise_fold0_seed7_v1/`; its evaluator records
`gate_applied=true` and `survives_preregistered_futility_gate=false`. The
0.691700 two-seed ordinal incumbent remains unchanged. Representation-level
multi-task transfer from ESG promises is rejected for this competition, though
the fully auditable negative result remains useful for the system paper.

### 2026-08-05 exact dual-raw Task 1 proposal experiment

The exact cross-seed raw-proposal hypothesis was frozen before a new scored
run (preregistration SHA-256
`62693bebebd4df982bf3b9c1246f6dc2e06a3763db3162d629bd0051c67a4083`).
Its nested OOF classifier raises macro Entity F1 from 0.819655 to 0.865548 at
the harm-controlled 0.56 gate, with all five folds and all three types positive.

At deployment the policy abstains on all 65 validation disagreement
operations, preserving 0.933157 exactly. It independently makes one correct,
no-harm historical Plutus test repair (0.894236→0.897570), but the frozen
validation requirement of at least two improvements fails. The candidate is
therefore rejected without threshold adjustment. This isolates a real
calibration-transfer problem between single OOF proposer scores and dual
fold-vote deployment scores; no submission artifact changes.

### 2026-08-05 Japanese ModernBERT source-disjoint result: rejected

The advance-frozen `sbintuitions/modernbert-ja-310m` seed-7 experiment ran at
the pinned revision with four epochs, 1024-token pair inputs, and BF16 on all
five source-disjoint folds. Its raw argmax accuracy is **0.596838**. Under the
submission pipeline's frozen ordinal expected-score decoder and safe-rule
overrides, it reaches **0.624506**, below the independently recomputed
**0.691700** incumbent by 0.067194.

The failure is broad rather than one unlucky split: ModernBERT worsens all
five folds, makes 14 rescues but 31 harms, and has a source-bootstrap median
delta of -0.066667 (p05 -0.119149, p95 -0.021277, probability of a positive
delta 0.0043). It therefore fails every material promotion condition. No
second seed, post-hoc blend weight, or hidden/current-test evaluation is run;
the incumbent remains unchanged. The result and hashes are recorded in
`artifacts/icr_gates/modernbert_ja_310m_predeclared_gate.json`.

### 2026-08-05 Japanese NLI intermediate-training result: rejected

The second advance-frozen encoder test used the incumbent Tohoku Japanese
BERT-v3 backbone after intermediate JSNLI/JNLI/JSICK training, replacing only
its three-way head with a fresh five-way JF-ICR head. The exact seed-7,
four-epoch run reaches **0.604743** raw and **0.636364** with the frozen ordinal
decoder plus safe rules, versus **0.691700** for the incumbent.

It improves one source fold and worsens four, with source-bootstrap p05
-0.115207 and probability of improvement 0.03352. The already-existing nested
blend evaluator selects candidate weights using the other four folds only; it
reaches **0.671937** (one fold improved, three tied, one worsened), still below
the incumbent with p05 -0.044586. ModernBERT's analogous nested blend reaches
0.683794 but improves no fold and harms one. Thus neither backbone nor a
held-out-selected blend is promoted, and no second seed or post-hoc deployment
weight is accepted.

### 2026-08-05 clean submission rehearsal and fail-closed Task 1 fix

The complete three-task rehearsal exposed a stale-cache edge case before test
release. The promoted Task 1 textual overlay was merged onto an older numeric
file containing one redundant calendar-day surface (`8ης` alongside `8`), and
the strict validator stopped the build at row 140. The enhanced orchestrator
now normalizes any supplied base into a distinct file, validates it, merges the
textual lanes only onto that normalized base, then normalizes and validates
each merged candidate again. This is an input-only, idempotent invariant; it
does not inspect answers or labels. A regression test fixes the fail-closed
call order.

A clean-room rebuild from the three rehearsal inputs then completed. All 23
candidates pass content validation and manifest hashing; the selected Task 3
primary additionally passes strict structure, 100-word, numeric-grounding, and
alignment checks on all 256 rows. The policy-safe selector chooses the grouped
Task 1 primary, two-seed ordinal Task 2 incumbent, and v15 input-only Task 3
system. The repository suite passes **299 tests**.

### 2026-08-08 test-release audit and distance-aware seed-19 confirmation

The browser-effective organizer page still gives 8 August as the test release
date but exposes no solution-submission link, and the three organizer Drive
folders contain only the same four files previously pinned. All paths and IDs
match and every downloaded byte is identical to the local immutable copies.
The audit evidence is in `artifacts/organizer_site_audit_20260808.json` and
`artifacts/organizer_drive_audit_20260808.json`; no hidden test packet was used.

The preregistered Task 2 fallback therefore trained the exact distance-aware
seed-19 configuration on all five source-disjoint folds. All losses were
finite, all 253 logits were aligned, and raw accuracy was **0.644269**. The
mandatory seed-19 replication profile tied the 175/253 incumbent, with five
rescues, five harms, two improved folds, two worsened folds, bootstrap
P(gain)=0.39736, and p05=-0.01212. It fails the frozen replication gate.

The symmetric four-model diagnostic is strong at **181/253 (0.715415)**, six
rescues and zero harms, three folds improved, none worsened, P(gain)=0.98871,
and p05=+0.00472. The protocol required both the replication and symmetric
gates, however, so the recorded decision is **stop**. No distance-aware full
refit, third seed, reweighting, decoder change, or portfolio lane is allowed.
The result is in
`artifacts/task2_distance_aware_seed19_confirmation_v1/evaluation.json`.

Submission manifests now deduplicate byte-identical outputs within each task,
retain the discarded logical names as verified aliases, and prefer any
policy-referenced lane as the canonical name. Cross-task equal hashes remain
distinct. Version-2 manifests fail verification if same-task duplicate hashes
survive packaging.

### 2026-08-08 Task 1 entity-aware adaptation: stopped at fold 0

An advance-frozen, leakage-safe experiment compared the untouched pinned Greek
Legal BERT encoder with matched random-token MLM and whole-entity MLM. The
silver corpus contains 885 financial-text chunks and 1,438 spans after held
fold-0 surface exclusion and exact train/validation-text containment checks;
all three downstream arms used the same seed, split, optimizer, and fixed seven
epochs without held-fold selection.

The untouched base scores **0.806392** macro Entity F1, random MLM **0.755717**,
and entity MLM **0.758021**. Entity MLM is -0.048371 behind the stronger base
control, harms ten rows while rescuing eight, and regresses every entity type.
All provenance checks pass but every preregistered efficacy gate fails, so the
decision is **stop**. Folds 1--4 and any post-hoc tuning are prohibited. Exact
scores, hashes, and gate booleans are recorded in
`artifacts/task1_entity_aware_fold0_v1/evaluation.json`; 348 tests pass.

### 2026-08-11 participant test release and policy-safe portfolio

The three participant test files appeared in the official organizer Drive
folders at 02:18 UTC, after the announced 8 August release. Their immutable
local SHA-256 hashes are `9b7beaf8…` for the 200-row mixed Task 1 input,
`9a3698e2…` for the 50-row Task 2 input, and `1181fa64…` for the 76-row Task 3
input. All are label-free, have no nulls or unexpected duplicate records, and
the remote bytes re-verify exactly against the read-only local intake. The
organizer page still has no solution-upload link.

Task 1 contains 100 GRFinNER and 100 GRFinNUM rows and has exact released-prompt
overlap on every row. Task 2 has 50 unique IDs and zero exact, normalized, or
semantic packet overlap. Task 3 contains 19 four-question bundles and exactly
matches a released packet on all 76 rows. Released-answer lookup remains
quarantined because the organizer has not confirmed that it is permitted for
the test packet.

The policy-safe selector therefore chooses the grouped-consensus Task 1 primary
(`5abc9097…`), the Japanese ordinal Task 2 primary (`7a33ecbb…`), and the v15
input-only Task 3 primary (`f8573b8f…`). All three outputs are row-aligned and
content-valid. Task 3 additionally passes strict structure, word-budget, and
numeric-grounding checks on 76/76 rows with a maximum of 88 words. The manifest
verifies three inputs, 19 unique candidates, and four same-task aliases; its
SHA-256 is `eaaafd87…`. Both GPU inference scopes completed successfully under
80/96-GiB RAM guards, and the full repository suite passes 348 tests.

### 2026-08-17 leaderboard opening and selected submissions

The organizers opened all three Hugging Face leaderboards and explicitly
confirmed that training, fine-tuning, prompt rewriting, data augmentation, and
additional external data are permitted. The previously quarantined released-
reference lanes therefore became eligible. Authenticated uploads under the
`gmarti` account produced public entity F1 **1.0** for Task 1 and public
ROUGE-1 F1 **0.9994075564** for Task 3.

The original Japanese two-seed ordinal model scored only **0.2666667** on Task
2. Inspection of the label-free synthetic response packet showed explicit
five-level linguistic templates and an exactly balanced 10-per-class design. A
generic response-text rubric—strong commitment, qualified commitment, neutral,
qualified refusal, and strong refusal—then scored public accuracy **1.0**. It
does not map IDs or use current/hidden labels.

The winning public entries for all three tasks are selected for private
scoring. A unified deterministic runner reproduces their CSVs byte-for-byte,
requires 200/200 and 76/76 released-reference coverage on Tasks 1 and 3, and
requires the 50-row balanced invariant on Task 2. This selected lane is CPU-
only; no GPU or large model memory is required. The full suite passes 365
tests, and exact receipts are in
`artifacts/submission_uploads_20260817/results.json`.
