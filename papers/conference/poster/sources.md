# Evidence for poster claims

This is a focused source map for the poster. Relative links point to the papers, research records, and aggregate evidence included in this public release.

| Poster claim | Evidence | Interpretation |
|---|---|---|
| Author names, affiliations, accepted title | [QA manuscript](../../task3/main.tex) | Match the accepted manuscript. |
| 1st NER, tied 1st ICR, 2nd QA; private scores | [Paper overview](../../README.md), three manuscript results sections | Different task metrics; do not compare their magnitudes as a common score. |
| Agent-led research process | Authors' description and [research log](../../../docs/research_log.md) | No exact agent model, autonomy percentage, person-hours, or productivity claim is made. |
| 209 scripts, 32 modules, 99 test files, 25 preregistration documents; 48,468 Python lines | `build.py` reads Git tree `557bf382e8e64a6742f117213419cff67484c18e` | 38,913 script lines + 4,622 module lines + 4,933 test lines. Physical lines include blanks/comments. Counts exclude the new conference files. |
| Greek model trials and method families | [Task 1 research log](../../../docs/task1_research_log.md) | Greek BERT, legal encoders, XLM-R and GLiNER were evaluated. |
| Japanese model trials and method families | [Research log](../../../docs/research_log.md) | Tohoku BERT, Rinna RoBERTa, ModernBERT and Japanese Nemotron 9B were evaluated. Not every trial completed full cross-validation. |
| QA model trials | [Task 3 log](../../../docs/task3_research_log.md), [general log](../../../docs/research_log.md), [57B compliance amendment](../../../docs/polyfiqa_large_content_planner_compliance_amendment.md) | Eva-4B adapter; Qwen 27B/35B realization pilots; 35B/57B fact-planning screens. A staged model is not counted as evaluated merely because it was downloaded. |
| Grounded compiler and sparse-edit oracle | [QA paper](../../task3/main.tex), [oracle summary](../../../artifacts/polyfiqa_sparse_edit_oracle_v1/summary.json) | 192 historical rows; baseline 0.3652420; oracle 0.4029502; delta 0.0377081. Oracle observes references and is not deployable. |
| Nested selector unchanged; temporal selector harms 4/4 changes | [Task 3 research log](../../../docs/task3_research_log.md), [QA paper](../../task3/main.tex) | Separate historical evaluation protocols, not private leaderboard results. |
| Entity-aware adaptation 0.8064 → 0.7580 | [Fold-0 evaluation](../../../artifacts/task1_entity_aware_fold0_v1/evaluation.json) | Macro entity F1; all three entity types regress versus base; gate fails. |
| ModernBERT 0.6917 → 0.6245; all folds worse; 14 corrections/31 new errors | [Research log, Japanese ModernBERT source-disjoint result](../../../docs/research_log.md) | Five source folds, frozen decoder and rules; replacement rejected. |
| Qwen 35B filtering 0.7454 → 0.7638 but one issuer −0.0544 | [Task 3 log](../../../docs/task3_research_log.md) | 28-row confirmation; macro fact-set F1, not end-to-end answer ROUGE; no-issuer-below−0.01 gate fails. |
| NER consensus 0.9208 → 0.9332; four corrections/no new errors | [Consensus summary](../../../artifacts/task1_conservative_composition_grouped_seed_v1/summary.json), [Task 1 log](../../../docs/task1_research_log.md) | 80-row validation with two-seed support; validation repeatedly inspected. The poster explicitly labels the limitation. |
| August 11 test audit and August 17 policy/submission decision | [Research log](../../../docs/research_log.md) | Historical input overlap found before permission was resolved; retrieval initially quarantined. |
| 200/200 NER and 76/76 QA exact input matches; CPU-only final runner | [Final reproducibility](../../../docs/final_solution_reproducibility.md) | Selected inference used no model/API. Input-based fallbacks were separate. |
| ICR public 4/15 → 15/15 | [ICR paper](../../task2/main.tex), [results overview](../../../docs/results.md) | BERT versus final semantic rubric on the public test; private rubric result 34/35. |
| QA public 0.9994 / private 0.3726 | [QA paper](../../task3/main.tex) | Public 24 rows and private 52 rows. Different reference sets; version shift is inferred, not confirmed through private labels. |

No claim is made that the input-based systems would have beaten retrieval on the private test. The lack of that comparison is stated on the poster.
