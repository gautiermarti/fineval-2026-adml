# Results and interpretation

The [official FinEval task website](https://www.finnlp2026sharedtask.info/) publishes three private leaderboards. The team appears as `gmarti`; the system papers use **ADML**. Results were checked on 15 October 2026.

| Task | Public result | Private result | Private rank |
|---|---:|---:|---|
| Greek NER | Entity F1 1.0000 | Entity F1 0.9676 | 1st |
| Japanese ICR | Accuracy 1.0000 (15/15) | Accuracy 0.9714 (34/35) | Tied 1st |
| Multilingual QA | ROUGE-1 F1 0.9994 (24 rows) | ROUGE-1 F1 0.3726 (52 rows) | 2nd |

The submitted NER and QA systems retrieved full-query matches from older public reference snapshots. The selected Japanese ICR system classified explicit commitment/refusal cues with a deterministic rubric. The organizers' 17 August external-data clarification permitted historical reference use. The selected workflows ran on CPU and used no model API or neural checkpoint.

All 200 NER and 76 QA test queries matched historical inputs. The QA public/private score gap is consistent with a reference-version change, but private reference answers were not available to establish that explanation directly. Historical oracle experiments are separate from the official test and must not be read as deployable leaderboard improvements.

The broader research produced encoders, conservative span consensus, ordinal classification, grounded fact compilation, and sparse-edit selection. These experiments are discussed in the papers and [research guide](research.md). We do not claim that these earlier input-based systems would have beaten the selected retrieval systems on the private test.

Machine-readable values and expected submission hashes are in [results.json](../results.json).
