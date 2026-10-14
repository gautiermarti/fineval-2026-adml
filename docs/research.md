# Research guide

The papers provide the stable account of each system. The research logs and preregistration documents record intermediate hypotheses, thresholds, corrections, and rejected candidates in chronological order; some paths refer to large local artifacts that are regenerated rather than redistributed.

| Task | Start here | Main implementation paths |
|---|---|---|
| Greek NER | [Paper](../papers/task1/main.pdf), [log](task1_research_log.md) | `scripts/train_task1_*`, `scripts/evaluate_task1_*`, `scripts/compose_task1_conservative_policies.py`, `src/fineval/task1_*` |
| Japanese ICR | [Paper](../papers/task2/main.pdf), [general log](research_log.md) | `scripts/train_icr_*`, `scripts/evaluate_icr_*`, `src/fineval/icr_*` |
| Financial QA | [Paper](../papers/task3/main.pdf), [log](task3_research_log.md) | `scripts/predict_polyfiqa.py`, `scripts/evaluate_polyfiqa_*`, `src/fineval/polyfiqa_*`, `src/fineval/finance.py` |

`requirements-research.txt` records the original broader dependencies. Research scripts may additionally require particular model revisions, CUDA, a compatible model server, and datasets specified in their arguments or the logs. Start each script with `--help` and reproduce the relevant prerequisite stages. Workstation-specific model-server launch scripts are not included; the Python clients accept model-server configuration.

## Selected evidence

- [NER consensus](../artifacts/task1_conservative_composition_grouped_seed_v1/summary.json): 80-row validation, four corrections and no new errors; this is development evidence after repeated validation inspection.
- [NER entity-aware adaptation](../artifacts/task1_entity_aware_fold0_v1/evaluation.json): failed the first-fold gate.
- [ICR seed confirmation](../artifacts/task2_distance_aware_seed19_confirmation_v1/summary.json): independent-seed evidence for the historical ordinal model.
- [QA sparse-edit oracle](../artifacts/polyfiqa_sparse_edit_oracle_v1/summary.json): candidate headroom when references are visible; not a deployable selector.

The poster summarizes selected decisions. Aggregate summaries are included; model weights and row-level dataset-derived records are not bundled. See the [poster claim map](../papers/conference/poster/sources.md).

Seven historical integration tests need unbundled reference data or generated artifacts. `python -m pytest tests -rs` reports those missing inputs as explicit skips; it does not suppress other test failures. After recreating the original data/artifact tree, use `python -m pytest tests --require-external-data` to require every selected integration test to run. The exact paths are listed in [tests/conftest.py](../tests/conftest.py).

## Research inventory

The poster's historical counts refer to a frozen source snapshot: 209 research scripts, 32 modules, 99 test files, and 25 preregistration documents. These are artifact counts, not a measure of person-hours, experiment count, agent autonomy, or productivity. The public release adds packaging utilities and preserves the [historical count record](../papers/conference/poster/research_inventory.json).
