"""Identify the historical integration tests requiring unbundled data artifacts."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_INPUTS = {
    'test_evaluate_polyfiqa_large_content_planner.py::test_real_screen_has_complete_cells_and_company_disjoint_demos': [
        'artifacts/polyfiqa_graph_to_reference_v5/loco',
        'artifacts/polyfiqa_graph_to_reference_v5/training.jsonl',
    ],
    'test_evaluate_polyfiqa_large_planner_intersection_confirmation.py::test_real_confirmation_screen_is_complete_and_disjoint_from_v1': [
        'artifacts/polyfiqa_graph_to_reference_v5/loco',
        'artifacts/polyfiqa_graph_to_reference_v5/training.jsonl',
        'artifacts/polyfiqa_large_content_planner_v1/manifest.json',
    ],
    'test_evaluate_polyfiqa_nemotron30b_fact_filter.py::test_frozen_full_population_is_unique_and_company_disjoint': [
        'artifacts/polyfiqa_graph_to_reference_v5/training.jsonl',
        'artifacts/polyfiqa_numeric_planner_v5/predictions.jsonl',
    ],
    'test_icr_edinet_exclusion_audit.py::test_every_reconstructed_source_group_has_an_identity': [
        'data/derived/icr_source_folds.csv',
    ],
    'test_polyfiqa_fact_graph.py::test_fact_graph_covers_all_eight_question_families': [
        'data/external/finmmeval_task2/task2_test_questions.jsonl',
    ],
    'test_polyfiqa_fact_graph.py::test_fact_graph_has_expected_typed_easy_facts': [
        'data/raw/subtask3/public-00000-of-00001.parquet',
    ],
    'test_train_icr_disagreement_weighted.py::test_disagreement_ids_are_frozen_fourteen_rows': [
        'data/external/ebisu_human_rationales/rationales.jsonl',
    ],
}


def pytest_addoption(parser):
    parser.addoption('--require-external-data', action='store_true',
                     help='Fail collection if a selected historical integration test lacks its data')


def pytest_collection_modifyitems(config, items):
    for item in items:
        key = item.nodeid.removeprefix('tests/')
        missing = [p for p in EXTERNAL_INPUTS.get(key, []) if not (ROOT/p).exists()]
        if missing:
            reason = 'Historical data not bundled; see docs/research.md: ' + ', '.join(missing)
            if config.getoption('--require-external-data'):
                raise pytest.UsageError(reason)
            item.add_marker(pytest.mark.skip(reason=reason))
