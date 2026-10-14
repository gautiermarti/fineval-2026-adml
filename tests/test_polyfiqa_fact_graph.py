import json
from pathlib import Path

import pandas as pd

from fineval.polyfiqa_facts import build_fact_graph, provenance_is_exact


ROOT = Path(__file__).resolve().parents[1]


def test_fact_graph_covers_all_eight_question_families() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl").read_text().splitlines()
    ]
    graphs = [build_fact_graph(row["task_id"], row["query"], row["question"]) for row in rows]
    assert {graph.family for graph in graphs} == {
        "revenue", "balance", "cash", "rd", "revenue_focus", "allocation", "margin", "capex",
    }
    assert all(provenance_is_exact(graph, row["query"]) for graph, row in zip(graphs, rows, strict=True))


def test_fact_graph_has_expected_typed_easy_facts() -> None:
    frame = pd.read_parquet(ROOT / "data/raw/subtask3/public-00000-of-00001.parquet")
    observed = {}
    for _, row in frame.iloc[:4].iterrows():
        graph = build_fact_graph(row.task_id, row.query, row.question)
        observed[graph.family] = {fact.metric for fact in graph.numeric}
    assert {"revenue", "revenue_change_percent"} <= observed["revenue"]
    assert {"current_assets", "total_liabilities", "total_equity", "liabilities_to_equity"} <= observed["balance"]
    assert {"operating_cash_flow", "investing_cash_flow", "financing_cash_flow"} <= observed["cash"]
    assert {"research_and_development", "revenue", "rd_ratio_percent"} <= observed["rd"]
