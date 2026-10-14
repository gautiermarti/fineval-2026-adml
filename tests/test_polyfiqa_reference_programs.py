from fineval.polyfiqa_facts import NumericFact, PolyFiQAFactGraph
from fineval.polyfiqa_programs import execute_program, induce_program


def graph(current: float, previous: float, change: float) -> PolyFiQAFactGraph:
    return PolyFiQAFactGraph(
        task_id="MSFT_20240101",
        tier="easy",
        family="revenue",
        numeric=(
            NumericFact("revenue", (current, previous), "millions"),
            NumericFact("revenue_change_percent", (change,), "percent", True),
        ),
        evidence=(),
    )


def test_induce_and_execute_cross_company_revenue_program() -> None:
    source = graph(35_000, 30_600, 14.4)
    answer = (
        "Answer: Microsoft's revenue increased from $30.6B in 2019 to "
        "$35.0B in 2020, a 14.4% increase.\nNews Evidence: None."
    )
    program, reason = induce_program("MSFT_20200429", answer, source)
    assert reason == "aligned" and program is not None
    target = graph(12_000, 10_000, 20.0)
    rendered, reason = execute_program(program, target, "HON_20200430")
    assert reason == "executed"
    assert rendered == (
        "Answer: Honeywell's revenue increased from $10.0B in 2019 to "
        "$12.0B in 2020, a 20.0% increase."
    )


def test_program_rejects_opposite_trend_literal() -> None:
    source = graph(35_000, 30_600, 14.4)
    answer = "Answer: Microsoft revenue increased from $30.6B to $35.0B, or 14.4%."
    program, _ = induce_program("MSFT_20200429", answer, source)
    assert program is not None
    rendered, reason = execute_program(program, graph(9_000, 10_000, -10.0), "HON_20200430")
    assert rendered is None
    assert reason == "semantic_state_mismatch"

