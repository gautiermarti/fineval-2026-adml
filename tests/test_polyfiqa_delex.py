from fineval.polyfiqa_delex import delexicalize_reference, restore_reference
from fineval.polyfiqa_facts import NumericFact, PolyFiQAFactGraph


def packet(statements: str, news: str = "None") -> str:
    return f"""Context: Financial Statements:
{statements}
English News: {news}
Chinese News: None
Japanese News: None
Spanish News: None
Greek News: None
Question: What happened?
Answer:"""


def graph():
    return PolyFiQAFactGraph(
        task_id="MSFT_20240101", tier="easy", family="revenue",
        numeric=(
            NumericFact("revenue", (62020.0, 52747.0), "millions"),
            NumericFact("revenue_change_percent", (17.58,), "percent", True),
        ),
        evidence=(),
    )


def test_fact_and_exact_copy_slots_round_trip():
    answer = "Microsoft's revenue was $62,020 million, up 17.6%; guidance was $61.1 billion."
    query = "Analyst guidance was $61.1 billion."
    record, reason = delexicalize_reference(
        "MSFT_20240101", answer, query, graph(),
    )
    assert reason == "aligned"
    assert record is not None
    assert [slot["source"] for slot in record["slots"]] == [
        "fact", "fact", "query_copy",
    ]
    assert restore_reference(record, graph(), "MSFT_20240101") == answer


def test_unverifiable_number_rejects_example():
    record, reason = delexicalize_reference(
        "MSFT_20240101", "Microsoft revenue was $999 million.", "No value here.", graph(),
    )
    assert record is None
    assert reason.startswith("unverifiable_number:")


def test_qualitative_reference_is_retained_without_fake_slots():
    record, reason = delexicalize_reference(
        "MSFT_20240101", "Microsoft's outlook remains constructive.", "No value.", graph(),
    )
    assert reason == "aligned_qualitative"
    assert record is not None and record["slots"] == []
    assert restore_reference(record, graph(), "MSFT_20240101") == (
        "Microsoft's outlook remains constructive."
    )


def test_ratio_unit_aligns_even_without_ratio_in_metric_name():
    ratio_graph = PolyFiQAFactGraph(
        task_id="MSFT_20200429", tier="easy", family="balance",
        numeric=(NumericFact("liabilities_to_equity", (1.492982594,), "ratio", True),),
        evidence=(),
    )
    answer = "Microsoft's liabilities-to-equity ratio was 1.49 in 2020."
    record, reason = delexicalize_reference(
        "MSFT_20200429", answer, "The ratio was calculated from the table.", ratio_graph,
    )
    assert reason == "aligned"
    assert record is not None and record["slots"][0]["kind"] == "ratio"
    assert restore_reference(record, ratio_graph, "MSFT_20200429") == answer


def test_unlabelled_jnj_amounts_are_millions_not_thousands():
    jnj_graph = PolyFiQAFactGraph(
        task_id="JNJ_20200724", tier="expert", family="capex",
        numeric=(NumericFact("capital_expenditures", (1287.0,), "units"),),
        evidence=(),
    )
    answer = "Johnson & Johnson's capital expenditures were $1.3B."
    record, reason = delexicalize_reference(
        "JNJ_20200724", answer, "Additions to property were 1,287.", jnj_graph,
    )
    assert reason == "aligned"
    assert record is not None
    assert restore_reference(record, jnj_graph, "JNJ_20200724") == answer


def test_unlabelled_uvv_amounts_are_thousands():
    uvv_graph = PolyFiQAFactGraph(
        task_id="UVV_20200805", tier="expert", family="capex",
        numeric=(NumericFact("capital_expenditures", (8386.0,), "units"),),
        evidence=(),
    )
    answer = "Universal Corporation's capital expenditures were $8.4M."
    record, reason = delexicalize_reference(
        "UVV_20200805", answer, "Capital expenditures were 8,386.", uvv_graph,
    )
    assert reason == "aligned"
    assert record is not None
    assert restore_reference(record, uvv_graph, "UVV_20200805") == answer


def test_unique_query_number_can_be_rendered_under_reference_style():
    empty_graph = PolyFiQAFactGraph(
        task_id="JNJ_20200724", tier="expert", family="allocation",
        numeric=(), evidence=(),
    )
    answer = "Johnson & Johnson invested $1.3B."
    record, reason = delexicalize_reference(
        "JNJ_20200724", answer,
        packet("| Additions to property | 1,287 |"), empty_graph,
    )
    assert reason == "aligned"
    assert record is not None
    slot = record["slots"][0]
    assert slot["source"] == "query_transform"
    assert slot["support_value"] == 1287.0
    assert slot["support_surface"].strip() == "1,287"
    assert restore_reference(record, empty_graph, "JNJ_20200724") == answer


def test_query_transform_rejects_ambiguous_rounded_values():
    empty_graph = PolyFiQAFactGraph(
        task_id="JNJ_20200724", tier="expert", family="allocation",
        numeric=(), evidence=(),
    )
    record, reason = delexicalize_reference(
        "JNJ_20200724", "Johnson & Johnson invested $1.3B.",
        packet("| Acquisition A | 1,287 |\n| Acquisition B | 1,340 |"),
        empty_graph,
    )
    assert record is None
    assert reason == "unverifiable_number:$1.3B"


def test_query_transform_uses_split_currency_cell_not_numeric_collision():
    empty_graph = PolyFiQAFactGraph(
        task_id="UVV_20240930", tier="expert", family="allocation",
        numeric=(), evidence=(),
    )
    statements = (
        "| Restructuring payments | (806) |\n"
        "| Dividends declared per common share | $ | 0.81 |"
    )
    answer = "Universal Corporation raised its quarterly dividend to $0.81."
    record, reason = delexicalize_reference(
        "UVV_20240930", answer, packet(statements), empty_graph,
    )
    assert reason == "aligned"
    assert record is not None
    slot = record["slots"][0]
    assert slot["source"] == "query_transform"
    assert slot["support_value"] == 0.81
    assert "Dividends declared" in slot["copy_context"]
    assert restore_reference(record, empty_graph, "UVV_20240930") == answer


def test_percent_fact_can_render_as_decimal_ratio():
    ratio_graph = PolyFiQAFactGraph(
        task_id="MSFT_20210427", tier="easy", family="rd",
        numeric=(NumericFact("rd_ratio_percent", (12.4778,), "percent", True),),
        evidence=(),
    )
    answer = "Microsoft's R&D ratio was 0.125."
    record, reason = delexicalize_reference(
        "MSFT_20210427", answer, packet("| R&D | 5,204 |"), ratio_graph,
    )
    assert reason == "aligned"
    assert record is not None
    assert record["slots"][0]["kind"] == "ratio_from_percent"
    assert restore_reference(record, ratio_graph, "MSFT_20210427") == answer


def test_calendar_days_and_office_product_number_are_not_slots():
    qualitative_graph = PolyFiQAFactGraph(
        task_id="MSFT_20210126", tier="expert", family="margin",
        numeric=(), evidence=(),
    )
    answer = (
        "Microsoft expanded Office 365 as of December 31, 2020."
    )
    record, reason = delexicalize_reference(
        "MSFT_20210126", answer, packet("| Revenue | 1 |"), qualitative_graph,
    )
    assert reason == "aligned_qualitative"
    assert record is not None and record["slots"] == []
    assert restore_reference(record, qualitative_graph, "MSFT_20210126") == answer
