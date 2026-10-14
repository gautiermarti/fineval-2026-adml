from fineval.polyfiqa_entailment import (
    NumericMention,
    clean_claim,
    evidence_quotes,
    fact_clauses,
    numeric_match,
    numeric_mentions,
    numeric_support,
    quotation_support,
)


def test_clean_claim_and_quote_support_accept_exact_source_quote():
    source = "Revenue reached $62 billion and increased by 18%."
    evidence = f'News Evidence: English news: "{source}"'
    assert evidence_quotes(evidence) == [source]
    assert quotation_support(evidence, [source]) == (1.0, 1.0)
    assert clean_claim("Answer: Revenue grew.") == "Revenue grew."


def test_quotation_support_rejects_invented_quote():
    exact, overlap = quotation_support(
        'News Evidence: "Revenue collapsed unexpectedly."',
        ["Revenue increased substantially."],
    )
    assert exact == 0.0
    assert overlap < 0.5
    assert quotation_support("News Evidence: None.", ["irrelevant"]) == (1.0, 1.0)


def test_numeric_mentions_normalize_units_and_ignore_years():
    mentions = numeric_mentions("In 2023, revenue was $62.0B, up 17.6%.")
    assert NumericMention(62_000.0, "millions") in mentions
    assert NumericMention(17.6, "percent") in mentions
    assert all(item.value != 2023 for item in mentions)
    assert numeric_mentions("Revenue was 62020 million.")[0].value == 62020


def test_numeric_support_uses_facts_and_source_only_numbers():
    graph = {
        "numeric": [{"metric": "revenue", "values": [62_020], "unit": "millions"}],
        "evidence": [{"text": "Analysts expected $61.1 billion."}],
    }
    rate, supported, unsupported = numeric_support(
        "Revenue was $62.0B versus expectations of $61.1B.", graph,
    )
    assert (rate, supported, unsupported) == (1.0, 2, 0)
    assert numeric_match(
        NumericMention(62_000, "millions"), NumericMention(62_020, "millions"),
    )


def test_fact_clauses_are_atomic():
    clauses = fact_clauses({
        "numeric": [
            {"metric": "revenue", "values": [10, 8], "unit": "millions"},
            {"metric": "growth", "values": [25], "unit": "percent", "derived": True},
        ],
    })
    assert len(clauses) == 2
    assert clauses[0].startswith("revenue")
    assert clauses[1].startswith("Derived growth")
