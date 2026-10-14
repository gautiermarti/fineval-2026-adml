import pytest

from fineval.metrics import accuracy, rouge1_f1
from fineval.parsing import parse_icr_query, parse_polyfiqa_query


def test_accuracy() -> None:
    assert accuracy(["+1", "0"], ["+1", "-1"]) == 0.5


def test_accuracy_rejects_bad_label() -> None:
    with pytest.raises(ValueError):
        accuracy(["+1"], ["positive"])


def test_rouge_identity() -> None:
    assert rouge1_f1(["revenue increased"], ["revenue increased"]) == 1.0


def test_polyfiqa_rouge_uses_nfkc_and_multilingual_tokens() -> None:
    assert rouge1_f1(["Ｒｅｖｅｎｕｅ 増加 12.5%"], ["revenue 増加 12.5"]) == 1.0


def test_icr_parser() -> None:
    parsed = parse_icr_query(
        "Financial Question: Will you invest?\nCompany Response: We will consider it.\n"
        "Directly output the chosen label"
    )
    assert parsed.question == "Will you invest?"
    assert parsed.response == "We will consider it."


def test_icr_parser_repairs_reversed_interrogative_fields() -> None:
    parsed = parse_icr_query(
        "Financial Question: 工期面の大きな問題は起きていない。\n"
        "Company Response: 工程遅延のリスクはあるか。\n"
        "Directly output the chosen label"
    )
    assert parsed.question == "工程遅延のリスクはあるか。"
    assert parsed.response == "工期面の大きな問題は起きていない。"


def test_polyfiqa_parser() -> None:
    parsed = parse_polyfiqa_query(
        "Context:\nFinancial Statements: S\nEnglish News: E\nChinese News: C\n"
        "Japanese News: J\nSpanish News: ES\nGreek News: G\nQuestion: Q\nAnswer:\n"
    )
    assert parsed.financial_statements == "S"
    assert parsed.spanish_news == "ES"
    assert parsed.question == "Q"
