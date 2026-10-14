import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_polyfiqa_hybrid import (  # noqa: E402
    infer_task_id_from_input, normalize_text, packet_fingerprint,
)


def row(query: str) -> pd.Series:
    return pd.Series({"query": query})


def test_routes_alibaba_periods_from_packet() -> None:
    assert infer_task_id_from_input(row("Alibaba report 2025年3月31日")) == "BABA_20250515"
    assert infer_task_id_from_input(row("Alibaba report 2025年6月30日")) == "BABA_20250829"


def test_routes_tencent_periods_from_packet() -> None:
    assert infer_task_id_from_input(row("腾讯 report 2025年3月31日")) == "TCEHY_20250514"
    assert infer_task_id_from_input(row("Tencent report 2025年6月30日")) == "TCEHY_20250813"


def test_generic_route_uses_company_and_latest_year() -> None:
    assert infer_task_id_from_input(
        row("Microsoft annual report 2023 with news updated in 2024")
    ) == "MSFT_20240101"


def test_packet_fingerprint_ignores_wrapper_and_whitespace() -> None:
    query = (
        "Old wrapper\nContext:\nFinancial Statements: revenue 10\n"
        "English News: sales rose\nChinese News: 无\nJapanese News: なし\n"
        "Spanish News: nada\nGreek News: κανένα\nQuestion: What trends?\nAnswer:"
    )
    changed = query.replace("Old wrapper", "A completely new prompt wrapper").replace(
        "revenue 10", "revenue    10"
    )
    assert packet_fingerprint(query, "What trends?") == packet_fingerprint(
        changed, "What trends?"
    )


def test_packet_fingerprint_changes_with_evidence() -> None:
    query = (
        "Context: Financial Statements: revenue 10 English News: sales rose "
        "Chinese News: 无 Japanese News: なし Spanish News: nada Greek News: κανένα "
        "Question: What trends? Answer:"
    )
    assert packet_fingerprint(query, "What trends?") != packet_fingerprint(
        query.replace("revenue 10", "revenue 11"), "What trends?"
    )


def test_normalize_text_nfkc_and_whitespace() -> None:
    assert normalize_text("Ａ  \n B") == "A B"
