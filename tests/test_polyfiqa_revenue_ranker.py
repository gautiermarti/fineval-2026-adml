from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_polyfiqa import (
    balance_health_phrase, balance_news_evidence, normalized_news_amounts,
    rank_revenue_candidates,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Revenue was $61.9 billion.", 61_900),
        ("Los ingresos fueron 70.066 millones de dólares.", 70_066),
        ("公司总营收为619亿美元。", 61_900),
        ("売上高は9億3719万ドルだった。", 937.19),
        ("売上高は29億4728万ドルだった。", 2_947.28),
        ("売上高は21.89Bドルだった。", 21_890),
    ],
)
def test_normalized_news_amounts(text, expected):
    assert normalized_news_amounts(text) == pytest.approx([expected])


def test_revenue_ranker_prefers_exact_total_over_high_semantic_segment():
    result = SimpleNamespace(current=61_858, change_percent=17.0, unit="millions")
    candidates = [
        {"text": "Azure cloud segment revenue grew 30%.", "semantic_score": .9},
        {"text": "Total revenue rose 17% to $61.9 billion.", "semantic_score": .3},
    ]
    assert rank_revenue_candidates(candidates, result)[0] == candidates[1]


def test_balance_evidence_requires_cue_and_matching_total():
    context = SimpleNamespace(
        english_news=(
            "Revenue reached $1.49 billion. "
            "Shareholders' equity improved to $1.49 billion from $1.46 billion."
        ),
        chinese_news="", japanese_news="", spanish_news="", greek_news="",
    )
    result = SimpleNamespace(
        current_assets=2_200_000, total_liabilities=1_550_000,
        total_equity=1_490_000, unit="thousands",
    )
    evidence = balance_news_evidence(context, result)
    assert evidence.startswith('English news: "Shareholders')
    assert "Revenue reached" not in evidence


@pytest.mark.parametrize(
    ("ratio", "cue"),
    [(0.75, "strong equity"), (1.0, "balanced"), (1.5, "moderate"), (3.0, "high")],
)
def test_balance_health_phrase_is_ratio_grounded(ratio, cue):
    assert cue in balance_health_phrase(ratio)
