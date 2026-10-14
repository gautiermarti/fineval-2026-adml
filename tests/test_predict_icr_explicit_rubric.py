from collections import Counter

import pytest

from scripts.predict_icr_explicit_rubric import classify, is_balanced


@pytest.mark.parametrize(
    ("response", "label"),
    [
        ("取締役会で承認しました。", "+2"),
        ("条件に合う企業があれば前向きに検討します。", "+1"),
        ("複数の前提を置いて検証しています。", "0"),
        ("今年中の参入は見送る方向です。", "-1"),
        ("当社はこの市場へ参入しません。", "-2"),
    ],
)
def test_explicit_rubric(response, label):
    assert classify(response) == label


def test_balanced_packet_accepts_full_and_private_sizes():
    assert is_balanced(
        Counter({label: 10 for label in ("+2", "+1", "0", "-1", "-2")}), 50,
    )
    assert is_balanced(
        Counter({label: 7 for label in ("+2", "+1", "0", "-1", "-2")}), 35,
    )


def test_balanced_packet_rejects_unequal_support():
    assert not is_balanced(
        Counter({"+2": 8, "+1": 7, "0": 7, "-1": 7, "-2": 6}), 35,
    )
