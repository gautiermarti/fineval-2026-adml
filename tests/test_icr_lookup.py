import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from predict_icr import record_id, semantic_query_key  # noqa: E402

import pandas as pd


def test_semantic_key_ignores_outer_prompt_wrapper() -> None:
    body = (
        "Financial Question: 今後の方針は。\n"
        "Company Response: 現時点では検討中です。\n"
        "Directly output the chosen label"
    )
    assert semantic_query_key("old instructions\n" + body) == semantic_query_key(
        "completely different instructions\n" + body
    )


def test_semantic_key_changes_with_response() -> None:
    first = (
        "Financial Question: 今後の方針は。 Company Response: 検討中です。 "
        "Directly output the chosen label"
    )
    second = first.replace("検討中です", "実施します")
    assert semantic_query_key(first) != semantic_query_key(second)


def test_record_id_preserves_string_ids() -> None:
    assert record_id(pd.Series({"id": "test_001"}), 7) == "test_001"
    assert record_id(pd.Series({}), 7) == 7
