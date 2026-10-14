from pathlib import Path

from fineval.icr_functional_expressions import (
    FunctionalExpression,
    informative_entries,
    load_tsutsuji,
    longest_nonoverlapping_matches,
    semantic_signature,
)


def test_load_tsutsuji_inherits_semantic_class(tmp_path: Path) -> None:
    xml = tmp_path / "tiny.xml"
    xml.write_text(
        "<ENTRIES><L1 MCLASS='G1'><L9 ID='a'>\nする.つもりだ\n</L9>"
        "<L2 MCLASS='y1'><L9 ID='b'>わけ.では.ない</L9></L2></L1></ENTRIES>",
        encoding="utf-8",
    )
    entries = load_tsutsuji(xml)
    assert {(row.surface, row.semantic_class) for row in entries} == {
        ("するつもりだ", "G1"), ("わけではない", "y1"),
    }


def test_longest_match_and_semantic_tail_signature() -> None:
    entries = informative_entries([
        FunctionalExpression("ではない", "y1", "short"),
        FunctionalExpression("わけではない", "y1", "long"),
        FunctionalExpression("予定である", "G3", "plan"),
        FunctionalExpression("で", "D4", "unsafe"),
    ])
    text = "現時点で約束したわけではないが、実行する予定である"
    matches = longest_nonoverlapping_matches(text, entries)
    assert [row.expression.identifier for row in matches] == ["long", "plan"]
    signature = semantic_signature(text, entries, tail_fraction=0.5)
    assert signature["class:y1:count"] == 1.0
    assert signature["class:G3:tail"] == 1.0
    assert "class:D4:count" not in signature
