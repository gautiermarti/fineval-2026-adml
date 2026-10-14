from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from collect_japanese_ir_qa import extract_pairs, is_benchmark_issuer


def test_extract_japanese_qa_marker_variants() -> None:
    text = """
【質問】今後の投資方針について教えてください。
【回答】需要を見極めながら、必要な投資を段階的に進める方針です。
【質問】増配を約束しますか。
【回答】現時点で決定した事実はなく、回答を差し控えます。
"""
    pairs = extract_pairs(text)
    assert len(pairs) == 2
    assert pairs[0][0].startswith("今後の投資方針")
    assert "段階的" in pairs[0][1]


def test_excludes_known_benchmark_issuers() -> None:
    assert is_benchmark_issuer("mufg", "https://www.mufg.jp/ir/")
    assert is_benchmark_issuer("unrelated", "https://www.nissan-global.com/JP/IR/")
    assert not is_benchmark_issuer("iij", "https://www.iij.ad.jp/ir/library/questions/")


def test_does_not_split_question_at_ai_line() -> None:
    text = """
Q. スマート製品の需要についてですが、
AI サーバーの機会をどのように見ていますか。
A. 需要は堅調で、生産能力を確保していきます。
Q2 次の投資方針を教えてください。
A2 必要な投資を段階的に進めます。
"""
    pairs = extract_pairs(text)
    assert len(pairs) == 2
    assert "AI サーバー" in pairs[0][0]
    assert pairs[0][1].startswith("需要は堅調")


def test_accepts_bare_qa_markers_with_required_whitespace() -> None:
    text = """
Q 今後の需要をどのように見ていますか。
A 中長期的に需要が拡大していくと想定しています。
Q 人員を増やす予定ですか。
A 必要な人材の採用を段階的に進めていきます。
"""
    pairs = extract_pairs(text)
    assert len(pairs) == 2
    assert pairs[0][0].startswith("今後の需要")
    assert pairs[1][1] == "必要な人材の採用を段階的に進めていきます。"


def test_accepts_vertical_japanese_colon_marker() -> None:
    text = """
Q︓ 今後の大型パネル需要はどのように見ていますか。
A︓ 生産性を向上させる取り組みを継続していきます。
"""
    assert extract_pairs(text) == [
        (
            "今後の大型パネル需要はどのように見ていますか。",
            "生産性を向上させる取り組みを継続していきます。",
        )
    ]
