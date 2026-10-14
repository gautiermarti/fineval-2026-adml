from pathlib import Path
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from curate_japanese_ir_qa import curate, trim_structural_tail


def test_trim_structural_navigation_tail() -> None:
    value = (
        "必要な投資を段階的に進める方針です。\n"
        "第２四半期 決算説明会（2020年10月7日開催）\n決算説明会資料\nPDF"
    )
    assert trim_structural_tail(value) == "必要な投資を段階的に進める方針です。"
    assert trim_structural_tail(
        "市場を拡大していきます。\n決算短信・決算説明会資料"
    ) == "市場を拡大していきます。"
    assert trim_structural_tail(
        "生産性向上に取り組みます。\nヘルスケア"
    ) == "生産性向上に取り組みます。"


def test_curator_rejects_general_faq() -> None:
    result, reason = curate({
        "source_url": "https://example.jp/contact/faq/",
        "question": "今後の方針を教えてください。",
        "response": "必要な投資を進める方針です。",
    })
    assert result is None
    assert reason == "general_faq_not_earnings_qa"
