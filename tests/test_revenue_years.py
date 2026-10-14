import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baseline_polyfiqa_revenue_rd import revenue_years


def test_recent_annual_report_uses_explicit_statement_years():
    assert revenue_years(
        "JNJ_20240216", "| December 31, | 2023 | 2022 |",
    ) == (2023, 2022)


def test_legacy_packet_keeps_validated_filing_date_heuristic():
    assert revenue_years("MSFT_20201027", "| 2019 | 2020 |") == (2021, 2020)
