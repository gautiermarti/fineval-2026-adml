#!/usr/bin/env python3
"""Build source-held-out folds from recovered public IR documents."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]

# Ranges sharing a name are intentionally consolidated even when non-contiguous.
# `confirmed` means an exact question was located in the linked public document;
# `inferred` means adjacent questions clearly share company/topic/layout but the
# exact document has not yet been individually verified.
BLOCKS = [
    (0, 6, "mufg_2025", "inferred", "https://www.mufg.jp/ir/presentation/2025/"),
    (7, 16, "mitsubishi_corp_2025_h1", "inferred", "https://www.mitsubishicorp.com/jp/ja/ir/library/meetings/"),
    (17, 24, "mitsubishi_corp_shareholder", "inferred", "https://www.mitsubishicorp.com/jp/ja/ir/"),
    (25, 47, "megabank_ir_2025", "inferred", ""),
    (48, 74, "mitsubishi_corp_2025_fy", "confirmed", "https://www.mitsubishicorp.com/jp/ja/ir/library/meetings/"),
    (75, 86, "itochu_ir", "inferred", "https://www.itochu.co.jp/ja/ir/"),
    (87, 93, "mitsubishi_corp_2025_fy", "inferred", "https://www.mitsubishicorp.com/jp/ja/ir/library/meetings/"),
    (94, 96, "unknown_94_96", "unknown", ""),
    (97, 100, "chiyoda_2020_fy", "confirmed", "https://www.chiyodacorp.com/jp/ir/assets/2019Q4_QA_J.pdf"),
    (101, 106, "nissan_legacy", "inferred", "https://www.nissan-global.com/JP/IR/"),
    (107, 110, "softbank_fy2022_q3", "confirmed", "https://group.softbank/system/files/pdf/ir/presentations/2022/earnings-presentation_q3fy2022_03_ja.pdf"),
    (111, 115, "shimizu_2020_fyq2", "confirmed", "https://pdf.irpocket.com/C1803/aMKh/cuPQ/Kqq8.pdf"),
    (116, 119, "nomura_2020", "inferred", "https://www.nomuraholdings.com/jp/investor/"),
    (120, 128, "mitsubishi_corp_legacy_q", "inferred", "https://www.mitsubishicorp.com/jp/ja/ir/library/meetings/"),
    (129, 134, "chiyoda_2020_fy", "confirmed", "https://www.chiyodacorp.com/jp/ir/assets/2019Q4_QA_J.pdf"),
    (135, 148, "mitsubishi_corp_2020_fy", "inferred", "https://www.mitsubishicorp.com/jp/ja/ir/library/meetings/"),
    (149, 152, "ana_2020_fy", "confirmed", "https://www.ana.co.jp/group/investors/data/kessan/pdf/2021_04_3.pdf"),
    (153, 165, "nomura_2020", "inferred", "https://www.nomuraholdings.com/jp/investor/"),
    (166, 175, "nissan_2024_fy", "confirmed", "https://www.nissan-global.com/JP/IR/FINANCIAL_RESULTS/ASSETS/DATA/2024/2024results_announcement_analyst_session_QA_j.pdf"),
    (176, 183, "shimizu_2020_fyq2", "confirmed", "https://pdf.irpocket.com/C1803/aMKh/cuPQ/Kqq8.pdf"),
    (184, 190, "softbank_fy2022_q3", "confirmed", "https://group.softbank/system/files/pdf/ir/presentations/2022/earnings-presentation_q3fy2022_03_ja.pdf"),
    (191, 195, "sumitomo_heavy_2025_fy", "confirmed", "https://www.shi.co.jp/ir/library/presentation/pdf/Financial_Q%26Asummary25.pdf"),
    (196, 214, "dnp_2025_h1", "confirmed", "https://www.dnp.co.jp/ir/library/presentation/pdf/dnp_25Q2qa.pdf"),
    (215, 230, "obayashi_2025_q3", "confirmed", "https://ir.obayashi.co.jp/ja/ir/ir-news/news202602121600/main/0/link/20260209QA.pdf"),
    (231, 246, "mitsui_2025_q3", "confirmed", "https://minkabu.jp/news/4436088"),
    (247, 252, "kajima_2025_q3", "confirmed", "https://www.kajima.co.jp/ir/presentation/pdf/presentation-qa-20260212-j.pdf"),
]


def main() -> None:
    semantic = pd.read_csv(ROOT / "data/derived/icr_semantic_folds.csv", dtype={"gold_label": str})
    records = []
    for _, row in semantic.iterrows():
        item_id = int(row["id"])
        matches = [block for block in BLOCKS if block[0] <= item_id <= block[1]]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one source block for id={item_id}, got {matches}")
        _, _, source, confidence, url = matches[0]
        records.append({**row.to_dict(), "source_group": source, "source_confidence": confidence, "source_url": url})
    frame = pd.DataFrame(records)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    frame["source_fold"] = -1
    for fold, (_, test) in enumerate(splitter.split(frame, frame["gold_label"], frame["source_group"])):
        frame.loc[test, "source_fold"] = fold
    output = ROOT / "data/derived/icr_source_folds.csv"
    frame.to_csv(output, index=False)
    print("source_groups", frame["source_group"].nunique())
    print("confidence", frame["source_confidence"].value_counts().to_dict())
    print("fold_sizes", frame["source_fold"].value_counts().sort_index().to_dict())
    print(pd.crosstab(frame["source_fold"], frame["gold_label"]).to_string())


if __name__ == "__main__":
    main()

