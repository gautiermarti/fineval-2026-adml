#!/usr/bin/env python3
"""Audit that every reconstructed JF-ICR source identity is excluded from EDINET MLM."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_icr_edinet_mlm import excluded_company  # noqa: E402


SOURCE_IDENTITIES = {
    "ana_2020_fy": ("ANA Holdings", ("全日本空輸", "ＡＮＡ", "ANA")),
    "chiyoda_2020_fy": ("Chiyoda Corporation", ("千代田化工",)),
    "dnp_2025_h1": ("Dai Nippon Printing", ("大日本印刷",)),
    "itochu_ir": ("ITOCHU", ("伊藤忠",)),
    "kajima_2025_q3": ("Kajima", ("鹿島建設",)),
    "megabank_ir_2025": ("SMFG / SMBC", ("三井住友",)),
    "mitsubishi_corp_2020_fy": ("Mitsubishi Corporation", ("三菱商事",)),
    "mitsubishi_corp_2025_fy": ("Mitsubishi Corporation", ("三菱商事",)),
    "mitsubishi_corp_2025_h1": ("Mitsubishi Corporation", ("三菱商事",)),
    "mitsubishi_corp_legacy_q": ("Mitsubishi Corporation", ("三菱商事",)),
    "mitsubishi_corp_shareholder": ("Mitsubishi Corporation", ("三菱商事",)),
    "mitsui_2025_q3": ("Mitsui & Co.", ("三井物産",)),
    "mufg_2025": ("MUFG", ("三菱ＵＦＪ", "三菱UFJ")),
    "nissan_2024_fy": ("Nissan", ("日産",)),
    "nissan_legacy": ("Nissan", ("日産",)),
    "nomura_2020": ("Nomura", ("野村",)),
    "obayashi_2025_q3": ("Obayashi", ("大林組",)),
    "shimizu_2020_fyq2": ("Shimizu", ("清水建設",)),
    "softbank_fy2022_q3": ("SoftBank", ("ソフトバンク",)),
    "sumitomo_heavy_2025_fy": ("Sumitomo Heavy Industries", ("住友重機械",)),
    "unknown_94_96": ("Sumitomo Corporation", ("住友商事",)),
}


def main() -> None:
    source = pd.read_csv(ROOT / "data/derived/icr_source_folds.csv")
    missing = sorted(set(source.source_group) - set(SOURCE_IDENTITIES))
    if missing:
        raise SystemExit(f"source groups lack issuer identity audit: {missing}")
    edinet = pd.read_json(ROOT / "data/external/edinet2023/2023.jsonl", lines=True)
    rows = []
    for group in sorted(SOURCE_IDENTITIES):
        identity, terms = SOURCE_IDENTITIES[group]
        matches = sorted({
            company for company in edinet.company.astype(str)
            if any(term in company for term in terms)
        })
        rows.append({
            "source_group": group,
            "identity": identity,
            "issuer_terms": " | ".join(terms),
            "benchmark_rows": int((source.source_group == group).sum()),
            "edinet_matches": len(matches),
            "matched_companies": " | ".join(matches),
            "all_matches_excluded": all(excluded_company(company) for company in matches),
        })
    result = pd.DataFrame(rows)
    if not result.all_matches_excluded.all():
        raise SystemExit("at least one matched EDINET issuer is not excluded")
    output = ROOT / "data/derived/icr_edinet_exclusion_audit.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"source_groups={len(result)} all_matches_excluded=true output={output}")


if __name__ == "__main__":
    main()
