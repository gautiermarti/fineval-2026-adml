#!/usr/bin/env python3
"""Quality curation and forward-intent indexing for Japanese IR Q&A."""

from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import re


STRUCTURAL_BOUNDARY = re.compile(
    r"\n(?:(?:第[一二三四１２３４0-9]+四半期|通期|中間期).{0,35}決算説明会|"
    r"【[^】]{0,40}(?:決算について|質問者\s*[0-9０-９]+)[^】]*】|"
    r"(?:決算短信・)?決算説明会資料\s*$|以\s*上\s*$|"
    r"(?:映像|精機|ヘルスケア|メディカル|インストルメンツ|コンポーネント|"
    r"デジタルマニュファクチャリング|アドバンストマニュファクチャリング|"
    r"その他|全体)(?:事業)?\s*$|"
    r"(?:PDF|Adobe®?\s*Reader®?|サイトポリシー).*)",
    re.IGNORECASE,
)
CONTAMINATION = (
    "Adobe Reader", "Adobe® Reader®", "サイトポリシー（PDFについて）",
    "決算説明会資料\nPDF", "アナリストレポート\nPDF",
)
FORWARD_CUES = (
    "今後", "見通し", "方針", "計画", "目標", "予定", "意向", "展望",
    "考えて", "考える", "想定", "進め", "取り組", "検討", "決定", "約束",
    "差し控え", "開示", "可能性", "目指", "注力", "継続", "強化", "実施",
    "対応する", "対応して", "していく", "してまい", "する予定", "しない",
)
REFUSAL_CUES = (
    "差し控え", "回答でき", "開示してい", "申し上げられ", "決定していな",
    "未定", "現時点では", "具体的には", "コメントでき",
)


def normalize(value: str) -> str:
    value = value.replace("\u3000", " ").replace("\ufeff", "")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def trim_structural_tail(value: str) -> str:
    match = STRUCTURAL_BOUNDARY.search(value)
    return normalize(value[:match.start()] if match else value).strip(" -―─:：")


def is_general_faq(url: str) -> bool:
    folded = url.casefold()
    return "/contact/faq" in folded or "/ir/irfaq" in folded


def curate(record: dict) -> tuple[dict | None, str]:
    if is_general_faq(record["source_url"]):
        return None, "general_faq_not_earnings_qa"
    question = trim_structural_tail(record["question"])
    response = trim_structural_tail(record["response"])
    if not (8 <= len(question) <= 500):
        return None, "question_length"
    if not (15 <= len(response) <= 1_400):
        return None, "response_length"
    if any(cue in response for cue in CONTAMINATION):
        return None, "navigation_contamination"
    japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", question + response))
    if japanese / max(len(question + response), 1) < 0.10:
        return None, "insufficient_japanese"
    pair_id = sha256((question + "\x1f" + response).encode()).hexdigest()
    combined = question + "\n" + response
    result = {
        **record,
        "pair_id": pair_id,
        "question": question,
        "response": response,
        "forward_cues": [cue for cue in FORWARD_CUES if cue in combined],
        "refusal_cues": [cue for cue in REFUSAL_CUES if cue in response],
    }
    result["forward_candidate"] = bool(result["forward_cues"] or result["refusal_cues"])
    return result, "accepted"


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path(__file__).resolve().parents[1] / "data/external/japanese_ir_qa/pairs.jsonl",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input.parent
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    accepted, rejected, seen = [], [], set()
    for record in records:
        result, reason = curate(record)
        if result is None:
            rejected.append({"reason": reason, **record})
            continue
        if result["pair_id"] in seen:
            rejected.append({"reason": "duplicate_after_cleaning", **record})
            continue
        seen.add(result["pair_id"])
        accepted.append(result)
    forward = [record for record in accepted if record["forward_candidate"]]
    output.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        ("pairs_curated.jsonl", accepted),
        ("forward_candidates.jsonl", forward),
        ("rejected_quality.jsonl", rejected),
    ):
        with (output / filename).open("w") as handle:
            for record in values:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "raw_pairs": len(records),
        "curated_pairs": len(accepted),
        "forward_candidates": len(forward),
        "issuers": len({record["issuer"] for record in accepted}),
        "pairs_by_issuer": dict(sorted(Counter(record["issuer"] for record in accepted).items())),
        "forward_by_issuer": dict(sorted(Counter(record["issuer"] for record in forward).items())),
        "rejection_reasons": dict(sorted(Counter(record["reason"] for record in rejected).items())),
        "refusal_candidate_pairs": sum(bool(record["refusal_cues"]) for record in accepted),
        "benchmark_similarity_max": max(
            (record["benchmark_max_similarity"] for record in accepted), default=0.0,
        ),
    }
    (output / "curation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
