#!/usr/bin/env python3
"""Compare direct and decomposed local judges on a stratified pilot."""

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fineval.metrics import accuracy  # noqa: E402
from fineval.ollama import generate_json  # noqa: E402
from fineval.parsing import parse_icr_query  # noqa: E402


STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "proposition": {"type": "string"},
        "responsiveness": {"type": "string", "enum": ["direct", "partial", "evasive"]},
        "direction": {"type": "string", "enum": ["accept", "unclear", "reject"]},
        "strength": {"type": "string", "enum": ["definitive", "qualified", "none"]},
        "conditionality": {"type": "string", "enum": ["none", "ordinary", "material"]},
        "decisive_span": {"type": "string"},
        "counterevidence_span": {"type": "string"},
        "label": {"type": "string", "enum": ["+2", "+1", "0", "-1", "-2"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": [
        "proposition", "responsiveness", "direction", "strength", "conditionality",
        "decisive_span", "counterevidence_span", "label", "confidence", "rationale",
    ],
}


RUBRIC = """ラベルは回答の感情ではなく、質問中の提案・行動・見通しに対する会社の実際の立場を示す。
+2: 明確で確定的な約束・同意・実施済みの方針。通常の説明上の留保だけでは弱めない。
+1: 肯定方向だが、条件付き、限定的、目標・意向・計画段階、または実行時期が未確定。
0: 肯否を示さない情報提供、真に未決定、両論併記、質問に対する立場が不明。
-1: 婉曲な拒否、実行可能性への否定的姿勢、延期、質問の前提を実質的に退ける。
-2: 明確で確定的な拒否・否定・不可能の表明。
特に回答末尾の述語、否定、モダリティを重視し、単なる前向きな語調を約束と誤認しない。"""

GUIDELINE_RUBRIC = """これは感情分析や質問への賛否判定ではない。回答が示す将来の行動、決定、方針、目標、戦略的方向性へのコミットメントまたは拒否の強さを分類する。
最初に、回答全体に将来志向の行動・決定・方針・目標・戦略的方向性があるか判定する。過去・現在の事実、背景、理由の説明だけなら、断定的でも0。
+2: 「実施します」「達成します」「決定・確定しています」等、明確で決定的なコミットメント。通常の説明上の留保では弱めない。
+1: 肯定方向だが「していきたい」「目指す」「考えている」「見込む」、条件、可能性、時期・具体性不足を伴う意向。
0: 真に曖昧、検討中、見通しを示せない、または将来方針を伴わない説明・事実・背景のみ。
-1: 提案された将来の行動・変更を、現時点・期間限定・条件付きで拒むため、再考の余地がある。
-2: 「予定はない」「行わない」「当社方針ではない」等、提案された将来の行動・決定・方針を明確かつ最終的に拒否する。
複数の信号があれば、孤立した語ではなく回答全体の支配的立場を確認した上で、適用可能な最も強いコミットメントまたは拒否を採る。丁寧な定型句だけをコミットメントとみなさない。"""


def direct_prompt(question: str, response: str) -> str:
    return f"""あなたは日本企業のIR・決算説明会に精通した金融実務家です。
次の質問に対する会社回答の暗黙的コミュニケーション意図を判定してください。
{RUBRIC}
質問: {question}
会社回答: {response}
根拠となる日本語の原文を短く引用し、最終ラベルを一つ選んでください。"""


def structured_prompt(question: str, response: str) -> str:
    return f"""あなたは日本語の語用論と日本企業のIRに精通した金融実務家です。
まず質問を会社が肯定・否定できる命題に直し、次に回答がその命題へ直接答えるか、方向、強さ、条件性を独立に分析してください。
最後にのみラベルへ写像してください。
{RUBRIC}
質問: {question}
会社回答: {response}
    回答中の決定的根拠と、それに反する根拠があれば両方を原文のまま短く引用してください。"""


def guideline_prompt(question: str, response: str) -> str:
    return f"""あなたは日本企業のIR回答を、公式の注釈指針に厳密に従って分類します。
{GUIDELINE_RUBRIC}
質問: {question}
会社回答: {response}
将来志向ゲート、方向、確定性、条件性、回答全体の支配的立場を順に検討し、決定的な原文を短く引用してラベルを一つ選んでください。"""


def contrastive_prompt(question: str, response: str, exemplars: dict) -> str:
    demonstrations = []
    for label in ["-2", "-1", "0", "+1", "+2"]:
        item = exemplars.get(label)
        if item is None:
            continue
        demonstrations.append(
            f"【既知例 {label}】\n質問: {item['question']}\n会社回答: {item['response']}"
        )
    examples_text = "\n\n".join(demonstrations)
    return f"""あなたは日本企業IRの暗黙的意図を判定します。
{RUBRIC}

以下は現在の対象と同じ出典グループを除外して検索した、各ラベルの意味的に近い既知例です。
表面的な単語一致ではなく、質問が作る命題と回答の関係を比較してください。
{examples_text}

【判定対象】
質問: {question}
会社回答: {response}

各隣接ラベルとの差を検討し、決定的な日本語原文を引用してラベルを一つ選んでください。"""


def select_pilot(frame: pd.DataFrame, per_common_class: int) -> pd.DataFrame:
    rare = frame[frame["answer"].isin(["-2", "-1"])]
    common = pd.concat(
        [
            group.sample(min(per_common_class, len(group)), random_state=2026)
            for _, group in frame[~frame["answer"].isin(["-2", "-1"])].groupby("answer")
        ]
    )
    return pd.concat([rare, common]).sort_values("id")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3.6:35b-a3b")
    parser.add_argument("--per-common-class", type=int, default=3)
    parser.add_argument("--max-id", type=int)
    parser.add_argument(
        "--tracks", nargs="+",
        choices=["direct", "structured", "guideline", "contrastive"],
        default=["direct", "structured", "guideline", "contrastive"],
    )
    args = parser.parse_args()
    frame = pd.read_parquet(ROOT / "data/raw/subtask2/JF-ICR_public_set.parquet")
    pilot = select_pilot(frame, args.per_common_class)
    if args.max_id is not None:
        pilot = pilot[pilot["id"] <= args.max_id]
    exemplar_path = ROOT / "data/derived/icr_contrastive_exemplars.jsonl"
    exemplar_map = {
        row["id"]: row["oof_exemplars"]
        for row in (json.loads(line) for line in exemplar_path.read_text().splitlines())
    }
    output_dir = ROOT / "artifacts/icr_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model = args.model.replace(":", "_").replace("/", "_")
    output_path = output_dir / f"{safe_model}.jsonl"
    failure_path = output_dir / f"{safe_model}.failures.jsonl"
    completed = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["id"], row["track"]))
    if failure_path.exists():
        for line in failure_path.read_text().splitlines():
            row = json.loads(line)
            completed.add((row["id"], row["track"]))

    for _, row in pilot.iterrows():
        example = parse_icr_query(row["query"])
        configured_tracks = [
            ("direct", direct_prompt(example.question, example.response), None),
            ("structured", structured_prompt(example.question, example.response), STRUCTURED_SCHEMA),
            ("guideline", guideline_prompt(example.question, example.response), STRUCTURED_SCHEMA),
            (
                "contrastive",
                contrastive_prompt(example.question, example.response, exemplar_map[int(row["id"])]),
                None,
            ),
        ]
        for track, prompt, schema in configured_tracks:
            if track not in args.tracks:
                continue
            if (int(row["id"]), track) in completed:
                continue
            started = time.time()
            try:
                prediction, raw = generate_json(args.model, prompt, *( [schema] if schema else [] ))
            except (json.JSONDecodeError, TimeoutError) as error:
                failure = {"id": int(row["id"]), "track": track, "error": repr(error)}
                with failure_path.open("a") as handle:
                    handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"id={row['id']} track={track} skipped={type(error).__name__}", flush=True)
                continue
            record = {
                "id": int(row["id"]), "gold": row["answer"], "track": track,
                "model": args.model, "prediction": prediction,
                "seconds": time.time() - started,
                "prompt_tokens": raw.get("prompt_eval_count"), "output_tokens": raw.get("eval_count"),
            }
            with output_path.open("a") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"id={row['id']} gold={row['answer']} track={track} pred={prediction['label']}", flush=True)

    results = pd.DataFrame(json.loads(line) for line in output_path.read_text().splitlines())
    results = results[results["id"].isin(pilot["id"]) & results["track"].isin(args.tracks)]
    for track, group in results.groupby("track"):
        predictions = group["prediction"].map(lambda item: item["label"]).tolist()
        print(track, "n=", len(group), "accuracy=", accuracy(group["gold"].tolist(), predictions))


if __name__ == "__main__":
    main()
