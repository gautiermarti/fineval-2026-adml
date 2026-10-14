#!/usr/bin/env python3
"""Rule-based GRFinNUM baseline; emits the organizer's entity,type answer format."""

from argparse import ArgumentParser
from pathlib import Path
import re
import json

import pandas as pd

NUMBER = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?|\d+(?:[,.]\d+)?)(?![\w])",
    re.IGNORECASE,
)
DATE = re.compile(r"(?<![\w])(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?(?![\w])")


def classify(span: str, left: str, right: str) -> str:
    context = f"{left[-90:]} {span} {right[:90]}".casefold()
    compact = span.casefold().replace(" ", "")
    if "%" in span or "‰" in span or re.search(r"ποσοστ|ρυθμ|margin|rate", context):
        return "ΠΟΣΟΣΤΑ"
    if re.search(r"€|\$|£|¥|ευρ|δολ|λίρ|\bεκ\.?\b|\bεκατ\.?\b|\bδισ\.?\b|million|billion", context):
        return "ΧΡΗΜΑΤΑ"
    if re.search(r"\b(?:19|20)\d{2}\b", compact) or re.search(
        r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?", span
    ) or re.search(r"χρήσ|έτος|έτη|τρίμην|μήν|ημέρ|περίοδο|year|quarter", context):
        return "ΧΡΟΝΙΚΑ"
    if re.search(r"καταστ.{0,4}μα|εργαζ|υπάλληλ|άτομ|μετοχ|μονάδ|εταιρεί|χώρ|προϊόν|τεμάχ|κατηγορ|θυγατρ|τ\.μ|mw|ισχύος|στην\s+κύπρο", context):
        return "ΠΟΣΟΤΗΤΕΣ"
    return "ΑΛΛΑ"


def predict(text: str) -> list[tuple[str, str]]:
    rows = []
    occupied = []
    for match in DATE.finditer(text):
        occupied.append((match.start(), match.end()))
        for part in match.groups():
            if part is None:
                continue
            # Locate each date component so the emitted entity is exactly the
            # numeric token used by the GRFinNUM annotations.
            start = text.find(part, match.start(), match.end())
            rows.append((part, "ΧΡΟΝΙΚΑ", start))
    for match in NUMBER.finditer(text):
        if any(a <= match.start() < b for a, b in occupied):
            continue
        span = match.group(0).strip().rstrip(".,;:)")
        if not span:
            continue
        if match.end() < len(text) and text[match.end():].lstrip().startswith(("%", "‰")):
            span += text[match.end():].lstrip()[0]
        rows.append((span, classify(span, text[:match.start()], text[match.end():]), match.start()))
    rows.sort(key=lambda item: item[2])
    return [(span, label) for span, label, _ in rows]


def parse_answer(answer: str) -> list[tuple[str, str]]:
    rows = []
    for line in str(answer or "").splitlines():
        if "," not in line:
            continue
        span, label = line.rsplit(",", 1)
        rows.append((span.strip(), label.strip()))
    return rows


def score(gold: list[tuple[str, str]], pred: list[tuple[str, str]]) -> tuple[float, float, float]:
    gold_set, pred_set = set(gold), set(pred)
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else float(not pred_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.input)
    with args.output.open("w") as handle:
        for position, row in frame.iterrows():
            pred = predict(row["text"])
            answer = "\n".join(f"{span}, {label}" for span, label in pred)
            record = {"row_index": int(position), "answer": answer, "entities": pred}
            if "answer" in row and pd.notna(row["answer"]):
                record["gold_f1"] = score(parse_answer(row["answer"]), pred)[2]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(frame)} predictions to {args.output}")
    if "answer" in frame:
        values = [score(parse_answer(row.answer), predict(row.text))[2] for _, row in frame.iterrows()]
        print(f"macro_entity_f1={sum(values)/len(values):.6f}")


if __name__ == "__main__":
    main()
