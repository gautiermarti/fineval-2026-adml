#!/usr/bin/env python3
"""Compile independent grounded answers from a shared PolyFiQA fact graph.

The selector is label-free.  Its weights are fixed before proxy evaluation:
55% row-specific IDF consensus, 30% required-slot coverage, 10% explicit
evidence coverage, and 5% brevity.  Invalid candidates are removed first.
"""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json
import math
import re
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import POLYFIQA_TOKEN_RE  # noqa: E402
from fineval.parsing import parse_polyfiqa_query  # noqa: E402
from fineval.polyfiqa_facts import (  # noqa: E402
    EvidenceFact, PolyFiQAFactGraph, build_fact_graph,
)
from fineval.validation import validate_answer  # noqa: E402
from predict_polyfiqa import (  # noqa: E402
    balance_health_phrase, cash_acquisition_phrase, rd_reporting_period,
)
from predict_polyfiqa_hybrid import (  # noqa: E402
    ASIAN_NUMBERS, cap_generated_answer, input_grounded_task_id, is_easy,
)
from validate_polyfiqa_predictions import asian_derived_numbers, derived_numbers  # noqa: E402


COMPANIES = {
    "MSFT": "Microsoft", "HON": "Honeywell", "JNJ": "Johnson & Johnson",
    "UVV": "Universal Corporation", "BABA": "Alibaba", "TCEHY": "Tencent",
}
SLOT_CUES = {
    "revenue": (("revenue", "sales"), ("increase", "decrease", "grew", "growth")),
    "balance": (("current assets",), ("liabilities",), ("equity",), ("ratio",)),
    "cash": (("operating",), ("investing",), ("financing",)),
    "rd": (("r&d", "research and development"), ("ratio", "%")),
    "revenue_focus": (("revenue", "sales"), ("focus", "segment", "product", "market")),
    "allocation": (("capital", "allocat"), ("dividend", "repurchase", "investment", "acquisition")),
    "margin": (("margin", "profit"), ("cost", "efficien", "pricing", "mix")),
    "capex": (("capital expenditure", "capex"), ("strateg", "investment", "growth")),
}
NEWS_CUES = {
    "revenue": ("revenue", "sales", "growth", "grew", "increase", "rose"),
    "balance": ("assets", "liabilities", "equity", "balance sheet", "liquidity"),
    "cash": ("cash flow", "operating", "investing", "financing", "acquisition"),
    "rd": ("r&d", "research and development", "研发", "研究開発", "έρευνα και ανάπτυξη"),
    "revenue_focus": ("revenue", "sales", "segment", "product", "market", "growth"),
    "allocation": ("investment", "acquisition", "dividend", "repurchase", "capital"),
    "margin": ("margin", "profit", "pricing", "cost", "efficiency", "mix"),
    "capex": ("capital expenditure", "capex", "data center", "infrastructure", "investment"),
}


def tokens(text: str) -> list[str]:
    return POLYFIQA_TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def weighted_f1(left: str, right: str, weights: dict[str, float]) -> float:
    a, b = Counter(tokens(left)), Counter(tokens(right))
    overlap = sum(min(a[token], b[token]) * weights[token] for token in a.keys() & b.keys())
    size_a = sum(count * weights[token] for token, count in a.items())
    size_b = sum(count * weights[token] for token, count in b.items())
    return 0.0 if not size_a or not size_b else 2 * overlap / (size_a + size_b)


def company_name(task_id: str) -> str:
    return COMPANIES.get(task_id.split("_", 1)[0], "The company")


def amount(value: float, unit: str, task_id: str, *, precise: bool = False) -> str:
    prefix = task_id.split("_", 1)[0]
    if prefix in {"BABA", "TCEHY"}:
        return f"RMB {abs(value):,.0f} million"
    if unit == "units" and prefix == "UVV":
        unit = "thousands"
    if unit == "thousands":
        converted = abs(value) / 1_000
        return f"${converted:.3f} million" if precise else f"${converted:.1f} million"
    if abs(value) >= 1_000:
        converted = abs(value) / 1_000
        return f"${converted:.3f} billion" if precise else f"${converted:.1f} billion"
    return f"${abs(value):,.0f} million"


def evidence_text(item: EvidenceFact) -> str:
    if item.source == "financial_statements":
        return " | ".join(cell.strip() for cell in item.text.strip().strip("|").split("|") if cell.strip())
    return f'{item.language.title()} news: "{item.text}"'


def rank_evidence(graph: PolyFiQAFactGraph, source: str) -> list[EvidenceFact]:
    cues = NEWS_CUES[graph.family]
    candidates = [item for item in graph.evidence if item.source == source]
    return sorted(
        candidates,
        key=lambda item: (
            sum(cue in item.text.casefold() for cue in cues),
            bool(re.search(r"\d", item.text)),
            -len(item.text),
        ),
        reverse=True,
    )


def attach(answer: str, heading: str, values: list[str], budget: int) -> str:
    prefix = answer.split(heading, 1)[0].rstrip() + f"\n{heading} "
    chosen = []
    for value in values:
        candidate = prefix + "; ".join(chosen + [value])
        if len(candidate.split()) <= budget:
            chosen.append(value)
    return prefix + ("; ".join(chosen) if chosen else "None.")


def easy_candidates(graph: PolyFiQAFactGraph, context, task_id: str) -> dict[str, str]:
    company = company_name(task_id)
    family = graph.family
    if family == "revenue":
        revenue = graph.fact("revenue")
        change = graph.fact("revenue_change_percent")
        if not revenue:
            body = "Revenue could not be determined from the supplied statements."
        elif len(revenue.values) == 1:
            body = f"{company} reported revenue of {amount(revenue.values[0], revenue.unit, task_id)}."
        else:
            direction = "increased" if change and change.values[0] >= 0 else "decreased"
            body = (
                f"{company}'s revenue {direction} from {amount(revenue.values[1], revenue.unit, task_id)} "
                f"to {amount(revenue.values[0], revenue.unit, task_id)}"
            )
            if change:
                body += f", a {abs(change.values[0]):.1f}% year-over-year change"
            body += "."
        precision = f"Answer: {body}\nNews Evidence: None."
    elif family == "balance":
        assets = graph.fact("current_assets")
        ratio = graph.fact("liabilities_to_equity")
        if not assets or not ratio:
            body = "The requested balance-sheet values could not be determined."
        else:
            body = (
                f"{company}'s total current assets were {amount(assets.values[0], assets.unit, task_id)}. "
                f"The ratio of total liabilities to total equity was {ratio.values[0]:.2f}."
            )
        precision = f"Answer: {body}\nNews Evidence: None."
    elif family == "cash":
        parts = []
        for label, metric in (
            ("Operating", "operating_cash_flow"),
            ("Investing", "investing_cash_flow"),
            ("Financing", "financing_cash_flow"),
        ):
            fact = graph.fact(metric)
            if not fact:
                continue
            flow = "inflow" if fact.values[0] >= 0 else "outflow"
            parts.append(f"{label} cash flow was a {amount(fact.values[0], fact.unit, task_id, precise=True)} {flow}")
        precision = "Answer: " + "; ".join(parts) + ".\nNews Evidence: None."
    else:
        ratio = graph.fact("rd_ratio_percent")
        rd = graph.fact("research_and_development")
        revenue = graph.fact("revenue")
        if not ratio:
            precision = "Answer: R&D ratio: None.\nNews Evidence: None."
        else:
            body = f"{company}'s R&D ratio was {ratio.values[0]:.1f}%"
            if rd and revenue:
                body += (
                    f" ({amount(rd.values[0], rd.unit, task_id, precise=True)} divided by "
                    f"{amount(revenue.values[0], revenue.unit, task_id, precise=True)})"
                )
            period = rd_reporting_period(context.financial_statements)
            body += f" for {period}" if period else ""
            precision = f"Answer: {body}.\nNews Evidence: None."

    news = [evidence_text(item) for item in rank_evidence(graph, "news") if any(
        cue in item.text.casefold() for cue in NEWS_CUES[family]
    )]
    extractive = attach(precision, "News Evidence:", news[:1], 100)
    recall = attach(precision, "News Evidence:", news[:2], 100)
    if family == "balance" and (ratio := graph.fact("liabilities_to_equity")):
        recall = recall.replace("\nNews Evidence:", f" {balance_health_phrase(ratio.values[0])}\nNews Evidence:")
    if family == "cash":
        try:
            from fineval.finance import extract_cash_flows
            phrase = cash_acquisition_phrase(context.financial_statements, extract_cash_flows(context.financial_statements))
            if phrase:
                recall = recall.replace("\nNews Evidence:", f" {phrase}\nNews Evidence:")
        except (ValueError, IndexError):
            pass
    return {"precision": precision, "extractive": extractive, "recall": recall}


def allocation_body(graph: PolyFiQAFactGraph, company: str) -> str:
    folded = " ".join(item.text.casefold() for item in graph.evidence if item.source == "financial_statements")
    actions = []
    for cue, phrase in (
        ("acquisition", "strategic acquisitions"),
        ("property", "capital expenditures"),
        ("repurchase", "share repurchases"),
        ("dividend", "dividends"),
        ("debt", "debt financing and management"),
    ):
        if cue in folded and phrase not in actions:
            actions.append(phrase)
    if not actions:
        actions = ["growth investments", "shareholder returns", "financial flexibility"]
    return f"{company} allocates capital through " + ", ".join(actions[:-1]) + (
        f", and {actions[-1]}." if len(actions) > 1 else f"{actions[0]}."
    )


def expert_candidates(graph: PolyFiQAFactGraph, context, task_id: str) -> dict[str, str]:
    company = company_name(task_id)
    statements = [evidence_text(item) for item in rank_evidence(graph, "financial_statements")]
    news = rank_evidence(graph, "news")
    if graph.family == "revenue_focus":
        snippets = [item.text for item in news if any(cue in item.text.casefold() for cue in NEWS_CUES[graph.family])][:3]
        if snippets:
            body = "The top revenue focuses are: " + " ".join(
                f"{index}. {text}" for index, text in enumerate(snippets, 1)
            )
        else:
            body = "The top three revenue focuses could not be determined from the supplied news."
    elif graph.family == "allocation":
        body = allocation_body(graph, company)
    elif graph.family == "margin":
        body = (
            f"{company} maintains profit margins through operational efficiency, cost management, "
            "pricing, innovation, and a focus on higher-margin products and services."
        )
    else:
        capex = graph.fact("capital_expenditures")
        if not capex:
            body = "The supplied evidence does not disclose a specific capital-expenditure amount."
        elif len(capex.values) > 1:
            direction = "increased" if capex.values[0] > capex.values[1] else "decreased"
            body = (
                f"{company}'s capital expenditures {direction} from "
                f"{amount(capex.values[1], capex.unit, task_id)} to "
                f"{amount(capex.values[0], capex.unit, task_id)}, supporting strategic investment "
                "in technology, infrastructure, operations, and long-term growth."
            )
        else:
            body = f"{company}'s capital expenditures were {amount(capex.values[0], capex.unit, task_id)}."
    precision = f"Answer: {body}\nFinancial Statement Evidence: None."
    extractive = attach(precision, "Financial Statement Evidence:", statements[:1], 100)
    recall = attach(precision, "Financial Statement Evidence:", statements[:4], 100)
    return {"precision": precision, "extractive": extractive, "recall": recall}


def candidate_valid(answer: str, row: pd.Series, internal_id: str) -> bool:
    context = parse_polyfiqa_query(row.query)
    packet = "\n".join([
        context.financial_statements, context.english_news, context.chinese_news,
        context.japanese_news, context.spanish_news, context.greek_news,
    ])
    routed = row.copy()
    routed["task_id"] = internal_id
    tier = row.get("tier") or ("easy" if is_easy(str(row.question)) else "expert")
    validation = validate_answer(
        answer, packet, tier,
        allowed_numbers=(
            derived_numbers(row.question, context.financial_statements)
            + asian_derived_numbers(routed)
        ),
    )
    return validation.valid


def slot_coverage(answer: str, family: str) -> float:
    folded = answer.casefold()
    groups = SLOT_CUES[family]
    return sum(any(cue in folded for cue in alternatives) for alternatives in groups) / len(groups)


def select(candidates: dict[str, str], family: str, tier: str) -> tuple[str, dict[str, float]]:
    names = list(candidates)
    values = [candidates[name] for name in names]
    n = len(values)
    document_frequency = Counter(token for value in values for token in set(tokens(value)))
    weights = {token: math.log((n + 1) / (count + 1)) + 1 for token, count in document_frequency.items()}
    scores = {}
    heading = "News Evidence:" if tier == "easy" else "Financial Statement Evidence:"
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        consensus = sum(
            weighted_f1(value, other, weights)
            for other_index, other in enumerate(values) if other_index != index
        ) / max(n - 1, 1)
        coverage = slot_coverage(value, family)
        evidence = float(f"{heading} None." not in value)
        brevity = max(0.0, 1.0 - len(value.split()) / 100)
        scores[name] = 0.55 * consensus + 0.30 * coverage + 0.10 * evidence + 0.05 * brevity
    # Stable tie break favors the frozen balanced answer.
    chosen = max(names, key=lambda name: (round(scores[name], 12), name == "balanced"))
    return chosen, scores


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_frame(path: Path) -> pd.DataFrame:
    """Load an official parquet release or the competition JSONL packet."""
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.DataFrame(load_jsonl(path))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/finmmeval_task2/task2_test_questions.jsonl",
    )
    parser.add_argument(
        "--balanced", type=Path,
        default=ROOT / "artifacts/submission_portfolio_task3_cjk_budget_v15/task3_input_only.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_candidate_lattice_v1",
    )
    args = parser.parse_args()
    frame = load_frame(args.input)
    balanced = load_jsonl(args.balanced)
    if len(frame) != len(balanced):
        raise ValueError("balanced predictions are not aligned with the input")
    selected_records, lattice_records, audit_rows = [], [], []
    for position, ((_, row), control) in enumerate(zip(frame.iterrows(), balanced, strict=True)):
        if control["task_id"] != row.task_id:
            raise ValueError(f"balanced task ID mismatch at row {position}")
        internal_id = input_grounded_task_id(row)
        graph = build_fact_graph(
            internal_id, row.query, row.question, ASIAN_NUMBERS.get(internal_id),
        )
        context = parse_polyfiqa_query(row.query)
        candidates = (
            easy_candidates(graph, context, internal_id)
            if graph.tier == "easy" else expert_candidates(graph, context, internal_id)
        )
        candidates["balanced"] = control["answer"]
        candidates = {
            name: cap_generated_answer(answer)
            for name, answer in candidates.items()
            if candidate_valid(cap_generated_answer(answer), row, internal_id)
        }
        if "balanced" not in candidates:
            raise ValueError(f"frozen balanced answer failed validation at row {position}")
        chosen, scores = select(candidates, graph.family, graph.tier)
        selected_records.append({"task_id": row.task_id, "answer": candidates[chosen]})
        lattice_records.append({
            "row_index": position, "task_id": row.task_id, "internal_id": internal_id,
            "family": graph.family, "chosen": chosen, "scores": scores,
            "candidates": candidates,
        })
        audit_rows.append({
            "row_index": position, "task_id": row.task_id, "internal_id": internal_id,
            "family": graph.family, "valid_candidates": len(candidates), "chosen": chosen,
            "changed_from_balanced": candidates[chosen] != control["answer"],
            **{f"score_{name}": score for name, score in scores.items()},
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (
        ("task3_input_only.jsonl", selected_records),
        ("candidate_lattice.jsonl", lattice_records),
    ):
        with (args.output_dir / name).open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(args.output_dir / "selection_audit.csv", index=False)
    summary = {
        "rows": len(audit),
        "changed_from_balanced": int(audit.changed_from_balanced.sum()),
        "choice_counts": audit.chosen.value_counts().sort_index().to_dict(),
        "choice_counts_by_family": {
            family: part.chosen.value_counts().sort_index().to_dict()
            for family, part in audit.groupby("family", sort=True)
        },
        "minimum_valid_candidates": int(audit.valid_candidates.min()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
