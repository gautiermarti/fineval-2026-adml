#!/usr/bin/env python3
"""Induce and cross-company evaluate strict PolyFiQA reference programs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import sys
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fineval.metrics import rouge1_f1  # noqa: E402
from fineval.polyfiqa_facts import build_fact_graph  # noqa: E402
from fineval.polyfiqa_programs import (  # noqa: E402
    COMPANY_NAMES, NUMBER_RE, answer_core, execute_program, induce_program,
    semantic_state, task_company,
)
from predict_polyfiqa_hybrid import cap_generated_answer  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def item_score(reference: str, prediction: str) -> float:
    return rouge1_f1([reference], [prediction])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ROOT / "data/external/polyfiqa_easy_july/test-00000-of-00001.parquet",
    )
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "artifacts/polyfiqa_easy_july/hybrid_input_only_cjk_budget_v15.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts/polyfiqa_reference_programs_v1",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).reset_index(drop=True)
    baseline = load_jsonl(args.baseline)
    if len(frame) != len(baseline):
        raise ValueError("input and baseline are not aligned")
    graphs = [
        build_fact_graph(row.task_id, row.query, row.question)
        for row in frame.itertuples()
    ]
    programs = []
    induction_reasons = Counter()
    for position, (row, graph) in enumerate(zip(frame.itertuples(), graphs, strict=True)):
        program, reason = induce_program(row.task_id, row.answer, graph)
        induction_reasons[reason] += 1
        if program is not None:
            programs.append((position, program))

    rows = []
    selected_outputs = []
    oracle_outputs = []
    for position, (row, graph, control) in enumerate(
        zip(frame.itertuples(), graphs, baseline, strict=True)
    ):
        if str(control["task_id"]) != str(row.task_id):
            raise ValueError(f"baseline mismatch at row {position}")
        evidence = control["answer"].split("\n", 1)[1] if "\n" in control["answer"] else "News Evidence: None."
        candidates = {"baseline": control["answer"]}
        program_candidates = []
        for donor_position, program in programs:
            if donor_position == position or program.source_company == task_company(row.task_id):
                continue
            core, reason = execute_program(program, graph, row.task_id)
            if core is None:
                continue
            answer = cap_generated_answer(core.rstrip() + "\n" + evidence)
            key = f"program:{donor_position}"
            candidates[key] = answer
            program_candidates.append((key, answer, program.source_company))

        # A label-free medoid requires evidence from at least two independent
        # donor companies. Unlike the rejected v1 lattice, these candidates
        # arise from independently authored historical references.
        donor_company_count = len({company for _, _, company in program_candidates})
        if donor_company_count >= 2:
            values = list(dict.fromkeys(answer for _, answer, _ in program_candidates))
            medoid = max(values, key=lambda value: (
                sum(item_score(value, other) for other in values) / max(len(values) - 1, 1),
                -len(value), value,
            ))
            selected_name = next(
                name for name, answer, _ in program_candidates if answer == medoid
            )
        else:
            selected_name = "baseline"
        score_by_name = {
            name: item_score(row.answer, answer) for name, answer in candidates.items()
        }
        oracle_name = max(score_by_name, key=score_by_name.get)
        selected_outputs.append({"task_id": row.task_id, "answer": candidates[selected_name]})
        oracle_outputs.append({"task_id": row.task_id, "answer": candidates[oracle_name]})
        rows.append({
            "row_index": position,
            "task_id": row.task_id,
            "family": graph.family,
            "candidate_count": len(candidates),
            "donor_company_count": donor_company_count,
            "selected": selected_name,
            "oracle": oracle_name,
            "baseline_score": score_by_name["baseline"],
            "selected_score": score_by_name[selected_name],
            "oracle_score": score_by_name[oracle_name],
        })

    audit = pd.DataFrame(rows)

    def execute_full(program, target_position: int) -> str | None:
        target = frame.iloc[target_position]
        target_graph = graphs[target_position]
        if program.source_company == task_company(target.task_id):
            return None
        core, _ = execute_program(program, target_graph, target.task_id)
        if core is None:
            return None
        control = baseline[target_position]["answer"]
        evidence = control.split("\n", 1)[1] if "\n" in control else "News Evidence: None."
        return cap_generated_answer(core.rstrip() + "\n" + evidence)

    # Fully nested company-held-out program quality. For held company H, a
    # program authored by A is scored only on B/C. It can route H only if it has
    # positive mean delta on at least six rows spanning both B and C.
    companies = sorted({task_company(task_id) for task_id in frame.task_id})
    nested_outputs: list[dict | None] = [None] * len(frame)
    nested_audit = []
    for held_company in companies:
        eligible_programs = [
            (source_position, program)
            for source_position, program in programs
            if program.source_company != held_company
        ]
        quality = {}
        for source_position, program in eligible_programs:
            deltas = []
            validation_companies = set()
            for target_position, target in frame.iterrows():
                target_company = task_company(target.task_id)
                if target_company in {held_company, program.source_company}:
                    continue
                candidate = execute_full(program, target_position)
                if candidate is None:
                    continue
                delta = item_score(target.answer, candidate) - item_score(
                    target.answer, baseline[target_position]["answer"],
                )
                deltas.append(delta)
                validation_companies.add(target_company)
            if len(deltas) >= 6 and len(validation_companies) >= 2:
                quality[source_position] = {
                    "mean_delta": sum(deltas) / len(deltas),
                    "minimum_company_count": len(validation_companies),
                    "validation_rows": len(deltas),
                }
        for target_position, target in frame.iterrows():
            if task_company(target.task_id) != held_company:
                continue
            choices = []
            for source_position, program in eligible_programs:
                estimate = quality.get(source_position)
                if estimate is None or estimate["mean_delta"] <= 0:
                    continue
                candidate = execute_full(program, target_position)
                if candidate is not None:
                    choices.append((
                        estimate["mean_delta"], estimate["validation_rows"],
                        -len(candidate), source_position, candidate,
                    ))
            if choices:
                chosen = max(choices)
                selected_answer = chosen[-1]
                source_position = chosen[-2]
                estimated_delta = chosen[0]
            else:
                selected_answer = baseline[target_position]["answer"]
                source_position = None
                estimated_delta = 0.0
            nested_outputs[target_position] = {
                "task_id": target.task_id, "answer": selected_answer,
            }
            reference = target.answer
            nested_audit.append({
                "row_index": target_position,
                "task_id": target.task_id,
                "held_company": held_company,
                "source_program_row": source_position,
                "estimated_training_delta": estimated_delta,
                "baseline_score": item_score(reference, baseline[target_position]["answer"]),
                "nested_score": item_score(reference, selected_answer),
            })
    if any(record is None for record in nested_outputs):
        raise AssertionError("nested company routing did not cover every row")
    nested_frame = pd.DataFrame(nested_audit).sort_values("row_index")

    def delex_style(text: str) -> str:
        value = answer_core(text)
        for company in sorted(COMPANY_NAMES.values(), key=len, reverse=True):
            value = re.sub(re.escape(company), " COMPANY ", value, flags=re.IGNORECASE)
        value = re.sub(r"\b(?:19|20)\d{2}\b", " YEAR ", value)
        return NUMBER_RE.sub(" VALUE ", value)

    # Nested reference-prior selector: estimate expected realization overlap
    # only from delexicalized references of non-held companies with the same
    # family and financial trend state. A donor's own company is excluded when
    # scoring its candidate, preventing the template from voting for itself.
    reference_prior_outputs: list[dict | None] = [None] * len(frame)
    reference_prior_audit = []
    for held_company in companies:
        held_programs = [
            (source_position, program)
            for source_position, program in programs
            if program.source_company != held_company
        ]
        for target_position, target in frame.iterrows():
            if task_company(target.task_id) != held_company:
                continue
            target_graph = graphs[target_position]
            choices = [("baseline", baseline[target_position]["answer"], None)]
            for source_position, program in held_programs:
                candidate = execute_full(program, target_position)
                if candidate is not None:
                    choices.append((f"program:{source_position}", candidate, program.source_company))
            scored_choices = []
            for name, candidate, donor_company in choices:
                reference_positions = [
                    index for index, reference in frame.iterrows()
                    if task_company(reference.task_id) != held_company
                    and (donor_company is None or task_company(reference.task_id) != donor_company)
                    and graphs[index].family == target_graph.family
                    and semantic_state(graphs[index]) == semantic_state(target_graph)
                ]
                reference_companies = {
                    task_company(frame.iloc[index].task_id) for index in reference_positions
                }
                minimum_companies = 2 if donor_company is not None else 3
                if len(reference_positions) < 4 or len(reference_companies) < minimum_companies:
                    continue
                candidate_style = delex_style(candidate)
                expected = sum(
                    item_score(delex_style(frame.iloc[index].answer), candidate_style)
                    for index in reference_positions
                ) / len(reference_positions)
                scored_choices.append((expected, -len(candidate_style), name, candidate))
            if scored_choices:
                expected, _, selected_name, selected_answer = max(scored_choices)
            else:
                expected, selected_name = 0.0, "baseline"
                selected_answer = baseline[target_position]["answer"]
            reference_prior_outputs[target_position] = {
                "task_id": target.task_id, "answer": selected_answer,
            }
            reference_prior_audit.append({
                "row_index": target_position,
                "task_id": target.task_id,
                "held_company": held_company,
                "selected": selected_name,
                "expected_delex_reference_overlap": expected,
                "baseline_score": item_score(target.answer, baseline[target_position]["answer"]),
                "selected_score": item_score(target.answer, selected_answer),
            })
    if any(record is None for record in reference_prior_outputs):
        raise AssertionError("reference-prior routing did not cover every row")
    reference_prior_frame = pd.DataFrame(reference_prior_audit).sort_values("row_index")
    summary = {
        "rows": len(frame),
        "induced_programs": len(programs),
        "induction_reasons": dict(induction_reasons),
        "rows_with_program_candidate": int((audit.candidate_count > 1).sum()),
        "rows_with_two_donor_companies": int((audit.donor_company_count >= 2).sum()),
        "baseline_rouge": float(audit.baseline_score.mean()),
        "label_free_selected_rouge": float(audit.selected_score.mean()),
        "nested_company_selected_rouge": float(nested_frame.nested_score.mean()),
        "oracle_lattice_rouge": float(audit.oracle_score.mean()),
        "oracle_improved_rows": int((audit.oracle_score > audit.baseline_score).sum()),
        "selected_changed_rows": int((audit.selected != "baseline").sum()),
        "nested_changed_rows": int(nested_frame.source_program_row.notna().sum()),
        "nested_improved_rows": int((nested_frame.nested_score > nested_frame.baseline_score).sum()),
        "nested_harmed_rows": int((nested_frame.nested_score < nested_frame.baseline_score).sum()),
        "reference_prior_selected_rouge": float(reference_prior_frame.selected_score.mean()),
        "reference_prior_changed_rows": int((reference_prior_frame.selected != "baseline").sum()),
        "reference_prior_improved_rows": int(
            (reference_prior_frame.selected_score > reference_prior_frame.baseline_score).sum()
        ),
        "reference_prior_harmed_rows": int(
            (reference_prior_frame.selected_score < reference_prior_frame.baseline_score).sum()
        ),
        "reference_prior_by_held_company": {
            company: {
                "rows": len(group),
                "baseline": float(group.baseline_score.mean()),
                "selected": float(group.selected_score.mean()),
                "changed": int((group.selected != "baseline").sum()),
            }
            for company, group in reference_prior_frame.groupby("held_company")
        },
        "nested_by_held_company": {
            company: {
                "rows": len(group),
                "baseline": float(group.baseline_score.mean()),
                "nested": float(group.nested_score.mean()),
                "changed": int(group.source_program_row.notna().sum()),
            }
            for company, group in nested_frame.groupby("held_company")
        },
        "by_family": {
            family: {
                "rows": len(group),
                "baseline": float(group.baseline_score.mean()),
                "selected": float(group.selected_score.mean()),
                "oracle": float(group.oracle_score.mean()),
                "candidate_rows": int((group.candidate_count > 1).sum()),
            }
            for family, group in audit.groupby("family")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "programs.jsonl").open("w") as handle:
        for position, program in programs:
            handle.write(json.dumps({"row_index": position, **program.to_dict()}) + "\n")
    with (args.output_dir / "selected.jsonl").open("w") as handle:
        for record in selected_outputs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "oracle.jsonl").open("w") as handle:
        for record in oracle_outputs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "nested_company_selected.jsonl").open("w") as handle:
        for record in nested_outputs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (args.output_dir / "reference_prior_selected.jsonl").open("w") as handle:
        for record in reference_prior_outputs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    audit.to_csv(args.output_dir / "audit.csv", index=False)
    nested_frame.to_csv(args.output_dir / "nested_audit.csv", index=False)
    reference_prior_frame.to_csv(args.output_dir / "reference_prior_audit.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
