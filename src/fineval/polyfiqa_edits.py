"""Sparse complete-answer edits for grounded PolyFiQA candidates."""

from __future__ import annotations

import re


EVIDENCE_RE = re.compile(r"(?m)(?:^|\n)(News Evidence:|Financial Statement Evidence:)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def split_answer(answer: str) -> tuple[str, str]:
    """Split answer core from its evidence section without changing text."""
    match = EVIDENCE_RE.search(answer)
    if match is None:
        return answer.strip(), ""
    heading_start = match.start(1)
    return answer[:heading_start].rstrip(), answer[heading_start:].strip()


def sentences(core: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(core.strip()) if part.strip()]


def join_answer(core: str, evidence: str) -> str:
    return core.rstrip() + (("\n" + evidence.strip()) if evidence.strip() else "")


def sparse_edits(baseline: str, alternatives: dict[str, str]) -> dict[str, str]:
    """Enumerate local edits around ``baseline`` and deduplicate exact outputs."""
    baseline_core, baseline_evidence = split_answer(baseline)
    baseline_sentences = sentences(baseline_core)
    candidates = {"baseline": baseline}
    for name, answer in alternatives.items():
        if answer == baseline:
            continue
        alternative_core, alternative_evidence = split_answer(answer)
        alternative_sentences = sentences(alternative_core)
        candidates[f"full:{name}"] = answer
        if alternative_core and baseline_evidence:
            candidates[f"core:{name}"] = join_answer(
                alternative_core, baseline_evidence,
            )
        if baseline_core and alternative_evidence:
            candidates[f"evidence:{name}"] = join_answer(
                baseline_core, alternative_evidence,
            )
        if baseline_sentences and alternative_sentences:
            candidates[f"first_sentence:{name}"] = join_answer(
                " ".join([alternative_sentences[0], *baseline_sentences[1:]]),
                baseline_evidence,
            )
    if len(baseline_sentences) > 1:
        for index in range(len(baseline_sentences)):
            retained = baseline_sentences[:index] + baseline_sentences[index + 1:]
            candidates[f"delete_sentence:{index}"] = join_answer(
                " ".join(retained), baseline_evidence,
            )

    unique = {}
    seen = set()
    for name, answer in candidates.items():
        if answer in seen:
            continue
        seen.add(answer)
        unique[name] = answer
    return unique
