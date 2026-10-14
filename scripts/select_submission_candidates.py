#!/usr/bin/env python3
"""Select one candidate per task using labels-free diagnostics and policy gates."""

from argparse import ArgumentParser
from pathlib import Path
import json


def choose_candidates(
    manifest: dict,
    *,
    allow_public_answer_reuse: bool = False,
    hard_word_limit: bool = False,
) -> dict:
    """Return auditable choices without reading gold labels or candidate scores."""
    available = {row["name"]: row for row in manifest.get("candidates", [])}
    support = manifest.get("decision_support", {})
    choices = {}

    if "task1" in support:
        diagnostics = support["task1"]
        overlap = float(diagnostics.get("public_overlap_fraction", 0.0))
        if allow_public_answer_reuse and overlap > 0:
            name = (
                "task1_grouped_lookup_safe"
                if "task1_grouped_lookup_safe" in available
                else "task1_lookup_safe"
            )
            reason = "public-answer reuse explicitly allowed and exact prompt overlap detected"
        else:
            name = (
                "task1_grouped_consensus_primary"
                if "task1_grouped_consensus_primary" in available
                else "task1_mixed_full_refit"
            )
            reason = (
                "policy-independent grouped-consensus textual primary"
                if name == "task1_grouped_consensus_primary"
                else "policy-independent prompt-routed full-data refit"
            )
        choices["task1"] = _selection(name, reason, available)

    if "task2" in support:
        diagnostics = support["task2"]
        semantic_overlap = float(diagnostics.get("semantic_overlap_fraction", 0.0))
        if allow_public_answer_reuse and semantic_overlap > 0:
            name = "task2_lookup_safe"
            reason = "public-answer reuse explicitly allowed and semantic packet overlap detected"
        else:
            name = "task2_japanese_ordinal"
            reason = (
                "policy-independent primary with the strongest source-disjoint "
                "validated ordinal decoding"
            )
        choices["task2"] = _selection(name, reason, available)

    if "task3" in support:
        diagnostics = support["task3"]
        packet_overlap = float(diagnostics.get("packet_overlap_fraction", 0.0))
        if allow_public_answer_reuse and packet_overlap > 0:
            name = "task3_overlap_capped" if hard_word_limit else "task3_overlap_enhanced"
            reason = (
                "public-answer reuse explicitly allowed; overlapping answers capped at 100 words"
                if hard_word_limit
                else "public-answer reuse explicitly allowed and exact document packet overlap detected"
            )
        else:
            name = "task3_input_only"
            reason = "policy-independent question-and-context-only system"
        choices["task3"] = _selection(name, reason, available)

    return {
        "team": manifest.get("team"),
        "source_manifest": manifest.get("built_at_utc"),
        "policy": {
            "allow_public_answer_reuse": allow_public_answer_reuse,
            "hard_word_limit": hard_word_limit,
        },
        "selected": choices,
    }


def _selection(name: str, reason: str, available: dict) -> dict:
    if name not in available:
        raise ValueError(f"required candidate is absent from manifest: {name}")
    candidate = available[name]
    return {
        "name": name,
        "path": candidate["path"],
        "sha256": candidate["sha256"],
        "reason": reason,
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-public-answer-reuse", action="store_true")
    parser.add_argument("--hard-word-limit", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    result = choose_candidates(
        manifest,
        allow_public_answer_reuse=args.allow_public_answer_reuse,
        hard_word_limit=args.hard_word_limit,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
