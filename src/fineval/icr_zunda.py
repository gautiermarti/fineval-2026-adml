"""Parsing and input-only summaries for Zunda extended-modality output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class ZundaEvent:
    event_id: int
    token_index: int
    surface: str
    lemma: str
    source: str
    tense: str
    conditionality: str
    modality: str
    actuality: str
    evaluation: str
    focus: str

    def to_dict(self) -> dict:
        return asdict(self)


def split_zunda_blocks(output: str) -> list[str]:
    """Split a batched Zunda stream into non-empty sentence blocks."""
    return [part.strip() for part in re.split(r"(?m)^EOS\s*$", output) if part.strip()]


def parse_zunda_block(block: str) -> tuple[list[ZundaEvent], list[str]]:
    """Parse event labels and token-aligned functional-expression labels."""
    raw_events: list[list[str]] = []
    functional: list[str] = []
    tokens: list[tuple[str, str]] = []
    for line in block.splitlines():
        if line.startswith("#EVENT"):
            fields = line.split("\t")
            if len(fields) != 9:
                raise ValueError(f"unexpected Zunda event line: {line!r}")
            raw_events.append(fields)
        elif line.startswith("#FUNCEXP\t"):
            functional = line.split("\t", 1)[1].split(",")
        elif line and not line.startswith(("#", "*", "EOS")):
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            morphology = fields[1].split(",")
            lemma = morphology[6] if len(morphology) > 6 else fields[0]
            tokens.append((fields[0], lemma if lemma != "*" else fields[0]))

    events = []
    for fields in raw_events:
        token_index = int(fields[1])
        if not 0 <= token_index < len(tokens):
            raise ValueError(
                f"Zunda event token {token_index} outside {len(tokens)} tokens"
            )
        event_id = int(fields[0].removeprefix("#EVENT"))
        surface, lemma = tokens[token_index]
        events.append(ZundaEvent(
            event_id=event_id,
            token_index=token_index,
            surface=surface,
            lemma=lemma,
            source=fields[2],
            tense=fields[3],
            conditionality=fields[4],
            modality=fields[5],
            actuality=fields[6],
            evaluation=fields[7],
            focus=fields[8],
        ))
    return events, functional


def zunda_signature(candidates: list[dict]) -> dict[str, float]:
    """Create compact query-conditioned multiple-instance modality features.

    Candidate ranking and weighting use only the question/response text and
    Zunda predictions.  No JF-ICR label or source statistic enters this step.
    """
    result: dict[str, float] = {
        "zunda:candidates": float(len(candidates)),
        "zunda:events": 0.0,
        "zunda:no_event_candidates": 0.0,
    }
    if not candidates:
        return result

    def add_event(prefix: str, event: dict, weight: float = 1.0) -> None:
        for field in (
            "tense", "conditionality", "modality", "actuality",
            "evaluation", "focus",
        ):
            value = str(event.get(field, ""))
            result[f"{prefix}:{field}={value}"] = (
                result.get(f"{prefix}:{field}={value}", 0.0) + weight
            )

    all_events: list[tuple[dict, dict]] = []
    for candidate in candidates:
        events = candidate.get("zunda_events", [])
        if not events:
            result["zunda:no_event_candidates"] += 1.0
        for event in events:
            all_events.append((candidate, event))
            add_event("global", event)
        for expression in candidate.get("functional_expressions", []):
            if expression != "O":
                category = expression.split(":", 1)[-1]
                key = f"global:function={category}"
                result[key] = result.get(key, 0.0) + 1.0

    result["zunda:events"] = float(len(all_events))
    denominator = max(float(len(all_events)), 1.0)
    for key in list(result):
        if key.startswith("global:"):
            result[key] /= denominator

    ranked_input = sorted(
        candidates,
        key=lambda row: (-float(row.get("input_only_score", 0.0)), int(row["start"])),
    )
    ranked_overlap = sorted(
        candidates,
        key=lambda row: (-float(row.get("question_overlap", 0.0)), int(row["start"])),
    )
    ranked_semantic = sorted(
        candidates,
        key=lambda row: (-float(row.get("ruri_similarity", -1.0)), int(row["start"])),
    )
    ranked_views = [
        ("top_input", ranked_input[0]),
        ("top_overlap", ranked_overlap[0]),
    ]
    if "ruri_similarity" in ranked_semantic[0]:
        ranked_views.append(("top_semantic", ranked_semantic[0]))
        semantic_scores = [float(row["ruri_similarity"]) for row in ranked_semantic]
        result["semantic:max"] = semantic_scores[0]
        result["semantic:margin"] = (
            semantic_scores[0] - semantic_scores[1]
            if len(semantic_scores) > 1 else semantic_scores[0]
        )
    for prefix, candidate in ranked_views:
        result[f"{prefix}:question_overlap"] = float(candidate.get("question_overlap", 0.0))
        result[f"{prefix}:input_score"] = float(candidate.get("input_only_score", 0.0))
        result[f"{prefix}:ruri_similarity"] = float(candidate.get("ruri_similarity", 0.0))
        result[f"{prefix}:position"] = float(candidate.get("start", 0))
        result[f"{prefix}:future_cues"] = float(candidate.get("future_cues", 0))
        result[f"{prefix}:commit_cues"] = float(candidate.get("commit_cues", 0))
        result[f"{prefix}:hedge_cues"] = float(candidate.get("hedge_cues", 0))
        result[f"{prefix}:refusal_cues"] = float(candidate.get("refusal_cues", 0))
        result[f"{prefix}:conditional_cues"] = float(candidate.get("conditional_cues", 0))
        for event in candidate.get("zunda_events", []):
            add_event(prefix, event)

    # A soft query-conditioned view avoids trusting one brittle event boundary.
    total_weight = 0.0
    for candidate, event in all_events:
        weight = 0.1 + max(0.0, float(candidate.get("question_overlap", 0.0)))
        weight += 0.05 * max(0.0, float(candidate.get("input_only_score", 0.0)))
        total_weight += weight
        add_event("query_weighted", event, weight)
    if total_weight:
        for key in list(result):
            if key.startswith("query_weighted:"):
                result[key] /= total_weight

    # Ruri semantic retrieval is a second, independently trained view of which
    # clause answers the question.  Normalize within each response so long
    # responses do not dominate merely because they contain more candidates.
    if "ruri_similarity" in ranked_semantic[0]:
        floor = min(float(row["ruri_similarity"]) for row in candidates)
        semantic_weight = 0.0
        for candidate, event in all_events:
            weight = 0.05 + max(0.0, float(candidate["ruri_similarity"]) - floor)
            semantic_weight += weight
            add_event("semantic_weighted", event, weight)
        if semantic_weight:
            for key in list(result):
                if key.startswith("semantic_weighted:"):
                    result[key] /= semantic_weight
    return result
