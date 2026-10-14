"""Auditable structured judgments for JF-ICR weak supervision."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping


LABELS = ("-2", "-1", "0", "+1", "+2")
RUBRIC_VERSION = "icr_factors_v1_20260805"

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "proposition": {"type": "string"},
        "future_action": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "direction": {"type": "string", "enum": ["commit", "neutral", "refuse"]},
        "strength": {
            "type": "string",
            "enum": ["strong", "qualified", "hedged", "none"],
        },
        "conditionality": {
            "type": "string",
            "enum": ["none", "conditional", "unknown"],
        },
        "temporal_scope": {
            "type": "string",
            "enum": ["permanent", "time_bounded", "unspecified"],
        },
        "responsiveness": {
            "type": "string",
            "enum": ["direct", "partial", "nonresponsive"],
        },
        "decisive_span": {"type": "string"},
        "counterevidence_span": {"type": "string"},
        "label": {"type": "string", "enum": list(LABELS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": [
        "proposition",
        "future_action",
        "direction",
        "strength",
        "conditionality",
        "temporal_scope",
        "responsiveness",
        "decisive_span",
        "counterevidence_span",
        "label",
        "confidence",
        "rationale",
    ],
    "additionalProperties": False,
}

COUNTERFACTUAL_SCHEMA = {
    "type": "object",
    "properties": {
        "source_span": {"type": "string"},
        "variants": {
            "type": "array",
            "minItems": 5,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(LABELS)},
                    "replacement_span": {"type": "string"},
                    "future_action": {
                        "type": "string", "enum": ["yes", "no", "unclear"],
                    },
                    "direction": {
                        "type": "string", "enum": ["commit", "neutral", "refuse"],
                    },
                    "strength": {
                        "type": "string",
                        "enum": ["strong", "qualified", "hedged", "none"],
                    },
                    "conditionality": {
                        "type": "string",
                        "enum": ["none", "conditional", "unknown"],
                    },
                    "temporal_scope": {
                        "type": "string",
                        "enum": ["permanent", "time_bounded", "unspecified"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "label", "replacement_span", "future_action", "direction",
                    "strength", "conditionality", "temporal_scope", "rationale",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["source_span", "variants"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AgreementDecision:
    accepted: bool
    label: str | None
    reason: str
    teachers: tuple[str, ...]
    rubrics: tuple[str, ...]
    judgment_count: int


@dataclass(frozen=True)
class DecisiveEventDecision:
    accepted: bool
    reason: str
    decisive_span: str | None
    factors: tuple[tuple[str, str], ...]
    teachers: tuple[str, ...]
    rubrics: tuple[str, ...]
    judgment_count: int


def normalize_evidence(text: str) -> str:
    """Normalize PDF line-wrap whitespace without changing Japanese content."""
    return re.sub(r"\s+", "", str(text or ""))


def locate_normalized_span(span: str, response: str) -> tuple[int, int] | None:
    """Locate a quote despite PDF whitespace inserted inside Japanese text."""
    normalized = []
    source_positions = []
    for index, character in enumerate(str(response or "")):
        if character.isspace():
            continue
        normalized.append(character)
        source_positions.append(index)
    target = normalize_evidence(span)
    if not target:
        return None
    joined = "".join(normalized)
    start = joined.find(target)
    if start < 0 or joined.find(target, start + 1) >= 0:
        return None
    return source_positions[start], source_positions[start + len(target) - 1] + 1


def protected_tokens(text: str) -> tuple[str, ...]:
    """Return values and Latin identifiers that a modality edit must not touch."""
    return tuple(re.findall(
        r"[-+]?\d[\d,.]*(?:%|％)?|[A-Za-z][A-Za-z0-9._/-]+", str(text or ""),
    ))


def evidence_is_grounded(span: str, response: str, *, allow_empty: bool = False) -> bool:
    normalized_span = normalize_evidence(span)
    if not normalized_span:
        return allow_empty
    return normalized_span in normalize_evidence(response)


def factor_consistency(judgment: Mapping[str, Any]) -> bool:
    """Reject impossible label/factor combinations without relabeling examples."""
    label = str(judgment.get("label", ""))
    direction = judgment.get("direction")
    strength = judgment.get("strength")
    future = judgment.get("future_action")
    temporal = judgment.get("temporal_scope")
    if label == "+2":
        return direction == "commit" and strength == "strong" and future == "yes"
    if label == "+1":
        return (
            direction == "commit"
            and strength in {"qualified", "hedged"}
            and future == "yes"
        )
    if label == "0":
        return direction == "neutral" and strength in {"hedged", "none"}
    if label == "-1":
        return (
            direction == "refuse"
            and strength in {"qualified", "hedged"}
            and temporal in {"time_bounded", "unspecified"}
            and future in {"yes", "unclear"}
        )
    if label == "-2":
        return direction == "refuse" and strength == "strong" and future in {"yes", "unclear"}
    return False


def validate_judgment(judgment: Mapping[str, Any], response: str) -> tuple[bool, str]:
    if not isinstance(judgment, Mapping):
        return False, "judgment_not_object"
    missing = [field for field in JUDGMENT_SCHEMA["required"] if field not in judgment]
    if missing:
        return False, "missing_fields:" + ",".join(missing)
    unexpected = sorted(set(judgment) - set(JUDGMENT_SCHEMA["properties"]))
    if unexpected:
        return False, "unexpected_fields:" + ",".join(unexpected)
    for field, specification in JUDGMENT_SCHEMA["properties"].items():
        value = judgment[field]
        if specification["type"] == "string" and not isinstance(value, str):
            return False, f"invalid_type:{field}"
        if specification["type"] == "number" and (
            not isinstance(value, (int, float)) or isinstance(value, bool)
        ):
            return False, f"invalid_type:{field}"
        if "enum" in specification and value not in specification["enum"]:
            return False, f"invalid_enum:{field}"
    try:
        confidence = float(judgment.get("confidence"))
    except (TypeError, ValueError):
        return False, "invalid_confidence"
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return False, "invalid_confidence"
    if not str(judgment.get("proposition", "")).strip():
        return False, "empty_proposition"
    if not evidence_is_grounded(str(judgment.get("decisive_span", "")), response):
        return False, "ungrounded_decisive_span"
    if not evidence_is_grounded(
        str(judgment.get("counterevidence_span", "")), response, allow_empty=True,
    ):
        return False, "ungrounded_counterevidence_span"
    if not factor_consistency(judgment):
        return False, "factor_label_inconsistency"
    return True, "valid"


def decide_agreement(
    records: Iterable[Mapping[str, Any]],
    response: str,
    *,
    minimum_teachers: int = 2,
    minimum_rubrics_per_teacher: int = 2,
) -> AgreementDecision:
    """Accept only rubric-invariant exact agreement across distinct teachers."""
    valid: list[Mapping[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for record in records:
        judgment = record.get("judgment", {})
        ok, reason = validate_judgment(judgment, response)
        if ok:
            valid.append(record)
        else:
            invalid_reasons[reason] += 1
    if not valid:
        reason = invalid_reasons.most_common(1)[0][0] if invalid_reasons else "no_judgments"
        return AgreementDecision(False, None, reason, (), (), 0)

    by_teacher: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in valid:
        teacher = str(record.get("teacher_id", "")).strip()
        rubric = str(record.get("rubric", "")).strip()
        if not teacher or not rubric:
            continue
        by_teacher[teacher].append(record)
    eligible = {
        teacher: rows
        for teacher, rows in by_teacher.items()
        if len({str(row["rubric"]) for row in rows}) >= minimum_rubrics_per_teacher
    }
    if len(eligible) < minimum_teachers:
        return AgreementDecision(
            False, None, "insufficient_teacher_rubric_coverage",
            tuple(sorted(eligible)),
            tuple(sorted({str(row.get("rubric", "")) for row in valid})),
            len(valid),
        )

    teacher_labels: dict[str, str] = {}
    for teacher, rows in eligible.items():
        labels = {str(row["judgment"]["label"]) for row in rows}
        if len(labels) != 1:
            return AgreementDecision(
                False, None, "within_teacher_rubric_disagreement",
                tuple(sorted(eligible)),
                tuple(sorted({str(row["rubric"]) for row in valid})),
                len(valid),
            )
        teacher_labels[teacher] = labels.pop()
    labels = set(teacher_labels.values())
    if len(labels) != 1:
        return AgreementDecision(
            False, None, "between_teacher_disagreement",
            tuple(sorted(eligible)),
            tuple(sorted({str(row["rubric"]) for row in valid})),
            len(valid),
        )
    label = labels.pop()
    accepted_teachers = tuple(sorted(eligible))
    accepted_rows = [row for teacher in accepted_teachers for row in eligible[teacher]]
    return AgreementDecision(
        True,
        label,
        "exact_teacher_rubric_agreement",
        accepted_teachers,
        tuple(sorted({str(row["rubric"]) for row in accepted_rows})),
        len(accepted_rows),
    )


DECISIVE_FACTOR_FIELDS = (
    "future_action", "direction", "strength", "conditionality",
    "temporal_scope", "responsiveness",
)

MODEL_FAMILY_MARKERS = (
    "qwen", "gemma", "llama", "mistral", "nemotron", "phi", "deepseek", "command-r",
    "aya", "internlm", "glm", "yi", "rinna", "elyza",
)


def canonical_model_family(model: str) -> str:
    """Return a conservative family key for a recorded teacher model."""
    normalized = re.sub(r"[^a-z0-9]+", "-", str(model).lower()).strip("-")
    for marker in MODEL_FAMILY_MARKERS:
        if marker in normalized:
            return marker
    return normalized.split("-", 1)[0] if normalized else ""


def audit_teacher_independence(
    records: Iterable[Mapping[str, Any]], *, minimum_teachers: int = 2,
) -> dict[str, Any]:
    """Fail closed on aliases, model drift, and same-family pseudo-teachers."""
    teacher_models: dict[str, set[str]] = defaultdict(set)
    teacher_revisions: dict[str, set[str]] = defaultdict(set)
    missing_identity = 0
    for record in records:
        teacher = str(record.get("teacher_id", "")).strip()
        model = str(record.get("model", "")).strip()
        revision = str(record.get("model_revision", "")).strip()
        if not teacher or not model or not revision:
            missing_identity += 1
            continue
        teacher_models[teacher].add(model)
        teacher_revisions[teacher].add(revision)

    drifting = {
        teacher: sorted(models) for teacher, models in teacher_models.items()
        if len(models) != 1
    }
    resolved = {
        teacher: next(iter(models)) for teacher, models in teacher_models.items()
        if len(models) == 1
    }
    revision_drift = {
        teacher: sorted(revisions)
        for teacher, revisions in teacher_revisions.items()
        if len(revisions) != 1
    }
    resolved_revisions = {
        teacher: next(iter(revisions))
        for teacher, revisions in teacher_revisions.items()
        if len(revisions) == 1
    }
    model_teachers: dict[str, list[str]] = defaultdict(list)
    for teacher, model in resolved.items():
        model_teachers[model].append(teacher)
    aliases = {
        model: sorted(teachers) for model, teachers in model_teachers.items()
        if len(teachers) != 1
    }
    families = {
        canonical_model_family(model) for model in resolved.values()
        if canonical_model_family(model)
    }
    passed = (
        missing_identity == 0
        and not drifting
        and not revision_drift
        and not aliases
        and len(resolved) >= minimum_teachers
        and len(resolved_revisions) >= minimum_teachers
        and len(set(resolved.values())) >= minimum_teachers
        and len(families) >= minimum_teachers
    )
    return {
        "passed": passed,
        "minimum_teachers": minimum_teachers,
        "teacher_models": dict(sorted(resolved.items())),
        "teacher_model_revisions": dict(sorted(resolved_revisions.items())),
        "model_families": sorted(families),
        "missing_identity_records": missing_identity,
        "teacher_model_drift": drifting,
        "teacher_revision_drift": revision_drift,
        "same_model_aliases": aliases,
    }


def decide_decisive_event_agreement(
    records: Iterable[Mapping[str, Any]],
    response: str,
    gold_label: str,
    *,
    minimum_teachers: int = 2,
    minimum_rubrics_per_teacher: int = 2,
) -> DecisiveEventDecision:
    """Require exact span and factor agreement for gold-anchored supervision.

    Label agreement is intentionally insufficient because every annotator is
    shown the known training label.  Each eligible teacher must instead return
    the same grounded decisive span and the same six modality factors across
    rubric views, and independent teachers must agree with one another.
    """
    valid: list[Mapping[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for record in records:
        judgment = record.get("judgment", {})
        ok, reason = validate_judgment(judgment, response)
        if ok and str(judgment.get("label")) != str(gold_label):
            ok, reason = False, "teacher_changed_fixed_gold_label"
        if ok:
            valid.append(record)
        else:
            invalid_reasons[reason] += 1
    if not valid:
        reason = invalid_reasons.most_common(1)[0][0] if invalid_reasons else "no_judgments"
        return DecisiveEventDecision(False, reason, None, (), (), (), 0)

    by_teacher: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in valid:
        teacher = str(record.get("teacher_id", "")).strip()
        rubric = str(record.get("rubric", "")).strip()
        if teacher and rubric:
            by_teacher[teacher].append(record)
    eligible = {
        teacher: rows for teacher, rows in by_teacher.items()
        if len({str(row["rubric"]) for row in rows}) >= minimum_rubrics_per_teacher
    }
    rubrics = tuple(sorted({str(row.get("rubric", "")) for row in valid}))
    if len(eligible) < minimum_teachers:
        return DecisiveEventDecision(
            False, "insufficient_teacher_rubric_coverage", None, (),
            tuple(sorted(eligible)), rubrics, len(valid),
        )

    teacher_views: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {}
    for teacher, rows in eligible.items():
        spans = {
            normalize_evidence(str(row["judgment"]["decisive_span"]))
            for row in rows
        }
        if len(spans) != 1:
            return DecisiveEventDecision(
                False, "within_teacher_span_disagreement", None, (),
                tuple(sorted(eligible)), rubrics, len(valid),
            )
        factors = {
            tuple((field, str(row["judgment"][field])) for field in DECISIVE_FACTOR_FIELDS)
            for row in rows
        }
        if len(factors) != 1:
            return DecisiveEventDecision(
                False, "within_teacher_factor_disagreement", None, (),
                tuple(sorted(eligible)), rubrics, len(valid),
            )
        teacher_views[teacher] = (spans.pop(), factors.pop())

    spans = {view[0] for view in teacher_views.values()}
    if len(spans) != 1:
        return DecisiveEventDecision(
            False, "between_teacher_span_disagreement", None, (),
            tuple(sorted(eligible)), rubrics, len(valid),
        )
    factor_sets = {view[1] for view in teacher_views.values()}
    if len(factor_sets) != 1:
        return DecisiveEventDecision(
            False, "between_teacher_factor_disagreement", None, (),
            tuple(sorted(eligible)), rubrics, len(valid),
        )
    normalized_span = spans.pop()
    location = locate_normalized_span(normalized_span, response)
    if location is None:
        return DecisiveEventDecision(
            False, "agreed_span_not_unique", None, (),
            tuple(sorted(eligible)), rubrics, len(valid),
        )
    left, right = location
    return DecisiveEventDecision(
        True, "exact_span_factor_agreement", response[left:right],
        factor_sets.pop(), tuple(sorted(eligible)), rubrics, len(valid),
    )


def decide_counterfactual_verification(
    records: Iterable[Mapping[str, Any]],
    response: str,
    target_label: str,
    generator_id: str,
) -> AgreementDecision:
    """Require independent exact recovery of a generated target label."""
    rows = list(records)
    if any(str(row.get("teacher_id", "")) == generator_id for row in rows):
        return AgreementDecision(
            False, None, "generator_cannot_self_verify", (), (), len(rows),
        )
    decision = decide_agreement(rows, response)
    if not decision.accepted:
        return decision
    if decision.label != target_label:
        return AgreementDecision(
            False, decision.label, "verified_label_misses_target",
            decision.teachers, decision.rubrics, decision.judgment_count,
        )
    return AgreementDecision(
        True, decision.label, "independent_target_recovery",
        decision.teachers, decision.rubrics, decision.judgment_count,
    )


def validate_counterfactual_lattice(
    lattice: Mapping[str, Any], response: str,
) -> tuple[bool, str, list[dict[str, Any]]]:
    """Construct minimal variants and prove that non-modality content is unchanged."""
    source_span = str(lattice.get("source_span", ""))
    location = locate_normalized_span(source_span, response)
    if location is None:
        return False, "source_span_not_unique_or_grounded", []
    left, right = location
    exact_source = response[left:right]
    if protected_tokens(exact_source):
        return False, "source_span_contains_protected_token", []
    variants = lattice.get("variants")
    if not isinstance(variants, list):
        return False, "variants_not_list", []
    labels = [str(item.get("label", "")) for item in variants if isinstance(item, Mapping)]
    if len(variants) != len(LABELS) or set(labels) != set(LABELS) or len(set(labels)) != len(LABELS):
        return False, "variants_must_cover_each_label_once", []

    original_protected = protected_tokens(response)
    replacements = set()
    constructed = []
    for item in variants:
        replacement = str(item.get("replacement_span", "")).strip()
        if not replacement or "\n" in replacement or len(replacement) > 160:
            return False, "invalid_replacement_span", []
        if not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", replacement):
            return False, "replacement_not_japanese", []
        normalized_replacement = normalize_evidence(replacement)
        if normalized_replacement in replacements:
            return False, "duplicate_replacement_span", []
        replacements.add(normalized_replacement)
        factors = {
            "label": item.get("label"),
            "future_action": item.get("future_action"),
            "direction": item.get("direction"),
            "strength": item.get("strength"),
            "temporal_scope": item.get("temporal_scope"),
        }
        if not factor_consistency(factors):
            return False, "counterfactual_factor_label_inconsistency", []
        candidate = response[:left] + replacement + response[right:]
        if protected_tokens(candidate) != original_protected:
            return False, "protected_token_changed", []
        constructed.append({
            "target_label": item["label"],
            "response": candidate,
            "source_span": exact_source,
            "replacement_span": replacement,
            "factors": {
                key: item[key]
                for key in (
                    "future_action", "direction", "strength", "conditionality",
                    "temporal_scope",
                )
            },
            "rationale": item.get("rationale", ""),
        })
    constructed.sort(key=lambda row: LABELS.index(row["target_label"]))
    return True, "valid", constructed
