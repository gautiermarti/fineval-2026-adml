"""Input-only event candidates for Japanese implicit-commitment recognition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True)
class EventCandidate:
    start: int
    end: int
    text: str
    question_overlap: float
    future_cues: int
    commit_cues: int
    hedge_cues: int
    refusal_cues: int
    conditional_cues: int
    historical_cues: int
    input_only_score: float

    def to_dict(self) -> dict:
        return asdict(self)


_CUES = {
    "future": (
        "今後", "将来", "来期", "中長期", "予定", "目標", "方針", "見通し",
    ),
    "commit": (
        "実施する", "行う", "取り組む", "進める", "目指す", "めざす",
        "計画して", "決定した", "継続する", "強化する", "拡大する",
    ),
    "hedge": (
        "考えている", "想定して", "可能性", "検討して", "見込んで",
        "期待して", "必要に応じ", "現時点", "方向で", "意向",
    ),
    "refusal": (
        "回答を控", "開示でき", "答えられ", "申し上げられ", "予定はない",
        "予定はありません",
        "考えていない", "わけではない", "困難", "未定", "決まっていない",
    ),
    "conditional": (
        "場合", "次第", "条件", "前提", "状況を見", "可能であれば",
        "環境によ", "必要があれば", "検討した上で",
    ),
    "historical": (
        "実施した", "行った", "取り組んだ", "であった", "だった", "結果",
        "これまで", "従来", "前期", "昨年",
    ),
}


def _ngrams(text: str, width: int = 2) -> set[str]:
    compact = re.sub(r"[^\w\u3040-\u30ff\u3400-\u9fff]", "", text.casefold())
    if len(compact) < width:
        return {compact} if compact else set()
    return {compact[index:index + width] for index in range(len(compact) - width + 1)}


def question_overlap(question: str, event: str) -> float:
    left, right = _ngrams(question), _ngrams(event)
    return len(left & right) / len(left) if left else 0.0


def _base_spans(response: str) -> list[tuple[int, int]]:
    spans = []
    for match in re.finditer(r"[^。！？!?\n]+(?:[。！？!?]+|\n|$)", response):
        left, right = match.span()
        while left < right and response[left].isspace():
            left += 1
        while right > left and response[right - 1].isspace():
            right -= 1
        if right > left:
            spans.append((left, right))
    return spans


def _subspans(response: str, left: int, right: int) -> list[tuple[int, int]]:
    """Add long-clause alternatives at Japanese discourse connectives."""
    if right - left < 70:
        return []
    text = response[left:right]
    boundaries = [0]
    for match in re.finditer(r"(?:が|ため|ので|一方|ただし|なお|ものの)、", text):
        boundaries.append(match.end())
    boundaries.append(len(text))
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        absolute_left, absolute_right = left + start, left + end
        while absolute_left < absolute_right and response[absolute_left].isspace():
            absolute_left += 1
        while absolute_right > absolute_left and response[absolute_right - 1].isspace():
            absolute_right -= 1
        if absolute_right - absolute_left >= 12:
            result.append((absolute_left, absolute_right))
    return result if len(result) > 1 else []


def event_candidates(question: str, response: str) -> list[EventCandidate]:
    """Return exact-substring event candidates ranked without using a label."""
    spans = _base_spans(response)
    spans.extend(
        subspan for left, right in list(spans)
        for subspan in _subspans(response, left, right)
    )
    seen = set()
    candidates = []
    for left, right in spans:
        if (left, right) in seen:
            continue
        seen.add((left, right))
        text = response[left:right]
        counts = {
            name: sum(text.count(cue) for cue in cues)
            for name, cues in _CUES.items()
        }
        overlap = question_overlap(question, text)
        score = (
            2.0 * overlap
            + 0.45 * counts["future"]
            + 0.55 * counts["commit"]
            + 0.45 * counts["hedge"]
            + 0.75 * counts["refusal"]
            + 0.35 * counts["conditional"]
            - 0.10 * counts["historical"]
        )
        candidates.append(EventCandidate(
            start=left, end=right, text=text, question_overlap=overlap,
            future_cues=counts["future"], commit_cues=counts["commit"],
            hedge_cues=counts["hedge"], refusal_cues=counts["refusal"],
            conditional_cues=counts["conditional"],
            historical_cues=counts["historical"], input_only_score=score,
        ))
    return sorted(
        candidates,
        key=lambda item: (-item.input_only_score, item.start, item.end),
    )


def candidates_are_exact(response: str, candidates: list[EventCandidate]) -> bool:
    return all(response[row.start:row.end] == row.text for row in candidates)
