"""Strict parsers for the organizer's instruction-formatted datasets."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ICRExample:
    question: str
    response: str


@dataclass(frozen=True)
class PolyFiQAContext:
    financial_statements: str
    english_news: str
    chinese_news: str
    japanese_news: str
    spanish_news: str
    greek_news: str
    question: str


_ICR_PATTERN = re.compile(
    r"Financial Question:\s*(?P<question>.*?)\s*"
    r"Company Response:\s*(?P<response>.*?)\s*"
    r"Directly output the chosen label",
    re.DOTALL,
)

_POLY_PATTERN = re.compile(
    r"Context:\s*Financial Statements:\s*(?P<statements>.*?)\s*"
    r"English News:\s*(?P<english>.*?)\s*"
    r"Chinese News:\s*(?P<chinese>.*?)\s*"
    r"Japanese News:\s*(?P<japanese>.*?)\s*"
    r"Spanish News:\s*(?P<spanish>.*?)\s*"
    r"Greek News:\s*(?P<greek>.*?)\s*"
    r"Question:\s*(?P<question>.*?)\s*Answer:\s*$",
    re.DOTALL,
)


def parse_icr_query(query: str) -> ICRExample:
    match = _ICR_PATTERN.search(query)
    if match is None:
        raise ValueError("unrecognized JF-ICR query format")
    question = match.group("question").strip()
    response = match.group("response").strip()
    # One released row has its fields reversed: the nominal response is an
    # interrogative while the nominal question is a long declarative answer.
    # Keep this narrowly linguistic so normal rhetorical questions are intact.
    interrogative = re.compile(r"(?:か|でしょうか)[。．]?$")
    if interrogative.search(response) and not interrogative.search(question):
        question, response = response, question
    return ICRExample(question=question, response=response)


def parse_polyfiqa_query(query: str) -> PolyFiQAContext:
    match = _POLY_PATTERN.search(query)
    if match is None:
        raise ValueError("unrecognized PolyFiQA query format")
    fields = {name: match.group(name).strip() for name in match.groupdict()}
    return PolyFiQAContext(
        financial_statements=fields["statements"],
        english_news=fields["english"],
        chinese_news=fields["chinese"],
        japanese_news=fields["japanese"],
        spanish_news=fields["spanish"],
        greek_news=fields["greek"],
        question=fields["question"],
    )
