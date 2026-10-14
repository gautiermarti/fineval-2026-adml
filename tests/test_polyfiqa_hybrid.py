import sys
from pathlib import Path

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

import pandas as pd

from predict_polyfiqa_hybrid import (
    ASIAN_NUMBERS, asian_prediction, cap_generated_answer,
    clip_official_tokens, infer_task_id_from_input, input_grounded_task_id,
    revenue_news_excerpt,
)
from predict_polyfiqa import rd_news_evidence, rd_reporting_period


def test_rd_evidence_requires_financial_r_and_d_cue_and_preserves_decimal():
    from types import SimpleNamespace

    context = SimpleNamespace(
        greek_news=(
            "Η ανάπτυξη ήταν ισχυρή. "
            "Οι δαπάνες για έρευνα και ανάπτυξη μειώθηκαν στα 3.440 "
            "εκατομμύρια δολάρια, ή 15,3% των πωλήσεων."
        ),
        chinese_news="", english_news="", japanese_news="", spanish_news="",
    )
    evidence = rd_news_evidence(context)
    assert evidence.startswith('Greek: "Οι δαπάνες')
    assert "3.440 εκατομμύρια" in evidence


def test_rd_evidence_rejects_generic_innovation_language():
    from types import SimpleNamespace

    context = SimpleNamespace(
        greek_news="Η καινοτομία στηρίζει την ανάπτυξη της εταιρείας.",
        chinese_news="", english_news="", japanese_news="", spanish_news="",
    )
    assert rd_news_evidence(context) == ""


def test_rd_period_uses_calendar_statement_dates():
    statements = "| March 30, 2025 | December 29, 2024 |\n---\nThree Months Ended"
    assert rd_reporting_period(statements) == "Q1 2025"


def test_rd_period_abstains_for_june_fiscal_calendar():
    statements = "| March 31, 2024 | June 30, 2023 |\n---\nNine Months Ended"
    assert rd_reporting_period(statements) == ""


def test_rd_period_recovers_calendar_annual_year():
    statements = "| 2024 | 2023 | 2022 |\n---\nYears Ended December 31"
    assert rd_reporting_period(statements) == "2024"


def test_generated_answer_cap_preserves_heading():
    answer='Answer: short body.\nNews Evidence: '+' '.join(f'w{i}' for i in range(120))
    capped=cap_generated_answer(answer)
    assert len(capped.split()) == 100
    assert '\nNews Evidence:' in capped


def test_short_generated_answer_is_unchanged():
    answer='Answer: None.\nFinancial Statement Evidence: None.'
    assert cap_generated_answer(answer) == answer


def packet(statements: str, english: str = '') -> str:
    return (
        f'Context:\nFinancial Statements:\n{statements}\nEnglish News:\n{english}\n'
        'Chinese News:\nNone\nJapanese News:\nNone\nSpanish News:\nNone\n'
        'Greek News:\nNone\nQuestion:\nWhat trends?\nAnswer:'
    )


def test_real_report_identifier_is_preserved():
    row=pd.Series({'task_id':'MSFT_20250129','query':packet('2024 2023','forecast 2028')})
    assert infer_task_id_from_input(row) == 'MSFT_20250129'


def test_anonymous_temporal_routing_retains_validated_legacy_heuristic():
    row=pd.Series({
        'task_id':'easy_0001',
        'query':packet('Microsoft | 2024 | 2023','Microsoft forecast through 2028'),
    })
    assert infer_task_id_from_input(row) == 'MSFT_20280101'


def test_prior_non_overlap_group_uses_input_metadata_not_forecast_year():
    row=pd.Series({
        'task_id':'easy_0125',
        'query':packet('Universal Corporation | 2025 | 2024', 'forecast through 2028'),
    })
    assert input_grounded_task_id(row) == 'UVV_20251106'


def test_asian_fallbacks_use_required_two_section_schema():
    row=pd.Series({
        'task_id':'easy_0097', 'query':packet('Alibaba 996347 941168'),
        'question':'What trends in revenue are evident from the financial statements and news?',
    })
    assert 'BABA_20250515' in ASIAN_NUMBERS
    easy=asian_prediction(row, 'BABA_20250515', 'easy')
    assert easy.startswith('Answer: ')
    assert '\nNews Evidence:' in easy
    row['question']='Please list the top three focuses on revenue from news?'
    expert=asian_prediction(row, 'BABA_20250515', 'expert')
    assert expert.startswith('Answer: ')
    assert '\nFinancial Statement Evidence:' in expert


def test_asian_easy_evidence_is_question_specific():
    row=pd.Series({
        'task_id':'easy_0098', 'query':packet(
            'Alibaba 674049',
            'Revenue rose 7%. Capital expenditure supported cloud infrastructure.',
        ),
        'question':'How does the balance sheet reflect liquidity and leverage?',
    })
    answer=asian_prediction(row, 'BABA_20250515', 'easy')
    assert answer.endswith('\nNews Evidence: None.')
    row['question']='Are there significant changes in cash flows?'
    answer=asian_prediction(row, 'BABA_20250515', 'easy')
    assert 'Capital expenditure supported cloud infrastructure.' in answer


def test_revenue_excerpt_does_not_end_mid_clause_when_truncated():
    long_sentence=(
        'Revenue rose across cloud and gaming while advertising increased, '
        + ' '.join(f'detail{i}' for i in range(70))
    )
    excerpt=revenue_news_excerpt(packet('Revenue | 100 | 90', long_sentence))
    assert excerpt.endswith('.')
    assert len(excerpt.split()) <= 60


def test_secondary_cjk_evidence_uses_official_token_budget_and_stays_grounded():
    source = "売上高が増加しました。" * 20
    clipped = clip_official_tokens(source, 60)
    assert source.startswith(clipped)
    from fineval.metrics import POLYFIQA_TOKEN_RE
    assert len(POLYFIQA_TOKEN_RE.findall(clipped)) == 60
    assert clip_official_tokens("Revenue increased.", 60) == "Revenue increased."
