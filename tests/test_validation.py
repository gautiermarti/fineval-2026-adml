from fineval.validation import validate_answer


def test_polyfiqa_structure_and_number_grounding() -> None:
    result = validate_answer(
        "Answer: Revenue was $120M, up 20%.\nNews Evidence: None.",
        "Revenue | $120M | growth 20%",
        "easy",
    )
    assert result.valid


def test_polyfiqa_validator_rejects_unsupported_number() -> None:
    result = validate_answer(
        "Answer: Revenue was $999M.\nFinancial Statement Evidence: Revenue | 120",
        "Revenue | 120",
        "expert",
    )
    assert result.unsupported_numbers == ("$999M",)


def test_polyfiqa_validator_understands_billion_scaling() -> None:
    supported = validate_answer(
        "Answer: Dividends were $3.856B.\nFinancial Statement Evidence: dividends | 3,856",
        "dividends | 3,856",
        "expert",
    )
    wrong_scale = validate_answer(
        "Answer: Dividends were $3.856M.\nFinancial Statement Evidence: dividends | 3,856",
        "dividends | 3,856",
        "expert",
    )
    assert not supported.unsupported_numbers
    assert wrong_scale.unsupported_numbers == ("$3.856M",)


def test_polyfiqa_validator_accepts_compact_display_rounding() -> None:
    result = validate_answer(
        "Answer: Capex was $3.8B.\nFinancial Statement Evidence: additions | 3,767",
        "additions | 3,767",
        "expert",
    )
    assert result.valid


def test_polyfiqa_validator_applies_table_thousands_unit() -> None:
    result = validate_answer(
        "Answer: Capex was $35.2M.\nFinancial Statement Evidence: capex | 35,227",
        "Amounts in thousands. capex | 35,227",
        "expert",
    )
    assert result.valid


def test_polyfiqa_validator_recovers_ocr_currency_table_unit() -> None:
    result = validate_answer(
        "Answer: Capex was $38.8M.\nFinancial Statement Evidence: capex | $ | 38,796",
        "Purchase of property | $ | 38,796 | $ | 21,099",
        "expert",
    )
    assert result.valid


def test_polyfiqa_validator_accepts_one_decimal_billion_rounding_near_one() -> None:
    result = validate_answer(
        "Answer: Capex was $1.0B.\nFinancial Statement Evidence: capex | 1,039",
        "Amounts in millions. capex | 1,039",
        "expert",
    )
    assert result.valid


def test_polyfiqa_validator_tolerates_dotted_source_section_numbers() -> None:
    result = validate_answer(
        "Answer: Cost control.\nFinancial Statement Evidence: None.",
        "Section 3.3.1 discusses cost control.",
        "expert",
    )
    assert result.valid
