from fineval.task1_span_surgery import span_surgeries


def test_span_surgeries_include_article_suffix_and_type_counterfactuals():
    text = "Η Τράπεζα Πειραιώς Α.Ε. ανακοίνωσε αποτελέσματα."
    start = text.index("Τράπεζα")
    end = text.index(" ανακοίνωσε")
    alternatives = span_surgeries(text, start, end, "ΟΡΓΑΝΙΣΜΟΣ")
    assert all(item.surface == text[item.start:item.end] for item in alternatives)
    assert any("add_leading_article" in item.operations for item in alternatives)
    assert any("drop_legal_suffix" in item.operations for item in alternatives)
    assert any(
        item.start == start and item.end == end and item.entity_type == "ΠΡΟΣΩΠΟ"
        for item in alternatives
    )


def test_span_surgeries_extract_parenthetical_alias():
    text = "Η Alpha Services (AS) ανακοίνωσε νέα σύμβαση."
    start = text.index("Alpha")
    end = text.index(" ανακοίνωσε")
    alternatives = span_surgeries(text, start, end, "ΟΡΓΑΝΙΣΜΟΣ")
    alias = [
        item for item in alternatives
        if "parenthetical_alias_only" in item.operations
    ]
    assert len(alias) == 1
    assert alias[0].surface == "AS"
