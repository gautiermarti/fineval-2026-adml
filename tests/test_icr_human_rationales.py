from fineval.icr_human_rationales import (
    choice_label,
    compact,
    compact_with_map,
    locate_unique_fragment,
    note_fragments,
)


def test_compact_nfkc_normalizes_width_and_preserves_source_offsets():
    text = "Ａ  B\nＣ"
    normalized, offsets = compact_with_map(text)
    assert normalized == "ABC"
    assert offsets == [0, 3, 5]
    assert compact(text) == "ABC"


def test_locate_unique_fragment_recovers_exact_source_span_across_spaces():
    text = "前段。売 上 高は増加した。後段。"
    start, end = locate_unique_fragment(text, "売上高は増加した")
    assert text[start:end] == "売 上 高は増加した"


def test_locate_unique_fragment_rejects_repeated_and_too_short_fragments():
    assert locate_unique_fragment("増加した。増加した。", "増加した") is None
    assert locate_unique_fragment("増加した。", "増") is None


def test_note_fragments_splits_tilde_and_ellipsis_delimiters():
    annotation = {
        "result": [
            {"type": "textarea", "value": {"text": [" 前半 〜 後半…結論。 "]}},
        ],
    }
    assert note_fragments(annotation) == ["前半", "後半", "結論"]


def test_choice_label_reads_valid_choice_prefix_only():
    assert choice_label({
        "result": [{"type": "choices", "value": {"choices": ["+1 commitment"]}}],
    }) == "+1"
    assert choice_label({
        "result": [{"type": "choices", "value": {"choices": ["positive"]}}],
    }) is None
    assert choice_label({"result": []}) is None
