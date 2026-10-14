import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_task1_neural_operation_ranker import marked_context


def test_marked_context_marks_exact_global_span_inside_window():
    text = "πριν Εταιρεία μετά"
    marked = marked_context(text, 5, 13)
    assert marked == "πριν  [MASK] Εταιρεία [MASK]  μετά"


def test_marked_context_accounts_for_context_offset():
    context = "Εταιρεία μετά"
    assert marked_context(context, 10, 18, context_start=10).startswith(
        " [MASK] Εταιρεία [MASK] "
    )


def test_marked_context_rejects_out_of_window_span():
    with pytest.raises(ValueError, match="outside"):
        marked_context("abc", 3, 5)
