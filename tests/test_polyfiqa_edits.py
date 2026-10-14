from fineval.polyfiqa_edits import sparse_edits, split_answer


def test_split_answer_preserves_evidence_heading():
    core, evidence = split_answer("Answer: A.\nNews Evidence: B.")
    assert core == "Answer: A."
    assert evidence == "News Evidence: B."


def test_sparse_edits_keep_baseline_and_make_local_hybrids():
    baseline = "Answer: A. Extra.\nNews Evidence: Base."
    alternative = "Answer: B.\nNews Evidence: Other."
    edits = sparse_edits(baseline, {"precision": alternative})
    assert edits["baseline"] == baseline
    assert edits["core:precision"] == "Answer: B.\nNews Evidence: Base."
    assert edits["evidence:precision"] == "Answer: A. Extra.\nNews Evidence: Other."
    assert edits["first_sentence:precision"] == (
        "Answer: B. Extra.\nNews Evidence: Base."
    )
    assert edits["delete_sentence:1"] == "Answer: A.\nNews Evidence: Base."
