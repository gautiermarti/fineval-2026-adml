from fineval.icr_events import candidates_are_exact, event_candidates


def test_event_candidates_preserve_exact_offsets_and_rank_refusal():
    question = "来期の増配を予定していますか。"
    response = (
        "前期は配当を増やしました。現時点では来期の増配を実施する予定はありません。"
        "市場環境を見ながら検討します。"
    )
    candidates = event_candidates(question, response)
    assert candidates_are_exact(response, candidates)
    assert "予定はありません" in candidates[0].text
    assert candidates[0].refusal_cues >= 1


def test_long_response_adds_exact_discourse_subclauses():
    question = "投資を続けますか。"
    response = (
        "当社はこれまで設備投資を進めてきましたが、"
        "市場環境が不透明であるため、今後の追加投資は状況を見ながら検討していく方針であり、"
        "具体的な金額については現時点で回答を控えます。"
    )
    candidates = event_candidates(question, response)
    assert candidates_are_exact(response, candidates)
    assert len(candidates) >= 2
    assert any(candidate.conditional_cues for candidate in candidates)
