from fineval.polyfiqa_family_gate import choose_family_route, exact_cluster_bootstrap


def test_family_route_requires_cross_company_support():
    rows = [
        {"company": "A", "baseline_score": 0.2, "ranker_score": 0.3},
        {"company": "A", "baseline_score": 0.3, "ranker_score": 0.4},
        {"company": "B", "baseline_score": 0.2, "ranker_score": 0.24},
    ]
    decision = choose_family_route(rows)
    assert decision["route"] == "family_ranker"
    assert decision["companies"] == 2


def test_family_route_rejects_concentrated_gain_with_bad_company():
    rows = [
        {"company": "A", "baseline_score": 0.1, "ranker_score": 0.5},
        {"company": "A", "baseline_score": 0.1, "ranker_score": 0.5},
        {"company": "B", "baseline_score": 0.6, "ranker_score": 0.5},
    ]
    decision = choose_family_route(rows)
    assert decision["mean_gain"] > 0
    assert decision["worst_company_gain"] < -0.02
    assert decision["route"] == "state_medoid"


def test_family_route_rejects_tiny_gain():
    rows = [
        {"company": "A", "baseline_score": 0.2, "ranker_score": 0.205},
        {"company": "B", "baseline_score": 0.3, "ranker_score": 0.305},
        {"company": "B", "baseline_score": 0.4, "ranker_score": 0.405},
    ]
    assert choose_family_route(rows)["route"] == "state_medoid"


def test_exact_cluster_bootstrap_preserves_positive_cluster_support():
    rows = [
        {"company": "A", "delta": 0.1},
        {"company": "A", "delta": 0.1},
        {"company": "B", "delta": 0.0},
    ]
    result = exact_cluster_bootstrap(
        rows, cluster_key="company", delta_key="delta",
    )
    assert result["resamples"] == 4
    assert result["p_nonnegative"] == 1.0
    assert result["p_gain"] == 0.75
    assert result["p05"] >= 0.0
