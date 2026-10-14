from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.predict_task1_gliner_fold_ensemble import ensemble


def test_fold_ensemble_uses_zero_for_absent_and_majority_for_vote():
    folds = []
    for fold in range(5):
        entities = []
        if fold < 3:
            entities.append({
                "start": 0, "end": 4, "label": "organization",
                "text": "ACME", "score": 0.5 + 0.1 * fold,
            })
        folds.append([{"row_index": 0, "fold": fold, "entities": entities}])
    mean, vote = ensemble(folds)
    assert mean[0]["entities"][0]["score"] == (0.5 + 0.6 + 0.7) / 5
    assert vote[0]["entities"][0]["score"] == (0.5 + 0.6 + 0.7) / 3
    assert vote[0]["entities"][0]["vote_count"] == 3
