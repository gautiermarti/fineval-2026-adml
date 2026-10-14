"""Leakage-controlled routing utilities for family-specific PolyFiQA compilers."""

from __future__ import annotations

from collections import defaultdict
from itertools import product


def choose_family_route(
    rows: list[dict],
    *,
    minimum_rows: int = 3,
    minimum_companies: int = 2,
    minimum_mean_gain: float = 0.01,
    maximum_company_loss: float = -0.02,
) -> dict:
    """Choose a ranker only when inner-company evidence is consistently positive.

    ``rows`` must contain ``company``, ``baseline_score``, and ``ranker_score``.
    The returned decision is deterministic and deliberately conservative: the
    ranker needs enough rows and companies, a material row-weighted gain, and
    no company-level loss below the declared tolerance.
    """
    company_deltas: dict[str, list[float]] = defaultdict(list)
    deltas = []
    for row in rows:
        delta = float(row["ranker_score"]) - float(row["baseline_score"])
        deltas.append(delta)
        company_deltas[str(row["company"])].append(delta)

    means = {
        company: sum(values) / len(values)
        for company, values in sorted(company_deltas.items())
    }
    mean_gain = sum(deltas) / len(deltas) if deltas else 0.0
    worst_company_gain = min(means.values()) if means else 0.0
    eligible = (
        len(rows) >= minimum_rows
        and len(means) >= minimum_companies
        and mean_gain >= minimum_mean_gain
        and worst_company_gain >= maximum_company_loss
    )
    return {
        "route": "family_ranker" if eligible else "state_medoid",
        "rows": len(rows),
        "companies": len(means),
        "mean_gain": mean_gain,
        "worst_company_gain": worst_company_gain,
        "company_mean_gains": means,
        "thresholds": {
            "minimum_rows": minimum_rows,
            "minimum_companies": minimum_companies,
            "minimum_mean_gain": minimum_mean_gain,
            "maximum_company_loss": maximum_company_loss,
        },
    }


def exact_cluster_bootstrap(
    rows: list[dict], *, cluster_key: str, delta_key: str,
) -> dict:
    """Enumerate the ordinary cluster bootstrap when cluster count is small."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(float(row[delta_key]))
    clusters = sorted(grouped)
    if not clusters:
        return {
            "resamples": 0, "p_gain": 0.0, "p_nonnegative": 0.0,
            "p05": 0.0, "median": 0.0, "p95": 0.0,
        }
    if len(clusters) > 8:
        raise ValueError("exact enumeration is reserved for at most eight clusters")
    estimates = []
    for sample in product(clusters, repeat=len(clusters)):
        values = [value for cluster in sample for value in grouped[cluster]]
        estimates.append(sum(values) / len(values))
    estimates.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(estimates) - 1)
        lower = int(position)
        upper = min(lower + 1, len(estimates) - 1)
        fraction = position - lower
        return estimates[lower] * (1 - fraction) + estimates[upper] * fraction

    return {
        "resamples": len(estimates),
        "p_gain": sum(value > 0 for value in estimates) / len(estimates),
        "p_nonnegative": sum(value >= 0 for value in estimates) / len(estimates),
        "p05": quantile(0.05),
        "median": quantile(0.5),
        "p95": quantile(0.95),
    }
