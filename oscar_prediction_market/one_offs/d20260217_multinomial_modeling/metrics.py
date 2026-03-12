"""CV metrics computation for multinomial modeling analysis.

Computes model performance metrics from cross-validation predictions.

All functions are stateless — they take DataFrames/dicts with standardised
column names and return metric dicts or DataFrames.  No I/O inside.

Key functions:

- :func:`extract_key_metrics` — pull headline numbers from a CV metrics JSON.
- :func:`compute_cv_metrics_from_df` — compute accuracy / Brier / log-loss /
  MRR / winner-prob from a predictions DataFrame.
"""

import numpy as np
import pandas as pd

# ============================================================================
# Metric extraction from CV JSON
# ============================================================================


def extract_key_metrics(metrics: dict) -> dict[str, float]:
    """Extract the headline metrics from a CV metrics dict.

    Works with both ``micro``- and ``macro``-keyed dicts (tries ``micro``
    first, then ``macro``).  Absent keys become ``nan``.

    Args:
        metrics: Dict returned by ``load_cv_metrics()``, with top-level keys
            like ``'macro'`` / ``'micro'`` and ``'prob_sum'``.

    Returns:
        Dict with keys: accuracy, top_3_accuracy, mrr, brier_score,
        log_loss, mean_winner_prob, prob_sum_mean, prob_sum_std.
    """
    agg = metrics.get("micro", metrics.get("macro", {}))
    prob_sum = metrics.get("prob_sum", {})
    return {
        "accuracy": agg.get("accuracy", float("nan")),
        "top_3_accuracy": agg.get("top_3_accuracy", float("nan")),
        "mrr": agg.get("mrr", float("nan")),
        "brier_score": agg.get("brier_score", float("nan")),
        "log_loss": agg.get("log_loss", float("nan")),
        "mean_winner_prob": agg.get("mean_winner_prob", float("nan")),
        "prob_sum_mean": prob_sum.get("mean", float("nan")),
        "prob_sum_std": prob_sum.get("std", float("nan")),
    }


# ============================================================================
# Metric computation from predictions DataFrame
# ============================================================================


def compute_cv_metrics_from_df(df: pd.DataFrame) -> dict[str, float]:
    """Compute key CV metrics from a predictions DataFrame.

    Averages per-ceremony metrics (LOYO macro-average), matching the
    evaluation strategy used by ``build_model``'s CV output.

    Args:
        df: DataFrame with columns: ``ceremony``, ``probability``,
            ``is_actual_winner``, ``rank``.

    Returns:
        Dict with keys: accuracy, top_3_accuracy, mrr, brier_score,
        log_loss, mean_winner_prob, prob_sum_mean, prob_sum_std.
    """
    ceremonies = df["ceremony"].unique()

    accuracies: list[float] = []
    top3_accuracies: list[float] = []
    mrrs: list[float] = []
    brier_scores: list[float] = []
    log_losses: list[float] = []
    winner_probs: list[float] = []
    prob_sums: list[float] = []

    for ceremony in ceremonies:
        year_df = df[df["ceremony"] == ceremony].sort_values("rank")
        probs = year_df["probability"].values
        actual = year_df["is_actual_winner"].astype(int).values

        prob_sums.append(float(probs.sum()))

        winner_idx = int(np.argmax(actual))
        winner_prob = float(probs[winner_idx])
        winner_probs.append(winner_prob)

        top_idx = int(np.argmax(probs))
        accuracies.append(float(actual[top_idx]))

        top3_indices = np.argsort(probs)[-3:]
        top3_accuracies.append(float(any(actual[i] for i in top3_indices)))

        ranks = np.argsort(np.argsort(-probs)) + 1
        winner_rank = int(ranks[winner_idx])
        mrrs.append(1.0 / winner_rank)

        brier = float(np.mean((probs - actual) ** 2))
        brier_scores.append(brier)

        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        ll = float(
            -np.mean(actual * np.log(probs_clipped) + (1 - actual) * np.log(1 - probs_clipped))
        )
        log_losses.append(ll)

    return {
        "accuracy": float(np.mean(accuracies)),
        "top_3_accuracy": float(np.mean(top3_accuracies)),
        "mrr": float(np.mean(mrrs)),
        "brier_score": float(np.mean(brier_scores)),
        "log_loss": float(np.mean(log_losses)),
        "mean_winner_prob": float(np.mean(winner_probs)),
        "prob_sum_mean": float(np.mean(prob_sums)),
        "prob_sum_std": float(np.std(prob_sums)),
    }
