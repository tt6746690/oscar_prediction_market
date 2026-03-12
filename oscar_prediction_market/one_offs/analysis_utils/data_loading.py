"""Analysis-time data loading.

Loads model predictions, CV metrics, market prices, feature importances, and
backtest results from the standard ``storage/`` directory layout produced by
``build_model.py`` and temporal snapshot experiment pipelines.

Designed to be category-agnostic: functions take explicit paths and parameters
rather than hard-coding Best Picture assumptions.

Key functions:

- **Prediction / metrics loading**: :func:`load_test_predictions`,
  :func:`load_cv_predictions`, :func:`load_cv_metrics`,
  :func:`load_all_test_predictions`.
- **Market data**: :func:`load_market_prices`, :func:`get_market_prob`,
  :func:`build_model_market_df`.
- **Backtest results**: :func:`load_backtest_results`.

Usage::

    from oscar_prediction_market.one_offs.analysis_utils.data_loading import (
        load_test_predictions,
        load_market_prices,
        build_model_market_df,
    )
"""

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from oscar_prediction_market.data.schema import OscarCategory
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.oscar_market import OscarMarket

logger = logging.getLogger(__name__)


# ============================================================================
# Prediction loading
# ============================================================================


def load_test_predictions(
    exp_dir: Path,
    model_type: str,
    snapshot_date: str,
) -> pd.DataFrame | None:
    """Load test predictions for a model x snapshot combination.

    Looks for predictions in order of preference:
    1. 5_final_predict/predictions_test.csv (feature selection pipeline)
    2. 2_final_predict/predictions_test.csv (simple pipeline)
    3. 2_predict/predictions_test.csv (legacy naming)

    Returns:
        DataFrame with columns: title, year, probability, rank, is_actual_winner, etc.
        None if no predictions found.
    """
    run_name = f"{model_type}_{snapshot_date}"
    base = exp_dir / "models" / model_type / snapshot_date / run_name

    for step_dir in ["5_final_predict", "2_final_predict", "2_predict"]:
        pred_path = base / step_dir / "predictions_test.csv"
        if pred_path.exists():
            return pd.read_csv(pred_path)
    return None


def load_cv_predictions(
    exp_dir: Path,
    model_type: str,
    snapshot_date: str,
) -> pd.DataFrame | None:
    """Load cross-validation predictions for calibration analysis.

    Looks for predictions in order of preference:
    1. 4_selected_cv/predictions.csv (feature selection pipeline)
    2. 1_cv/predictions.csv (simple pipeline)

    Returns:
        DataFrame with columns: ceremony, title, probability, is_actual_winner, rank, etc.
        None if no predictions found.
    """
    run_name = f"{model_type}_{snapshot_date}"
    base = exp_dir / "models" / model_type / snapshot_date / run_name

    for cv_dir in ["4_selected_cv", "1_cv"]:
        pred_path = base / cv_dir / "predictions.csv"
        if pred_path.exists():
            return pd.read_csv(pred_path)
    return None


def load_cv_metrics(
    exp_dir: Path,
    model_type: str,
    snapshot_date: str,
) -> dict | None:
    """Load CV metrics JSON for a model x snapshot combination.

    Looks for metrics in order of preference:
    1. 4_selected_cv/metrics.json (feature selection pipeline)
    2. 1_cv/metrics.json (simple pipeline)

    Returns:
        Metrics dict with keys: macro, prob_sum, num_years, etc.
        None if no metrics found.
    """
    run_name = f"{model_type}_{snapshot_date}"
    base = exp_dir / "models" / model_type / snapshot_date / run_name

    for cv_dir in ["4_selected_cv", "1_cv"]:
        metrics_path = base / cv_dir / "metrics.json"
        if metrics_path.exists():
            return json.loads(metrics_path.read_text())
    return None


def load_all_test_predictions(
    exp_dir: Path,
    model_types: list[str],
) -> pd.DataFrame:
    """Load test predictions for all model types x snapshot dates.

    Discovers available snapshot dates from the filesystem.

    Returns:
        DataFrame with columns: title, year, probability, rank, is_actual_winner,
            model_type, snapshot_date
    """
    models_dir = exp_dir / "models"
    rows: list[pd.DataFrame] = []

    for model_type in model_types:
        mt_dir = models_dir / model_type
        if not mt_dir.exists():
            continue
        for date_dir in sorted(mt_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            snap_date = date_dir.name
            preds = load_test_predictions(exp_dir, model_type, snap_date)
            if preds is not None:
                preds["model_type"] = model_type
                preds["snapshot_date"] = snap_date
                rows.append(preds)

    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def get_available_snapshots(exp_dir: Path) -> list[str]:
    """Get sorted snapshot dates that have models built for any model type."""
    models_dir = exp_dir / "models"
    if not models_dir.exists():
        return []

    all_dates: set[str] = set()
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            for date_dir in model_dir.iterdir():
                if date_dir.is_dir():
                    all_dates.add(date_dir.name)

    return sorted(all_dates)


# ============================================================================
# Market price loading
# ============================================================================


def load_market_prices(
    start_date: date = date(2025, 11, 25),
    end_date: date = date(2026, 2, 14),
) -> pd.DataFrame:
    """Load Kalshi Best Picture market prices in dollar-probability units.

    Returns:
        DataFrame with columns: nominee, date, yes_price, probability
        Empty DataFrame if no data available.
    """
    bp_data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    mkt = OscarMarket(event_ticker=bp_data.event_ticker, nominee_tickers=bp_data.nominee_tickers)
    candles = mkt.get_daily_prices(start_date=start_date, end_date=end_date)
    if not candles:
        logger.warning("No market price data from Kalshi API")
        return pd.DataFrame()
    df = pd.DataFrame(
        [
            {
                "nominee": c.nominee,
                "date": c.date,
                "yes_price": c.yes_price,
                "close": c.close,
            }
            for c in candles
        ]
    )
    df["probability"] = df["yes_price"]
    return df


def get_market_prob(
    market_df: pd.DataFrame,
    snapshot_date: str,
    title: str,
) -> float | None:
    """Get market probability for a nominee at a given date.

    Falls back to the most recent prior date if exact date not available.
    Returns None if nominee not in market or no data prior to date.
    """
    bp_nominee_tickers = OSCAR_MARKETS.get_nominee_tickers(OscarCategory.BEST_PICTURE, 2026)
    if title not in bp_nominee_tickers:
        return None
    target = date.fromisoformat(snapshot_date)
    nom_df = market_df[market_df["nominee"] == title]
    if nom_df.empty:
        return None
    exact = nom_df[nom_df["date"] == target]
    if not exact.empty:
        return float(exact.iloc[0]["probability"])
    prior = nom_df[nom_df["date"] <= target]
    if not prior.empty:
        return float(prior.iloc[-1]["probability"])
    return None


# ============================================================================
# Joining model predictions with market prices
# ============================================================================


def build_model_market_df(
    preds_df: pd.DataFrame,
    market_df: pd.DataFrame,
    test_year: int = 2026,
) -> pd.DataFrame:
    """Join model predictions with market prices at each snapshot.

    Args:
        preds_df: All predictions (from load_all_test_predictions)
        market_df: Market prices (from load_market_prices)
        test_year: Year to filter to (default 2026)

    Returns:
        DataFrame with columns:
            snapshot_date, model_type, title, model_prob, market_prob, divergence, rank
    """
    test = preds_df[preds_df["year"] == test_year].copy()
    rows: list[dict] = []
    for _, row in test.iterrows():
        mkt_p = get_market_prob(market_df, row["snapshot_date"], row["title"])
        rows.append(
            {
                "snapshot_date": row["snapshot_date"],
                "model_type": row["model_type"],
                "title": row["title"],
                "model_prob": row["probability"],
                "market_prob": mkt_p,
                "divergence": row["probability"] - mkt_p if mkt_p is not None else None,
                "rank": row["rank"],
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# Feature importance loading
# ============================================================================


def load_feature_importances(
    exp_dir: Path,
    model_types: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Load feature importance: {model_type: {date_str: {feature_name: importance}}}.

    Reads feature_importance.csv from the final prediction step of each model.
    Works for both linear models (abs_coefficient) and tree models (importance).
    """
    result: dict[str, dict[str, dict[str, float]]] = {}
    models_dir = exp_dir / "models"

    for model_type in model_types:
        mt_dir = models_dir / model_type
        if not mt_dir.exists():
            continue
        result[model_type] = {}
        for date_dir in sorted(mt_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            snap_date = date_dir.name
            run_name = f"{model_type}_{snap_date}"
            fi_path = date_dir / run_name / "5_final_predict" / "feature_importance.csv"
            if not fi_path.exists():
                candidates = list(date_dir.glob("*/5_final_predict/feature_importance.csv"))
                fi_path_candidate = candidates[0] if candidates else None
            else:
                fi_path_candidate = fi_path
            if fi_path_candidate is None or not fi_path_candidate.exists():
                continue
            fi_df = pd.read_csv(fi_path_candidate)
            imp_col = "abs_coefficient" if "abs_coefficient" in fi_df.columns else "importance"
            result[model_type][snap_date] = {
                row["feature"]: float(row[imp_col]) for _, row in fi_df.iterrows()
            }
    return result


# ============================================================================
# Backtest results loading
# ============================================================================


def load_backtest_results(exp_dir: Path) -> dict | None:
    """Load backtest_results.json if it exists.

    Returns:
        Dict with keys: config, backtests, settlements, etc.
        None if no backtest results found.
    """
    path = exp_dir / "backtest" / "backtest_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None
