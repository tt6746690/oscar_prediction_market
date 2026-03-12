"""Price utilities for old one-off experiments.

Moved from ``trading.price_utils`` — these functions are only used by
archived one-off experiments (d20260214 trade signal ablation/backtest,
d20260219 backtest regression).

They operate on DataFrames with columns ``(date, nominee, close)`` and
use dollar price units throughout.
"""

from datetime import date
from typing import Any

import pandas as pd


def get_market_prices_on_date(
    daily_prices: pd.DataFrame,
    target_date: date,
) -> dict[str, float]:
    """Get market close prices for all outcomes on a specific date.

    Falls back to the nearest earlier date if the exact date has no data.
    This handles weekends and holidays when markets may not trade.

    Args:
        daily_prices: DataFrame with columns: date, nominee, close.
        target_date: Date to look up prices for.

    Returns:
        {outcome_name: price} in dollars [0.0, 1.0], empty if no data available.
    """
    df = daily_prices[daily_prices["date"] <= target_date].copy()
    if df.empty:
        return {}

    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    prices: dict[str, float] = {}
    for _, row in latest.iterrows():
        prices[row["nominee"]] = float(row["close"])

    return prices


def build_prices_by_date(
    daily_prices: pd.DataFrame,
    snapshot_dates: list[str],
) -> dict[str, dict[str, float]]:
    """Build a prices lookup dict from daily prices and snapshot dates.

    For each snapshot date, finds the market close prices (falling back to
    the nearest earlier date if needed).
    """
    prices_by_date: dict[str, dict[str, float]] = {}
    for snap_date in snapshot_dates:
        target_date = date.fromisoformat(snap_date)
        prices = get_market_prices_on_date(daily_prices, target_date)
        if prices:
            prices_by_date[snap_date] = prices
    return prices_by_date


def serialize_daily_prices(daily_prices: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialize daily prices DataFrame for JSON storage (e.g., for parallel workers).

    Converts date column to string for JSON compatibility.
    """
    df = daily_prices.copy()
    df["date"] = df["date"].astype(str)
    return df.to_dict(orient="records")  # type: ignore[return-value]


def deserialize_daily_prices(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Deserialize daily prices from JSON records back to DataFrame.

    Restores date column from string.
    """
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df
