"""Summarize lag audit results at per-event granularity.

Reads inferred and fixed-time lag audit CSVs and writes reproducible summary
artifacts used in the one-off README.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
    d20260223_timing_leakage_audit.analyze_event_level_lag
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path("storage/d20260223_timing_leakage_audit/2025")
INFERRED_CSV = BASE_DIR / "lag_audit_inferred" / "uniform_lag_audit.csv"
FIXED_CSV = BASE_DIR / "lag_audit_fixed_2100" / "uniform_lag_audit.csv"
OUT_DIR = BASE_DIR / "analysis"


def _load_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    baseline = df[(df["price_source"] == "daily_close_same_date") & (df["lag_hours"] == 0)].copy()
    lagged = df[df["price_source"] == "intraday_lag_hours"].copy()
    return baseline, lagged


def _event_pivot(baseline: pd.DataFrame, lagged: pd.DataFrame) -> pd.DataFrame:
    baseline_event = (
        baseline.groupby(["snapshot_date", "event_labels"], as_index=False)[["total_pnl"]]
        .sum()
        .rename(columns={"total_pnl": "baseline_pnl"})
    )
    lag_event = lagged.groupby(["snapshot_date", "event_labels", "lag_hours"], as_index=False)[
        "total_pnl"
    ].sum()

    pivot = lag_event.pivot_table(
        index=["snapshot_date", "event_labels"],
        columns="lag_hours",
        values="total_pnl",
        aggfunc="first",
    ).reset_index()

    pivot = pivot.rename(
        columns={1: "lag_1h_pnl", 6: "lag_6h_pnl", 12: "lag_12h_pnl", 24: "lag_24h_pnl"}
    )
    pivot = pivot.merge(baseline_event, on=["snapshot_date", "event_labels"], how="left")
    pivot = pivot[
        [
            "snapshot_date",
            "event_labels",
            "baseline_pnl",
            "lag_1h_pnl",
            "lag_6h_pnl",
            "lag_12h_pnl",
            "lag_24h_pnl",
        ]
    ]
    return pivot.sort_values("snapshot_date").reset_index(drop=True)


def _add_shares(table: pd.DataFrame) -> pd.DataFrame:
    total = float(table["baseline_pnl"].sum())
    out = table.copy()
    out["baseline_share_pct"] = (out["baseline_pnl"] / total) * 100.0
    return out


def _to_markdown_currency(table: pd.DataFrame) -> str:
    df = table.copy()
    money_cols = ["baseline_pnl", "lag_1h_pnl", "lag_6h_pnl", "lag_12h_pnl", "lag_24h_pnl"]
    for col in money_cols:
        df[col] = df[col].map(lambda v: f"{v:+,.2f}")
    if "baseline_share_pct" in df.columns:
        df["baseline_share_pct"] = df["baseline_share_pct"].map(lambda v: f"{v:,.1f}%")
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---" for _ in columns]) + "|"
    rows = []
    for row in df.itertuples(index=False):
        vals = [str(v).replace("|", "\\|") for v in row]
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, separator, *rows])


def _save(table: pd.DataFrame, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.md").write_text(_to_markdown_currency(table) + "\n")


def main() -> None:
    inferred_base, inferred_lag = _load_tables(INFERRED_CSV)
    fixed_base, fixed_lag = _load_tables(FIXED_CSV)

    inferred_table = _add_shares(_event_pivot(inferred_base, inferred_lag))
    fixed_table = _add_shares(_event_pivot(fixed_base, fixed_lag))

    _save(inferred_table, "event_level_lag_inferred")
    _save(fixed_table, "event_level_lag_fixed_2100")

    print(f"Saved: {OUT_DIR / 'event_level_lag_inferred.csv'}")
    print(f"Saved: {OUT_DIR / 'event_level_lag_fixed_2100.csv'}")


if __name__ == "__main__":
    main()
