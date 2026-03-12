"""Discover all Oscar market tickers from Kalshi API.

Queries the Kalshi events API for all known Oscar series tickers and outputs
a comprehensive inventory of categories × years × nominee tickers.

Writes ``ticker_inventory.json`` in this directory, which is the canonical
data source for the :mod:`market_data` registry.  Commit the updated JSON
after running.

Usage::

    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.trading.market_data.discover_tickers
"""

import json
import time
from pathlib import Path

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"

# All known Oscar series tickers on Kalshi
# Discovered by crawling Kalshi URLs and API queries on 2026-02-20.
OSCAR_SERIES: dict[str, str] = {
    # Major categories (our 9 modeled categories)
    "KXOSCARPIC": "Best Picture",
    "KXOSCARDIR": "Best Director",
    "KXOSCARACTO": "Best Actor (Lead)",
    "KXOSCARACTR": "Best Actress (Lead)",
    "KXOSCARSUPACTO": "Best Supporting Actor",
    "KXOSCARSUPACTR": "Best Supporting Actress",
    "KXOSCARSPLAY": "Best Original Screenplay",
    "KXOSCARASPLAY": "Best Adapted Screenplay",
    "KXOSCARANIMATED": "Best Animated Feature",
    "KXOSCARCINE": "Best Cinematography",
    # Additional categories with Kalshi markets
    "KXOSCARCOSTUME": "Best Costume Design",
    "KXOSCARPROD": "Best Production Design",
    "KXOSCARSOUND": "Best Sound",
    "KXOSCARSONG": "Best Original Song",
    "KXOSCARSCORE": "Best Original Score",
    "KXOSCAREDIT": "Best Film Editing",
    "KXOSCARINTLFILM": "Best International Feature Film",
}


def fetch_events(series_ticker: str) -> list[dict]:
    """Fetch all events for a series ticker."""
    time.sleep(0.5)
    params: dict[str, str | int] = {"series_ticker": series_ticker, "limit": 50}
    resp = requests.get(f"{BASE}/events", params=params)
    if resp.status_code == 429:
        print(f"  Rate limited on {series_ticker}, retrying in 5s...")
        time.sleep(5)
        resp = requests.get(f"{BASE}/events", params=params)
    if resp.status_code != 200:
        print(f"  {series_ticker}: HTTP {resp.status_code}")
        return []
    return resp.json().get("events", [])


def fetch_markets(event_ticker: str) -> list[dict]:
    """Fetch all nominee markets for an event ticker."""
    time.sleep(0.5)
    params: dict[str, str | int] = {"event_ticker": event_ticker, "limit": 200}
    resp = requests.get(f"{BASE}/markets", params=params)
    if resp.status_code == 429:
        time.sleep(5)
        resp = requests.get(f"{BASE}/markets", params=params)
    if resp.status_code != 200:
        return []
    return resp.json().get("markets", [])


def extract_nominee_name(market: dict) -> str:
    """Extract nominee name from a market's custom_strike or title."""
    cs = market.get("custom_strike", {})
    if isinstance(cs, dict):
        nominee = cs.get("Nominee", cs.get("Movie", str(cs)))
        return str(nominee)
    title: str = market.get("title", "")
    return title[:60]


def main() -> None:
    print("=" * 80)
    print("KALSHI OSCAR MARKET INVENTORY")
    print("=" * 80)

    # {series: {event_ticker: [{ticker, nominee, status}, ...]}}
    inventory: dict[str, dict[str, list[dict]]] = {}

    for series, name in sorted(OSCAR_SERIES.items(), key=lambda x: x[1]):
        print(f"\n--- {name} ({series}) ---")
        events = fetch_events(series)

        if not events:
            print("  No events found")
            continue

        series_data: dict[str, list[dict]] = {}

        for event in sorted(events, key=lambda e: e["event_ticker"]):
            et = event["event_ticker"]
            markets = fetch_markets(et)

            nominees = []
            for m in sorted(markets, key=lambda x: x["ticker"]):
                nominee_name = extract_nominee_name(m)
                nominees.append(
                    {
                        "ticker": m["ticker"],
                        "nominee": nominee_name,
                        "status": m["status"],
                    }
                )

            series_data[et] = nominees
            active = sum(1 for n in nominees if n["status"] == "active")
            finalized = sum(1 for n in nominees if n["status"] == "finalized")
            print(f"  {et}: {len(nominees)} markets (active={active}, finalized={finalized})")
            for n in nominees:
                print(f"    {n['ticker']}: {n['nominee']} [{n['status']}]")

        inventory[series] = series_data

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Categories × Years")
    print("=" * 80)

    # Collect all year suffixes
    all_suffixes: set[str] = set()
    for series_data in inventory.values():
        for et in series_data:
            # Extract year suffix: e.g. OSCARPIC-22 -> 22, KXOSCARPIC-25 -> 25
            suffix = et.split("-")[-1]
            if suffix.endswith("B"):
                suffix = suffix[:-1]  # KXOSCARANIMATED-26B -> 26
            all_suffixes.add(suffix)

    year_cols = sorted(all_suffixes)

    # Header
    header = f"{'Category':<30} | " + " | ".join(f"  {y}  " for y in year_cols)
    print(header)
    print("-" * len(header))

    for series in sorted(inventory, key=lambda s: OSCAR_SERIES[s]):
        name = OSCAR_SERIES[series]
        row = f"{name:<30} | "
        for y in year_cols:
            # Find event with this year suffix
            found = False
            for et, nominees in inventory[series].items():
                if et.endswith(f"-{y}") or et.endswith(f"-{y}B"):
                    n_noms = len(nominees)
                    row += f"  {n_noms:2d}  | "
                    found = True
                    break
            if not found:
                row += "   —  | "
        print(row)

    # Save raw inventory as JSON (same directory as this script)
    output_path = Path(__file__).parent / "ticker_inventory.json"
    with open(output_path, "w") as f:
        json.dump(inventory, f, indent=2)
        f.write("\n")
    print(f"\nSaved full inventory to {output_path}")


if __name__ == "__main__":
    main()
