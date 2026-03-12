# Live 2026 Oscar Prediction Market Trading

**Storage:** `storage/d20260224_live_2026/`

Pre-ceremony analysis for the 98th Academy Awards (March 15, 2026).
Models are retrained after each precursor award, generating updated
trading signals across 9 categories on Kalshi prediction markets.

## Motivation

Apply the full modeling + trading pipeline from 2024-2025 backtests to live
2026 markets. The methodology is validated in
[d20260225_buy_hold_backtest](../d20260225_buy_hold_backtest/) -- which showed
avg_ensemble dominates cross-year rank stability (ρ=0.862) with the lowest
EV inflation (1.10×) and 100% both-year profitability.

This experiment tracks model predictions vs market prices as each precursor
award resolves, generating buy-hold recommendations with explicit risk bounds.

## Setup

### Data Pipeline

Each precursor event triggers a 4-step pipeline:

1. **Refresh** -- Fetch latest precursor winners (`refresh_data.py`)
2. **Build datasets** -- Feature engineering for each snapshot (`build_datasets.sh`)
3. **Train models** -- 4 model types x 9 categories x N snapshots (`train_models.sh`)
4. **Analysis** -- Buy-hold scenarios + report (`run_buy_hold.py` + `generate_report.py`)

### Trading Config

**Model:** avg_ensemble (4-model average: lr, clogit, gbt, cal_sgbt)
**Allocation:** maxedge_100 (signal-proportional bankroll allocation)

Selected from [d20260305_config_selection_sweep](../d20260305_config_selection_sweep/)
(highest cross-year rank stability ρ=0.862, lowest EV inflation 1.10×) and
[d20260305_portfolio_kelly](../d20260305_portfolio_kelly/) (best avg_rank 2.5,
best worst_rank 4).

| Parameter | Value |
|-----------|-------|
| Model | avg_ensemble |
| Edge Threshold | 0.20 |
| Kelly Fraction | 0.05 |
| Kelly Mode | multi_outcome |
| Fee Type | taker |
| Directions | YES + NO |
| Allocation | maxedge_100 (proportional to max model-market edge per category) |

<details>
<summary>Previous config (pre-Mar 7): 3 clogit configs</summary>

All 3 configs used **clogit** (conditional logit), selected from the CVaR-5%
Pareto frontier in the [cross-year analysis](../d20260225_buy_hold_backtest/).

| Config | Kelly Frac | Edge Thresh | Kelly Mode | Directions | CVaR-5% Bound |
|--------|:---------:|:-----------:|:----------:|:----------:|:-------------:|
| **Conservative** | 0.35 | 0.15 | independent | YES only | L <= 10% |
| **Moderate** | 0.50 | 0.05 | independent | YES+NO | L <= 20% |
| **Aggressive** | 0.15 | 0.15 | multi_outcome | YES+NO | L <= 30%+ |

</details>

### Available Snapshots (all precursors resolved)

| # | Event | Date | Key Changes |
|---|-------|------|-------------|
| 1 | Oscar Nominations | Jan 22 | Baseline feature set |
| 2 | DGA Winner | Feb 7 | +directing signal |
| 3 | Annie Winner | Feb 21 | +animated signal |
| 4 | BAFTA Winners | Feb 22 | +8 category signals |
| 5 | PGA Winner | Feb 28 | +best_picture signal |
| 6 | SAG Winners | Mar 1 | +4 acting signals |
| 7 | ASC Winner | Mar 8 | +cinematography signal (One Battle After Another) |
| 8 | WGA Winner | Mar 8 | +original_screenplay signal (Sinners) |

All 10 precursor orgs resolved. Final models trained. Oscar ceremony: **Mar 15**.

## Findings

### Mar 9 — Final pre-ceremony update (ASC + WGA)

Retrained with ASC + WGA results (both Mar 8). 8 precursor changes detected:
ASC cinematography winner (One Battle After Another), WGA original winner
(Sinners), plus PGA/SAG winners picked up from prior refresh. 18 new datasets
built, 72 new models trained. All 10 precursor orgs now resolved — these are
the final models before the ceremony.

#### Key Model-Market Divergences (Live Orderbook, Mar 9 21:57 ET)

| Category | Nominee | avg_ensemble | Market | Gap |
|----------|---------|:------------:|:------:|:---:|
| actress_supporting | Amy Madigan | 75.6% | 45.5% | **+30.1pp** |
| animated_feature | KPop Demon Hunters | 66.3% | 93.5% | **-27.2pp** |
| actress_supporting | Teyana Taylor | 3.7% | 28.0% | **-24.3pp** |
| animated_feature | Zootopia 2 | 26.9% | 5.5% | **+21.4pp** |
| actor_leading | Timothée Chalamet | 50.8% | 30.5% | **+20.3pp** |
| actor_supporting | Stellan Skarsgård | 37.6% | 17.5% | **+20.1pp** |
| actor_leading | Michael B. Jordan | 21.1% | 59.5% | **-38.4pp** |

**ASC impact on Cinematography:** Model now gives One Battle After Another
84.9% (up from 82.7% pre-ASC) after it won the ASC award. Market agrees at
73.5% — edge narrowed, Cinematography dropped below edge_20 threshold. The
model-market gap on Sinners also narrowed (4.3% model vs 17.5% market, down
from the earlier 13pp gap).

**WGA impact on Original Screenplay:** Sinners won WGA, reinforcing its
already-dominant position. Model: 96.9%, market: 94.5% — edge is tiny (2.4pp),
no trade triggered. All models agree Sinners is >89%.

#### Portfolio Recommendations (Live, $1,000 Bankroll)

| Config | E[P&L] | CVaR-5% | $ Deployed | % Bankroll | E[ROIC] | P(profit) | Active Cats |
|--------|:------:|:-------:|:----------:|:----------:|:-------:|:---------:|:-----------:|
| **edge_10** | +$375 | -$188 | $483 | 48.3% | +77.6% | 83% | 6/9 |
| **edge_15** | +$439 | -$219 | $480 | 48.0% | +91.4% | 82% | 5/9 |
| **edge_20** | +$525 | -$315 | $561 | 56.1% | +93.7% | 94% | 3/9 |

**edge_20 active positions** (3 categories, 4 trades):

| Category | Position | Allocated ($) |
|----------|----------|:-------------:|
| Actor (Leading) | NO Michael B. Jordan | $141 |
| Actress (Supporting) | YES Amy Madigan | $151 |
| Actress (Supporting) | NO Teyana Taylor | $196 |
| Animated Feature | NO KPop Demon Hunters | $72 |

5/9 categories inactive at edge_20 threshold: Actor Supporting, Actress Leading,
Best Picture, Cinematography, Directing, Original Screenplay. Edge_10 adds
Actor Supporting, Animated (Zootopia YES), Best Picture (Sinners NO), and
Cinematography (Sinners NO).

#### Comparison: SAG (Mar 8) → ASC+WGA (Mar 9)

Markets continued converging toward model predictions. Key changes:

| Category | Effect of ASC/WGA |
|----------|-------------------|
| Cinematography | ASC winner confirmed model's top pick → edge_20 threshold no longer triggered (edge dropped from ~25pp to 11pp) |
| Original Screenplay | WGA confirmed Sinners → market already priced in (94.5%), model agrees (96.9%) |
| Actress (Supporting) | Unchanged — Amy Madigan edge remains large (+30pp), strongest conviction trade |
| Animated Feature | KPop Demon Hunters still 27pp above model — persistent disagreement |

#### Model Comparison (Live, edge_20)

| Category | avg_ensemble | cal_sgbt | clogit | gbt | lr |
|----------|:------:|:------:|:------:|:------:|:------:|
| Actor (Leading) | $+51 | $+34 | $+39 | $+51 | $+51 |
| Actor (Supporting) | — | $+23 | $-37 | $+13 | $+54 |
| Actress (Supporting) | $-19 | $-27 | $-21 | $-28 | $-28 |
| Animated Feature | $+255 | — | $+286 | $+85 | $+61 |
| **TOTAL** | **$+287** | **$+30** | **$+225** | **$+195** | **$+137** |

All 6 models profitable. avg_ensemble leads at $+287. clogit close at $+225.

<details>
<summary>Mar 7 — Config regime change</summary>

Switched from 3 clogit configs (conservative/moderate/aggressive) with uniform
$1K/category allocation to a single avg_ensemble config with maxedge_100
signal-proportional allocation. Based on two studies:

1. **[d20260305_config_selection_sweep](../d20260305_config_selection_sweep/)** —
   avg_ensemble highest cross-year rank stability (ρ=0.862), lowest EV inflation (1.10×).

2. **[d20260305_portfolio_kelly](../d20260305_portfolio_kelly/)** —
   maxedge_100 best avg_rank (2.5), captures 2–4× P&L of uniform.

| Aspect | Old (pre-Mar 7) | New (Mar 7+) |
|--------|-----------------|--------------|
| Model | clogit | avg_ensemble |
| Configs | 3 (conservative/moderate/aggressive) | 1 (recommended) |
| Edge threshold | 0.05 / 0.15 / 0.15 | 0.20 |
| Category allocation | Uniform $1K/category | maxedge_100 (signal-proportional) |

</details>

<details>
<summary>Mar 3 — Live orderbook pricing (SAG models, live market prices)</summary>

Live orderbook mid-prices fetched at 2026-03-03 00:59 ET via Kalshi public API.
Model predictions from latest available snapshot (SAG, Mar 1). Spread penalty
uses orderbook half-spread (best bid–ask) rather than trade-history estimates.

Per-config reports: [reports/](reports/) (one per config, dated)

#### Key Model-Market Divergences (Live Orderbook Prices)

Largest disagreements between clogit and live market orderbook mid-prices:

| Category | Nominee | Clogit | Market | Gap |
|----------|---------|:------:|:------:|:---:|
| animated_feature | KPop Demon Hunters | 52.7% | 92.5% | **-39.8pp** |
| actress_supporting | Amy Madigan | 78.6% | 42.0% | **+36.6pp** |
| actress_supporting | Teyana Taylor | 1.4% | 30.0% | **-28.6pp** |
| animated_feature | Zootopia 2 | 30.4% | 5.5% | **+24.9pp** |
| cinematography | One Battle after Another | 89.6% | 66.5% | **+23.1pp** |
| actor_leading | Timothée Chalamet | 31.1% | 54.0% | **-22.9pp** |
| actor_supporting | Sean Penn | 93.8% | 76.0% | **+17.8pp** |

#### Portfolio Recommendations ($1,000 Bankroll)

Monte Carlo simulation (10k samples, winners drawn from clogit probabilities):

| Config | E[P&L] | CVaR-5% | CVaR-10% | $ Deployed | % Bankroll | E[ROIC] | P(profit) | P(loss>10% bankroll) | P(loss>20% bankroll) | Active Cats |
|--------|:------:|:-------:|:--------:|:----------:|:----------:|:-------:|:---------:|:--------------------:|:--------------------:|:-----------:|
| **Conservative** | +$41 | -$33 | -$27 | $56 | 5.6% | +73.5% | 80% | 0% | 0% | 3/9 |
| **Moderate** | +$164 | -$66 | -$42 | $290 | 29.0% | +56.5% | 87% | 1% | 0% | 6/9 |
| **Aggressive** | +$193 | -$94 | -$59 | $237 | 23.7% | +81.2% | 88% | 1% | 0% | 4/9 |

All three configs show positive expected value with strong risk profiles.
Moderate is the recommended default — $290 deployed (29% bankroll) with 87%
probability of profit and E[P&L] of +$164. The aggressive config achieves
the highest E[ROIC] (+81.2%) with 88% P(profit).

#### Live vs SAG Comparison (market movement in ~24h)

Markets moved toward model predictions in several categories, reducing edge:

| Metric | SAG (Mar 2) | Live (Mar 3) | Change |
|--------|:-----------:|:------------:|:------:|
| **Conservative** E[P&L] | +$66 | +$41 | -$25 |
| **Moderate** E[P&L] | +$288 | +$164 | -$124 |
| **Aggressive** E[P&L] | +$306 | +$193 | -$113 |
| Moderate $ Deployed | $430 | $290 | -$140 |
| Moderate Active Cats | 6/9 | 6/9 | 0 |

**Key price movements** (SAG candle close → live orderbook mid-price):

| Nominee | SAG Price | Live Price | Move | Effect on Edge |
|---------|:---------:|:----------:|:----:|:---------------|
| Amy Madigan (Supp. Actress) | 31.0% | 42.0% | +11pp | Edge still large (model: 78.6%) |
| Teyana Taylor (Supp. Actress) | 47.0% | 30.0% | -17pp | Still overpriced (model: 1.4%) |
| Sean Penn (Supp. Actor) | 55.0% | 76.0% | +21pp | Market moved toward model (93.8%) |
| Timothée Chalamet (Lead Actor) | 68.0% | 54.0% | -13.5pp | Edge increased — now NO side |
| KPop Demon Hunters (Animated) | 90.0% | 92.5% | +2.5pp | Slightly wider vs model (52.7%) |

Same 6/9 moderate categories remain active, but with smaller positions due to
orderbook spread penalty (real bid-ask spread vs trade-history estimates). Total
deployed dropped from $430 → $290. E[P&L] halved primarily from spread-adjusted
sizing, not lost edge.

**Animated feature** remains the strongest signal — KPop Demon Hunters still 40pp
above model (92.5% market vs 52.7% clogit), and Zootopia 2 still 25pp below
(5.5% market vs 30.4% clogit).

#### Model Agreement Analysis (SAG snapshot)

Cross-model comparison (clogit, lr, gbt, cal_sgbt) reveals where predictions
diverge most. High disagreement (>15pp spread) flags positions where clogit-only
trading is risky — the signal may be model-specific rather than fundamental.

**Category risk levels:**

| Category | Max Spread | Risk | Notes |
|----------|:----------:|:----:|-------|
| Actor (Supporting) | 71.5pp | HIGH | lr/clogit/cal_sgbt diverge on Penn & Skarsgård |
| Actor (Leading) | 50.8pp | HIGH | clogit vs gbt/cal_sgbt on Chalamet (31% vs 81%) |
| Actress (Supporting) | 41.5pp | HIGH | lr low on Madigan vs cal_sgbt high (55% vs 96%) |
| Animated Feature | 38.7pp | HIGH | clogit low on KPop vs cal_sgbt high (53% vs 92%) |
| Cinematography | 24.2pp | HIGH | gbt low on One Battle after Another |
| Directing | 17.1pp | MED | |
| Original Screenplay | 15.2pp | MED | |
| Best Picture | 12.3pp | MED | |
| Actress (Leading) | 6.6pp | LOW | All models agree on Buckley (~95%+) |

**Top disagreement nominees with positions (moderate config, SAG prices):**

| Nominee | Position | Capital | clogit | lr | gbt | cal_sgbt | Spread |
|---------|:--------:|:-------:|:------:|:--:|:---:|:--------:|:------:|
| Sean Penn | YES | $35 | 93.8% | 59.8% | 50.2% | 22.3% | 71.5pp |
| Stellan Skarsgård | NO | $25 | 6.2% | 18.1% | 42.9% | 76.9% | 70.7pp |
| Timothée Chalamet | NO | $21 | 31.1% | 31.3% | 81.9% | 81.0% | 50.8pp |
| Amy Madigan | YES | $35 | 78.6% | 54.7% | 87.8% | 96.2% | 41.5pp |
| KPop Demon Hunters | NO | $24 | 52.7% | 62.3% | 61.6% | 91.5% | 38.7pp |

8 positions have HIGH model disagreement (>20pp spread). Two key concerns:
- **Skarsgård NO** ($25): clogit says 6.2% but cal_sgbt says 76.9% — diametrically opposite. If cal_sgbt is right, this NO bet loses.
- **Chalamet NO** ($21): clogit/lr agree (~31%) but gbt/cal_sgbt say ~81%. A 50pp split on the favorite.

Positions where all models agree (LOW risk): Actress (Leading) — Buckley is
unanimous at 93-99%. Cinematography is mostly aligned (clogit/cal_sgbt both ~90%).

> Note: Model agreement analysis uses SAG-era market prices (Mar 2). Live
> orderbook prices are similar but not identical. The disagreement patterns
> reflect structural model differences, not stale data.

</details>

### Previous Updates

<details>
<summary>Mar 2 -- SAG update (6 snapshots, market prices as of 2026-03-02 04:00 ET)</summary>

Retrained with PGA + SAG results. 6 precursor changes detected (PGA bp/animated,
SAG ensemble/lead_actor/lead_actress/supporting_actress). 18 new datasets built,
72 new models trained.

**Portfolio Recommendations (SAG entry, $1,000 bankroll):**

| Config | E[P&L] | CVaR-5% | $ Deployed | % Bankroll | E[ROIC] | P(profit) | Active Cats |
|--------|:------:|:-------:|:----------:|:----------:|:-------:|:---------:|:-----------:|
| **Conservative** | +$66 | -$35 | $86 | 8.6% | +76.7% | 85% | 4/9 |
| **Moderate** | +$288 | -$65 | $430 | 43.0% | +67.0% | 91% | 6/9 |
| **Aggressive** | +$306 | -$92 | $296 | 29.6% | +103.4% | 93% | 5/9 |

**Model agreement analysis:** 43% of moderate-config capital in HIGH disagreement
positions (clogit vs cal_sgbt spread >40pp on Sean Penn, Skarsgård, Chalamet).
See model agreement section above for full cross-model comparison.
</details>

<details>
<summary>Feb 28 -- Buy-hold analysis (4 snapshots, old A/B/C configs)</summary>

Initial centers-only run with old config structure (avg_ensemble, clogit,
clogit_cal_sgbt_ensemble). Key findings:
- actress_supporting had largest model-market divergence (Mosaku 60-90% model vs 2% market)
- 5-8 categories active depending on config option
- 1,125 scenario_pnl rows across 4 entry points
</details>

<details>
<summary>Feb 27 -- Revamp to buy-and-hold methodology</summary>

- Deleted continuous-rebalancing pipeline (run_predictions.py)
- Created generate_configs.py and run_buy_hold.py
- Extended year_config.py for 2026 support
</details>

<details>
<summary>Feb 24 -- Initial run (4 snapshots)</summary>

- Refreshed precursors (8 BAFTA changes, Annie manually patched)
- Built 36 datasets, trained 144 models (0 failures)
</details>

---

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"

# Full pipeline (refresh -> build -> train -> analyze -> report):
bash oscar_prediction_market/one_offs/d20260224_live_2026/run.sh \
    2>&1 | tee storage/d20260224_live_2026/run.log

# Buy-hold analysis only (~2 min with fresh market fetch):
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.run_buy_hold

# Live orderbook pricing (fetches current orderbook mid-prices):
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.run_buy_hold --live

# Generate report ($1000 bankroll):
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.generate_report --bankroll 1000

# Generate report for latest live entry:
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.generate_report --live --bankroll 1000

# Full live pipeline (fetch prices + generate reports):
bash oscar_prediction_market/one_offs/d20260224_live_2026/run_live.sh

# Model agreement analysis:
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.analyze_model_agreement --bankroll 1000

# Orderbook & liquidity analysis:
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.analyze_orderbook --bankroll 1000

# Specific categories:
uv run python -m oscar_prediction_market.one_offs.\
d20260224_live_2026.run_buy_hold --categories best_picture directing
```

### After a New Precursor Event

1. Update `_TODAY` in `__init__.py` to today's date
2. Add new snapshot key to `build_datasets.sh` and `train_models.sh`
3. Run:
   ```bash
   # Refresh precursor data
   uv run python -m oscar_prediction_market.one_offs.\
   d20260224_live_2026.refresh_data

   # Build new datasets
   bash oscar_prediction_market/one_offs/d20260224_live_2026/build_datasets.sh

   # Train new models
   bash oscar_prediction_market/one_offs/d20260224_live_2026/train_models.sh

   # Clear market data cache (to get latest prices)
   rm storage/d20260224_live_2026/market_data/candles/*_hourly_candles.parquet
   rm -rf storage/cache/kalshi/

   # Re-run analysis + report
   uv run python -m oscar_prediction_market.one_offs.\
   d20260224_live_2026.run_buy_hold
   uv run python -m oscar_prediction_market.one_offs.\
   d20260224_live_2026.generate_report --bankroll 1000
   uv run python -m oscar_prediction_market.one_offs.\
   d20260224_live_2026.analyze_model_agreement --bankroll 1000
   ```

## Output Structure

```
one_offs/d20260224_live_2026/
+-- reports/                            # Per-config markdown reports (checked into git)
|   +-- YYYY-MM-DDTHH:MM_<config>.md   # Live-timestamped reports
+-- run_live.sh                         # Live pipeline orchestration

storage/d20260224_live_2026/
+-- shared/
|   +-- precursor_awards.json       # Refreshed precursor data
|   +-- film_metadata.json
|   +-- refresh_diff_summary.json   # Audit trail
+-- datasets/{category}/{snapshot_date}/
+-- models/{category}/{model}/{snapshot}/
+-- market_data/
|   +-- candles/                    # Daily + hourly (parquet, cached)
|   +-- trades/
+-- results/
|   +-- scenario_pnl.csv           # Per (cat, model, entry, config, winner)
|   +-- scenario_pnl_agg.csv       # Min/mean/max aggregated across winners
|   +-- model_vs_market.csv         # Probability divergences
|   +-- position_summary.csv        # Held positions per entry point
|   +-- report_summary.csv          # Machine-readable report summary
|   +-- model_agreement_analysis.md # Cross-model disagreement analysis
|   +-- orderbook_analysis.md       # Orderbook depth & liquidity
+-- run.log
```
