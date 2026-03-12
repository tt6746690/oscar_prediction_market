# oscar_prediction_market


# Motivations

This project started after reading far too many of Matt Levine's
[*Money Stuff*](https://www.bloomberg.com/opinion/authors/ARbTQlRLRjE/matthew-s-levine)
columns, where prediction markets come up frequently. The way he writes about
them is contagious. And rightly so — prediction markets have been experiencing
explosive growth. 

They first entered the broader public conversation during the 2024 United States presidential election, and by late 2025 and early 2026 it felt like an inflection point. I wanted to understand them better and maybe try building something around them myself.

The Academy Awards felt like a great test case: relatively liquid, reasonably
structured, and amenable to machine learning. Plus, I like watching films, so it seemed like a fun domain to work in.

Another force in action is projects like this have become far more feasible now that coding agents have gotten so much better. With these tools, building a full pipeline for a relatively small-volume market — especially compared to
equities — suddenly seemed realistic. It's also exactly the kind of niche that institutional investors would likely ignore. The potential gains are simply too small.

About five weeks later, this repo had grown into a reasonable pipeline: Oscar data → ML predictions → market edge → Kelly-sized trades → backtested P&L, along side a [research journal](oscar_prediction_market/one_offs/experiment_index.md) documenting 30+
one-off investigations into things like what model to use, do backtesting, etc.. It's still a bit rough around the edges, but it works well enough that I'd be willing to throw away $1k at. these.

The whole process has been a lot of fun — from experimenting with agentic coding tools, to learning more about prediction markets, to digging into how the Oscars actually work. As a bonus, it even gave me a reason to watch a few films that caught my interest along the way!

## (Partially) What I Learned

The biggest surprise from this project was how predictable the Oscars are. [Precursor awards](oscar_prediction_market/one_offs/d20260220_feature_ablation/) — especially the guilds (DGA, PGA, SAG), along with BAFTA and Critics Choice — dominate almost every model. Features like box office, runtime, or genre [add almost nothing](oscar_prediction_market/one_offs/d20260220_feature_ablation/#all-models-agree-precursor_winners--precursor_noms-is-the-sweet-spot) by comparison.

[Model choice](oscar_prediction_market/one_offs/d20260305_config_selection_sweep/#2-which-model) also matters quite a bit. Different algorithms produced [several-fold differences](oscar_prediction_market/one_offs/d20260207_model_comparison/#gbt-is-the-best-single-model-xgboost-adds-no-value) in backtest profit. In the end, a simple equal-weight ensemble of a few model types proved the most reliable, consistently finishing near the top across [bootstrap resamples](oscar_prediction_market/one_offs/d20260305_config_selection_sweep/#bootstrap-model-ranking) and backtest years.

The problem is also [inherently temporal](oscar_prediction_market/one_offs/d20260211_temporal_model_snapshots/), which was interesting coming from mostly working with static datasets. New precursor signals [arrive throughout awards season](oscar_prediction_market/one_offs/d20260211_temporal_model_snapshots/#model-accuracy-improves-as-precursor-information-arrives), so the model retrains periodically to incorporate the newly available information.

On the trading side, restraint helps. Most small model–market differences are likely noise — the dataset is small and the models aren't that reliable — so the final strategy used only trades when the [edge exceeds ~20%](oscar_prediction_market/one_offs/d20260214_trade_signal_ablation/#parameter-sensitivity) after fees. [Concentrating capital](oscar_prediction_market/one_offs/d20260305_portfolio_kelly/#31-pl-is-extremely-concentrated) in the few categories with the largest edges also [improved returns substantially](oscar_prediction_market/one_offs/d20260305_portfolio_kelly/#32-signal-scorecard-ev-leads-on-combined-pl-maxedge-leads-on-rank).

Speed also matters. In backtesting, [delaying trades by just 6 hours](oscar_prediction_market/one_offs/d20260220_backtest_strategies/#signal-delay-analysis-2025-02-24) after a precursor result — a realistic 6hr lag for someone trading manually — [costs roughly half the P&L](oscar_prediction_market/one_offs/d20260220_backtest_strategies/#headline-results-core-categories-are-robust) compared to instantaneous execution. Markets do correct toward model predictions over time, so the edge is real but it fades.

The biggest limitation is data. Kalshi Oscar markets only began in 2024, leaving just [two seasons](oscar_prediction_market/one_offs/d20260305_config_selection_sweep/#1-how-reliable-is-this-backtest) for backtesting. That's not enough for [sophisticated portfolio optimization](oscar_prediction_market/one_offs/d20260225_buy_hold_backtest/), so simple robustness checks, like [bootstrap stability](oscar_prediction_market/one_offs/d20260305_config_selection_sweep/#what-can-we-trust), turned out to be the most reliable guide.

## 2026 Predictions (Pre-Ceremony)

*Market prices as of 2026-03-11 22:20 ET. Ceremony is March 15.*

The model currently disagrees with the market most on these nominees:

| Category | Nominee | Direction | Model | Market | Net Edge | Allocation |
|----------|---------|-----------|------:|-------:|---------:|-----------:|
| Actor (Leading) | Michael B. Jordan | NO | 21% | 50% | +25.9% | 20% |
| Animated Feature | KPop Demon Hunters | NO | 66% | 93% | +24.7% | 37% |
| Actress (Supporting) | Amy Madigan | YES | 76% | 52% | +21.6% | 43% |

Only 3 of 9 categories pass the 20% edge threshold — the remaining 6 sit in
cash. Full report with all categories, bet sizing, and scenario P&L:
[2026-03-11 taker report](oscar_prediction_market/one_offs/d20260224_live_2026/reports/2026-03-11T22:21_edge_20_taker.md).

## Quick Start

```bash
git clone https://github.com/tt6746690/oscar_prediction_market
cd oscar_prediction_market
uv sync --all-extras                # install dependencies
# Populate .env with API keys before running (see .env for instructions)
# they are required for building datasets and fetching live market data

# full pipeline: build data, train models, generate reports
bash oscar_prediction_market/one_offs/d20260224_live_2026/run.sh
# just generate live report (if data/models already in storage/)
bash oscar_prediction_market/one_offs/d20260224_live_2026/run_live.sh
```

This produces a markdown report like
[this one](oscar_prediction_market/one_offs/d20260224_live_2026/reports/2026-03-11T22:21_edge_20_taker.md) containing model-vs-market probabilities for every nominee, edge calculations, Kelly bet sizing, and scenario P&L for each category.

