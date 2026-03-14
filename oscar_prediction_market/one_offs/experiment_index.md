# Experiment Index: Oscar Prediction Market

Chronological research journal for the Oscar prediction and trading pipeline —
from first baselines through feature engineering, model selection, multi-category
expansion, backtesting, and live trading. Each entry links to a self-contained
one-off directory with a detailed README, reproduction scripts, and results.

---

**[d20260313_reddit_predictions_post](d20260313_reddit_predictions_post/)** (2026-03-13)
- Reddit-style Oscar predictions post backed by model data for all 9 categories.
- Conversational write-up with links to precursor breakdowns, historical analogs, and model agreement analysis.
- Biggest contrarian calls: Amy Madigan for Supporting Actress (model 88% vs market 31%), Chalamet over MBJ for Best Actor (model 52% vs market 29%).

**[d20260310_best_actor_diagnostics](d20260310_best_actor_diagnostics/)** (2026-03-10)
- Deep diagnostic on Best Actor: model says Chalamet (51%) > MBJ (21%); market says MBJ (61%) > Chalamet (29%). All 5 model sub-components favor Chalamet.
- Historical base rate: 2 precursor wins → 33% Oscar win rate vs 1 win → 10%. Model's 2.4× ratio is consistent with base rates.
- But 0/3 exact analogs with Chalamet's specific winner profile (GG-Musical + CC) ever won. SAG winner (MBJ) is historically the single strongest precursor.

**[d20260305_portfolio_kelly](d20260305_portfolio_kelly/)** (2026-03-05, updated 2026-03-09)
- 39 bankroll allocation strategies across 6 signal families. Reallocation captures 2–4× the P&L of uniform $1k/category.
- **maxedge_100 recommended** (avg_rank 3.5, top-6 in both years). ev_100 has highest combined P&L ($115K) but ranks 28th in 2024.
- With N_eff ≈ 1.1, fine strategy distinctions are near the noise floor. All signal families at 100% aggressiveness substantially beat uniform.

**[d20260305_config_selection_sweep](d20260305_config_selection_sweep/)** (2026-03-05)
- Targeted 27-config × 6-model sweep (taker fees, multi_outcome Kelly, all directions fixed). Finer edge threshold grid (0.02–0.25).
- Clear 3-tier model ranking: Tier 1 (avg_ensemble + cal_sgbt, >98% bootstrap top-3), Tier 2 (clogit_cal_sgbt_ensemble), Tier 3 (clogit, LR, GBT).
- Model rankings are unstable across years (Kendall τ = −0.067). Two years is enough to identify the right tier but not the single optimal config.

**[d20260301_renormalization_analysis](d20260301_renormalization_analysis/)** (2026-03-01)
- Renormalization is a no-op for production models (clogit, cal_sgbt, ensembles) — they already sum to 1.0 by construction.
- Only affects LR (mean prob_sum 0.87) and GBT (mean 0.87). Renorm helps GBT (+62% P&L) but hurts LR (−16%).
- No code change needed since production models are already normalized.

**[d20260225_buy_hold_backtest](d20260225_buy_hold_backtest/)** (2026-02-25, updated 2026-02-28)
- Buy-and-hold across 2 years, 588 configs, 6 models. 89.3% of configs profitable in both years ($30k best combined P&L).
- clogit and lr achieve 100% cross-year profitability; cal_sgbt at 66.7%. clogit highest Spearman ρ (0.893).
- Directing dominates P&L. 2024 harder (67–100% prof by model); 2025 all models 100% profitable.

**[d20260224_live_2026](d20260224_live_2026/)** (2026-02-24, updated 2026-02-28)
- Pre-ceremony analysis for 98th Oscars (9 categories, 4 snapshots through BAFTA). Uses avg_ensemble with maxedge_100 allocation.
- Biggest edge: actress_supporting — Wunmi Mosaku 60–90% model vs 2% market; Teyana Taylor 0.5–16% model vs 71% market.
- Option B (clogit) has broadest coverage (8/9 categories, 25 positions, $5,737 capital). Option C most selective (4/9 categories, 15 positions).

**[d20260223_timing_leakage_audit](d20260223_timing_leakage_audit/)** (2026-02-23)
- Timing-leakage audit for multi-category backtests with lag grid (1h/6h/12h/24h) under inferred and fixed (21:00 ET) event-time assumptions.
- Baseline snapshot P&L is highly concentrated on Feb 7–8 (Critics Choice + DGA/PGA window); lagging execution materially reduces headline edge.
- Recommends framework-level timestamped execution policy (event time + lag) to replace date-only same-day close joins.

**[d20260220_backtest_strategies](d20260220_backtest_strategies/)** (2026-02-22)
- Multi-category backtest against 2025 Kalshi markets. Best Picture: clogit +$414.72 (+41.5%), cal_sgbt +$382.22, gbt +$357.33. LR $0.
- Calendar bug fix (`f2cf592`) was critical — reversed Best Picture from 0% profitable to 31–46% profitable configs.
- Precursor features (DGA/PGA) drive a massive model-vs-market edge window (Feb 8–16).

**[d20260220_feature_ablation](d20260220_feature_ablation/)** (2026-02-20, updated 2026-02-21)
- Multi-category feature ablation: `lr_full` + feature selection beats curated feature groups across 9 categories × 2 models (clogit, cal_sgbt).
- **t=0.90 is the recommended universal feature selection threshold** — best cross-model average (Brier 0.0406, Acc 88.2%, −28% vs no feature selection).
- Clogit wins 8/9 categories; cal_sgbt wins only animated_feature. Per-category feature configs now canonical in `modeling/configs/features/`.

**[d20260219_backtest_regression](d20260219_backtest_regression/)** (2026-02-19)
- Regression test for refactored backtest code. 12/24 configs passed; 12 failed due to intentional behavioral changes.
- Fixed directional spread handling (sells now use `close − spread` instead of `close + spread`) and no-prediction sell price (was $0, now `close − spread`).
- All failures are correctness improvements, not regressions.

**[d20260218_build_all_datasets](d20260218_build_all_datasets/)** (2026-02-18)
- Built raw datasets for all 9 Oscar categories (ceremonies 72–98). All pass completeness checks.
- Screenwriter TMDb enrichment is only ~4% (structural TMDb gap for writers). All other person categories ≥98%.
- Precursor coverage is 100% for 7 categories; animated_feature has BAFTA (84%) and PGA (87%) gaps due to late category introduction.

**[d20260217_multinomial_modeling](d20260217_multinomial_modeling/)** (2026-02-18)
- Conditional Logit matches Binary LR accuracy (65.8%) with better calibration (Brier 0.0674 vs 0.0688) and perfect prob-sum normalization.
- Softmax GBT fails badly (50.4% accuracy) — multi-class XGBoost overfits on ~26 ceremony training set.
- Post-hoc normalization of binary models provides only ~2% Brier improvement — structural constraint (clogit) is better.

**[d20260217_multi_category_expansion](d20260217_multi_category_expansion/)** (2026-02-17, updated 2026-02-20)
- Planning document (PLAN.md) for expanding Best Picture pipeline to all 9 Kalshi Oscar categories.
- Key decisions: nested precursor schema, single `feature_engineering.py`, per-category feature configs, 2000–2026 training range.
- Phases 0 (schema refactor) and 1 (all datasets built) complete. Feeds into d20260218 and d20260220 experiments.

**[d20260214_trade_signal_ablation](d20260214_trade_signal_ablation/)** (2026-02-15, updated 2026-02-17)
- 878-config parameter sweep: GBT + α=0.15 market blend + maker fees → **+23.9% return** (new best). Pure GBT baseline: +23.5%.
- Normalization ablation: normalizing probabilities to sum=1.0 hurts GBT (−19.9pp) but helps LR (+3.6pp). Don't normalize for GBT.
- Deep dives: best config trade-by-trade replay (+$242 Sinners trade), worst config failure anatomy, GBT vs LR probability concentration analysis.

**[d20260214_trade_signal_backtest](d20260214_trade_signal_backtest/)** (2026-02-14)
- All 6 configs lost −29% to −46%. Fees (7% taker) are the dominant loss driver.
- GBT outperforms LR (−33% vs −46%) due to fewer trades. Dynamic bankroll slightly better (+2–4pp).
- Motivated the parameter ablation follow-up (d20260214_trade_signal_ablation).

**[d20260213_feature_ablation](d20260213_feature_ablation/)** (2026-02-13–14)
- LR instability root cause: pure L2 grid retains all features. Fix: tighter grid or importance thresholding.
- 3-group subset (precursor winners + noms + oscar noms) is optimal: 80.8% accuracy, 0.0502 Brier.
- Expanding 29→47 features hurts uniformly. GBT interaction features counterproductive.

**[d20260212_counterfactual_analysis](d20260212_counterfactual_analysis/)** (2026-02-12)
- 5 what-if DGA winner scenarios with retrained models. Upset outcomes create 20–48pp edges vs market.
- Expected outcome (One Battle wins) yields only +2.2pp — not actionable.

**[d20260211_temporal_model_snapshots](d20260211_temporal_model_snapshots/)** (2026-02-11)
- LR/GBT trained at 11 dates across awards season. Models lag market confidence throughout.
- Critics Choice winner is the pivotal event (+7.7pp LR, +23.1pp GBT accuracy).
- GBT's 57pp collapse at DGA winner is the most dramatic single-event shift.

**[d20260209_relative_features](d20260209_relative_features/)** (2026-02-09)
- Percentile-in-year and z-score features. GBT baseline (80.8%) with absolute features remains strongest.
- Mixed results; later shown to be noise in Feb 13–14 ablation.

**[d20260208_dga_price_impact](d20260208_dga_price_impact/)** (2026-02-08)
- Real-money trading debrief after DGA ceremony. Market barely moved on DGA confirmation (One Battle +3¢).
- Pre-DGA prices already embedded the expected outcome. Model edge evaporated within minutes.

**[d20260207_dga_sensitivity](d20260207_dga_sensitivity/)** (2026-02-07)
- DGA winner is the single most impactful feature — accuracy jumps 5–8pp (GBT: 73.1% → 80.8%).
- DGA → Oscar winner in 69.2% of historical years. Market already prices in the expected DGA outcome.

**[d20260207_cv_splitting](d20260207_cv_splitting/)** (2026-02-07)
- Stratified, Grouped, LOYO, LOPO tested. Performance increases monotonically with more training data.
- LOYO is the only rigorous strategy for temporal prediction markets (73.1% GBT accuracy).

**[d20260207_voting_strategy](d20260207_voting_strategy/)** (2026-02-07)
- IRV era features tested. Both models assign zero importance to voting strategy features.
- Oscar voting format (IRV switch in 2009) does not affect Best Picture prediction.

**[d20260207_larger_dataset](d20260207_larger_dataset/)** (2026-02-07)
- Expanding training data from 2000 to 1980 (47 years). Pre-2000 data is noise; no improvement.
- Precursor landscape changed since the 1980s. Stick with 2000–2025 (26 years).

**[d20260207_bagged_ensemble](d20260207_bagged_ensemble/)** (2026-02-07)
- 100-bag bootstrap for uncertainty quantification. Bagging does not improve accuracy (GBT single 73.1% → bagged 69.2%).
- Real value is per-bag probability distributions — wider confidence intervals on long-shots.

**[d20260207_model_comparison](d20260207_model_comparison/)** (2026-02-07)
- LR vs GBT vs XGBoost. GBT best accuracy (80.8%); LR best calibration. XGBoost overfits (identical errors to LR).

**[d20260207_feature_ablation](d20260207_feature_ablation/)** (2026-02-07)
- Initial feature group analysis. **Precursor awards are the only feature group that matters** (LR: 73% precursor-only vs 59% full 21 features).
- Commercial/timing features actively hurt. Critic scores are noise for LR.

**[d20260206_hyperparameter_tuning](d20260206_hyperparameter_tuning/)** (2026-02-05–06)
- 4 rounds of grid search + nested CV. Nested CV gives honest deployment estimates: 59.1% expanding window vs 63% LOYO.
- Established LR and GBT baselines with proper evaluation framework.

**[d20260201_build_dataset](d20260201_build_dataset/)** (2026-02-01)
- 4-stage pipeline: Oscar nominations → film metadata (OMDb+TMDb) → precursor awards (Wikipedia) → merged dataset.
- ~260 Best Picture nominees (ceremonies 72–98). All API calls cached via diskcache.
- The `--as-of-date` mechanism controls what the model "knows" at prediction time — critical for temporal experiments.

**[d20260201_baseline_models](d20260201_baseline_models/)** (2026-02-01)
- First LR vs GBT comparison. LR 77.3% on small feature sets; GBT 54.5% (overfits with few features).
- Simpler models with fewer features (8–26) vastly outperform full feature sets (45). Precursor awards are the dominant signal.
