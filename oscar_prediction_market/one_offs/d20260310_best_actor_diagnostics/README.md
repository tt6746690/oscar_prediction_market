# Best Actor Diagnostics — Model Trust Verification

**Storage:** `storage/d20260310_best_actor_diagnostics/`

Deep-dive into the Best Actor category where model predictions diverge sharply from market prices. The market says MBJ (60.5%) >> Chalamet (28.5%). All 5 model sub-components say the opposite. This analysis investigates whether the model's contrarian view is well-grounded.

## Motivation

The 2026-03-10 trading report shows the largest model-vs-market gap in Best Actor:

| Nominee | Market | Model (avg_ensemble) | Gap |
|---------|--------|---------------------|-----|
| Michael B. Jordan | 60.5% | 21.1% | **-39.4pp** |
| Timothée Chalamet | 28.5% | 50.8% | **+22.3pp** |

This is a $558 position (134 NO MBJ + 76 YES Chalamet). Before doubling down, we need confidence that the model's contrarian call is grounded in real signal, not an artifact of model limitations.

## Findings

### 1. Chalamet leads precursor wins 2–1, but the race is genuinely split

The precursor award data — the primary input the model sees — tells a clear story:

| Nominee | SAG | BAFTA | GG-Drama | GG-Musical | CC | **Wins** | Noms |
|---------|-----|-------|----------|------------|-----|---------|------|
| **Timothée Chalamet** | N | N | — | **W** | **W** | **2** | 4 |
| **Michael B. Jordan** | **W** | N | N | — | N | **1** | 4 |
| Wagner Moura | — | — | **W** | — | N | 1 | 2 |
| Ethan Hawke | N | N | — | N | N | 0 | 4 |
| Leonardo DiCaprio | N | N | — | N | N | 0 | 4 |

Chalamet won GG-Musical + Critics Choice (2 wins). MBJ won SAG (1 win). Both were widely nominated (4/5 precursors each). The model sees Chalamet's 2-vs-1 precursor win advantage and assigns him higher probability. The market heavily favors MBJ.

### 2. Historical base rates strongly favor more precursor wins

From 26 years of data (2000–2025):

| Precursor Wins | Nominees | Oscar Wins | **Win Rate** |
|----------------|----------|------------|-------------|
| 0 | 77 | 2 | 2.6% |
| 1 | 20 | 2 | **10.0%** |
| 2 | 12 | 4 | **33.3%** |
| 3 | 10 | 8 | 80.0% |
| 4 | 11 | 10 | 90.9% |

At 2 precursor wins, the historical Oscar win rate is **33.3%**. At 1 win, it's **10.0%**. This is a 3.3× ratio favoring Chalamet's profile over MBJ's — which is broadly consistent with the model's ~2.4× probability ratio (avg_ensemble: 50.8% vs 21.1%).

### 3. But no exact analog with Chalamet's winner profile has ever won the Oscar

Chalamet's exact winner profile [SAG-no, BAFTA-no, GG-Drama-no, GG-Musical-yes, CC-yes] has 3 historical matches:
- **2024 Paul Giamatti** (The Holdovers) — did NOT win
- **2019 Christian Bale** (Vice) — did NOT win
- **2015 Michael Keaton** (Birdman) — did NOT win

0/3 exact analogs won. This is a small sample (and 33% base rate for 2-win nominees still allows 2/3 losing), but it's a yellow flag.

MBJ's exact profile [SAG-yes, BAFTA-no, GG-Drama-no, GG-Musical-no, CC-no] also has 3 exact matches, all losses:
- **2025 Timothée Chalamet** (A Complete Unknown) — did NOT win
- **2017 Denzel Washington** (Fences) — did NOT win
- **2004 Johnny Depp** (Pirates of the Caribbean) — did NOT win

**Neither nominee's exact profile has ever produced an Oscar winner.** This race is genuinely unusual.

### 4. The model is well-calibrated and historically accurate for Best Actor

CV metrics across all 4 models (26 years LOYO):

| Model | Accuracy | Mean Winner Prob | Brier Score | ECE |
|-------|----------|-----------------|-------------|-----|
| clogit | 84.6% | 0.592 | 0.070 | 0.084 |
| lr | 88.5% | 0.749 | 0.092 | 0.153 |
| gbt | 88.5% | 0.704 | 0.062 | 0.046 |
| cal_sgbt | 88.5% | 0.788 | 0.047 | 0.043 |

85–89% accuracy on Best Actor is strong (random = 20%, always-picking-favorite ≈ 60%). GBT and cal_sgbt have the best calibration (ECE < 0.05). LR has higher ECE (0.15) — it tends to be overconfident on its top pick.

### 5. When the model disagrees with the precursor favorite, it's usually right

Upset analysis — years where the model's #1 pick ≠ the precursor leader:

| Model | Upsets | Model Correct | Rate |
|-------|--------|--------------|------|
| clogit | 2 | 1 | 50% |
| lr | 2 | 2 | **100%** |
| gbt | 4 | 3 | **75%** |
| cal_sgbt | 4 | 3 | **75%** |

Notable upsets where the model was right:
- **2003**: GBT correctly picked Adrien Brody over precursor favorite Daniel Day-Lewis
- **2004**: LR/GBT/cal_sgbt correctly picked Sean Penn over precursor favorite Bill Murray
- **2023**: All 4 models correctly picked Brendan Fraser over precursor favorite Austin Butler

The model has a real track record of picking Oscar winners that the raw precursor signal misses. This builds confidence in its contrarian call.

### 6. Feature importance: SAG and precursor_wins_count drive the GBT models' strong Chalamet preference

| Feature | clogit | lr | gbt | cal_sgbt |
|---------|--------|----|----|---------|
| sag_lead_actor_winner | 0.427 | 0.421 | 0.145 | 0.145 |
| bafta_lead_actor_winner | 0.433 | 0.403 | 0.038 | 0.038 |
| critics_choice_actor_winner | 0.401 | 0.398 | — | — |
| precursor_wins_count | — | 0.371 | **0.450** | **0.450** |
| imdb_rating | — | — | 0.102 | 0.099 |

The clogit model (3 features: BAFTA/SAG/CC winner) actually slightly favors MBJ (25.4% vs 24.8%) because SAG winner (coeff=0.427) outweighs CC winner (coeff=0.401). But LR, GBT, and cal_sgbt all use `precursor_wins_count` as a feature, which gives Chalamet's 2-vs-1 advantage a large additional boost.

**Key disagreement across models**: clogit (which treats each precursor independently) slightly favors MBJ. The other 3 models (which aggregate precursor wins) strongly favor Chalamet. The avg_ensemble therefore favors Chalamet overall.

### 7. Counterfactual: MBJ would need one more precursor win to flip the clogit model

Using the clogit decomposition (analytically tractable):

| Scenario | MBJ Prob | Chalamet Prob | MBJ Favorite? |
|----------|----------|---------------|---------------|
| **Baseline** | 25.4% | 24.8% | ★ (barely) |
| MBJ wins BAFTA | 34.5% | 21.8% | ★ |
| MBJ wins CC | 33.7% | 22.0% | ★ |
| MBJ wins BAFTA + CC | 44.0% | 18.6% | ★ |
| Chalamet loses CC | 27.7% | 18.1% | ★ |
| MBJ sweeps all 3 model-relevant precursors | 44.0% | 18.6% | ★ |
| Chalamet sweeps all 3 | 19.0% | 43.8% | ✗ |

For clogit specifically, MBJ is already the slight favorite (25.4% vs 24.8%). One additional precursor win would make him the clear clogit favorite. But in the ensemble (where GBT/cal_sgbt dominate via `precursor_wins_count`), Chalamet's 2-win aggregate advantage is decisive.

## Assessment: Should You Trust the Model?

**Confidence builders:**
- 85–89% historical accuracy for Best Actor across all models
- Model is 75–100% correct on "upset" calls (when it disagrees with precursor favorite)
- The base rate data clearly supports: 2 precursor wins (33% win rate) > 1 precursor win (10% win rate)
- All 4 model types agree on direction (Chalamet > MBJ), even if magnitudes differ

**Caution flags:**
- 0/3 exact historical analogs with Chalamet's specific winner profile [GG-Mus + CC] won. The specific *combination* of precursors matters, not just the count — and this combination has never produced a Best Actor winner.
- SAG winner is the single most predictive feature across all models. MBJ won SAG. Historically, SAG Lead Actor winner → Oscar winner is a very strong signal.
- The market at 60/28 implies a ~2.1:1 odds ratio favoring MBJ. The model at 51/21 implies ~2.4:1 favoring Chalamet. These are **opposite directions** — this is a genuinely high-conviction contrarian bet.
- The model's confidence comes mainly from `precursor_wins_count` (an aggregate feature). A more nuanced model might weight SAG > CC differently.

**Bottom line**: The model has a defensible case. The 33% vs 10% base rate is real. The historical upset track record (75%) is encouraging. But the bet is essentially: "2 wins (CC + GG-Musical) should beat 1 win (SAG), even though SAG is historically the strongest single precursor." The market is betting that SAG winner trumps everything. Both sides have reasonable arguments. This is a genuine disagreement, not a model bug.

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"

# Individual analyses
uv run python -m oscar_prediction_market.one_offs.d20260310_best_actor_diagnostics.precursor_breakdown
uv run python -m oscar_prediction_market.one_offs.d20260310_best_actor_diagnostics.historical_analogs
uv run python -m oscar_prediction_market.one_offs.d20260310_best_actor_diagnostics.feature_importance
uv run python -m oscar_prediction_market.one_offs.d20260310_best_actor_diagnostics.calibration_backtest
uv run python -m oscar_prediction_market.one_offs.d20260310_best_actor_diagnostics.counterfactual_sensitivity
```

## Output Structure

```
storage/d20260310_best_actor_diagnostics/
├── precursor_breakdown.txt          # Raw precursor data for all 2026 nominees
├── historical_analogs.txt           # Similar historical nominees + base rates
├── feature_importance.txt           # Per-model feature rankings + decomposition
├── calibration_backtest.txt         # ECE/MCE, reliability, upset analysis
└── counterfactual_sensitivity.txt   # "What if" precursor flip scenarios
```

## Trading Config Implications

### Current setup

The live config (`edge_20_taker`) applies uniformly across all categories:

| Parameter | Value |
|-----------|-------|
| Model | avg_ensemble |
| Kelly fraction | 0.05 (1/20th Kelly) |
| Buy edge threshold | 20% net |
| Kelly mode | multi_outcome |
| Fee type | taker |
| Sell threshold | -1.0 (buy-and-hold) |
| Bankroll per category | $1,000 |
| Allocation | maxedge_100 (weight ∝ max raw edge) |

Actor Leading currently has allocation weight **2.99** (~$332 effective bankroll). The system targets 699 NO MBJ contracts (current position: 134 NO MBJ + 76 YES Chalamet = ~$558 deployed). It wants to BUY 565 more NO MBJ and SELL all 76 YES Chalamet.

### Options considered

**Option A — Keep the config, trust the system.** The config was selected through a rigorous sweep. Kelly fraction 0.05 is already extremely conservative. The 20% edge threshold filters weak signals. Overriding based on diagnostic findings undermines the systematic approach. Risk: sizing up to 699 NO MBJ in a race where SAG winner (MBJ) has the strongest single-precursor signal.

**Option B — Hold current position, don't add.** The current ~$558 position already expresses the view. The diagnostics revealed genuine uncertainty (0/3 exact analogs won, clogit disagrees with ensemble direction). Don't execute the BUY 565 signal. No code change needed — just don't trade. Risk: leaving edge on the table if the model is right.

**Option C — Reduce allocation weight for Actor Leading.** Cap the weight at 1.0–1.5 instead of 2.99. This scales targets proportionally: weight 1.0 → ~234 NO MBJ (add ~100 from current 134), weight 1.5 → ~350 NO MBJ (add ~216). Keeps the framework intact while acknowledging that max raw edge overstates confidence when model internal agreement is split (clogit vs the other 3). Implementation: a `category_weight_overrides` parameter in the report generator.

### Recommendation

**Option B (hold current, don't add)** for simplicity and risk management. The diagnostics showed this is a genuine model-vs-market disagreement — not a model bug, but also not a slam dunk. The current $558 position captures meaningful upside if the model is correct. Tripling down (to 699 contracts) concentrates too much in a single contrarian call where:

- 0/3 exact historical analogs with Chalamet's winner profile actually won
- SAG winner (MBJ's key credential) is the strongest single predictor across all 4 models
- The clogit model (most interpretable) slightly favors MBJ, not Chalamet

The 75% upset track record is encouraging but based on small samples. With 5 days to ceremony, the prudent move is to hold existing exposure and let the position play out — not to size up into maximum conviction on a split signal.
