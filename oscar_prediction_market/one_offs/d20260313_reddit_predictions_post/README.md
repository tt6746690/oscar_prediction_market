# 2026 Oscar Predictions — Model-Backed Analysis

**Storage:** `storage/d20260313_reddit_predictions_post/`

Data-driven predictions for the 98th Academy Awards (March 15, 2026), using a precursor-award model trained on 26 years of Oscar data (2000–2025). The model tracks which nominees won which precursor awards (SAG, BAFTA, Golden Globes, Critics Choice, DGA, PGA, WGA, ASC, Annie) and uses an ensemble of 4 statistical models to predict Oscar winners.

For the companion Reddit post, see [REDDIT_POST.md](REDDIT_POST.md).

## Table of Contents

- [How the Model Works](#how-the-model-works)
- [Best Picture](#best-picture)
- [Directing](#directing)
- [Best Actor](#best-actor)
- [Best Actress](#best-actress)
- [Best Supporting Actor](#best-supporting-actor)
- [Best Supporting Actress](#best-supporting-actress)
- [Original Screenplay](#original-screenplay)
- [Cinematography](#cinematography)
- [Animated Feature](#animated-feature)
- [Full Prediction Summary](#full-prediction-summary)
- [Model Accuracy & Calibration](#model-accuracy--calibration)

---

## How the Model Works

The model ensemble averages 4 independent models, each trained with leave-one-year-out cross-validation on 26 years of data:

| Model | Description | Strengths |
|-------|-------------|-----------|
| **Conditional Logit** (clogit) | Each precursor is an independent binary feature | Most interpretable; shows which single award matters most |
| **Logistic Regression** (LR) | Adds aggregate features like `precursor_wins_count` | Captures cumulative win momentum |
| **Gradient Boosting** (GBT) | Tree-based; captures non-linear interactions | Best raw accuracy (~80% across categories) |
| **Calibrated Softmax GBT** (cal_sgbt) | GBT with post-hoc probability calibration | Best calibrated probabilities |

The **avg_ensemble** (simple average of all 4) is used for all predictions below. It was selected from a [27-config × 6-model sweep](../d20260305_config_selection_sweep/) as the most robust choice — best cross-year rank stability (ρ=0.862) and lowest expected value inflation.

**Precursor snapshots:** The model ingests 8 precursor events as they occur during awards season. All 2026 precursors were resolved by March 8 (ASC + WGA). The final model snapshot (`2026-03-08_wga`) incorporates all available information.

---

## Best Picture

### Prediction: One Battle After Another (65.1%)

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **One Battle After Another** | **65.1%** | **5 / 7** | PGA, DGA, BAFTA, GG-Musical, CC |
| Sinners | 6.3% | 1 / 7 | SAG Ensemble |
| Marty Supreme | 4.9% | 0 / 7 | — |
| Hamnet | 3.8% | 1 / 7 | GG-Drama |
| Sentimental Value | 2.9% | 0 / 7 | — |

### Why One Battle After Another

OBAA swept 5 of 7 precursor awards: PGA, DGA, BAFTA Film, Golden Globe Musical/Comedy, and Critics Choice. The only precursor it didn't win was SAG Ensemble (Sinners) and Golden Globe Drama (Hamnet).

**Historical context:** Films with 5+ precursor wins have a **near-perfect track record** — historically, every Best Picture nominee with 6 wins won the Oscar (100%), and 3 of 5 nominees with 5 wins also won (60%). OBAA's closest historical analog is **The Artist** (2012), which also won PGA + DGA + BAFTA + GG-Musical + CC and went on to win Best Picture. La La Land (2017) had the same profile but famously lost to Moonlight, making it the cautionary tale (1/2 exact matches won).

### Why Not Sinners?

Sinners' only precursor win is SAG Ensemble. While SAG Ensemble is a meaningful signal — **Parasite** (2020) won with this exact same profile [SAG only, 0 other wins] — historically only 2 of 9 films with exactly this profile won the Oscar (22%). The market seems to have Sinners overpriced relative to this base rate.

### Model Agreement

All 4 models agree OBAA is the heavy favorite, with a range of 46.9% (GBT) to 74.5% (LR). The GBT model's lower estimate (46.9%) reflects its ability to capture non-linear patterns — it may be picking up on the La La Land scenario where a dominant precursor leader was upset.

---

## Directing

### Prediction: Paul Thomas Anderson (92.1%)

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **Paul Thomas Anderson** | **92.1%** | **4 / 4** | DGA, BAFTA, GG, CC |
| Chloé Zhao | 3.2% | 0 / 4 | — |
| Josh Safdie | 2.9% | 0 / 4 | — |
| Ryan Coogler | 2.9% | 0 / 4 | — |
| Joachim Trier | 2.7% | 0 / 4 | — |

### Why PTA Is a Lock

PTA swept all 4 directing precursors — DGA, BAFTA Director, Golden Globe Director, and Critics Choice Director. Of the 10 historical nominees who did the same, **9 won the Oscar** (90%). The only exception was Sam Mendes (2020) for 1917, who swept all 4 precursors but lost to Bong Joon-ho (Parasite).

This is one of the safest predictions on the board. Even the most conservative model (GBT at 81.7%) gives him a commanding lead.

---

## Best Actor

### Prediction: Timothée Chalamet (51.8%) — but this is genuinely close

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **Timothée Chalamet** | **51.8%** | **2 / 5** | GG-Musical, CC |
| Michael B. Jordan | 21.5% | 1 / 5 | SAG |
| Wagner Moura | 12.0% | 1 / 5 | GG-Drama |
| Leonardo DiCaprio | 8.8% | 0 / 5 | — |
| Ethan Hawke | 7.7% | 0 / 5 | — |

### The Model's Case for Chalamet

Chalamet leads precursor wins 2-to-1 over MBJ. He won Golden Globe Musical/Comedy and Critics Choice — two wins that historically correlate with Oscar success. **The base rate is clear:** nominees with 2 precursor wins have a 33.3% Oscar win rate, vs. 10.0% for nominees with 1 win. That 3.3× ratio is what drives the model's Chalamet preference.

The model has a strong track record on "upset" calls in Best Actor — when it disagrees with the precursor favorite, it's historically been correct 75% of the time. Notable examples: correctly picking Brendan Fraser over Austin Butler (2023), Sean Penn over Bill Murray (2004), and Adrien Brody over Daniel Day-Lewis (2003).

### The Case Against (Why MBJ Could Win)

MBJ's case rests on one powerful credential: SAG Lead Actor winner. SAG is historically the single strongest precursor for Best Actor — it has the highest individual feature weight in 3 of 4 models.

**The yellow flag for Chalamet:** His exact winner profile [GG-Musical + CC, no SAG/BAFTA] has been matched by 3 historical nominees — **Paul Giamatti** (2024), **Christian Bale** (2019), and **Michael Keaton** (2015). **All 3 lost the Oscar.** While a 0/3 sample is small, it's notable that this specific combination of wins has never produced an Oscar winner.

MBJ's profile [SAG only, no others] also has 0/3 historical winners — **Timothée Chalamet himself** (2025, A Complete Unknown), **Denzel Washington** (2017, Fences), and **Johnny Depp** (2004, Pirates). Neither profile has a winning precedent.

### Model Disagreement Is High

This is the category with the most model disagreement:
- **cal_sgbt** (81.0% Chalamet) and **GBT** (48.6% Chalamet) strongly favor Chalamet
- **clogit** (34.2% MBJ vs 31.1% Chalamet) actually slightly favors MBJ
- **LR** (46.4% Chalamet) splits the difference

The key difference: clogit treats each precursor independently (SAG winner is a slightly heavier feature than CC winner), while GBT/cal_sgbt use `precursor_wins_count`, which gives Chalamet's 2-vs-1 advantage a large boost.

**Bottom line:** The model says Chalamet, but with genuine uncertainty. This is the kind of race that makes awards season interesting.

For deeper analysis, see the [Best Actor Diagnostics](../d20260310_best_actor_diagnostics/README.md).

---

## Best Actress

### Prediction: Jessie Buckley (98.2%)

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **Jessie Buckley** | **98.2%** | **4 / 5** | SAG, BAFTA, GG-Drama, CC |
| Kate Hudson | 1.5% | 0 / 5 | — |
| Rose Byrne | 1.2% | 1 / 5 | GG-Musical |
| Renate Reinsve | 0.5% | 0 / 5 | — |
| Emma Stone | 0.5% | 0 / 5 | — |

### Why Buckley is the Biggest Lock of the Night

Buckley won SAG, BAFTA, Golden Globe Drama, and Critics Choice — 4 of 5 precursors. **Every single historical nominee with this exact profile won the Oscar: 8 for 8.** This includes Renée Zellweger (2020), Frances McDormand (2018), Brie Larson (2016), Julianne Moore (2015), and Cate Blanchett (2014).

Moreover, among Leading Actress nominees with 4 precursor wins, the historical Oscar win rate is **100%** (9/9). Buckley's only missing precursor is Golden Globe Musical/Comedy, which didn't apply since Hamnet is a drama.

All 4 models agree: the lowest probability is GBT at 95.5%. Rose Byrne (GG-Musical winner for If I Had Legs I'd Kick You) is the nominal runner-up but her Golden Globe Musical win translates to only ~1% model probability. Historically, nominees with just 1 GG-Musical win and nothing else have a 0% Oscar win rate from this specific profile (0/10).

---

## Best Supporting Actor

### Prediction: Sean Penn (55.0%) — closer than expected

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **Sean Penn** | **55.0%** | **2 / 4** | SAG, BAFTA |
| Stellan Skarsgård | 37.0% | 1 / 4 | GG |
| Jacob Elordi | 5.1% | 1 / 4 | CC |
| Delroy Lindo | 1.2% | 0 / 4 | — |
| Benicio del Toro | — | 0 / 4 | — |

### Penn's Precursor Advantage

Penn won both SAG and BAFTA Supporting Actor — the two most industry-facing precursors. Historically, nominees with 2 precursor wins in Supporting Actor have a **50% Oscar win rate** (5/10), jumping to 83% for 3 wins.

His closest historical analog is **Christopher Walken** (2003, Catch Me If You Can) — same SAG + BAFTA combo, no GG or CC. Walken lost that year. But at one Hamming distance away, **Troy Kotsur** (2022, CODA: SAG + BAFTA + CC) won, as did **Mark Rylance** (2016: BAFTA only) — suggesting that BAFTA especially carries weight in this category.

### Why Skarsgård Is the Serious Threat

Skarsgård's Golden Globe win makes him a real contender. The model assigns him 37% — not far behind Penn. Historically, 1-win nominees have a 21.7% base rate in Supporting Actor. Skarsgård's specific profile [GG only] has one notable winner: **George Clooney** (2006, Syriana), who won the Oscar with only a Golden Globe to his name.

### Extreme Model Disagreement

This is the category with the **widest model disagreement in the entire slate:**
- **clogit** says Penn at 93.7% (near-lock) — because SAG + BAFTA are its two features
- **cal_sgbt** says Skarsgård at 76.8% — its calibrated GBT sees something the simpler models miss
- The ensemble splits the difference at 55/37

This split means the model itself is uncertain about what matters more: the raw precursor count (Penn) or some non-linear interaction the tree model is capturing (Skarsgård). Keep an eye on this one.

---

## Best Supporting Actress

### Prediction: Amy Madigan (88.4%) — the model's biggest contrarian call

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **Amy Madigan** | **88.4%** | **2 / 4** | SAG, CC |
| Wunmi Mosaku | 21.2% | 1 / 4 | BAFTA |
| Teyana Taylor | 4.3% | 1 / 4 | GG |
| Inga Ibsdotter Lilleaas | 3.1% | 0 / 4 | — |
| Elle Fanning | — | 0 / 4 | — |

### Why Madigan Is the Model's Top Pick

Madigan won SAG and Critics Choice — 2 of 4 Supporting Actress precursors. The base rates strongly support her: nominees with 2 precursor wins have a **66.7% Oscar win rate** in this category (6/9). More importantly, her exact winner profile [SAG + CC] has 2 historical matches, and **both won the Oscar**: **Lupita Nyong'o** (2014, 12 Years a Slave) and **Alicia Vikander** (2016, The Danish Girl). That's 2/2 — a perfect track record for this specific combination.

The SAG + CC combination is particularly powerful because it captures both industry (SAG) and critical (CC) support — suggesting broad Academy appeal.

### Why Mosaku Is the Main Alternative

Mosaku has the BAFTA, which is meaningful — but her profile [BAFTA only] historically matches **Kerry Condon** (2023), **Rachel Weisz** (2019), and **Helena Bonham Carter** (2011), among others. Of 6 similar nominees, **2 won** (Penélope Cruz in 2009 and Tilda Swinton in 2008), giving a 33% win rate. Respectable, but less than Madigan's 66.7% base rate with 2 wins.

BAFTA and SAG splitting is notable. In past years when they split in this category, the Oscar typically goes to the nominee from the **stronger Best Picture contender**. Both Sinners (Mosaku's film) and Weapons (Madigan's film) are in the conversation, but OBAA — the Best Picture favorite — features supporting actress nominee Teyana Taylor (who won the Golden Globe but that's it).

### Why Taylor Isn't the Threat

Taylor's only win is the Golden Globe. Her exact profile [GG only, no SAG/BAFTA/CC] matches 4 historical nominees, and **none won the Oscar** (0/4). Notable examples: Cate Blanchett (2008, I'm Not There), Natalie Portman (2005, Closer). The Golden Globe Supporting Actress has historically been the weakest predictor in this category.

### This Is Where the Real Money Is

The market (as of March 11) had Madigan at only ~31% and Teyana Taylor at ~47%. The model at 88.4% vs market at 31% represents a **57 percentage point disagreement** — the largest gap across all nominees in all categories. Either the model sees something the market doesn't (Madigan's dominant precursor profile), or the market knows something the model can't capture (perhaps narrative momentum, campaign strength, or the "Best Picture coattails" effect for Taylor being in OBAA).

---

## Original Screenplay

### Prediction: Sinners / Ryan Coogler (96.8%)

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **Sinners (Ryan Coogler)** | **96.8%** | **3 / 4** | WGA, BAFTA, CC |
| Marty Supreme (Bronstein/Safdie) | 1.8% | 0 / 4 | — |

### Why Sinners Is Another Lock

Coogler won WGA, BAFTA Original Screenplay, and Critics Choice — 3 of 4 precursors (missing only Golden Globe Screenplay, where he was nominated). **Every historical nominee with this exact profile won the Oscar: 5 for 5.** This includes Emerald Fennell (2021, Promising Young Woman), Tom McCarthy (2016, Spotlight), Diablo Cody (2008, Juno), and Michael Arndt (2007, Little Miss Sunshine).

General base rate: 3-win nominees in Original Screenplay have a 100% historical Oscar win rate (9/9).

---

## Cinematography

### Prediction: One Battle After Another / Michael Bauman (88.9%)

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **One Battle After Another (Michael Bauman)** | **88.9%** | **2 / 3** | ASC, BAFTA |
| Marty Supreme (Darius Khondji) | 4.7% | 0 / 3 | — |
| Frankenstein (Dan Laustsen) | 4.6% | 0 / 3 | — |
| Sinners (Autumn Durald Arkapaw) | 4.5% | 0 / 3 | — |
| Train Dreams (Adolpho Veloso) | 2.1% | 1 / 3 | CC |

### Why Bauman Has This

Bauman won ASC and BAFTA — the two most heavyweight cinematography precursors. His exact profile [ASC + BAFTA, no CC] has 7 historical matches, with **5 winning the Oscar** (71%). Recent winners with this profile include Greig Fraser (2022, Dune) and Anthony Dod Mantle (2009, Slumdog Millionaire).

General base rate: nominees with 2 cinematography precursor wins have a 69.2% Oscar win rate. With the specific ASC + BAFTA combo, it's even higher (71%). Bauman's only missing precursor is CC (won by Adolpho Veloso for Train Dreams), but CC Cinematography has historically been the weakest predictor in this category.

---

## Animated Feature

### Prediction: KPop Demon Hunters (76.8%) — but Zootopia 2 is interesting

| Nominee | Model | Precursor Wins | Key Wins |
|---------|-------|----------------|----------|
| **KPop Demon Hunters** | **76.8%** | **4 / 5** | Annie, PGA, GG, CC |
| Zootopia 2 | 31.1% | 1 / 5 | BAFTA |
| Little Amélie | 3.6% | 0 / 5 | — |
| Elio | 2.4% | 0 / 5 | — |
| Arco | 1.9% | 0 / 5 | — |

### Why KPop Demon Hunters

KPop Demon Hunters won 4 of 5 animated feature precursors: Annie Award, PGA Animated, Golden Globe Animated, and Critics Choice Animated. The only precursor it missed was BAFTA Animated (where it wasn't eligible — Zootopia 2 won). Nominees with 4 precursor wins in this category have a **75% historical Oscar win rate** (3/4).

Its exact winner profile [Annie + PGA + GG + CC, no BAFTA] matches **Zootopia** (2017, the original) and **Cars** (2007). Zootopia won the Oscar; Cars did not (losing to Happy Feet). So its exact analog is 1/2 — a coin flip at the profile level, but the 4-win base rate (75%) provides more context.

### Why Zootopia 2 Could Surprise

Zootopia 2's BAFTA win is notable — it beat the field at an awards body that often reflects Academy taste. However, its profile [BAFTA only, 1 win] has 3 historical matches: **Wallace & Gromit: Vengeance Most Fowl** (2025, lost), **Kubo and the Two Strings** (2017, lost), and **Happy Feet** (2007, won). The 1/3 win rate (33%) and general 1-win base rate (28.6%) make it a real dark horse, but not the favorite.

The big caveat: KPop Demon Hunters wasn't BAFTA-eligible. If it had been, it might have won BAFTA too, pushing it to 5/5 precursors (which has a 100% historical Oscar win rate). The BAFTA win for Zootopia 2 may be somewhat inflated by the absence of the strongest competitor.

### Model Disagreement

Clogit (most interpretable) gives KPop Demon Hunters only 52.7% — the lowest of any model — because it treats each precursor independently and BAFTA is missing. Cal_sgbt (best calibrated) gives 91.5% — the highest — because it captures the aggregate 4-win advantage. The spread (52.7%–91.5%) is wide, reflecting genuine uncertainty about how much the missing BAFTA matters.

---

## Full Prediction Summary

| Category | Prediction | Model Prob | Precursor Wins | Confidence |
|----------|-----------|------------|----------------|------------|
| **Best Picture** | One Battle After Another | 65.1% | 5/7 | High |
| **Directing** | Paul Thomas Anderson | 92.1% | 4/4 | Very High |
| **Best Actor** | Timothée Chalamet | 51.8% | 2/5 | Low |
| **Best Actress** | Jessie Buckley | 98.2% | 4/5 | Very High |
| **Supporting Actor** | Sean Penn | 55.0% | 2/4 | Moderate |
| **Supporting Actress** | Amy Madigan | 88.4% | 2/4 | High |
| **Original Screenplay** | Sinners (Ryan Coogler) | 96.8% | 3/4 | Very High |
| **Cinematography** | One Battle After Another | 88.9% | 2/3 | High |
| **Animated Feature** | KPop Demon Hunters | 76.8% | 4/5 | High |

### Categories where model and market notably disagree (as of March 11)

| Category | Nominee | Model | Market | Gap |
|----------|---------|-------|--------|-----|
| Supporting Actress | Amy Madigan | 88.4% | ~31% | **+57pp** |
| Actor (Leading) | M.B. Jordan | 21.5% | ~60% | **-39pp** |
| Actor (Leading) | T. Chalamet | 51.8% | ~29% | **+23pp** |
| Animated Feature | KPop DH | 76.8% | ~90% | **-13pp** |
| Animated Feature | Zootopia 2 | 31.1% | ~9% | **+22pp** |
| Supporting Actor | Skarsgård | 37.0% | ~26% | **+11pp** |

---

## Model Accuracy & Calibration

Across all categories and 26 years of leave-one-year-out cross-validation:

| Metric | avg_ensemble | GBT | cal_sgbt | LR | clogit |
|--------|-------------|-----|----------|----|--------|
| Accuracy (all categories) | ~80% | 80.8% | ~80% | ~78% | ~76% |
| Best Actor accuracy | 85-89% per model | 88.5% | 88.5% | 88.5% | 84.6% |
| Calibration (ECE) | Low | 0.046 | 0.043 | 0.153 | 0.084 |

The model has been validated through:
- [Cross-validation studies](../d20260207_cv_splitting/) — LOYO is the only rigorous temporal strategy
- [Config selection sweep](../d20260305_config_selection_sweep/) — 27 configs × 6 models evaluated
- [Kelly portfolio backtesting](../d20260305_portfolio_kelly/) — 39 bankroll strategies tested
- [Historical backtest](../d20260225_buy_hold_backtest/) — 89.3% of configurations profitable both test years
- [Feature ablation](../d20260207_feature_ablation/) — Precursor awards are the only feature group that matters

## How to Run

```bash
cd "$(git rev-parse --show-toplevel)"
uv run python -m oscar_prediction_market.one_offs.d20260313_reddit_predictions_post.gather_evidence
```

## Output Structure

```
storage/d20260313_reddit_predictions_post/
├── all_categories_evidence.txt    # Raw analysis output for all 9 categories
```
