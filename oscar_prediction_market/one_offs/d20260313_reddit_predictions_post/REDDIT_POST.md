# I built a statistical model to predict the Oscars — here are my picks for every category, backed by 26 years of data

I've been working on a prediction model that tracks which Oscar nominees won precursor awards — SAG, BAFTA, Golden Globes, Critics Choice, DGA, PGA, WGA, ASC, and the Annies — going back to 2000. It's trained on 26 years of data and is around **80% accurate** historically across all categories.

All precursors for the 2026 season are now resolved (WGA and ASC on March 8 were the last), so the model has its final predictions. Here's what it says — and more importantly, *why* it says it.

**Quick summary:**

| Category | Pick | Confidence |
|----------|------|------------|
| Best Actress | Jessie Buckley | 🟢 Lock (98%) |
| Original Screenplay | Sinners | 🟢 Lock (97%) |
| Directing | PTA | 🟢 Near-lock (92%) |
| Cinematography | OBAA | 🟢 Strong (89%) |
| Supporting Actress | Amy Madigan | 🟢 Strong (88%) |
| Animated Feature | KPop Demon Hunters | 🟡 Solid (77%) |
| Best Picture | OBAA | 🟡 Solid (65%) |
| Supporting Actor | Sean Penn | 🟡 Lean (55%) |
| Best Actor | Chalamet | 🔴 Coin flip (52%) |

Now the reasoning — category by category.

---

**Best Picture: One Battle After Another**

OBAA swept 5 of 7 major precursors — PGA, DGA, BAFTA Film, Golden Globe Musical/Comedy, and Critics Choice. The only things it didn't win were SAG Ensemble (Sinners) and Golden Globe Drama (Hamnet). The model gives it [**65%**](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#best-picture).

Historically, Best Picture nominees with 5+ precursor wins have an extraordinary track record. OBAA's exact profile matches The Artist (2012), which also won PGA + DGA + BAFTA + GG + CC and [won the Oscar](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#why-one-battle-after-another). The cautionary analog is La La Land (2017), same profile, which famously lost to Moonlight. So it's 1 out of 2 exact matches — but the broader base rate for 5-win nominees is very strong.

Sinners' only precursor win is SAG Ensemble. This happened with Parasite in 2020 — [SAG Ensemble as the only win — and Parasite won BP](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#why-not-sinners). But it's been a 2/9 hit rate historically for that profile. The SAG Ensemble win feels more like "the cast of Sinners is beloved" than "Sinners is the Best Picture frontrunner."

---

**Directing: Paul Thomas Anderson**

This is essentially a lock. PTA swept all 4 directing precursors: DGA, BAFTA, Golden Globe, and Critics Choice. [**9 out of 10** historical nominees who did this won the Oscar](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#why-pta-is-a-lock) — the only exception being Sam Mendes (1917, 2020), who lost to Bong Joon-ho for Parasite. Model gives PTA **92%**. Moving on.

---

**Best Actor: Timothée Chalamet — but this is genuinely anyone's race**

This is the most fascinating category this year, and where [the model and the betting markets diverge the hardest](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#best-actor). The market has MBJ around 60%, Chalamet around 29%. The model flips it: Chalamet **52%**, MBJ **21%**.

Why? Chalamet won Golden Globe Musical/Comedy and Critics Choice — that's 2 precursor wins. MBJ won SAG — that's 1. The historical base rates are clear: nominees with 2 wins have a [33% Oscar win rate, vs 10% for 1-win nominees](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260310_best_actor_diagnostics/README.md#2-historical-base-rates-strongly-favor-more-precursor-wins). The model just counts wins and says more is better.

But here's the thing — [Chalamet's exact winner profile (GG-Musical + CC, no SAG, no BAFTA) has been matched by 3 previous nominees](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260310_best_actor_diagnostics/README.md#3-but-no-exact-analog-with-chalamets-winner-profile-has-ever-won-the-oscar): Paul Giamatti (2024), Christian Bale (2019), and Michael Keaton (2015). All 3 lost. MBJ's profile (SAG only) also has 0 winners out of 3. So *neither* profile has a winning precedent, making this a genuinely unusual race.

The market's bet on MBJ comes down to SAG being the single most predictive individual award for Best Actor. And that's true. The model's bet on Chalamet comes down to 2 > 1 in total precursor wins. Both arguments are reasonable.

What's interesting is how [the race shifted through awards season](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260310_best_actor_diagnostics/README.md#1-chalamet-leads-precursor-wins-21-but-the-race-is-genuinely-split). After the Golden Globes and Critics Choice, Chalamet was sitting on 2 wins with real momentum. Then MBJ won SAG right in the middle of Oscar voting, and the market swung hard to MBJ. But the model doesn't care about *recency* — it just sees 2 wins vs 1 and says the numbers favor Chalamet. If I had to bet my life on it, I'd say Chalamet, but I wouldn't feel great about it.

---

**Best Actress: Jessie Buckley**

The biggest lock of the night. Buckley won SAG, BAFTA, Golden Globe Drama, and Critics Choice — 4 of 5 precursors. [**Every nominee in history with this exact profile won the Oscar: 8 for 8.**](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#why-buckley-is-the-biggest-lock-of-the-night) Renée Zellweger, Frances McDormand, Brie Larson, Julianne Moore, Cate Blanchett... the list goes on. Model gives her **98%**. This is as close to inevitable as Oscar predictions get.

---

**Supporting Actor: Sean Penn — but Skarsgård is a real threat**

Penn won both SAG and BAFTA — the two most industry-facing precursors, giving him 2 out of 4. [Model gives him 55%](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#best-supporting-actor). The base rate for 2-win nominees in this category is 50%.

But under the hood, this category has the [widest internal disagreement](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#extreme-model-disagreement) of any category — different components of the model range from a near-lock on Penn to favoring Skarsgård. The ensemble settles at 55/37, but this isn't a confident call.

Skarsgård won the Golden Globe, and his profile has a notable precedent: George Clooney (2006, Syriana) won the Oscar with only a Golden Globe to his name. The model treats Skarsgård as a legitimate dark horse at 37%. Don't be shocked if he pulls the upset.

---

**Supporting Actress: Amy Madigan**

Here's where my model is most contrarian. Prediction markets (as of March 11) have Teyana Taylor at ~47% and Madigan at only ~31%. The model says Madigan at a commanding [**88%**](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#best-supporting-actress).

Why? Madigan won SAG and Critics Choice — 2 precursor wins. In Supporting Actress, [**nominees with 2 wins have a 67% Oscar win rate**](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#why-madigan-is-the-models-top-pick). And here's the kicker: her *exact* profile [SAG + CC] has been matched by **Lupita Nyong'o** (12 Years a Slave) and **Alicia Vikander** (The Danish Girl) — and **both won the Oscar**. That's **2 for 2**.

The market seems to be betting on the "Best Picture coattails" effect for Taylor (she's in OBAA, the BP frontrunner) or Mosaku (she's in Sinners). But Taylor's only win is the Golden Globe, and [her exact profile has 0/4 historical winners](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#why-taylor-isnt-the-threat). Mosaku won BAFTA, which is more meaningful, but still just 1 win.

This is the category I'm most confident the market is mispricing.

---

**Original Screenplay: Sinners (Ryan Coogler)**

Coogler won WGA, BAFTA, and Critics Choice — 3 of 4 precursors. [**Every historical nominee with this exact profile won: 5 for 5**](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#original-screenplay), including Emerald Fennell (Promising Young Woman), Tom McCarthy (Spotlight), and Diablo Cody (Juno). Model gives it **97%**.

---

**Cinematography: One Battle After Another (Michael Bauman)**

Bauman won ASC and BAFTA — the two most important cinematography precursors. [**71% of nominees with this exact profile won the Oscar**](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#cinematography) (5/7), including Greig Fraser for Dune (2022). Model gives it **89%**.

---

**Animated Feature: KPop Demon Hunters — though Zootopia 2's BAFTA win is interesting**

KPop Demon Hunters won 4 of 5 precursors — Annie, PGA Animated, Golden Globe, and Critics Choice. It wasn't eligible for BAFTA (where Zootopia 2 won). Model gives it [**77%**](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md#animated-feature).

The nuance is that KPop DH's BAFTA ineligibility means Zootopia 2's BAFTA win might be inflated — it won by default against a weaker field. Still, BAFTA carries weight, and Zootopia 2 gets **31%** from the model. Its profile matches Happy Feet (2007), which actually won the Oscar with only a BAFTA. Not impossible, but the 4-win base rate favors KPop DH.

---

**The OBAA Sweep**

One thing worth noting: the model has OBAA winning **Best Picture, Directing, Cinematography**, and its cast member Sean Penn for **Supporting Actor** — a potential 4-trophy haul. PTA could join the rare club of directors who won both BP and Director in the same year. Meanwhile, Sinners' consolation prizes are Original Screenplay (a lock at 97%) and possibly Best Actor if MBJ pulls it off. This feels like one of those years where two films split the awards rather than one movie running the table.

---

**Most Likely Upsets**

If I had to pick where the model is most likely to be *wrong*, it's these three:

1. **Best Actor — MBJ over Chalamet.** The model says Chalamet at 52%, but SAG is the strongest single predictor, MBJ won it, and the momentum from winning during Oscar voting is real. This is a genuine coin flip.

2. **Supporting Actor — Skarsgård over Penn.** The model says Penn at 55%, but Skarsgård has a clear path via the Golden Globe, and previous GG-only winners like George Clooney (2006) have pulled this off.

3. **Animated Feature — Zootopia 2 over KPop Demon Hunters.** KPop DH is the clear favorite at 77%, but the BAFTA win for Zootopia 2 is legitimate, and Happy Feet (2007) proved a BAFTA-only animated film can win the Oscar.

---

The model's biggest bets against conventional wisdom are **Amy Madigan for Supporting Actress** and **Chalamet over MBJ for Best Actor**. If those two hit, it would validate the "count the precursor wins" approach. If they miss, it might mean SAG alone trumps everything.

Sunday can't come soon enough. If you want a deeper dive on any specific category — historical analogs, base rates, or why the runner-up might or might not pull the upset — ask and I'll dig into the data.

---

*Full methodology, precursor breakdowns, historical analogies, and model details: [README.md](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260313_reddit_predictions_post/README.md)*

*Best Actor deep dive (the most contentious race): [Best Actor Diagnostics](https://github.com/tt6746690/oscar_prediction_market/blob/main/oscar_prediction_market/one_offs/d20260310_best_actor_diagnostics/README.md)*

*Full source code and experiment history: [oscar_prediction_market](https://github.com/tt6746690/oscar_prediction_market)*
