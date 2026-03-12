# DGA Price Impact & Trading Retrospective

**Storage:** `storage/d20260208_dga_price_impact/`

Analysis of Kalshi market reaction to One Battle after Another winning DGA (Feb 8, 2026),
plus a retrospective on pre/post-DGA trading strategy execution. A real-money trading
debrief.

## Findings

### Pre-DGA strategy (planned Feb 6-7)

Thesis: DGA is the #1 model feature. The market (One Battle at 73c) already embeds the
expected DGA outcome. Upside is in buying cheap upset insurance and placing post-DGA
limit orders.

| Phase | Plan | Budget |
|-------|------|--------|
| Pre-DGA upset bets | Marty Yes at 2c ($8), Hamnet Yes at 7c ($7-14) | $15-22 |
| Post-DGA limit orders | Scenario-specific, only matching DGA winner fills | $20-35 |
| Reserve | Hold for BAFTA/PGA | $250+ |

### Market barely moved on DGA confirmation — it was already priced in

| Nominee | Pre-DGA (Feb 7) | Post-DGA (Feb 8) | Change |
|---------|:---:|:---:|:---:|
| One Battle | 70c | 73c | **+3c** |
| Sinners | 18c | 19c | +1c |
| Hamnet | 10c | 7c | -3c |
| Marty Supreme | 3c | 2c | -1c |

One Battle moved only +3c despite model predicting 81c post-DGA. The pre-DGA price
already embedded the expected outcome:
$P(\text{Oscar}) = 0.75 \times 85\% + 0.25 \times 40\% \approx 73\%$.
DGA confirmation carried only 0.42 bits of information ($-\log_2 0.75$).

### Execution deviated from plan — sizing and timing errors

| Trade | Planned | Actual | Fees | Delta |
|-------|---------|--------|------|-------|
| Sinners NO | 427 cts @ 82c ($350) | 500 cts @ 83c ($415) | $4.94 | +$65 |
| One Battle YES | 205 cts @ 73c ($150) | 250 cts @ 73c ($183) | $3.45 | +$33 |
| Hamnet YES (pre-DGA) | 100-200 cts @ 7c ($7-14) | 400 cts @ 8c ($32) | $3.78 | 2-4x oversized |
| Marty YES | Pre-DGA at 2c ($8) | Post-DGA at 3c ($7.50) | $0.69 | Wrong timing |

Closed Hamnet/Marty same day for -$15.42 (including $4.47 fees) instead of holding
as cheap options.

### Lessons learned

- **Market prices expected catalysts in advance** — DGA confirmation was a non-event
- **Stick to planned sizing and timing** — 1/4 Kelly exists for a reason
- **Keep cheap options open** — closing eliminates optionality for minimal cash recovery
- **Fees matter disproportionately on small stakes** (11.3% effective rate on $39.50)
- **Biggest edges are in relative pricing** (Sinners overpriced at 18c), not absolute
  moves after catalysts
- **Volume spike** (48K contracts on DGA day, 2.5x normal) without price movement =
  strong market consensus
