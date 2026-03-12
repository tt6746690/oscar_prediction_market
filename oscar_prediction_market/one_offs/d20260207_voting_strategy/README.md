# Voting Strategy Features — Does IRV Era Matter?

**Storage:** `storage/d20260207_bp_voting_strategy/`

Best Picture winner selection changed from plurality to instant-runoff voting (IRV)
starting at ceremony 82 (2009). Tests whether encoding this regime change improves
predictions. Spoiler: it doesn't.

## Setup

- New features: `is_irv_era` (bool, ceremony >= 82) and `nominees_in_year` (int)
- CV: Leave-One-Year-Out (26 years: 2000-2025), as-of 2026-02-06
- LR: C=0.5, l1_ratio=1.0 | GBT: n=25, lr=0.1, depth=2
- 8 configs: nonzero-importance and full feature sets, each with/without voting features

### Voting system timeline

| Period | Ceremonies | Winner Selection | Nominees |
|--------|-----------|-----------------|----------|
| Pre-2009 | 1-81 | Plurality | 5 (from #17 onward) |
| 2009-2010 | 82-83 | IRV | Fixed 10 |
| 2011-2020 | 84-93 | IRV | Variable 5-10 |
| 2021+ | 94+ | IRV | Fixed 10 |

## Findings

### Both models completely ignore voting strategy features

| Config | #Feat | Accuracy | Top-3 | Log-Loss | MRR |
|--------|-------|----------|-------|----------|-----|
| **gbt_baseline** | **10** | **73.1%** | **88.5%** | **0.2203** | **0.8157** |
| gbt_with_voting | 12 | 73.1% | 88.5% | 0.2201 | 0.8164 |
| **lr_baseline** | **8** | **69.2%** | **84.6%** | **0.2297** | **0.7956** |
| lr_with_voting | 10 | 69.2% | 84.6% | 0.2297 | 0.7956 |

Both `is_irv_era` and `nominees_in_year` received **zero importance** across all
models. GBT gave both 0.0 importance; L1 zeroed them out completely for LR.

### Why the IRV switch doesn't matter

The same precursor signals (PGA, DGA, SAG, etc.) predict the winner under both
voting systems. The type of film that wins hasn't changed enough to be captured
by a binary era indicator.

`nominees_in_year` is redundant — the model already implicitly accounts for base
rates through the number of rows per ceremony and through features like
`oscar_total_nominations`.

The 7 incorrect GBT years are identical across all configs — voting features don't
help with any of the hard cases.
