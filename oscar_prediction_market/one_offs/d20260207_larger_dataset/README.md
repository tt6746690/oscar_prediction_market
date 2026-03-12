# Larger Dataset — Does More Training Data Help?

**Storage:** `storage/d20260207_larger_dataset/`

The [CV splitting comparison](../d20260207_cv_splitting/) showed performance increases
monotonically with data size. This experiment tests whether expanding the training data
beyond 2000-2025 (26 years) to 1980-2026 (47 years, 306 films) improves the model.

## Setup

- Dataset rebuilt to cover 1980-2026 (47 years, 306 films)
- LOYO CV on 2000-2025
- Same hyperparameters and feature sets as the CV splitting experiment

## Findings

### More data does not help — older data adds noise

Extending the dataset to 1980 did not improve LOYO accuracy on the 2000-2025 test
period. The precursor award landscape has changed substantially since the 1980s
(some awards didn't exist, ceremony formats changed), so pre-2000 data introduces
distributional shift rather than additional signal.

The model trained on 2000-2025 already captures the modern award-season dynamics.
Adding older data dilutes these patterns with outdated correlations.

### Recommendation

Stick with 2000-2025 (26 years) as the training set. The small-data regime is a
fundamental constraint — the solution is better features and regularization rather
than older, noisier data.
