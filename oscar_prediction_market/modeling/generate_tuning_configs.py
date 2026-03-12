"""Single source of truth for all hyperparameter grids.

Generates param grid JSON files consumed by nested CV (evaluate_cv.py). Each grid
is a Cartesian product of parameter axes with fixed "infrastructure" params. The
generator functions are authoritative — there are no hand-edited JSON grid files.

**Validation**: Every config dict is validated against its Pydantic schema before
writing. This catches field mismatches (typos, missing fields, extra fields) at
generation time rather than at training time.

**Provenance**: Grid ranges come from prior tuning experiments. Each generator
docstring explains *why* those ranges were chosen and references the experiments.

Usage:
    # Generate all param grids to a directory
    uv run python -m oscar_prediction_market.modeling.generate_tuning_configs \
        --output-dir storage/d20260220_feature_ablation/configs/param_grids

    # Generate for a single model type
    uv run python -m oscar_prediction_market.modeling.generate_tuning_configs \
        --model-type logistic_regression \
        --output-dir storage/d20260220_feature_ablation/configs/param_grids
"""

import argparse
import json
import logging
from collections.abc import Callable
from itertools import product
from pathlib import Path

from pydantic import BaseModel, ValidationError

from oscar_prediction_market.modeling.models import (
    MODEL_INFO,
    TUNABLE_MODEL_TYPES,
    ModelType,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Validation
# ============================================================================


def validate_grid(model_type: ModelType, grid: list[dict]) -> None:
    """Validate every config dict against its Pydantic schema.

    Catches field mismatches (typos, missing fields, extra fields via
    ``model_config = {"extra": "forbid"}``), type errors, and constraint
    violations at generation time.

    Raises:
        ValueError: If any config fails validation, with details.
    """
    info = MODEL_INFO.get(model_type)
    if info is None:
        raise ValueError(
            f"No model info registered for {model_type}. Known types: {list(MODEL_INFO.keys())}"
        )
    config_class: type[BaseModel] = info.config_class

    errors: list[str] = []
    for i, params in enumerate(grid):
        config_dict = {"model_type": model_type.value, **params}
        try:
            config_class.model_validate(config_dict)
        except ValidationError as e:
            errors.append(f"  Config [{i}]: {e}")

    if errors:
        raise ValueError(
            f"Grid validation failed for {model_type.value} "
            f"({len(errors)}/{len(grid)} configs invalid):\n" + "\n".join(errors)
        )


# ============================================================================
# Hyperparameter Grids
# ============================================================================


def get_lr_param_grid() -> list[dict]:
    """Logistic Regression hyperparameter grid.

    Intuition: LR with elastic net regularization. With ~33 features and ~250
    training rows, moderate regularization (C=0.1–0.5) prevents overfitting while
    keeping the model interpretable. L1 (lasso) tends to outperform L2 here because
    many features are correlated precursor awards — L1 selects the most informative
    ones and zeros out the rest.

    Grid axes:
    - C: [0.05, 0.1, 0.2, 0.5, 1.0] — inverse regularization strength.
      Centered around C=0.1 which was optimal in d20260206_hyperparameter_tuning.
      Lower C = stronger regularization. Extreme values (0.001, 10.0) removed
      as they consistently underperformed.
    - l1_ratio: [0.0, 0.5, 1.0] — elastic net mixing.
      0.0 = pure L2 (ridge), 0.5 = balanced, 1.0 = pure L1 (lasso).
      l1_ratio=1.0 was consistently optimal — L1 sparsity helps with correlated
      precursor features.
    - class_weight: ["balanced", None] — imbalance handling.
      ~10% positive rate (1 winner per ~10 nominees). "balanced" upweights the
      minority class. Both settings are competitive depending on category.

    Fixed params:
    - solver: "saga" — required for elastic net. Supports L1, L2, and elastic net.
    - max_iter: 10000 — high enough to always converge. Previously a grid dimension
      (2000 vs 4000) but this is wasteful: if the lower value converges, results
      are identical; if it doesn't, the higher value is strictly better.

    Total: 5 × 3 × 2 = 30 configurations.
    Provenance: d20260206_hyperparameter_tuning, refined in d20260220_feature_ablation.
    """
    grid = []
    for C, l1_ratio, class_weight in product(
        [0.05, 0.1, 0.2, 0.5, 1.0],
        [0.0, 0.5, 1.0],
        ["balanced", None],
    ):
        grid.append(
            {
                "C": C,
                "l1_ratio": l1_ratio,
                "solver": "saga",
                "max_iter": 10000,
                "class_weight": class_weight,
            }
        )
    return grid


def get_gbt_param_grid() -> list[dict]:
    """Gradient Boosting (sklearn) hyperparameter grid.

    Intuition: With ~250 training rows and stochastic subsampling, GBT needs to be
    kept simple — shallow trees (depth 1–3) and few boosting rounds (25–100). Over-
    fitting is the main risk, so the grid is deliberately biased toward simpler models.

    Grid axes:
    - n_estimators: [25, 50, 75, 100] — number of boosting stages.
      50 was optimal in d20260206_hyperparameter_tuning (at the minimum of the old
      grid [50, 100, 200, 300]). Extended lower to check if even fewer trees suffice.
    - max_depth: [1, 2, 3] — tree depth. Controls interaction order.
      depth=1 = stumps (main effects only), depth=2 = pairwise interactions.
      depth=2 was optimal; extended to depth=1 to test pure additive model.
    - learning_rate: [0.025, 0.05, 0.1] — shrinkage per stage.
      Lower rate + more trees = smoother fit. lr=0.05 was optimal; added 0.025
      to explore slower learning.

    Fixed params:
    - min_samples_split: 5, min_samples_leaf: 2 — prevent tiny splits with so
      few rows. These were fixed from the start and never tuned individually.
    - subsample: 0.8 — stochastic GBT reduces variance, standard practice.
    - random_state: 42 — reproducibility.

    Total: 4 × 3 × 3 = 36 configurations.
    Provenance: d20260206_hyperparameter_tuning (grid shift toward simpler models).
    """
    grid = []
    for n_est, depth, lr in product(
        [25, 50, 75, 100],
        [1, 2, 3],
        [0.025, 0.05, 0.1],
    ):
        grid.append(
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "subsample": 0.8,
                "random_state": 42,
            }
        )
    return grid


def get_xgb_param_grid() -> list[dict]:
    """XGBoost hyperparameter grid.

    Intuition: XGBoost differs from sklearn's GBT primarily in its built-in L1/L2
    regularization (reg_alpha, reg_lambda) and column subsampling per tree. This
    allows slightly deeper trees (depth 4–6) without as much overfitting risk.

    Grid axes:
    - n_estimators: [25, 50, 75, 100] — same range as GBT for comparability.
    - max_depth: [2, 3, 4, 6] — slightly deeper than GBT due to regularization.
      Includes depth=6 (XGBoost default) to test if regularization compensates.
    - learning_rate: [0.01, 0.05, 0.1] — includes 0.01 because XGBoost benefits
      from lower learning rates paired with more trees.
    - colsample_bytree: [0.8, 1.0] — random feature subsampling per tree.
      0.8 adds decorrelation between trees (like random forest effect).

    Fixed params:
    - min_child_weight: 2 — analogous to min_samples_leaf in GBT.
    - subsample: 0.8 — row subsampling (stochastic boosting).
    - reg_alpha: 0.0 — no L1 regularization on leaf weights.
    - reg_lambda: 1.0 — L2 regularization on leaf weights (XGBoost default).
    - random_state: 42 — reproducibility.

    Total: 4 × 4 × 3 × 2 = 96 configurations.
    Note: Larger grid than GBT due to the extra colsample axis. Could prune
    if compute budget is tight.
    """
    grid = []
    for n_est, depth, lr, colsample in product(
        [25, 50, 75, 100],
        [2, 3, 4, 6],
        [0.01, 0.05, 0.1],
        [0.8, 1.0],
    ):
        grid.append(
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "min_child_weight": 2,
                "subsample": 0.8,
                "colsample_bytree": colsample,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "random_state": 42,
            }
        )
    return grid


def get_conditional_logit_param_grid() -> list[dict]:
    """Conditional Logit (McFadden's choice model) hyperparameter grid.

    Intuition: Conditional logit models P(win | ceremony) directly as a softmax
    over nominee utilities. Unlike binary LR, probabilities sum to 1 within each
    ceremony by construction — no post-hoc normalization needed. Uses statsmodels'
    elastic net regularization (fit_regularized).

    The penalty strength matters more than in binary LR because the effective
    sample size is ~25 ceremony-level observations (not ~250 nominee rows), so
    overfit risk is high.

    Grid axes:
    - alpha: [0.001, 0.002, 0.005, 0.01, 0.05, 0.1] — overall regularization strength.
      Wide log-scale range. Low values (0.001–0.01) target the small-data regime.
      alpha=0.005 was at the grid boundary in d20260220 Round 1 for all 9 categories,
      0.002 added in Round 2 and selected by several categories. Shifted again to add
      0.001 and remove 0.5 (never selected) to probe the low-regularization frontier.
    - L1_wt: [0.0, 0.25, 0.5, 0.75, 1.0] — elastic net mixing.
      0.0 = pure L2 (ridge), 1.0 = pure L1 (lasso). Finer grid than LR because
      the model is more sensitive to the sparsity pattern with fewer effective samples.

    Total: 6 × 5 = 30 configurations.
    Provenance: d20260217_multinomial_modeling, refined in d20260220_feature_ablation.
    """
    grid = []
    for alpha, l1_wt in product(
        [0.001, 0.002, 0.005, 0.01, 0.05, 0.1],
        [0.0, 0.25, 0.5, 0.75, 1.0],
    ):
        grid.append(
            {
                "alpha": alpha,
                "L1_wt": l1_wt,
            }
        )
    return grid


def get_softmax_gbt_param_grid() -> list[dict]:
    """Softmax GBT (multi-class XGBoost) hyperparameter grid.

    Intuition: Two-stage approach — a binary model ranks nominees, then the top-K
    enter a multi-class XGBoost with softmax objective (multi:softprob). The feature
    space is K × F (one feature set per slot), creating a very wide matrix (~K×33
    features for ~25 training years). Strong regularization and simple trees are
    essential to avoid overfitting this wide-and-short dataset.

    Grid axes:
    - n_estimators: [25, 50, 75, 100] — boosting rounds. Same range as binary GBT.
    - max_depth: [1, 2, 3] — shallow trees are critical with K×F features.
    - learning_rate: [0.025, 0.05, 0.1] — lower rates help with multi-class.

    Fixed params (heavy regularization for the wide feature space):
    - min_child_weight: 1 — XGBoost default; not a main regularization lever here.
    - subsample: 0.8 — stochastic boosting.
    - colsample_bytree: 0.8 — column subsampling per tree helps significantly
      when features are K copies of the same base features.
    - reg_alpha: 0.1 — L1 regularization on leaf weights. Higher than binary XGBoost
      due to the wider feature space.
    - reg_lambda: 1.0 — L2 regularization (XGBoost default).
    - top_k: 10 — number of nominees per ceremony. Determines the number of classes
      and the feature matrix width. 10 captures nearly all eventual winners.
    - random_state: 42 — reproducibility.

    Total: 4 × 3 × 3 = 36 configurations.
    Provenance: d20260217_multinomial_modeling.
    """
    grid = []
    for n_est, depth, lr in product(
        [25, 50, 75, 100],
        [1, 2, 3],
        [0.025, 0.05, 0.1],
    ):
        grid.append(
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "top_k": 10,
                "random_state": 42,
            }
        )
    return grid


def get_calibrated_softmax_gbt_param_grid() -> list[dict]:
    """Calibrated Softmax GBT hyperparameter grid.

    Intuition: Wraps a binary GBT with temperature-scaled softmax post-processing.
    Binary GBT outputs are converted to log-odds, then softmax with temperature T
    gives calibrated within-ceremony probabilities:

        p_hat_i = exp(logit(p_i) / T) / Σ_j exp(logit(p_j) / T)

    Temperature controls sharpness: T < 1 is sharper (more confident on
    frontrunners), T > 1 is smoother (more uniform). T = 1 is equivalent to
    raw logit → softmax (no rescaling). The binary GBT base params are a subset
    of the full GBT grid — intentionally smaller because the temperature axis
    adds a multiplicative factor.

    Grid axes:
    - n_estimators: [25, 50] — fewer options than full GBT to keep grid manageable.
    - max_depth: [1, 2] — shallow trees; deeper ones overfit at this sample size.
    - learning_rate: [0.025, 0.05, 0.1] — same range as GBT.
    - temperature: [0.5, 0.75, 1.0, 1.25, 1.5] — the calibration lever.
      0.5 = very sharp (strong frontrunner signal), 1.5 = smooth (hedged bets).
      Range informed by d20260217_multinomial_modeling.

    Fixed params:
    - min_samples_split: 5, min_samples_leaf: 2 — same as sklearn GBT.
    - subsample: 0.8 — stochastic boosting.
    - random_state: 42 — reproducibility.

    Total: 2 × 2 × 3 × 5 = 60 configurations.
    Note: The original hand-crafted JSON had 50 entries due to 2 accidentally
    missing (n_est, depth, lr) combos. Clean Cartesian product gives 60.
    Provenance: d20260217_multinomial_modeling.
    """
    grid = []
    for n_est, depth, lr, temp in product(
        [25, 50],
        [1, 2],
        [0.025, 0.05, 0.1],
        [0.5, 0.75, 1.0, 1.25, 1.5],
    ):
        grid.append(
            {
                "n_estimators": n_est,
                "max_depth": depth,
                "learning_rate": lr,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "subsample": 0.8,
                "temperature": temp,
                "random_state": 42,
            }
        )
    return grid


# ============================================================================
# Grid registry — maps ModelType → generator function
# ============================================================================

_GRID_GENERATORS: dict[ModelType, Callable[[], list[dict]]] = {
    ModelType.LOGISTIC_REGRESSION: get_lr_param_grid,
    ModelType.GRADIENT_BOOSTING: get_gbt_param_grid,
    ModelType.XGBOOST: get_xgb_param_grid,
    ModelType.CONDITIONAL_LOGIT: get_conditional_logit_param_grid,
    ModelType.SOFTMAX_GBT: get_softmax_gbt_param_grid,
    ModelType.CALIBRATED_SOFTMAX_GBT: get_calibrated_softmax_gbt_param_grid,
}


def get_param_grid(model_type: ModelType) -> list[dict]:
    """Get the hyperparameter grid for a model type.

    This is the main entry point for getting grids programmatically.

    Args:
        model_type: One of the tunable model types.

    Returns:
        List of parameter dicts (without ``model_type`` key — callers add it).

    Raises:
        ValueError: If model_type has no registered grid generator.
    """
    generator = _GRID_GENERATORS.get(model_type)
    if generator is None:
        raise ValueError(
            f"No param grid generator for {model_type}. "
            f"Known types: {list(_GRID_GENERATORS.keys())}"
        )
    return generator()


# ============================================================================
# Config naming
# ============================================================================


def config_to_name(model_type: ModelType, params: dict) -> str:
    """Generate a descriptive filename stem from config parameters."""
    if model_type == ModelType.LOGISTIC_REGRESSION:
        cw = "bal" if params.get("class_weight") == "balanced" else "none"
        return f"lr_C{params['C']}_l1r{params['l1_ratio']}_cw{cw}"
    elif model_type == ModelType.XGBOOST:
        return (
            f"xgb_n{params['n_estimators']}_d{params['max_depth']}"
            f"_lr{params['learning_rate']}_cs{params['colsample_bytree']}"
        )
    elif model_type == ModelType.CONDITIONAL_LOGIT:
        return f"clogit_a{params['alpha']}_l1w{params['L1_wt']}"
    elif model_type == ModelType.SOFTMAX_GBT:
        return f"smgbt_n{params['n_estimators']}_d{params['max_depth']}_lr{params['learning_rate']}"
    elif model_type == ModelType.CALIBRATED_SOFTMAX_GBT:
        return (
            f"csmgbt_n{params['n_estimators']}_d{params['max_depth']}"
            f"_lr{params['learning_rate']}_t{params['temperature']}"
        )
    else:
        # GBT (default)
        return (
            f"gbt_n{params['n_estimators']}_d{params['max_depth']}"
            f"_lr{params['learning_rate']}_leaf{params['min_samples_leaf']}"
        )


# ============================================================================
# File generation
# ============================================================================


def generate_individual_configs(
    model_type: ModelType,
    output_dir: Path,
) -> list[Path]:
    """Generate individual model config JSON files.

    Each file is a complete config dict that can be passed to --model-config.

    Args:
        model_type: Model type to generate configs for.
        output_dir: Directory to write config files.

    Returns:
        List of paths to created config files.
    """
    grid = get_param_grid(model_type)
    validate_grid(model_type, grid)

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    for params in grid:
        config_name = config_to_name(model_type, params)
        config_dict = {"model_type": model_type.value, **params}

        output_path = output_dir / f"{config_name}.json"
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        created_files.append(output_path)
        logger.info(f"Created: {output_path}")

    return created_files


def generate_param_grid_file(
    model_type: ModelType,
    output_dir: Path,
) -> Path:
    """Generate a param grid JSON file for nested CV.

    Output format::

        {
            "model_type": "logistic_regression",
            "grid": [
                {"C": 0.05, "l1_ratio": 0.0, ...},
                {"C": 0.1, "l1_ratio": 0.0, ...},
                ...
            ]
        }

    Args:
        model_type: Model type to generate grid for.
        output_dir: Directory to write the grid file.

    Returns:
        Path to the created file.
    """
    grid = get_param_grid(model_type)
    validate_grid(model_type, grid)

    short_name = model_type.short_name

    output_dir.mkdir(parents=True, exist_ok=True)
    param_grid_dict = {
        "model_type": model_type.value,
        "grid": grid,
    }

    output_path = output_dir / f"{short_name}_grid.json"
    with open(output_path, "w") as f:
        json.dump(param_grid_dict, f, indent=2)

    logger.info(f"Created param grid: {output_path} ({len(grid)} configs)")
    return output_path


def generate_all_param_grids(output_dir: Path) -> dict[ModelType, Path]:
    """Generate param grid files for all tunable model types.

    Args:
        output_dir: Directory to write all grid files.

    Returns:
        Mapping of model type to created file path.
    """
    results = {}
    for model_type in TUNABLE_MODEL_TYPES:
        path = generate_param_grid_file(model_type, output_dir)
        results[model_type] = path
    return results


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate hyperparameter grid JSON files for nested CV tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Generate all grids:\n"
            "  %(prog)s --output-dir storage/experiment/configs/param_grids\n\n"
            "  # Generate one model type:\n"
            "  %(prog)s --model-type logistic_regression --output-dir configs/param_grids\n"
        ),
    )
    parser.add_argument(
        "--model-type",
        type=ModelType,
        choices=TUNABLE_MODEL_TYPES,
        default=None,
        help="Model type to generate grid for (default: all tunable types)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for param grid JSON files",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    output_dir = Path(args.output_dir)

    if args.model_type is not None:
        # Single model type
        model_types = [args.model_type]
    else:
        # All tunable types
        model_types = TUNABLE_MODEL_TYPES

    logger.info(f"Generating param grids -> {output_dir}/")

    total_configs = 0
    for model_type in model_types:
        path = generate_param_grid_file(model_type, output_dir)
        grid = get_param_grid(model_type)
        total_configs += len(grid)
        logger.info(f"  {model_type.value}: {len(grid)} configs -> {path.name}")

    logger.info(f"\nTotal: {len(model_types)} grids, {total_configs} configurations")


if __name__ == "__main__":
    main()
