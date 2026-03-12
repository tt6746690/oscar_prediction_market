"""Generate ablation grid of backtest configs.

Creates one JSON config file per parameter combination. Includes:

1. **Main grid** (1728 configs): full sweep over model_type, kelly_fraction,
   min_edge, sell_edge_threshold, min_price, fee_type, kelly_mode.

2. **α-blend ablation** (10 configs): sweep market_blend_alpha ∈
   {0.0, 0.15, 0.30, 0.50, 0.85} for LR and GBT, with all other
   params fixed at best-known values.

3. **Probability normalization ablation** (4 configs): test whether
   normalizing model probabilities to sum to 1 improves performance,
   for LR and GBT × {True, False}, with best-known params.

Total: 1728 + 10 + 4 = 1742 configs (14 overlap with main grid → 1742 unique files
but the dedup depends on ID convention; the actual file count includes overlaps
since α-blend/normalize configs have distinct IDs).

Usage:
    cd "$(git rev-parse --show-toplevel)"
    uv run python -m oscar_prediction_market.one_offs.\
d20260214_trade_signal_ablation.generate_configs \
        --output-dir storage/d20260214_trade_signal_ablation/configs
"""

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Grid definition
# ============================================================================

GRID: dict[str, list[Any]] = {
    "model_type": ["lr", "gbt", "avg", "market_blend"],
    "kelly_fraction": [0.10, 0.15, 0.25],
    "min_edge": [0.05, 0.08, 0.10, 0.15],
    "sell_edge_threshold": [-0.03, -0.05, -0.10],
    "min_price": [0.0, 0.10, 0.20],
    "fee_type": ["taker", "maker"],
    "kelly_mode": ["independent", "multi_outcome"],
}

# Fixed params (not ablated)
FIXED = {
    "bankroll_dollars": 1000,
    "max_position_per_outcome_dollars": 250,
    "max_total_exposure_dollars": 500,
    "spread_penalty_mode": "trade_data",
    "fixed_spread_penalty": 0.02,
    "price_start_date": "2025-12-01",
    "price_end_date": "2026-02-15",
    "bankroll_mode": "dynamic",
}


def generate_configs(output_dir: Path) -> int:
    """Generate all config permutations and write to JSON files.

    Returns the number of configs generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    # ---- Experiment 1: Main grid (864 configs) ----
    keys = list(GRID.keys())
    values = list(GRID.values())

    for combo in itertools.product(*values):
        params: dict[str, Any] = dict(zip(keys, combo, strict=True))

        # Build config dict
        config = {**FIXED}
        config["model_types"] = [params["model_type"]]
        config["kelly_fraction"] = params["kelly_fraction"]
        config["min_edge"] = params["min_edge"]
        config["sell_edge_threshold"] = params["sell_edge_threshold"]
        config["min_price"] = params["min_price"]
        config["fee_type"] = params["fee_type"]
        config["kelly_mode"] = params["kelly_mode"]

        # Generate descriptive config ID
        config_id = (
            f"{params['model_type']}"
            f"_kelly{params['kelly_fraction']:.2f}"
            f"_edge{params['min_edge']:.2f}"
            f"_sell{abs(params['sell_edge_threshold']):.2f}"
            f"_floor{params['min_price']:.2f}"
            f"_{params['fee_type']}"
            f"_{params['kelly_mode']}"
        )

        config_path = output_dir / f"{config_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        count += 1

    # ---- Experiment 2: α-blend ablation (12 configs) ----
    # Best-known fixed params from experiment 1
    best_fixed = {
        **FIXED,
        "kelly_fraction": 0.10,
        "min_edge": 0.05,
        "sell_edge_threshold": -0.03,
        "min_price": 0.0,
        "fee_type": "maker",
    }
    alpha_values = [0.0, 0.15, 0.30, 0.50, 0.85]
    # Skip α=1.0 (pure market → zero edge → zero trades)

    for base_model in ["lr", "gbt"]:
        for alpha in alpha_values:
            config = {**best_fixed}
            config["model_types"] = [base_model]
            config["market_blend_alpha"] = alpha

            config_id = f"{base_model}_alpha{alpha:.2f}_kelly0.10_edge0.05_sell0.03_floor0_maker"
            config_path = output_dir / f"{config_id}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            count += 1

    # ---- Experiment 3: Probability normalization ablation (4 configs) ----
    for base_model in ["lr", "gbt"]:
        for normalize in [True, False]:
            config = {**best_fixed}
            config["model_types"] = [base_model]
            config["normalize_probabilities"] = normalize

            norm_str = "norm" if normalize else "raw"
            config_id = f"{base_model}_{norm_str}_kelly0.10_edge0.05_sell0.03_floor0_maker"
            config_path = output_dir / f"{config_id}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            count += 1

    logger.info("Generated %d configs in %s", count, output_dir)
    return count


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Generate ablation config grid")
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    n = generate_configs(Path(args.output_dir))

    # Verify count matches expected
    main_grid = 1
    for v in GRID.values():
        main_grid *= len(v)
    alpha_configs = 2 * 5  # 2 models × 5 α values (skip α=1.0)
    norm_configs = 2 * 2  # 2 models × {True, False}
    expected = main_grid + alpha_configs + norm_configs
    assert n == expected, f"Generated {n} configs, expected {expected}"
    print(
        f"Generated {n} configs (main grid: {main_grid}, α-blend: {alpha_configs}, normalize: {norm_configs})"
    )


if __name__ == "__main__":
    main()
