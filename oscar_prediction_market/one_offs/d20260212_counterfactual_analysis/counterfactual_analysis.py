"""Counterfactual analysis for feature events.

Generalizable framework for answering: "When a new binary feature becomes available,
how does each possible outcome change model win probabilities, and where are the
trading edges vs the market?"

The DGA winner analysis is the first concrete instance:
- Baseline: model trained with as_of_date=2026-02-06 (no DGA winner feature)
- Scenario: model trained with as_of_date=2026-02-07 (DGA winner feature available)
- Counterfactuals: for each DGA nominee, set dga_winner=True for that nominee only
  and predict. Derived features (precursor_wins_count, has_pga_dga_combo) are
  recomputed automatically because we modify raw data and re-run feature engineering.

Usage:
    uv run python -m oscar_prediction_market.one_offs.d20260212_counterfactual_analysis.counterfactual_analysis \
        --output-dir storage/d20260212_counterfactual_analysis/dga_winner
"""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from oscar_prediction_market.data.schema import (
    NominationDataset,
    NominationRecord,
    OscarCategory,
    PrecursorKey,
)
from oscar_prediction_market.modeling.data_loader import (
    load_raw_dataset,
    prepare_model_data,
)
from oscar_prediction_market.modeling.feature_engineering import (
    FEATURE_REGISTRY,
    FeatureSet,
    transform_dataset,
)
from oscar_prediction_market.modeling.models import (
    ModelType,
    PredictionModel,
    load_model,
)
from oscar_prediction_market.trading.market_data import OSCAR_MARKETS
from oscar_prediction_market.trading.oscar_market import OscarMarket

logger = logging.getLogger(__name__)

TARGET_CEREMONY = 98  # 2026 Oscars


# ============================================================================
# Feature Event abstraction
# ============================================================================


class FeatureEvent(BaseModel):
    """A feature event: a moment when a binary feature becomes available.

    Defines what to modify in the raw data for counterfactual analysis.
    """

    model_config = {"extra": "forbid"}

    name: str
    description: str
    pre_event_date: date  # as_of_date for baseline model (feature not yet available)
    post_event_date: date  # as_of_date for scenario model (feature available)
    eligible_nominees: list[str]  # Film titles that could have feature=True
    field_name: str  # Flat field name for display (e.g. "dga_directing_winner")
    precursor_key: PrecursorKey  # Key in NominationRecord.precursors dict
    field_type: str = "winner"  # "winner" or "nominee"
    market_price_date: date  # Date to fetch Kalshi prices for comparison


class ScenarioPrediction(BaseModel):
    """Prediction for a single nominee under a specific scenario."""

    model_config = {"extra": "forbid"}

    title: str
    probability: float
    rank: int


class CounterfactualResult(BaseModel):
    """Result of a single counterfactual scenario for one model."""

    model_config = {"extra": "forbid"}

    model_name: str
    scenario_nominee: str  # Which nominee gets the feature=True
    predictions: list[ScenarioPrediction]  # All nominees' predictions under this scenario


class CounterfactualSummary(BaseModel):
    """Full summary for a feature event: baselines + all counterfactuals + market prices."""

    model_config = {"extra": "forbid"}

    event: FeatureEvent
    baselines: dict[str, list[ScenarioPrediction]]  # model_name -> predictions
    counterfactuals: list[CounterfactualResult]  # All model × scenario combinations
    market_prices: dict[str, float]  # nominee title -> market price (0-1 scale)


# ============================================================================
# DGA Winner Feature Event
# ============================================================================

DGA_NOMINEES_2026 = [
    "Frankenstein",
    "Hamnet",
    "Marty Supreme",
    "One Battle after Another",
    "Sinners",
]

DGA_WINNER_EVENT = FeatureEvent(
    name="dga_winner",
    description="DGA Outstanding Director winner announcement",
    pre_event_date=date(2026, 2, 6),
    post_event_date=date(2026, 2, 7),
    eligible_nominees=DGA_NOMINEES_2026,
    field_name="dga_directing_winner",
    precursor_key=PrecursorKey.DGA_DIRECTING,
    field_type="winner",
    market_price_date=date(2026, 2, 6),  # Pre-DGA market prices
)


# ============================================================================
# Raw data modification
# ============================================================================


def create_counterfactual_dataset(
    raw_dataset: NominationDataset,
    event: FeatureEvent,
    winner_title: str,
    ceremony: int = TARGET_CEREMONY,
) -> NominationDataset:
    """Create a modified copy of the raw dataset for a counterfactual scenario.

    Sets the nested award result field to True for the winner_title and False for
    all other nominees in the target ceremony. Does NOT modify records from other
    ceremonies.

    Args:
        raw_dataset: Original raw dataset
        event: Feature event definition
        winner_title: Title of the nominee who "wins" in this scenario
        ceremony: Target ceremony number

    Returns:
        Deep copy of dataset with modified records
    """
    modified_records: list[NominationRecord] = []
    for record in raw_dataset.records:
        if record.ceremony == ceremony:
            # Deep copy to avoid mutating original
            modified = record.model_copy(deep=True)
            # Ensure the precursor key exists in the dict
            if event.precursor_key not in modified.precursors:
                from oscar_prediction_market.data.schema import AwardResult

                modified.precursors[event.precursor_key] = AwardResult()
            award_result = modified.precursors[event.precursor_key]
            if record.film.title == winner_title:
                setattr(award_result, event.field_type, True)
            else:
                setattr(award_result, event.field_type, False)
            modified_records.append(modified)
        else:
            modified_records.append(record)

    return NominationDataset(
        category=raw_dataset.category,
        year_start=raw_dataset.year_start,
        year_end=raw_dataset.year_end,
        record_count=raw_dataset.record_count,
        records=modified_records,
    )


# ============================================================================
# Model loading and prediction
# ============================================================================

MODEL_CONFIGS = [
    ("lr", ModelType.LOGISTIC_REGRESSION),
    ("gbt", ModelType.GRADIENT_BOOSTING),
    ("xgb", ModelType.XGBOOST),
]


def load_trained_model(output_dir: Path, model_prefix: str, role: str) -> PredictionModel:
    """Load a trained model from build_model output.

    Args:
        output_dir: Base output directory (e.g., storage/.../dga_winner/)
        model_prefix: Model prefix (e.g., "lr", "gbt", "xgb")
        role: "baseline" or "scenario"

    Returns:
        Loaded trained model
    """
    model_dir = output_dir / f"{model_prefix}_{role}" / "5_final_predict"
    if not model_dir.exists():
        # Try without feature selection
        model_dir = output_dir / f"{model_prefix}_{role}" / "2_final_predict"
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info(f"Loading model: {model_path}")
    return load_model(model_path)


def load_selected_features(output_dir: Path, model_prefix: str, role: str) -> FeatureSet:
    """Load selected feature set from build_model output.

    Args:
        output_dir: Base output directory
        model_prefix: Model prefix (e.g., "lr", "gbt", "xgb")
        role: "baseline" or "scenario"

    Returns:
        FeatureSet with selected features
    """
    features_path = output_dir / f"{model_prefix}_{role}" / "3_selected_features.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Selected features not found: {features_path}")
    with open(features_path) as f:
        data = json.load(f)
    return FeatureSet(**data)


def predict_ceremony(
    model: PredictionModel,
    raw_dataset: NominationDataset,
    feature_set: FeatureSet,
    model_type: ModelType,
    as_of_date: date,
    ceremony: int = TARGET_CEREMONY,
) -> list[ScenarioPrediction]:
    """Run prediction for a single ceremony using a trained model.

    Runs feature engineering on the (possibly modified) raw dataset,
    then predicts with the trained model.

    Args:
        model: Trained model
        raw_dataset: Raw dataset (possibly modified for counterfactual)
        feature_set: Feature set to use
        model_type: Model type for feature engineering
        as_of_date: Date for feature availability filtering
        ceremony: Target ceremony

    Returns:
        List of ScenarioPrediction sorted by probability descending
    """
    all_features = list(FEATURE_REGISTRY.values())
    df = transform_dataset(raw_dataset, all_features, as_of_date)

    # Filter to target ceremony
    test_df = df[df["ceremony"] == ceremony].copy()
    if len(test_df) == 0:
        raise ValueError(f"No records for ceremony {ceremony}")

    X_test, _, metadata = prepare_model_data(test_df, feature_set)
    probabilities = model.predict_proba(X_test)

    # Build predictions
    preds = []
    for title, prob in zip(metadata["title"].tolist(), probabilities.tolist(), strict=True):
        preds.append(ScenarioPrediction(title=title, probability=prob, rank=0))

    # Assign ranks (1 = highest probability)
    preds.sort(key=lambda p: p.probability, reverse=True)
    for i, pred in enumerate(preds):
        pred.rank = i + 1

    return preds


# ============================================================================
# Kalshi market prices
# ============================================================================


def fetch_market_prices(price_date: date) -> dict[str, float]:
    """Fetch Kalshi closing prices for all nominees on a given date.

    Returns:
        Dict of nominee title -> price in [0, 1] scale
    """
    logger.info(f"Fetching Kalshi prices for {price_date}")
    bp_data = OSCAR_MARKETS.get_category_data(OscarCategory.BEST_PICTURE, 2026)
    mkt = OscarMarket(event_ticker=bp_data.event_ticker, nominee_tickers=bp_data.nominee_tickers)
    candles = mkt.get_daily_prices(
        start_date=price_date,
        end_date=price_date,
    )

    prices: dict[str, float] = {}
    if not candles:
        logger.warning(f"No price data for {price_date}")
        return prices

    for c in candles:
        # Kalshi nominee names may differ from dataset titles
        prices[c.nominee] = c.close

    return prices


# ============================================================================
# Run analysis
# ============================================================================


def run_counterfactual_analysis(
    event: FeatureEvent,
    output_dir: Path,
    raw_path: Path,
) -> CounterfactualSummary:
    """Run full counterfactual analysis for a feature event.

    1. Load baseline and scenario models
    2. Get baseline predictions (pre-event model)
    3. For each eligible nominee: create counterfactual dataset → predict
    4. Fetch market prices

    Args:
        event: Feature event definition
        output_dir: Directory with trained baseline/scenario models + output destination
        raw_path: Path to raw dataset JSON.

    Returns:
        CounterfactualSummary with all results
    """
    if raw_path is None:
        raise ValueError("raw_path is required")
    raw_dataset = load_raw_dataset(raw_path)

    baselines: dict[str, list[ScenarioPrediction]] = {}
    counterfactuals: list[CounterfactualResult] = []

    for model_prefix, model_type in MODEL_CONFIGS:
        # --- Baseline predictions (pre-event model) ---
        baseline_model = load_trained_model(output_dir, model_prefix, "baseline")
        baseline_features = load_selected_features(output_dir, model_prefix, "baseline")

        baseline_preds = predict_ceremony(
            model=baseline_model,
            raw_dataset=raw_dataset,
            feature_set=baseline_features,
            model_type=model_type,
            as_of_date=event.pre_event_date,
        )
        baselines[model_prefix] = baseline_preds
        logger.info(
            f"Baseline {model_prefix}: top={baseline_preds[0].title} "
            f"({baseline_preds[0].probability:.1%})"
        )

        # --- Scenario predictions (post-event model with counterfactuals) ---
        scenario_model = load_trained_model(output_dir, model_prefix, "scenario")
        scenario_features = load_selected_features(output_dir, model_prefix, "scenario")

        for winner_title in event.eligible_nominees:
            cf_dataset = create_counterfactual_dataset(raw_dataset, event, winner_title)
            cf_preds = predict_ceremony(
                model=scenario_model,
                raw_dataset=cf_dataset,
                feature_set=scenario_features,
                model_type=model_type,
                as_of_date=event.post_event_date,
            )
            counterfactuals.append(
                CounterfactualResult(
                    model_name=model_prefix,
                    scenario_nominee=winner_title,
                    predictions=cf_preds,
                )
            )
            # Log the counterfactual winner's probability
            winner_pred = next(p for p in cf_preds if p.title == winner_title)
            logger.info(
                f"  {model_prefix} if {winner_title} wins: "
                f"P({winner_title})={winner_pred.probability:.1%}"
            )

    # --- Fetch market prices ---
    market_prices = fetch_market_prices(event.market_price_date)

    return CounterfactualSummary(
        event=event,
        baselines=baselines,
        counterfactuals=counterfactuals,
        market_prices=market_prices,
    )


# ============================================================================
# Output formatting
# ============================================================================


def get_pred_for_nominee(preds: list[ScenarioPrediction], title: str) -> ScenarioPrediction | None:
    """Find prediction for a specific nominee."""
    for p in preds:
        if p.title == title:
            return p
    return None


def format_summary_tables(summary: CounterfactualSummary) -> str:
    """Format summary as human-readable text tables."""
    lines: list[str] = []
    event = summary.event
    model_names = sorted(summary.baselines.keys())

    lines.append(f"{'=' * 90}")
    lines.append(f"Counterfactual Analysis: {event.name}")
    lines.append(f"{'=' * 90}")
    lines.append(f"Pre-event date:  {event.pre_event_date}")
    lines.append(f"Post-event date: {event.post_event_date}")
    lines.append(f"Market price date: {event.market_price_date}")
    lines.append(f"Eligible nominees: {', '.join(event.eligible_nominees)}")
    lines.append("")

    # --- Baseline predictions ---
    lines.append(f"{'=' * 90}")
    lines.append("BASELINE PREDICTIONS (pre-event model, no DGA winner feature)")
    lines.append(f"{'=' * 90}")
    lines.append("")

    # Get all nominees from first model's baseline
    all_nominees = [
        p.title for p in sorted(summary.baselines[model_names[0]], key=lambda p: p.rank)
    ]

    header = f"{'Nominee':<30}"
    for m in model_names:
        header += f" {m.upper():>10}"
    header += f" {'Kalshi':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for title in all_nominees:
        row = f"{title:<30}"
        for m in model_names:
            pred = get_pred_for_nominee(summary.baselines[m], title)
            row += f" {pred.probability:>9.1%}" if pred else f" {'?':>10}"
        kalshi_name = title
        market_p = summary.market_prices.get(kalshi_name)
        row += f" {market_p:>9.1%}" if market_p is not None else f" {'?':>10}"
        lines.append(row)

    lines.append("")

    # --- Counterfactual scenario predictions ---
    lines.append(f"{'=' * 90}")
    lines.append("COUNTERFACTUAL SCENARIOS")
    lines.append(f"{'=' * 90}")

    for winner_title in event.eligible_nominees:
        lines.append("")
        lines.append(f"--- If {winner_title} wins {event.name.upper()} ---")
        lines.append("")

        header = f"{'Nominee':<30}"
        for m in model_names:
            header += f" {m.upper():>10}"
        lines.append(header)
        lines.append("-" * len(header))

        for title in all_nominees:
            row = f"{title:<30}"
            for m in model_names:
                cf = next(
                    (
                        c
                        for c in summary.counterfactuals
                        if c.model_name == m and c.scenario_nominee == winner_title
                    ),
                    None,
                )
                if cf:
                    pred = get_pred_for_nominee(cf.predictions, title)
                    row += f" {pred.probability:>9.1%}" if pred else f" {'?':>10}"
                else:
                    row += f" {'?':>10}"
            lines.append(row)

    lines.append("")

    # --- Probability shift table ---
    lines.append(f"{'=' * 90}")
    lines.append("PROBABILITY SHIFT: Baseline → Scenario (for the DGA winner nominee)")
    lines.append(f"{'=' * 90}")
    lines.append("")

    header = f"{'DGA Winner':<25}"
    for m in model_names:
        header += f" {m.upper() + ' base':>12} {m.upper() + ' scen':>12} {'delta':>8}"
    header += f" {'Mean scen':>10} {'Kalshi':>8} {'Edge':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for winner_title in event.eligible_nominees:
        row = f"{winner_title:<25}"
        scenario_probs = []
        for m in model_names:
            base_pred = get_pred_for_nominee(summary.baselines[m], winner_title)
            cf = next(
                (
                    c
                    for c in summary.counterfactuals
                    if c.model_name == m and c.scenario_nominee == winner_title
                ),
                None,
            )
            scen_pred = get_pred_for_nominee(cf.predictions, winner_title) if cf else None

            base_p = base_pred.probability if base_pred else 0
            scen_p = scen_pred.probability if scen_pred else 0
            delta = scen_p - base_p
            scenario_probs.append(scen_p)

            row += f" {base_p:>11.1%} {scen_p:>11.1%} {delta:>+7.1%}"

        mean_scen = sum(scenario_probs) / len(scenario_probs) if scenario_probs else 0
        kalshi_name = winner_title
        market_p = summary.market_prices.get(kalshi_name, 0)
        edge = mean_scen - market_p

        row += f" {mean_scen:>9.1%} {market_p:>7.1%} {edge:>+7.1%}"
        lines.append(row)

    lines.append("")

    # --- Impact on other nominees ---
    lines.append(f"{'=' * 90}")
    lines.append("IMPACT ON NON-WINNER NOMINEES (scenario prob - baseline prob, in pp)")
    lines.append(f"{'=' * 90}")
    lines.append("")

    # Average across models for cleaner presentation
    header = f"{'Nominee':<30}"
    for winner in event.eligible_nominees:
        header += f" {winner[:12]:>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for title in all_nominees:
        if title in event.eligible_nominees:
            continue  # Skip the winner rows (shown above)
        row = f"{title:<30}"
        for winner_title in event.eligible_nominees:
            deltas = []
            for m in model_names:
                base_pred = get_pred_for_nominee(summary.baselines[m], title)
                cf = next(
                    (
                        c
                        for c in summary.counterfactuals
                        if c.model_name == m and c.scenario_nominee == winner_title
                    ),
                    None,
                )
                scen_pred = get_pred_for_nominee(cf.predictions, title) if cf else None
                if base_pred and scen_pred:
                    deltas.append(scen_pred.probability - base_pred.probability)
            mean_delta = sum(deltas) / len(deltas) if deltas else 0
            row += f" {mean_delta * 100:>+11.1f}pp"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def build_summary_json(summary: CounterfactualSummary) -> dict:
    """Build structured JSON summary for programmatic consumption."""
    event = summary.event

    result: dict = {
        "event": {
            "name": event.name,
            "description": event.description,
            "pre_event_date": event.pre_event_date.isoformat(),
            "post_event_date": event.post_event_date.isoformat(),
            "market_price_date": event.market_price_date.isoformat(),
            "eligible_nominees": event.eligible_nominees,
        },
        "baselines": {},
        "scenarios": {},
        "market_prices": {},
    }

    # Baselines
    for model_name, preds in summary.baselines.items():
        result["baselines"][model_name] = {
            p.title: {"probability": round(p.probability, 4), "rank": p.rank} for p in preds
        }

    # Scenarios
    for cf in summary.counterfactuals:
        key = f"{cf.model_name}_{cf.scenario_nominee}"
        result["scenarios"][key] = {
            "model": cf.model_name,
            "scenario_nominee": cf.scenario_nominee,
            "predictions": {
                p.title: {"probability": round(p.probability, 4), "rank": p.rank}
                for p in cf.predictions
            },
        }

    # Market prices
    for title in [p.title for p in next(iter(summary.baselines.values()))]:
        price = summary.market_prices.get(title)
        if price is not None:
            result["market_prices"][title] = round(price, 4)

    return result


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Run counterfactual analysis for feature events")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory with trained baseline/scenario models + output destination",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        required=True,
        help="Path to raw dataset JSON file.",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    output_dir = Path(args.output_dir)
    raw_path = Path(args.raw_path)

    # Run DGA winner counterfactual analysis
    print(f"\nRunning counterfactual analysis: {DGA_WINNER_EVENT.name}")
    print(f"Output directory: {output_dir}")

    summary = run_counterfactual_analysis(DGA_WINNER_EVENT, output_dir, raw_path=raw_path)

    # Format and save text summary
    text_output = format_summary_tables(summary)
    print(text_output)

    summary_txt_path = output_dir / "counterfactual_summary.txt"
    with open(summary_txt_path, "w") as f:
        f.write(text_output)
    print(f"\nSaved text summary: {summary_txt_path}")

    # Save JSON summary
    json_output = build_summary_json(summary)
    summary_json_path = output_dir / "counterfactual_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"Saved JSON summary: {summary_json_path}")


if __name__ == "__main__":
    main()
