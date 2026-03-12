"""Model persistence — save and load trained models."""

import pickle
from pathlib import Path

from oscar_prediction_market.modeling.models.base import PredictionModel


def save_model(model: PredictionModel, output_path: Path) -> None:
    """Save trained model to pickle file.

    Args:
        model: Trained PredictionModel instance
        output_path: Path to save the pickle file
    """
    with open(output_path, "wb") as f:
        pickle.dump(model, f)


def load_model(model_path: Path) -> PredictionModel:
    """Load trained model from pickle file.

    Args:
        model_path: Path to the pickle file

    Returns:
        Loaded PredictionModel instance
    """
    with open(model_path, "rb") as f:
        return pickle.load(f)
