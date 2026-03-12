"""Tests for Oscar-specific prediction loading and remapping."""

from pathlib import Path

import pytest

from oscar_prediction_market.data.schema import (
    NominationDataset,
    OscarCategory,
)
from oscar_prediction_market.modeling.models import ModelType
from oscar_prediction_market.modeling.prediction_io import load_predictions
from oscar_prediction_market.trading.oscar_prediction_source import (
    load_all_snapshot_predictions,
    load_nomination_dataset,
)
from tests.modeling.conftest import _make_actor_record


def _write_prediction_csv(
    models_dir: Path,
    category: OscarCategory,
    model_type: ModelType,
    snapshot_key: str,
    rows: list[tuple[str, float]],
) -> None:
    short_name = model_type.short_name
    pred_dir = (
        models_dir
        / category.slug
        / short_name
        / snapshot_key
        / f"{short_name}_{snapshot_key}"
        / "5_final_predict"
    )
    pred_dir.mkdir(parents=True, exist_ok=True)
    lines = ["title,probability"]
    lines.extend(f"{title},{prob}" for title, prob in rows)
    (pred_dir / "predictions_test.csv").write_text("\n".join(lines) + "\n")


def test_load_predictions_requires_canonical_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_predictions(
            category=OscarCategory.BEST_PICTURE,
            model_type=ModelType.LOGISTIC_REGRESSION,
            snapshot_key="2025-01-23_oscar_noms",
            models_dir=tmp_path,
        )


def test_load_nomination_dataset_and_remap_actor_titles(tmp_path: Path) -> None:
    dataset = NominationDataset(
        category=OscarCategory.ACTOR_LEADING,
        year_start=2025,
        year_end=2025,
        record_count=2,
        records=[
            _make_actor_record(
                title="Film A",
                film_id="tt0000001",
                ceremony=97,
                year_film=2024,
                winner=True,
                total_noms=5,
                noms_by_category={OscarCategory.ACTOR_LEADING.value: 1},
                person_name="Actor A",
            ),
            _make_actor_record(
                title="Film B",
                film_id="tt0000002",
                ceremony=97,
                year_film=2024,
                winner=False,
                total_noms=3,
                noms_by_category={OscarCategory.ACTOR_LEADING.value: 1},
                person_name="Actor B",
            ),
        ],
    )
    datasets_dir = tmp_path / "datasets"
    dataset_dir = datasets_dir / "actor_leading" / "2025-01-23_oscar_noms"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "oscar_actor_leading_raw.json").write_text(dataset.model_dump_json())

    loaded_dataset = load_nomination_dataset(
        datasets_dir=datasets_dir,
        category=OscarCategory.ACTOR_LEADING,
        snapshot_key="2025-01-23_oscar_noms",
    )
    assert loaded_dataset.build_title_to_person_map(2025) == {
        "Film A": "Actor A",
        "Film B": "Actor B",
    }

    models_dir = tmp_path / "models"
    _write_prediction_csv(
        models_dir=models_dir,
        category=OscarCategory.ACTOR_LEADING,
        model_type=ModelType.LOGISTIC_REGRESSION,
        snapshot_key="2025-01-23_oscar_noms",
        rows=[("Film A", 0.7), ("Film B", 0.3)],
    )

    preds = load_all_snapshot_predictions(
        category=OscarCategory.ACTOR_LEADING,
        model_type=ModelType.LOGISTIC_REGRESSION,
        snapshot_keys=["2025-01-23_oscar_noms"],
        models_dir=models_dir,
        dataset=loaded_dataset,
        ceremony_year=2025,
    )
    assert preds == {"2025-01-23_oscar_noms": {"Actor A": 0.7, "Actor B": 0.3}}


def test_load_all_snapshot_predictions_screenplay_uses_film_titles_directly(tmp_path: Path) -> None:
    """ORIGINAL_SCREENPLAY predictions already use film titles (from the CSV
    ``title`` column), so no remapping is needed — keys pass through as-is."""
    record_a = _make_actor_record(
        title="Film A",
        film_id="tt0000011",
        ceremony=97,
        year_film=2024,
        winner=True,
        total_noms=4,
        noms_by_category={OscarCategory.ORIGINAL_SCREENPLAY.value: 1},
        person_name="Writer A",
    ).model_copy(update={"category": OscarCategory.ORIGINAL_SCREENPLAY})
    record_b = _make_actor_record(
        title="Film B",
        film_id="tt0000012",
        ceremony=97,
        year_film=2024,
        winner=False,
        total_noms=2,
        noms_by_category={OscarCategory.ORIGINAL_SCREENPLAY.value: 1},
        person_name="Writer B",
    ).model_copy(update={"category": OscarCategory.ORIGINAL_SCREENPLAY})
    dataset = NominationDataset(
        category=OscarCategory.ORIGINAL_SCREENPLAY,
        year_start=2025,
        year_end=2025,
        record_count=2,
        records=[record_a, record_b],
    )

    datasets_dir = tmp_path / "datasets"
    dataset_dir = datasets_dir / "original_screenplay" / "2025-01-23_oscar_noms"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "oscar_original_screenplay_raw.json").write_text(dataset.model_dump_json())

    models_dir = tmp_path / "models"
    # Predictions use film titles directly (not writer names)
    _write_prediction_csv(
        models_dir=models_dir,
        category=OscarCategory.ORIGINAL_SCREENPLAY,
        model_type=ModelType.LOGISTIC_REGRESSION,
        snapshot_key="2025-01-23_oscar_noms",
        rows=[("Film A", 0.55), ("Film B", 0.45)],
    )

    preds = load_all_snapshot_predictions(
        category=OscarCategory.ORIGINAL_SCREENPLAY,
        model_type=ModelType.LOGISTIC_REGRESSION,
        snapshot_keys=["2025-01-23_oscar_noms"],
        models_dir=models_dir,
        dataset=dataset,
        ceremony_year=2025,
    )
    # Film titles pass through without remapping
    assert preds == {"2025-01-23_oscar_noms": {"Film A": 0.55, "Film B": 0.45}}
