"""Shared utility functions for the modeling module.

Contains:
- Path resolution utilities
- Year/ceremony conversion utilities
- CV config loading with Pydantic TypeAdapter
"""

from pathlib import Path

from pydantic import TypeAdapter

from oscar_prediction_market.modeling.cv_splitting import (
    CVSplitConfig,
)

# Module-level paths
MODELING_DIR = Path(__file__).parent


# ============================================================================
# Path Resolution
# ============================================================================


def resolve_config_path(config_path: str) -> Path:
    """Resolve config path relative to modeling directory if not absolute.

    Args:
        config_path: Path string (absolute or relative)

    Returns:
        Resolved absolute Path

    Raises:
        FileNotFoundError: If config file not found
    """
    path = Path(config_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    modeling_relative = MODELING_DIR / path
    if modeling_relative.exists():
        return modeling_relative
    raise FileNotFoundError(f"Config file not found: {config_path}")


# ============================================================================
# Year/Ceremony Conversion
# ============================================================================


def year_to_ceremony(year: int) -> int:
    """Convert year to Oscar ceremony number.

    Example: 2026 -> 98
    """
    return year - 1928


def ceremony_to_year(ceremony: int) -> int:
    """Convert Oscar ceremony number to year.

    Example: 98 -> 2026 (1st ceremony was 1929)
    """
    return ceremony + 1928


def parse_year_range(year_range: str) -> tuple[int, int]:
    """Parse year range string like '2000-2025' or single year '2026'.

    Args:
        year_range: String like '2000-2025' or '2026'

    Returns:
        (start_year, end_year) tuple

    Raises:
        ValueError: If format is invalid
    """
    if "-" in year_range:
        parts = year_range.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid year range format: {year_range}. Expected 'START-END'")
        return int(parts[0]), int(parts[1])
    else:
        year = int(year_range)
        return year, year


# ============================================================================
# CV Config Loading with Pydantic TypeAdapter
# ============================================================================

# TypeAdapter for CV split config
_cv_split_adapter: TypeAdapter[CVSplitConfig] = TypeAdapter(CVSplitConfig)


def load_cv_split_config(config_path: Path) -> CVSplitConfig:
    """Load CV split config using Pydantic TypeAdapter.

    Args:
        config_path: Path to JSON config file

    Returns:
        Validated CVSplitConfig
    """
    with open(config_path, "rb") as f:
        return _cv_split_adapter.validate_json(f.read())
