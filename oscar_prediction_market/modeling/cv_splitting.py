"""Cross-validation splitters for temporal data in Oscar Best Picture prediction.

Provides multiple CV splitting strategies for splitting data along the TEMPORAL dimension.
These splitters define how to partition years into train/test sets for backtesting.

┌─────────────────────┬──────────────┬──────────┬───────────────┬─────────────────────────────────────────┐
│ Splitter            │ Respects     │ Variance │ Training Data │ Best For / Motivation                   │
│                     │ Time Order?  │          │               │                                         │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Expanding Window    │ ✓ Yes        │ Higher   │ Grows over    │ REALISTIC DEPLOYMENT ESTIMATE.          │
│ (current default)   │              │          │ time          │ Mimics actual prediction scenario where │
│                     │              │          │               │ you only have past data. Use as primary │
│                     │              │          │               │ metric for model comparison.            │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Leave-One-Year-Out  │ ✗ No         │ Lower    │ Maximum       │ CEILING ESTIMATE. Uses all available    │
│ (LOYO)              │ (uses future)│          │ (N-1 years)   │ data for training. Shows "best possible"│
│                     │              │          │               │ performance if time didn't matter.      │
│                     │              │          │               │ Compare to Expanding to see if temporal │
│                     │              │          │               │ structure matters.                      │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Sliding Window      │ ✓ Yes        │ Higher   │ Fixed size    │ REGIME CHANGE DETECTION. Uses only      │
│                     │              │          │ (recent N)    │ recent N years for training. If this    │
│                     │              │          │               │ beats Expanding, old data is hurting.   │
│                     │              │          │               │ Helps detect if Oscar dynamics shifted. │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Bootstrap           │ Depends on   │ N/A      │ Varies        │ CONFIDENCE INTERVALS. Resamples years   │
│                     │ base splitter│ (CI)     │               │ with replacement to estimate variance.  │
│                     │              │          │               │ Use for statistical significance and    │
│                     │              │          │               │ uncertainty quantification on metrics.  │
└─────────────────────┴──────────────┴──────────┴───────────────┴─────────────────────────────────────────┘

Usage:
    from cv_splitting import (
        ExpandingWindowSplitter,
        LeaveOneYearOutSplitter,
        SlidingWindowSplitter,
        BootstrapSplitter,
    )

    # Create splitter
    splitter = ExpandingWindowSplitter(min_train_years=5)

    # Generate train/test splits
    for fold in splitter.generate_folds(ceremony_years):
        train_ceremonies = fold.train_ceremonies
        test_ceremonies = fold.test_ceremonies
        # ... train and evaluate
"""

import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Fold Definition
# ============================================================================


class CVFold(BaseModel):
    """A single cross-validation fold with train and test ceremony years.

    All fields required - no defaults.
    """

    fold_idx: int = Field(..., description="Fold index (0-based)", ge=0)
    train_ceremonies: list[int] = Field(
        ..., description="Ceremony years for training", min_length=1
    )
    test_ceremonies: list[int] = Field(..., description="Ceremony years for testing", min_length=1)
    description: str = Field(..., description="Human-readable description of this fold")

    model_config = {"extra": "forbid"}


# ============================================================================
# Strategy Configurations (Pydantic)
# ============================================================================


class ExpandingWindowConfig(BaseModel):
    """Configuration for Expanding Window (anchored walk-forward) strategy.

    Trains on all years from start up to test year, expands forward.
    Respects temporal ordering - no future data leakage.
    """

    strategy_type: Literal["expanding_window"] = Field(..., description="Strategy type identifier")
    min_train_years: int = Field(
        ..., description="Minimum years of training data before first test", ge=1
    )
    num_test_years: int = Field(..., description="Number of consecutive test years per fold", ge=1)

    model_config = {"extra": "forbid"}


class LeaveOneYearOutConfig(BaseModel):
    """Configuration for Leave-One-Year-Out (LOYO) strategy.

    For each test year, trains on ALL other years (past and future).
    Does NOT respect temporal ordering - provides ceiling estimate.
    """

    strategy_type: Literal["leave_one_year_out"] = Field(
        ..., description="Strategy type identifier"
    )

    model_config = {"extra": "forbid"}


class SlidingWindowConfig(BaseModel):
    """Configuration for Sliding Window strategy.

    Trains on most recent N years, tests on next year.
    Useful for detecting regime changes - if this beats Expanding, old data hurts.
    """

    strategy_type: Literal["sliding_window"] = Field(..., description="Strategy type identifier")
    train_window_size: int = Field(..., description="Number of years in training window", ge=1)

    model_config = {"extra": "forbid"}


class BootstrapConfig(BaseModel):
    """Configuration for Bootstrap splitter.

    Resamples years with replacement to estimate confidence intervals.
    Uses a base splitter for actual train/test splits within each bootstrap sample.
    """

    strategy_type: Literal["bootstrap"] = Field(..., description="Splitter type identifier")
    n_bootstrap: int = Field(..., description="Number of bootstrap samples", ge=10)
    base_strategy: "CVSplitConfig" = Field(
        ..., description="Base splitter to use within each bootstrap sample"
    )
    random_state: int = Field(..., description="Random seed for reproducibility")

    model_config = {"extra": "forbid"}


# Discriminated union for all CV split configs
CVSplitConfig = (
    ExpandingWindowConfig | LeaveOneYearOutConfig | SlidingWindowConfig | BootstrapConfig
)


# ============================================================================
# Base Splitter Interface
# ============================================================================


class CVSplitter(ABC):
    """Abstract base class for temporal cross-validation splitters.

    Splitters define how to partition ceremony years into train/test sets.
    All splitting is done along the temporal (year) dimension.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable splitter name."""
        pass

    @property
    @abstractmethod
    def respects_time_order(self) -> bool:
        """Whether this splitter respects temporal ordering (no future leakage)."""
        pass

    @abstractmethod
    def generate_folds(self, ceremony_years: list[int]) -> list[CVFold]:
        """Generate train/test folds for cross-validation.

        Args:
            ceremony_years: Sorted list of all ceremony years in dataset

        Returns:
            List of CVFold objects defining train/test splits
        """
        pass

    @abstractmethod
    def get_config(self) -> CVSplitConfig:
        """Return the configuration for this splitter."""
        pass


# ============================================================================
# Expanding Window Splitter
# ============================================================================


class ExpandingWindowSplitter(CVSplitter):
    """Expanding window (anchored walk-forward) cross-validation.

    For each test year t:
    - Train on years [1, 2, ..., t-1]
    - Test on year t

    Respects temporal ordering. Primary splitter for realistic deployment estimates.
    """

    def __init__(self, min_train_years: int, num_test_years: int):
        """
        Args:
            min_train_years: Minimum training years before first test
            num_test_years: Number of consecutive test years per fold
        """
        self.min_train_years = min_train_years
        self.num_test_years = num_test_years

    @property
    def name(self) -> str:
        return "Expanding Window"

    @property
    def respects_time_order(self) -> bool:
        return True

    def generate_folds(self, ceremony_years: list[int]) -> list[CVFold]:
        """Generate expanding window folds."""
        ceremonies = sorted(ceremony_years)
        min_ceremony = min(ceremonies)
        max_ceremony = max(ceremonies)

        # First test year needs min_train_years of data
        first_test_year = min_ceremony + self.min_train_years
        last_test_start = max_ceremony - self.num_test_years + 1

        folds = []
        fold_idx = 0
        current_test_start = first_test_year

        while current_test_start <= last_test_start:
            # Train on all years before test window
            train_ceremonies = [c for c in ceremonies if c < current_test_start]

            # Test on num_test_years consecutive years
            test_ceremonies = [
                c
                for c in ceremonies
                if current_test_start <= c < current_test_start + self.num_test_years
            ]

            if len(train_ceremonies) >= self.min_train_years and len(test_ceremonies) > 0:
                folds.append(
                    CVFold(
                        fold_idx=fold_idx,
                        train_ceremonies=train_ceremonies,
                        test_ceremonies=test_ceremonies,
                        description=(
                            f"Train [{min(train_ceremonies)}-{max(train_ceremonies)}] "
                            f"({len(train_ceremonies)} yrs), "
                            f"Test [{min(test_ceremonies)}-{max(test_ceremonies)}]"
                        ),
                    )
                )
                fold_idx += 1

            current_test_start += 1

        return folds

    def get_config(self) -> ExpandingWindowConfig:
        return ExpandingWindowConfig(
            strategy_type="expanding_window",
            min_train_years=self.min_train_years,
            num_test_years=self.num_test_years,
        )


# ============================================================================
# Leave-One-Year-Out Splitter
# ============================================================================


class LeaveOneYearOutSplitter(CVSplitter):
    """Leave-One-Year-Out (LOYO) cross-validation.

    For each test year t:
    - Train on ALL other years (past AND future)
    - Test on year t

    Does NOT respect temporal ordering. Provides ceiling estimate of performance.
    """

    @property
    def name(self) -> str:
        return "Leave-One-Year-Out"

    @property
    def respects_time_order(self) -> bool:
        return False

    def generate_folds(self, ceremony_years: list[int]) -> list[CVFold]:
        """Generate LOYO folds - one fold per year."""
        ceremonies = sorted(ceremony_years)
        folds = []

        for fold_idx, test_year in enumerate(ceremonies):
            train_ceremonies = [c for c in ceremonies if c != test_year]

            folds.append(
                CVFold(
                    fold_idx=fold_idx,
                    train_ceremonies=train_ceremonies,
                    test_ceremonies=[test_year],
                    description=(
                        f"Train on all except {test_year} "
                        f"({len(train_ceremonies)} yrs), Test [{test_year}]"
                    ),
                )
            )

        return folds

    def get_config(self) -> LeaveOneYearOutConfig:
        return LeaveOneYearOutConfig(strategy_type="leave_one_year_out")


# ============================================================================
# Sliding Window Splitter
# ============================================================================


class SlidingWindowSplitter(CVSplitter):
    """Sliding window cross-validation with fixed training size.

    For each test year t:
    - Train on years [t-N, t-N+1, ..., t-1] (most recent N years)
    - Test on year t

    Useful for detecting regime changes. If this beats Expanding Window,
    old data is hurting performance.
    """

    def __init__(self, train_window_size: int):
        """
        Args:
            train_window_size: Number of years in training window
        """
        self.train_window_size = train_window_size

    @property
    def name(self) -> str:
        return f"Sliding Window (N={self.train_window_size})"

    @property
    def respects_time_order(self) -> bool:
        return True

    def generate_folds(self, ceremony_years: list[int]) -> list[CVFold]:
        """Generate sliding window folds."""
        ceremonies = sorted(ceremony_years)
        folds = []
        fold_idx = 0

        for test_year in ceremonies:
            # Train on N most recent years before test year
            train_candidates = [c for c in ceremonies if c < test_year]
            train_ceremonies = train_candidates[-self.train_window_size :]

            if len(train_ceremonies) == self.train_window_size:
                folds.append(
                    CVFold(
                        fold_idx=fold_idx,
                        train_ceremonies=train_ceremonies,
                        test_ceremonies=[test_year],
                        description=(
                            f"Train [{min(train_ceremonies)}-{max(train_ceremonies)}] "
                            f"(last {self.train_window_size} yrs), Test [{test_year}]"
                        ),
                    )
                )
                fold_idx += 1

        return folds

    def get_config(self) -> SlidingWindowConfig:
        return SlidingWindowConfig(
            strategy_type="sliding_window",
            train_window_size=self.train_window_size,
        )


# ============================================================================
# Bootstrap Splitter
# ============================================================================


class BootstrapSplitter(CVSplitter):
    """Bootstrap splitter for confidence interval estimation.

    Resamples years with replacement, then applies a base splitter.
    Returns multiple sets of folds for uncertainty quantification.
    """

    def __init__(
        self,
        base_splitter: CVSplitter,
        n_bootstrap: int,
        random_state: int,
    ):
        """
        Args:
            base_splitter: Splitter to use within each bootstrap sample
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.base_splitter = base_splitter
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @property
    def name(self) -> str:
        return f"Bootstrap ({self.n_bootstrap} samples, base={self.base_splitter.name})"

    @property
    def respects_time_order(self) -> bool:
        return self.base_splitter.respects_time_order

    def generate_folds(self, ceremony_years: list[int]) -> list[CVFold]:
        """Generate folds - uses base splitter on original data.

        For bootstrap, use generate_bootstrap_samples() instead.
        """
        return self.base_splitter.generate_folds(ceremony_years)

    def generate_bootstrap_samples(
        self, ceremony_years: list[int]
    ) -> list[tuple[list[int], list[CVFold]]]:
        """Generate bootstrap samples, each with its own folds.

        Args:
            ceremony_years: Original ceremony years

        Returns:
            List of (sampled_years, folds) tuples for each bootstrap sample
        """
        ceremonies = sorted(ceremony_years)
        samples = []

        for _ in range(self.n_bootstrap):
            # Sample years with replacement
            sampled = self._rng.choice(ceremonies, size=len(ceremonies), replace=True).tolist()
            sampled_unique = sorted(set(sampled))

            # Generate folds on sampled years
            folds = self.base_splitter.generate_folds(sampled_unique)

            if len(folds) > 0:
                samples.append((sampled_unique, folds))

        return samples

    def get_config(self) -> BootstrapConfig:
        return BootstrapConfig(
            strategy_type="bootstrap",
            n_bootstrap=self.n_bootstrap,
            base_strategy=self.base_splitter.get_config(),
            random_state=self.random_state,
        )


# ============================================================================
# Factory Function
# ============================================================================


def create_splitter(config: CVSplitConfig) -> CVSplitter:
    """Factory function to create a CV splitter from configuration."""
    if isinstance(config, ExpandingWindowConfig):
        return ExpandingWindowSplitter(
            min_train_years=config.min_train_years,
            num_test_years=config.num_test_years,
        )
    elif isinstance(config, LeaveOneYearOutConfig):
        return LeaveOneYearOutSplitter()
    elif isinstance(config, SlidingWindowConfig):
        return SlidingWindowSplitter(train_window_size=config.train_window_size)
    elif isinstance(config, BootstrapConfig):
        base_splitter = create_splitter(config.base_strategy)
        return BootstrapSplitter(
            base_splitter=base_splitter,
            n_bootstrap=config.n_bootstrap,
            random_state=config.random_state,
        )
    else:
        raise ValueError(f"Unknown splitter config type: {type(config)}")


# ============================================================================
# Utility Functions
# ============================================================================


def print_splitter_comparison() -> None:
    """Print comparison table of all splitters."""
    print(
        """
┌─────────────────────┬──────────────┬──────────┬───────────────┬─────────────────────────────────────────┐
│ Splitter            │ Respects     │ Variance │ Training Data │ Best For / Motivation                   │
│                     │ Time Order?  │          │               │                                         │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Expanding Window    │ ✓ Yes        │ Higher   │ Grows over    │ REALISTIC DEPLOYMENT ESTIMATE.          │
│ (default)           │              │          │ time          │ Primary metric for model comparison.    │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Leave-One-Year-Out  │ ✗ No         │ Lower    │ Maximum       │ CEILING ESTIMATE. Compare to Expanding  │
│ (LOYO)              │ (uses future)│          │ (N-1 years)   │ to see if temporal structure matters.   │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Sliding Window      │ ✓ Yes        │ Higher   │ Fixed size    │ REGIME CHANGE DETECTION. If beats       │
│                     │              │          │ (recent N)    │ Expanding, old data is hurting.         │
├─────────────────────┼──────────────┼──────────┼───────────────┼─────────────────────────────────────────┤
│ Bootstrap           │ Depends on   │ N/A      │ Varies        │ CONFIDENCE INTERVALS for metrics.       │
│                     │ base splitter│ (CI)     │               │ Uncertainty quantification.             │
└─────────────────────┴──────────────┴──────────┴───────────────┴─────────────────────────────────────────┘
"""
    )
