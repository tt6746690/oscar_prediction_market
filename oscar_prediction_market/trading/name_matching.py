"""Nominee name matching between model predictions and Kalshi markets.

Model predictions use names from oscars.csv / TMDb (e.g., ``"Adrien Brody"``),
while Kalshi markets use their own names (e.g., ``"Adrien Brody"`` for
people, ``"The Brutalist"`` for films). These mostly overlap but have edge
cases: accent variations, "Written by" prefixes for screenplay, and different
title formatting.

Matching strategy (applied in order):
1. Exact match after normalization (lowercase, strip accents, collapse whitespace)
2. Fuzzy match via ``thefuzz`` (ratio > 85)
3. Manual override dict for known mismatches

Usage::

    from oscar_prediction_market.trading.name_matching import (
        match_nominees,
        normalize_name,
    )

    mapping = match_nominees(
        model_names=["Adrien Brody", "Timothée Chalamet"],
        kalshi_names=["Adrien Brody", "Timothee Chalamet"],
        category=OscarCategory.ACTOR_LEADING,
        ceremony_year=2025,
    )
    # {"Adrien Brody": "Adrien Brody", "Timothée Chalamet": "Timothee Chalamet"}
"""

import logging
import re

from oscar_prediction_market.data.schema import (
    OscarCategory,
)
from oscar_prediction_market.data.utils import normalize_person_name

logger = logging.getLogger(__name__)

# ============================================================================
# Manual overrides: {(category_slug, ceremony_year): {model_name: kalshi_name}}
# Add entries here when automated matching fails.
# ============================================================================

MANUAL_OVERRIDES: dict[tuple[str, int], dict[str, str]] = {
    # Screenplay category: model has "Written by X" while Kalshi has film title
    # These are handled by the film_title fallback, not manual overrides
}


# ============================================================================
# Name normalisation
# ============================================================================


def normalize_name(name: str) -> str:
    """Normalize a name for matching: strip accents, lowercase, collapse whitespace.

    For film titles, additionally strips common articles and punctuation.
    Reuses ``normalize_person_name()`` for accent stripping.

    Args:
        name: Raw name string.

    Returns:
        Normalized string for comparison.
    """
    # Use the existing accent-stripping normalizer
    normalized = normalize_person_name(name)
    # Strip common prefixes for screenplays
    normalized = re.sub(r"^(written by|screenplay by|story by)\s+", "", normalized)
    # Strip "the " prefix for film titles
    normalized = re.sub(r"^the\s+", "", normalized)
    # Remove punctuation except spaces
    normalized = re.sub(r"[^\w\s]", "", normalized)
    # Collapse whitespace
    return " ".join(normalized.split())


def _fuzzy_ratio(a: str, b: str) -> int:
    """Compute fuzzy string similarity ratio (0-100).

    Uses thefuzz.fuzz.ratio for character-level Levenshtein similarity.
    """
    from thefuzz import fuzz

    return fuzz.ratio(a, b)


def match_nominees(
    model_names: list[str],
    kalshi_names: list[str],
    category: OscarCategory,
    ceremony_year: int,
    fuzzy_threshold: int = 85,
    model_film_titles: dict[str, str] | None = None,
) -> dict[str, str]:
    """Match model nominee names to Kalshi market names.

    For person categories (acting, directing, cinematography), matches on
    the person name. For film categories (best picture, animated feature),
    matches on the film title.

    For screenplay, model names may be "Written by Sean Baker" while Kalshi
    uses the film title "Anora". In this case, pass ``model_film_titles``
    mapping model names → film titles as a fallback.

    Args:
        model_names: Names from model predictions (e.g., from predictions_test.csv).
        kalshi_names: Names from Kalshi markets (e.g., from ticker inventory).
        category: Oscar category (determines matching strategy).
        ceremony_year: Ceremony year (for manual overrides).
        fuzzy_threshold: Minimum fuzzy ratio to accept a match (0-100).
        model_film_titles: Optional {model_name: film_title} for film-title fallback
            (used for screenplay category where model has person names but Kalshi has titles).

    Returns:
        ``{model_name: kalshi_name}`` mapping. Only includes matched names.
    """
    cat_slug = category.slug
    overrides = MANUAL_OVERRIDES.get((cat_slug, ceremony_year), {})

    # Normalize all names for comparison
    norm_kalshi = {normalize_name(kn): kn for kn in kalshi_names}
    matched: dict[str, str] = {}
    unmatched_model: list[str] = []

    for model_name in model_names:
        # Check manual overrides first
        if model_name in overrides:
            matched[model_name] = overrides[model_name]
            continue

        norm_model = normalize_name(model_name)

        # Step 1: Exact match after normalization
        if norm_model in norm_kalshi:
            matched[model_name] = norm_kalshi[norm_model]
            continue

        # Step 2: Fuzzy match
        best_score = 0
        best_kalshi = None
        for norm_k, orig_k in norm_kalshi.items():
            score = _fuzzy_ratio(norm_model, norm_k)
            if score > best_score:
                best_score = score
                best_kalshi = orig_k

        if best_score >= fuzzy_threshold and best_kalshi is not None:
            matched[model_name] = best_kalshi
            logger.debug(
                "Fuzzy match: '%s' -> '%s' (score=%d)", model_name, best_kalshi, best_score
            )
            continue

        # Step 3: For screenplay/person categories, try film title fallback
        if model_film_titles and model_name in model_film_titles:
            film_title = model_film_titles[model_name]
            norm_film = normalize_name(film_title)
            if norm_film in norm_kalshi:
                matched[model_name] = norm_kalshi[norm_film]
                logger.debug(
                    "Film title fallback: '%s' (film='%s') -> '%s'",
                    model_name,
                    film_title,
                    norm_kalshi[norm_film],
                )
                continue

            # Fuzzy on film title
            for norm_k, orig_k in norm_kalshi.items():
                score = _fuzzy_ratio(norm_film, norm_k)
                if score >= fuzzy_threshold:
                    matched[model_name] = orig_k
                    logger.debug(
                        "Film title fuzzy: '%s' (film='%s') -> '%s' (score=%d)",
                        model_name,
                        film_title,
                        orig_k,
                        score,
                    )
                    break
            else:
                unmatched_model.append(model_name)
        else:
            unmatched_model.append(model_name)

    if unmatched_model:
        logger.warning(
            "Unmatched model names for %s %d: %s",
            category.name,
            ceremony_year,
            unmatched_model,
        )

    return matched


def validate_matching(
    matched: dict[str, str],
    model_names: list[str],
    kalshi_names: list[str],
    category: OscarCategory,
    ceremony_year: int,
) -> None:
    """Print matching validation report for human review.

    Args:
        matched: Result from ``match_nominees()``.
        model_names: All model names.
        kalshi_names: All Kalshi names.
        category: Oscar category.
        ceremony_year: Ceremony year.
    """
    print(f"\n{'=' * 70}")
    print(f"Nominee Matching: {category.name} {ceremony_year}")
    print(f"{'=' * 70}")
    print(f"  Model names:  {len(model_names)}")
    print(f"  Kalshi names: {len(kalshi_names)}")
    print(f"  Matched:      {len(matched)}")

    print(f"\n  {'Model Name':<40} {'Kalshi Name':<30}")
    print(f"  {'-' * 40} {'-' * 30}")
    for mn, kn in sorted(matched.items()):
        print(f"  {mn:<40} {kn:<30}")

    unmatched_model = [n for n in model_names if n not in matched]
    if unmatched_model:
        print(f"\n  UNMATCHED model names: {unmatched_model}")

    unmatched_kalshi = [n for n in kalshi_names if n not in matched.values()]
    if unmatched_kalshi:
        print(f"  UNMATCHED Kalshi names: {unmatched_kalshi}")
