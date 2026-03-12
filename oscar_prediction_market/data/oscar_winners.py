"""Oscar ceremony winners and Kalshi market conventions by year.

Consolidates per-year winner data and Kalshi-specific category conventions
so they are defined in one place and imported everywhere.
"""

from oscar_prediction_market.data.schema import OscarCategory

# ============================================================================
# Winners by ceremony year
# ============================================================================

WINNERS_2024: dict[OscarCategory, str] = {
    OscarCategory.BEST_PICTURE: "Oppenheimer",
    OscarCategory.DIRECTING: "Christopher Nolan",
    OscarCategory.ACTOR_LEADING: "Cillian Murphy",
    OscarCategory.ACTRESS_LEADING: "Emma Stone",
    OscarCategory.ACTOR_SUPPORTING: "Robert Downey Jr.",
    OscarCategory.ACTRESS_SUPPORTING: "Da'Vine Joy Randolph",
    OscarCategory.ORIGINAL_SCREENPLAY: "Anatomy of a Fall",
    OscarCategory.ANIMATED_FEATURE: "The Boy and the Heron",
    # No Cinematography market on Kalshi for 2024
}

WINNERS_2025: dict[OscarCategory, str] = {
    OscarCategory.BEST_PICTURE: "Anora",
    OscarCategory.DIRECTING: "Sean Baker",
    OscarCategory.ACTOR_LEADING: "Adrien Brody",
    OscarCategory.ACTRESS_LEADING: "Mikey Madison",
    OscarCategory.ACTOR_SUPPORTING: "Kieran Culkin",
    OscarCategory.ACTRESS_SUPPORTING: "Zoe Saldaña",
    OscarCategory.ORIGINAL_SCREENPLAY: "Anora",
    OscarCategory.ANIMATED_FEATURE: "Flow",
    OscarCategory.CINEMATOGRAPHY: "The Brutalist",
}

WINNERS_BY_YEAR: dict[int, dict[OscarCategory, str]] = {
    2024: WINNERS_2024,
    2025: WINNERS_2025,
}


# ============================================================================
# Kalshi market conventions
# ============================================================================

#: Categories where Kalshi uses person names (not film titles) as ticker labels.
#: This is a subset of data.schema.PERSON_CATEGORIES (which includes Original
#: Screenplay and Cinematography at the data level). Here we only list the 5
#: categories where the Kalshi *ticker labels* use person names.
KALSHI_PERSON_NAME_CATEGORIES: set[OscarCategory] = {
    OscarCategory.DIRECTING,
    OscarCategory.ACTOR_LEADING,
    OscarCategory.ACTRESS_LEADING,
    OscarCategory.ACTOR_SUPPORTING,
    OscarCategory.ACTRESS_SUPPORTING,
}
