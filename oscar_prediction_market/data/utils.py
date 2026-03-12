"""Text-processing utilities for Oscar data pipeline.

Name normalisation and screenplay credit parsing used across fetching, building,
and matching stages.
"""

import re
import unicodedata

# ============================================================================
# Person Name Normalisation
# ============================================================================


def normalize_person_name(name: str) -> str:
    """Normalize a person name for matching: strip accents, lowercase, collapse whitespace.

    Used for matching person names across data sources (oscars.csv, precursor award
    Wikipedia tables, TMDb) which may spell the same name differently due to accent
    variations or whitespace inconsistencies.

    Example::

        >>> normalize_person_name("Timothée Chalamet")
        'timothee chalamet'
        >>> normalize_person_name("Renée Zellweger")
        'renee zellweger'
        >>> normalize_person_name("Robert De Niro")
        'robert de niro'
    """
    # NFD decomposition splits accented chars into base + combining mark
    nfkd = unicodedata.normalize("NFKD", name)
    # Strip combining marks (accents)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase + collapse whitespace
    return " ".join(stripped.lower().split())


# ============================================================================
# Screenplay Credit Parsing
# ============================================================================

# Prefixes that appear before writer names in oscars.csv (ceremony 73+).
# Ordered longest-first to avoid partial matches.
_SCREENPLAY_CREDIT_PREFIXES = [
    "Screen Story by",
    "Screen Story and Screenplay by",
    "Screenplay by",
    "Written by",
    "Story by",
    "Written for the Screen by",
    "Original Screenplay by",
]

# Compiled pattern: match any prefix (case-insensitive) at start of a credit block
_PREFIX_PATTERN = re.compile(
    r"^(?:" + "|".join(re.escape(p) for p in _SCREENPLAY_CREDIT_PREFIXES) + r")\s*",
    re.IGNORECASE,
)


def clean_screenplay_names(raw_name: str) -> str:
    """Clean screenplay nominee names from oscars.csv credit format.

    oscars.csv ceremony 73+ has names like:
    - "Written by Cameron Crowe"
    - "Screenplay by A and B; Story by C"
    - "Written for the Screen by X & Y"

    Returns comma-separated individual writer names, suitable for TMDb lookup
    and precursor matching.

    Example::

        >>> clean_screenplay_names("Written by Cameron Crowe")
        'Cameron Crowe'
        >>> clean_screenplay_names("Screenplay by A and B; Story by C")
        'A, B, C'
        >>> clean_screenplay_names("Written for the Screen by X & Y")
        'X, Y'
        >>> clean_screenplay_names("Robert Towne")
        'Robert Towne'
    """
    if not raw_name or not raw_name.strip():
        return raw_name

    # Split on ";" to separate credit blocks (e.g., "Screenplay by A; Story by B")
    blocks = [b.strip() for b in raw_name.split(";")]

    all_names: list[str] = []
    for block in blocks:
        # Strip credit prefixes ("Written by", "Screenplay by", etc.)
        cleaned = _PREFIX_PATTERN.sub("", block).strip()
        if not cleaned:
            continue

        # Split on " and " or " & " to get individual writers
        # Be careful not to split names like "Simon and Garfunkel" (unlikely for writers)
        parts = re.split(r"\s+and\s+|\s*&\s*", cleaned)
        for part in parts:
            part = part.strip().rstrip(",").strip()
            if part:
                all_names.append(part)

    if not all_names:
        # No prefixes found — return original (already a clean name)
        return raw_name.strip()

    return ", ".join(all_names)
