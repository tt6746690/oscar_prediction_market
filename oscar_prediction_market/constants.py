"""Project-wide path constants and API keys.

All resolved paths and shared configuration live here so that different
modules (data fetchers, modeling, etc.) import from a single source of truth.

Path resolution uses the package location to find the repo root, so everything
works regardless of the current working directory.

API keys are loaded from environment variables (populated via ``.env`` file
at the repo root). See ``.env`` for key descriptions and setup instructions.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Root Paths
# ============================================================================

# Repository root (two levels up from this file: constants.py -> oscar_prediction_market/ -> repo)
REPO_ROOT = Path(__file__).parent.parent

# Shared storage directory (experiment results, caches, artifacts)
STORAGE_DIR = REPO_ROOT / "storage"

# ============================================================================
# Cache Directories (under storage/ so worktrees can share via symlink)
# ============================================================================

CACHE_DIR = STORAGE_DIR / "cache"
TMDB_CACHE_DIR = CACHE_DIR / "tmdb"
TMDB_PERSON_CACHE_DIR = CACHE_DIR / "tmdb_person"
OMDB_CACHE_DIR = CACHE_DIR / "omdb"
PRECURSOR_AWARDS_CACHE_DIR = CACHE_DIR / "precursor_awards"

# ============================================================================
# API Keys — only needed for specific tasks (see README.md § API Keys)
# ============================================================================

# TMDb + OMDb: required only for building datasets from source.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")

# Kalshi: required only for fetching live market data or executing trades.
# Not needed for backtesting with cached market data.
#
# Credentials are read from files (RSA keys belong in files, not env vars).
# Override via env vars: KALSHI_ACCESS_KEY_FILE, KALSHI_PRIVATE_KEY_FILE
# Actual file reading happens lazily in KalshiClient.__init__, not at import time.
KALSHI_ACCESS_KEY_FILE = Path(os.environ.get("KALSHI_ACCESS_KEY_FILE", "")).expanduser()
KALSHI_PRIVATE_KEY_FILE = Path(os.environ.get("KALSHI_PRIVATE_KEY_FILE", "")).expanduser()
