"""Oscar market metadata — typed registry of all Kalshi Oscar tickers.

Re-exports from :mod:`.registry` for convenient imports::

    from oscar_prediction_market.trading.market_data import (
        OSCAR_MARKETS,
        CategoryCeremonyData,
        OscarMarketsRegistry,
    )
"""

from oscar_prediction_market.trading.market_data.registry import (
    OSCAR_MARKETS,
    CategoryCeremonyData,
    OscarMarketsRegistry,
    RawInventory,
)

__all__ = [
    "OSCAR_MARKETS",
    "CategoryCeremonyData",
    "OscarMarketsRegistry",
    "RawInventory",
]
