from __future__ import annotations

import logging
from pathlib import Path

import yaml

from .fetch import RawQuote
from .models import Quote

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_symbols_config(path: Path | None = None) -> dict:
    """Load symbols.yaml and return the parsed dict."""
    if path is None:
        path = CONFIG_DIR / "symbols.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize(raw_quotes: dict[str, RawQuote], symbols_config: dict | None = None) -> list[Quote]:
    """Convert raw provider data into Quote objects using symbols.yaml metadata."""
    if symbols_config is None:
        symbols_config = load_symbols_config()

    quotes: list[Quote] = []

    for symbol, raw in raw_quotes.items():
        cfg = symbols_config.get(symbol)
        if cfg is None:
            logger.warning("Symbol %s not in symbols.yaml, skipping", symbol)
            continue

        quote = Quote(
            symbol=symbol,
            name_he=cfg["name_he"],
            gender=cfg["gender"],
            category=cfg["category"],
            last=raw.last,
            change_pct=round(raw.change_pct, 4),
            unit=cfg.get("unit"),
            decimals=cfg.get("decimals", 2),
        )
        quotes.append(quote)

    return quotes
