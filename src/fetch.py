from __future__ import annotations

import logging
import re
import time
from typing import Protocol, runtime_checkable
from urllib.parse import quote as url_quote

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


@runtime_checkable
class Provider(Protocol):
    """Fetch raw market data for a list of symbols."""

    def fetch(self, symbols: list[str]) -> dict[str, RawQuote]:
        """Return {symbol: RawQuote} for every symbol that could be fetched."""
        ...


class RawQuote:
    """Minimal container for raw provider data before normalization."""

    __slots__ = ("symbol", "last", "change_pct")

    def __init__(self, symbol: str, last: float, change_pct: float) -> None:
        self.symbol = symbol
        self.last = last
        self.change_pct = change_pct

    def __repr__(self) -> str:
        return f"RawQuote({self.symbol!r}, last={self.last}, change_pct={self.change_pct})"


class YahooProvider:
    """Fetches quotes via the yfinance library."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, dict[str, RawQuote]]] = {}
        self._cache_ttl = 60  # seconds

    def fetch(self, symbols: list[str]) -> dict[str, RawQuote]:
        now = time.time()

        # Check cache
        cache_key = ",".join(sorted(symbols))
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                logger.debug("Returning cached data (age %.1fs)", now - ts)
                return data

        result: dict[str, RawQuote] = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                last = float(info.last_price)
                prev_close = float(info.previous_close)

                if prev_close == 0:
                    logger.warning("Previous close is 0 for %s, skipping", symbol)
                    continue

                change_pct = ((last - prev_close) / prev_close) * 100
                result[symbol] = RawQuote(symbol, last, change_pct)
                logger.debug("Fetched %s: last=%.4f change=%.4f%%", symbol, last, change_pct)

            except Exception:
                logger.warning("Failed to fetch %s, omitting", symbol, exc_info=True)

        self._cache[cache_key] = (now, result)
        return result


class CnbcProvider:
    """Fetches quotes via the CNBC quote.cnbc.com JSON API."""

    API_URL = (
        "https://quote.cnbc.com/quote-html-webservice/restQuote/symbolType/symbol"
    )

    # Map Yahoo symbols (used in symbols.yaml) → CNBC symbols
    YAHOO_TO_CNBC = {
        "^GDAXI": ".GDAXI",
        "^FCHI": ".FCHI",
        "^FTSE": ".FTSE",
        "^STOXX": ".STOXX",
        "^DJI": ".DJI",
        "^GSPC": ".SPX",
        "^IXIC": ".IXIC",
        "YM=F": "@DJ.1",
        "ES=F": "@SP.1",
        "NQ=F": "@ND.1",
        "BZ=F": "@LCO.1",
        "CL=F": "@CL.1",
        "NG=F": "@NG.1",
        "GC=F": "@GC.1",
        "BTC-USD": "BTC.CM=",
    }

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, dict[str, RawQuote]]] = {}
        self._cache_ttl = 60  # seconds

    def fetch(self, symbols: list[str]) -> dict[str, RawQuote]:
        now = time.time()

        cache_key = ",".join(sorted(symbols))
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                logger.debug("CNBC: returning cached data (age %.1fs)", now - ts)
                return data

        # Build reverse map: cnbc_symbol → yahoo_symbol
        cnbc_to_yahoo: dict[str, str] = {}
        cnbc_symbols: list[str] = []
        for yahoo_sym in symbols:
            cnbc_sym = self.YAHOO_TO_CNBC.get(yahoo_sym)
            if cnbc_sym is None:
                logger.warning("No CNBC mapping for %s, skipping", yahoo_sym)
                continue
            cnbc_symbols.append(cnbc_sym)
            cnbc_to_yahoo[cnbc_sym] = yahoo_sym

        if not cnbc_symbols:
            logger.error("No valid CNBC symbols to fetch")
            return {}

        # Single batch request with pipe-separated symbols
        params = {
            "symbols": "|".join(cnbc_symbols),
            "requestMethod": "itv",
            "noform": "1",
            "partnerId": "2",
            "fund": "1",
            "exthrs": "1",
            "output": "json",
        }

        try:
            resp = requests.get(
                self.API_URL,
                params=params,
                headers={"User-Agent": self.USER_AGENT},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("CNBC API request failed", exc_info=True)
            return {}

        result: dict[str, RawQuote] = {}

        quotes_raw = data.get("FormattedQuoteResult", {}).get("FormattedQuote", [])
        # API returns a single dict instead of a list when there's only one result
        if isinstance(quotes_raw, dict):
            quotes_raw = [quotes_raw]

        for q in quotes_raw:
            cnbc_sym = q.get("symbol", "")
            yahoo_sym = cnbc_to_yahoo.get(cnbc_sym)
            if yahoo_sym is None:
                logger.debug("CNBC returned unexpected symbol %s, ignoring", cnbc_sym)
                continue

            try:
                # Parse last price: "25,468.26" → 25468.26
                last_str = q.get("last", "").replace(",", "")
                last = float(last_str)

                # Parse change_pct: "+0.42%" or "-1.82%" or "UNCH" → float
                pct_str = q.get("change_pct", "").strip()
                if pct_str.upper() == "UNCH":
                    change_pct = 0.0
                else:
                    change_pct = float(pct_str.replace("%", "").replace("+", ""))

                result[yahoo_sym] = RawQuote(yahoo_sym, last, change_pct)
                logger.debug("CNBC fetched %s (%s): last=%.4f change=%.4f%%",
                             yahoo_sym, cnbc_sym, last, change_pct)

            except (ValueError, TypeError):
                logger.warning("CNBC: failed to parse quote for %s (%s), skipping",
                               yahoo_sym, cnbc_sym, exc_info=True)

        self._cache[cache_key] = (now, result)
        return result
