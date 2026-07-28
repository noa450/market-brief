from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Quote:
    symbol: str
    name_he: str          # "מדד הדאקס בפרנקפורט"
    gender: str           # "m" | "f"
    category: str         # europe | us_futures | commodities | crypto
    last: float
    change_pct: float     # signed
    unit: str | None      # "דולר לחבית" | "דולר לאונקיה" | None
    decimals: int         # display precision


@dataclass
class AnnotatedQuote:
    """Quote enriched with deterministic rule outputs."""
    quote: Quote
    intensity_up: str      # Hebrew intensity word for positive move
    intensity_down: str    # Hebrew intensity word for negative move
    direction: str         # "up" | "down" | "flat"

    @property
    def intensity(self) -> str:
        if self.direction == "up":
            return self.intensity_up
        elif self.direction == "down":
            return self.intensity_down
        return "יציב / כמעט ללא שינוי"

    def format_change_pct(self) -> str:
        return f"{abs(self.quote.change_pct):.1f}%"

    def format_last(self) -> str:
        val = self.quote.last
        decimals = self.quote.decimals
        if decimals == 0:
            formatted = f"{int(round(val)):,}"
        else:
            formatted = f"{val:,.{decimals}f}"
        return formatted


@dataclass
class CategorySummary:
    category: str
    sentiment: str         # positive | negative | mixed | flat
    quotes: list[AnnotatedQuote]


@dataclass
class BriefData:
    """Complete pre-LLM data package."""
    headline: str
    categories: list[CategorySummary]
    generated_at: str      # ISO 8601
