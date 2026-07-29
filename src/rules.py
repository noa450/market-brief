from __future__ import annotations

from collections import defaultdict

from .models import AnnotatedQuote, BriefData, CategorySummary, Quote

# ---------- 6.1  Magnitude → Hebrew intensity ----------

INTENSITY_TABLE: list[tuple[float, str, str]] = [
    # (upper_bound_exclusive, up_word, down_word)
    (0.15, "יציב / כמעט ללא שינוי", "יציב / כמעט ללא שינוי"),
    (0.5,  "עלייה קלה, מוסיף",      "ירידה קלה, נסוג קלות"),
    (1.5,  "עלייה נאה, מטפס",       "ירידה, מאבד גובה"),
    (3.0,  "קפיצה, זינוק",          "ירידה חדה, צניחה"),
]

INTENSITY_MAX_UP = "זינוק חד, ראלי"
INTENSITY_MAX_DOWN = "צניחה חדה, קריסה"


def classify_intensity(change_pct: float) -> tuple[str, str, str]:
    """Return (intensity_up, intensity_down, direction)."""
    abs_change = abs(change_pct)

    if abs_change < 0.15:
        direction = "flat"
    elif change_pct > 0:
        direction = "up"
    else:
        direction = "down"

    for upper, up_word, down_word in INTENSITY_TABLE:
        if abs_change < upper:
            return up_word, down_word, direction

    return INTENSITY_MAX_UP, INTENSITY_MAX_DOWN, direction


def annotate_quote(quote: Quote) -> AnnotatedQuote:
    """Enrich a Quote with deterministic rule outputs."""
    up, down, direction = classify_intensity(quote.change_pct)
    return AnnotatedQuote(
        quote=quote,
        intensity_up=up,
        intensity_down=down,
        direction=direction,
    )


# ---------- 6.2  Category sentiment ----------

CATEGORY_ORDER = ["europe", "us", "us_futures", "commodities", "crypto"]


def classify_sentiment(quotes: list[Quote]) -> str:
    """Classify category sentiment from its quotes' change_pct values."""
    if not quotes:
        return "flat"

    DEAD_ZONE = 0.05  # treat ±0.05% as zero

    changes = [q.change_pct for q in quotes]
    mean = sum(changes) / len(changes)

    # Clamp tiny moves to zero for sign classification
    clamped = [0.0 if abs(c) < DEAD_ZONE else c for c in changes]

    all_positive = all(c >= 0 for c in clamped)
    all_negative = all(c <= 0 for c in clamped)

    if all_positive and mean > 0.2:
        return "positive"
    if all_negative and mean < -0.2:
        return "negative"

    has_positive = any(c > 0 for c in clamped)
    has_negative = any(c < 0 for c in clamped)
    if has_positive and has_negative:
        return "mixed"

    return "flat"


# ---------- 6.3  Headline ----------

CATEGORY_LABELS = {
    "europe": "באירופה",
    "us": "בוול סטריט",
    "us_futures": "בחוזים בארה\"ב",
    "commodities": "בסחורות",
    "crypto": "בקריפטו",
}

SENTIMENT_WORDS = {
    "positive": "אופטימיות",
    "negative": "פסימיות",
    "mixed": "מגמה מעורבת",
    "flat": "יציבות",
}


def _notable_movers(all_quotes: list[Quote]) -> list[str]:
    """Return Hebrew names of assets exceeding 1.5% absolute move."""
    movers = []
    for q in all_quotes:
        if abs(q.change_pct) >= 1.5:
            # Determine short display name for headline
            direction = "עולה" if q.change_pct > 0 else "יורד" if q.gender == "m" else "יורדת"
            # Use a shorter name form for the headline
            name = q.name_he
            # Strip leading ה if present for more natural headline phrasing
            if name.startswith("ה"):
                name = name  # keep as-is; Hebrew convention
            movers.append(f"{name} {direction}")
    return movers


def build_headline(categories: list[CategorySummary]) -> str:
    """Build the deterministic headline string."""
    clauses = []
    for cat in categories:
        label = CATEGORY_LABELS.get(cat.category, cat.category)
        word = SENTIMENT_WORDS[cat.sentiment]
        clauses.append(f"{word} {label}")

    # Gather all quotes for notable movers
    all_quotes = [aq.quote for cat in categories for aq in cat.quotes]
    movers = _notable_movers(all_quotes)

    headline = "שווקי העולם: "
    if len(clauses) >= 2:
        headline += ", ".join(clauses[:-1]) + ", " + clauses[-1]
    elif clauses:
        headline += clauses[0]

    if movers:
        headline += "; " + ", ".join(movers)

    return headline


# ---------- 6.4  Full pipeline ----------

def apply_rules(quotes: list[Quote], categories_filter: list[str] | None = None) -> BriefData:
    """Run the full rules layer: annotate, group, classify, headline."""
    from datetime import datetime, timezone

    # Group by category
    by_cat: dict[str, list[Quote]] = defaultdict(list)
    for q in quotes:
        if categories_filter and q.category not in categories_filter:
            continue
        by_cat[q.category].append(q)

    # Build CategorySummary in spec order, skip empty
    category_summaries: list[CategorySummary] = []
    for cat_name in CATEGORY_ORDER:
        cat_quotes = by_cat.get(cat_name)
        if not cat_quotes:
            continue
        sentiment = classify_sentiment(cat_quotes)
        annotated = [annotate_quote(q) for q in cat_quotes]
        category_summaries.append(CategorySummary(
            category=cat_name,
            sentiment=sentiment,
            quotes=annotated,
        ))

    headline = build_headline(category_summaries)

    return BriefData(
        headline=headline,
        categories=category_summaries,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
