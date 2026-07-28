"""Table-driven tests for rules.py — covers threshold boundaries and sentiment classification."""

import pytest

from src.models import Quote
from src.rules import (
    CATEGORY_ORDER,
    annotate_quote,
    apply_rules,
    build_headline,
    classify_intensity,
    classify_sentiment,
)


# ─── 6.1  Intensity threshold boundaries ───


@pytest.mark.parametrize(
    "change_pct, expected_direction",
    [
        (0.0,    "flat"),
        (0.10,   "flat"),
        (0.149,  "flat"),
        (-0.10,  "flat"),
        (-0.149, "flat"),
        (0.15,   "up"),
        (0.3,    "up"),
        (0.49,   "up"),
        (-0.15,  "down"),
        (-0.49,  "down"),
        (0.5,    "up"),
        (1.0,    "up"),
        (-0.5,   "down"),
        (-1.49,  "down"),
        (1.5,    "up"),
        (2.5,    "up"),
        (-1.5,   "down"),
        (-2.99,  "down"),
        (3.0,    "up"),
        (5.0,    "up"),
        (-3.0,   "down"),
        (-10.0,  "down"),
    ],
)
def test_classify_direction(change_pct: float, expected_direction: str):
    _, _, direction = classify_intensity(change_pct)
    assert direction == expected_direction


@pytest.mark.parametrize(
    "change_pct, expected_up_fragment",
    [
        (0.0,   "יציב"),
        (0.149, "יציב"),
        (0.15,  "עלייה קלה"),
        (0.49,  "עלייה קלה"),
        (0.5,   "עלייה נאה"),
        (1.49,  "עלייה נאה"),
        (1.5,   "קפיצה"),
        (2.99,  "קפיצה"),
        (3.0,   "זינוק חד"),
        (5.0,   "זינוק חד"),
    ],
)
def test_classify_intensity_up(change_pct: float, expected_up_fragment: str):
    up, _, _ = classify_intensity(change_pct)
    assert expected_up_fragment in up


@pytest.mark.parametrize(
    "change_pct, expected_down_fragment",
    [
        (0.0,    "יציב"),
        (-0.149, "יציב"),
        (-0.15,  "ירידה קלה"),
        (-0.49,  "ירידה קלה"),
        (-0.5,   "ירידה,"),
        (-1.49,  "ירידה,"),
        (-1.5,   "ירידה חדה"),
        (-2.99,  "ירידה חדה"),
        (-3.0,   "צניחה חדה"),
        (-10.0,  "צניחה חדה"),
    ],
)
def test_classify_intensity_down(change_pct: float, expected_down_fragment: str):
    _, down, _ = classify_intensity(change_pct)
    assert expected_down_fragment in down


# ─── 6.2  Category sentiment ───


def _make_quotes(changes: list[float], category: str = "europe") -> list[Quote]:
    return [
        Quote(
            symbol=f"SYM{i}",
            name_he=f"נכס {i}",
            gender="m",
            category=category,
            last=100.0,
            change_pct=c,
            unit=None,
            decimals=2,
        )
        for i, c in enumerate(changes)
    ]


@pytest.mark.parametrize(
    "changes, expected_sentiment",
    [
        # All positive, mean > 0.2 → positive
        ([0.5, 0.3, 0.4], "positive"),
        ([1.0, 2.0, 3.0], "positive"),
        # All negative, mean < -0.2 → negative
        ([-0.5, -0.3, -0.4], "negative"),
        ([-1.0, -2.0], "negative"),
        # Mixed signs → mixed
        ([0.5, -0.3], "mixed"),
        ([1.0, -0.1], "mixed"),
        # All positive but mean ≤ 0.2 → flat
        ([0.1, 0.1, 0.1], "flat"),
        ([0.0, 0.2, 0.1], "flat"),
        # All negative but mean ≥ -0.2 → flat
        ([-0.1, -0.1, -0.1], "flat"),
        # All zeros → flat
        ([0.0, 0.0], "flat"),
        # Edge: mean exactly 0.2 (not > 0.2) → flat
        ([0.2, 0.2], "flat"),
        # Edge: mean exactly -0.2 (not < -0.2) → flat
        ([-0.2, -0.2], "flat"),
    ],
)
def test_classify_sentiment(changes: list[float], expected_sentiment: str):
    quotes = _make_quotes(changes)
    assert classify_sentiment(quotes) == expected_sentiment


# ─── 6.3  Headline ───


def test_headline_contains_all_categories():
    """Headline should mention each category's sentiment."""
    quotes = (
        _make_quotes([0.5, 0.4], category="europe")
        + _make_quotes([-0.5, -0.3], category="us_futures")
    )
    brief = apply_rules(quotes)
    assert "שווקי העולם:" in brief.headline
    assert "אופטימיות" in brief.headline  # europe positive
    assert "פסימיות" in brief.headline     # us negative


def test_headline_notable_movers():
    """Assets with |change_pct| >= 1.5 should appear in headline."""
    quotes = _make_quotes([2.0], category="commodities")
    quotes[0] = Quote(
        symbol="BZ=F", name_he="נפט מסוג ברנט", gender="m",
        category="commodities", last=85.0, change_pct=-2.0, unit="דולר לחבית", decimals=1,
    )
    brief = apply_rules(quotes)
    assert "נפט מסוג ברנט" in brief.headline


def test_headline_no_notable_movers_below_threshold():
    """Assets with |change_pct| < 1.5 should NOT appear as notable movers."""
    quotes = _make_quotes([0.5, 0.3], category="europe")
    brief = apply_rules(quotes)
    # The headline should not have a semicolon (no movers clause)
    assert ";" not in brief.headline


# ─── 6.4  Ordering ───


def test_category_order():
    """Categories must appear in spec order: europe → us_futures → commodities → crypto."""
    quotes = (
        _make_quotes([0.5], category="crypto")
        + _make_quotes([0.5], category="europe")
        + _make_quotes([-0.5], category="commodities")
        + _make_quotes([0.3], category="us_futures")
    )
    brief = apply_rules(quotes)
    cat_names = [c.category for c in brief.categories]
    assert cat_names == ["europe", "us_futures", "commodities", "crypto"]


def test_empty_categories_skipped():
    """Categories with no quotes should not appear."""
    quotes = _make_quotes([0.5], category="europe")
    brief = apply_rules(quotes)
    cat_names = [c.category for c in brief.categories]
    assert cat_names == ["europe"]


def test_annotated_quote_formatting():
    """Verify format_change_pct and format_last produce correct strings."""
    q = Quote(
        symbol="GC=F", name_he="הזהב", gender="m",
        category="commodities", last=4024.0, change_pct=-1.3, unit="דולר לאונקיה", decimals=0,
    )
    aq = annotate_quote(q)
    assert aq.format_change_pct() == "1.3%"
    assert aq.format_last() == "4,024"
    assert aq.direction == "down"
    assert "ירידה" in aq.intensity


def test_categories_filter():
    """--categories flag should filter to only selected categories."""
    quotes = (
        _make_quotes([0.5], category="europe")
        + _make_quotes([0.5], category="crypto")
    )
    brief = apply_rules(quotes, categories_filter=["crypto"])
    cat_names = [c.category for c in brief.categories]
    assert cat_names == ["crypto"]


# ─── Full pipeline integration ───


def test_full_pipeline_minimum_categories():
    """apply_rules should work with at least 2 categories."""
    quotes = (
        _make_quotes([0.7, 0.6, 0.6, 0.4], category="europe")
        + _make_quotes([0.3, -0.1, -0.9], category="us_futures")
    )
    brief = apply_rules(quotes)
    assert len(brief.categories) == 2
    assert brief.headline.startswith("שווקי העולם:")
    assert brief.generated_at  # non-empty ISO timestamp
