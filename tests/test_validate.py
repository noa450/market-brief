"""
Tests for validate.py — the critical safety component.

Includes a correct reference text and deliberately corrupted variants.
Each corruption must be detected and rejected.
"""

import pytest

from src.models import AnnotatedQuote, BriefData, CategorySummary, Quote
from src.rules import annotate_quote
from src.validate import ValidationResult, _extract_numbers, validate


# ─── Test fixtures ───

def _q(symbol, name_he, gender, category, last, change_pct, unit=None, decimals=2):
    return Quote(
        symbol=symbol, name_he=name_he, gender=gender,
        category=category, last=last, change_pct=change_pct,
        unit=unit, decimals=decimals,
    )


def _build_brief() -> BriefData:
    """Build BriefData matching the reference output in the spec."""
    europe = [
        _q("^GDAXI", "מדד הדאקס בפרנקפורט", "m", "europe", 18500, 0.7, decimals=0),
        _q("^FCHI", "מדד הקאק בפריז", "m", "europe", 7600, 0.6, decimals=0),
        _q("^FTSE", "מדד הפוטסי בלונדון", "m", "europe", 8200, 0.6, decimals=0),
        _q("^STOXX", "מדד ה-STOXX 600 הכלל-אירופי", "m", "europe", 510, 0.4, decimals=0),
    ]
    us = [
        _q("YM=F", "החוזים על מדד הדאו ג'ונס", "m", "us_futures", 39500, 0.3, decimals=0),
        _q("ES=F", "החוזים על ה-S&P 500", "m", "us_futures", 5400, -0.1, decimals=0),
        _q("NQ=F", 'החוזים על הנאסד"ק', "m", "us_futures", 18800, -0.9, decimals=0),
    ]
    commodities = [
        _q("BZ=F", "נפט מסוג ברנט", "m", "commodities", 85.9, -2.8, "דולר לחבית", 1),
        _q("CL=F", "החבית האמריקאית (WTI)", "f", "commodities", 80.5, -2.5, "דולר", 1),
        _q("GC=F", "הזהב", "m", "commodities", 4024, -1.3, "דולר לאונקיה", 0),
    ]
    crypto = [
        _q("BTC-USD", "הביטקוין", "m", "crypto", 63268, -2.5, "דולר", 0),
    ]

    def _cat(name, quotes):
        return CategorySummary(
            category=name,
            sentiment="positive" if all(q.change_pct > 0 for q in quotes) else
                      "negative" if all(q.change_pct < 0 for q in quotes) else "mixed",
            quotes=[annotate_quote(q) for q in quotes],
        )

    return BriefData(
        headline='שווקי העולם: אופטימיות באירופה, מגמה מעורבת בחוזים בארה"ב; הנפט והזהב יורדים',
        categories=[
            _cat("europe", europe),
            _cat("us_futures", us),
            _cat("commodities", commodities),
            _cat("crypto", crypto),
        ],
        generated_at="2025-01-15T10:00:00+00:00",
    )


# The reference correct output from the spec (section 9)
CORRECT_TEXT = (
    'שווקי העולם: אופטימיות באירופה, מגמה מעורבת בחוזים בארה"ב; הנפט והזהב יורדים\n'
    "\n"
    "אירופה ירוקה: המסחר בבורסות היבשת מתנהל בעליות שערים נאות. "
    "מדד הדאקס בפרנקפורט מוביל את המגמה עם קפיצה של כ-0.7%, "
    "מדד הקאק בפריז ומדד הפוטסי בלונדון מטפסים בכ-0.6% כל אחד, "
    "ומדד ה-STOXX 600 הכלל-אירופי מוסיף כ-0.4%.\n"
    "\n"
    'וול סטריט (חוזים עתידיים): בשווקים בארה"ב מסתמנת מגמה מעורבת לקראת פתיחת המסחר. '
    "החוזים על מדד הדאו ג'ונס מטפסים בכ-0.3%, "
    "בעוד החוזים על ה-S&P 500 נסוגים קלות בכ-0.1%. "
    'הלחץ מורגש בעיקר בסקטור הטכנולוגיה, כאשר החוזים על הנאסד"ק מאבדים כ-0.9%.\n'
    "\n"
    "סחורות ואנרגיה: מחירי הנפט רושמים ירידות חדות — "
    "נפט מסוג ברנט יורד בכ-2.8% למחיר של כ-85.9 דולר לחבית, "
    "והחבית האמריקאית (WTI) מאבדת כ-2.5% ל-80.5 דולר. "
    "גם הזהב נחלש בכ-1.3% ונסחר סביב רמה של 4,024 דולר לאונקיה.\n"
    "\n"
    'קריפטו: הביטקוין מאבד גובה ונסוג בכ-2.5%, כשהוא נסחר סביב רמה של 63,268 דולר.'
)


# ─── Number extraction tests ───


class TestNumberExtraction:
    def test_plain_number(self):
        assert 0.7 in _extract_numbers("כ-0.7%")

    def test_comma_separated(self):
        nums = _extract_numbers("4,024 דולר")
        assert 4024.0 in nums

    def test_large_comma(self):
        nums = _extract_numbers("63,268 דולר")
        assert 63268.0 in nums

    def test_multiple(self):
        nums = _extract_numbers("כ-0.7% וגם כ-85.9 דולר")
        assert 0.7 in nums
        assert 85.9 in nums

    def test_no_numbers(self):
        assert _extract_numbers("אין מספרים כאן") == []

    def test_percentage_without_prefix(self):
        nums = _extract_numbers("ירידה של 2.8%")
        assert 2.8 in nums


# ─── Correct text should pass ───


class TestCorrectText:
    def test_reference_output_passes(self):
        brief = _build_brief()
        result = validate(CORRECT_TEXT, brief)
        assert result.passed, f"Reference text should pass validation:\n{result.error_summary()}"


# ─── Deliberately corrupted texts ───


class TestFlippedSign:
    """A negative change described with a positive verb → reject."""

    def test_brent_up_instead_of_down(self):
        brief = _build_brief()
        # Brent has change_pct=-2.8, but we say it "מטפס" (climbs)
        corrupted = CORRECT_TEXT.replace(
            "נפט מסוג ברנט יורד בכ-2.8%",
            "נפט מסוג ברנט מטפס בכ-2.8%",
        )
        result = validate(corrupted, brief)
        assert not result.passed
        assert any(e.check == "direction" for e in result.errors)

    def test_dax_down_instead_of_up(self):
        brief = _build_brief()
        # DAX has change_pct=+0.7, but we say it "יורד" (drops)
        corrupted = CORRECT_TEXT.replace(
            "מדד הדאקס בפרנקפורט מוביל את המגמה עם קפיצה של כ-0.7%",
            "מדד הדאקס בפרנקפורט יורד בכ-0.7%",
        )
        result = validate(corrupted, brief)
        assert not result.passed
        assert any(e.check == "direction" for e in result.errors)


class TestInventedNumber:
    """A number that doesn't appear in the input → reject."""

    def test_invented_percentage(self):
        brief = _build_brief()
        # Replace 0.7% with 0.9% — no asset has 0.9% change in the input
        corrupted = CORRECT_TEXT.replace("כ-0.7%", "כ-0.8%")
        result = validate(corrupted, brief)
        assert not result.passed
        assert any(e.check == "membership" for e in result.errors)

    def test_invented_price(self):
        brief = _build_brief()
        # Replace 4,024 with 4,125 — doesn't match any last price
        corrupted = CORRECT_TEXT.replace("4,024", "4,125")
        result = validate(corrupted, brief)
        assert not result.passed
        assert any(e.check == "membership" for e in result.errors)


class TestDroppedAsset:
    """An asset from the input is missing in the output → reject."""

    def test_missing_bitcoin(self):
        brief = _build_brief()
        # Remove the entire crypto paragraph
        corrupted = CORRECT_TEXT.replace(
            'קריפטו: הביטקוין מאבד גובה ונסוג בכ-2.5%, כשהוא נסחר סביב רמה של 63,268 דולר.',
            "",
        )
        result = validate(corrupted, brief)
        assert not result.passed
        assert any(e.check == "completeness" for e in result.errors)

    def test_missing_gold(self):
        brief = _build_brief()
        corrupted = CORRECT_TEXT.replace("הזהב", "המתכת")
        result = validate(corrupted, brief)
        assert not result.passed
        assert any(e.check == "completeness" for e in result.errors)


class TestSubstitutedAssetName:
    """An asset name that doesn't appear in symbols.yaml → reject."""

    def test_hallucinated_name(self):
        brief = _build_brief()
        # Replace הזהב with הזכוכית (glass instead of gold) — the spec example
        corrupted = CORRECT_TEXT.replace("הזהב", "הזכוכית")
        result = validate(corrupted, brief)
        assert not result.passed
        # Should fail completeness (הזהב is missing)
        assert any(e.check == "completeness" for e in result.errors)


class TestLatinCharacters:
    """Unexpected Latin characters outside the whitelist → reject."""

    def test_english_word(self):
        brief = _build_brief()
        corrupted = CORRECT_TEXT + "\nThis is bullish for the market."
        result = validate(corrupted, brief)
        assert not result.passed
        assert any(e.check == "language" for e in result.errors)

    def test_whitelisted_tickers_allowed(self):
        """DAX, CAC, FTSE, STOXX, S&P 500, WTI should not trigger language check."""
        brief = _build_brief()
        result = validate(CORRECT_TEXT, brief)
        # These are all in the correct text and should not cause errors
        assert not any(e.check == "language" for e in result.errors)


class TestMultipleErrors:
    """Multiple issues in one text should all be reported."""

    def test_invented_number_and_dropped_asset(self):
        brief = _build_brief()
        corrupted = CORRECT_TEXT.replace("כ-0.7%", "כ-9.9%").replace(
            'קריפטו: הביטקוין מאבד גובה ונסוג בכ-2.5%, כשהוא נסחר סביב רמה של 63,268 דולר.',
            "",
        )
        result = validate(corrupted, brief)
        assert not result.passed
        checks = {e.check for e in result.errors}
        assert "membership" in checks
        assert "completeness" in checks
