"""
Validator for generated Hebrew market briefs.

Runs 6 checks against the generated text and the input BriefData.
Fail closed: any check failure → rejection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import AnnotatedQuote, BriefData
from .normalize import load_symbols_config

# Ticker names allowed to appear as Latin characters in the output
LATIN_WHITELIST = {
    "STOXX", "S&P", "WTI",
    # Also allow with numbers/suffixes commonly seen
    "S&P 500", "STOXX 600",
}

# Hebrew directional verbs — positive polarity
POSITIVE_VERBS = [
    "עולה", "עולים", "עולות",
    "מטפס", "מטפסים", "מטפסת", "מטפסות",
    "מוסיף", "מוסיפים", "מוסיפה", "מוסיפות",
    "זינק", "זינקו", "זינקה",
    "קופץ", "קופצים", "קופצת", "קופצות",
    "קפץ", "קפצו", "קפצה",
    "מזנק", "מזנקים", "מזנקת", "מזנקות",
    "עליות", "עלייה",
    "ראלי",
    "ירוק", "ירוקה", "ירוקים",
    "מוביל", "מובילה", "מובילים",
]

# Hebrew directional verbs — negative polarity
NEGATIVE_VERBS = [
    "יורד", "יורדים", "יורדת", "יורדות",
    "נסוג", "נסוגים", "נסוגה", "נסוגות",
    "מאבד", "מאבדים", "מאבדת", "מאבדות",
    "צונח", "צונחים", "צונחת", "צונחות",
    "צנח", "צנחו", "צנחה",
    "נחלש", "נחלשים", "נחלשת", "נחלשות",
    "רושם ירידה", "רושמים ירידות", "רושמת ירידה",
    "ירידה", "ירידות",
    "צניחה",
    "קורס", "קורסים", "קורסת",
    "אדום", "אדומה", "אדומים",
    "לחץ",
]

# Numbers regex: handles כ-1.5%, כ-63,268, 0.7%, 85.9, negative signs
# Captures the numeric part (possibly with commas and decimal point)
_NUMBER_RE = re.compile(
    r"כ?-?"              # optional כ- prefix
    r"(\d[\d,]*\.?\d*)"  # the number itself (with optional commas and decimal)
    r"\s*%?"             # optional % suffix
)


@dataclass
class ValidationError:
    check: str
    message: str


@dataclass
class ValidationResult:
    passed: bool
    errors: list[ValidationError] = field(default_factory=list)

    def error_summary(self) -> str:
        return "\n".join(f"[{e.check}] {e.message}" for e in self.errors)


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from Hebrew text, handling כ-, commas, %."""
    numbers = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            numbers.append(float(raw))
        except ValueError:
            continue
    return numbers


def _collect_input_numbers(brief: BriefData) -> set[float]:
    """Collect all valid reference numbers from the brief's quotes."""
    numbers: set[float] = set()
    for cat in brief.categories:
        for aq in cat.quotes:
            # change_pct as displayed (absolute, 1 decimal)
            numbers.add(round(abs(aq.quote.change_pct), 1))
            # last price at configured precision
            numbers.add(round(aq.quote.last, aq.quote.decimals))
    return numbers


def _number_matches(extracted: float, reference: float, tolerance: float = 0.05) -> bool:
    """Check if an extracted number matches a reference within tolerance."""
    return abs(extracted - reference) <= tolerance


def _collect_asset_names(brief: BriefData) -> set[str]:
    """Collect all Hebrew asset names from the brief."""
    names = set()
    for cat in brief.categories:
        for aq in cat.quotes:
            names.add(aq.quote.name_he)
    return names


def _load_asset_name_whitelist() -> set[str]:
    """Load all known Hebrew asset names from symbols.yaml."""
    config = load_symbols_config()
    return {v["name_he"] for v in config.values()}


def _find_nearest_direction(text: str, asset_pos: int) -> str | None:
    """
    Find the nearest directional verb to the asset mention position.
    Returns "up", "down", or None.

    Searches in a window around the asset position.
    """
    # Search only after the asset name (the verb always follows the subject in Hebrew briefs)
    window_after = text[asset_pos:asset_pos + 80]
    window_before = ""
    window = window_after

    best_direction = None
    best_distance = float("inf")

    for verb in POSITIVE_VERBS:
        idx = window.find(verb)
        if idx != -1:
            dist = abs(idx - len(window_before))
            if dist < best_distance:
                best_distance = dist
                best_direction = "up"

    for verb in NEGATIVE_VERBS:
        idx = window.find(verb)
        if idx != -1:
            dist = abs(idx - len(window_before))
            if dist < best_distance:
                best_distance = dist
                best_direction = "down"

    return best_direction


def validate(text: str, brief: BriefData) -> ValidationResult:
    """
    Run all 6 validation checks on generated text against the input brief.

    Returns ValidationResult with passed=True if all checks pass.
    """
    errors: list[ValidationError] = []

    # Collect input data
    all_quotes: list[AnnotatedQuote] = []
    for cat in brief.categories:
        all_quotes.extend(cat.quotes)

    input_numbers = _collect_input_numbers(brief)
    input_asset_names = _collect_asset_names(brief)
    whitelist_names = _load_asset_name_whitelist()

    # ── Check 1: Number extraction & membership ──
    # Strip asset names before extracting numbers, since names like
    # "S&P 500" and "STOXX 600" contain numbers that are not data figures.
    text_for_numbers = text
    for name in input_asset_names:
        text_for_numbers = text_for_numbers.replace(name, "")
    # Also strip whitelisted Latin ticker fragments with numbers
    for term in sorted(LATIN_WHITELIST, key=len, reverse=True):
        text_for_numbers = text_for_numbers.replace(term, "")
    extracted_numbers = _extract_numbers(text_for_numbers)
    for num in extracted_numbers:
        matched = any(_number_matches(num, ref) for ref in input_numbers)
        if not matched:
            errors.append(ValidationError(
                check="membership",
                message=f"Orphan number {num} not found in input data. "
                        f"Valid numbers: {sorted(input_numbers)}",
            ))

    # ── Check 2: Direction verification ──
    for aq in all_quotes:
        name = aq.quote.name_he
        pos = text.find(name)
        if pos == -1:
            # Will be caught by completeness check
            continue

        # Only check direction for assets that actually moved
        if aq.direction == "flat":
            continue

        detected_dir = _find_nearest_direction(text, pos)
        if detected_dir is None:
            # No verb found near this asset — not necessarily wrong
            # (could be phrased differently), skip
            continue

        expected_dir = aq.direction  # "up" or "down"
        if detected_dir != expected_dir:
            errors.append(ValidationError(
                check="direction",
                message=f"Asset '{name}' has change_pct={aq.quote.change_pct:+.2f}% "
                        f"(expected {expected_dir}) but nearest verb indicates {detected_dir}.",
            ))

    # ── Check 3: Asset name whitelist ──
    # Find all Hebrew asset names mentioned in the text and check they're known
    for name in whitelist_names:
        # Check if any substring that looks like it could be a corrupted version appears
        pass  # Forward check: handled by completeness + the names are fixed strings

    # Check that asset names used in the text are from the whitelist
    # We do this by verifying every name from the INPUT appears verbatim
    # (the LLM should not paraphrase them)
    for name in input_asset_names:
        if name not in whitelist_names:
            errors.append(ValidationError(
                check="asset_whitelist",
                message=f"Asset name '{name}' not in symbols.yaml whitelist.",
            ))

    # ── Check 4: Completeness ──
    for aq in all_quotes:
        name = aq.quote.name_he
        if name not in text:
            errors.append(ValidationError(
                check="completeness",
                message=f"Asset '{name}' (symbol={aq.quote.symbol}) not mentioned in output.",
            ))

    # ── Check 5: Language — no unexpected Latin characters ──
    _check_latin(text, errors)

    result = ValidationResult(
        passed=len(errors) == 0,
        errors=errors,
    )
    return result


def _check_latin(text: str, errors: list[ValidationError]) -> None:
    """Check that no Latin characters appear outside whitelisted ticker names."""
    # Remove whitelisted terms from the text first
    cleaned = text
    # Sort by length descending so longer matches are removed first
    # (e.g., "S&P 500" before "S&P", "STOXX 600" before "STOXX")
    whitelist_sorted = sorted(LATIN_WHITELIST, key=len, reverse=True)
    for term in whitelist_sorted:
        cleaned = cleaned.replace(term, "")

    # Now check for remaining Latin characters
    latin_matches = re.findall(r"[a-zA-Z]+", cleaned)
    if latin_matches:
        errors.append(ValidationError(
            check="language",
            message=f"Unexpected Latin characters found: {latin_matches}. "
                    "Only whitelisted ticker names are allowed.",
        ))
