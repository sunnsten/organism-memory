from __future__ import annotations

import pytest
from unittest.mock import patch

from organism.eval.runner.matchers import normalize, contains_all, contains_none


# ---------------------------------------------------------------------------
# normalize() — thousands separator stripping
# ---------------------------------------------------------------------------

class TestNormalizeThousandsSeparators:
    """normalize() must strip thousands separators so 1,300 and 1300 match."""

    def test_comma_separator_en(self):
        result = normalize("€1,300/month")
        assert "1300" in result

    def test_dot_separator_de(self):
        result = normalize("1.300 EUR")
        assert "1300" in result

    def test_no_separator(self):
        result = normalize("budget 1300 EUR")
        assert "1300" in result

    def test_comma_mixed_ru(self):
        result = normalize("€1,300/mo")
        assert "1300" in result

    def test_large_number(self):
        # 1,300,000 → 1300000
        result = normalize("1,300,000 residents")
        assert "1300" in result

    def test_non_thousands_decimal_untouched(self):
        # 1,5 has only 1 digit after comma — NOT a thousands separator, must not strip
        result = normalize("coefficient 1,5")
        # "1" should still appear (number is split/modified but comma-stripped only for 3-digit groups)
        assert "1" in result
        # The result should NOT turn "1,5" into "15"
        assert "15" not in result

    def test_plain_decimal_with_two_digits(self):
        # 1,50 — 2 digits after comma, not a thousands separator
        result = normalize("price 1,50 EUR")
        assert "150" not in result


# ---------------------------------------------------------------------------
# normalize() — fallback mode (no NLP libs)
# ---------------------------------------------------------------------------

class TestNormalizeFallback:
    """When HAS_NLP_LIBS is False, basic normalization must still work."""

    def test_fallback_lowercase(self):
        with patch("organism.eval.runner.matchers.HAS_NLP_LIBS", False):
            result = normalize("Hello World")
        assert result == "hello world"

    def test_fallback_yo_to_ye(self):
        with patch("organism.eval.runner.matchers.HAS_NLP_LIBS", False):
            result = normalize("ёжик")
        assert result == "ежик"

    def test_fallback_thousands_separator(self):
        # Thousands stripping happens BEFORE the HAS_NLP_LIBS branch
        with patch("organism.eval.runner.matchers.HAS_NLP_LIBS", False):
            result = normalize("€1,300/month")
        assert "1300" in result

    def test_fallback_whitespace_collapse(self):
        with patch("organism.eval.runner.matchers.HAS_NLP_LIBS", False):
            result = normalize("  too   many   spaces  ")
        assert result == "too many spaces"


# ---------------------------------------------------------------------------
# contains_all() — number format variants
# ---------------------------------------------------------------------------

class TestContainsAllNumbers:
    """contains_all() must match numbers regardless of thousands separator format."""

    def test_en_format_comma(self):
        assert contains_all("Your budget is €1,300/month", ["1300"])

    def test_ru_format_no_separator(self):
        assert contains_all("Your budget 1300 EUR per month", ["1300"])

    def test_de_format_dot(self):
        assert contains_all("Budget €1.300 pro Monat", ["1300"])

    def test_wrong_number_fails(self):
        assert not contains_all("budget 1400", ["1300"])

    def test_multiple_keywords(self):
        assert contains_all("Donaustadt district, budget €1,300/month", ["Donaustadt", "1300"])

    def test_keyword_missing(self):
        assert not contains_all("Donaustadt district", ["Donaustadt", "1300"])

    def test_empty_keywords(self):
        assert contains_all("any text", [])

    def test_empty_text(self):
        assert not contains_all("", ["1300"])


# ---------------------------------------------------------------------------
# contains_none() — exclusion checks
# ---------------------------------------------------------------------------

class TestContainsNone:
    """contains_none() returns True when none of the tokens are in the text."""

    def test_absent_number(self):
        assert contains_none("budget €1,300/month", ["1500"])

    def test_present_number(self):
        assert not contains_none("old budget 1500 EUR", ["1500"])

    def test_present_number_with_separator(self):
        # "1,500" in text, checking for "1500" — should detect it
        assert not contains_none("budget €1,500/month", ["1500"])

    def test_multiple_tokens_all_absent(self):
        assert contains_none("Donaustadt, 1300 EUR", ["1500", "Neubau"])

    def test_multiple_tokens_one_present(self):
        assert not contains_none("Donaustadt, 1300 EUR, Neubau mentioned", ["1500", "Neubau"])

    def test_empty_tokens(self):
        assert contains_none("any text", [])


# ---------------------------------------------------------------------------
# contains_all() — Russian morphology (requires pymorphy2)
# ---------------------------------------------------------------------------

try:
    from pymorphy2 import MorphAnalyzer
    MorphAnalyzer()  # fails on Python 3.13 (inspect.getargspec removed)
    HAS_PYMORPHY2 = True
except Exception:
    HAS_PYMORPHY2 = False


@pytest.mark.skipif(not HAS_PYMORPHY2, reason="pymorphy2 not installed")
class TestContainsAllRussianMorphology:
    """Russian inflected forms should match their nominative keyword."""

    def test_state_school_genitive(self):
        # "государственной школе" should match keyword "государственная"
        assert contains_all("государственной школе", ["государственная"])

    def test_state_school_full_phrase(self):
        assert contains_all(
            "австрийской государственной школы с курсом немецкого",
            ["государственная", "школ"],
        )

    def test_vienna_in_different_cases(self):
        # "Вену" (accusative) should match "Вена" (nominative keyword)
        assert contains_all("переезд в Вену", ["Вена"])

    def test_rent_forms(self):
        # "аренду" should match root "аренд"
        assert contains_all("стоимость аренды квартиры", ["аренд"])

    def test_donaustadt_unchanged(self):
        # Proper nouns / unknown words pass through unchanged
        assert contains_all("Район Донауштадт (22-й)", ["Донауштадт"])
