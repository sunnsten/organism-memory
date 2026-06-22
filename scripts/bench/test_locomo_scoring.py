import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.bench.locomo import _normalize, _token_f1


def test_iso_date_matches_natural():
    """ISO date should match natural-language date after normalization."""
    assert _token_f1("2023-05-08", "8 May 2023") >= 0.5


def test_iso_date_year_only():
    """Year-only answer should match itself."""
    assert _token_f1("2022", "2022") == 1.0


def test_natural_date_matches_iso():
    """Natural date should match ISO after normalization."""
    assert _token_f1("8 May 2023", "2023-05-08") >= 0.5


def test_plural_singular_match():
    """'books' and 'book' should match after lemmatization."""
    assert _token_f1("books", "book") >= 0.5


def test_mental_health_synonym():
    """'mental health' should match 'mental health support'."""
    assert _token_f1("mental health support", "mental health") >= 0.5


def test_gerund_normalization():
    """'running' and 'run' should match."""
    assert _token_f1("running", "run") >= 0.5
