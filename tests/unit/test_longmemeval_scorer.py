import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))  # add repo root to path

import pytest
from scripts.bench.longmemeval import token_f1, score_response


def test_exact_match_is_1():
    assert token_f1("matcha latte", "matcha latte") == pytest.approx(1.0)


def test_partial_overlap():
    f1 = token_f1("matcha latte no sugar", "latte")
    assert 0.0 < f1 < 1.0


def test_no_overlap_is_0():
    assert token_f1("cappuccino", "matcha") == 0.0


def test_empty_predicted_is_0():
    assert token_f1("", "matcha latte") == 0.0


def test_score_response_pass():
    result = score_response(predicted="matcha latte", ground_truth="matcha latte")
    assert result["f1"] == pytest.approx(1.0)
    assert result["pass"] is True


def test_score_response_fail():
    result = score_response(predicted="I don't know", ground_truth="matcha latte")
    assert result["pass"] is False


def test_score_response_multi_answer_takes_best():
    """run_instance picks the best F1 across multiple ground truths."""
    from scripts.bench.longmemeval import score_response
    # Simulate the max() logic from run_instance
    ground_truths = ["matcha latte", "latte"]
    predicted = "matcha latte"
    best = max(
        (score_response(predicted, gt) for gt in ground_truths),
        key=lambda r: r["f1"],
    )
    assert best["f1"] == pytest.approx(1.0)
    assert best["ground_truth"] == "matcha latte"
