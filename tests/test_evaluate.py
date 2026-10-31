import pytest

from src.evaluate import score


def test_perfect_match():
    s = score(["Guten Morgen , wie geht es dir ?"], ["Guten Morgen , wie geht es dir ?"])
    assert s["bleu"] == pytest.approx(100.0)
    assert s["chrf"] == pytest.approx(100.0)
    assert "13a" in s["bleu_sig"]


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        score(["a", "b"], ["a"])


def test_empty_hypothesis():
    s = score([""], ["Guten Morgen ."])
    assert s["bleu"] == 0.0
