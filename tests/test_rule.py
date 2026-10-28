"""Проверки правила из двух условий (src/choose.py)."""
import numpy as np
import pytest

from choose import THRESHOLDS, choose_estimator


def test_clean_normal_sample_gets_huber():
    x = np.random.default_rng(0).standard_normal(300)
    assert choose_estimator(x)["estimator"] == "huber"


def test_one_sided_outliers_get_median():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(200)
    x[:30] = 10 + rng.standard_normal(30)
    r = choose_estimator(x)
    assert r["estimator"] == "median"
    assert r["value"] == pytest.approx(np.median(x))


def test_obvious_typo_triggers_median():
    assert choose_estimator([10, 11, 12, 11, 10, 95])["estimator"] == "median"


def test_cautious_is_stricter_than_soft():
    a_c, b_c = THRESHOLDS["cautious"]
    a_s, b_s = THRESHOLDS["soft"]
    assert a_c < a_s and b_c < b_s


def test_ignores_missing_values_and_validates_input():
    r = choose_estimator([1.0, 2.0, np.nan, 3.0, 2.5])
    assert r["n"] == 4
    with pytest.raises(ValueError):
        choose_estimator([1.0, 2.0])
    with pytest.raises(ValueError):
        choose_estimator([1.0, 2.0, 3.0], variant="unknown")


def test_regional_wages_2025():
    import pandas as pd
    w = pd.read_csv("data/wages_2025.csv")["wage"]
    r = choose_estimator(w)
    assert r["n"] == 85 and r["estimator"] == "median" and r["value"] == 75525
