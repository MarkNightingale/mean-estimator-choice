"""Проверки оценок, статистики Хогга и характеристик выборки."""
import numpy as np
import pytest

from estimator_choice import estimators as est


def row(x):
    return np.asarray(x, dtype=float)[None, :]


def test_mean_median_trimmed():
    x = row(list(range(1, 11)) + [1000])
    assert est.est_mean(x)[0] == pytest.approx(np.mean(x))
    assert est.est_median(x)[0] == 6
    # 10 % усечение 11 наблюдений отбрасывает по одному значению с каждой стороны
    assert est.est_trim(x)[0] == pytest.approx(np.mean(range(2, 11)))


def test_hodges_lehmann_small_example():
    # полусуммы (i <= j) для {0, 1, 10}: 0; 0,5; 5; 1; 5,5; 10 -> медиана 3
    assert est.est_hl(row([0, 1, 10]))[0] == pytest.approx(3.0)


def test_huber_resists_outliers_and_matches_center():
    rng = np.random.default_rng(1)
    clean = rng.standard_normal((200, 100))
    dirty = clean.copy()
    dirty[:, :10] += 50
    assert np.abs(est.est_huber(clean).mean()) < 0.03
    assert est.est_huber(dirty).mean() < 0.3 < est.est_mean(dirty).mean()


def test_estimators_are_location_scale_equivariant():
    rng = np.random.default_rng(2)
    x = rng.standard_t(3, (50, 40))
    E = est.all_estimates(x)
    E2 = est.all_estimates(3 * x + 7)
    np.testing.assert_allclose(E2, 3 * E + 7, rtol=1e-6, atol=1e-6)


def test_hogg_q_normal_reference_value():
    # для нормального распределения Q ≈ 2,59
    rng = np.random.default_rng(3)
    q = est.hogg_q(rng.standard_normal((20, 20000))).mean()
    assert q == pytest.approx(2.59, abs=0.03)


def test_hogg_rule_thresholds():
    np.testing.assert_array_equal(est.hogg_rule(np.array([2.0, 2.6, 3.0, 3.2, 4.0])), [0, 0, 2, 2, 1])


def test_features_invariant_to_location_scale_and_sign():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((30, 60)) + (rng.random((30, 60)) < 0.1) * 6
    F = est.features(x)
    np.testing.assert_allclose(est.features(5 * x - 2), F, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(est.features(-x), F, rtol=1e-6, atol=1e-9)


def test_clean_distributions_share_iqr():
    rng = np.random.default_rng(5)
    for d in est.DISTS:
        z = est.draw_clean(d, 400000, rng)
        iqr = np.subtract(*np.percentile(z, [75, 25]))
        assert iqr == pytest.approx(est.IQR_N, rel=0.02), d
