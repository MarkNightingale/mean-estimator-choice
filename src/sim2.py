"""Расширенная версия эксперимента (v2).
Добавлено: оценка Ходжеса – Лемана; распределения t(2) и логистическое; близкие выбросы (k = 3),
малая доля выбросов (1 %), выборки из 10 наблюдений.
"""
import numpy as np
from scipy import stats
import sim  # базовые оценки и признаки

IQR_N = sim.IQR_N

def draw_clean(dist, size, rng):
    if dist in ("normal", "t3", "laplace"):
        return sim.draw_clean(dist, size, rng)
    if dist == "t2":
        return rng.standard_t(2, size) * IQR_N / (2 * stats.t.ppf(0.75, 2))
    if dist == "logistic":
        return rng.logistic(0, 1, size) * IQR_N / (2 * stats.logistic.ppf(0.75))
    raise ValueError(dist)

def draw_samples(dist, n, eps, otype, k, R, rng):
    x = draw_clean(dist, (R, n), rng)
    if eps > 0:
        mask = rng.random((R, n)) < eps
        if otype == "sym":
            sign = rng.choice([-1.0, 1.0], size=(R, n))
        else:
            sign = np.repeat(rng.choice([-1.0, 1.0], size=(R, 1)), n, axis=1)
        x = np.where(mask, sign * k + rng.standard_normal((R, n)), x)
    return x

def est_hl(x, chunk=None):
    """Оценка Ходжеса – Лемана: медиана полусумм x_i + x_j (i <= j)."""
    R, n = x.shape
    i, j = np.triu_indices(n)
    if chunk is None:
        chunk = max(1, int(2.5e7 // len(i)))
    out = np.empty(R)
    for a in range(0, R, chunk):
        xb = x[a:a + chunk]
        out[a:a + chunk] = np.median((xb[:, i] + xb[:, j]) / 2, axis=1)
    return out

EST_NAMES = ["mean", "median", "trim10", "huber", "hl"]
EST_RU = {"mean": "Средняя", "median": "Медиана", "trim10": "Усечённая средняя (10 %)",
          "huber": "Оценка Хьюбера", "hl": "Оценка Ходжеса – Лемана"}

def all_estimates(x):
    return np.column_stack([sim.est_mean(x), sim.est_median(x), sim.est_trim(x), sim.est_huber(x), est_hl(x)])

features = sim.features
hogg_rule = sim.hogg_rule
FEAT_NAMES = sim.FEAT_NAMES

DISTS = ("normal", "t3", "laplace", "t2", "logistic")
def scenario_grid(dists=DISTS):
    grid = []
    for d in dists:
        for n in (10, 20, 50, 100, 300):
            grid.append((d, n, 0.0, "none", 0))
            for eps in (0.01, 0.05, 0.10, 0.20):
                for ot in ("sym", "one"):
                    for k in (3, 5, 10):
                        grid.append((d, n, eps, ot, k))
    return grid

BIG = 0.25  # «серьёзный промах»: ошибка больше четверти единицы масштаба

def err_metrics(e):
    """e — вектор ошибок оценки (theta = 0)."""
    return dict(mse=float((e ** 2).mean()), bias=float(e.mean()), mae=float(np.abs(e).mean()),
                big=float((np.abs(e) > BIG).mean()))
