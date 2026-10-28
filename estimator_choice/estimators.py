"""Оценки типичного уровня, статистика Хогга, характеристики выборки и генерация засорённых выборок.

Все функции векторизованы по строкам: на вход подаётся массив формы (R, n) —
R выборок по n наблюдений, на выходе — массив длины R.

Модель засорения Тьюки – Хьюбера: X ~ (1 − ε)·F(x − θ) + ε·G, θ = 0.
Все «чистые» распределения F нормированы к межквартильному размаху стандартного
нормального (около 1,349), поэтому единица масштаба для них одна и та же.
"""
import numpy as np
from scipy import stats

IQR_N = 2 * stats.norm.ppf(0.75)  # 1,349 — межквартильный размах N(0, 1)
MAD_SCALE = 0.6745                 # MAD / 0,6745 = MAD × 1,483 совпадает с σ для нормального

DISTS = ("normal", "t3", "laplace", "t2", "logistic")
EST_NAMES = ["mean", "median", "trim10", "huber", "hl"]
EST_RU = {"mean": "Средняя", "median": "Медиана", "trim10": "Усечённая средняя (10 %)",
          "huber": "Оценка Хьюбера", "hl": "Оценка Ходжеса – Лемана"}
FEAT_NAMES = ["log n", "|асимметрия|", "эксцесс", "Q Хогга", "SD/MAD", "доля за «усами»", "|средняя − медиана|/MAD"]
BIG = 0.25  # «серьёзный промах»: ошибка больше четверти единицы масштаба


# ---------- генерация выборок ----------
def draw_clean(dist, size, rng):
    """«Чистое» распределение с межквартильным размахом IQR_N."""
    if dist == "normal":
        return rng.standard_normal(size)
    if dist == "t3":
        return rng.standard_t(3, size) * (IQR_N / (2 * stats.t.ppf(0.75, 3)))
    if dist == "laplace":
        return rng.laplace(0, 1, size) * (IQR_N / (2 * stats.laplace.ppf(0.75)))
    if dist == "t2":
        return rng.standard_t(2, size) * IQR_N / (2 * stats.t.ppf(0.75, 2))
    if dist == "logistic":
        return rng.logistic(0, 1, size) * IQR_N / (2 * stats.logistic.ppf(0.75))
    raise ValueError(dist)


def draw_samples(dist, n, eps, otype, k, R, rng):
    """R выборок объёма n: доля eps наблюдений заменяется выбросами sign·k + N(0, 1).

    otype = "sym" — знак случаен для каждого наблюдения; "one" — все выбросы выборки
    по одну сторону (сторона случайна для каждой выборки).
    """
    x = draw_clean(dist, (R, n), rng)
    if eps > 0:
        mask = rng.random((R, n)) < eps
        if otype == "sym":
            sign = rng.choice([-1.0, 1.0], size=(R, n))
        else:
            sign = np.repeat(rng.choice([-1.0, 1.0], size=(R, 1)), n, axis=1)
        x = np.where(mask, sign * k + rng.standard_normal((R, n)), x)
    return x


def scenario_grid(dists=DISTS):
    """План эксперимента: (распределение, n, ε, расположение, k) — 25 сценариев на пару «распределение — n»."""
    grid = []
    for d in dists:
        for n in (10, 20, 50, 100, 300):
            grid.append((d, n, 0.0, "none", 0))
            for eps in (0.01, 0.05, 0.10, 0.20):
                for ot in ("sym", "one"):
                    for k in (3, 5, 10):
                        grid.append((d, n, eps, ot, k))
    return grid


# ---------- оценки ----------
def _mad(x, med):
    mad = np.median(np.abs(x - med[:, None]), axis=1) / MAD_SCALE
    return np.where(mad <= 0, 1e-12, mad)


def est_mean(x):
    return x.mean(axis=1)


def est_median(x):
    return np.median(x, axis=1)


def est_trim(x, a=0.1):
    """Усечённая средняя: отбрасывается доля a наименьших и наибольших значений."""
    return stats.trim_mean(x, a, axis=1)


def est_huber(x, c=1.345, iters=50, tol=1e-8):
    """M-оценка Хьюбера; масштаб — нормированное MAD; итерационно перевзвешенные наименьшие квадраты."""
    med = np.median(x, axis=1)
    mad = _mad(x, med)
    mu = med.copy()
    for _ in range(iters):
        r = (x - mu[:, None]) / mad[:, None]
        w = np.where(np.abs(r) <= c, 1.0, c / np.maximum(np.abs(r), 1e-12))
        mu_new = (w * x).sum(axis=1) / w.sum(axis=1)
        if np.max(np.abs(mu_new - mu)) < tol:
            mu = mu_new
            break
        mu = mu_new
    return mu


def est_hl(x, chunk=None):
    """Оценка Ходжеса – Лемана: медиана полусумм (x_i + x_j)/2, i <= j."""
    R, n = x.shape
    i, j = np.triu_indices(n)
    if chunk is None:
        chunk = max(1, int(2.5e7 // len(i)))
    out = np.empty(R)
    for a in range(0, R, chunk):
        xb = x[a:a + chunk]
        out[a:a + chunk] = np.median((xb[:, i] + xb[:, j]) / 2, axis=1)
    return out


def all_estimates(x):
    """Матрица (R, 5) в порядке EST_NAMES."""
    return np.column_stack([est_mean(x), est_median(x), est_trim(x), est_huber(x), est_hl(x)])


# ---------- правило Хогга ----------
def _tail_means(xs, beta):
    m = max(1, int(round(beta * xs.shape[1])))
    return xs[:, -m:].mean(axis=1), xs[:, :m].mean(axis=1)


def hogg_q(x):
    """Q = (U_0,05 − L_0,05) / (U_0,5 − L_0,5): средние крайних 5 % и половин выборки."""
    xs = np.sort(x, axis=1)
    u05, l05 = _tail_means(xs, 0.05)
    u50, l50 = _tail_means(xs, 0.5)
    return (u05 - l05) / np.maximum(u50 - l50, 1e-12)


def hogg_rule(q):
    """Индекс выбранной оценки: 0 — средняя (Q <= 2,6), 2 — усечённая средняя (Q <= 3,2), 1 — медиана."""
    return np.where(q <= 2.6, 0, np.where(q <= 3.2, 2, 1))


# ---------- характеристики выборки ----------
def features(x):
    """Семь характеристик (FEAT_NAMES), рассчитанных после центрирования по медиане и деления на MAD."""
    n = x.shape[1]
    med = np.median(x, axis=1)
    z = (x - med[:, None]) / _mad(x, med)[:, None]
    sd = z.std(axis=1, ddof=1)
    skew = np.abs(stats.skew(z, axis=1))
    kurt = stats.kurtosis(z, axis=1)
    q1, q3 = np.percentile(z, [25, 75], axis=1)
    iqr = q3 - q1
    out_share = ((z < q1[:, None] - 1.5 * iqr[:, None]) | (z > q3[:, None] + 1.5 * iqr[:, None])).mean(axis=1)
    mean_med = np.abs(z.mean(axis=1))
    return np.column_stack([np.full(len(x), np.log(n)), skew, kurt, hogg_q(x), sd, out_share, mean_med])


def err_metrics(e):
    """Показатели ошибки по вектору ошибок e (θ = 0): MSE, смещение, MAE, доля серьёзных промахов."""
    return dict(mse=float((e ** 2).mean()), bias=float(e.mean()), mae=float(np.abs(e).mean()),
                big=float((np.abs(e) > BIG).mean()))
