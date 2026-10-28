"""Имитационный эксперимент: выбор оценки параметра положения при выбросах.

Модель засорения: X = (1-eps)*F + eps*G, theta = 0 (оценки эквивариантны по сдвигу).
Все «чистые» распределения нормированы к одинаковому межквартильному размаху
(как у N(0,1): IQR = 1.349), поэтому sigma = 1 для всех F.
"""
import numpy as np
from scipy import stats

IQR_N = 2 * stats.norm.ppf(0.75)  # 1.349

def draw_clean(dist, size, rng):
    if dist == "normal":
        return rng.standard_normal(size)
    if dist == "t3":
        s = IQR_N / (2 * stats.t.ppf(0.75, 3))
        return rng.standard_t(3, size) * s
    if dist == "laplace":
        s = IQR_N / (2 * stats.laplace.ppf(0.75))
        return rng.laplace(0, 1, size) * s
    raise ValueError(dist)

def draw_samples(dist, n, eps, otype, k, R, rng):
    x = draw_clean(dist, (R, n), rng)
    if eps > 0:
        mask = rng.random((R, n)) < eps
        if otype == "sym":
            sign = rng.choice([-1.0, 1.0], size=(R, n))
        else:  # one-sided: все выбросы выборки по одну сторону, сторона случайна
            sign = np.repeat(rng.choice([-1.0, 1.0], size=(R, 1)), n, axis=1)
        out = sign * k + rng.standard_normal((R, n))
        x = np.where(mask, out, x)
    return x

# ---------- оценки (векторизованы по строкам) ----------
def est_mean(x):
    return x.mean(axis=1)

def est_median(x):
    return np.median(x, axis=1)

def est_trim(x, a=0.1):
    return stats.trim_mean(x, a, axis=1)

def est_huber(x, c=1.345, iters=50, tol=1e-8):
    med = np.median(x, axis=1)
    mad = np.median(np.abs(x - med[:, None]), axis=1) / 0.6745
    mad = np.where(mad <= 0, 1e-12, mad)
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

EST_NAMES = ["mean", "median", "trim10", "huber"]
EST_RU = {"mean": "Средняя", "median": "Медиана", "trim10": "Усечённая средняя (10 %)", "huber": "Оценка Хьюбера"}

def all_estimates(x):
    return np.column_stack([est_mean(x), est_median(x), est_trim(x), est_huber(x)])

# ---------- статистики формы и признаки ----------
def _tail_means(xs, beta):
    n = xs.shape[1]
    m = max(1, int(round(beta * n)))
    return xs[:, -m:].mean(axis=1), xs[:, :m].mean(axis=1)

def hogg_q(x):
    xs = np.sort(x, axis=1)
    u05, l05 = _tail_means(xs, 0.05)
    u50, l50 = _tail_means(xs, 0.5)
    return (u05 - l05) / np.maximum(u50 - l50, 1e-12)

def hogg_rule(q):
    """0=mean, 1=median, 2=trim10 (пороги 2,6 и 3,2 по Хоггу)."""
    return np.where(q <= 2.6, 0, np.where(q <= 3.2, 2, 1))

def features(x):
    n = x.shape[1]
    med = np.median(x, axis=1)
    mad = np.median(np.abs(x - med[:, None]), axis=1) / 0.6745
    mad = np.where(mad <= 0, 1e-12, mad)
    z = (x - med[:, None]) / mad[:, None]
    sd = z.std(axis=1, ddof=1)
    skew = np.abs(stats.skew(z, axis=1))
    kurt = stats.kurtosis(z, axis=1)
    q1, q3 = np.percentile(z, [25, 75], axis=1)
    iqr = q3 - q1
    out_share = ((z < q1[:, None] - 1.5 * iqr[:, None]) | (z > q3[:, None] + 1.5 * iqr[:, None])).mean(axis=1)
    mean_med = np.abs(z.mean(axis=1))  # |mean - median|/MAD
    return np.column_stack([np.full(len(x), np.log(n)), skew, kurt, hogg_q(x), sd, out_share, mean_med])

FEAT_NAMES = ["log n", "|асимметрия|", "эксцесс", "Q Хогга", "SD/MAD", "доля за «усами»", "|средняя − медиана|/MAD"]

def scenario_grid(dists=("normal", "t3")):
    grid = []
    for d in dists:
        for n in (20, 50, 100, 300):
            grid.append((d, n, 0.0, "none", 0))
            for eps in (0.05, 0.10, 0.20):
                for ot in ("sym", "one"):
                    for k in (5, 10):
                        grid.append((d, n, eps, ot, k))
    return grid
