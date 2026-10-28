"""Правило из двух условий для выбора оценки типичного уровня.

Оценка Хьюбера, если
  * доля наблюдений за «усами» диаграммы размаха не больше a и
  * |средняя − медиана| / MAD не больше b,
иначе — медиана. MAD нормирован (умножен на 1,483).

Пороги подобраны в имитационном эксперименте (см. docs/methodology.md):
  cautious (осторожное, рекомендуется): a = 0,05; b = 0,20
  soft (мягкое):                        a = 0,16; b = 0,30

Пример использования из Python:
    >>> from choose import choose_estimator
    >>> choose_estimator([10, 11, 12, 11, 10, 95])["estimator"]
    'median'

Из командной строки:
    python src/choose.py data/wages_2025.csv
    python src/choose.py data/unemp_2025.txt --variant soft
"""
import argparse
import sys

import numpy as np

import sim

THRESHOLDS = {"cautious": (0.05, 0.20), "soft": (0.16, 0.30)}


def summary(x):
    """Характеристики выборки, на которых основано правило."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        raise ValueError("нужно хотя бы 3 наблюдения")
    feats = sim.features(x[None, :])[0]
    return dict(n=int(x.size), share_outside=float(feats[5]), mean_median_gap=float(feats[6]))


def choose_estimator(x, variant="cautious"):
    """Выбрать оценку типичного уровня по правилу из двух условий.

    Возвращает словарь: выбранная оценка ('huber' или 'median'), её значение,
    характеристики выборки и значения средней, медианы и оценки Хьюбера для сравнения.
    """
    if variant not in THRESHOLDS:
        raise ValueError(f"variant должен быть одним из {list(THRESHOLDS)}")
    a, b = THRESHOLDS[variant]
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    s = summary(x)
    use_huber = s["share_outside"] <= a + 1e-9 and s["mean_median_gap"] <= b
    est = {"mean": float(x.mean()), "median": float(np.median(x)), "huber": float(sim.est_huber(x[None, :])[0])}
    name = "huber" if use_huber else "median"
    return dict(estimator=name, value=est[name], variant=variant, **s, estimates=est)


def _read(path):
    import pandas as pd
    if path.endswith(".txt"):
        return pd.read_csv(path, sep=";", header=None).iloc[:, -1].to_numpy(float)
    df = pd.read_csv(path)
    return df.select_dtypes("number").iloc[:, -1].to_numpy(float)


def _fmt(v):
    return f"{v:,.0f}".replace(",", " ") if abs(v) >= 1000 else f"{v:.3g}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Выбор оценки типичного уровня по правилу из двух условий")
    p.add_argument("path", help="CSV (берётся последний числовой столбец) или TXT в формате «название;значение»")
    p.add_argument("--variant", choices=list(THRESHOLDS), default="cautious")
    args = p.parse_args(argv)
    r = choose_estimator(_read(args.path), args.variant)
    ru = {"huber": "оценка Хьюбера", "median": "медиана"}
    print(f"Наблюдений: {r['n']}")
    print(f"Доля за «усами»: {r['share_outside']:.1%}; |средняя − медиана| / MAD: {r['mean_median_gap']:.2f}")
    e = r["estimates"]
    print(f"Средняя {_fmt(e['mean'])}; медиана {_fmt(e['median'])}; оценка Хьюбера {_fmt(e['huber'])}")
    print(f"Правило ({r['variant']}): {ru[r['estimator']]} = {_fmt(r['value'])}")


if __name__ == "__main__":
    sys.exit(main())
