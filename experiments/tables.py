"""Строки таблиц статьи (results/tables.json); все числа берутся из расчётных файлов."""
import pandas as pd, numpy as np, json
from common import DATA, FIGURES, QUICK, RESULTS, quick
L = pd.read_csv(RESULTS / "long.csv"); B = pd.read_csv(RESULTS / "bias.csv"); E = pd.read_csv(RESULTS / "regions.csv"); S = pd.read_csv(RESULTS / "semi.csv")
def f(x, d=2):
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s
def fr(x):  # относительная ошибка: 2 знака, большие — меньше знаков
    return f(x, 2) if x < 10 else (f(x, 1) if x < 100 else f(x, 0))
RU = {"mean": "Средняя", "median": "Медиана", "trim10": "Усечённая средняя (10 %)", "huber": "Оценка Хьюбера",
      "hl": "Оценка Ходжеса – Лемана", "hogg": "Правило Хогга", "logit": "Логистическая регрессия", "rf": "Случайный лес",
      "rule": "Правило из двух условий"}
FIX = ["mean", "median", "trim10", "huber", "hl", "hogg"]
T = {}
# Таблица 2: обучающие распределения, вариант A, перекрёстная проверка
d = L[(L.setting == "A") & L.dist.isin(["normal", "t3"])]
rows = []
for m in FIX + ["logit", "rf", "rule"]:
    x = d[d.method == m]; c, k = x[x.eps == 0], x[x.eps > 0]
    name = RU[m] if m != "rule" else "Правило из двух условий (мягкое)"
    rows.append([name, fr(x.rel.mean()), fr(x.rel.max()), fr(c.rel.mean()), fr(k.rel.mean()),
                 fr((c.rel.mean() + k.rel.mean()) / 2), f(k.big.mean() * 100, 1)])
T["t2"] = rows
# Таблица 3: смещение
rows = []
g = B.groupby("eps").mean(numeric_only=True)
for m, nm in [("mean", RU["mean"]), ("median", RU["median"]), ("trim10", RU["trim10"]), ("huber", RU["huber"]), ("hl", RU["hl"]),
              ("hogg", RU["hogg"]), ("rf", RU["rf"]), ("soft", "Мягкое правило"), ("caut", "Осторожное правило")]:
    rows.append([nm] + [f(g.loc[e, m], 2) for e in (0.05, 0.10, 0.20)])
T["t3"] = rows
# Таблица 4: перенос
rows = []
def cell(st, m, dist):
    x = L[(L.setting == st) & (L.method == m) & (L.dist == dist)]
    return [fr(x.rel.mean()), fr(x.rel.max())]
for m in FIX:
    rows.append([RU[m]] + sum([cell("A", m, dd) for dd in ("laplace", "t2", "logistic")], []))
rows.append(["sub", "Обучение на нормальном распределении и распределении Стьюдента"])
for m, nm in [("logit", RU["logit"]), ("rf", RU["rf"]), ("rule", "Мягкое правило")]:
    rows.append([nm] + sum([cell("A", m, dd) for dd in ("laplace", "t2", "logistic")], []))
rows.append(["sub", "Обучение с добавлением распределения Лапласа"])
for m, nm in [("logit", RU["logit"]), ("rf", RU["rf"]), ("rule", "Осторожное правило")]:
    rows.append([nm, "–", "–"] + sum([cell("B", m, dd) for dd in ("t2", "logistic")], []))
T["t4"] = rows
# Таблица 5: региональные показатели
E = E.set_index("key"); K = ["wage", "income", "unemp", "food", "housing"]
def dig(k): return 1 if k == "unemp" else (2 if k == "food" else 0)
def val(k, c): return f(E.loc[k, c], 2 if k == "unemp" else dig(k))
rows = [["Число субъектов"] + [str(int(E.loc[k, "n"])) for k in K],
        ["Значение по Российской Федерации"] + [f(E.loc[k, "rus"], 1 if k == "unemp" else dig(k)) for k in K],
        ["Невзвешенная средняя"] + [val(k, "est_mean") for k in K],
        ["Усечённая средняя (10 %)"] + [val(k, "est_trim10") for k in K],
        ["Оценка Хьюбера"] + [val(k, "est_huber") for k in K],
        ["Оценка Ходжеса – Лемана"] + [val(k, "est_hl") for k in K],
        ["Медиана"] + [val(k, "est_median") for k in K],
        ["Доля субъектов за «усами», %"] + [f(E.loc[k, "out_share"] * 100, 1) for k in K],
        ["|средняя – медиана| / MAD"] + [f(E.loc[k, "mm"], 2) for k in K],
        ["Статистика Хогга Q"] + [f(E.loc[k, "q"], 2) for k in K],
        ["Выбор правила Хогга и обоих правил из двух условий"] + ["медиана" for k in K],
        ["Выбор случайного леса (вероятность)"] + [("медиана" if E.loc[k, "rf"] == "median" else "оценка Хьюбера") + f" ({f(E.loc[k, 'rf_p'], 2)})" for k in K],
        ["Превышение средней над медианой, %"] + [f((E.loc[k, "est_mean"] / E.loc[k, "est_median"] - 1) * 100, 1) for k in K]]
T["t5"] = rows
# Таблица 6: полусинтетический эксперимент
NM = {"wage": "Заработная плата", "income": "Денежные доходы", "unemp": "Уровень безработицы", "food": "Стоимость набора продуктов", "housing": "Цена жилья"}
rows = []
for k in K:
    for _, q in S[S.key == k].iterrows():
        rows.append([NM[k] if q.m == 2 else "", f"{int(q.m)}"] + [f(q[f"{e}_mean"], 1) for e in ("mean", "trim10", "huber", "hl", "median", "rf")])
T["t6"] = rows
json.dump(T, open(RESULTS / "tables.json", "w"), ensure_ascii=False, indent=1)
for k, v in T.items():
    print(k); [print("  ", r) for r in v]
