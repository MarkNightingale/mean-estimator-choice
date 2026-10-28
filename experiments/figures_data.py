"""Исходные данные рисунков в Excel (требование журнала)."""
import numpy as np, pandas as pd, pickle
from common import DATA, FIGURES, QUICK, RESULTS, quick
L = pd.read_csv(RESULTS / "long.csv"); sc = pd.read_csv(RESULTS / "scenarios.csv")
RU = {"mean": "Средняя", "trim10": "Усечённая средняя (10 %)", "hl": "Оценка Ходжеса – Лемана", "huber": "Оценка Хьюбера",
      "median": "Медиана", "rf": "Случайный лес"}
tr = L[(L.setting == "A") & L.dist.isin(["normal", "t3"])]
rows = []
for ot, lab in [("sym", "Симметричные выбросы"), ("one", "Односторонние выбросы")]:
    d = tr[(tr.eps == 0) | (tr.otype == ot)]
    for eps in sorted(d.eps.unique()):
        r = {"Панель": lab, "Доля выбросов, %": eps * 100}
        for m, nm in RU.items(): r[nm] = round(d[(d.method == m) & (d.eps == eps)].rel.mean(), 4)
        rows.append(r)
f1 = pd.DataFrame(rows)
REL = pickle.load(open(RESULTS / "rule_grid.pkl", "rb"))
A_GRID = np.round(np.arange(0.02, 0.21, 0.01), 2); B_GRID = np.round(np.arange(0.05, 0.61, 0.05), 2)
def obj(dists):
    ids = sc.loc[sc.dist.isin(dists), "sid"]
    c = [REL[s] for s in ids if sc.loc[s, "eps"] == 0]; k = [REL[s] for s in ids if sc.loc[s, "eps"] > 0]
    return (np.mean(c, 0) + np.mean(k, 0)) / 2
f2 = {}
for nm, dd in [("Рис2_норм_Стьюдент", ["normal", "t3"]), ("Рис2_с_Лапласом", ["normal", "t3", "laplace"])]:
    O = obj(dd)
    f2[nm] = pd.DataFrame(np.round(O, 4), index=[f"{a*100:.0f}" for a in A_GRID], columns=[f"{b:.2f}" for b in B_GRID])
    f2[nm].index.name = "Порог доли за «усами», % \\ порог |средняя − медиана|/MAD"
def load(p): return pd.read_csv(p, sep=";", header=None, names=["Субъект РФ", "Значение"])
w = pd.read_csv(DATA / "wages_2025.csv").rename(columns={"region": "Субъект РФ", "wage": "Значение"})
IND = [("Заработная плата, руб.", w), ("Денежные доходы, руб. в месяц", load(DATA / "income_2025.txt")),
       ("Уровень безработицы, %", load(DATA / "unemp_2025.txt")), ("Стоимость набора продуктов, руб.", load(DATA / "food_2025.txt")),
       ("Цена жилья на вторичном рынке, руб. за кв. м", load(DATA / "housing_2025.txt"))]
parts = []
for nm, d in IND:
    v = d["Значение"].astype(float); med = v.median(); mad = (v - med).abs().median() / 0.6745
    parts.append(d.assign(**{"Показатель": nm, "(значение − медиана)/MAD": ((v - med) / mad).round(3)})[["Показатель", "Субъект РФ", "Значение", "(значение − медиана)/MAD"]])
f3 = pd.concat(parts)
with pd.ExcelWriter(FIGURES / "figures_data.xlsx") as xw:
    pd.DataFrame({"Лист": ["Рис1", "Рис2_норм_Стьюдент", "Рис2_с_Лапласом", "Рис3"],
                  "Содержание": ["Отношение MSE к MSE лучшей оценки сценария, среднее по сценариям (нормальное распределение и распределение Стьюдента, 250 сценариев)",
                                 "Средняя относительная ошибка правила из двух условий (равные веса), обучение на нормальном и Стьюдента",
                                 "То же с добавлением распределения Лапласа",
                                 "Значения показателей по субъектам РФ за 2025 г. и робастная шкала"]}).to_excel(xw, sheet_name="Описание", index=False)
    f1.to_excel(xw, sheet_name="Рис1", index=False)
    for nm, d in f2.items(): d.to_excel(xw, sheet_name=nm)
    f3.to_excel(xw, sheet_name="Рис3", index=False)
print(f1.round(2).to_string())
