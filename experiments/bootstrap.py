"""Шаг 4. Бутстреп-интервалы для разностей средней относительной MSE между методами.

Сценарии (250 на нормальном распределении и распределении Стьюдента, перекрёстная проверка)
переизвлекаются с возвращением; разность считается на одних и тех же сценариях (парное сравнение).
Сценарии без выбросов и с выбросами переизвлекаются отдельно, чтобы их соотношение сохранялось.
"""
import numpy as np
import pandas as pd

from common import DATA, FIGURES, QUICK, RESULTS, quick

L = pd.read_csv(RESULTS / "long.csv")
d = L[(L.setting == "A") & L.dist.isin(["normal", "t3"])]
P = d.pivot(index="sid", columns="method", values="rel")
clean = d.drop_duplicates("sid").set_index("sid").eps.reindex(P.index) == 0
PAIRS = [("rf", "median"), ("rf", "huber"), ("rule", "median"), ("rule", "huber"), ("rf", "rule"), ("rf", "logit")]
B = quick(10000, 200)
rng = np.random.default_rng(2026)
ic, ik = np.where(clean.to_numpy())[0], np.where(~clean.to_numpy())[0]
rows = []
for a, b in PAIRS:
    diff = (P[a] - P[b]).to_numpy()
    eq = lambda v, c, k: (v[c].mean() + v[k].mean()) / 2
    bs_all, bs_eq = np.empty(B), np.empty(B)
    for i in range(B):
        c = rng.choice(ic, len(ic)); k = rng.choice(ik, len(ik))
        idx = np.concatenate([c, k])
        bs_all[i] = diff[idx].mean(); bs_eq[i] = eq(diff, c, k)
    rows.append(dict(a=a, b=b, diff_all=diff.mean(), lo_all=np.percentile(bs_all, 2.5), hi_all=np.percentile(bs_all, 97.5),
                     diff_eq=eq(diff, ic, ik), lo_eq=np.percentile(bs_eq, 2.5), hi_eq=np.percentile(bs_eq, 97.5),
                     share_better=(diff < 0).mean()))
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "bootstrap.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))
