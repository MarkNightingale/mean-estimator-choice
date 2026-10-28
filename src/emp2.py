import numpy as np, pandas as pd, pickle
import sim2

rf = pickle.load(open("results/rf_B.pkl", "rb"))
RR = pickle.load(open("results/rule2_res.pkl", "rb"))
THR = {"soft": RR[("A", "thr")], "cautious": RR[("B", "thr")]}
iM, iH = sim2.EST_NAMES.index("median"), sim2.EST_NAMES.index("huber")
class Rule:
    def __init__(self, a, b): self.a, self.b = a, b
    def predict(self, F): return np.where((F[:, 5] <= self.a + 1e-9) & (F[:, 6] <= self.b), iH, iM)
rule = Rule(*THR["cautious"]); soft = Rule(*THR["soft"])
print("пороги", THR)
EST = sim2.EST_NAMES

def load(path, sep=";"):
    return pd.read_csv(path, sep=sep, header=None, names=["region", "value"])["value"].to_numpy(float)

IND = {
    "wage": ("Среднемесячная номинальная начисленная заработная плата, руб.", pd.read_csv("data/wages_2025.csv")["wage"].to_numpy(float), 100360),
    "income": ("Среднедушевые денежные доходы населения, руб. в месяц", load("data/income_2025.txt"), 74845),
    "unemp": ("Уровень безработицы, %", load("data/unemp_2025.txt"), 2.2),
    "food": ("Стоимость минимального набора продуктов питания, руб. (декабрь)", load("data/food_2025.txt"), 7347.16),
    "housing": ("Средняя цена на вторичном рынке жилья, руб. за кв. м", load("data/housing_2025.txt"), 130359),
}
rows = []
for key, (name, v, rus) in IND.items():
    x = v[None, :]
    E = sim2.all_estimates(x)[0]; F = sim2.features(x)
    pr = rf.predict_proba(F)[0]
    r = dict(key=key, name=name, n=len(v), rus=rus, **{f"est_{e}": E[j] for j, e in enumerate(EST)},
             out_share=F[0, 5], mm=F[0, 6], q=F[0, 3], kurt=F[0, 2],
             rf=EST[rf.classes_[np.argmax(pr)]], rf_p=pr.max(), rule=EST[rule.predict(F)[0]], soft=EST[soft.predict(F)[0]],
             hogg=["mean", "median", "trim10"][sim2.hogg_rule(F[:, 3])[0]])
    r["chosen_rf_value"] = r[f"est_{r['rf']}"]
    r["mean_vs_rf_pct"] = (r["est_mean"] / r["chosen_rf_value"] - 1) * 100
    rows.append(r)
emp = pd.DataFrame(rows)
pd.set_option("display.width", 250); print(emp.round(3).T)
emp.to_csv("results/emp2.csv", index=False)

# Полусинтетическая проверка: «лишний разряд» (×10) в m регионах, для всех показателей
rs = np.random.default_rng(11)
B = 2000
semi = []
for key, (name, v, rus) in IND.items():
    x = v[None, :]
    E0 = sim2.all_estimates(x)[0]; F0 = sim2.features(x)
    ref_rf = E0[rf.predict(F0)[0]]; ref_rule = E0[rule.predict(F0)[0]]
    for m in (2, 4, 8):
        xm = np.repeat(x, B, axis=0)
        for b in range(B):
            xm[b, rs.choice(len(v), m, replace=False)] *= 10
        Em = sim2.all_estimates(xm); Fm = sim2.features(xm)
        prf = rf.predict(Fm); pru = rule.predict(Fm)
        row = dict(key=key, m=m, share=m / len(v) * 100)
        for j, e in enumerate(EST):
            d = np.abs(Em[:, j] / E0[j] - 1) * 100
            row[f"{e}_mean"] = d.mean(); row[f"{e}_p95"] = np.percentile(d, 95)
        d = np.abs(Em[np.arange(B), prf] / ref_rf - 1) * 100; row["rf_mean"] = d.mean(); row["rf_p95"] = np.percentile(d, 95)
        d = np.abs(Em[np.arange(B), pru] / ref_rule - 1) * 100; row["rule_mean"] = d.mean(); row["rule_p95"] = np.percentile(d, 95)
        for j, e in enumerate(EST): row[f"rfpick_{e}"] = (prf == j).mean()
        semi.append(row)
semi = pd.DataFrame(semi)
print(semi.round(2).to_string())
semi.to_csv("results/semi2.csv", index=False)
