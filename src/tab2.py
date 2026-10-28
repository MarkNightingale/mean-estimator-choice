"""Сводные таблицы для текста статьи (v2)."""
import numpy as np, pandas as pd, pickle, json
from sklearn.model_selection import StratifiedGroupKFold
import sim2
sc = pd.read_csv("results/scenarios2.csv"); r = pd.read_csv("results/ml2_results.csv")
store = pickle.load(open("results/store2.pkl", "rb")); REL = pickle.load(open("results/rule_grid.pkl", "rb"))
iM, iH = sim2.EST_NAMES.index("median"), sim2.EST_NAMES.index("huber")
A_GRID = np.round(np.arange(0.02, 0.21, 0.01), 2); B_GRID = np.round(np.arange(0.05, 0.61, 0.05), 2)
SET = {"A": (["normal", "t3"], ["laplace", "t2", "logistic"]), "B": (["normal", "t3", "laplace"], ["t2", "logistic"])}

def obj(ids):
    c = [REL[s] for s in ids if sc.loc[s, "eps"] == 0]; k = [REL[s] for s in ids if sc.loc[s, "eps"] > 0]
    return (np.mean(c, 0) + np.mean(k, 0)) / 2
def thr_of(ids):
    i, j = np.unravel_index(np.argmin(obj(ids)), (len(A_GRID), len(B_GRID))); return A_GRID[i], B_GRID[j]
def rule_metrics(s, a, b):
    F, E = store[s]["F"], store[s]["E"].astype(float)
    use = (F[:, 5] <= a + 1e-9) & (F[:, 6] <= b)
    return sim2.err_metrics(np.where(use, E[:, iH], E[:, iM]))

rows = []
for e in sim2.EST_NAMES + ["hogg"]:
    for _, q in sc.iterrows():
        for st in "AB":
            rows.append(dict(setting=st, method=e, sid=q.sid, mse=q[f"mse_{e}"], big=q[f"big_{e}"], mae=q[f"mae_{e}"]))
for _, q in r[r.method.isin(["rf", "logit"])].iterrows():
    rows.append(dict(setting=q.setting, method=q.method, sid=q.sid, mse=q.mse, big=q.big, mae=q.mae))
thr_full = {}
for st, (trd, ted) in SET.items():
    ids = sc.loc[sc.dist.isin(trd), "sid"].to_numpy()
    strat = np.array([f"{sc.loc[s, 'dist']}_{sc.loc[s, 'eps'] > 0}" for s in ids])
    folds = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=3).split(ids, strat, ids):
        a, b = thr_of(ids[tr]); folds.append((a, b))
        for s in ids[te]: rows.append(dict(setting=st, method="rule", sid=s, **{k: v for k, v in rule_metrics(s, a, b).items() if k != "bias"}))
    a, b = thr_of(ids); thr_full[st] = (float(a), float(b), [tuple(map(float, f)) for f in folds])
    for s in sc.loc[sc.dist.isin(ted), "sid"]:
        rows.append(dict(setting=st, method="rule", sid=s, **{k: v for k, v in rule_metrics(s, a, b).items() if k != "bias"}))
L = pd.DataFrame(rows).merge(sc[["sid", "dist", "n", "eps", "otype", "k", "mse_oracle"]], on="sid")
L["rel"] = L.mse / L.mse_oracle
L.to_csv("results/long2.csv", index=False)
print("пороги", thr_full)

def summ(d):
    c, k = d[d.eps == 0], d[d.eps > 0]
    return pd.Series(dict(rel=d.rel.mean(), mx=d.rel.max(), clean=c.rel.mean(), cont=k.rel.mean(),
                          eq=(c.rel.mean() + k.rel.mean()) / 2, big_cont=k.big.mean() * 100, big_clean=c.big.mean() * 100,
                          mae=d.mae.mean(), ns=len(d)))
pd.set_option("display.width", 250)
for st, (trd, ted) in SET.items():
    print(f"==== {st} CV {trd}")
    print(L[(L.setting == st) & L.dist.isin(trd)].groupby("method").apply(summ).round(3))
    for d in ted:
        print(f"-- {st} OOD {d}")
        print(L[(L.setting == st) & (L.dist == d)].groupby("method").apply(summ)[["rel", "mx", "clean", "cont", "eq", "big_cont"]].round(3))
