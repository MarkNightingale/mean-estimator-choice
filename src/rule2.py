"""Простое правило: оценка Хьюбера, если доля за «усами» <= a и |средняя − медиана|/MAD <= b, иначе медиана.
Пороги подбираются по сетке на обучающих сценариях (критерий — средняя относительная MSE
с равными весами «чистых» и засорённых сценариев), качество проверяется на отложенных сценариях."""
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import StratifiedGroupKFold
import sim2

sc = pd.read_csv("results/scenarios2.csv")
store = pickle.load(open("results/store2.pkl", "rb"))
iM, iH = sim2.EST_NAMES.index("median"), sim2.EST_NAMES.index("huber")
A_GRID = np.round(np.arange(0.02, 0.21, 0.01), 2)
B_GRID = np.round(np.arange(0.05, 0.61, 0.05), 2)

# для каждого сценария и пары порогов — MSE правила
def rule_mse(s):
    F, E = store[s]["F"], store[s]["E"].astype(np.float64)
    share, mm = F[:, 5], F[:, 6]
    eM, eH = E[:, iM] ** 2, E[:, iH] ** 2
    out = np.empty((len(A_GRID), len(B_GRID)))
    for i, a in enumerate(A_GRID):
        ok_a = share <= a + 1e-9
        for j, b in enumerate(B_GRID):
            use_h = ok_a & (mm <= b)
            out[i, j] = np.where(use_h, eH, eM).mean()
    return out
REL = {s: rule_mse(s) / sc.loc[s, "mse_oracle"] for s in sc.sid}
pickle.dump(REL, open("results/rule_grid.pkl", "wb"))

def objective(ids):
    clean = [REL[s] for s in ids if sc.loc[s, "eps"] == 0]
    cont = [REL[s] for s in ids if sc.loc[s, "eps"] > 0]
    return (np.mean(clean, axis=0) + np.mean(cont, axis=0)) / 2

def best_thr(ids):
    obj = objective(ids); i, j = np.unravel_index(np.argmin(obj), obj.shape)
    return i, j

res = {}
SETTINGS = {"A": (["normal", "t3"], ["laplace", "t2", "logistic"]), "B": (["normal", "t3", "laplace"], ["t2", "logistic"])}
for st, (trd, ted) in SETTINGS.items():
    ids = sc.loc[sc.dist.isin(trd), "sid"].to_numpy()
    strat = np.array([f"{sc.loc[s_, 'dist']}_{sc.loc[s_, 'eps'] > 0}" for s_ in ids])
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=3).split(ids, strat, ids):
        i, j = best_thr(ids[tr])
        for s in ids[te]:
            res[(st, s)] = REL[s][i, j]
    i, j = best_thr(ids)
    print(st, "пороги:", A_GRID[i], B_GRID[j])
    for s in sc.loc[sc.dist.isin(ted), "sid"]:
        res[(st, s)] = REL[s][i, j]
    res[(st, "thr")] = (A_GRID[i], B_GRID[j])
pickle.dump(res, open("results/rule2_res.pkl", "wb"))

def show(st, dists):
    d = sc[sc.dist.isin(dists)].copy()
    d["rel"] = [res[(st, s)] for s in d.sid]
    c, k = d[d.eps == 0].rel, d[d.eps > 0].rel
    return dict(rel_all=d.rel.mean(), max_all=d.rel.max(), rel_clean=c.mean(), rel_cont=k.mean(), rel_eq=(c.mean() + k.mean()) / 2)
for st, (trd, ted) in SETTINGS.items():
    print(st, "CV", {k: round(v, 3) for k, v in show(st, trd).items()})
    for d in ted:
        print(st, d, {k: round(v, 3) for k, v in show(st, [d]).items()})
