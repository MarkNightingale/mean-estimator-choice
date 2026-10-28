import numpy as np, pandas as pd, pickle, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedGroupKFold
import sim2

df = pd.read_csv("results/scenarios2.csv")
store = pickle.load(open("results/store2.pkl", "rb"))
EST = sim2.EST_NAMES
LAB = {e: i for i, e in enumerate(EST)}
N_PER = 400
rng = np.random.default_rng(7)
SETTINGS = {"A": (["normal", "t3"], ["laplace", "t2", "logistic"]),
            "B": (["normal", "t3", "laplace"], ["t2", "logistic"])}

def build(ids):
    X, y, g = [], [], []
    for s in ids:
        F = store[s]["F"]; idx = rng.choice(len(F), N_PER, replace=False)
        X.append(F[idx]); y.append(np.full(N_PER, LAB[df.loc[s, "best"]])); g.append(np.full(N_PER, s))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)

def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=1)
def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
def surrogate(model, X):
    return DecisionTreeClassifier(max_depth=2, random_state=1).fit(X, model.predict(X))

def evaluate(model, s):
    F, E = store[s]["F"], store[s]["E"]
    pick = model.predict(F)
    e = E[np.arange(len(E)), pick].astype(np.float64)
    m = sim2.err_metrics(e)
    m["picks"] = np.bincount(pick, minlength=len(EST)) / len(pick)
    return m

out = {}   # (setting, method, sid) -> metrics
trees = {}
t0 = time.time()
for st, (train_d, test_d) in SETTINGS.items():
    tr_ids = df.loc[df.dist.isin(train_d), "sid"].to_numpy()
    te_ids = df.loc[df.dist.isin(test_d), "sid"].to_numpy()
    X, y, g = build(tr_ids)
    strat = np.array([f"{df.loc[s_, 'dist']}_{df.loc[s_, 'eps'] > 0}" for s_ in g])
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=3).split(X, strat, g):
        mr = rf().fit(X[tr], y[tr]); ml = logit().fit(X[tr], y[tr]); mt = surrogate(mr, X[tr])
        for s in np.unique(g[te]):
            for name, m in (("rf", mr), ("logit", ml), ("rule", mt)):
                out[(st, name, s)] = evaluate(m, s)
    mr = rf().fit(X, y); ml = logit().fit(X, y); mt = surrogate(mr, X)
    trees[st] = dict(tree=mt, text=export_text(mt, feature_names=sim2.FEAT_NAMES), importances=mr.feature_importances_, rf=mr)
    for s in te_ids:
        for name, m in (("rf", mr), ("logit", ml), ("rule", mt)):
            out[(st, name, s)] = evaluate(m, s)
    print(st, "done", round(time.time() - t0), flush=True)
    print(trees[st]["text"])

rows = []
for (st, name, s), m in out.items():
    r = dict(setting=st, method=name, sid=s, **{k: v for k, v in m.items() if k != "picks"})
    for j, e in enumerate(EST): r[f"pick_{e}"] = m["picks"][j]
    rows.append(r)
res = pd.DataFrame(rows).merge(df[["sid", "dist", "n", "eps", "otype", "k", "mse_oracle"]], on="sid")
res.to_csv("results/ml2_results.csv", index=False)
pickle.dump({k: {kk: vv for kk, vv in v.items() if kk != "rf"} for k, v in trees.items()}, open("results/ml2_trees.pkl", "wb"))
pickle.dump(trees["B"]["rf"], open("results/rf_B.pkl", "wb"))
pickle.dump(trees["A"]["rf"], open("results/rf_A.pkl", "wb"))
