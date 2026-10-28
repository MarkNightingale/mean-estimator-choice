"""Шаг 4. Смещение оценок при односторонних выбросах (сторона зафиксирована: справа)."""
import numpy as np, pandas as pd, pickle
from common import DATA, FIGURES, QUICK, RESULTS, quick
from estimator_choice import estimators as est
rf = pickle.load(open(RESULTS / "rf_A.pkl", "rb"))
rng = np.random.default_rng(5)
R = quick(5000, 200); rows = []
for d in ("normal", "t3"):
  for n in (20, 100):
    for k in (3, 5, 10):
        for eps in (0.05, 0.10, 0.20):
            x = est.draw_clean(d, (R, n), rng)
            mask = rng.random((R, n)) < eps
            x = np.where(mask, k + rng.standard_normal((R, n)), x)
            E = est.all_estimates(x); F = est.features(x)
            pick = rf.predict(F)
            soft = np.where((F[:, 5] <= 0.16 + 1e-9) & (F[:, 6] <= 0.30), 3, 1)
            caut = np.where((F[:, 5] <= 0.05 + 1e-9) & (F[:, 6] <= 0.20), 3, 1)
            h = est.hogg_rule(F[:, 3])
            row = dict(dist=d, n=n, k=k, eps=eps)
            for j, e in enumerate(est.EST_NAMES): row[e] = E[:, j].mean()
            row["hogg"] = E[np.arange(R), h].mean(); row["rf"] = E[np.arange(R), pick].mean()
            row["soft"] = E[np.arange(R), soft].mean(); row["caut"] = E[np.arange(R), caut].mean()
            rows.append(row)
b = pd.DataFrame(rows); b.to_csv(RESULTS / "bias.csv", index=False)
pd.set_option("display.width", 250); print(b.round(3).to_string())
print(b.groupby("eps")[["mean","median","trim10","huber","hl","hogg","rf","soft","caut"]].mean().round(3))
