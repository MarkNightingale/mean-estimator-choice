"""Шаг 1. Имитационный эксперимент: 625 сценариев × 5 000 выборок.

Для каждой выборки сохраняются ошибки пяти оценок и семь характеристик (results/store.pkl, ~150 МБ),
для каждого сценария — MSE, смещение, MAE, доля серьёзных промахов и правило Хогга (results/scenarios.csv).
"""
import numpy as np, pandas as pd, pickle, time
from common import DATA, FIGURES, QUICK, RESULTS, quick
from estimator_choice import estimators as est

R = quick(5000, 120)
rng = np.random.default_rng(20261028)
rows, store = [], {}
t0 = time.time()
grid = est.scenario_grid()
for sid, (d, n, eps, ot, k) in enumerate(grid):
    x = est.draw_samples(d, n, eps, ot, k, R, rng)
    E = est.all_estimates(x)
    F = est.features(x)
    h = est.hogg_rule(F[:, 3])
    row = dict(sid=sid, dist=d, n=n, eps=eps, otype=ot, k=k)
    for j, nm in enumerate(est.EST_NAMES):
        for key, v in est.err_metrics(E[:, j]).items():
            row[f"{key}_{nm}"] = v
        row[f"se_{nm}"] = float((E[:, j] ** 2).std(ddof=1) / np.sqrt(R))
    for key, v in est.err_metrics(E[np.arange(R), h]).items():
        row[f"{key}_hogg"] = v
    mses = [row[f"mse_{nm}"] for nm in est.EST_NAMES]
    row["best"] = est.EST_NAMES[int(np.argmin(mses))]
    row["mse_oracle"] = min(mses)
    rows.append(row)
    store[sid] = dict(E=E.astype(np.float32), F=F.astype(np.float32))
    if sid % 50 == 0:
        print(sid, len(grid), round(time.time() - t0), flush=True)
df = pd.DataFrame(rows)
df.to_csv(RESULTS / "scenarios.csv", index=False)
pickle.dump(store, open(RESULTS / "store.pkl", "wb"))
print("done", round(time.time() - t0), df.best.value_counts().to_dict())
