import numpy as np, pandas as pd, pickle, time
import sim2

R = 5000
rng = np.random.default_rng(20261028)
rows, store = [], {}
t0 = time.time()
grid = sim2.scenario_grid()
for sid, (d, n, eps, ot, k) in enumerate(grid):
    x = sim2.draw_samples(d, n, eps, ot, k, R, rng)
    E = sim2.all_estimates(x)
    F = sim2.features(x)
    h = sim2.hogg_rule(F[:, 3])
    row = dict(sid=sid, dist=d, n=n, eps=eps, otype=ot, k=k)
    for j, nm in enumerate(sim2.EST_NAMES):
        for key, v in sim2.err_metrics(E[:, j]).items():
            row[f"{key}_{nm}"] = v
        row[f"se_{nm}"] = float((E[:, j] ** 2).std(ddof=1) / np.sqrt(R))
    for key, v in sim2.err_metrics(E[np.arange(R), h]).items():
        row[f"{key}_hogg"] = v
    mses = [row[f"mse_{nm}"] for nm in sim2.EST_NAMES]
    row["best"] = sim2.EST_NAMES[int(np.argmin(mses))]
    row["mse_oracle"] = min(mses)
    rows.append(row)
    store[sid] = dict(E=E.astype(np.float32), F=F.astype(np.float32))
    if sid % 50 == 0:
        print(sid, len(grid), round(time.time() - t0), flush=True)
df = pd.DataFrame(rows)
df.to_csv("results/scenarios2.csv", index=False)
pickle.dump(store, open("results/store2.pkl", "wb"))
print("done", round(time.time() - t0), df.best.value_counts().to_dict())
