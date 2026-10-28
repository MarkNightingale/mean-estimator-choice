"""Рисунки статьи (оттенки серого, 300 dpi) в figures/."""
import numpy as np, pandas as pd, pickle
from common import DATA, FIGURES, QUICK, RESULTS, quick
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
comma = FuncFormatter(lambda v, p: (f"{v:g}").replace(".", ","))
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.edgecolor": "black",
                     "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
L = pd.read_csv(RESULTS / "long.csv")
tr = L[(L.setting == "A") & L.dist.isin(["normal", "t3"])]

# ---------- Рис. 1 ----------
styles = {"mean": ("Средняя", "-", "o", "black", "none"), "trim10": ("Усечённая средняя", ":", "^", "dimgray", "none"),
          "hl": ("Оценка Ходжеса – Лемана", (0, (5, 1.5, 1, 1.5, 1, 1.5)), "v", "0.45", "none"),
          "huber": ("Оценка Хьюбера", "-.", "D", "gray", "none"), "median": ("Медиана", "--", "s", "black", "none"),
          "rf": ("Случайный лес", "-", "o", "black", "black")}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
for ax, ot, title in zip(axes, ["sym", "one"], ["Симметричные выбросы", "Односторонние выбросы"]):
    d = tr[(tr.eps == 0) | (tr.otype == ot)]
    for m, (lab, ls, mk, col, fc) in styles.items():
        y = d[d.method == m].groupby("eps").rel.mean()
        ax.plot(y.index * 100, y.values, linestyle=ls, marker=mk, color=col, mfc=fc, label=lab, ms=4.5, lw=1.2 if m != "rf" else 1.8)
    ax.set_yscale("log"); ax.set_title(title, fontsize=10)
    ax.set_xlabel("Доля выбросов, %"); ax.set_xticks([0, 1, 5, 10, 20])
    ax.grid(True, which="major", color="0.85", lw=0.6)
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5, 10])); ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_major_formatter(comma)
axes[0].set_ylabel("MSE / MSE лучшей оценки (лог. шкала)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(FIGURES / "fig1.png", dpi=300); plt.close(fig)

# ---------- Рис. 2: критерий подбора порогов правила ----------
REL = pickle.load(open(RESULTS / "rule_grid.pkl", "rb")); sc = pd.read_csv(RESULTS / "scenarios.csv")
A_GRID = np.round(np.arange(0.02, 0.21, 0.01), 2); B_GRID = np.round(np.arange(0.05, 0.61, 0.05), 2)
def obj(dists):
    ids = sc.loc[sc.dist.isin(dists), "sid"]
    c = [REL[s] for s in ids if sc.loc[s, "eps"] == 0]; k = [REL[s] for s in ids if sc.loc[s, "eps"] > 0]
    return (np.mean(c, 0) + np.mean(k, 0)) / 2
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, dists, title in zip(axes, [["normal", "t3"], ["normal", "t3", "laplace"]],
                            ["Нормальное и Стьюдента", "Нормальное, Стьюдента и Лапласа"]):
    O = obj(dists)
    X, Y = np.meshgrid(A_GRID * 100, B_GRID)
    lev = [1.08, 1.10, 1.12, 1.15, 1.20, 1.30]
    cs = ax.contour(X, Y, O.T, levels=lev, colors="black", linewidths=0.8)
    ax.clabel(cs, fmt=lambda v: f"{v:.2f}".replace(".", ","), fontsize=7.5)
    for a, b, lab in [(16, 0.30, "мягкое"), (5, 0.20, "осторожное")]:
        ax.plot(a, b, marker="s" if lab == "осторожное" else "o", mfc="white", mec="black", ms=8, ls="none")
        ax.annotate(lab, (a, b), xytext=(6, 6) if lab == "мягкое" else (-4, -16), textcoords="offset points", fontsize=8.5,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_title(title, fontsize=10); ax.set_xlabel("Порог доли наблюдений за «усами», %")
    ax.yaxis.set_major_formatter(comma)
axes[0].set_ylabel("Порог |средняя − медиана| / MAD")
fig.tight_layout(); fig.savefig(FIGURES / "fig2.png", dpi=300); plt.close(fig)

# ---------- Рис. 3: региональные показатели в робастной шкале ----------
def load(p): return pd.read_csv(p, sep=";", header=None, names=["r", "v"]).v.to_numpy(float)
IND = [("Заработная плата", pd.read_csv(DATA / "wages_2025.csv").wage.to_numpy(float)),
       ("Денежные доходы", load(DATA / "income_2025.txt")), ("Уровень безработицы", load(DATA / "unemp_2025.txt")),
       ("Стоимость набора\nпродуктов", load(DATA / "food_2025.txt")), ("Цена жилья\n(вторичный рынок)", load(DATA / "housing_2025.txt"))]
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(7.2, 4.1))
for i, (lab, v) in enumerate(IND):
    med = np.median(v); mad = np.median(np.abs(v - med)) / 0.6745
    z = (v - med) / mad; y = len(IND) - 1 - i
    zc = z[z <= 12]
    ax.scatter(zc, y + rng.uniform(-0.18, 0.18, len(zc)), s=9, facecolors="none", edgecolors="0.35", linewidths=0.6)
    out = z > 12
    if out.any(): ax.annotate(f"ещё {out.sum()} за пределами шкалы (до {z.max():.0f})".replace(".", ","), (12, y + 0.3), ha="right", fontsize=7.5)
    ax.plot([0, 0], [y - 0.32, y + 0.32], color="black", lw=1.8)
    m = (v.mean() - med) / mad
    ax.plot(m, y, marker="v", color="black", ms=7, ls="none")
ax.set_yticks(range(len(IND))); ax.set_yticklabels([l for l, _ in IND][::-1], fontsize=9)
ax.set_xlim(-4.3, 12.3); ax.set_xlabel("(значение − медиана) / MAD")
ax.axvline(0, color="0.8", lw=0.6, zorder=0)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color="black", lw=1.8), Line2D([], [], marker="v", color="black", ls="none")],
          ["медиана", "невзвешенная средняя"], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=8.5)
ax.xaxis.set_major_formatter(comma); ax.grid(True, axis="x", color="0.9", lw=0.5)
fig.tight_layout(); fig.savefig(FIGURES / "fig3.png", dpi=300); plt.close(fig)
