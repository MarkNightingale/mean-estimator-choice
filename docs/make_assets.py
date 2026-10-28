"""Иллюстрации для README (светлая и тёмная темы) и docs/results.md.

Запуск из корня репозитория: python docs/make_assets.py
Цвета проверены на различимость при нарушениях цветового зрения
(синий — медиана и предложенные методы, охра — средняя, серый — остальные).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2328", muted="#59636e", faint="#d1d9e0", dots="#9aa3ad",
                  blue="#1f6fae", ochre="#c0661a"),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", faint="#30363d", dots="#6e7681",
                 blue="#4b9be0", ochre="#d27b2e"),
}
SERIF = "Lora" if any("Lora" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Serif"
SANS = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"svg.fonttype": "path", "font.family": SANS, "svg.hashsalt": "mean-estimator-choice"})  # одинаковые SVG при повторном запуске
parse = lambda s: float(s.replace(" ", "").replace(",", "."))
comma = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")


def banner(theme):
    c = THEMES[theme]
    w = pd.read_csv(ROOT / "data" / "wages_2025.csv")["wage"].to_numpy() / 1000
    mean, med = w.mean(), np.median(w)
    q1, q3 = np.percentile(w, [25, 75]); hi = q3 + 1.5 * (q3 - q1)
    fig = plt.figure(figsize=(12.8, 4.0), dpi=100, facecolor=c["bg"])
    fig.text(0.045, 0.86, "Какую среднюю выбрать,\nесли в данных есть выбросы", family=SERIF, size=27,
             color=c["ink"], va="top", linespacing=1.15)
    fig.text(0.045, 0.42, "Имитационное исследование, модель машинного обучения\nи простое правило для статистической практики",
             size=14, color=c["muted"], va="top", linespacing=1.45)
    ax = fig.add_axes([0.565, 0.24, 0.40, 0.48], facecolor=c["bg"])
    rng = np.random.default_rng(7)
    y = rng.uniform(-0.32, 0.32, len(w))
    out = w > hi
    ax.scatter(w[~out], y[~out], s=40, facecolors="none", edgecolors=c["dots"], linewidths=1.1, zorder=2)
    ax.scatter(w[out], y[out], s=40, facecolors=c["dots"], edgecolors=c["bg"], linewidths=1.0, zorder=2)
    for x, col, lab, ha, dx in [(med, c["blue"], f"медиана {comma(med)}", "right", -1.6),
                                (mean, c["ochre"], f"средняя {comma(mean)}", "left", 1.6)]:
        ax.plot([x, x], [-0.55, 0.55], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.text(x + dx, 0.68, lab, color=col, ha=ha, va="bottom", size=14.5, weight="semibold")
    ax.annotate("", xy=(mean, -0.66), xytext=(med, -0.66),
                arrowprops=dict(arrowstyle="-|>", color=c["ochre"], lw=1.4, shrinkA=0, shrinkB=0, mutation_scale=11))
    ax.text(hi + 4, -0.62, f"{out.sum()} регионов за «усами»", color=c["muted"], size=12.5, va="top")
    ax.set_xlim(40, 225); ax.set_ylim(-0.95, 1.05)
    ax.set_yticks([])
    for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["faint"])
    ax.set_xticks([50, 100, 150, 200]); ax.tick_params(colors=c["muted"], labelsize=12, length=0, pad=6)
    ax.set_xlabel("Среднемесячная зарплата в 85 субъектах РФ, 2025 г., тыс. руб.", color=c["muted"], size=12.5, labelpad=8)
    fig.savefig(OUT / f"banner-{theme}.svg", facecolor=c["bg"], metadata={"Date": None})
    plt.close(fig)


def social():
    """Превью для ссылок (Open Graph), 1280×640: загружается вручную в Settings → Social preview."""
    c = THEMES["light"]
    w = pd.read_csv(ROOT / "data" / "wages_2025.csv")["wage"].to_numpy() / 1000
    mean, med = w.mean(), np.median(w)
    q1, q3 = np.percentile(w, [25, 75]); hi = q3 + 1.5 * (q3 - q1)
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=c["bg"])
    fig.text(0.06, 0.86, "Какую среднюю выбрать,\nесли в данных есть выбросы", family=SERIF, size=40,
             color=c["ink"], va="top", linespacing=1.12)
    fig.text(0.06, 0.55, "Имитационное исследование, машинное обучение и правило из двух условий",
             size=17, color=c["muted"], va="top")
    ax = fig.add_axes([0.06, 0.14, 0.88, 0.30], facecolor=c["bg"])
    rng = np.random.default_rng(7)
    y = rng.uniform(-0.32, 0.32, len(w)); out = w > hi
    ax.scatter(w[~out], y[~out], s=60, facecolors="none", edgecolors=c["dots"], linewidths=1.3, zorder=2)
    ax.scatter(w[out], y[out], s=60, facecolors=c["dots"], edgecolors=c["bg"], linewidths=1.0, zorder=2)
    for x, col, lab, ha, dx in [(med, c["blue"], f"медиана {comma(med)}", "right", -1.5),
                                (mean, c["ochre"], f"средняя {comma(mean)}", "left", 1.5)]:
        ax.plot([x, x], [-0.5, 0.5], color=col, lw=3, solid_capstyle="round", zorder=3)
        ax.text(x + dx, 0.62, lab, color=col, ha=ha, va="bottom", size=17, weight="semibold")
    ax.set_xlim(40, 225); ax.set_ylim(-0.7, 1.15); ax.set_yticks([])
    for sp in ("left", "right", "top"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(c["faint"])
    ax.set_xticks([50, 100, 150, 200]); ax.tick_params(colors=c["muted"], labelsize=13, length=0, pad=6)
    fig.text(0.06, 0.035, "Зарплата в 85 субъектах РФ, 2025 г., тыс. руб.", size=13, color=c["muted"])
    fig.text(0.94, 0.035, "github.com/MarkNightingale/mean-estimator-choice", size=13, color=c["muted"], ha="right")
    fig.savefig(OUT / "social-preview.png", facecolor=c["bg"], dpi=100)
    plt.close(fig)


def results(theme):
    c = THEMES[theme]
    t2 = json.load(open(ROOT / "results" / "tables.json"))["t2"]
    rows = {r[0]: (parse(r[1]), parse(r[2])) for r in t2}
    order = [("Случайный лес", "Случайный лес", "ours"), ("Правило из двух условий (мягкое)", "Правило из двух условий", "ours"),
             ("Медиана", "Медиана", "base"), ("Оценка Хьюбера", "Оценка Хьюбера", "base"),
             ("Оценка Ходжеса – Лемана", "Оценка Ходжеса – Лемана", "base"), ("Правило Хогга", "Правило Хогга", "base"),
             ("Средняя", "Средняя", "mean")]
    fig, ax = plt.subplots(figsize=(9.0, 3.9), dpi=100, facecolor=c["bg"])
    ax.set_facecolor(c["bg"])
    for i, (key, lab, kind) in enumerate(order):
        avg, worst = rows[key]
        y = len(order) - 1 - i
        col = {"ours": c["blue"], "mean": c["ochre"], "base": c["dots"]}[kind]
        ax.plot([avg, worst], [y, y], color=col, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([avg], [y], s=80, color=col, edgecolors=c["bg"], linewidths=2, zorder=3)
        ax.scatter([worst], [y], s=80, facecolors=c["bg"], edgecolors=col, linewidths=2, zorder=3)
        ax.text(0.86, y, lab, ha="right", va="center", size=11.5, color=c["ink"],
                weight="semibold" if kind == "ours" else "normal")
    ax.set_xscale("log"); ax.set_xlim(0.9, 60)
    ticks = [1, 1.5, 2, 3, 5, 10, 20, 50]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t).replace(".", ",") for t in ticks]); ax.minorticks_off()
    ax.set_yticks([]); ax.set_ylim(-0.7, len(order) - 0.3)
    for s in ("left", "right", "top"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(c["faint"])
    ax.grid(axis="x", color=c["faint"], lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(colors=c["muted"], labelsize=10, length=0, pad=6)
    ax.set_xlabel("MSE метода / MSE лучшей оценки сценария (лог. шкала)", color=c["muted"], size=10.5, labelpad=8)
    ax.scatter([], [], s=70, color=c["muted"], label="в среднем по 250 сценариям")
    ax.scatter([], [], s=70, facecolors=c["bg"], edgecolors=c["muted"], linewidths=2, label="в худшем сценарии")
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5, labelcolor=c["ink"], handletextpad=0.3)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.97, bottom=0.2)
    fig.savefig(OUT / f"results-{theme}.svg", facecolor=c["bg"], metadata={"Date": None})
    plt.close(fig)


LAB = {"mean": "среднее", "median": "медиана", "trim10": "усеч. 10 %", "huber": "Хьюбер", "hl": "Ходжес–Леман",
       "hogg": "схема Хогга", "logit": "логит", "rf": "лес", "rule": "2 порога", "soft": "2 порога (soft)", "caut": "2 порога (cautious)"}


def num(v, d=3):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def md(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join([":--"] + ["--:"] * (len(head) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def results_md():
    R = ROOT / "results"
    L = pd.read_csv(R / "long.csv")
    parts = ["# Результаты в числах", "",
             "Страница собирается скриптом `docs/make_assets.py` прямо из CSV-файлов в `results/` и показывает числа с большей точностью, чем в тексте статьи. "
             "Везде, где не сказано иное, `rel` — это MSE метода, делённая на MSE «оракула» (лучшей из пяти оценок в том же сценарии).", ""]
    # 1. обучающие распределения
    d = L[(L.setting == "A") & L.dist.isin(["normal", "t3"])]
    rows = []
    for m in ["mean", "median", "trim10", "huber", "hl", "hogg", "logit", "rf", "rule"]:
        x = d[d.method == m]; c, k = x[x.eps == 0], x[x.eps > 0]
        rows.append([LAB[m] if m != "rule" else "2 порога (soft)", num(x.rel.mean()), num(x.rel.max()), num(c.rel.mean()), num(k.rel.mean()),
                     num((c.rel.mean() + k.rel.mean()) / 2), num(k.big.mean() * 100, 2), num(x.mae.mean())])
    parts += ["## N(0, 1) и t(3): 250 сценариев, перекрёстная проверка", "",
              md(["метод", "rel, среднее", "rel, максимум", "rel при ε = 0", "rel при ε > 0", "rel, 50/50", "промахи > 0,25 при ε > 0, %", "MAE"], rows), "",
              "«50/50» — сценарии без засорения и с засорением учитываются с одинаковым весом.", ""]
    # 2. перенос: строки — распределение и набор обучения
    rows = []
    for dist, st in [("laplace", "A"), ("t2", "A"), ("t2", "B"), ("logistic", "A"), ("logistic", "B")]:
        x = L[(L.setting == st) & (L.dist == dist)]
        g = x.groupby("method").rel
        rows.append([f"{dist}, обучение {st}"] + [f"{num(g.mean()[m])} / {num(g.max()[m], 2)}" for m in ["median", "huber", "hl", "hogg", "logit", "rf", "rule"]])
    parts += ["## Распределения, отложенные от обучения", "",
              "В ячейке — среднее и максимум `rel` по 125 сценариям. Набор A обучен на N(0, 1) и t(3), набор B — ещё и на Лапласе; «2 порога» в наборе A — вариант soft, в наборе B — cautious.", "",
              md(["распределение", "медиана", "Хьюбер", "Ходжес–Леман", "схема Хогга", "логит", "лес", "2 порога"], rows), ""]
    # 3. смещение: строки — доля выбросов
    B = pd.read_csv(R / "bias.csv").groupby("eps").mean(numeric_only=True)
    cols = ["mean", "median", "trim10", "huber", "hl", "hogg", "rf", "soft", "caut"]
    rows = [[f"ε = {num(e, 2)}"] + [num(B.loc[e, m]) for m in cols] for e in B.index]
    parts += ["## Смещение при засорении с одной стороны", "",
              "Среднее смещение в единицах масштаба по 36 сценариям (N(0, 1) и t(3); n = 20 и 100; k = 3, 5, 10), все выбросы справа.", "",
              md(["доля выбросов"] + [LAB[m] for m in cols], rows), ""]
    # 4. регионы
    E = pd.read_csv(R / "regions.csv").set_index("key")
    NM = {"wage": "зарплата, тыс. руб.", "income": "доходы, тыс. руб./мес.", "unemp": "безработица, %", "food": "набор продуктов, тыс. руб.", "housing": "жильё, тыс. руб./м²"}
    sc = {"wage": 1e3, "income": 1e3, "unemp": 1, "food": 1e3, "housing": 1e3}
    rows = []
    for k in NM:
        f = sc[k]
        rows.append([NM[k], int(E.loc[k, "n"]), num(E.loc[k, "est_mean"] / f, 3), num(E.loc[k, "est_median"] / f, 3), num(E.loc[k, "est_huber"] / f, 3),
                     num(E.loc[k, "est_hl"] / f, 3), num(E.loc[k, "out_share"] * 100, 2), num(E.loc[k, "mm"]), f"{LAB[E.loc[k, 'rf']]} ({num(E.loc[k, 'rf_p'])})"])
    parts += ["## Росстат, 2025 г.: оценки центра по субъектам", "",
              md(["ряд", "n", "среднее", "медиана", "Хьюбер", "Ходжес–Леман", "за «усами», %", "отрыв x̄ от Me, в MAD", "выбор леса (вероятность)"], rows), "",
              "Оба варианта двухпорогового правила и схема Хогга для всех пяти рядов выбирают медиану.", ""]
    # 5. стресс-тест
    S = pd.read_csv(R / "semi.csv")
    rows = [[NM[q.key].split(",")[0], int(q.m), num(q.mean_mean, 2), num(q.median_mean, 2), num(q.median_p95, 2), num(q.huber_mean, 2), num(q.rf_mean, 2)] for q in S.itertuples()]
    parts += ["## Стресс-тест «лишний ноль»", "",
              "m случайных значений ряда умножаются на 10; 2 000 повторов. Средний |сдвиг| оценки в процентах; для медианы дан и 95-й процентиль.", "",
              md(["ряд", "m", "среднее", "медиана", "медиана, p95", "Хьюбер", "лес"], rows), ""]
    # 6. бутстреп
    b = pd.read_csv(R / "bootstrap.csv")
    rows = [[f"{LAB[r.a] if r.a != 'rule' else '2 порога (soft)'} − {LAB[r.b] if r.b != 'rule' else '2 порога (soft)'}", num(r.diff_all, 4), f"[{num(r.lo_all, 4)}; {num(r.hi_all, 4)}]",
             num(r.diff_eq, 4), f"[{num(r.lo_eq, 4)}; {num(r.hi_eq, 4)}]", f"{r.share_better:.0%}"] for r in b.itertuples()]
    parts += ["## Парный бутстреп по сценариям", "",
              "Разность средних `rel` (минус — первый метод точнее) и 2,5–97,5 % перцентили по 10 000 повторам; последний столбец — доля сценариев, где первый метод точнее.", "",
              md(["пара", "Δ rel", "интервал", "Δ rel, 50/50", "интервал, 50/50", "первый точнее"], rows), ""]
    (ROOT / "docs" / "results.md").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    for th in THEMES:
        banner(th); results(th)
    social()
    results_md()
    print("готово:", sorted(p.name for p in OUT.iterdir()))
