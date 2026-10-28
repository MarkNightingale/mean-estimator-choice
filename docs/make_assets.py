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
num = lambda s: float(s.replace(" ", "").replace(",", "."))
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
    rows = {r[0]: (num(r[1]), num(r[2])) for r in t2}
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


def md_table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join([":--"] + ["--:"] * (len(head) - 1)) + "|"]
    for r in rows:
        if r[0] == "sub":
            out.append(f"| *{r[1]}* |" + " |" * (len(head) - 1))
        else:
            out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def boot_rows():
    b = pd.read_csv(ROOT / "results" / "bootstrap.csv")
    ru = {"rf": "Случайный лес", "median": "медиана", "huber": "оценка Хьюбера", "rule": "мягкое правило", "logit": "логит"}
    f = lambda v: f"{v:.3f}".replace(".", ",").replace("-", "−")
    first = lambda t: t[0].upper() + t[1:]
    return [[f"{first(ru[r.a])} − {ru[r.b]}", f(r.diff_all), f"{f(r.lo_all)}…{f(r.hi_all)}", f(r.diff_eq), f"{f(r.lo_eq)}…{f(r.hi_eq)}",
             f"{r.share_better * 100:.0f}"] for r in b.itertuples()]


def results_md():
    T = json.load(open(ROOT / "results" / "tables.json"))
    parts = ["# Результаты: все таблицы статьи", "",
             "Файл создаётся скриптом `docs/make_assets.py` из `results/tables.json`; числа совпадают со статьёй. "
             "Отношение MSE — MSE метода, делённая на MSE лучшей для сценария оценки (1,00 — как у лучшей оценки).", "",
             "## Точность на обучающих распределениях", "",
             "Нормальное распределение и распределение Стьюдента (3 ст. св.), 250 сценариев, перекрёстная проверка по сценариям.", "",
             md_table(T["t2"], ["Метод", "В среднем", "Наибольшее", "Без выбросов", "С выбросами", "Равные веса", "Серьёзные промахи, %"]), "",
             "«Без выбросов» и «с выбросами» — средние по 10 и 240 сценариям, «равные веса» — их полусумма. "
             "Серьёзный промах — ошибка больше 0,25 единицы масштаба; доля приведена для сценариев с выбросами.", "",
             "## Смещение при односторонних выбросах", "",
             "Среднее смещение в единицах масштаба; все выбросы справа; 36 сценариев (нормальное и Стьюдента, n = 20 и 100, k = 3, 5, 10).", "",
             md_table(T["t3"], ["Метод", "5 % выбросов", "10 %", "20 %"]), "",
             "## Перенос на новые распределения", "",
             md_table(T["t4"], ["Метод", "Лаплас, в среднем", "Лаплас, наиб.", "Стьюдент (2), в среднем", "Стьюдент (2), наиб.",
                                "Логистическое, в среднем", "Логистическое, наиб."]), "",
             "Для моделей, обученных с распределением Лапласа, результаты на нём не приводятся.", "",
             "## Региональные показатели, 2025 г.", "",
             md_table(T["t5"], ["Характеристика", "Зарплата, руб.", "Доходы, руб. в месяц", "Безработица, %", "Набор продуктов, руб.", "Жильё, руб. за кв. м"]), "",
             "## Ошибки с лишним разрядом", "",
             "Среднее по 2 000 повторениям абсолютное изменение оценки, % от значения по исходным данным; m — число субъектов, где значение умножено на 10.", "",
             md_table(T["t6"], ["Показатель", "m", "Средняя", "Усечённая", "Хьюбер", "Ходжес – Леман", "Медиана", "Случайный лес"]), "",
             "## Устойчивость различий: бутстреп по сценариям", "",
             "Разность средней относительной MSE двух методов (отрицательная — первый точнее) и 95-процентный интервал парного бутстрепа "
             "по 250 сценариям (10 000 повторений). «Чаще точнее» — доля сценариев, где первый метод точнее второго.", "",
             md_table(boot_rows(), ["Сравнение", "Разность", "Интервал", "Разность (равные веса)", "Интервал (равные веса)", "Чаще точнее, %"]), ""]
    (ROOT / "docs" / "results.md").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    for th in THEMES:
        banner(th); results(th)
    social()
    results_md()
    print("готово:", sorted(p.name for p in OUT.iterdir()))
