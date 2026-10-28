"""Общие пути и режим быстрой проверки для скриптов эксперимента.

Переменные окружения:
  RESULTS_DIR, FIGURES_DIR — куда писать результаты и рисунки (по умолчанию results/ и figures/);
  QUICK=1 — сокращённый прогон для проверки, что весь конвейер работает (числа не совпадут со статьёй).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # пакет estimator_choice доступен и без установки

DATA = ROOT / "data"
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
FIGURES = Path(os.environ.get("FIGURES_DIR", ROOT / "figures"))
QUICK = os.environ.get("QUICK") == "1"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def quick(full, small):
    """Значение параметра: полное или сокращённое в режиме QUICK."""
    return small if QUICK else full
