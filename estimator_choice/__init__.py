"""Выбор оценки типичного уровня при выбросах.

Основное — функция choose_estimator (правило из двух условий) и модуль estimators
с оценками, статистикой Хогга и характеристиками выборки.
"""
from .rule import THRESHOLDS, choose_estimator, summary

__all__ = ["choose_estimator", "summary", "THRESHOLDS"]
__version__ = "1.1.0"
