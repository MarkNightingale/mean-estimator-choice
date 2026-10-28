PY ?= python

.PHONY: all simulate models rules summary empirical tables figures assets quick test check help

help:  ## список команд
	@grep -E "^[a-z]+:.*##" Makefile | sed -E "s/:.*## /\t/"

all: simulate models rules summary empirical quick  ## полный расчёт с нуля (около часа)

simulate:  ## имитационный эксперимент: 625 сценариев x 5 000 выборок
	$(PY) src/run2.py

models:  ## случайный лес и логистическая регрессия, перекрёстная проверка
	$(PY) src/ml2.py

rules:  ## подбор порогов правила из двух условий
	$(PY) src/rule2.py

summary:  ## сводная таблица по методам и смещение при односторонних выбросах
	$(PY) src/tab2.py
	$(PY) src/bias2.py

empirical:  ## региональные показатели и полусинтетический эксперимент
	$(PY) src/emp2.py

tables:  ## таблицы статьи из сохранённых результатов
	$(PY) src/tables2.py

figures:  ## рисунки статьи и их исходные данные
	$(PY) src/figs2.py
	$(PY) src/figdata.py

assets:  ## иллюстрации README и docs/results.md
	$(PY) docs/make_assets.py

quick: tables figures assets  ## всё, что считается за секунды по сохранённым результатам

test:  ## автотесты
	$(PY) -m pytest -q

check: tables  ## таблицы из сохранённых результатов совпадают с опубликованными
	git diff --exit-code results/tables2.json
