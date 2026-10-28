PY ?= python
X = $(PY) experiments

.PHONY: help all simulate models rules summary regions tables figures assets quick test check smoke

help:  ## список команд
	@grep -E "^[a-z]+:.*##" Makefile | sed -E "s/:.*## /\t/"

all: simulate models rules summary regions quick  ## полный расчёт с нуля (около 35 минут)

simulate:  ## шаг 1: 625 сценариев x 5 000 выборок
	$(X)/simulate.py

models:  ## шаг 2: случайный лес и логистическая регрессия
	$(X)/train_models.py

rules:  ## шаг 3: пороги правила из двух условий
	$(X)/tune_rule.py

summary:  ## шаг 4: сводная таблица, смещение, бутстреп-интервалы
	$(X)/summarize.py
	$(X)/bias.py
	$(X)/bootstrap.py

regions:  ## шаг 5: показатели Росстата и ошибки с лишним разрядом
	$(X)/regions.py

tables:  ## таблицы статьи из сохранённых результатов
	$(X)/tables.py

figures:  ## рисунки статьи и их данные в Excel
	$(X)/figures.py
	$(X)/figures_data.py

assets:  ## иллюстрации README и docs/results.md
	$(PY) docs/make_assets.py

quick: tables figures assets  ## всё, что считается за секунды по сохранённым результатам

test:  ## автотесты
	$(PY) -m pytest -q

check: tables  ## таблицы из сохранённых результатов совпадают с опубликованными
	git diff --exit-code results/tables.json

smoke:  ## весь конвейер в сокращённом виде во временной папке (около 2 минут)
	$(eval TMP := $(shell mktemp -d))
	QUICK=1 RESULTS_DIR=$(TMP) FIGURES_DIR=$(TMP) $(MAKE) --no-print-directory simulate models rules summary regions tables figures
	@echo "Сокращённый прогон прошёл: $(TMP)"
