# Данные

Показатели Росстата по субъектам Российской Федерации за 2025 г., полученные через сервис StatBase
(дата обращения 28.10.2026). Формат файлов `.txt` — «субъект;значение», без заголовка.

| Файл | Показатель | Субъектов | Значение по РФ |
|---|---|---|---|
| `wages_2025.csv` | Среднемесячная номинальная начисленная заработная плата работников организаций, руб. | 85 | 100 360 |
| `income_2025.txt` | Среднедушевые денежные доходы населения, руб. в месяц | 85 | 74 845 |
| `unemp_2025.txt` | Уровень безработицы, % | 85 | 2,2 |
| `food_2025.txt` | Стоимость минимального набора продуктов питания, руб. (декабрь) | 85 | 7 347,16 |
| `housing_2025.txt` | Средняя цена на вторичном рынке жилья, руб. за кв. м | 83 | 130 359 |

Источники:
- https://statbase.ru/datasets/national-statistics-of-russia/labor-income-prices/average-monthly-nominal-wages-by-region-national-stat/
- https://statbase.ru/data/rus-average-income-per-capita-by-region-national-stat/
- https://statbase.ru/data/rus-unemployment-rate-by-region-national-stat/
- https://statbase.ru/data/rus-cost-of-minimum-set-of-food-products-national-stat/
- https://statbase.ru/data/ru99-average-prices-in-the-secondary-housing-market-by-region-national-stat/

Перед использованием в других работах значения стоит сверить с публикациями Росстата.

## Контрольные суммы (SHA-256)

Позволяют убедиться, что файлы не менялись.

| Файл | SHA-256 |
|---|---|
| `food_2025.txt` | `aa5fe556ea0d5145e5ebd546305c50ffad5f4747d5cd0ff09834e3db3cefdace` |
| `housing_2025.txt` | `5c955b88f32e01f08042d772d71295da61016bd4dc98df09422ccb4b37a5b21d` |
| `income_2025.txt` | `00ee3e5798015bd52059fcd222cd512b3b324865d901a9ab94bd8f20e32035e9` |
| `unemp_2025.txt` | `47c960556f676ba0ea2d0505465768bef741b8d3316b270c3a5182d771b2ef5e` |
| `wages_2025.csv` | `32c3a40911d7f4117d162ae10755c3f6ecbde5073bb201e513fe781c75bfb48a` |
