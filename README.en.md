<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img alt="Average monthly wages in 85 Russian regions, 2025: median 75.5 thousand roubles, unweighted mean 85.9 thousand pulled up by nine regions beyond the boxplot whiskers" src="docs/assets/banner-light.svg" width="100%">
</picture>

[![Tests](https://github.com/MarkNightingale/mean-estimator-choice/actions/workflows/ci.yml/badge.svg)](https://github.com/MarkNightingale/mean-estimator-choice/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-1f6fae)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-1f6fae)](requirements.txt)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-c0661a)](CITATION.cff)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarkNightingale/mean-estimator-choice/blob/main/notebooks/demo.ipynb)

**[Русская версия](README.md)**. Figures and detailed documentation are in Russian.

The arithmetic mean misdescribes the typical level of an indicator when the data contain outliers, while the median loses precision on clean data. Using 625 Monte Carlo scenarios, we test whether a machine learning model can choose the estimator from the characteristics of a particular sample, and reduce what it learns to a rule that needs no programming.

Code, data and results for a paper by E.Yu. Shimanskaya, M.V. Solovev and A.E. Skhvediani (Peter the Great St. Petersburg Polytechnic University). The paper is in preparation.

## The two-condition rule

> If **at most 5%** of observations lie beyond the boxplot whiskers and the mean differs from the median by **at most 0.20 MAD**, use the Huber estimator. Otherwise use the median.

MAD is scaled by 1.483. This is the cautious version, recommended when nothing is known about the data in advance. The soft version (16% and 0.30 MAD) is more precise on clean data but can miss close one-sided outliers.

Try it in the [Colab notebook](https://colab.research.google.com/github/MarkNightingale/mean-estimator-choice/blob/main/notebooks/demo.ipynb) without installing anything, or locally:

```bash
pip install git+https://github.com/MarkNightingale/mean-estimator-choice
choose-estimator data/wages_2025.csv            # or: --variant soft
```

From Python: `from estimator_choice import choose_estimator; choose_estimator(values)`.

## Findings

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-dark.svg">
  <img alt="MSE relative to the scenario-best estimator, on average and in the worst scenario: random forest 1.05 and 1.52; two-condition rule 1.08 and 2.39; median 1.11 and 1.60; Huber 1.27 and 3.20; Hodges–Lehmann 1.46 and 8.73; Hogg's rule 2.41 and 30.0; mean 5.00 and 42.8" src="docs/assets/results-light.svg" width="100%">
</picture>

*Normal and Student (3 df) distributions, 250 scenarios, scenario-grouped cross-validation. A value of 1 means the precision of the scenario-best estimator.*

- **No estimator is best everywhere.** The median wins with outliers, the Huber estimator without them, the mean almost never.
- **The model beats every fixed estimator** on average and in the worst case, but the gain over the median is modest: 5% vs 11% excess MSE (scenario bootstrap 95% interval for the difference: 3.9–7.2 percentage points).
- **Hogg's (1974) rule fails with one-sided outliers**: with 20% of them its MSE is 8.6 times the minimum.
- **Transfer is partial.** On the Laplace and Student (2 df) distributions the model trails the median by 5–8%; on the logistic it beats it.
- **Regional data.** For five Rosstat indicators in 2025 almost all procedures choose the median; the unweighted mean exceeds it by 7–41%.

## Reproduce

```bash
pip install -r requirements.txt -r requirements-dev.txt
make quick   # tables and figures from saved results, seconds
make test    # unit tests
make smoke   # the whole pipeline in reduced form, about 2 minutes
make all     # full run from scratch, about 35 minutes
```

All random seeds are fixed. Large intermediate files (`store.pkl`, `rf_*.pkl`, ~370 MB) are not stored; `experiments/simulate.py` and `experiments/train_models.py` recreate them. See [docs/methodology.md](docs/methodology.md) for the design and [docs/results.md](docs/results.md) for all tables.

## Citation

Use **Cite this repository** on the right (from [CITATION.cff](CITATION.cff)). The paper reference will be added after publication.

## Authors

- **Elizaveta Yu. Shimanskaya** — Master’s Student
- **Mark V. Solovev** — Assistant, Graduate School of Industrial Economics ([ORCID 0000-0003-4626-6595](https://orcid.org/0000-0003-4626-6595))
- **Angi E. Skhvediani** — Cand. Sci. (Econ.), Associate Professor, Graduate School of Industrial Economics ([ORCID 0000-0001-7171-7357](https://orcid.org/0000-0001-7171-7357))

Peter the Great St. Petersburg Polytechnic University, 29 Politekhnicheskaya St., St. Petersburg, 195251, Russia. Questions about the code: [Issues](https://github.com/MarkNightingale/mean-estimator-choice/issues).

## License

Code: [MIT](LICENSE). Data: open data of Rosstat (Federal State Statistics Service of Russia); please cite the original source.
