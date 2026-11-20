# 📈 Section 1: Advanced Regression

Predict medical insurance charges with the three big boosting libraries. Every notebook follows the same path: baseline, cross-validation, hyperparameter search, tuned model.

> 📂 The notebooks read their data from `../data/`: open them from this folder in Jupyter (the default) and the paths resolve.

| # | Notebook | What it covers | Dataset |
|:-:|----------|----------------|---------|
| 1 | [`01-xgboost-regressor.ipynb`](01-xgboost-regressor.ipynb) | XGBoost vs Random Forest: before and after hyperparameter tuning | `data/insurance.csv` |
| 2 | [`02-lightgbm-regressor.ipynb`](02-lightgbm-regressor.ipynb) | LightGBM, with XGBoost and Random Forest as references: cross-validation and tuning | `data/insurance.csv` |
| 3 | [`03-catboost-regressor.ipynb`](03-catboost-regressor.ipynb) | CatBoost: grid and randomized hyperparameter search (720 fits), final metrics | `data/insurance.csv` |

---

[← Quickstart](../00-quickstart/) &nbsp;·&nbsp; [Advanced Classification →](../02-advanced-classification/)

*Built by [Evgenii Matveev](https://github.com/evgeniimatveev) · Data Analyst / Analytics & MLOps Engineer*
