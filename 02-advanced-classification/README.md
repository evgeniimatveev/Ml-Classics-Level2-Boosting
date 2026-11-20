# 🎯 Section 2: Advanced Classification

Binary classification with accuracy, F1 and ROC-AUC: which bank customers will leave, and which insurance policies cost more than the median.

> 📂 The notebooks read their data from `../data/`: open them from this folder in Jupyter (the default) and the paths resolve.

| # | Notebook | What it covers | Dataset |
|:-:|----------|----------------|---------|
| 1 | [`01-xgboost-classifier.ipynb`](01-xgboost-classifier.ipynb) | XGBoost classifier: train/test accuracy, cross-validation, ROC-AUC, tuning | `data/churn-modelling.csv` |
| 2 | [`02-lightgbm-classifier.ipynb`](02-lightgbm-classifier.ipynb) | LightGBM classifier: is the insurance charge above the median? Compared with XGBoost and Random Forest, then tuned | `data/insurance.csv` |
| 3 | [`03-catboost-classifier.ipynb`](03-catboost-classifier.ipynb) | CatBoost classifier and a voting ensemble, final comparison table | `data/churn-modelling.csv` |

---

[← Advanced Regression](../01-advanced-regression/) &nbsp;·&nbsp; [Model Comparison →](../03-model-comparison/)

*Built by [Evgenii Matveev](https://github.com/evgeniimatveev) · Data Analyst / Analytics & MLOps Engineer*
