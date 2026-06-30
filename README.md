# Customer Churn Prediction — Decision Tree Tuning

A Moringa School Data Science (Module 4) lab that builds and tunes a decision tree classifier to predict customer churn for a telecommunications company.

## Business Context

Telecom companies lose significant revenue when customers churn (cancel their service). Since acquiring a new customer typically costs 5–25x more than retaining an existing one, this project prioritizes **recall** over precision — it's more costly to miss an at-risk customer (false negative) than to flag a loyal one unnecessarily (false positive).

## Dataset

`telecom_churn_processed.csv` — 20,000 customer records, 13 pre-processed numeric features (account age, monthly charges, subscription tier, support calls, usage decline, contract type, etc.) and a binary `churn` target (0 = retained, 1 = churned). Class distribution is moderately imbalanced (~34% churn).

## Workflow

1. **Baseline model** — Default `DecisionTreeClassifier`, evaluated with 5-fold cross-validated recall.
2. **Randomized search** — Broad exploration over `criterion`, `max_depth`, `min_samples_split`, and `min_samples_leaf` to find promising hyperparameter regions.
3. **Grid search** — Focused search around the best randomized-search parameters to fine-tune the model.
4. **Final validation** — Best model evaluated on a held-out test set with recall score, confusion matrix, and feature importance.

## Key Results

| Stage | CV / Train Recall |
|---|---|
| Baseline (default params) | Train: 1.000 · CV: 0.879 (overfit) |
| Randomized search (best) | Train: 0.935 · CV: 0.899 |
| Grid search (best) | Train: 0.933 · CV: 0.891 |
| **Final test set** | **0.888** |

The baseline model overfits heavily (perfect training recall vs. lower cross-validated recall). Hyperparameter tuning narrows this gap substantially, and the final test recall (0.888) tracks closely with cross-validation performance, indicating a well-generalized model.

**Top churn predictors:** `monthly_charges`, `usage_decline`, and `subscription_tier`.

## Repository Contents

- `customer_churn_dt.ipynb` — Full notebook: data exploration, baseline model, randomized search, grid search, final evaluation, confusion matrix, and feature importance.
- `telecom_churn_processed.csv` — Pre-processed dataset used in the notebook.

## Requirements

```
pandas
numpy
matplotlib
scikit-learn
```

## Running

```bash
jupyter notebook customer_churn_dt.ipynb
```

Ensure `telecom_churn_processed.csv` is in the same directory as the notebook.
