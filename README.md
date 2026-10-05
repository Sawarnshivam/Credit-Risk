# Credit Default Risk — Probability of Default model

This is a project I built to predict the probability that a loan applicant will
default, using the public **Home Credit Default Risk** dataset from Kaggle. It
goes from the raw application and history tables all the way to a calibrated
model, a cost-based decision threshold, and SHAP explanations.

Everything here is trained on public Kaggle data — none of it is proprietary or
from any company.

The numbers in this README come from actually running the pipeline. I tried hard
not to hand-wave any results.

## What it does

- Cleans the application data and fixes a few known issues in it.
- Builds 375 features: affordability ratios and score combinations from the
  application table, plus aggregates over the bureau, previous-application,
  POS-cash and credit-card history tables.
- Trains and compares four models (logistic regression, random forest, XGBoost,
  LightGBM).
- Studies the class imbalance (the default rate is only ~8%) and compares class
  weighting, SMOTE and threshold tuning.
- Tunes the boosted models with Optuna, calibrates the best one, and picks a
  decision threshold from a simple cost model.
- Explains the model with SHAP and wraps everything in a small prediction class.
- Tracks the model runs with MLflow.

## Results (on a 20% hold-out of the training data)

Best model is **LightGBM**:

| model | ROC-AUC | PR-AUC | KS | Brier |
|-------|--------:|-------:|---:|------:|
| Logistic Regression | 0.772 | 0.263 | 0.406 | 0.067 |
| Random Forest | 0.762 | 0.246 | 0.392 | 0.068 |
| XGBoost | 0.786 | 0.285 | 0.436 | 0.066 |
| **LightGBM** | **0.787** | **0.289** | 0.433 | 0.066 |

I focused on ROC-AUC, PR-AUC, KS and Brier instead of accuracy, because with an
8% default rate you can get ~92% accuracy by predicting "no default" for
everyone, so accuracy is meaningless here.

A few things I found interesting while doing this:

- **The threshold matters more than resampling.** At the default 0.5 cutoff the
  model almost never flags a default (recall ~0.04) even though its ranking is
  good. Class weighting brings recall up to ~0.64, and just moving the threshold
  to 0.15 gives the best F1 (0.33).
- **SMOTE made things worse** (AUC dropped from 0.78 to 0.76). I left it in the
  comparison because a negative result is still a result — on this data,
  synthesising minority points in a 375-dim space added noise.
- **Tuning barely helped.** Optuna moved CV PR-AUC by a tiny amount over the
  defaults. The feature engineering did much more than the hyperparameters.
- The top SHAP features are the external credit scores and my engineered ratios
  (`EXT_SOURCE_MEAN`, `CREDIT_ANNUITY_RATIO`, `CREDIT_GOODS_RATIO`), which was
  nice to see because it means the feature work actually mattered.

The full write-up with all the tables is in
[`reports/model_report.md`](reports/model_report.md).

## Decision bands

From the cost sweep (assuming a false negative costs 5x a false positive), the
lowest-cost threshold is about 0.15, which gives these bands:

- **APPROVE** if PD < 0.15
- **REVIEW** if 0.15 ≤ PD < 0.20
- **REJECT** if PD ≥ 0.20

## Project layout

```
configs/config.yaml         all paths and settings
src/
  data/                     loading + cleaning + encoding
  features/                 feature engineering + streaming aggregation
  models/                   model builders, calibration, prediction
  evaluation/               metrics, threshold sweep, plotting, MLflow
  explainability/           SHAP
main.py                     runs each stage from the command line
tests/                      unit tests (pytest)
notebooks/                  EDA, feature engineering, modeling, explainability
reports/                    metrics tables, figures, and the model report
```

## How to run it

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Download the Home Credit data from Kaggle and put the CSVs in `data/raw/`
   (see [`data/README.md`](data/README.md) for the file list).

3. Run the stages in order (or use `all`):

   ```bash
   python main.py features     # build the feature matrices
   python main.py train        # compare the four models
   python main.py imbalance    # class-weight / SMOTE / threshold study
   python main.py tune         # Optuna tuning (XGBoost + LightGBM)
   python main.py calibrate    # fit + calibrate the final model
   python main.py threshold    # cost-based threshold + risk bands
   python main.py shap         # SHAP importance + explanations
   python main.py predict      # score a few raw applicants end to end
   ```

   Each stage writes its outputs to `reports/` and `models/`.

4. See the MLflow runs:

   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

## A note on the machine I ran this on

I built this on a small box (about 4 GB of RAM and a single CPU core), so a few
stages run on stratified subsamples to fit in memory, and I kept the Optuna
search small. I've written those choices down in the code comments and in the
model report so it's clear what was full-data and what wasn't. The gradient
boosting comparison uses all the training rows; the linear/RF baselines, the
imbalance study, and tuning use subsamples. On a bigger machine you can raise
those limits in `configs/config.yaml`.

## Tests

```bash
python -m pytest tests/ -q
```

The tests check the cleaning logic, the feature ratios, the metric functions,
and — most importantly — that the streaming aggregator gives the same answer as
a plain pandas groupby.
