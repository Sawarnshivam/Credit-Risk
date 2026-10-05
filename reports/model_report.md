# Model report — Credit default risk / probability of default

All numbers below come from actual runs on the Home Credit data. The model is
evaluated on a single stratified hold-out split (20% of `application_train`,
seed 42). Where a stage was run on a subsample for memory reasons, that is
stated.

## 1. Data and features

- Training rows: 307,511. Hold-out (test-of-train) split: 20%.
- Default rate: **8.07%** — a strongly imbalanced problem.
- Raw application columns: 120 (after dropping id/target).
- Final feature count: **375**.

Features come from three places:

1. **Application-level ratios / interactions** (built in
   `application_features`): affordability ratios such as
   `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_ANNUITY_RATIO`,
   `CREDIT_GOODS_RATIO`, per-person / per-child income, age and employment in
   years, and combinations of the three external credit scores
   (`EXT_SOURCE_MEAN/MIN/MAX/STD/PROD`).
2. **Relational aggregates** over the bureau, bureau-balance, previous
   application, POS-cash and credit-card tables — counts, means, min/max, sums
   and a few rates (e.g. approval rate, DPD rate). These are prefixed `BURO_`,
   `BB_`, `PREV_`, `POS_`, `CC_`.
3. The installments table was not available in this run, so its aggregates are
   not included (see `data/README.md`).

The aggregates are computed with a **two-pass streaming group aggregator**
(`ChunkedAggregator`) rather than a single in-memory groupby, because the source
tables are large (bureau_balance alone is ~27M rows) and the machine used here
had ~4 GB of RAM. The aggregator's arithmetic is unit-tested against a plain
pandas groupby (`tests/test_features.py`).

A quick leakage check on the built matrix showed the highest absolute
correlations with `TARGET` were the external-score features (|corr| ≈ 0.15–0.22),
which is expected and legitimate — no single feature was suspiciously predictive.

## 2. Model comparison

Four models were trained on the same features and evaluated on the same hold-out
set. Metrics are ranked by ROC-AUC / PR-AUC; because the classes are imbalanced,
**PR-AUC, KS and Brier matter more than accuracy** (accuracy is not reported —
predicting "no default" for everyone already scores ~92%).

| model | ROC-AUC | PR-AUC | KS | Gini | Brier |
|-------|--------:|-------:|---:|-----:|------:|
| Logistic Regression (baseline) | 0.7718 | 0.2627 | 0.4063 | 0.5435 | 0.0669 |
| Random Forest | 0.7616 | 0.2463 | 0.3916 | 0.5233 | 0.0682 |
| XGBoost | 0.7860 | 0.2854 | 0.4363 | 0.5721 | 0.0657 |
| **LightGBM** | **0.7874** | **0.2888** | 0.4326 | **0.5747** | **0.0656** |

Notes on how these were run, for honesty about the compute:

- Logistic Regression and Random Forest were fit on a stratified 150,000-row
  subsample (the machine could not hold the dense, one-hot + scaled matrix for
  logistic regression, and sklearn's Random Forest is memory-heavy). XGBoost and
  LightGBM used all 246,008 training rows.
- LightGBM and XGBoost are almost tied. **LightGBM was chosen as the final
  model** — it had the best PR-AUC and Brier, trains fast, uses little memory and
  handles NaNs natively, so it was the practical choice to carry through tuning,
  calibration and explanation.

## 3. Class imbalance

Using LightGBM on a 100,000-row subsample, four ways of handling the 8% positive
rate were compared (evaluated on the same hold-out set):

| strategy | ROC-AUC | PR-AUC | precision | recall | F1 |
|----------|--------:|-------:|----------:|-------:|---:|
| none (threshold 0.5) | 0.7788 | 0.2770 | 0.552 | 0.044 | 0.081 |
| class weights (`balanced`) | 0.7788 | 0.2735 | 0.200 | 0.636 | 0.304 |
| SMOTE (train only) | 0.7605 | 0.2430 | 0.485 | 0.042 | 0.078 |
| threshold tuning (τ = 0.15) | 0.7788 | 0.2770 | 0.264 | 0.453 | 0.333 |

Takeaways:

- At the default 0.5 threshold the model almost never predicts "default"
  (recall 0.04) even though ranking quality (AUC) is fine. This is the classic
  imbalance symptom.
- **Class weighting** recovers recall (0.04 → 0.64) at the same AUC, at the cost
  of precision.
- **Threshold tuning** gives the best F1 (0.333) without retraining — for a
  ranking model, moving the decision threshold is the cheapest and most
  effective lever.
- **SMOTE actually hurt** here (AUC 0.779 → 0.760). Synthesising minority rows
  in a 375-dimensional space added noise rather than signal. This matches common
  experience on real tabular credit data and is a good argument against reaching
  for SMOTE by default.

The practical conclusion carried forward: keep the well-calibrated ranking model
and choose an operating threshold from a cost analysis, rather than resampling.

## 4. Hyperparameter tuning (Optuna)

Both gradient-boosted models were tuned with Optuna (TPE sampler), maximising
3-fold CV **average precision (PR-AUC)** on a 60,000-row subsample, 15 trials
each. The search was deliberately small and single-threaded because of the
one-core / 4 GB box; studies are persisted to SQLite so they can resume.

| model | best CV PR-AUC | trials |
|-------|---------------:|-------:|
| XGBoost | 0.2527 | 15 |
| LightGBM | 0.2550 | 15 |

Best LightGBM parameters: `n_estimators=500, learning_rate≈0.037, num_leaves=41,
max_depth=4, min_child_samples=144, subsample≈0.81, colsample_bytree≈0.73`.

Honest note: tuning moved the needle only slightly over the defaults. On this
dataset the feature engineering matters much more than the hyperparameters,
which is worth stating plainly rather than overselling the search.

## 5. Calibration

The tuned LightGBM was calibrated with a held-out calibration split (Platt /
sigmoid and isotonic), using a prefit/frozen base estimator so the base model is
never retrained on the calibration data.

| variant | Brier |
|---------|------:|
| uncalibrated | 0.065867 |
| sigmoid (selected) | 0.065862 |
| isotonic | 0.065902 |

The differences are tiny — **LightGBM was already well-calibrated** on this data.
Sigmoid was selected as marginally best and saved as the final model. Its
hold-out metrics: ROC-AUC 0.7844, PR-AUC 0.2841, KS 0.4288, Gini 0.5689,
Brier 0.0659. (Slightly below the untuned full-data LightGBM because the final
model's base was fit on 80% of the training rows and the tuned parameters were
selected on a subsample — an honest cost of the memory-limited setup.)

## 6. Threshold selection with a cost framework

Default risk is asymmetric: approving someone who defaults (false negative)
costs more than declining someone who would have repaid (false positive). Using
a simple cost ratio of **5 : 1** (FN : FP), the expected cost per applicant was
swept across thresholds:

| threshold | precision | recall | approval rate | expected cost/app |
|-----------|----------:|-------:|--------------:|------------------:|
| 0.05 | 0.141 | 0.844 | 0.517 | 0.478 |
| 0.10 | 0.206 | 0.631 | 0.753 | 0.345 |
| **0.15** | 0.264 | 0.469 | 0.857 | **0.320** |
| 0.20 | 0.318 | 0.346 | 0.912 | 0.324 |
| 0.25 | 0.365 | 0.252 | 0.944 | 0.337 |
| 0.50 | 0.598 | 0.040 | 0.995 | 0.390 |

The minimum expected cost is at **τ ≈ 0.15**. From the sweep the pipeline derives
three risk bands used by the prediction service:

- **APPROVE** (LOW): PD < 0.15
- **REVIEW** (MEDIUM): 0.15 ≤ PD < 0.20
- **REJECT** (HIGH): PD ≥ 0.20

## 7. Explainability (SHAP)

TreeSHAP was run on the final LightGBM (unwrapped from the calibrator) over a
4,000-row validation sample. Top features by mean |SHAP|:

| rank | feature | mean \|SHAP\| |
|------|---------|-----------:|
| 1 | EXT_SOURCE_MEAN | 0.397 |
| 2 | CODE_GENDER | 0.113 |
| 3 | CREDIT_ANNUITY_RATIO | 0.097 |
| 4 | CREDIT_GOODS_RATIO | 0.088 |
| 5 | AMT_ANNUITY | 0.081 |
| 6 | NAME_EDUCATION_TYPE | 0.081 |
| 7 | EXT_SOURCE_1 | 0.072 |
| 8 | OWN_CAR_AGE | 0.071 |
| 9 | EXT_SOURCE_MAX | 0.070 |
| 10 | DAYS_EMPLOYED | 0.061 |

Two things worth highlighting:

- The **engineered features earn their place**: `EXT_SOURCE_MEAN`,
  `CREDIT_ANNUITY_RATIO` and `CREDIT_GOODS_RATIO` are all in the top four, and
  relational aggregates (`BURO_AMT_CREDIT_SUM_DEBT_MEAN`, `POS_MONTHS_BALANCE_STD`,
  `PREV_ASK_TO_GRANT_MAX`) appear further down. The feature work is doing real
  work, not just padding the column count.
- SHAP is used for **attribution, not causation** — it explains what drives a
  given score, which is what is needed for adverse-action style reason codes, not
  a claim about what causes default.

Per-applicant explanations (top risk-increasing and risk-reducing features for
five example applicants) are saved in
`reports/tables/sample_predictions.json`.

## 8. Limitations

- One hold-out split, not repeated CV, for the headline metrics (CV was used
  inside tuning). Numbers would wobble a little across seeds.
- The installments table is missing, so one family of features is absent.
- The linear and Random-Forest baselines, the imbalance study and tuning all ran
  on subsamples to fit in ~4 GB of RAM. The gradient-boosted comparison used the
  full training rows.
- Single-record scoring in `predict.py` uses application features only;
  relational aggregates default to NaN for a brand-new applicant with no history
  in the other tables.
