# Data

This project uses the **Home Credit Default Risk** dataset, which is public and
available on Kaggle:
https://www.kaggle.com/competitions/home-credit-default-risk/data

The raw CSVs are **not** included in this repo (they are ~2.5 GB together). To
run the pipeline, download them and put them in `data/raw/`.

## Expected files in `data/raw/`

| file | rows | what it is |
|------|------|------------|
| `application_train.csv` | 307,511 | one row per loan application, with `TARGET` |
| `application_test.csv` | 48,744 | same columns, no `TARGET` |
| `bureau.csv` | 1,716,428 | past credits from other institutions (Credit Bureau) |
| `bureau_balance.csv` | 27,299,925 | monthly balances for the bureau credits |
| `previous_application.csv` | 1,670,214 | the client's earlier Home Credit applications |
| `POS_CASH_balance.csv` | 10,001,358 | monthly POS/cash loan balances |
| `credit_card_balance.csv` | 3,840,312 | monthly credit-card balances |
| `installments_payments.csv` | — | repayment history (see note below) |

Paths are configured in `configs/config.yaml` under `paths.raw_dir` and
`data.files`.

## Note on `installments_payments.csv`

The version of the data I worked with did **not** include
`installments_payments.csv`, so the installments aggregates are not part of the
current feature set. The code handles this gracefully — the loaders check
whether each table exists and simply skip the ones that are missing, so the
pipeline runs end to end with the seven tables that are present. If you add the
file to `data/raw/`, you can extend `feature_engineering.py` with an
installments aggregate in the same style as the others.

## Generated data (not committed)

Running `python main.py features` writes:

- `data/processed/train.parquet`, `data/processed/test.parquet` — the built
  feature matrices (375 columns).
- `data/processed/val_scores.npz` — validation probabilities from the final
  model, saved by the calibrate stage and reused by the threshold/SHAP stages.

These are ignored by git; rebuild them from the raw CSVs.
