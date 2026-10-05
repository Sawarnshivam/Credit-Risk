"""Generate the four analysis notebooks from code, so the JSON is always valid.

Run once: python build_notebooks.py
The notebooks load the artifacts produced by main.py (metrics tables, SHAP,
etc.) so they display the project's real results.
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = "notebooks"


def nb(cells):
    n = new_notebook()
    n["cells"] = cells
    n["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    return n


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


SETUP = """import os, sys, json
import numpy as np
import pandas as pd
# run from repo root so `src` and configs resolve
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
sys.path.insert(0, os.getcwd())
pd.set_option("display.max_columns", 60)"""

# --------------------------------------------------------------------------- #
# 01 EDA
# --------------------------------------------------------------------------- #
eda = nb([
    md("# 01 — Exploratory data analysis\n\n"
       "A first look at the Home Credit application data: target balance, "
       "missingness and the strongest raw signals. This uses the public Kaggle "
       "dataset in `data/raw/`."),
    code(SETUP),
    code("from src.data.load_data import load_config, load_application\n"
         "cfg = load_config()\n"
         "app = load_application(cfg, 'train')\n"
         "print(app.shape)\n"
         "app.head()"),
    md("## Target balance\n\nThe positive class (default) is rare, which drives "
       "most of the modelling choices later."),
    code("rate = app['TARGET'].mean()\n"
         "print(f'default rate: {rate:.4f}  ({app.TARGET.sum():,} of {len(app):,})')\n"
         "app['TARGET'].value_counts().plot.bar(title='TARGET (0 = repaid, 1 = default)')"),
    md("## Missing values\n\nSeveral columns are mostly empty; the tree models "
       "handle NaNs natively, and the linear baseline imputes them."),
    code("miss = app.isna().mean().sort_values(ascending=False)\n"
         "miss.head(15)"),
    md("## External scores vs default\n\nThe three `EXT_SOURCE_*` columns are the "
       "single strongest raw signals in this dataset."),
    code("for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']:\n"
         "    print(c, 'corr with TARGET:', round(app[c].corr(app['TARGET']), 4))"),
    code("import matplotlib.pyplot as plt\n"
         "fig, ax = plt.subplots(1,3, figsize=(12,3))\n"
         "for i,c in enumerate(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']):\n"
         "    app.groupby('TARGET')[c].plot.hist(bins=30, alpha=0.5, ax=ax[i], legend=True, title=c, density=True)\n"
         "plt.tight_layout()"),
    md("## A couple of known data issues\n\n"
       "`DAYS_EMPLOYED` has a sentinel value (365243) for people who are not "
       "employed, and `CODE_GENDER` has a few `XNA` values. Both are handled in "
       "`clean_application`."),
    code("print('DAYS_EMPLOYED == 365243:', int((app['DAYS_EMPLOYED']==365243).sum()))\n"
         "print('CODE_GENDER value counts:')\n"
         "print(app['CODE_GENDER'].value_counts())"),
])

# --------------------------------------------------------------------------- #
# 02 Feature engineering
# --------------------------------------------------------------------------- #
feat = nb([
    md("# 02 — Feature engineering\n\n"
       "How the 375-column feature matrix is built: application-level ratios, "
       "and streaming aggregates over the large history tables."),
    code(SETUP),
    md("## Application ratios\n\nInterpretable affordability ratios and "
       "combinations of the external scores."),
    code("from src.data.preprocessing import clean_application\n"
         "from src.features.feature_engineering import application_features\n"
         "from src.data.load_data import load_config, load_application\n"
         "cfg = load_config()\n"
         "app = load_application(cfg, 'train')\n"
         "eng = application_features(clean_application(app))\n"
         "new_cols = [c for c in eng.columns if c not in app.columns]\n"
         "print('added', len(new_cols), 'application features:')\n"
         "print(new_cols)"),
    code("eng[['CREDIT_INCOME_RATIO','ANNUITY_INCOME_RATIO','CREDIT_ANNUITY_RATIO',\n"
         "     'EXT_SOURCE_MEAN','AGE_YEARS']].describe()"),
    md("## Streaming aggregation over history tables\n\n"
       "The history tables are large (bureau_balance is ~27M rows), so aggregates "
       "are computed in a two-pass streaming way that keeps memory proportional "
       "to the number of groups, not rows. The maths is unit-tested against a "
       "plain groupby in `tests/test_features.py`."),
    code("from src.features.feature_engineering import ChunkedAggregator\n"
         "# tiny demo of the aggregator\n"
         "demo = pd.DataFrame({'SK_ID_CURR':[1,1,2,2,2],'AMT':[10.,20.,3.,4.,5.]})\n"
         "agg = ChunkedAggregator(key='SK_ID_CURR', value_cols=['AMT'])\n"
         "agg.update(demo)\n"
         "agg.result(prefix='DEMO_')"),
    md("## The built matrix\n\nRun `python main.py features` to build the full "
       "matrices; here we just load the metadata it saved."),
    code("meta = json.load(open('reports/tables/feature_meta.json'))\n"
         "meta"),
    code("# if the matrix has been built, peek at it\n"
         "import os\n"
         "p = 'data/processed/train.parquet'\n"
         "if os.path.exists(p):\n"
         "    tr = pd.read_parquet(p)\n"
         "    print('matrix shape:', tr.shape)\n"
         "    print('example aggregate columns:', [c for c in tr.columns if c.startswith(('BURO_','PREV_','POS_','CC_'))][:10])\n"
         "else:\n"
         "    print('run: python main.py features')"),
])

# --------------------------------------------------------------------------- #
# 03 Modeling
# --------------------------------------------------------------------------- #
model = nb([
    md("# 03 — Modeling\n\n"
       "Model comparison, the class-imbalance study, tuning, calibration and the "
       "cost-based threshold. This notebook loads the result tables written by "
       "`main.py`, so the numbers are exactly what the pipeline produced."),
    code(SETUP),
    md("## Model comparison\n\nFour models on the same hold-out set. PR-AUC / KS "
       "/ Brier matter more than accuracy at an 8% default rate."),
    code("cmp = pd.read_csv('reports/tables/model_comparison.csv')\n"
         "cmp[['model','roc_auc','pr_auc','ks','gini','brier']].round(4)"),
    code("import matplotlib.pyplot as plt\n"
         "ax = cmp.set_index('model')[['roc_auc','pr_auc']].plot.bar(rot=20, figsize=(8,4))\n"
         "ax.set_title('Model comparison'); plt.tight_layout()"),
    md("## Class imbalance\n\nNone / class-weight / SMOTE / threshold tuning."),
    code("imb = pd.read_csv('reports/tables/imbalance_comparison.csv')\n"
         "imb.round(4)"),
    md("Class weighting recovers recall at the same AUC; threshold tuning gives "
       "the best F1; SMOTE hurt AUC here."),
    md("## Tuning (Optuna)"),
    code("best = json.load(open('reports/tables/optuna_best.json'))\n"
         "for m, d in best.items():\n"
         "    print(m, 'CV PR-AUC:', round(d['cv_pr_auc'],4), '| trials:', d['n_trials'])\n"
         "best['lightgbm']['best_params']"),
    md("## Calibration\n\nThe final model was calibrated; Brier barely changed "
       "because LightGBM was already well-calibrated."),
    code("cal = json.load(open('reports/tables/calibration.json'))\n"
         "print('Brier by variant:', {k: round(v,5) for k,v in cal['briers'].items()})\n"
         "print('selected:', cal['selected'])\n"
         "print('final hold-out metrics:', {k: round(v,4) for k,v in cal['final_val_metrics'].items() if k in ['roc_auc','pr_auc','ks','gini','brier']})"),
    md("## Cost-based threshold\n\nFalse negative : false positive assumed 5 : 1. "
       "The minimum expected cost picks the operating threshold and risk bands."),
    code("sweep = pd.read_csv('reports/tables/threshold_sweep.csv')\n"
         "sweep.round(4)"),
    code("ax = sweep.set_index('threshold')[['precision','recall','f1','expected_cost_per_app']].plot(marker='o', figsize=(8,4))\n"
         "ax.set_title('Threshold sweep'); plt.tight_layout()\n"
         "print('risk bands:', json.load(open('reports/tables/risk_bands.json')))"),
])

# --------------------------------------------------------------------------- #
# 04 Explainability
# --------------------------------------------------------------------------- #
expl = nb([
    md("# 04 — Model explainability\n\n"
       "SHAP global importance and per-applicant explanations for the final "
       "LightGBM model. SHAP is used for attribution (what drives a score), not "
       "causation."),
    code(SETUP),
    md("## Global importance\n\nMean absolute SHAP value per feature."),
    code("imp = pd.read_csv('reports/tables/shap_importance.csv')\n"
         "imp.head(20)"),
    code("import matplotlib.pyplot as plt\n"
         "top = imp.head(15).iloc[::-1]\n"
         "plt.figure(figsize=(7,5))\n"
         "plt.barh(top['feature'], top['mean_abs_shap'])\n"
         "plt.title('Top 15 features by mean |SHAP|'); plt.tight_layout()"),
    md("The engineered ratios (`CREDIT_ANNUITY_RATIO`, `CREDIT_GOODS_RATIO`, "
       "`EXT_SOURCE_MEAN`) and relational aggregates (`BURO_*`, `POS_*`, `PREV_*`) "
       "rank highly, which shows the feature engineering was worthwhile."),
    md("## Saved SHAP plots\n\n`main.py shap` also saves summary, bar and "
       "dependence plots under `reports/figures/`."),
    code("from IPython.display import Image, display\n"
         "import os\n"
         "for f in ['shap_summary.png','shap_bar.png','shap_dependence_EXT_SOURCE_MEAN.png']:\n"
         "    p = os.path.join('reports/figures', f)\n"
         "    if os.path.exists(p):\n"
         "        display(Image(p))"),
    md("## Per-applicant explanations\n\nTop risk-increasing and risk-reducing "
       "features for a few example applicants, with the decision the bands imply."),
    code("examples = json.load(open('reports/tables/sample_predictions.json'))\n"
         "for e in examples[:3]:\n"
         "    print(f\"applicant {e['applicant_id']}: PD={e['probability_of_default']} -> {e['decision']} ({e['risk_category']})\")\n"
         "    print('   increases risk:', e['top_risk_increasing'][:3])\n"
         "    print('   reduces risk:  ', e['top_risk_reducing'][:3])\n"
         "    print()"),
    md("## Scoring a raw applicant\n\nThe `CreditRiskPredictor` ties it together: "
       "raw record in, probability + decision out."),
    code("from src.models.predict import CreditRiskPredictor\n"
         "from src.data.load_data import load_config, raw_path\n"
         "cfg = load_config()\n"
         "predictor = CreditRiskPredictor()\n"
         "raw = pd.read_csv(raw_path(cfg,'application_test'), nrows=3)\n"
         "for _, row in raw.iterrows():\n"
         "    print(predictor.predict(row.to_dict()))"),
])

for name, n in [("01_eda", eda), ("02_feature_engineering", feat),
                ("03_modeling", model), ("04_model_explainability", expl)]:
    path = os.path.join(OUT, f"{name}.ipynb")
    with open(path, "w") as fh:
        nbf.write(n, fh)
    print("wrote", path)
