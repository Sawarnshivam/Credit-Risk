"""Command-line entry point for the credit-risk pipeline.

Each stage is a separate subcommand so it runs in its own process and memory is
released between stages (this box has ~4 GB RAM). Artifacts are written to disk
and picked up by the next stage.

    python main.py features     # build train/test feature matrices
    python main.py train        # baseline + 4-model comparison
    python main.py imbalance    # class-imbalance experiments
    python main.py tune         # Optuna for XGBoost + LightGBM
    python main.py calibrate    # probability calibration -> final model
    python main.py threshold    # cost-based threshold / risk bands
    python main.py shap         # SHAP global + local explanations
    python main.py predict      # demo predictions on validation rows
    python main.py all          # run the whole thing in order
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import file_exists, load_application, load_config, raw_path
from src.data.preprocessing import (
    TreeEncoder,
    clean_application,
    replace_infinities,
    split_feature_types,
)
from src.evaluation.metrics import full_metrics
from src.features.feature_engineering import (
    aggregate_bureau,
    aggregate_bureau_balance,
    aggregate_credit_card,
    aggregate_pos_cash,
    aggregate_previous_application,
    application_features,
)

CFG = "configs/config.yaml"
EXCLUDE = ["SK_ID_CURR", "TARGET"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _proc(cfg) -> str:
    os.makedirs(cfg["paths"]["processed_dir"], exist_ok=True)
    return cfg["paths"]["processed_dir"]


def _tables(cfg) -> str:
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    return cfg["paths"]["tables_dir"]


def load_matrix(cfg, which: str = "train") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(_proc(cfg), f"{which}.parquet"))


def get_split(cfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    df = load_matrix(cfg, "train")
    y = df["TARGET"].astype("int8")
    feats = [c for c in df.columns if c not in EXCLUDE]
    X = df[feats]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=cfg["validation"]["test_size"],
        stratify=y, random_state=cfg["seed"],
    )
    return X_tr, X_val, y_tr, y_val, feats


def encode_trees(X_tr, X_val, cat_cols):
    enc = TreeEncoder(cat_cols=cat_cols).fit(X_tr)
    return enc.transform(X_tr), enc.transform(X_val), enc


# --------------------------------------------------------------------------- #
# stage: features
# --------------------------------------------------------------------------- #
def stage_features(cfg) -> None:
    t0 = time.time()
    chunk = cfg["feature_engineering"]["chunksize"]
    print("Loading application tables ...")
    train = clean_application(load_application(cfg, "train"))
    test = clean_application(load_application(cfg, "test"))
    train = application_features(train)
    test = application_features(test)
    n_app_feats = train.shape[1] - 2  # minus id + target
    print(f"  application features: {n_app_feats} columns")

    aggregates = []
    print("Aggregating bureau ...")
    aggregates.append(aggregate_bureau(raw_path(cfg, "bureau"), chunk))
    gc.collect()
    print("Aggregating bureau_balance ...")
    aggregates.append(
        aggregate_bureau_balance(raw_path(cfg, "bureau_balance"), raw_path(cfg, "bureau"), chunk)
    )
    gc.collect()
    print("Aggregating previous_application ...")
    aggregates.append(aggregate_previous_application(raw_path(cfg, "previous_application"), chunk))
    gc.collect()
    print("Aggregating POS_CASH ...")
    aggregates.append(aggregate_pos_cash(raw_path(cfg, "pos_cash"), chunk))
    gc.collect()
    print("Aggregating credit_card ...")
    aggregates.append(aggregate_credit_card(raw_path(cfg, "credit_card"), chunk))
    gc.collect()

    def merge_all(app: pd.DataFrame) -> pd.DataFrame:
        for agg in aggregates:
            app = app.merge(agg, on="SK_ID_CURR", how="left")
        return app

    train = merge_all(train)
    test = merge_all(test)
    train = replace_infinities(train)
    test = replace_infinities(test)

    # Downcast floats again after merges.
    for df in (train, test):
        for c in df.select_dtypes(include=["float64"]).columns:
            df[c] = df[c].astype("float32")

    proc = _proc(cfg)
    train.to_parquet(os.path.join(proc, "train.parquet"), index=False)
    test.to_parquet(os.path.join(proc, "test.parquet"), index=False)

    feats = [c for c in train.columns if c not in EXCLUDE]
    json.dump(feats, open(os.path.join(cfg["paths"]["models_dir"], "feature_order.json"), "w"))
    os.makedirs(cfg["paths"]["models_dir"], exist_ok=True)
    json.dump(feats, open(os.path.join(cfg["paths"]["models_dir"], "feature_order.json"), "w"))

    meta = {
        "n_train_rows": int(train.shape[0]),
        "n_test_rows": int(test.shape[0]),
        "n_application_features": int(n_app_feats),
        "n_total_features": int(len(feats)),
        "target_positive_rate": float(train["TARGET"].mean()),
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(meta, open(os.path.join(_tables(cfg), "feature_meta.json"), "w"), indent=2)
    print("Feature build done:", json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# stage: train (baseline + comparison)
# --------------------------------------------------------------------------- #
MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]


def _fit_single_model(cfg, name: str) -> None:
    """Fit exactly one model in its own process and append its metrics row.

    Each model is trained in a fresh interpreter (see ``stage_train``) so that
    peak memory is fully reclaimed between models -- on a 4 GB box the tree
    models each approach the ceiling and would otherwise OOM when sharing one
    process.
    """
    from src.evaluation.evaluation import (
        log_to_mlflow, plot_confusion, plot_pr, plot_roc,
    )
    from src.models.train_baseline import build_logreg
    from src.models.train_lightgbm import build_lightgbm
    from src.models.train_random_forest import build_random_forest
    from src.models.train_xgboost import build_xgboost

    seed = cfg["seed"]
    figdir = cfg["paths"]["figures_dir"]
    mdir = cfg["paths"]["models_dir"]
    os.makedirs(mdir, exist_ok=True)
    sub_rows = cfg["compute_budget"]["logreg_max_rows"]

    X_tr, X_val, y_tr, y_val, feats = get_split(cfg)
    num_cols, cat_cols = split_feature_types(load_matrix(cfg, "train"), EXCLUDE)

    is_tree = name != "logistic_regression"
    params = {"model": name}

    if is_tree:
        Xtr_e, Xval_e, enc = encode_trees(X_tr, X_val, cat_cols)
        joblib.dump(enc, os.path.join(mdir, "tree_encoder.joblib"))
        del X_tr, X_val
        gc.collect()
        Xt, Xv, yt = Xtr_e, Xval_e, y_tr
        # Random Forest (sklearn, dense) is fit on a stratified subsample to
        # cap memory; the gradient-boosted models use all rows.
        if name == "random_forest":
            n = min(sub_rows, len(y_tr))
            if n < len(y_tr):
                keep, _ = train_test_split(np.arange(len(y_tr)), train_size=n,
                                           stratify=y_tr.values, random_state=seed)
                Xt, yt = Xtr_e.iloc[keep], y_tr.iloc[keep]
                print(f"  (fit on stratified subsample of {len(yt)} rows)")
            model = build_random_forest(seed=seed)
            params.update({"n_estimators": 200, "max_depth": 12, "fit_rows": len(yt)})
        elif name == "xgboost":
            model = build_xgboost(seed=seed)
            params.update({"scale_pos_weight": 1.0})
        else:
            model = build_lightgbm(seed=seed)
    else:
        n = min(sub_rows, len(X_tr))
        if n < len(X_tr):
            X_tr, _, y_tr, _ = train_test_split(X_tr, y_tr, train_size=n,
                                                stratify=y_tr, random_state=seed)
            print(f"  (fit on stratified subsample of {n} rows)")
        Xt, Xv, yt = X_tr, X_val, y_tr
        model = build_logreg(num_cols, cat_cols, seed=seed)
        params.update({"C": 0.1, "fit_rows": n})

    t0 = time.time()
    model.fit(Xt, yt)
    sec = time.time() - t0
    proba = model.predict_proba(Xv)[:, 1]
    m = full_metrics(y_val.values, proba, threshold=0.5)
    m["train_seconds"] = round(sec, 1)

    plot_roc(y_val, proba, name, os.path.join(figdir, f"roc_{name}.png"))
    plot_pr(y_val, proba, name, os.path.join(figdir, f"pr_{name}.png"))
    plot_confusion(y_val, (proba >= 0.5).astype(int), name,
                   os.path.join(figdir, f"cm_{name}.png"))
    if is_tree:
        joblib.dump(model, os.path.join(mdir, f"{name}.joblib"))
    log_to_mlflow(name, params, m,
                  artifact_paths=[os.path.join(figdir, f"roc_{name}.png")])

    row = {"model": name, **m}
    with open(os.path.join(_tables(cfg), "_model_rows.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"  {name}: ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
          f"KS={m['ks']:.4f} Brier={m['brier']:.4f} ({sec:.1f}s)")


def stage_train(cfg) -> None:
    """Orchestrate the model comparison, one isolated subprocess per model.

    The parent process deliberately holds almost nothing in memory (it only
    reads the target column for the summary line) so that each child model has
    the full memory budget to itself.
    """
    rows_path = os.path.join(_tables(cfg), "_model_rows.jsonl")
    if os.path.exists(rows_path):
        os.remove(rows_path)

    y = pd.read_parquet(os.path.join(_proc(cfg), "train.parquet"),
                        columns=["TARGET"])["TARGET"]
    n_tr = int(round(len(y) * (1 - cfg["validation"]["test_size"])))
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    meta = json.load(open(os.path.join(_tables(cfg), "feature_meta.json")))
    print(f"train={n_tr} val={len(y) - n_tr} pos_weight={pos_weight:.2f} "
          f"n_feats={meta['n_total_features']}")
    del y
    gc.collect()

    for name in MODEL_ORDER:
        print(f"{name} ...")
        env = dict(os.environ,
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__),
                            "_train_one", "--model", name],
                           cwd=os.path.dirname(os.path.abspath(__file__)), env=env)
        if r.returncode != 0:
            print(f"  !! {name} exited with code {r.returncode} "
                  f"(likely out of memory); skipping")

    if not os.path.exists(rows_path):
        print("No models completed.")
        return
    rows = [json.loads(l) for l in open(rows_path)]
    order = {n: i for i, n in enumerate(MODEL_ORDER)}
    rows.sort(key=lambda d: order.get(d["model"], 99))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(_tables(cfg), "model_comparison.csv"), index=False)
    print(tbl[["model", "roc_auc", "pr_auc", "ks", "gini", "brier"]].to_string(index=False))


# --------------------------------------------------------------------------- #
# stage: imbalance
# --------------------------------------------------------------------------- #
def stage_imbalance(cfg) -> None:
    from imblearn.over_sampling import SMOTE
    from src.models.train_lightgbm import build_lightgbm

    seed = cfg["seed"]
    X_tr, X_val, y_tr, y_val, feats = get_split(cfg)
    _, cat_cols = split_feature_types(load_matrix(cfg, "train"), EXCLUDE)
    Xtr_e, Xval_e, _ = encode_trees(X_tr, X_val, cat_cols)
    del X_tr, X_val
    gc.collect()

    # Subsample training for the SMOTE comparison to keep memory bounded; the
    # same subsample is used for every variant so the comparison is fair.
    n_sub = min(100000, len(Xtr_e))
    keep, _ = train_test_split(np.arange(len(Xtr_e)), train_size=n_sub,
                               stratify=y_tr.values, random_state=seed)
    Xs = Xtr_e.iloc[keep].reset_index(drop=True)
    ys = y_tr.iloc[keep].reset_index(drop=True)
    del Xtr_e  # only the subsample and the validation matrix are needed now
    gc.collect()
    print(f"imbalance subsample: {len(ys)} rows, pos_rate={ys.mean():.4f}", flush=True)
    pos_weight = float((ys == 0).sum() / (ys == 1).sum())

    rows = []

    def evaluate(name, model, Xt, yt):
        model.fit(Xt, yt)
        proba = model.predict_proba(Xval_e)[:, 1]
        m = full_metrics(y_val.values, proba, threshold=0.5)
        rows.append({"strategy": name, **{k: m[k] for k in
                    ["roc_auc", "pr_auc", "precision", "recall", "f1", "ks"]}})
        print(f"  {name}: ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
              f"recall={m['recall']:.3f} F1={m['f1']:.3f}", flush=True)
        gc.collect()

    print("1) no correction", flush=True)
    evaluate("none", build_lightgbm(seed=seed), Xs, ys)
    print("2) class weights (balanced)", flush=True)
    evaluate("class_weight", build_lightgbm(class_weight="balanced", seed=seed), Xs, ys)
    print("3) SMOTE (train only)", flush=True)
    # A single-threaded neighbour search avoids a joblib/OpenMP fork deadlock
    # that can hang parallel KNN on this constrained container.
    from sklearn.neighbors import NearestNeighbors
    knn = NearestNeighbors(n_neighbors=6, n_jobs=1)
    Xr, yr = SMOTE(random_state=seed, k_neighbors=knn).fit_resample(Xs.fillna(Xs.median()), ys)
    evaluate("smote", build_lightgbm(seed=seed), Xr, yr)
    del Xr, yr; gc.collect()

    # 4) threshold optimisation on the "none" model (reuse its probabilities).
    base = build_lightgbm(seed=seed).fit(Xs, ys)
    proba = base.predict_proba(Xval_e)[:, 1]
    from src.evaluation.metrics import threshold_metrics
    best = max([0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
               key=lambda t: threshold_metrics(y_val.values, proba, t)["f1"])
    tm = threshold_metrics(y_val.values, proba, best)
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows.append({"strategy": f"threshold@{best}", "roc_auc": roc_auc_score(y_val, proba),
                 "pr_auc": average_precision_score(y_val, proba),
                 "precision": tm["precision"], "recall": tm["recall"], "f1": tm["f1"],
                 "ks": None})
    print(f"4) threshold tuning: best F1 threshold={best} -> recall={tm['recall']:.3f} F1={tm['f1']:.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(_tables(cfg), "imbalance_comparison.csv"), index=False)


# --------------------------------------------------------------------------- #
# stage: tune (Optuna)
# --------------------------------------------------------------------------- #
def _xgb_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 400, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 6),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    )


def _lgbm_space(trial):
    return dict(
        n_estimators=trial.suggest_int("n_estimators", 300, 500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 16, 64),
        max_depth=trial.suggest_int("max_depth", 3, 8),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    )


def _tune_model(cfg, model_name: str) -> None:
    """Tune a single model with a SQLite-backed Optuna study.

    The study is persisted, so if the process is killed mid-search it resumes
    from the completed trials on the next invocation rather than starting over.
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    seed = cfg["seed"]
    n_rows = cfg["compute_budget"]["optuna_sample_rows"]
    n_trials = cfg["compute_budget"]["optuna_trials"]
    tables = _tables(cfg)

    # Load, take the training split, and subsample BEFORE encoding so peak
    # memory stays low (encoding the full 246k matrix is the memory spike).
    df = load_matrix(cfg, "train")
    y = df["TARGET"].astype("int8")
    feats = [c for c in df.columns if c not in EXCLUDE]
    tr_idx, _ = train_test_split(np.arange(len(df)),
                                 test_size=cfg["validation"]["test_size"],
                                 stratify=y.values, random_state=seed)
    sub = min(n_rows, len(tr_idx))
    keep, _ = train_test_split(tr_idx, train_size=sub,
                               stratify=y.values[tr_idx], random_state=seed)
    _, cat_cols = split_feature_types(df, EXCLUDE)
    Xsub = df[feats].iloc[keep].reset_index(drop=True)
    ys = y.iloc[keep].reset_index(drop=True)
    del df, y
    gc.collect()
    from src.data.preprocessing import TreeEncoder
    Xs = TreeEncoder(cat_cols=cat_cols).fit_transform(Xsub)
    del Xsub
    gc.collect()
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    if model_name == "xgboost":
        from src.models.train_xgboost import build_xgboost
        space, build = _xgb_space, build_xgboost
    else:
        from src.models.train_lightgbm import build_lightgbm
        space, build = _lgbm_space, build_lightgbm

    def obj(trial):
        model = build(space(trial), seed=seed)
        return cross_val_score(model, Xs, ys, cv=cv,
                               scoring="average_precision", n_jobs=1).mean()

    def cb(study, trial):
        print(f"    [{model_name}] trial {trial.number + 1} "
              f"value={trial.value:.4f} best={study.best_value:.4f}", flush=True)
        gc.collect()  # release per-trial model memory before the next trial

    storage = f"sqlite:///{os.path.join(tables, f'optuna_{model_name}.db')}"
    study = optuna.create_study(direction="maximize", study_name=model_name,
                                storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=seed))
    done = len([t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - done)
    print(f"{model_name}: {done} trials done, running {remaining} more "
          f"(target {n_trials}) on {len(ys)} rows", flush=True)
    if remaining:
        study.optimize(obj, n_trials=remaining, show_progress_bar=False,
                       callbacks=[cb])
    print(f"{model_name}: best PR-AUC(cv)={study.best_value:.4f}", flush=True)


def stage_tune(cfg) -> None:
    """Orchestrate per-model tuning in isolated, resumable subprocesses."""
    import optuna
    tables = _tables(cfg)
    n_trials = cfg["compute_budget"]["optuna_trials"]

    for model_name in ["xgboost", "lightgbm"]:
        # Resume until the target trial count is reached; each attempt is a
        # fresh process so a mid-search kill only costs the current trial.
        for attempt in range(1, 8):
            storage = f"sqlite:///{os.path.join(tables, f'optuna_{model_name}.db')}"
            try:
                st = optuna.load_study(study_name=model_name, storage=storage)
                done = len([t for t in st.trials
                            if t.state == optuna.trial.TrialState.COMPLETE])
            except Exception:
                done = 0
            if done >= n_trials:
                break
            print(f"=== tuning {model_name} (attempt {attempt}, {done}/{n_trials} done) ===",
                  flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1")
            subprocess.run([sys.executable, os.path.abspath(__file__),
                            "_tune_one", "--model", model_name],
                           cwd=os.path.dirname(os.path.abspath(__file__)), env=env)

    # Collect best params from both studies.
    out = {}
    for model_name in ["xgboost", "lightgbm"]:
        storage = f"sqlite:///{os.path.join(tables, f'optuna_{model_name}.db')}"
        try:
            st = optuna.load_study(study_name=model_name, storage=storage)
            out[model_name] = {"best_params": st.best_params,
                               "cv_pr_auc": st.best_value,
                               "n_trials": len([t for t in st.trials
                                    if t.state == optuna.trial.TrialState.COMPLETE])}
        except Exception as e:
            print(f"  ({model_name} study unavailable: {e})")
    json.dump(out, open(os.path.join(tables, "optuna_best.json"), "w"), indent=2)
    print("Saved optuna_best.json:", json.dumps(
        {k: {"cv_pr_auc": round(v["cv_pr_auc"], 4), "n_trials": v["n_trials"]}
         for k, v in out.items()}, indent=2))


# --------------------------------------------------------------------------- #
# stage: calibrate  (fits final model)
# --------------------------------------------------------------------------- #
def stage_calibrate(cfg) -> None:
    from src.evaluation.evaluation import plot_calibration
    from src.models.calibration import calibrate
    from src.models.train_lightgbm import build_lightgbm

    seed = cfg["seed"]
    mdir = cfg["paths"]["models_dir"]
    X_tr, X_val, y_tr, y_val, feats = get_split(cfg)
    _, cat_cols = split_feature_types(load_matrix(cfg, "train"), EXCLUDE)
    Xtr_e, Xval_e, enc = encode_trees(X_tr, X_val, cat_cols)
    del X_tr, X_val
    gc.collect()

    # tuned params if available
    best_path = os.path.join(_tables(cfg), "optuna_best.json")
    params = None
    if os.path.exists(best_path):
        params = json.load(open(best_path))["lightgbm"]["best_params"]
        print("Using tuned LightGBM params.", flush=True)

    # split train -> base-fit / calibration
    Xb, Xc, yb, yc = train_test_split(Xtr_e, y_tr, test_size=0.2, stratify=y_tr, random_state=seed)
    del Xtr_e, y_tr
    gc.collect()
    base = build_lightgbm(params, seed=seed).fit(Xb, yb)
    del Xb, yb
    gc.collect()

    proba_raw = base.predict_proba(Xval_e)[:, 1]
    sig = calibrate(base, Xc, yc, method="sigmoid")
    iso = calibrate(base, Xc, yc, method="isotonic")
    proba_sig = sig.predict_proba(Xval_e)[:, 1]
    proba_iso = iso.predict_proba(Xval_e)[:, 1]

    from sklearn.metrics import brier_score_loss
    briers = {
        "uncalibrated": brier_score_loss(y_val, proba_raw),
        "sigmoid": brier_score_loss(y_val, proba_sig),
        "isotonic": brier_score_loss(y_val, proba_iso),
    }
    print("Brier scores:", {k: round(v, 5) for k, v in briers.items()})
    plot_calibration(
        {"uncalibrated": (y_val, proba_raw),
         "sigmoid": (y_val, proba_sig),
         "isotonic": (y_val, proba_iso)},
        os.path.join(cfg["paths"]["figures_dir"], "calibration_curves.png"),
    )

    best_method = min(briers, key=briers.get)
    final = {"uncalibrated": base, "sigmoid": sig, "isotonic": iso}[best_method]
    joblib.dump(final, os.path.join(mdir, "final_model.joblib"))

    # metrics of final model on val
    proba_final = final.predict_proba(Xval_e)[:, 1]
    m = full_metrics(y_val.values, proba_final, threshold=0.5)
    json.dump(
        {"briers": briers, "selected": best_method, "final_val_metrics": m},
        open(os.path.join(_tables(cfg), "calibration.json"), "w"), indent=2,
    )
    # stash val probabilities + labels for threshold/shap stages
    np.savez(os.path.join(_proc(cfg), "val_scores.npz"),
             y=y_val.values, proba=proba_final)
    print(f"Selected '{best_method}' as final. Saved final_model.joblib")


# --------------------------------------------------------------------------- #
# stage: threshold
# --------------------------------------------------------------------------- #
def stage_threshold(cfg) -> None:
    from src.evaluation.threshold import choose_bands, sweep_thresholds

    d = np.load(os.path.join(_proc(cfg), "val_scores.npz"))
    y_val, proba = d["y"], d["proba"]
    cf = cfg["cost_framework"]
    ths = [round(x, 2) for x in np.arange(0.05, 0.55, 0.05)]
    sweep = sweep_thresholds(y_val, proba, ths, cf["cost_false_negative"], cf["cost_false_positive"])
    sweep.to_csv(os.path.join(_tables(cfg), "threshold_sweep.csv"), index=False)
    bands = choose_bands(sweep)
    json.dump(bands, open(os.path.join(_tables(cfg), "risk_bands.json"), "w"), indent=2)
    print(sweep.to_string(index=False))
    print("Risk bands:", bands)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    plt.plot(sweep["threshold"], sweep["precision"], marker="o", label="precision")
    plt.plot(sweep["threshold"], sweep["recall"], marker="o", label="recall")
    plt.plot(sweep["threshold"], sweep["f1"], marker="o", label="F1")
    plt.plot(sweep["threshold"], sweep["expected_cost_per_app"], marker="s", label="expected cost/app")
    plt.axvline(bands["approve_below"], color="green", ls="--", alpha=0.6)
    plt.axvline(bands["reject_above"], color="red", ls="--", alpha=0.6)
    plt.xlabel("threshold")
    plt.title("Threshold analysis with cost overlay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["paths"]["figures_dir"], "threshold_analysis.png"), dpi=110)
    plt.close()


# --------------------------------------------------------------------------- #
# stage: shap
# --------------------------------------------------------------------------- #
def stage_shap(cfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    from src.explainability.shap_analysis import (compute_shap, global_importance,
                                                   local_explanation)

    seed = cfg["seed"]
    mdir = cfg["paths"]["models_dir"]
    figdir = cfg["paths"]["figures_dir"]
    n_shap = cfg["compute_budget"]["shap_sample_rows"]

    X_tr, X_val, y_tr, y_val, feats = get_split(cfg)
    _, cat_cols = split_feature_types(load_matrix(cfg, "train"), EXCLUDE)
    Xtr_e, Xval_e, _ = encode_trees(X_tr, X_val, cat_cols)

    # SHAP on the underlying tree model (unwrap calibrator + FrozenEstimator).
    final = joblib.load(os.path.join(mdir, "final_model.joblib"))
    tree_model = final
    if hasattr(tree_model, "calibrated_classifiers_"):
        tree_model = tree_model.calibrated_classifiers_[0].estimator
    try:
        from sklearn.frozen import FrozenEstimator
        if isinstance(tree_model, FrozenEstimator):
            tree_model = tree_model.estimator
    except ImportError:
        pass
    from src.models.train_lightgbm import build_lightgbm
    if not hasattr(tree_model, "booster_"):
        tree_model = build_lightgbm(seed=seed).fit(Xtr_e, y_tr)

    samp = Xval_e.sample(min(n_shap, len(Xval_e)), random_state=seed)
    explainer, sv = compute_shap(tree_model, samp)

    imp = global_importance(sv, list(samp.columns))
    imp.to_csv(os.path.join(_tables(cfg), "shap_importance.csv"), index=False)
    print("Top SHAP features:\n", imp.head(15).to_string(index=False))

    shap.summary_plot(sv, samp, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "shap_summary.png"), dpi=110); plt.close()
    shap.summary_plot(sv, samp, plot_type="bar", show=False, max_display=20)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "shap_bar.png"), dpi=110); plt.close()

    for feat in imp["feature"].head(3):
        try:
            shap.dependence_plot(feat, sv, samp, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(figdir, f"shap_dependence_{feat}.png"), dpi=110)
            plt.close()
        except Exception as e:
            print("dependence plot skipped for", feat, e)

    # local explanations for 5 applicants + full decision output
    from src.models.predict import CreditRiskPredictor  # noqa
    bands = json.load(open(os.path.join(_tables(cfg), "risk_bands.json")))
    proba = final.predict_proba(samp)[:, 1]
    examples = []
    ids = X_val.loc[samp.index].index  # positional; use SK_ID via train matrix
    df_ids = load_matrix(cfg, "train").loc[samp.index, "SK_ID_CURR"].values
    for j in range(5):
        inc, red = local_explanation(sv, list(samp.columns), j, top_k=5)
        p = float(proba[j])
        cat = "LOW" if p < bands["approve_below"] else ("HIGH" if p >= bands["reject_above"] else "MEDIUM")
        dec = {"LOW": "APPROVE", "MEDIUM": "REVIEW", "HIGH": "REJECT"}[cat]
        examples.append({
            "applicant_id": int(df_ids[j]),
            "probability_of_default": round(p, 4),
            "risk_category": cat,
            "decision": dec,
            "top_risk_increasing": [f"{n} ({v:+.3f})" for n, v in inc],
            "top_risk_reducing": [f"{n} ({v:+.3f})" for n, v in red],
        })
    json.dump(examples, open(os.path.join(_tables(cfg), "sample_predictions.json"), "w"), indent=2)
    print("Saved 5 sample explanations.")


# --------------------------------------------------------------------------- #
# stage: predict (demo)
# --------------------------------------------------------------------------- #
def stage_predict(cfg) -> None:
    from src.models.predict import CreditRiskPredictor
    predictor = CreditRiskPredictor(CFG)
    # Score raw application records (as the pipeline would receive them in use),
    # not the pre-built feature matrix.
    raw = pd.read_csv(raw_path(cfg, "application_test"), nrows=3)
    for _, row in raw.iterrows():
        out = predictor.predict(row.to_dict())
        out["applicant_id"] = int(row["SK_ID_CURR"])
        print(json.dumps(out))


# --------------------------------------------------------------------------- #
def main():
    cfg = load_config(CFG)
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=[
        "features", "train", "imbalance", "tune", "calibrate",
        "threshold", "shap", "predict", "all", "_train_one", "_tune_one",
    ])
    ap.add_argument("--model", default=None,
                    help="model name for the internal _train_one / _tune_one stages")
    args = ap.parse_args()

    if args.stage == "_train_one":
        _fit_single_model(cfg, args.model)
        return
    if args.stage == "_tune_one":
        _tune_model(cfg, args.model)
        return

    stages = {
        "features": stage_features, "train": stage_train, "imbalance": stage_imbalance,
        "tune": stage_tune, "calibrate": stage_calibrate, "threshold": stage_threshold,
        "shap": stage_shap, "predict": stage_predict,
    }
    if args.stage == "all":
        for s in ["features", "train", "imbalance", "tune", "calibrate", "threshold", "shap"]:
            print(f"\n===== {s} =====")
            stages[s](cfg)
    else:
        stages[args.stage](cfg)


if __name__ == "__main__":
    main()
