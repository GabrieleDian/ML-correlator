#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================== HARD-PIN THREADS BEFORE ANY NUMPY/SCI-KIT IMPORTS ==================
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ================== STD LIB ==================
import json
import time
import re
import logging
import sys
import warnings
import tempfile, shutil, atexit
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ================== THIRD-PARTY ==================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from joblib import Parallel, delayed, parallel_backend

import xgboost as xgb
from xgboost import XGBClassifier

# scikit-optimize (install if missing)
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("Installing scikit-optimize...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

# Optional parquet engine notice
try:
    import pyarrow  # noqa: F401
except Exception:
    print("Note: pyarrow not found. Pandas may fallback to fastparquet if installed.")

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*sklearn\.utils\.extmath")
warnings.filterwarnings("ignore")

# ================== ONE SHARED LOKY SCRATCH + CLEANUP ==================
JOBLIB_TMPDIR = tempfile.mkdtemp(prefix="joblib-loky-")
@atexit.register
def _cleanup_joblib_tmpdir():
    try:
        shutil.rmtree(JOBLIB_TMPDIR, ignore_errors=True)
    except Exception:
        pass

# =======================================================================================
#                                  LOGGING (INLINE)
# =======================================================================================
def setup_logging(log_dir: str | Path, log_prefix: str = "run", level=logging.INFO) -> Path:
    """
    Dual logging: console + file. Returns log file path.
    If `rich` is installed, uses pretty console logs; otherwise plain text.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{log_prefix}_{ts}.log"

    # Reset root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []

    # File logs
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(fh)

    # Console logs (pretty if rich is available)
    try:
        from rich.logging import RichHandler  # type: ignore
        ch = RichHandler(rich_tracebacks=True, show_time=False, markup=True)
        ch.setLevel(level)
        root.addHandler(ch)
    except Exception:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(ch)

    logging.info("=" * 90)
    logging.info(f"Logging started - {log_file}")
    logging.info("=" * 90)
    return log_file

def _json_safe(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")

# =======================================================================================
#                       XGBOOST VERSION-AGNOSTIC HELPERS
# =======================================================================================
def _fit_xgb_with_early_stop(model, X_train, y_train, X_val, y_val, early_rounds=50):
    """Try classic kwarg, then callback API, else plain fit."""
    try:
        return model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=early_rounds
        )
    except TypeError:
        try:
            from xgboost.callback import EarlyStopping
            return model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=[EarlyStopping(rounds=early_rounds, save_best=True)]
            )
        except Exception:
            return model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

def _predict_proba_best_iter(model, X):
    """Use best_iteration / ntree_limit if available."""
    try:
        if getattr(model, "best_iteration", None) is not None:
            return model.predict_proba(X, iteration_range=(0, model.best_iteration + 1))
    except TypeError:
        pass
    try:
        if getattr(model, "best_ntree_limit", None) is not None:
            return model.predict_proba(X, ntree_limit=model.best_ntree_limit)
    except Exception:
        pass
    return model.predict_proba(X)

# =======================================================================================
#                                  DATA LOADING
# =======================================================================================
def _infer_loop_from_name(p: Path) -> int | None:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else None

def load_and_align_parquet_dataset(
    dataset_dir: str | Path,
    loops: list[int] = list(range(5, 12)),
    target_col: str = "COEFFICIENTS",
    loop_col: str = "loops"
) -> pd.DataFrame:
    """
    Load *.parquet for specified loops, enforce a common feature space (intersection),
    and return DataFrame: [features, loop_col, target_col].
    """
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {dataset_dir}")

    loaded, feature_sets = [], []
    for p in files:
        df = pd.read_parquet(p)

        # ensure loop column
        if loop_col not in df.columns:
            lg = _infer_loop_from_name(p)
            if lg is None:
                raise ValueError(f"Cannot infer loop id from filename: {p.name}")
            df[loop_col] = lg

        # filter loops
        df = df[df[loop_col].isin(loops)]
        if df.empty:
            continue

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' missing in {p.name}")

        feats = [c for c in df.columns if c not in {target_col, loop_col}]
        feature_sets.append(set(feats))
        loaded.append(df)

    if not loaded:
        raise ValueError(f"No rows found for loops {loops} in {dataset_dir}")

    common_feats = sorted(list(set.intersection(*feature_sets)))
    if not common_feats:
        raise ValueError("No common feature columns across parquet files.")

    merged = pd.concat([d[common_feats + [loop_col, target_col]].copy() for d in loaded], ignore_index=True)
    return merged

# =======================================================================================
#                              METRICS / THRESHOLD
# =======================================================================================
def _best_threshold_from_scores(y_true, y_score, metric="f1"):
    """Scan thresholds 0.01..0.99 and return (best_t, best_val) for f1 or balanced_accuracy."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_val = 0.5, -1.0
    y_true_bool = y_true.astype(bool)
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = np.sum((y_pred == 1) & y_true_bool)
        fp = np.sum((y_pred == 1) & ~y_true_bool)
        fn = np.sum((y_pred == 0) & y_true_bool)
        tn = np.sum((y_pred == 0) & ~y_true_bool)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        sens = rec
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal  = 0.5 * (sens + spec)
        val  = f1 if metric == "f1" else bal
        if val > best_val:
            best_val, best_t = val, float(t)
    return best_t, best_val

# =======================================================================================
#                        BAYES OPT + FINAL RETRAIN (ALL-IN-ONE)
# =======================================================================================
@dataclass
class BayesConfig:
    n_calls: int = 30
    n_splits: int = 5
    n_workers: int = 5
    random_state: int = 42
    results_dir: str | Path = "bayes_results_merged"
    save_plot: bool = True
    threshold_metric: str = "f1"  # or "balanced_accuracy"
    holdout_size: float = 0.1      # for early stopping to discover best_iteration
    device: str = "cpu"            # "cpu" or "cuda" (XGB >=2.0 sklearn API supports this)

def bayes_opt_on_merged_stratified(
    df: pd.DataFrame,
    *,
    target_col: str = "COEFFICIENTS",
    loop_col: str = "loops",
    cfg: BayesConfig = BayesConfig()
):
    """
    - Merged CV stratified by (loop, target).
    - Bayesian optimization for XGB hyperparams.
    - OOF threshold selection.
    - Final retrain on 100% with n_estimators = best_iteration + 1.
    - Save artifacts (model, scaler, features, threshold, curves, summaries).
    """
    # Features / arrays
    data_cols = [c for c in df.columns if c not in {target_col, loop_col}]
    X_all = df[data_cols].values
    y_all = df[target_col].values.ravel()
    strat_all = df[loop_col].astype(str) + "_" + pd.Series(y_all).astype(str)

    # Run dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.results_dir) / f"merged_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Search space
    dimensions = [
        Integer(200, 1200, name='n_estimators'),
        Integer(3, 12, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0.0, 5.0, name='reg_alpha'),
        Real(0.0, 10.0, name='reg_lambda'),
    ]

    kf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    fold_jobs = max(1, min(int(cfg.n_workers), cfg.n_splits))

    import gc
    history_rows = []

    def _fit_eval_fold(train_idx, test_idx, params):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=cfg.random_state,
            n_jobs=1,
            tree_method="hist",
            max_bin=256,
            device=cfg.device,   # << no predictor kwarg
            **params
        )
        _fit_xgb_with_early_stop(model, X_train, y_train, X_test, y_test, early_rounds=50)
        y_scores = _predict_proba_best_iter(model, X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_scores)

        del model, X_train, X_test, y_scores, scaler
        gc.collect()
        return auc_val

    @use_named_args(dimensions=dimensions)
    def objective(**params):
        try:
            with parallel_backend("loky", inner_max_num_threads=1):
                with Parallel(
                    n_jobs=fold_jobs,
                    backend="loky",
                    prefer=None,
                    temp_folder=JOBLIB_TMPDIR
                ) as parallel:
                    fold_aucs = parallel(
                        delayed(_fit_eval_fold)(tr, te, params)
                        for tr, te in kf.split(X_all, strat_all)
                    )
            mean_auc = float(np.mean(fold_aucs))
            std_auc = float(np.std(fold_aucs))
            logging.info(f"[MERGED CV] AUC: {mean_auc:.3f} ± {std_auc:.3f} | {params}")

            history_rows.append({
                **params,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "timestamp": time.time()
            })
            return -mean_auc
        except Exception as e:
            logging.error(f"Error with params {params}: {e}")
            history_rows.append({**params, "mean_auc": np.nan, "std_auc": np.nan,
                                 "error": str(e), "timestamp": time.time()})
            return 1.0

    logging.info("\n" + "="*70)
    logging.info("Starting MERGED Bayesian Hyperparameter Optimization (loop+label-stratified)")
    logging.info("=" * 70)

    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=cfg.n_calls,
        random_state=cfg.random_state,
        acq_func='EI'
    )

    # Optimization history
    hist_df = pd.DataFrame(history_rows)
    hist_path = run_dir / "history.csv"
    hist_df.to_csv(hist_path, index=False)

    # Best params
    best_params = dict(zip([d.name for d in dimensions], result.x))
    best_score = float(-result.fun)
    logging.info(f"Best CV AUC: {best_score:.4f}")
    for k, v in best_params.items():
        logging.info(f"  {k}: {v}")

    # ================= OOF CAPTURE (for threshold & curves) =================
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = np.zeros_like(thresholds, dtype=float)
    recalls = np.zeros_like(thresholds, dtype=float)
    f1s = np.zeros_like(thresholds, dtype=float)
    balanced_accuracies = np.zeros_like(thresholds, dtype=float)

    def _scores_for_fold(train_idx, test_idx):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=cfg.random_state,
            n_jobs=1,
            tree_method="hist",
            max_bin=256,
            device=cfg.device,
            **best_params
        )
        _fit_xgb_with_early_stop(model, X_train, y_train, X_test, y_test, early_rounds=50)
        y_scores = _predict_proba_best_iter(model, X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_scores)
        return y_test, y_scores, auc_val

    with parallel_backend("loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=fold_jobs,
            backend="loky",
            prefer=None,
            temp_folder=JOBLIB_TMPDIR
        ) as parallel:
            fold_results = parallel(
                delayed(_scores_for_fold)(tr, te)
                for tr, te in kf.split(X_all, strat_all)
            )

    roc_aucs, y_true_oof, y_score_oof = [], [], []
    for y_test, y_scores, auc_val in fold_results:
        roc_aucs.append(auc_val)
        y_true_oof.append(y_test)
        y_score_oof.append(y_scores)

        yb = y_test.astype(bool)
        for i, t in enumerate(thresholds):
            y_pred = (y_scores >= t).astype(int)
            tp = np.sum((y_pred == 1) & yb)
            fp = np.sum((y_pred == 1) & ~yb)
            fn = np.sum((y_pred == 0) & yb)
            tn = np.sum((y_pred == 0) & ~yb)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            sens = rec
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            bal  = 0.5 * (sens + spec)
            precisions[i] += prec
            recalls[i] += rec
            f1s[i] += f1
            balanced_accuracies[i] += bal

    y_true_oof = np.concatenate(y_true_oof)
    y_score_oof = np.concatenate(y_score_oof)

    n_folds = kf.get_n_splits()
    precisions /= n_folds; recalls /= n_folds; f1s /= n_folds; balanced_accuracies /= n_folds

    mean_auc = float(np.mean(roc_aucs))
    std_auc  = float(np.std(roc_aucs))
    logging.info(f"CV ROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    thr_df = pd.DataFrame({
        "threshold": thresholds,
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "balanced_accuracy": balanced_accuracies
    })
    thr_path = run_dir / "thresholds.csv"
    thr_df.to_csv(thr_path, index=False)

    plot_path = None
    if cfg.save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.plot(thresholds, f1s, label="F1 Score")
        plt.plot(thresholds, balanced_accuracies, label="Balanced Accuracy")
        plt.xlabel("Threshold"); plt.ylabel("Metric Score")
        plt.title(f"Merged CV Metrics vs. Threshold\nROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plot_path = run_dir / "metrics.png"
        plt.savefig(plot_path, dpi=160); plt.close()

    # Best threshold from OOF
    best_thr, best_thr_val = _best_threshold_from_scores(
        y_true_oof, y_score_oof, metric=cfg.threshold_metric
    )
    logging.info(f"Best threshold ({cfg.threshold_metric}): t={best_thr:.3f} (score={best_thr_val:.4f})")

    # ================= FINAL RETRAIN (100% DATA) =================
    # Discover best_iteration with small stratified holdout
    X_tr, X_val, y_tr, y_val, strat_tr, strat_val = train_test_split(
        X_all, y_all, strat_all, test_size=cfg.holdout_size,
        random_state=cfg.random_state, stratify=strat_all
    )
    scaler_tmp = StandardScaler()
    X_tr = scaler_tmp.fit_transform(X_tr).astype(np.float32)
    X_val = scaler_tmp.transform(X_val).astype(np.float32)

    model_tmp = XGBClassifier(
        eval_metric='logloss',
        random_state=cfg.random_state,
        n_jobs=1,
        tree_method="hist",
        max_bin=256,
        device=cfg.device,
        **best_params
    )
    _fit_xgb_with_early_stop(model_tmp, X_tr, y_tr, X_val, y_val, early_rounds=50)
    best_iter = getattr(model_tmp, "best_iteration", None)
    if best_iter is None:
        best_iter = best_params["n_estimators"] - 1
    del model_tmp, scaler_tmp

    # Fit scaler on 100% and retrain with best_iter + 1
    scaler_full = StandardScaler()
    X_all_scaled = scaler_full.fit_transform(X_all).astype(np.float32)

    final_params = dict(best_params)
    final_params["n_estimators"] = int(best_iter + 1)

    final_model = XGBClassifier(
        eval_metric='logloss',
        random_state=cfg.random_state,
        n_jobs=1,
        tree_method="hist",
        max_bin=256,
        device=cfg.device,
        **final_params
    )
    final_model.fit(X_all_scaled, y_all)

    # Persist artifacts
    try:
        import joblib
        joblib.dump(final_model, run_dir / "final_model_sklearn.pkl")
        final_model.get_booster().save_model(str(run_dir / "final_model_booster.json"))
        joblib.dump(scaler_full, run_dir / "scaler.pkl")
    except Exception as e:
        logging.warning(f"joblib/xgboost save failed: {e}; falling back to pickle.")
        import pickle
        with open(run_dir / "final_model_sklearn.pkl", "wb") as f:
            pickle.dump(final_model, f)
        with open(run_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler_full, f)
        try:
            final_model.get_booster().save_model(str(run_dir / "final_model_booster.json"))
        except Exception:
            pass

    with open(run_dir / "feature_names.json", "w") as f:
        json.dump(data_cols, f, indent=2)
    with open(run_dir / "decision_threshold.txt", "w") as f:
        f.write(f"{best_thr:.6f}\n")
    with open(run_dir / "inference_readme.txt", "w") as f:
        f.write(
            "Inference recipe:\n"
            "1) Load scaler.pkl and final_model_sklearn.pkl\n"
            "2) Ensure input columns match feature_names.json order\n"
            "3) X_scaled = scaler.transform(X)\n"
            "4) p = final_model.predict_proba(X_scaled)[:,1]\n"
            f"5) y_hat = (p >= {best_thr:.6f}).astype(int)\n"
        )

    summary = {
        "mode": "merged_parquet",
        "timestamp": ts,
        "n_calls": int(cfg.n_calls),
        "n_splits": int(cfg.n_splits),
        "n_workers": int(cfg.n_workers),
        "random_state": int(cfg.random_state),
        "feature_count": int(len(data_cols)),
        "best_params_cv": best_params,
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc,
        "selected_threshold_metric": cfg.threshold_metric,
        "selected_threshold": float(best_thr),
        "selected_threshold_score": float(best_thr_val),
        "final_model_n_estimators": int(final_params["n_estimators"]),
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "history_csv": str(hist_path.resolve()),
            "thresholds_csv": str(thr_path.resolve()),
            "plot_png": str(plot_path.resolve()) if plot_path else None,
            "final_model_sklearn_pkl": str((run_dir / "final_model_sklearn.pkl").resolve()),
            "final_model_booster_json": str((run_dir / "final_model_booster.json").resolve()),
            "scaler_pkl": str((run_dir / "scaler.pkl").resolve()),
            "feature_names_json": str((run_dir / "feature_names.json").resolve()),
            "decision_threshold_txt": str((run_dir / "decision_threshold.txt").resolve()),
        }
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)

    return {
        'best_params': best_params,
        'best_cv_auc': best_score,
        'cv_auc_mean': mean_auc,
        'cv_auc_std': std_auc,
        'artifacts_dir': str(run_dir.resolve()),
        'feature_names': data_cols,
        'decision_threshold': float(best_thr),
        'final_model_paths': summary["paths"]
    }

# =======================================================================================
#                                     RUNNER
# =======================================================================================
def run_bayes_on_merged_parquet(
    dataset_dir="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset",
    loops=range(5, 12),
    log_root="bayes_results_merged",
    cfg: BayesConfig = BayesConfig()
):
    setup_logging(log_root, log_prefix="merged_bayes")
    logging.info(f"Loading parquet dataset from: {dataset_dir}")
    merged = load_and_align_parquet_dataset(dataset_dir, loops=list(loops))
    logging.info(f"Merged shape: {merged.shape}")
    logging.info(f"Class distribution: {dict(zip(*np.unique(merged['COEFFICIENTS'], return_counts=True)))}")
    logging.info(f"Loop distribution: {dict(zip(*np.unique(merged['loops'], return_counts=True)))}")

    out = bayes_opt_on_merged_stratified(
        merged,
        target_col="COEFFICIENTS",
        loop_col="loops",
        cfg=cfg
    )
    logging.info("\n" + "="*90)
    logging.info("Merged / loop+label-stratified optimization complete.")
    logging.info(f"Best CV AUC: {out['best_cv_auc']:.4f}")
    logging.info(f"Artifacts dir: {out['artifacts_dir']}")
    logging.info("="*90)
    return out

# =======================================================================================
#                                     ENTRY POINT
# =======================================================================================
if __name__ == "__main__":
    config = BayesConfig(
        n_calls=30,
        n_splits=5,
        n_workers=5,
        random_state=42,
        results_dir="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/bayes_results/bayes_results_merged",
        save_plot=True,
        threshold_metric="f1",  # or "balanced_accuracy"
        holdout_size=0.1,
        device="cpu"            # set "cuda" for GPU
    )

    # Initialize logging first so everything is captured
    _lf = setup_logging(config.results_dir, log_prefix="merged_bayes")
    print(f"Log file: {_lf}")

    out = run_bayes_on_merged_parquet(
        dataset_dir="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset",
        loops=range(5, 12),
        log_root=config.results_dir,
        cfg=config
    )

    print("\n=== Final artifacts ===")
    print(json.dumps(out["final_model_paths"], indent=2))
    print(f"Decision threshold: {out['decision_threshold']:.6f}")
