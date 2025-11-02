#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ======= HARD PIN NATIVE THREADS *BEFORE ANY NUMPY/SCIKIT/XGBOOST IMPORTS* =======
from __future__ import print_function
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ======= PYTHON / STDLIB =======

import json
import time
import warnings
import logging
import sys
from pathlib import Path
from datetime import datetime
import tempfile, shutil, atexit  # <<< for loky temp dir & cleanup

# ======= THIRD-PARTY =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_sample_weight, resample
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, mean_squared_error, r2_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# joblib: include parallel_backend and temp_folder handling
from joblib import Parallel, delayed, parallel_backend

import xgboost as xgb
from xgboost import XGBClassifier

# Install scikit-optimize if not available
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    print("Installing scikit-optimize...")
    import subprocess
    subprocess.check_call(["pip", "install", "scikit-optimize"])
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

try:
    import shap
except ImportError:
    print("Installing shap...")
    import subprocess
    subprocess.check_call(["pip", "install", "shap"])
    import shap

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*sklearn\.utils\.extmath")
warnings.filterwarnings("ignore")

# ======= One shared loky scratch folder + guaranteed cleanup =======
JOBLIB_TMPDIR = tempfile.mkdtemp(prefix="joblib-loky-")
@atexit.register
def _cleanup_joblib_tmpdir():
    try:
        shutil.rmtree(JOBLIB_TMPDIR, ignore_errors=True)
    except Exception:
        pass

# ============================== LOGGING ==============================
def setup_logging(log_dir: str | Path, log_prefix: str = "run"):
    """
    Set up logging to both console and file with timestamp.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{log_prefix}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    logging.info("=" * 80)
    logging.info(f"Logging started - Log file: {log_file}")
    logging.info("=" * 80)

    return log_file


def log_print(*args, **kwargs):
    """Print to console and also log the message."""
    message = ' '.join(str(arg) for arg in args)
    logging.info(message)
    print(*args, **kwargs, flush=True)


def _json_safe(o):
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    elif isinstance(o, (np.float32, np.float64)):
        return float(o)
    elif isinstance(o, (np.ndarray,)):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


# =================== XGBOOST VERSION-AGNOSTIC EARLY STOP HELPERS ===================
def _fit_xgb_with_early_stop(model, X_train, y_train, X_val, y_val, early_rounds=50):
    """
    Version-agnostic XGB fit with early stopping.
    Tries: early_stopping_rounds kwarg -> callbacks -> plain fit.
    """
    try:
        # Most versions support this kwarg
        return model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=early_rounds
        )
    except TypeError:
        # Older/newer variants: use callback API if available
        try:
            from xgboost.callback import EarlyStopping
            return model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=[EarlyStopping(rounds=early_rounds, save_best=True)]
            )
        except Exception:
            # No early stopping available — do a normal fit
            return model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

def _predict_proba_best_iter(model, X):
    """
    Use the best iteration/ntree_limit when available, across XGBoost versions.
    """
    # Newer API: iteration_range
    try:
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            return model.predict_proba(X, iteration_range=(0, model.best_iteration + 1))
    except TypeError:
        pass
    # Older API: ntree_limit
    try:
        if hasattr(model, "best_ntree_limit") and model.best_ntree_limit is not None:
            return model.predict_proba(X, ntree_limit=model.best_ntree_limit)
    except Exception:
        pass
    # Fallback
    return model.predict_proba(X)


# =================== CORE: BAYESIAN OPTIMIZATION (SEQUENTIAL, NO SCALER) ===================
def intra_bayesian_optimization(
    dfs,
    loop,
    *,
    n_calls=20,
    n_splits=5,
    n_workers=1,                 # << forced sequential
    results_dir="bayes_results",
    save_plot=True,
    random_state=42
):
    """
    Sequential CV with memory-safe XGBoost settings. No feature scaling.
    """

    # ------------------------------- data -------------------------------
    data = dfs[dfs['loops'] == loop]
    logging.info(f"Data shape: {data.shape}")
    data_cols = [c for c in data.columns if 'COEFFICIENTS' not in c and 'loop' not in c]
    target_col = 'COEFFICIENTS'
    # Cast once to float32
    X = data[data_cols].to_numpy(dtype=np.float32, copy=False)
    y = data[target_col].to_numpy().ravel()

    logging.info(f"Features: {len(data_cols)}")
    logging.info(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Output dir
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(results_dir) / f"loop{loop}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- search space (n_estimators back to 1000) ---------------------
    dimensions = [
        Integer(50, 1000, name='n_estimators'),
        Integer(3, 12, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0.0, 5.0, name='reg_alpha'),
        Real(0.0, 10.0, name='reg_lambda'),
    ]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_jobs = 1  # << force sequential folds to minimize memory

    import gc

    def _fit_eval_fold(train_idx, test_idx, params):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=1,                     # avoid nested parallelism
            tree_method="hist",           # memory efficient on CPU
            max_bin=256,
            predictor="cpu_predictor",
            **params
        )

        # Version-agnostic early stopping
        _fit_xgb_with_early_stop(model, X_train, y_train, X_test, y_test, early_rounds=50)

        y_scores = _predict_proba_best_iter(model, X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_scores)

        # cleanup
        del model, X_train, X_test, y_scores
        gc.collect()
        return auc_val

    history_rows = []

    @use_named_args(dimensions=dimensions)
    def objective(**params):
        try:
            # Sequential folds; still use loky backend for isolation but n_jobs=1
            with parallel_backend("loky", inner_max_num_threads=1):
                with Parallel(
                    n_jobs=fold_jobs,
                    backend="loky",
                    prefer=None,
                    temp_folder=JOBLIB_TMPDIR
                ) as parallel:
                    fold_aucs = parallel(
                        delayed(_fit_eval_fold)(train_idx, test_idx, params)
                        for train_idx, test_idx in kf.split(X, y)
                    )
            mean_auc = float(np.mean(fold_aucs))
            std_auc = float(np.std(fold_aucs))
            logging.info(f"CV Score: {mean_auc:.3f} ± {std_auc:.3f} | Params: {params}")

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

    # -------------------------- run optimization --------------------------
    logging.info("\n" + "="*60)
    logging.info("Starting Bayesian Hyperparameter Optimization")
    logging.info("="*60)

    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=random_state,
        acq_func='EI'
    )

    # Save optimization history CSV
    hist_df = pd.DataFrame(history_rows)
    hist_path = run_dir / "history.csv"
    hist_df.to_csv(hist_path, index=False)

    # ------------------------------- results -------------------------------
    logging.info("\n" + "="*60)
    logging.info("OPTIMIZATION RESULTS")
    logging.info("="*60)

    best_params = dict(zip([d.name for d in dimensions], result.x))
    best_score = float(-result.fun)
    logging.info(f"Best CV Score: {best_score:.4f}")
    logging.info("Best Parameters:")
    for k, v in best_params.items():
        logging.info(f"  {k}: {v}")

    # ----------------------- final evaluation (sequential) -----------------------
    logging.info("\n" + "="*60)
    logging.info("FINAL EVALUATION WITH BEST PARAMETERS")
    logging.info("="*60)

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = np.zeros_like(thresholds, dtype=float)
    recalls = np.zeros_like(thresholds, dtype=float)
    f1s = np.zeros_like(thresholds, dtype=float)
    balanced_accuracies = np.zeros_like(thresholds, dtype=float)

    def _scores_for_fold(train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            max_bin=256,
            predictor="cpu_predictor",
            **best_params
        )

        _fit_xgb_with_early_stop(model, X_train, y_train, X_test, y_test, early_rounds=50)
        y_scores = _predict_proba_best_iter(model, X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_scores)

        # cleanup
        import gc
        del X_train, X_test
        gc.collect()
        return y_test, y_scores, auc_val

    # Sequential final evaluation
    with parallel_backend("loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=1,  # force sequential
            backend="loky",
            prefer=None,
            temp_folder=JOBLIB_TMPDIR
        ) as parallel:
            fold_results = parallel(
                delayed(_scores_for_fold)(train_idx, test_idx)
                for train_idx, test_idx in kf.split(X, y)
            )

    roc_aucs = []
    for fold, (y_test, y_scores, auc_val) in enumerate(fold_results, start=1):
        roc_aucs.append(auc_val)
        logging.info(f"Fold {fold} ROC AUC: {auc_val:.3f}")

        y_test_bool = y_test.astype(bool)
        for i, t in enumerate(thresholds):
            y_pred = (y_scores >= t).astype(int)
            tp = np.sum((y_pred == 1) & y_test_bool)
            fp = np.sum((y_pred == 1) & ~y_test_bool)
            fn = np.sum((y_pred == 0) & y_test_bool)
            tn = np.sum((y_pred == 0) & ~y_test_bool)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            sens = rec
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            bal_acc = (sens + spec) / 2.0

            precisions[i] += prec
            recalls[i] += rec
            f1s[i] += f1
            balanced_accuracies[i] += bal_acc

    n_folds = kf.get_n_splits()
    precisions /= n_folds
    recalls    /= n_folds
    f1s        /= n_folds
    balanced_accuracies /= n_folds

    mean_auc = float(np.mean(roc_aucs))
    std_auc  = float(np.std(roc_aucs))
    logging.info(f"\nAverage ROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    # Save threshold sweep CSV
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
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.plot(thresholds, f1s, label="F1 Score")
        plt.plot(thresholds, balanced_accuracies, label="Balanced Accuracy")
        plt.xlabel("Threshold")
        plt.ylabel("Metric Score")
        plt.title(f"Cross-Validated Metrics vs. Threshold (Optimized)\nROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = run_dir / "metrics.png"
        plt.savefig(plot_path, dpi=160)
        plt.close()

    logging.info("\n" + "="*60)
    logging.info("SUMMARY")
    logging.info("="*60)
    logging.info(f"✓ Bayesian optimization completed with {len(result.func_vals)} iterations")
    logging.info(f"✓ Best CV score: {best_score:.4f}")
    logging.info(f"Artifacts saved to: {run_dir.resolve()}")

    summary = {
        "loop": int(loop),
        "timestamp": ts,
        "n_calls": int(n_calls),
        "n_splits": int(n_splits),
        "n_workers": int(1),
        "random_state": int(random_state),
        "feature_count": int(len(data_cols)),
        "best_params": best_params,
        "best_cv_score": best_score,
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc,
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "history_csv": str(hist_path.resolve()),
            "thresholds_csv": str(thr_path.resolve()),
            "plot_png": str(plot_path.resolve()) if plot_path else None
        }
    }

    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_safe)

    return {
        'best_params': best_params,
        'best_cv_score': best_score,
        'cv_auc': mean_auc,
        'cv_auc_std': std_auc,
        'optimization_history_df': hist_df,
        'thresholds_df': thr_df,
        'artifacts_dir': str(run_dir.resolve())
    }


# =================== SHAP (no scaler; sample-friendly) ===================
def calculate_global_shap_values(
    dfs, feature_columns, best_params, target_loop=6,
    shap_sample_size=None, random_state=42
):
    """
    Calculate global SHAP values with memory-safe XGB settings.
    No scaling; inputs cast to float32 once.
    """
    data = dfs[dfs['loops'] == target_loop]
    X = data[feature_columns].to_numpy(dtype=np.float32, copy=False)
    y = data['COEFFICIENTS'].to_numpy().ravel()

    model = XGBClassifier(
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=1,                 # SHAP compatibility + avoid nested threads
        tree_method="hist",
        max_bin=256,
        predictor="cpu_predictor",
        **best_params
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    if shap_sample_size is not None and X.shape[0] > shap_sample_size:
        np.random.seed(random_state)
        sample_indices = np.random.choice(X.shape[0], size=shap_sample_size, replace=False)
        X_sample = X[sample_indices]
        shap_values = explainer.shap_values(X_sample)
        X_used = X_sample
    else:
        shap_values = explainer.shap_values(X)
        X_used = X

    return shap_values, explainer, model, X_used


def get_global_feature_importance(shap_values, feature_names):
    global_importance = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance
    }).sort_values('importance', ascending=False)
    return importance_df


def get_global_feature_direction(shap_values, feature_names):
    mean_shap = np.mean(shap_values, axis=0)
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_signed_shap': mean_shap
    }).sort_values('mean_signed_shap', ascending=False)
    return df


# =================== PIPELINE (end-to-end; n_workers=1, no scaler) ===================
def run_bayes_shap_for_loop(
    i: int,
    *,
    input_csv_template="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/new2_merged/{i}loops_merged.csv",
    output_root="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/results/bayes_shap_result",
    n_calls=20,
    n_splits=5,
    n_workers=1,                   # << forced sequential usage
    random_state=42,
    save_plot_threshold_curves=False,
    column_defs: dict | None = None,
    shap_sample_size: int = 500
):
    """
    End-to-end pipeline with memory-safe training/evaluation.
    No StandardScaler; all CV/seeding sequential (n_workers=1).
    """
    # ---------------------------
    # Setup logging
    # ---------------------------
    output_root_path = Path(output_root)
    log_file = setup_logging(
        log_dir=output_root_path,
        log_prefix=f"loop{i}"
    )
    logging.info(f"Starting run_bayes_shap_for_loop for loop {i}")
    logging.info(f"Configuration:")
    logging.info(f"  n_calls: {n_calls}")
    logging.info(f"  n_splits: {n_splits}")
    logging.info(f"  n_workers: {n_workers} (forced to 1 in CV)")
    logging.info(f"  shap_sample_size: {shap_sample_size}")
    logging.info(f"  random_state: {random_state}")
    logging.info(f"  output_root: {output_root}")

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _oof_scores_with_best_params(dfs, feature_cols, loop, best_params, *,
                                     n_splits=5, n_workers=1, random_state=42):
        """Out-of-fold predictions (no leakage), sequential, no scaling."""
        data = dfs[dfs['loops'] == loop]
        X = data[feature_cols].to_numpy(dtype=np.float32, copy=False)
        y = data['COEFFICIENTS'].to_numpy().ravel()

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_jobs = 1  # << force sequential

        def _fit_predict_fold(train_idx, test_idx):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = XGBClassifier(
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=1,                 # avoid nested
                tree_method="hist",
                max_bin=256,
                predictor="cpu_predictor",
                **best_params
            )

            _fit_xgb_with_early_stop(model, X_train, y_train, X_test, y_test, early_rounds=50)
            y_scores = _predict_proba_best_iter(model, X_test)[:, 1]
            return y_test, y_scores

        # Sequential OOF
        with parallel_backend("loky", inner_max_num_threads=1):
            with Parallel(
                n_jobs=fold_jobs,
                backend="loky",
                prefer=None,
                temp_folder=JOBLIB_TMPDIR
            ) as parallel:
                fold_out = parallel(
                    delayed(_fit_predict_fold)(tr, te) for tr, te in kf.split(X, y)
                )

        y_true = np.concatenate([yt for yt, _ in fold_out])
        y_score = np.concatenate([ys for _, ys in fold_out])
        return y_true, y_score

    def _make_auc_and_decile_plots(y_true, y_score, out_dir, prefix):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # ROC (AUC)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1], [0,1], lw=1, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = out_dir / f"{prefix}_roc_curve.png"
        plt.savefig(roc_path, dpi=180)
        plt.close()

        # PR curve (AP)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        plt.figure(figsize=(6,6))
        plt.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend(loc="upper right")
        plt.tight_layout()
        pr_path = out_dir / f"{prefix}_pr_curve.png"
        plt.savefig(pr_path, dpi=180)
        plt.close()

        # Deciles
        dfm = pd.DataFrame({"y": y_true, "score": y_score})
        score_clean = (
            pd.Series(dfm["score"], copy=True)
            .replace([np.inf, -np.inf], pd.NA)
        )
        r_desc = score_clean.rank(pct=True, ascending=False, method="average")
        decile = (21 - np.ceil(r_desc * 20)).astype("Int64")
        dfm["decile_rank"] = decile

        total_pos = pd.Series(dfm["y"]).sum()
        dec = (
            dfm.dropna(subset=["decile_rank"])
            .groupby("decile_rank", as_index=False)
            .agg(
                n=("y", "size"),
                positives=("y", "sum"),
                score_min=("score", "min"),
                score_max=("score", "max"),
                score_mean=("score", "mean"),
            )
            .sort_values("decile_rank")
        )

        dec["negatives"] = dec["n"] - dec["positives"]
        dec["precision"] = dec["positives"] / dec["n"].replace(0, np.nan)
        dec["recall_incremental"] = dec["positives"] / (total_pos if total_pos > 0 else 1)
        dec["recall_cumulative"] = dec["recall_incremental"].cumsum()

        dec_path = out_dir / f"{prefix}_deciles_metrics.csv"
        dec.to_csv(dec_path, index=False)

        # Precision vs decile
        plt.figure(figsize=(8,5))
        plt.plot(dec["decile_rank"], dec["precision"], marker="o")
        plt.xticks(dec["decile_rank"])
        plt.xlabel("Decile rank (1=highest scores)")
        plt.ylabel("Precision")
        plt.title("Precision by Score Decile")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        dec_precision_path = out_dir / f"{prefix}_decile_precision.png"
        plt.savefig(dec_precision_path, dpi=180)
        plt.close()

        # Cumulative recall vs decile
        plt.figure(figsize=(8,5))
        plt.plot(dec["decile_rank"], dec["recall_cumulative"], marker="o")
        plt.xticks(dec["decile_rank"])
        plt.xlabel("Decile rank (1=highest scores)")
        plt.ylabel("Cumulative Recall")
        plt.title("Cumulative Recall by Score Decile")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        dec_recall_path = out_dir / f"{prefix}_decile_recall_cumulative.png"
        plt.savefig(dec_recall_path, dpi=180)
        plt.close()

        return {
            "roc_curve_png": str(roc_path),
            "pr_curve_png": str(pr_path),
            "deciles_csv": str(dec_path),
            "decile_precision_png": str(dec_precision_path),
            "decile_recall_cumulative_png": str(dec_recall_path),
            "auc": float(roc_auc),
            "average_precision": float(ap),
        }

    # ---------------------------
    # Load data
    # ---------------------------
    csv_path = input_csv_template.format(i=i)
    logging.info(f"Loading data from: {csv_path}")
    try:
        dfs = pd.read_csv(csv_path)
        dfs["loops"] = i
        logging.info(f"Data loaded successfully. Shape: {dfs.shape}")
    except Exception as e:
        logging.error(f"Error loading data from {csv_path}: {e}")
        raise

    centrality_cols = [
        'Centrality_betweenness_max',
        'Centrality_betweenness_mean',
        'Centrality_betweenness_skew',
        'Centrality_betweenness_std',
        'Centrality_closeness_max',
        'Centrality_closeness_max_norm',
        'Centrality_closeness_mean',
        'Centrality_closeness_mean_norm',
        'Centrality_closeness_skew',
        'Centrality_closeness_std',
        'Centrality_eigenvector_max',
        'Centrality_eigenvector_mean',
        'Centrality_eigenvector_skew',
        'Centrality_eigenvector_std'
    ]

    repeat_motifs = ['Motif_induced_C4',
    'Motif_induced_C4_per_Cn4',
    'Motif_induced_Diamond',
    'Motif_induced_Diamond_per_Cn4',
    'Motif_induced_K1_3',
    'Motif_induced_K1_3_per_Cn4',
    'Motif_induced_K4',
    'Motif_induced_K4_per_Cn4',
    'Motif_induced_P4',
    'Motif_induced_P4_per_Cn4',
    'Motif_induced_TailedTriangle',
    'Motif_induced_TailedTriangle_per_Cn4']

    if column_defs is None:
        all_cols = [x for x in dfs.columns if (x not in ["loops", "COEFFICIENTS"] and x not in repeat_motifs)]
        cols0 = [x for x in all_cols if ('motif' in x.lower()) or ('eig_' in x.lower() and 'centrality' not in x.lower())]
        cols00 = cols0 + centrality_cols
        cols1 = [x for x in all_cols if 'motif' in x.lower()]
        cols5 = [x for x in all_cols if ('eig_' in x.lower() and 'centrality' not in x.lower())]
        cols2 = [x for x in all_cols if ('eig_' in x.lower() and 'centrality' not in x.lower()) or ('spectral' in x.lower() and 'centrality' not in x.lower())]
        column_defs = {
            #"all_columns": all_cols,
            "motifs_eig": cols0,
            "motifs": cols1,
            #"spectral": cols2,
            "eig": cols5,
            "motifs_eig_centrality": cols00
        }

    output_root = Path(output_root)
    (output_root / f"loop{i}").mkdir(parents=True, exist_ok=True)
    ts = ""  # overwrite

    index_rows = []
    auc_summary_rows = []

    # ---------------------------
    # Per column-set pipeline
    # ---------------------------
    for colset_name, feature_cols in column_defs.items():
        if not feature_cols:
            logging.info(f"[loop {i} | {colset_name}] Skipping (no features).")
            continue

        logging.info(f"\n=== Loop {i} | Colset: {colset_name} | {len(feature_cols)} features ===")

        # Uniqueness stats
        feature_df = dfs[feature_cols]
        def _uniqueness_stats(df_subset: pd.DataFrame) -> dict:
            n = len(df_subset)
            if n == 0:
                return {"n_rows": 0, "n_unique_rows": 0, "uniqueness_ratio": 1.0, "duplicate_rows": 0}
            n_unique = len(df_subset.drop_duplicates())
            dup = n - n_unique
            return {
                "n_rows": int(n),
                "n_unique_rows": int(n_unique),
                "uniqueness_ratio": float(n_unique / n),
                "duplicate_rows": int(dup),
            }

        uniq_stats = _uniqueness_stats(feature_df)
        logging.info(
            f"[uniqueness] rows={uniq_stats['n_rows']}, "
            f"unique={uniq_stats['n_unique_rows']}, "
            f"UR={uniq_stats['uniqueness_ratio']:.6f}, "
            f"dupes={uniq_stats['duplicate_rows']}"
        )

        # Optional dup groups (can be heavy on very large data; keep as-is)
        dup_groups_csv = None
        if uniq_stats["duplicate_rows"] > 0:
            dup_groups = (
                feature_df
                .groupby(list(feature_df.columns), dropna=False)
                .size()
                .reset_index(name="count")
                .query("count > 1")
                .sort_values("count", ascending=False)
            )
            out_dir_for_dupes = output_root / f"loop{i}" / f"{colset_name}_{ts}"
            out_dir_for_dupes.mkdir(parents=True, exist_ok=True)
            dup_groups_csv_path = out_dir_for_dupes / f"loop{i}_{colset_name}_duplicate_groups.csv"
            dup_groups.to_csv(dup_groups_csv_path, index=False)
            dup_groups_csv = str(dup_groups_csv_path)

        df_for_bayes = dfs[feature_cols + ["loops", "COEFFICIENTS"]]

        # Bayesian optimization (sequential)
        logging.info(f"Starting Bayesian optimization for {colset_name}...")
        results = intra_bayesian_optimization(
            df_for_bayes,
            i,
            n_calls=n_calls,
            n_splits=n_splits,
            n_workers=1,  # << enforce sequential
            results_dir="bayes_results",
            save_plot=save_plot_threshold_curves,
            random_state=random_state
        )
        logging.info(f"Bayesian optimization completed for {colset_name}")

        # Output folder
        out_dir = output_root / f"loop{i}" / f"{colset_name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save best params + sweep + thresholds
        best_params = results["best_params"]
        with open(out_dir / f"loop{i}_{colset_name}_best_params.json", "w") as f:
            json.dump(best_params, f, indent=2, default=_json_safe)

        results["optimization_history_df"].to_csv(out_dir / f"loop{i}_{colset_name}_bayes_history.csv", index=False)
        results["thresholds_df"].to_csv(out_dir / f"loop{i}_{colset_name}_thresholds.csv", index=False)

        # SHAP (no scaler)
        logging.info(f"Computing SHAP values for {colset_name}...")
        shap_values, explainer, model, X_used = calculate_global_shap_values(
            dfs, feature_columns=feature_cols, best_params=best_params, target_loop=i,
            shap_sample_size=shap_sample_size, random_state=random_state
        )
        logging.info(f"SHAP computation completed. Shape: {np.array(shap_values).shape}")

        importance_df = get_global_feature_importance(shap_values, feature_cols)
        direction_df  = get_global_feature_direction(shap_values, feature_cols)
        shap_full = importance_df.merge(direction_df, on="feature")
        shap_full_path = out_dir / f"loop{i}_{colset_name}_shap_full.csv"
        shap_full.to_csv(shap_full_path, index=False)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_used, feature_names=feature_cols, max_display=20, show=False)
        plt.title(f"SHAP Summary (loop {i} | {colset_name})")
        shap_png_path = out_dir / f"loop{i}_{colset_name}_shap_summary.png"
        plt.savefig(shap_png_path, bbox_inches='tight', dpi=200)
        plt.close()

        # SHAP sample csv
        if shap_sample_size is not None and len(shap_values) <= shap_sample_size:
            shap_sample = pd.DataFrame(shap_values, columns=feature_cols)
            shap_sample["sample_index"] = np.arange(len(shap_sample))
        else:
            n_samples = min(shap_sample_size, len(shap_values))
            shap_sample = resample(
                pd.DataFrame(shap_values, columns=feature_cols),
                n_samples=n_samples,
                random_state=random_state
            )
            shap_sample["sample_index"] = np.arange(len(shap_sample))

        shap_sample_path = out_dir / f"loop{i}_{colset_name}_shap_sample.csv"
        shap_sample.to_csv(shap_sample_path, index=False)

        # OOF + curves/deciles (sequential)
        y_true_oof, y_score_oof = _oof_scores_with_best_params(
            dfs, feature_cols, i, best_params,
            n_splits=n_splits, n_workers=1, random_state=random_state
        )
        curve_artifacts = _make_auc_and_decile_plots(
            y_true_oof, y_score_oof,
            out_dir=out_dir,
            prefix=f"loop{i}_{colset_name}"
        )

        # AUC summary
        auc_summary_rows.append({
            "loop": i,
            "colset": colset_name,
            "n_features": len(feature_cols),
            "auc": curve_artifacts["auc"],
            "average_precision": curve_artifacts["average_precision"],
            "uniqueness_ratio": uniq_stats["uniqueness_ratio"],
            "n_rows": uniq_stats["n_rows"],
            "n_unique_rows": uniq_stats["n_unique_rows"],
            "duplicate_rows": uniq_stats["duplicate_rows"],
        })

        # Index artifacts
        index_rows.append({
            "loop": i,
            "colset": colset_name,
            "timestamp": ts,
            "best_params_json": str(out_dir / f"loop{i}_{colset_name}_best_params.json"),
            "bayes_history_csv": str(out_dir / f"loop{i}_{colset_name}_bayes_history.csv"),
            "thresholds_csv": str(out_dir / f"loop{i}_{colset_name}_thresholds.csv"),
            "shap_full_csv": str(shap_full_path),
            "shap_summary_png": str(shap_png_path),
            "shap_sample_csv": str(shap_sample_path),
            "roc_curve_png": curve_artifacts["roc_curve_png"],
            "pr_curve_png": curve_artifacts["pr_curve_png"],
            "deciles_csv": curve_artifacts["deciles_csv"],
            "decile_precision_png": curve_artifacts["decile_precision_png"],
            "decile_recall_cumulative_png": curve_artifacts["decile_recall_cumulative_png"],
            "auc": curve_artifacts["auc"],
            "average_precision": curve_artifacts["average_precision"],
            "uniqueness_ratio": uniq_stats["uniqueness_ratio"],
            "n_rows": uniq_stats["n_rows"],
            "n_unique_rows": uniq_stats["n_unique_rows"],
            "duplicate_rows": uniq_stats["duplicate_rows"],
            "duplicate_groups_csv": dup_groups_csv,
            "optimizer_artifacts_dir": results.get("artifacts_dir")
        })

    # Artifact index for the loop
    idx_path = None
    if index_rows:
        idx_df = pd.DataFrame(index_rows)
        idx_path = output_root / f"loop{i}" / f"loop{i}_artifact_index_{ts}.csv"
        idx_df.to_csv(idx_path, index=False)
        logging.info(f"\n✅ Artifact index saved to: {idx_path}")
    else:
        logging.warning("\n⚠️ No column sets were processed (empty definitions?).")

    # AUC summary for the loop
    auc_path = None
    auc_df = pd.DataFrame(auc_summary_rows)
    if not auc_df.empty:
        auc_df = auc_df.sort_values(["auc", "uniqueness_ratio"], ascending=[False, False])
        auc_path = output_root / f"loop{i}" / f"loop{i}_auc_summary_{ts}.csv"
        auc_df.to_csv(auc_path, index=False)
        logging.info("\nAUC summary (descending):")
        try:
            logging.info(auc_df.to_string(index=False))
        except Exception:
            logging.info(str(auc_df.head()))
        logging.info(f"\n📄 AUC summary saved to: {auc_path}")

    logging.info("\n" + "="*80)
    logging.info(f"Pipeline completed for loop {i}")
    logging.info(f"All outputs logged to: {log_file}")
    logging.info("="*80)

    return {
        "artifact_index_csv": str(idx_path) if idx_path else None,
        "auc_summary_csv": str(auc_path) if auc_path else None,
        "auc_summary_df": auc_df
    }
