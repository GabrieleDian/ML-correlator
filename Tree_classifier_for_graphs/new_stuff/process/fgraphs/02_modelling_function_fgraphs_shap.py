#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All-in-one script for SHAP analysis of fgraph features.
Computes SHAP values separately for: all columns, NUM, DEN, and TOTAL.
Saves results in fgraphs_loop_# folders.
"""

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
import re
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import atexit
import gc

# ======= THIRD-PARTY =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import resample
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
)

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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize"])
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args

try:
    import shap
except ImportError:
    print("Installing shap...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
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

from fractions import Fraction

def to_numeric(x):
    try:
        return float(Fraction(str(x)))
    except Exception:
        return None   # or 0, depending on how you want to handle errors

# ============================== LOGGING ==============================
def setup_logging(log_dir: str | Path, log_prefix: str = "run"):
    """Set up logging to both console and file with timestamp."""
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

def _json_safe(o):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(o, (np.int32, np.int64, np.int8, np.int16)):
        return int(o)
    elif isinstance(o, (np.float32, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

# =================== XGBOOST VERSION-AGNOSTIC EARLY STOP HELPERS ===================
def _fit_xgb_with_early_stop(model, X_train, y_train, X_val, y_val, early_rounds=50):
    """Version-agnostic XGB fit with early stopping."""
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
    """Use the best iteration/ntree_limit when available."""
    try:
        if hasattr(model, "best_iteration") and model.best_iteration is not None:
            return model.predict_proba(X, iteration_range=(0, model.best_iteration + 1))
    except TypeError:
        pass
    try:
        if hasattr(model, "best_ntree_limit") and model.best_ntree_limit is not None:
            return model.predict_proba(X, ntree_limit=model.best_ntree_limit)
    except Exception:
        pass
    return model.predict_proba(X)

# =================== BAYESIAN OPTIMIZATION WITH 80/20 SPLIT ===================
def intra_bayesian_optimization(
    dfs,
    loop,
    feature_cols,
    *,
    n_calls=20,
    n_splits=5,
    n_workers=1,  # kept for API, but training is sequential
    results_dir="bayes_results",
    save_plot=True,
    random_state=42,
    colset_name: str = ""
):
    """
    Sequential CV with memory-safe XGBoost settings.
    Uses a stratified 80/20 train/hold-out split.
    CV + Bayesian optimization are done on the 80% train subset.
    Then we train on that 80% and evaluate once on the 20% hold-out,
    producing a second AUC.
    """
    # ------------------------------- data -------------------------------
    data = dfs[dfs['loops'] == loop].replace(np.inf, np.nan)
    target_col = 'COEFFICIENTS'

    # Cast once to float32
    X = data[feature_cols].to_numpy(dtype=np.float32, copy=False)
    target_series = data[target_col].apply(to_numeric)
    y = target_series.ne(0).astype(int).to_numpy().ravel()

    logging.info(f"Features: {len(feature_cols)}")
    logging.info(f"Target distribution (full): {dict(zip(*np.unique(y, return_counts=True)))}")

    # ------------- stratified 80/20 train / hold-out split -------------
    X_cv, X_holdout, y_cv, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state
    )

    logging.info(
        "Train/Hold-out split (stratified): "
        f"train={X_cv.shape[0]}, holdout={X_holdout.shape[0]}"
    )
    logging.info(
        f"Train target dist: {dict(zip(*np.unique(y_cv, return_counts=True)))}"
    )
    logging.info(
        f"Hold-out target dist: {dict(zip(*np.unique(y_holdout, return_counts=True)))}"
    )

    # Memory optimization: sample training data if dataset is very large
    # This reduces memory usage during optimization and final evaluation
    # With 200k samples, results remain statistically valid for large datasets
    max_samples_for_optimization = 200000  # Use max 200k samples
    original_size = X_cv.shape[0]
    
    if X_cv.shape[0] > max_samples_for_optimization:
        logging.info(
            f"Large dataset detected ({original_size} samples). "
            f"Sampling {max_samples_for_optimization} samples to reduce memory usage. "
            f"This will be used for both optimization and final evaluation."
        )
        # Stratified sampling to preserve class distribution
        X_cv_sampled, _, y_cv_sampled, _ = train_test_split(
            X_cv,
            y_cv,
            train_size=max_samples_for_optimization,
            stratify=y_cv,
            random_state=random_state
        )
        # Replace with sampled data and free original
        del X_cv, y_cv
        gc.collect()
        X_cv = X_cv_sampled
        y_cv = y_cv_sampled
        logging.info(
            f"Using {X_cv.shape[0]} samples (down from {original_size}) for optimization and evaluation."
        )
    else:
        logging.info(f"Using full training set ({X_cv.shape[0]} samples).")

    # Output dir for this loop/colset
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path(results_dir)
    if colset_name:
        run_dir = results_dir / f"loop{loop}_{colset_name}_{ts}"
    else:
        run_dir = results_dir / f"loop{loop}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # --------------------- search space ---------------------
    dimensions = [
        Integer(50, 1000, name='n_estimators'),
        Integer(3, 12, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0.0, 5.0, name='reg_alpha'),
        Real(0.0, 10.0, name='reg_lambda'),
    ]

    # CV is only over the training portion (X_cv, y_cv)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_jobs = 1  # force sequential folds to minimize memory

    def _fit_eval_fold(train_idx, test_idx, params):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=1,               # avoid nested parallelism
            tree_method="hist",     # memory efficient on CPU
            max_bin=256,
            predictor="cpu_predictor",
            **params
        )

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
            with parallel_backend("loky", inner_max_num_threads=1):
                with Parallel(
                    n_jobs=fold_jobs,
                    backend="loky",
                    prefer=None,
                    temp_folder=JOBLIB_TMPDIR
                ) as parallel:
                    fold_aucs = parallel(
                        delayed(_fit_eval_fold)(train_idx, test_idx, params)
                        for train_idx, test_idx in kf.split(X_cv, y_cv)
                    )
            mean_auc = float(np.mean(fold_aucs))
            std_auc = float(np.std(fold_aucs))
            logging.info(f"CV Score (train 80%): {mean_auc:.3f} ± {std_auc:.3f} | Params: {params}")

            # Cleanup after each optimization iteration
            del fold_aucs
            gc.collect()

            history_rows.append({
                **params,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "timestamp": time.time()
            })
            return -mean_auc
        except Exception as e:
            logging.error(f"Error with params {params}: {e}")
            history_rows.append({
                **params,
                "mean_auc": np.nan,
                "std_auc": np.nan,
                "error": str(e),
                "timestamp": time.time()
            })
            return 1.0

    # -------------------------- run optimization --------------------------
    logging.info("\n" + "="*60)
    logging.info("Starting Bayesian Hyperparameter Optimization (on 80% train)")
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
    logging.info("OPTIMIZATION RESULTS (CV on train set)")
    logging.info("="*60)

    best_params = dict(zip([d.name for d in dimensions], result.x))
    best_score = float(-result.fun)
    logging.info(f"Best CV Score (train 80%): {best_score:.4f}")
    logging.info("Best Parameters:")
    for k, v in best_params.items():
        logging.info(f"  {k}: {v}")

    # ----------------------- final CV evaluation (train only, sequential) -----------------------
    logging.info("\n" + "="*60)
    logging.info("FINAL CV EVALUATION WITH BEST PARAMETERS (train 80%)")
    logging.info("="*60)

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = np.zeros_like(thresholds, dtype=float)
    recalls = np.zeros_like(thresholds, dtype=float)
    f1s = np.zeros_like(thresholds, dtype=float)
    balanced_accuracies = np.zeros_like(thresholds, dtype=float)

    def _scores_for_fold(train_idx, test_idx):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]

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
        del X_train, X_test
        gc.collect()
        return y_test, y_scores, auc_val

    with parallel_backend("loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=1,  # force sequential
            backend="loky",
            prefer=None,
            temp_folder=JOBLIB_TMPDIR
        ) as parallel:
            fold_results = parallel(
                delayed(_scores_for_fold)(train_idx, test_idx)
                for train_idx, test_idx in kf.split(X_cv, y_cv)
            )

    roc_aucs = []
    for fold, (y_test, y_scores, auc_val) in enumerate(fold_results, start=1):
        roc_aucs.append(auc_val)
        logging.info(f"Fold {fold} ROC AUC (train CV): {auc_val:.3f}")

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
    recalls /= n_folds
    f1s /= n_folds
    balanced_accuracies /= n_folds

    mean_auc = float(np.mean(roc_aucs))
    std_auc = float(np.std(roc_aucs))
    logging.info(f"\nAverage ROC AUC (train CV): {mean_auc:.3f} ± {std_auc:.3f}")

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

    # OOF AUC on 80% training split (single and mean over folds)
    logging.info("\n" + "="*60)
    logging.info("OOF AUC COMPUTATION ON 80% TRAINING SPLIT")
    logging.info("="*60)

    def _oof_scores_for_train_split(train_idx, test_idx):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]

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
        auc_fold = float(roc_auc_score(y_test, y_scores))
        return y_test, y_scores, auc_fold

    with parallel_backend("loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=1,
            backend="loky",
            prefer=None,
            temp_folder=JOBLIB_TMPDIR
        ) as parallel:
            fold_out_train = parallel(
                delayed(_oof_scores_for_train_split)(train_idx, test_idx)
                for train_idx, test_idx in kf.split(X_cv, y_cv)
            )

    y_true_oof_train = np.concatenate([yt for yt, _, _ in fold_out_train])
    y_score_oof_train = np.concatenate([ys for _, ys, _ in fold_out_train])
    fold_aucs_train = [auc_fold for _, _, auc_fold in fold_out_train]
    oof_auc_train = float(roc_auc_score(y_true_oof_train, y_score_oof_train))
    oof_auc_train_mean = float(np.mean(fold_aucs_train))
    logging.info(
        f"OOF AUC (80% training split): {oof_auc_train:.4f} (single), "
        f"{oof_auc_train_mean:.4f} (mean over folds)"
    )

    # Optional plot of metrics vs threshold
    plot_path = None
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label="Precision")
        plt.plot(thresholds, recalls, label="Recall")
        plt.plot(thresholds, f1s, label="F1 Score")
        plt.plot(thresholds, balanced_accuracies, label="Balanced Accuracy")
        plt.xlabel("Threshold")
        plt.ylabel("Metric Score")
        plt.title(
            "Cross-Validated Metrics vs. Threshold (Optimized, train 80%)\n"
            f"ROC AUC: {mean_auc:.3f} ± {std_auc:.3f}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = run_dir / "metrics.png"
        plt.savefig(plot_path, dpi=160)
        plt.close()

    # ----------------------- hold-out evaluation -----------------------
    logging.info("\n" + "="*60)
    logging.info("HOLD-OUT EVALUATION WITH BEST PARAMETERS (20% test set)")
    logging.info("="*60)

    holdout_model = XGBClassifier(
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=1,
        tree_method="hist",
        max_bin=256,
        predictor="cpu_predictor",
        **best_params
    )
    # Train on all CV-train data (80%)
    holdout_model.fit(X_cv, y_cv)

    y_holdout_scores = holdout_model.predict_proba(X_holdout)[:, 1]
    holdout_auc = float(roc_auc_score(y_holdout, y_holdout_scores))
    logging.info(f"Hold-out ROC AUC: {holdout_auc:.3f}")

    logging.info("\n" + "="*60)
    logging.info("SUMMARY")
    logging.info("="*60)
    logging.info(f"✓ Bayesian optimization completed with {len(result.func_vals)} iterations")
    logging.info(f"✓ Best train-CV score: {best_score:.4f}")
    logging.info(f"✓ OOF AUC (80% training split): {oof_auc_train:.4f}")
    logging.info(f"✓ Hold-out ROC AUC: {holdout_auc:.4f}")
    logging.info(f"Artifacts saved to: {run_dir.resolve()}")

    summary = {
        "loop": int(loop),
        "colset": colset_name,
        "timestamp": ts,
        "n_calls": int(n_calls),
        "n_splits": int(n_splits),
        "n_workers": int(1),
        "random_state": int(random_state),
        "feature_count": int(len(feature_cols)),
        "best_params": best_params,
        "best_cv_score": best_score,      # CV on train
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc,
        "oof_auc_train": oof_auc_train,   # OOF AUC on 80% training split (single)
        "oof_auc_train_mean": oof_auc_train_mean,  # mean over folds
        "holdout_auc": holdout_auc,
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
        'oof_auc_train': oof_auc_train,
        'oof_auc_train_mean': oof_auc_train_mean,
        'holdout_auc': holdout_auc,
        'optimization_history_df': hist_df,
        'thresholds_df': thr_df,
        'artifacts_dir': str(run_dir.resolve())
    }

# =================== OUT-OF-FOLD PREDICTIONS (FULL DATA) ===================
def _oof_scores_with_best_params(dfs, feature_cols, loop, best_params, *,
                                 n_splits=5, n_workers=1, random_state=42):
    """Out-of-fold predictions (no leakage), sequential, no scaling, on full data for the loop."""
    data = dfs[dfs['loops'] == loop].replace(np.inf, np.nan)
    X = data[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y = data['COEFFICIENTS'].apply(to_numeric).ne(0).astype(int).to_numpy().ravel()

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_jobs = 1  # force sequential

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
        auc_fold = float(roc_auc_score(y_test, y_scores))
        return y_test, y_scores, auc_fold

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

    y_true = np.concatenate([yt for yt, _, _ in fold_out])
    y_score = np.concatenate([ys for _, ys, _ in fold_out])
    fold_aucs = [auc_fold for _, _, auc_fold in fold_out]
    mean_fold_auc = float(np.mean(fold_aucs))
    return y_true, y_score, mean_fold_auc

# =================== SHAP COMPUTATION ===================
def calculate_global_shap_values(
    dfs, feature_columns, best_params, target_loop,
    shap_sample_size=None, random_state=42
):
    """Calculate global SHAP values with memory-safe XGB settings."""
    data = dfs[dfs['loops'] == target_loop].replace(np.inf, np.nan)
    X = data[feature_columns].to_numpy(dtype=np.float32, copy=False)
    y = data['COEFFICIENTS'].apply(to_numeric).ne(0).astype(int).to_numpy().ravel()

    model = XGBClassifier(
        eval_metric='logloss',
        random_state=random_state,
        n_jobs=1,
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
    """Get global feature importance from SHAP values."""
    global_importance = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance
    }).sort_values('importance', ascending=False)
    return importance_df

def get_global_feature_direction(shap_values, feature_names):
    """Get mean signed SHAP values (feature direction)."""
    mean_shap = np.mean(shap_values, axis=0)
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_signed_shap': mean_shap
    }).sort_values('mean_signed_shap', ascending=False)
    return df

# =================== COLUMN GROUPING ===================
def get_column_sets(dfs, loop, column_defs=None):
    """
    Get column sets as defined in modelling_function_instance_faster.py.
    Returns a dictionary with column sets (motifs_eig, motifs, eig, motifs_eig_centrality).
    """
    data = dfs[dfs['loops'] == loop].replace(np.inf, np.nan)
    
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

    repeat_motifs = [
        'Motif_induced_C4',
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
        'Motif_induced_TailedTriangle_per_Cn4'
    ]

    if column_defs is None:
        all_cols = [x for x in data.columns if (x not in ["loops", "COEFFICIENTS"] and x not in repeat_motifs)]
        cols0 = [x for x in all_cols if ('motif' in x.lower()) or ('eig_' in x.lower() and 'centrality' not in x.lower())]
        cols00 = cols0 + centrality_cols
        cols1 = [x for x in all_cols if 'motif' in x.lower()]
        cols5 = [x for x in all_cols if ('eig_' in x.lower() and 'centrality' not in x.lower())]
        cols2 = [x for x in all_cols if ('eig_' in x.lower() and 'centrality' not in x.lower()) or ('spectral' in x.lower() and 'centrality' not in x.lower())]
        column_defs = {
            #"all_columns": all_cols,
            "motifs_eig": cols0,
            "motifs": cols1,
            "eig": cols5,
            "motifs_eig_centrality": cols00,
            #"spectral": cols2
        }
    
    return column_defs

def extract_columns_by_suffix(dfs, loop, base_column_names):
    """
    Extract columns matching base names with NUM, DEN, TOTAL suffixes.
    Returns a dictionary with 'NUM', 'DEN', 'TOTAL', 'all_suffixes' keys.
    """
    data = dfs[dfs['loops'] == loop]
    all_available_cols = [c for c in data.columns if c not in ['loops', 'COEFFICIENTS', 'Unnamed: 0']]
    
    # Extract base names (remove suffix if present)
    base_names = set()
    for col in base_column_names:
        base = col
        for suffix in ['_NUM', '_DEN', '_TOTAL']:
            if col.endswith(suffix):
                base = col[:-len(suffix)]
                break
        base_names.add(base)
    
    num_cols = []
    den_cols = []
    total_cols = []
    all_suffixes_cols = []  # all columns regardless of suffix
    
    for base in base_names:
        num_col = f"{base}_NUM"
        den_col = f"{base}_DEN"
        total_col = f"{base}_TOTAL"
        
        if num_col in all_available_cols:
            num_cols.append(num_col)
            all_suffixes_cols.append(num_col)
        if den_col in all_available_cols:
            den_cols.append(den_col)
            all_suffixes_cols.append(den_col)
        if total_col in all_available_cols:
            total_cols.append(total_col)
            all_suffixes_cols.append(total_col)
    
    return {
        'NUM': num_cols,
        'DEN': den_cols,
        'TOTAL': total_cols,
        'all_suffixes': all_suffixes_cols
    }

# =================== MAIN PIPELINE ===================
def run_bayes_shap_for_loop(
    i: int,
    *,
    input_csv_template="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/new_features/new2_merged/merged/dataset/dataset/dataset/{i}loops_merged.parquet",
    output_root="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/results/bayes_shap_result",
    n_calls=20,
    n_splits=5,
    n_workers=1,
    random_state=42,
    save_plot_threshold_curves=False,
    shap_sample_size: int = 500
):
    """
    End-to-end pipeline for fgraph features with separate SHAP analysis for:
    - All columns
    - NUM columns
    - DEN columns
    - TOTAL columns
    
    Saves results in fgraphs_loop_# folders.
    """
    # Setup logging
    output_root_path = Path(output_root)
    log_file = setup_logging(
        log_dir=output_root_path,
        log_prefix=f"fgraphs_loop{i}"
    )
    logging.info(f"Starting fgraphs SHAP analysis for loop {i}")
    logging.info(f"Configuration:")
    logging.info(f"  n_calls: {n_calls}")
    logging.info(f"  n_splits: {n_splits}")
    logging.info(f"  n_workers: {n_workers} (forced to 1 in CV)")
    logging.info(f"  shap_sample_size: {shap_sample_size}")
    logging.info(f"  random_state: {random_state}")
    logging.info(f"  save_plot_threshold_curves: {save_plot_threshold_curves}")
    logging.info(f"  output_root: {output_root}")

    # Load data with memory-efficient chunked reading for large CSVs
    csv_path = input_csv_template.format(i=i)
    logging.info(f"Loading data from: {csv_path}")
    try:
        if csv_path.endswith('.parquet'):
            try:
                import pyarrow.parquet as pq  # noqa: F401
                dfs = pd.read_parquet(csv_path)
            except ImportError:
                logging.warning("pyarrow not available, trying pandas read_parquet...")
                dfs = pd.read_parquet(csv_path)
        else:
            # Memory-efficient chunked reading for large CSV files
            file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
            logging.info(f"File size: {file_size_mb:.1f} MB")
            
            if file_size_mb > 500:  # Use chunked reading for files > 500MB
                logging.info("Using chunked reading for memory efficiency...")
                chunk_list = []
                chunk_size = 100000  # Read 100k rows at a time
                
                # Read a sample to identify columns that need object dtype (contain non-numeric values)
                sample = pd.read_csv(csv_path, nrows=min(10000, chunk_size))
                dtypes_dict = {}
                # Identify columns that contain non-numeric string values (like '1/2', '-1/2')
                for col in sample.columns:
                    # Check if column contains values that look like fractions or non-numeric strings
                    sample_values = sample[col].astype(str).head(100)
                    has_fraction = sample_values.str.contains(r'^[-+]?\d+/\d+', na=False).any()
                    # Also check COEFFICIENTS column specifically
                    if col == 'COEFFICIENTS' or has_fraction:
                        dtypes_dict[col] = 'object'
                        logging.info(f"  Keeping column '{col}' as object type (contains fractional/non-numeric values)")
                
                # Read in chunks with dtype specification for problematic columns
                chunk_reader = pd.read_csv(csv_path, chunksize=chunk_size, dtype=dtypes_dict, low_memory=False)
                for chunk in chunk_reader:
                    # Downcast numeric columns after reading (errors='coerce' handles non-numeric values)
                    for col in chunk.select_dtypes(include=['float64']).columns:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce', downcast='float')
                    for col in chunk.select_dtypes(include=['int64']).columns:
                        # Only downcast if all values are numeric
                        try:
                            chunk[col] = pd.to_numeric(chunk[col], errors='coerce', downcast='integer')
                        except (ValueError, TypeError):
                            # If conversion fails, keep as object/string
                            pass
                    chunk_list.append(chunk)
                    logging.info(f"  Loaded chunk: {len(chunk)} rows (total so far: {sum(len(c) for c in chunk_list)})")
                
                dfs = pd.concat(chunk_list, ignore_index=True)
                del chunk_list, sample
                gc.collect()
                logging.info(f"Chunked loading complete. Final shape: {dfs.shape}")
            else:
                # For smaller files, read normally but still optimize dtypes
                dfs = pd.read_csv(csv_path, low_memory=False)
                # Downcast numeric columns to save memory
                for col in dfs.select_dtypes(include=['float64']).columns:
                    dfs[col] = pd.to_numeric(dfs[col], errors='coerce', downcast='float')
                for col in dfs.select_dtypes(include=['int64']).columns:
                    dfs[col] = pd.to_numeric(dfs[col], errors='coerce', downcast='integer')
        
        dfs["loops"] = i
        logging.info(f"Data loaded successfully. Shape: {dfs.shape}, Memory usage: {dfs.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    except Exception as e:
        logging.error(f"Error loading data from {csv_path}: {e}")
        raise

    # Get column sets (same as modelling_function_instance_faster.py)
    column_sets = get_column_sets(dfs, i, column_defs=None)
    
    # Get all available columns for optimization
    data = dfs[dfs['loops'] == i].replace(np.inf, np.nan)
    all_columns = [c for c in data.columns if c not in ['loops', 'COEFFICIENTS', 'Unnamed: 0']]
    
    # Filter to numeric columns only
    numeric_all_cols = []
    for col in all_columns:
        try:
            pd.to_numeric(data[col], errors='raise')
            numeric_all_cols.append(col)
        except (ValueError, TypeError):
            continue

    if not numeric_all_cols:
        logging.error("No feature columns found!")
        return None

    # Create output directory
    output_dir = output_root_path / f"fgraphs_loop_{i}"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    index_rows = []
    auc_summary_rows = []

    # =================== PROCESS EACH COLUMN SET ===================
    for colset_name, base_column_names in column_sets.items():
        if not base_column_names:
            logging.info(f"[loop {i} | {colset_name}] Skipping (no base columns).")
            continue

        logging.info(f"\n{'='*80}")
        logging.info(f"Processing column set: {colset_name}")
        logging.info(f"{'='*80}")

        # Extract all columns for this column set (with all suffixes)
        suffix_groups = extract_columns_by_suffix(dfs, i, base_column_names)
        
        # Get all columns for this column set (NUM + DEN + TOTAL)
        all_colset_cols = suffix_groups['all_suffixes']
        
        if not all_colset_cols:
            logging.info(f"  [{colset_name}] Skipping (no matching columns found).")
            continue

        # Create directory for this column set
        colset_dir = output_dir / colset_name
        colset_dir.mkdir(parents=True, exist_ok=True)

        # =================== STEP 1: OPTIMIZE FOR THIS COLUMN SET ===================
        logging.info(f"  Starting Bayesian optimization for {colset_name} ({len(all_colset_cols)} features)...")
        results = intra_bayesian_optimization(
            dfs,
            i,
            all_colset_cols,
            n_calls=n_calls,
            n_splits=n_splits,
            n_workers=1,
            results_dir=output_dir / "bayes_results",
            save_plot=save_plot_threshold_curves,
            random_state=random_state,
            colset_name=colset_name
        )
        logging.info(
            f"  Bayesian optimization completed for {colset_name}. "
            f"Train-CV AUC={results.get('cv_auc'):.3f}, "
            f"Hold-out AUC={results.get('holdout_auc'):.3f}"
        )

        best_params = results["best_params"]
        
        # Save best params and optimization history for this column set
        with open(colset_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2, default=_json_safe)
        
        results["optimization_history_df"].to_csv(
            colset_dir / "bayes_history.csv", index=False
        )
        # Save thresholds for this column set
        results["thresholds_df"].to_csv(
            colset_dir / "thresholds.csv", index=False
        )

        # =================== STEP 2: COMPUTE OUT-OF-FOLD AUC ON FULL DATA ===================
        logging.info(f"  Computing out-of-fold predictions for {colset_name} on full loop data...")
        y_true_oof, y_score_oof, oof_auc_mean_full = _oof_scores_with_best_params(
            dfs, all_colset_cols, i, best_params,
            n_splits=n_splits, n_workers=1, random_state=random_state
        )
        auc_all_suffixes = roc_auc_score(y_true_oof, y_score_oof)
        logging.info(
            f"  Model AUC (OOF, all_suffixes): {auc_all_suffixes:.4f} (single), "
            f"{oof_auc_mean_full:.4f} (mean over folds)"
        )

        # =================== STEP 3: TRAIN MODEL AND COMPUTE SHAP FOR THIS COLUMN SET ===================
        logging.info(f"  Computing SHAP values for {colset_name}...")
        shap_values_all, explainer, model, X_used_all = calculate_global_shap_values(
            dfs, 
            feature_columns=all_colset_cols, 
            best_params=best_params, 
            target_loop=i,
            shap_sample_size=shap_sample_size, 
            random_state=random_state
        )
        logging.info(f"  SHAP computation completed. Shape: {np.array(shap_values_all).shape}")

        # Create a mapping from feature name to index for this column set
        feature_to_idx = {feat: idx for idx, feat in enumerate(all_colset_cols)}

        holdout_auc = results.get("holdout_auc")
        oof_auc_train = results.get("oof_auc_train")
        oof_auc_train_mean = results.get("oof_auc_train_mean")

        # =================== STEP 4: FILTER SHAP VALUES BY SUFFIX ===================
        for suffix_name, feature_cols in suffix_groups.items():
            if not feature_cols:
                logging.info(f"    [{colset_name} | {suffix_name}] Skipping (no features).")
                continue

            logging.info(f"\n    Creating SHAP visualization: {colset_name} | {suffix_name} | {len(feature_cols)} features")

            # Create subdirectory for this suffix
            suffix_dir = colset_dir / suffix_name
            suffix_dir.mkdir(parents=True, exist_ok=True)

            # Get indices of features in this group
            group_indices = [feature_to_idx[feat] for feat in feature_cols if feat in feature_to_idx]
            
            if not group_indices:
                logging.warning(f"    No matching features found for {colset_name} | {suffix_name}")
                continue

            # Extract SHAP values for this group
            shap_values_group = shap_values_all[:, group_indices]
            X_used_group = X_used_all[:, group_indices]

            # Compute feature importance and direction for this group
            importance_df = get_global_feature_importance(shap_values_group, feature_cols)
            direction_df = get_global_feature_direction(shap_values_group, feature_cols)
            shap_full = importance_df.merge(direction_df, on="feature")
            
            # Save SHAP results
            shap_full_path = suffix_dir / "shap_full.csv"
            shap_full.to_csv(shap_full_path, index=False)
            logging.info(f"    Saved SHAP full results to: {shap_full_path}")

            # SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_group, X_used_group, feature_names=feature_cols, max_display=20, show=False)
            plt.title(f"SHAP Summary (loop {i} | {colset_name} | {suffix_name})")
            shap_png_path = suffix_dir / "shap_summary.png"
            plt.savefig(shap_png_path, bbox_inches='tight', dpi=200)
            plt.close()
            logging.info(f"    Saved SHAP summary plot to: {shap_png_path}")

            # SHAP sample CSV
            if shap_sample_size is not None and len(shap_values_group) <= shap_sample_size:
                shap_sample = pd.DataFrame(shap_values_group, columns=feature_cols)
                shap_sample["sample_index"] = np.arange(len(shap_sample))
            else:
                n_samples = min(shap_sample_size, len(shap_values_group)) if shap_sample_size else len(shap_values_group)
                shap_sample = resample(
                    pd.DataFrame(shap_values_group, columns=feature_cols),
                    n_samples=n_samples,
                    random_state=random_state
                )
                shap_sample["sample_index"] = np.arange(len(shap_sample))

            shap_sample_path = suffix_dir / "shap_sample.csv"
            shap_sample.to_csv(shap_sample_path, index=False)
            logging.info(f"    Saved SHAP sample CSV to: {shap_sample_path}")

            # Use the same AUC for all suffix groups (from the single model trained on all_suffixes)
            auc_val = auc_all_suffixes if suffix_name == 'all_suffixes' else None

            # Index artifacts
            index_rows.append({
                "loop": i,
                "colset": colset_name,
                "suffix": suffix_name,
                "n_features": len(feature_cols),
                "auc": float(auc_all_suffixes) if auc_val is not None else None,
                "best_params_json": str(colset_dir / "best_params.json"),
                "bayes_history_csv": str(colset_dir / "bayes_history.csv"),
                "thresholds_csv": str(colset_dir / "thresholds.csv"),
                "shap_full_csv": str(shap_full_path),
                "shap_summary_png": str(shap_png_path),
                "shap_sample_csv": str(shap_sample_path),
                "oof_auc_train": oof_auc_train,
                "oof_auc_train_mean": oof_auc_train_mean,
                "holdout_auc": holdout_auc,
                "optimizer_artifacts_dir": results.get("artifacts_dir")
            })

            if suffix_name == 'all_suffixes':
                auc_summary_rows.append({
                    "loop": i,
                    "colset": colset_name,
                    "suffix": suffix_name,
                    "n_features": len(feature_cols),
                    "auc": float(auc_all_suffixes),
                    "oof_auc_train": oof_auc_train,
                    "oof_auc_train_mean": oof_auc_train_mean,
                    "holdout_auc": holdout_auc,
                    "best_cv_score": results.get("best_cv_score"),
                    "cv_auc_mean": results.get("cv_auc"),
                    "cv_auc_std": results.get("cv_auc_std"),
                })

            logging.info(f"    ✓ Completed SHAP visualization for {colset_name} | {suffix_name}")

    # Save artifact index
    if index_rows:
        idx_df = pd.DataFrame(index_rows)
        idx_path = output_dir / f"artifact_index_{ts}.csv"
        idx_df.to_csv(idx_path, index=False)
        logging.info(f"\n✅ Artifact index saved to: {idx_path}")

    # Save AUC summary
    if auc_summary_rows:
        auc_df = pd.DataFrame(auc_summary_rows)
        auc_df = auc_df.sort_values("auc", ascending=False)
        auc_path = output_dir / f"auc_summary_{ts}.csv"
        auc_df.to_csv(auc_path, index=False)
        logging.info(f"\n📄 AUC summary saved to: {auc_path}")
        logging.info("\nAUC Summary (descending):")
        logging.info(auc_df.to_string(index=False))

    logging.info("\n" + "="*80)
    logging.info(f"Pipeline completed for loop {i}")
    logging.info(f"All outputs saved to: {output_dir}")
    logging.info(f"All outputs logged to: {log_file}")
    logging.info("="*80)

    return {
        "output_dir": str(output_dir),
        "artifact_index": str(idx_path) if index_rows else None,
        "auc_summary": str(auc_path) if auc_summary_rows else None,
    }

# =================== MAIN ENTRY POINT ===================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SHAP analysis for fgraph features")
    parser.add_argument("--loop", type=int, required=True, help="Loop number")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/new_features/new2_merged/merged/dataset/dataset/dataset/{i}loops_merged.parquet",
        help="Input CSV/Parquet template (use {i} for loop number)"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/results/bayes_shap_result",
        help="Output root directory"
    )
    parser.add_argument("--n-calls", type=int, default=20, help="Number of Bayesian optimization calls")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--shap-sample-size", type=int, default=500, help="SHAP sample size")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument(
        "--save-plot-threshold-curves",
        action="store_true",
        help="If set, save metrics vs threshold plots during optimization"
    )
    
    args = parser.parse_args()
    
    run_bayes_shap_for_loop(
        i=args.loop,
        input_csv_template=args.input,
        output_root=args.output_root,
        n_calls=args.n_calls,
        n_splits=args.n_splits,
        shap_sample_size=args.shap_sample_size,
        random_state=args.random_state,
        save_plot_threshold_curves=args.save_plot_threshold_curves
    )
