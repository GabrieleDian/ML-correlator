# Import the libraries we need to use in this lab
from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ast
import time
# model trainin set
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
#models 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
#model evaluation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*sklearn\.utils\.extmath")


# Minimal Bayesian Hyperparameter Optimization for 6-Loop Intra-Loop Experiment
# =============================================================================
# Based on your existing intra-loop code structure

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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


import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args


# Add this small helper near the top of your script (or inside the function)
def _json_safe(o):
    if isinstance(o, (np.int32, np.int64)):
        return int(o)
    elif isinstance(o, (np.float32, np.float64)):
        return float(o)
    elif isinstance(o, (np.ndarray,)):
        return o.tolist()
    else:
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def intra_bayesian_optimization(
    dfs,
    loop,
    *,
    n_calls=20,
    n_splits=5,
    n_workers=1,                 # <— parallel workers (fold-level & XGB)
    results_dir="bayes_results", # <— base folder to save all artifacts
    save_plot=True,
    random_state=42
):
    """
    Improved: parallel CV, parallel XGBoost training, and persistent results.

    Files created (under {results_dir}/{timestamp}/):
      - summary.json  : best params/scores, CV AUCs, config, paths
      - history.csv   : per-iteration params and CV score from gp_minimize
      - thresholds.csv: threshold grid with mean precision/recall/F1/balAcc
      - metrics.png   : (optional) thresholds plot
    """

    # =============================================================================
    # SETUP: data
    # =============================================================================
    data = dfs[dfs['loops'] == loop]
    print(f"Data shape: {data.shape}")
    data_cols = [c for c in data.columns
                 if 'COEFFICIENTS' not in c and 'loop' not in c]
    target_col = 'COEFFICIENTS'
    X = data[data_cols].values
    y = data[target_col].values.ravel()

    print(f"Features: {len(data_cols)}")
    print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Results directory (timestamped)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(results_dir) / f"loop{loop}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # =============================================================================
    # BAYESIAN OPTIMIZATION SPACE
    # =============================================================================
    dimensions = [
        Integer(50, 1000, name='n_estimators'),
        Integer(3, 20, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0.0, 10.0, name='reg_alpha'),
        Real(0.0, 10.0, name='reg_lambda'),
    ]

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Helper: one fold training/eval (so we can parallelize)
    def _fit_eval_fold(train_idx, test_idx, params):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=n_workers,   # XGBoost parallelism within each fit
            **params
        )
        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_scores)

    # Track optimization history in memory (we’ll also save to CSV)
    history_rows = []

    @use_named_args(dimensions=dimensions)
    def objective(**params):
        try:
            # Parallelize across CV folds
            fold_aucs = Parallel(n_jobs=n_workers, prefer="threads")(
                delayed(_fit_eval_fold)(train_idx, test_idx, params)
                for train_idx, test_idx in kf.split(X, y)
            )
            mean_auc = float(np.mean(fold_aucs))
            std_auc = float(np.std(fold_aucs))
            print(f"CV Score: {mean_auc:.3f} ± {std_auc:.3f} | Params: {params}")

            # Append to history
            history_rows.append({
                **params,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "timestamp": time.time()
            })

            # Minimize => return negative AUC
            return -mean_auc

        except Exception as e:
            print(f"Error with params {params}: {e}")
            history_rows.append({**params, "mean_auc": np.nan, "std_auc": np.nan,
                                 "error": str(e), "timestamp": time.time()})
            return 1.0

    # =============================================================================
    # RUN BAYESIAN OPTIMIZATION
    # =============================================================================
    print("\n" + "="*60)
    print("Starting Bayesian Hyperparameter Optimization")
    print("="*60)

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

    # =============================================================================
    # RESULTS
    # =============================================================================
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)

    best_params = dict(zip([d.name for d in dimensions], result.x))
    best_score = float(-result.fun)
    print(f"Best CV Score: {best_score:.4f}")
    print("Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # =============================================================================
    # FINAL EVALUATION WITH BEST PARAMETERS (parallelized)
    # =============================================================================
    print("\n" + "="*60)
    print("FINAL EVALUATION WITH BEST PARAMETERS")
    print("="*60)

    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = np.zeros_like(thresholds, dtype=float)
    recalls = np.zeros_like(thresholds, dtype=float)
    f1s = np.zeros_like(thresholds, dtype=float)
    balanced_accuracies = np.zeros_like(thresholds, dtype=float)

    # Evaluate folds in parallel to collect scores for threshold sweep
    def _scores_for_fold(train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=n_workers,
            **best_params
        )
        model.fit(X_train, y_train)
        y_scores = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_scores)
        return y_test, y_scores, auc

    fold_results = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(_scores_for_fold)(train_idx, test_idx)
        for train_idx, test_idx in kf.split(X, y)
    )

    roc_aucs = []
    for fold, (y_test, y_scores, auc) in enumerate(fold_results, start=1):
        roc_aucs.append(auc)
        print(f"Fold {fold} ROC AUC: {auc:.3f}")

        for i, t in enumerate(thresholds):
            y_pred = (y_scores >= t).astype(int)
            precisions[i] += precision_score(y_test, y_pred, zero_division=0)
            recalls[i]    += recall_score(y_test, y_pred)
            f1s[i]        += f1_score(y_test, y_pred)
            balanced_accuracies[i] += balanced_accuracy_score(y_test, y_pred)

    n_folds = kf.get_n_splits()
    precisions /= n_folds
    recalls    /= n_folds
    f1s        /= n_folds
    balanced_accuracies /= n_folds

    mean_auc = float(np.mean(roc_aucs))
    std_auc  = float(np.std(roc_aucs))
    print(f"\nAverage ROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")

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

    # Plot & save
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

    # =============================================================================
    # SUMMARY + SAVE
    # =============================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Bayesian optimization completed with {len(result.func_vals)} iterations")
    print(f"✓ Best CV score: {best_score:.4f}")
    print(f"Artifacts saved to: {run_dir.resolve()}")

    summary = {
        "loop": int(loop),
        "timestamp": ts,
        "n_calls": int(n_calls),
        "n_splits": int(n_splits),
        "n_workers": int(n_workers),
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

    # Return results programmatically too
    bayesian_results = {
        'best_params': best_params,
        'best_cv_score': best_score,
        'cv_auc': mean_auc,
        'cv_auc_std': std_auc,
        'optimization_history_df': hist_df,   # convenient to inspect in-session
        'thresholds_df': thr_df,
        'artifacts_dir': str(run_dir.resolve())
    }
    return bayesian_results

import shap
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def calculate_global_shap_values(dfs, feature_columns, best_params, target_loop=6):
    """
    Calculate global SHAP values for the model.
    
    Args:
        dfs: DataFrame with features and target
        feature_columns: List of feature column names
        best_params: Best hyperparameters for the model
        target_loop: Which loop to analyze
        
    Returns:
        shap_values, shap_explainer, model, X_scaled
    """
    
    # Prepare data
    data = dfs[dfs['loops'] == target_loop]
    X = data[feature_columns].values
    y = data['COEFFICIENTS'].values.ravel()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model with best parameters
    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_jobs=1,  # Use n_jobs=1 for SHAP compatibility
        **best_params
    )
    model.fit(X_scaled, y)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for all samples
    shap_values = explainer.shap_values(X_scaled)
    
    return shap_values, explainer, model, X_scaled

# Calculate global importance (mean absolute SHAP values)
def get_global_feature_importance(shap_values, feature_names):
    """
    Get global feature importance from SHAP values.
    """
    # Calculate mean absolute SHAP values for each feature
    global_importance = np.mean(np.abs(shap_values), axis=0)
    
    # Create DataFrame with feature names and importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': global_importance
    }).sort_values('importance', ascending=False)
    
    return importance_df

def get_global_feature_direction(shap_values, feature_names):
    """
    Compute mean signed SHAP values (directional global effects).
    """
    mean_shap = np.mean(shap_values, axis=0)
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_signed_shap': mean_shap
    }).sort_values('mean_signed_shap', ascending=False)
    return df


import os, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from joblib import Parallel, delayed
from sklearn.utils import resample
import shap
from xgboost import XGBClassifier

# Assumes these helpers already exist in your environment:
# _json_safe, intra_bayesian_optimization,
# calculate_global_shap_values, get_global_feature_importance, get_global_feature_direction


def run_bayes_shap_for_loop(
    i: int,
    *,
    input_csv_template="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/new_merged/{i}loops_merged.csv",
    output_root="/Users/rezadoobary/Documents/MLCORRELATORS/ML-correlator/Tree_classifier_for_graphs/new_stuff/results/bayes_shap_result",
    n_calls=20,
    n_splits=5,
    n_workers=8,
    random_state=42,
    save_plot_threshold_curves=False,
    column_defs: dict | None = None,
    shap_sample_size: int = 500
):
    """
    End-to-end pipeline for a given loop `i`:
      • Bayesian optimization per column-set
      • Save best params, sweep history, thresholds
      • SHAP: full table, beeswarm (dots) summary plot, sampled "dot" values
      • OOF predictions: ROC(AUC), PR curve
      • Decile metrics: precision/recall by score deciles (CSV + plots)
      • Uniqueness stats per column-set (feature-space duplicates)
      • Duplicate pattern export per column-set (if duplicates exist)
      • Artifact index CSV
      • AUC summary CSV across column-sets (also returned)
    Artifacts saved under: {output_root}/loop{i}/{colset_name}_{timestamp}/
    """

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _oof_scores_with_best_params(dfs, feature_cols, loop, best_params, *, n_splits=5, n_workers=8, random_state=42):
        """Create out-of-fold predictions (no leakage). Returns y_true, y_score."""
        data = dfs[dfs['loops'] == loop]
        X = data[feature_cols].values
        y = data['COEFFICIENTS'].values.ravel()

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        def _fit_predict_fold(train_idx, test_idx):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            model = XGBClassifier(
                eval_metric='logloss',
                random_state=random_state,
                n_jobs=n_workers,
                **best_params
            )
            model.fit(X_train, y_train)
            y_scores = model.predict_proba(X_test)[:, 1]
            return y_test, y_scores

        fold_out = Parallel(n_jobs=n_workers, prefer="threads")(
            delayed(_fit_predict_fold)(tr, te) for tr, te in kf.split(X, y)
        )

        y_true = np.concatenate([yt for yt, _ in fold_out])
        y_score = np.concatenate([ys for _, ys in fold_out])
        return y_true, y_score

    def _make_auc_and_decile_plots(y_true, y_score, out_dir, prefix):
        """Save ROC/PR curves and decile precision/recall artifacts. Return paths + metrics."""
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
        # Deciles (robust to ties/duplicates/NaNs/±inf), 1 = best (highest score)
        dfm = pd.DataFrame({"y": y_true, "score": y_score})

        # Clean score for ranking
        score_clean = (
            pd.Series(dfm["score"], copy=True)
            .replace([np.inf, -np.inf], pd.NA)
        )

        # Percentile rank with ties averaged; highest score -> pct near 1.0
        r_desc = score_clean.rank(pct=True, ascending=False, method="average")

        # Map to deciles 1..20, with 1 = best
        # r_desc in (0,1]; top -> 1, bottom -> 20
        decile = (21 - np.ceil(r_desc * 20)).astype("Int64")

        dfm["decile_rank"] = decile  # Int64 keeps <NA> for missing scores

        total_pos = pd.Series(dfm["y"]).sum()

        dec = (
            dfm.dropna(subset=["decile_rank"])  # drop rows with missing score/decile
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

    def _uniqueness_stats(df_subset: pd.DataFrame) -> dict:
        """
        Compute uniqueness on the given feature subset (rows identical across all columns count as duplicates).
        Returns: n_rows, n_unique_rows, uniqueness_ratio, duplicate_rows
        """
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

    # ---------------------------
    # Load data
    # ---------------------------
    csv_path = input_csv_template.format(i=i)
    dfs = pd.read_csv(csv_path)
    dfs["loops"] = i

    if column_defs is None:
        all_cols = [x for x in dfs.columns if x not in ["loops", "COEFFICIENTS"]]
        cols0 = [x for x in dfs.columns if ('motif' in x.lower()) or ('eig_' in x.lower() and 'centrality' not in x.lower())]
        cols1 = [x for x in dfs.columns if 'motif' in x.lower()]
        cols5 = [x for x in dfs.columns if ('eig_' in x.lower() and 'centrality' not in x.lower())]
        cols2 = [x for x in dfs.columns if ('eig_' in x.lower() and 'centrality' not in x.lower()) or ('spectral' in x.lower() and 'centrality' not in x.lower())]
        column_defs = {"all_columns":all_cols, "motifs_eig": cols0, "motifs": cols1, "spectral": cols2,"eig":cols5}

    output_root = Path(output_root)
    (output_root / f"loop{i}").mkdir(parents=True, exist_ok=True)
    # ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    ts = ""  # overwrite

    index_rows = []
    auc_summary_rows = []

    # ---------------------------
    # Per column-set pipeline
    # ---------------------------
    for colset_name, feature_cols in column_defs.items():
        if not feature_cols:
            print(f"[loop {i} | {colset_name}] Skipping (no features).")
            continue

        print(f"\n=== Loop {i} | Colset: {colset_name} | {len(feature_cols)} features ===")

        # ---- Uniqueness stats on feature space ----
        feature_df = dfs[feature_cols]
        uniq_stats = _uniqueness_stats(feature_df)
        print(
            f"[uniqueness] rows={uniq_stats['n_rows']}, "
            f"unique={uniq_stats['n_unique_rows']}, "
            f"UR={uniq_stats['uniqueness_ratio']:.6f}, "
            f"dupes={uniq_stats['duplicate_rows']}"
        )

        # Optional: write duplicate groups (only if duplicates exist)
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

        # ---- Bayesian optimization (writes its own artifacts under bayes_results/*) ----
        results = intra_bayesian_optimization(
            df_for_bayes,
            i,
            n_calls=n_calls,
            n_splits=n_splits,
            n_workers=n_workers,
            results_dir="bayes_results",
            save_plot=save_plot_threshold_curves,
            random_state=random_state
        )

        # ---- Output folder for this column set ----
        out_dir = output_root / f"loop{i}" / f"{colset_name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- Save best params + sweep + thresholds ----
        best_params = results["best_params"]
        with open(out_dir / f"loop{i}_{colset_name}_best_params.json", "w") as f:
            json.dump(best_params, f, indent=2, default=_json_safe)

        results["optimization_history_df"].to_csv(out_dir / f"loop{i}_{colset_name}_bayes_history.csv", index=False)
        results["thresholds_df"].to_csv(out_dir / f"loop{i}_{colset_name}_thresholds.csv", index=False)

        # ---- SHAP: full table + beeswarm (dots) + sample of dots ----
        shap_values, explainer, model, X_scaled = calculate_global_shap_values(
            dfs, feature_columns=feature_cols, best_params=best_params, target_loop=i
        )

        importance_df = get_global_feature_importance(shap_values, feature_cols)
        direction_df  = get_global_feature_direction(shap_values, feature_cols)
        shap_full = importance_df.merge(direction_df, on="feature")
        shap_full_path = out_dir / f"loop{i}_{colset_name}_shap_full.csv"
        shap_full.to_csv(shap_full_path, index=False)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_scaled, feature_names=feature_cols, max_display=20, show=False)
        plt.title(f"SHAP Summary (loop {i} | {colset_name})")
        shap_png_path = out_dir / f"loop{i}_{colset_name}_shap_summary.png"
        plt.savefig(shap_png_path, bbox_inches='tight', dpi=200)
        plt.close()

        n_samples = min(shap_sample_size, X_scaled.shape[0])
        shap_sample = resample(
            pd.DataFrame(shap_values, columns=feature_cols),
            n_samples=n_samples,
            random_state=random_state
        )
        shap_sample["sample_index"] = np.arange(len(shap_sample))
        shap_sample_path = out_dir / f"loop{i}_{colset_name}_shap_sample.csv"
        shap_sample.to_csv(shap_sample_path, index=False)

        # ---- OOF scores -> ROC/PR & deciles ----
        y_true_oof, y_score_oof = _oof_scores_with_best_params(
            dfs, feature_cols, i, best_params,
            n_splits=n_splits, n_workers=n_workers, random_state=random_state
        )
        curve_artifacts = _make_auc_and_decile_plots(
            y_true_oof, y_score_oof,
            out_dir=out_dir,
            prefix=f"loop{i}_{colset_name}"
        )

        # ---- AUC summary row (now includes uniqueness stats) ----
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

        # ---- Index artifacts (includes uniqueness stats + path to duplicate groups CSV) ----
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

    # ---- Write artifact index for the loop ----
    idx_path = None
    if index_rows:
        idx_df = pd.DataFrame(index_rows)
        idx_path = output_root / f"loop{i}" / f"loop{i}_artifact_index_{ts}.csv"
        idx_df.to_csv(idx_path, index=False)
        print(f"\n✅ Artifact index saved to: {idx_path}")
    else:
        print("\n⚠️ No column sets were processed (empty definitions?).")

    # ---- Write AUC summary for the loop ----
    auc_path = None
    auc_df = pd.DataFrame(auc_summary_rows)
    if not auc_df.empty:
        # break ties by uniqueness (prefer higher UR when AUC ties)
        auc_df = auc_df.sort_values(["auc", "uniqueness_ratio"], ascending=[False, False])
        auc_path = output_root / f"loop{i}" / f"loop{i}_auc_summary_{ts}.csv"
        auc_df.to_csv(auc_path, index=False)
        print("\nAUC summary (descending):")
        try:
            print(auc_df.to_string(index=False))
        except Exception:
            print(auc_df.head())
        print(f"\n📄 AUC summary saved to: {auc_path}")

    return {
        "artifact_index_csv": str(idx_path) if idx_path else None,
        "auc_summary_csv": str(auc_path) if auc_path else None,
        "auc_summary_df": auc_df
    }
