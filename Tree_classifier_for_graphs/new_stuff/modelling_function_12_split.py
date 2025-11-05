#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json, time, re, logging, sys, warnings, tempfile, shutil, atexit, gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

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

try:
    import pyarrow  # noqa
except Exception:
    print("Note: pyarrow not found. Pandas may fallback to fastparquet if installed.")

warnings.filterwarnings("ignore")

# ---------- temp dir & cleanup ----------
JOBLIB_TMPDIR = tempfile.mkdtemp(prefix="ultralean-")
@atexit.register
def _cleanup():
    try: shutil.rmtree(JOBLIB_TMPDIR, ignore_errors=True)
    except Exception: pass

# ---------- logging ----------
def setup_logging(log_dir: str | Path, log_prefix: str = "run", level=logging.INFO) -> Path:
    log_dir = Path(log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{log_prefix}_{ts}.log"
    root = logging.getLogger(); root.setLevel(level); root.handlers = []
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s","%Y-%m-%d %H:%M:%S"))
    root.addHandler(fh)
    try:
        from rich.logging import RichHandler  # type: ignore
        ch = RichHandler(rich_tracebacks=True, show_time=False, markup=True)
    except Exception:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
    ch.setLevel(level); root.addHandler(ch)
    logging.info("="*90); logging.info(f"Logging started - {log_file}"); logging.info("="*90)
    return log_file

def _json_safe(o):
    import numpy as _np
    if isinstance(o, (_np.integer,)): return int(o)
    if isinstance(o, (_np.floating,)): return float(o)
    if isinstance(o, (_np.ndarray,)): return o.tolist()
    raise TypeError(f"{type(o).__name__} not JSON serializable")

# ---------- helpers ----------
def _fit_xgb_with_early_stop(model, X_train, y_train, X_val, y_val, early_rounds=50):
    try:
        return model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False,
                         early_stopping_rounds=early_rounds)
    except TypeError:
        try:
            from xgboost.callback import EarlyStopping
            return model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False,
                             callbacks=[EarlyStopping(rounds=early_rounds, save_best=True)])
        except Exception:
            return model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

def _predict_proba_best_iter(model, X):
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

def _infer_loop_from_name(p: Path) -> int | None:
    m = re.search(r"(\d+)", p.stem); return int(m.group(1)) if m else None

# ---------- memory-lean loader (two-pass + downcast) ----------
def load_and_align_parquet_dataset_lean(
    dataset_dir: str | Path,
    loops: list[int] = list(range(5, 12)),
    target_col: str = "COEFFICIENTS",
    loop_col: str = "loops",
) -> pd.DataFrame:
    """
    Two-pass loader:
      - Pass 1: discover common feature columns across files (filtered to requested loops)
      - Pass 2: read only needed columns, downcast dtypes to reduce memory
    """
    import pyarrow.parquet as pq

    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {dataset_dir}")

    # -------- PASS 1: find common features among files that have rows for the target loops
    feature_sets = []
    present_files = []

    for p in files:
        try:
            schema_names = set(pq.ParquetFile(p).schema.names)
        except Exception as e:
            raise RuntimeError(f"Failed to read schema for {p}: {e}")

        if target_col not in schema_names:
            raise ValueError(f"Target column '{target_col}' missing in {p.name}")

        minimal_cols = [c for c in (loop_col, target_col) if c in schema_names]
        df_head = pd.read_parquet(p, columns=minimal_cols, engine="pyarrow")

        if loop_col not in df_head.columns:
            lg = _infer_loop_from_name(p)
            if lg is None:
                raise ValueError(f"Cannot infer loop id from filename: {p.name}")
            df_head[loop_col] = lg

        df_head = df_head[df_head[loop_col].isin(loops)]
        if df_head.empty:
            del df_head
            continue

        feats = list(schema_names - {target_col, loop_col})
        feature_sets.append(set(feats))
        present_files.append(p)
        del df_head

    if not present_files:
        raise ValueError(f"No rows found for loops {loops} in {dataset_dir}")

    common_feats = sorted(list(set.intersection(*feature_sets)))
    if not common_feats:
        raise ValueError("No common feature columns across parquet files.")

    # -------- PASS 2: read only the needed columns, downcast dtypes, and concatenate
    wanted_cols = common_feats + [loop_col, target_col]
    chunks = []

    for p in present_files:
        try:
            schema_names = set(pq.ParquetFile(p).schema.names)
        except Exception as e:
            raise RuntimeError(f"Failed to read schema for {p}: {e}")

        cols_to_read = [c for c in wanted_cols if c in schema_names]
        df = pd.read_parquet(p, columns=cols_to_read, engine="pyarrow")

        if loop_col not in df.columns:
            df[loop_col] = _infer_loop_from_name(p)

        df = df[df[loop_col].isin(loops)]
        if df.empty:
            del df
            continue

        for c in common_feats:
            if c in df.columns and df[c].dtype != np.float32:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

        if target_col in df.columns:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(np.int8)
        df[loop_col] = pd.to_numeric(df[loop_col], errors="coerce").astype(np.int16)

        chunks.append(df)

    if not chunks:
        raise ValueError("After filtering and downcasting, no data remained to merge.")

    merged = pd.concat(chunks, ignore_index=True)
    return merged

def _best_threshold_from_scores(y_true, y_score, metric="f1"):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_val = 0.5, -1.0
    y_true_bool = y_true.astype(bool)
    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int8)
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
        if val > best_val: best_val, best_t = val, float(t)
    return best_t, best_val

@dataclass
class BayesConfig:
    n_calls: int = 30
    n_splits: int = 5
    random_state: int = 42
    results_dir: str | Path = "bayes_results_merged"
    save_plot: bool = True
    threshold_metric: str = "f1"
    holdout_size: float = 0.1
    device: str = "cpu"  # "cuda" if you have a GPU
    use_oof_memmap: bool = True  # write OOF arrays to disk instead of RAM

def bayes_opt_on_merged_stratified(
    df: pd.DataFrame,
    *,
    target_col: str = "COEFFICIENTS",
    loop_col: str = "loops",
    cfg: BayesConfig = BayesConfig()
):
    # Build arrays once, compact, and free the frame
    data_cols = [c for c in df.columns if c not in {target_col, loop_col}]
    df[data_cols] = df[data_cols].astype(np.float32)
    X_all = np.ascontiguousarray(df[data_cols].to_numpy(), dtype=np.float32)
    y_all = df[target_col].to_numpy(dtype=np.int8, copy=True).ravel()
    strat_all = (df[loop_col].astype(str) + "_" + pd.Series(y_all, index=df.index).astype(str)).to_numpy()
    del df; gc.collect()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.results_dir) / f"merged_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # === SAME SEARCH SPACE (unchanged) ===
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
    history_rows = []

    def _fit_eval_fold(train_idx, test_idx, params):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=cfg.random_state,
            n_jobs=1,
            tree_method="hist",
            max_bin=256,
            device=cfg.device,
            **params
        )
        _fit_xgb_with_early_stop(model, X_train, y_train, X_test, y_test, early_rounds=50)
        y_scores = _predict_proba_best_iter(model, X_test)[:, 1]
        auc_val = roc_auc_score(y_test, y_scores)

        del model, X_train, X_test, y_scores, y_train, y_test
        gc.collect()
        return auc_val

    @use_named_args(dimensions=dimensions)
    def objective(**params):
        try:
            fold_aucs = []
            # SERIAL folds (no joblib)
            for tr, te in kf.split(X_all, strat_all):
                fold_aucs.append(_fit_eval_fold(tr, te, params))
            mean_auc = float(np.mean(fold_aucs))
            std_auc = float(np.std(fold_aucs))
            logging.info(f"[MERGED CV] AUC: {mean_auc:.3f} ± {std_auc:.3f} | {params}")
            history_rows.append({**params, "mean_auc": mean_auc, "std_auc": std_auc, "ts": time.time()})
            return -mean_auc
        except Exception as e:
            logging.error(f"Error with params {params}: {e}")
            history_rows.append({**params, "mean_auc": np.nan, "std_auc": np.nan, "error": str(e), "ts": time.time()})
            return 1.0

    logging.info("\n" + "="*70)
    logging.info("Starting MERGED Bayesian Hyperparameter Optimization (serial, ultra-lean, no scaling)")
    logging.info("=" * 70)

    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=cfg.n_calls,
        random_state=cfg.random_state,
        acq_func='EI'
    )

    hist_df = pd.DataFrame(history_rows)
    hist_path = run_dir / "history.csv"; hist_df.to_csv(hist_path, index=False)

    best_params = dict(zip([d.name for d in dimensions], result.x))
    best_score = float(-result.fun)
    logging.info(f"Best CV AUC: {best_score:.4f}")
    for k, v in best_params.items(): logging.info(f"  {k}: {v}")

    # ---------- OOF capture (optionally memmap to disk) ----------
    thresholds = np.linspace(0.01, 0.99, 99)
    prec = np.zeros_like(thresholds, dtype=np.float64)
    rec  = np.zeros_like(thresholds, dtype=np.float64)
    f1s  = np.zeros_like(thresholds, dtype=np.float64)
    bal  = np.zeros_like(thresholds, dtype=np.float64)

    if cfg.use_oof_memmap:
        oof_scores = np.memmap(str(Path(JOBLIB_TMPDIR) / "oof_scores.dat"), dtype="float32", mode="w+", shape=(X_all.shape[0],))
        oof_true   = np.memmap(str(Path(JOBLIB_TMPDIR) / "oof_true.dat"),   dtype="int8",    mode="w+", shape=(X_all.shape[0],))
    else:
        oof_scores = np.empty((X_all.shape[0],), dtype=np.float32)
        oof_true   = y_all.astype(np.int8, copy=True)

    roc_aucs = []
    for tr, te in kf.split(X_all, strat_all):
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=cfg.random_state,
            n_jobs=1,
            tree_method="hist",
            max_bin=256,
            device=cfg.device,
            **best_params
        )
        _fit_xgb_with_early_stop(model, X_tr, y_tr, X_te, y_te, early_rounds=50)
        ys = _predict_proba_best_iter(model, X_te)[:, 1].astype(np.float32, copy=False)
        roc_aucs.append(roc_auc_score(y_te, ys))
        oof_scores[te] = ys
        oof_true[te]   = y_te.astype(np.int8, copy=False)

        yb = y_te.astype(bool, copy=False)
        for i, t in enumerate(thresholds):
            yp = (ys >= t).astype(np.int8, copy=False)
            tp = np.sum((yp == 1) & yb)
            fp = np.sum((yp == 1) & ~yb)
            fn = np.sum((yp == 0) & yb)
            tn = np.sum((yp == 0) & ~yb)
            p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            s  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec[i] += p; rec[i] += r; f1s[i] += f1; bal[i] += 0.5*(r + s)

        del model, X_tr, X_te, y_tr, y_te, ys, yb
        gc.collect()

    n_folds = kf.get_n_splits()
    prec /= n_folds; rec /= n_folds; f1s /= n_folds; bal /= n_folds

    mean_auc = float(np.mean(roc_aucs)); std_auc = float(np.std(roc_aucs))
    logging.info(f"CV ROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")

    thr_df = pd.DataFrame({"threshold": thresholds, "precision": prec, "recall": rec, "f1": f1s, "balanced_accuracy": bal})
    thr_path = run_dir / "thresholds.csv"; thr_df.to_csv(thr_path, index=False)

    plot_path = None
    if cfg.save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, prec, label="Precision")
        plt.plot(thresholds, rec,  label="Recall")
        plt.plot(thresholds, f1s,  label="F1 Score")
        plt.plot(thresholds, bal,  label="Balanced Accuracy")
        plt.xlabel("Threshold"); plt.ylabel("Metric Score")
        plt.title(f"Merged CV Metrics vs. Threshold\nROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plot_path = run_dir / "metrics.png"; plt.savefig(plot_path, dpi=160); plt.close()

    # Best threshold from full OOF
    best_thr, best_thr_val = _best_threshold_from_scores(
        np.array(oof_true, dtype=np.int8),
        np.array(oof_scores, dtype=np.float32),
        metric=cfg.threshold_metric
    )
    logging.info(f"Best threshold ({cfg.threshold_metric}): t={best_thr:.3f} (score={best_thr_val:.4f})")

    # ---------- FINAL RETRAIN ----------
    X_tr, X_val, y_tr, y_val, strat_tr, strat_val = train_test_split(
        X_all, y_all, strat_all, test_size=cfg.holdout_size, random_state=cfg.random_state, stratify=strat_all
    )

    model_tmp = XGBClassifier(
        eval_metric='logloss', random_state=cfg.random_state, n_jobs=1,
        tree_method="hist", max_bin=256, device=cfg.device, **best_params
    )
    _fit_xgb_with_early_stop(model_tmp, X_tr, y_tr, X_val, y_val, early_rounds=50)
    best_iter = getattr(model_tmp, "best_iteration", None)
    if best_iter is None: best_iter = best_params["n_estimators"] - 1
    del model_tmp, X_tr, X_val, y_tr, y_val; gc.collect()

    final_params = dict(best_params); final_params["n_estimators"] = int(best_iter + 1)
    final_model = XGBClassifier(
        eval_metric='logloss', random_state=cfg.random_state, n_jobs=1,
        tree_method="hist", max_bin=256, device=cfg.device, **final_params
    )
    final_model.fit(X_all, y_all)

    # persist artifacts
    try:
        import joblib
        joblib.dump(final_model, run_dir / "final_model_sklearn.pkl")
        final_model.get_booster().save_model(str(run_dir / "final_model_booster.json"))
    except Exception as e:
        logging.warning(f"joblib/xgboost save failed: {e}; falling back to pickle.")
        import pickle
        with open(run_dir / "final_model_sklearn.pkl", "wb") as f: pickle.dump(final_model, f)
        try: final_model.get_booster().save_model(str(run_dir / "final_model_booster.json"))
        except Exception: pass

    with open(run_dir / "feature_names.json", "w") as f: json.dump(list(data_cols), f, indent=2)
    with open(run_dir / "decision_threshold.txt", "w") as f: f.write(f"{best_thr:.6f}\n")
    with open(run_dir / "inference_readme.txt", "w") as f:
        f.write(
            "Inference recipe:\n"
            "1) Load final_model_sklearn.pkl\n"
            "2) Ensure input columns match feature_names.json order\n"
            "3) Convert input to float32 ndarray with same column order\n"
            "4) p = model.predict_proba(X)[:,1]\n"
            f"5) y_hat = (p >= {best_thr:.6f}).astype(int)\n"
        )

    summary = {
        "mode": "merged_parquet",
        "timestamp": ts,
        "n_calls": int(cfg.n_calls),
        "n_splits": int(cfg.n_splits),
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
            "history_csv": str((run_dir / 'history.csv').resolve()),
            "thresholds_csv": str(thr_path.resolve()),
            "plot_png": str(plot_path.resolve()) if plot_path else None,
            "final_model_sklearn_pkl": str((run_dir / "final_model_sklearn.pkl").resolve()),
            "final_model_booster_json": str((run_dir / "final_model_booster.json").resolve()),
            "feature_names_json": str((run_dir / "feature_names.json").resolve()),
            "decision_threshold_txt": str((run_dir / "decision_threshold.txt").resolve()),
        }
    }
    with open(run_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2, default=_json_safe)

    return {
        'best_params': best_params,
        'best_cv_auc': best_score,
        'cv_auc_mean': mean_auc,
        'cv_auc_std': std_auc,
        'artifacts_dir': str(run_dir.resolve()),
        'feature_names': list(data_cols),
        'decision_threshold': float(best_thr),
        'final_model_paths': summary["paths"]
    }

# ---------- runner ----------
def run_bayes_on_merged_parquet(
    dataset_dir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset",
    loops=range(5, 12),
    log_root="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/bayes_results/bayes_results_merged",
    cfg: BayesConfig = BayesConfig()
):
    setup_logging(log_root, log_prefix="merged_bayes_ultralean_noscale")
    logging.info(f"Loading parquet dataset from: {dataset_dir}")
    merged = load_and_align_parquet_dataset_lean(dataset_dir, loops=list(loops))
    logging.info(f"Merged shape: {merged.shape}")
    logging.info(f"Class distribution: {dict(zip(*np.unique(merged['COEFFICIENTS'], return_counts=True)))}")
    logging.info(f"Loop distribution: {dict(zip(*np.unique(merged['loops'], return_counts=True)))}")

    out = bayes_opt_on_merged_stratified(merged, target_col="COEFFICIENTS", loop_col="loops", cfg=cfg)
    logging.info("\n" + "="*90)
    logging.info("Merged / loop+label-stratified optimization complete.")
    logging.info(f"Best CV AUC: {out['best_cv_auc']:.4f}")
    logging.info(f"Artifacts dir: {out['artifacts_dir']}")
    logging.info("="*90)
    return out

if __name__ == "__main__":
    config = BayesConfig(
        n_calls=30,
        n_splits=5,
        random_state=42,
        results_dir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/bayes_results/bayes_results_merged",
        save_plot=True,
        threshold_metric="f1",
        holdout_size=0.1,
        device="cpu",          # switch to "cuda" if your instance has a GPU
        use_oof_memmap=True
    )
    _lf = setup_logging(config.results_dir, log_prefix="merged_bayes_ultralean_noscale")
    print(f"Log file: {_lf}")

    out = run_bayes_on_merged_parquet(
        dataset_dir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset",
        loops=range(5, 12),
        log_root=config.results_dir,
        cfg=config
    )

    print("\n=== Final artifacts ===")
    print(json.dumps(out["final_model_paths"], indent=2))
    print(f"Decision threshold: {out['decision_threshold']:.6f}")
