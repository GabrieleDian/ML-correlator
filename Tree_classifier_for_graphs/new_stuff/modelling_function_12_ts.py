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

from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

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

# SHAP (install if missing)
try:
    import shap
except ImportError:
    print("Installing shap...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "shap"])
    except subprocess.CalledProcessError:
        print("Standard pip install failed, trying with --no-build-isolation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-build-isolation", "shap"])
    import shap

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
    random_state: int = 42
    results_dir: str | Path = "bayes_results_merged"
    save_plot: bool = True
    threshold_metric: str = "f1"
    device: str = "cpu"  # "cuda" if you have a GPU
    use_oof_memmap: bool = True  # write OOF arrays to disk instead of RAM
    shap_sample_size: int = 10000  # Number of samples for SHAP computation

def bayes_opt_on_merged_loop_order(
    df: pd.DataFrame,
    *,
    target_col: str = "COEFFICIENTS",
    loop_col: str = "loops",
    loops: list[int] | None = None,
    cfg: BayesConfig = BayesConfig()
):
    """
    Bayesian optimization with loop-order-based validation.
    
    Args:
        df: DataFrame with features, target, and loop columns
        target_col: Name of target column
        loop_col: Name of loop column
        loops: List of consecutive loop orders to use (e.g., [5, 6, 7, 8, 9, 10, 11]).
                If None, will infer from data. Must be consecutive integers.
        cfg: Configuration object
    
    Validation splits are generated automatically from loops:
        loops[0] -> loops[1]
        loops[0:2] -> loops[2]
        loops[0:3] -> loops[3]
        ...
    """
    logging.info("="*70)
    logging.info("Starting Bayesian optimization with loop-order validation")
    logging.info("="*70)
    
    # Build arrays once, compact, and free the frame
    logging.info("Preparing data arrays...")
    print("Preparing data arrays...", flush=True)
    data_cols = [c for c in df.columns if c not in {target_col, loop_col}]
    logging.info(f"Feature columns: {len(data_cols)}")
    df[data_cols] = df[data_cols].astype(np.float32)
    X_all = np.ascontiguousarray(df[data_cols].to_numpy(), dtype=np.float32)
    y_all = df[target_col].to_numpy(dtype=np.int8, copy=True).ravel()
    loop_all = df[loop_col].to_numpy(dtype=np.int16, copy=True)
    
    # Determine which loops to use
    if loops is None:
        # Infer from data - get unique loops and sort
        available_loops = sorted(np.unique(loop_all).tolist())
        loops = available_loops
        logging.info(f"No loops specified, using all available loops from data: {loops}")
    else:
        # Ensure loops are sorted and convert to list
        loops = sorted(list(loops))
        # Verify loops are consecutive
        if len(loops) > 1:
            for i in range(1, len(loops)):
                if loops[i] != loops[i-1] + 1:
                    raise ValueError(f"Loops must be consecutive integers. Found gap between {loops[i-1]} and {loops[i]}")
        logging.info(f"Using specified loops: {loops}")
    
    # Verify all specified loops exist in data
    available_loops_in_data = set(np.unique(loop_all).tolist())
    missing_loops = set(loops) - available_loops_in_data
    if missing_loops:
        raise ValueError(f"Requested loops {sorted(missing_loops)} not found in data. Available loops: {sorted(available_loops_in_data)}")
    
    logging.info(f"Data arrays prepared: X shape={X_all.shape}, y shape={y_all.shape}")
    logging.info(f"Loop distribution: {dict(zip(*np.unique(loop_all, return_counts=True)))}")
    print(f"Data arrays prepared: {X_all.shape[0]:,} samples, {X_all.shape[1]} features", flush=True)
    print(f"Using loops: {loops}", flush=True)
    
    del df; gc.collect()
    logging.info("DataFrame deleted, memory freed")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.results_dir) / f"merged_looporder_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Results directory: {run_dir}")
    print(f"Results directory: {run_dir}", flush=True)

    # Generate validation splits dynamically from loops
    # Special rule: If loop 5 is present, pack it with loop 6 (start from [5,6] -> 7)
    # For loops [5, 6, 7, 8, 9, 10, 11]:
    #   [5, 6] -> 7  (skip [5] -> 6)
    #   [5, 6, 7] -> 8
    #   [5, 6, 7, 8] -> 9
    #   ...
    if len(loops) < 2:
        raise ValueError(f"Need at least 2 loops for validation splits. Got: {loops}")
    
    validation_splits = []
    start_idx = 1  # Default: start from second loop
    
    # If loop 5 is in the list, pack it with loop 6 (skip the [5] -> 6 split)
    if 5 in loops and len(loops) > 2:
        # Find index of 5
        idx_5 = loops.index(5)
        # If 6 comes right after 5, start from index 2 (so first split is [5,6] -> 7)
        if idx_5 + 1 < len(loops) and loops[idx_5 + 1] == 6:
            start_idx = 2
            logging.info("Loop 5 detected: packing with loop 6, skipping [5] -> 6 split")
            print("Loop 5 detected: packing with loop 6, skipping [5] -> 6 split", flush=True)
    
    for i in range(start_idx, len(loops)):
        train_loops = loops[:i]
        val_loop = loops[i]
        validation_splits.append((train_loops, val_loop))
    
    logging.info(f"Validation splits defined: {validation_splits}")
    print(f"Validation splits: {validation_splits}", flush=True)

    # === SAME SEARCH SPACE ===
    dimensions = [
        Integer(200, 1200, name='n_estimators'),
        Integer(3, 12, name='max_depth'),
        Real(0.01, 0.3, name='learning_rate'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0.0, 5.0, name='reg_alpha'),
        Real(0.0, 10.0, name='reg_lambda'),
    ]

    history_rows = []

    def _fit_eval_split(train_loops, val_loop, params):
        """Train on train_loops, validate on val_loop"""
        train_mask = np.isin(loop_all, train_loops)
        val_mask = (loop_all == val_loop)
        
        n_train = np.sum(train_mask)
        n_val = np.sum(val_mask)
        
        if not np.any(train_mask):
            logging.warning(f"No training data for loops {train_loops}")
            return np.nan
        if not np.any(val_mask):
            logging.warning(f"No validation data for loop {val_loop}")
            return np.nan
        
        logging.info(f"  Split {train_loops}->{val_loop}: training on {n_train:,} samples, validating on {n_val:,} samples")
        X_train, X_val = X_all[train_mask], X_all[val_mask]
        y_train, y_val = y_all[train_mask], y_all[val_mask]

        logging.info(f"  Training model with params: {params}")
        model = XGBClassifier(
            eval_metric='logloss',
            random_state=cfg.random_state,
            n_jobs=1,
            tree_method="hist",
            max_bin=256,
            device=cfg.device,
            **params
        )
        _fit_xgb_with_early_stop(model, X_train, y_train, X_val, y_val, early_rounds=50)
        logging.info(f"  Model training completed (best_iter={getattr(model, 'best_iteration', 'N/A')})")
        y_scores = _predict_proba_best_iter(model, X_val)[:, 1]
        auc_val = roc_auc_score(y_val, y_scores)
        logging.info(f"  Validation AUC: {auc_val:.4f}")

        del model, X_train, X_val, y_scores, y_train, y_val
        gc.collect()
        return auc_val

    iteration_count = [0]  # Use list to allow modification in nested function
    
    @use_named_args(dimensions=dimensions)
    def objective(**params):
        iteration_count[0] += 1
        iter_num = iteration_count[0]
        start_time = time.time()
        
        logging.info(f"\n{'='*70}")
        logging.info(f"Optimization iteration {iter_num}/{cfg.n_calls}")
        logging.info(f"Testing parameters: {params}")
        print(f"\n[{iter_num}/{cfg.n_calls}] Testing parameters...", flush=True)
        
        try:
            split_aucs = []
            # Loop-order validation splits
            for idx, (train_loops, val_loop) in enumerate(validation_splits, 1):
                logging.info(f"  Split {idx}/{len(validation_splits)}: {train_loops} -> {val_loop}")
                print(f"  Split {idx}/{len(validation_splits)}: {train_loops} -> {val_loop}...", flush=True)
                auc = _fit_eval_split(train_loops, val_loop, params)
                if not np.isnan(auc):
                    split_aucs.append(auc)
                    print(f"    AUC: {auc:.4f}", flush=True)
            
            if len(split_aucs) == 0:
                logging.warning(f"No valid splits for params {params}")
                history_rows.append({**params, "mean_auc": np.nan, "std_auc": np.nan, "error": "No valid splits", "ts": time.time()})
                return 1.0
            
            mean_auc = float(np.mean(split_aucs))
            std_auc = float(np.std(split_aucs))
            elapsed = time.time() - start_time
            logging.info(f"[ITER {iter_num}] Mean AUC: {mean_auc:.4f} ± {std_auc:.4f} (took {elapsed:.1f}s)")
            print(f"[ITER {iter_num}] Mean AUC: {mean_auc:.4f} ± {std_auc:.4f} (took {elapsed:.1f}s)", flush=True)
            history_rows.append({**params, "mean_auc": mean_auc, "std_auc": std_auc, "ts": time.time()})
            return -mean_auc
        except Exception as e:
            elapsed = time.time() - start_time
            logging.error(f"Error with params {params} (took {elapsed:.1f}s): {e}")
            print(f"  ERROR: {e}", flush=True)
            import traceback
            logging.error(traceback.format_exc())
            history_rows.append({**params, "mean_auc": np.nan, "std_auc": np.nan, "error": str(e), "ts": time.time()})
            return 1.0

    logging.info("\n" + "="*70)
    logging.info("Starting LOOP-ORDER Bayesian Hyperparameter Optimization")
    splits_str = ", ".join([f"{t}->{v}" for t, v in validation_splits])
    logging.info(f"Validation splits: {splits_str}")
    logging.info(f"Total iterations: {cfg.n_calls}")
    logging.info("=" * 70)
    print(f"\nStarting Bayesian optimization ({cfg.n_calls} iterations)...", flush=True)
    print(f"Validation splits: {splits_str}", flush=True)

    opt_start_time = time.time()
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=cfg.n_calls,
        random_state=cfg.random_state,
        acq_func='EI'
    )
    opt_elapsed = time.time() - opt_start_time
    logging.info(f"\nOptimization completed in {opt_elapsed:.1f} seconds ({opt_elapsed/60:.1f} minutes)")
    print(f"\nOptimization completed in {opt_elapsed/60:.1f} minutes", flush=True)

    logging.info("Saving optimization history...")
    print("Saving optimization history...", flush=True)
    hist_df = pd.DataFrame(history_rows)
    hist_path = run_dir / "history.csv"; hist_df.to_csv(hist_path, index=False)
    logging.info(f"History saved to: {hist_path}")

    best_params = dict(zip([d.name for d in dimensions], result.x))
    best_score = float(-result.fun)
    logging.info("\n" + "="*70)
    logging.info("OPTIMIZATION RESULTS")
    logging.info("="*70)
    logging.info(f"Best CV AUC: {best_score:.4f}")
    print(f"\nBest CV AUC: {best_score:.4f}", flush=True)
    logging.info("Best Parameters:")
    print("Best Parameters:", flush=True)
    for k, v in best_params.items():
        logging.info(f"  {k}: {v}")
        print(f"  {k}: {v}", flush=True)

    # ---------- Evaluate on each split with best params ----------
    logging.info("\n" + "="*70)
    logging.info("Evaluating best parameters on all validation splits")
    logging.info("="*70)
    print("\nEvaluating best parameters on all validation splits...", flush=True)
    
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
    split_results = []
    
    for idx, (train_loops, val_loop) in enumerate(validation_splits, 1):
        logging.info(f"\nProcessing split {idx}/{len(validation_splits)}: {train_loops} -> {val_loop}")
        print(f"  Split {idx}/{len(validation_splits)}: {train_loops} -> {val_loop}...", flush=True)
        
        train_mask = np.isin(loop_all, train_loops)
        val_mask = (loop_all == val_loop)
        
        if not np.any(train_mask) or not np.any(val_mask):
            logging.warning(f"Skipping split {train_loops}->{val_loop} (missing data)")
            print(f"    WARNING: Skipping (missing data)", flush=True)
            continue
        
        n_train = np.sum(train_mask)
        n_val = np.sum(val_mask)
        logging.info(f"  Training on {n_train:,} samples, validating on {n_val:,} samples")
        print(f"    Training: {n_train:,} samples, Validation: {n_val:,} samples", flush=True)
        
        X_tr, X_te = X_all[train_mask], X_all[val_mask]
        y_tr, y_te = y_all[train_mask], y_all[val_mask]

        logging.info(f"  Training model with best parameters...")
        print(f"    Training model...", flush=True)
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
        logging.info(f"  Model trained (best_iter={getattr(model, 'best_iteration', 'N/A')})")
        ys = _predict_proba_best_iter(model, X_te)[:, 1].astype(np.float32, copy=False)
        auc_val = roc_auc_score(y_te, ys)
        roc_aucs.append(auc_val)
        
        logging.info(f"  Split {train_loops}->{val_loop}: AUC = {auc_val:.4f}")
        print(f"    AUC: {auc_val:.4f}", flush=True)
        split_results.append({
            "train_loops": train_loops,
            "val_loop": val_loop,
            "auc": float(auc_val),
            "n_train": int(n_train),
            "n_val": int(n_val)
        })
        
        # Store OOF predictions
        val_indices = np.where(val_mask)[0]
        oof_scores[val_indices] = ys
        oof_true[val_indices] = y_te.astype(np.int8, copy=False)

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

    n_splits = len(roc_aucs)
    if n_splits > 0:
        prec /= n_splits; rec /= n_splits; f1s /= n_splits; bal /= n_splits

    mean_auc = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    std_auc = float(np.std(roc_aucs)) if len(roc_aucs) > 1 else 0.0
    logging.info(f"\nOverall CV ROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"\nOverall CV ROC AUC: {mean_auc:.3f} ± {std_auc:.3f}", flush=True)

    logging.info("Saving threshold analysis and split results...")
    print("Saving threshold analysis and split results...", flush=True)
    thr_df = pd.DataFrame({"threshold": thresholds, "precision": prec, "recall": rec, "f1": f1s, "balanced_accuracy": bal})
    thr_path = run_dir / "thresholds.csv"; thr_df.to_csv(thr_path, index=False)
    logging.info(f"Thresholds saved to: {thr_path}")

    # Save split results
    split_df = pd.DataFrame(split_results)
    split_path = run_dir / "split_results.csv"; split_df.to_csv(split_path, index=False)
    logging.info(f"Split results saved to: {split_path}")

    plot_path = None
    if cfg.save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, prec, label="Precision")
        plt.plot(thresholds, rec,  label="Recall")
        plt.plot(thresholds, f1s,  label="F1 Score")
        plt.plot(thresholds, bal,  label="Balanced Accuracy")
        plt.xlabel("Threshold"); plt.ylabel("Metric Score")
        plt.title(f"Loop-Order CV Metrics vs. Threshold\nROC AUC: {mean_auc:.3f} ± {std_auc:.3f}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plot_path = run_dir / "metrics.png"; plt.savefig(plot_path, dpi=160); plt.close()

    # Best threshold from full OOF
    oof_mask = (oof_scores != 0) | (np.array(oof_true) != 0)  # Use all validation data
    if np.any(oof_mask):
        best_thr, best_thr_val = _best_threshold_from_scores(
            np.array(oof_true)[oof_mask], 
            np.array(oof_scores)[oof_mask],
            metric=cfg.threshold_metric
        )
    else:
        best_thr, best_thr_val = 0.5, 0.0
    logging.info(f"Best threshold ({cfg.threshold_metric}): t={best_thr:.3f} (score={best_thr_val:.4f})")

    # ---------- FINAL RETRAIN on all available data ----------
    loops_str = f"{loops[0]}-{loops[-1]}" if len(loops) > 1 else str(loops[0])
    logging.info("\n" + "="*70)
    logging.info(f"Training final model on all available data (loops {loops_str})")
    logging.info("="*70)
    print(f"\nTraining final model on all data (loops {loops_str})...", flush=True)
    final_params = dict(best_params)
    
    # Use best iteration from last split for n_estimators
    logging.info("Determining best number of estimators...")
    print("Determining best number of estimators...", flush=True)
    model_tmp = XGBClassifier(
        eval_metric='logloss', random_state=cfg.random_state, n_jobs=1,
        tree_method="hist", max_bin=256, device=cfg.device, **best_params
    )
    # Use last split for determining best iteration
    if validation_splits:
        last_train_loops, last_val_loop = validation_splits[-1]
        train_mask = np.isin(loop_all, last_train_loops)
        val_mask = (loop_all == last_val_loop)
    else:
        # Fallback: use all but last loop for training, last loop for validation
        train_mask = np.isin(loop_all, loops[:-1])
        val_mask = (loop_all == loops[-1])
        last_train_loops = loops[:-1]
        last_val_loop = loops[-1]
    
    if np.any(train_mask) and np.any(val_mask):
        train_loops_str = ",".join(map(str, last_train_loops))
        logging.info(f"Using split {train_loops_str} -> {last_val_loop} to determine best iteration")
        X_tr, X_val = X_all[train_mask], X_all[val_mask]
        y_tr, y_val = y_all[train_mask], y_all[val_mask]
        _fit_xgb_with_early_stop(model_tmp, X_tr, y_tr, X_val, y_val, early_rounds=50)
        best_iter = getattr(model_tmp, "best_iteration", None)
        if best_iter is None: best_iter = best_params["n_estimators"] - 1
        logging.info(f"Best iteration determined: {best_iter + 1}")
        print(f"Best iteration: {best_iter + 1}", flush=True)
        del model_tmp, X_tr, X_val, y_tr, y_val
    else:
        best_iter = best_params["n_estimators"] - 1
        logging.info(f"Using default n_estimators: {best_iter + 1}")
    gc.collect()

    final_params["n_estimators"] = int(best_iter + 1)
    logging.info(f"Training final model with n_estimators={final_params['n_estimators']} on all data...")
    print(f"Training final model (n_estimators={final_params['n_estimators']}) on all {X_all.shape[0]:,} samples...", flush=True)
    final_model = XGBClassifier(
        eval_metric='logloss', random_state=cfg.random_state, n_jobs=1,
        tree_method="hist", max_bin=256, device=cfg.device, **final_params
    )
    final_model.fit(X_all, y_all)
    logging.info("Final model training completed")
    print("Final model training completed", flush=True)

    # persist artifacts
    logging.info("Saving model artifacts...")
    print("Saving model artifacts...", flush=True)
    try:
        import joblib
        joblib.dump(final_model, run_dir / "final_model_sklearn.pkl")
        final_model.get_booster().save_model(str(run_dir / "final_model_booster.json"))
        logging.info("Model saved successfully (joblib)")
        print("Model saved successfully", flush=True)
    except Exception as e:
        logging.warning(f"joblib/xgboost save failed: {e}; falling back to pickle.")
        print(f"Warning: joblib save failed, using pickle...", flush=True)
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

    # ---------- SHAP COMPUTATION ----------
    logging.info("\n" + "="*70)
    logging.info("Computing SHAP values for final model")
    logging.info("="*70)
    print("\n" + "="*70, flush=True)
    print("Computing SHAP values for final model", flush=True)
    print("="*70, flush=True)
    
    try:
        shap_results = calculate_shap_values_for_final_model(
            model=final_model,
            X_all=X_all,
            feature_names=data_cols,
            results_dir=run_dir,
            shap_sample_size=cfg.shap_sample_size,
            random_state=cfg.random_state
        )
        logging.info("SHAP computation completed successfully")
        print("SHAP computation completed successfully", flush=True)
    except Exception as e:
        logging.error(f"SHAP computation failed: {e}")
        print(f"ERROR: SHAP computation failed: {e}", flush=True)
        import traceback
        logging.error(traceback.format_exc())
        shap_results = None

    # ---------- FINAL RETRAIN ON ENTIRE DATASET FOR INFERENCE ----------
    logging.info("\n" + "="*70)
    logging.info("Retraining model on ENTIRE dataset for inference on unseen loop orders")
    logging.info("="*70)
    print("\n" + "="*70, flush=True)
    print("Retraining model on ENTIRE dataset for inference", flush=True)
    print("="*70, flush=True)
    
    # Get all unique loops in the dataset
    unique_loops = sorted(np.unique(loop_all).tolist())
    logging.info(f"Training on all available loops: {unique_loops}")
    logging.info(f"Total samples: {X_all.shape[0]:,}")
    print(f"Training on all available loops: {unique_loops}", flush=True)
    print(f"Total samples: {X_all.shape[0]:,}", flush=True)
    
    # Use the best parameters from optimization
    inference_params = dict(best_params)
    
    # Determine best n_estimators using the last validation split
    logging.info("Determining optimal number of estimators for full dataset training...")
    print("Determining optimal number of estimators...", flush=True)
    model_tmp = XGBClassifier(
        eval_metric='logloss', random_state=cfg.random_state, n_jobs=1,
        tree_method="hist", max_bin=256, device=cfg.device, **best_params
    )
    # Use last split for determining best iteration
    if validation_splits:
        last_train_loops, last_val_loop = validation_splits[-1]
        train_mask = np.isin(loop_all, last_train_loops)
        val_mask = (loop_all == last_val_loop)
    else:
        # Fallback: use all but last loop for training, last loop for validation
        train_mask = np.isin(loop_all, loops[:-1])
        val_mask = (loop_all == loops[-1])
        last_train_loops = loops[:-1]
        last_val_loop = loops[-1]
    
    if np.any(train_mask) and np.any(val_mask):
        train_loops_str = ",".join(map(str, last_train_loops))
        logging.info(f"Using split {train_loops_str} -> {last_val_loop} to determine best iteration")
        X_tr, X_val = X_all[train_mask], X_all[val_mask]
        y_tr, y_val = y_all[train_mask], y_all[val_mask]
        _fit_xgb_with_early_stop(model_tmp, X_tr, y_tr, X_val, y_val, early_rounds=50)
        best_iter = getattr(model_tmp, "best_iteration", None)
        if best_iter is None: best_iter = best_params["n_estimators"] - 1
        logging.info(f"Best iteration determined: {best_iter + 1}")
        print(f"Best iteration: {best_iter + 1}", flush=True)
        del model_tmp, X_tr, X_val, y_tr, y_val
    else:
        best_iter = best_params["n_estimators"] - 1
        logging.info(f"Using default n_estimators: {best_iter + 1}")
    gc.collect()
    
    inference_params["n_estimators"] = int(best_iter + 1)
    
    # Train on ENTIRE dataset
    logging.info(f"Training inference model with n_estimators={inference_params['n_estimators']} on ALL {X_all.shape[0]:,} samples...")
    print(f"Training inference model (n_estimators={inference_params['n_estimators']}) on ALL {X_all.shape[0]:,} samples...", flush=True)
    inference_model = XGBClassifier(
        eval_metric='logloss', random_state=cfg.random_state, n_jobs=1,
        tree_method="hist", max_bin=256, device=cfg.device, **inference_params
    )
    inference_model.fit(X_all, y_all)
    logging.info("Inference model training completed")
    print("Inference model training completed", flush=True)
    
    # Save inference model
    logging.info("Saving inference model artifacts...")
    print("Saving inference model artifacts...", flush=True)
    try:
        import joblib
        joblib.dump(inference_model, run_dir / "inference_model_sklearn.pkl")
        inference_model.get_booster().save_model(str(run_dir / "inference_model_booster.json"))
        logging.info("Inference model saved successfully (joblib)")
        print("Inference model saved successfully", flush=True)
    except Exception as e:
        logging.warning(f"joblib/xgboost save failed: {e}; falling back to pickle.")
        print(f"Warning: joblib save failed, using pickle...", flush=True)
        import pickle
        with open(run_dir / "inference_model_sklearn.pkl", "wb") as f: 
            pickle.dump(inference_model, f)
        try: 
            inference_model.get_booster().save_model(str(run_dir / "inference_model_booster.json"))
        except Exception: 
            pass
    
    # Update inference readme
    with open(run_dir / "inference_readme.txt", "w") as f:
        f.write(
            "INFERENCE MODEL (for unseen loop orders):\n"
            "==========================================\n"
            "This model was trained on the ENTIRE dataset (all available loops).\n"
            "Use this model for prediction on unseen loop orders (e.g., loop 12, 13, etc.).\n\n"
            "Inference recipe:\n"
            "1) Load inference_model_sklearn.pkl\n"
            "2) Ensure input columns match feature_names.json order\n"
            "3) Convert input to float32 ndarray with same column order\n"
            "4) p = model.predict_proba(X)[:,1]\n"
            f"5) y_hat = (p >= {best_thr:.6f}).astype(int)\n\n"
            "Trained on loops: " + ", ".join(map(str, unique_loops)) + "\n"
            f"Total training samples: {X_all.shape[0]:,}\n"
            f"Model parameters: n_estimators={inference_params['n_estimators']}, "
            f"max_depth={inference_params['max_depth']}, "
            f"learning_rate={inference_params['learning_rate']:.6f}\n"
        )
    
    logging.info("Inference model ready for use on unseen loop orders")
    print("Inference model ready for use on unseen loop orders", flush=True)

    summary = {
        "mode": "merged_loop_order",
        "timestamp": ts,
        "n_calls": int(cfg.n_calls),
        "random_state": int(cfg.random_state),
        "feature_count": int(len(data_cols)),
        "loops_used": loops,
        "validation_splits": validation_splits,
        "best_params_cv": best_params,
        "cv_auc_mean": mean_auc,
        "cv_auc_std": std_auc,
        "split_results": split_results,
        "selected_threshold_metric": cfg.threshold_metric,
        "selected_threshold": float(best_thr),
        "selected_threshold_score": float(best_thr_val),
        "final_model_n_estimators": int(final_params["n_estimators"]),
        "inference_model_n_estimators": int(inference_params["n_estimators"]),
        "inference_model_trained_on_loops": unique_loops,
        "inference_model_total_samples": int(X_all.shape[0]),
        "paths": {
            "run_dir": str(run_dir.resolve()),
            "history_csv": str((run_dir / 'history.csv').resolve()),
            "split_results_csv": str(split_path.resolve()),
            "thresholds_csv": str(thr_path.resolve()),
            "plot_png": str(plot_path.resolve()) if plot_path else None,
            "final_model_sklearn_pkl": str((run_dir / "final_model_sklearn.pkl").resolve()),
            "final_model_booster_json": str((run_dir / "final_model_booster.json").resolve()),
            "inference_model_sklearn_pkl": str((run_dir / "inference_model_sklearn.pkl").resolve()),
            "inference_model_booster_json": str((run_dir / "inference_model_booster.json").resolve()),
            "feature_names_json": str((run_dir / "feature_names.json").resolve()),
            "decision_threshold_txt": str((run_dir / "decision_threshold.txt").resolve()),
            "inference_readme_txt": str((run_dir / "inference_readme.txt").resolve()),
        }
    }
    
    if shap_results:
        summary["paths"]["shap_full_csv"] = shap_results.get("shap_full_path")
        summary["paths"]["shap_summary_png"] = shap_results.get("shap_summary_png")
        summary["paths"]["shap_sample_csv"] = shap_results.get("shap_sample_path")
    
    logging.info("Saving summary.json...")
    print("Saving summary.json...", flush=True)
    with open(run_dir / "summary.json", "w") as f: json.dump(summary, f, indent=2, default=_json_safe)
    logging.info(f"Summary saved to: {run_dir / 'summary.json'}")

    logging.info("\n" + "="*70)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY")
    logging.info("="*70)
    logging.info(f"All artifacts saved to: {run_dir}")
    print("\n" + "="*70, flush=True)
    print("PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    print(f"All artifacts saved to: {run_dir}", flush=True)
    print("="*70, flush=True)

    return {
        'best_params': best_params,
        'best_cv_auc': best_score,
        'cv_auc_mean': mean_auc,
        'cv_auc_std': std_auc,
        'artifacts_dir': str(run_dir.resolve()),
        'feature_names': list(data_cols),
        'decision_threshold': float(best_thr),
        'final_model_paths': summary["paths"],
        'shap_results': shap_results
    }

# ---------- SHAP computation functions ----------
def calculate_shap_values_for_final_model(
    model,
    X_all: np.ndarray,
    feature_names: list[str],
    results_dir: Path,
    shap_sample_size: int = 10000,
    random_state: int = 42
):
    """Calculate SHAP values for the final trained model."""
    logging.info(f"Computing SHAP values on {X_all.shape[0]} samples...")
    print(f"Computing SHAP values on {X_all.shape[0]} samples...", flush=True)
    
    # Ensure X is float32
    if X_all.dtype != np.float32:
        X_all = X_all.astype(np.float32, copy=False)
    
    # Create SHAP explainer
    logging.info("Creating SHAP TreeExplainer...")
    print("Creating SHAP TreeExplainer...", flush=True)
    
    try:
        # Try using booster from sklearn model
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            explainer = shap.TreeExplainer(booster)
            logging.info("Using XGBoost booster from sklearn model")
        else:
            explainer = shap.TreeExplainer(model)
            logging.info("Using sklearn model directly")
    except Exception as e:
        logging.error(f"Failed to create SHAP TreeExplainer: {e}")
        raise
    
    # Sample data if needed
    if shap_sample_size is not None and X_all.shape[0] > shap_sample_size:
        logging.info(f"Sampling {shap_sample_size} rows from {X_all.shape[0]} total rows...")
        print(f"Sampling {shap_sample_size} rows from {X_all.shape[0]} total rows...", flush=True)
        np.random.seed(random_state)
        sample_indices = np.random.choice(X_all.shape[0], size=shap_sample_size, replace=False)
        X_sample = X_all[sample_indices]
        logging.info(f"Computing SHAP values on {X_sample.shape[0]} samples (this may take several minutes)...")
        print(f"Computing SHAP values on {X_sample.shape[0]} samples (this may take several minutes)...", flush=True)
        print("Progress: Starting SHAP computation...", flush=True)
        
        # For large samples, compute in batches with progress updates
        if X_sample.shape[0] > 1000:
            batch_size = 1000
            n_batches = (X_sample.shape[0] + batch_size - 1) // batch_size
            logging.info(f"Computing SHAP in {n_batches} batches of ~{batch_size} samples each...")
            print(f"Computing SHAP in {n_batches} batches of ~{batch_size} samples each...", flush=True)
            
            shap_batches = []
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X_sample.shape[0])
                batch = X_sample[start_idx:end_idx]
                
                logging.info(f"Processing batch {i+1}/{n_batches} (samples {start_idx+1}-{end_idx})...")
                print(f"Progress: Batch {i+1}/{n_batches} (samples {start_idx+1}-{end_idx})...", flush=True)
                
                batch_shap = explainer.shap_values(batch)
                shap_batches.append(batch_shap)
                
                # Progress update
                if (i + 1) % max(1, n_batches // 10) == 0 or i == n_batches - 1:
                    progress_pct = ((i + 1) / n_batches) * 100
                    logging.info(f"Progress: {progress_pct:.1f}% complete ({i+1}/{n_batches} batches)")
                    print(f"Progress: {progress_pct:.1f}% complete ({i+1}/{n_batches} batches)", flush=True)
            
            # Concatenate batches
            shap_values = np.concatenate(shap_batches, axis=0)
            logging.info("All batches processed, concatenating results...")
            print("All batches processed, concatenating results...", flush=True)
        else:
            shap_values = explainer.shap_values(X_sample)
            print("SHAP computation completed!", flush=True)
        
        X_used = X_sample
    else:
        logging.info(f"Computing SHAP values on full dataset ({X_all.shape[0]} samples)...")
        print(f"Computing SHAP values on full dataset ({X_all.shape[0]} samples)...", flush=True)
        print("Progress: Starting SHAP computation (this may take a while)...", flush=True)
        shap_values = explainer.shap_values(X_all)
        print("SHAP computation completed!", flush=True)
        X_used = X_all
    
    logging.info(f"SHAP values computed. Shape: {np.array(shap_values).shape}")
    print(f"SHAP values computed. Shape: {np.array(shap_values).shape}", flush=True)
    
    # Get feature importance and direction
    logging.info("Computing feature importance...")
    print("Computing feature importance...", flush=True)
    importance_df = get_global_feature_importance(shap_values, feature_names)
    direction_df = get_global_feature_direction(shap_values, feature_names)
    
    # Merge importance and direction
    shap_full = importance_df.merge(direction_df, on="feature")
    shap_full_path = results_dir / "shap_full.csv"
    shap_full.to_csv(shap_full_path, index=False)
    logging.info(f"Saved SHAP full results to: {shap_full_path}")
    print(f"Saved SHAP full results to: {shap_full_path}", flush=True)
    
    # Create SHAP summary plot
    logging.info("Creating SHAP summary plot...")
    print("Creating SHAP summary plot...", flush=True)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_used, feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Summary (Loop-Order Model)")
    shap_png_path = results_dir / "shap_summary.png"
    plt.savefig(shap_png_path, bbox_inches='tight', dpi=200)
    plt.close()
    logging.info(f"Saved SHAP summary plot to: {shap_png_path}")
    print(f"Saved SHAP summary plot to: {shap_png_path}", flush=True)
    
    # Save SHAP sample CSV
    logging.info("Saving SHAP sample CSV...")
    print("Saving SHAP sample CSV...", flush=True)
    if shap_sample_size is not None and len(shap_values) <= shap_sample_size:
        shap_sample = pd.DataFrame(shap_values, columns=feature_names)
        shap_sample["sample_index"] = np.arange(len(shap_sample))
    else:
        n_samples = min(shap_sample_size, len(shap_values)) if shap_sample_size else len(shap_values)
        shap_sample = resample(
            pd.DataFrame(shap_values, columns=feature_names),
            n_samples=n_samples,
            random_state=random_state
        )
        shap_sample["sample_index"] = np.arange(len(shap_sample))
    
    shap_sample_path = results_dir / "shap_sample.csv"
    shap_sample.to_csv(shap_sample_path, index=False)
    logging.info(f"Saved SHAP sample CSV to: {shap_sample_path}")
    print(f"Saved SHAP sample CSV to: {shap_sample_path}", flush=True)
    
    # Log top features
    logging.info("\n" + "="*60)
    logging.info("Top 20 Features by SHAP Importance")
    logging.info("="*60)
    logging.info(shap_full.head(20).to_string(index=False))
    
    return {
        'shap_values': shap_values,
        'explainer': explainer,
        'shap_full_df': shap_full,
        'shap_full_path': str(shap_full_path),
        'shap_summary_png': str(shap_png_path),
        'shap_sample_path': str(shap_sample_path)
    }

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

# ---------- runner ----------
def run_bayes_on_merged_parquet(
    dataset_dir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset",
    loops=range(5, 12),
    log_root="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/bayes_results/bayes_results_merged",
    cfg: BayesConfig = BayesConfig()
):
    """
    Run Bayesian optimization with loop-order validation.
    
    Args:
        dataset_dir: Directory containing parquet files
        loops: Consecutive loop orders to use (e.g., range(5, 12) for loops 5-11)
        log_root: Root directory for logs
        cfg: Configuration object
    """
    setup_logging(log_root, log_prefix="merged_bayes_looporder")
    logging.info(f"Loading parquet dataset from: {dataset_dir}")
    
    # Convert loops to list and ensure it's sorted
    loops_list = sorted(list(loops))
    logging.info(f"Requested loops: {loops_list}")
    
    merged = load_and_align_parquet_dataset_lean(dataset_dir, loops=loops_list)
    logging.info(f"Merged shape: {merged.shape}")
    logging.info(f"Class distribution: {dict(zip(*np.unique(merged['COEFFICIENTS'], return_counts=True)))}")
    logging.info(f"Loop distribution: {dict(zip(*np.unique(merged['loops'], return_counts=True)))}")

    out = bayes_opt_on_merged_loop_order(
        merged, 
        target_col="COEFFICIENTS", 
        loop_col="loops", 
        loops=loops_list,
        cfg=cfg
    )
    logging.info("\n" + "="*90)
    logging.info("Loop-order optimization complete.")
    logging.info(f"Best CV AUC: {out['best_cv_auc']:.4f}")
    logging.info(f"Artifacts dir: {out['artifacts_dir']}")
    logging.info("="*90)
    return out

if __name__ == "__main__":
    # Specify which consecutive loop orders to use
    # Example: range(5, 12) uses loops 5, 6, 7, 8, 9, 10, 11
    # Loops must be consecutive integers (e.g., [5, 6, 7, 8, 9, 10, 11])
    LOOPS_TO_USE = [5,6,7, 8, 9, 10, 11]   # Change this to specify your desired loops
    
    config = BayesConfig(
        n_calls=30,
        random_state=42,
        results_dir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/bayes_results/bayes_results_merged",
        save_plot=True,
        threshold_metric="f1",
        device="cpu",          # switch to "cuda" if your instance has a GPU
        use_oof_memmap=True,
        shap_sample_size=10000
    )
    _lf = setup_logging(config.results_dir, log_prefix="merged_bayes_looporder")
    print(f"Log file: {_lf}")
    print(f"Using loops: {list(LOOPS_TO_USE)}")

    out = run_bayes_on_merged_parquet(
        dataset_dir="/home/ec2-user/ML-correlator/Tree_classifier_for_graphs/new_stuff/features/merged/dataset/dataset",
        loops=LOOPS_TO_USE,
        log_root=config.results_dir,
        cfg=config
    )

    print("\n=== Final artifacts ===")
    print(json.dumps(out["final_model_paths"], indent=2))
    print(f"Decision threshold: {out['decision_threshold']:.6f}")
